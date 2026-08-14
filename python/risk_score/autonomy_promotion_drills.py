"""Run evidence-backed promotion drills in an isolated disposable runtime.

The input runtime is treated as read-only.  Every controller, worker, evaluator,
checkpoint, candidate, and rollout path used by a drill is remapped below one
explicit disposable root.  A drill first finalizes a detailed receipt, then
atomically renames that root out of service, publishes bootstrap gate evidence,
and finally removes the renamed tree.  Failures preserve the tree for forensics.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from risk_score.autonomy_bootstrap import publish_gate_evidence
from risk_score.promotion_auditor import (
    AuditorRuntime,
    EvaluationArtifacts,
    PromotionAuditor,
    tree_manifest,
)
from risk_score.promotion_controller import (
    PROMOTION_FAILURE_STEPS,
    PromotionController,
    RuntimeConfig,
    SafetyHalt,
    inspect_candidate,
)
from risk_score.promotion_evidence import (
    PromotionEvidenceError,
    build_controller_evidence,
    publish_controller_evidence,
)
from risk_score.promotion_state import (
    CandidateState,
    GenerationState,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    fsync_directory,
    load_champion,
    sha256_bytes,
    sha256_file,
)

SCHEMA_VERSION = 1
DRILL_SPEC_CONTRACT = "risk-score-autonomy-promotion-drill-spec-v1"
DRILL_DETAIL_CONTRACT = "risk-score-autonomy-promotion-drill-detail-v1"
DRILL_FAILURE_CONTRACT = "risk-score-autonomy-promotion-drill-failure-v1"
EVALUATOR_JOB_CONTRACT = "risk-score-disposable-evaluator-job-v1"
EVALUATOR_SUMMARY_CONTRACT = "risk-score-disposable-evaluator-summary-v1"
WORKER_RECEIPT_CONTRACT = "risk-score-disposable-worker-receipt-v1"
SUPERVISOR_RECEIPT_CONTRACT = "risk-score-disposable-supervisor-receipt-v1"
COMMAND_RECEIPT_CONTRACT = "risk-score-disposable-command-receipt-v1"

DRILL_GATES = (
    "disposable-canary-drill",
    "crash-replay-drill",
    "rollback-before-admission-drill",
    "rollback-after-admission-drill",
    "shadow-controller-replay",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
_MAX_CANONICAL_JSON_BYTES = 64 * 1024 * 1024
_ALL_QUIESCENT_ROLES = [
    "selfplay",
    "shuffler",
    "trainer",
    "exporter",
    "evaluator",
]
_SENTINEL_PATHS = (
    ("runtime-config", None),
    ("promotion-root", "promotion_root"),
    ("champion-projection", "champion_path"),
    ("trainer-checkpoint", "trainer_checkpoint"),
    ("original-model", "original_model_path"),
    ("candidate-inbox", "candidate_inbox"),
    ("accepted-models", "accepted_models"),
    ("admitted-selfplay", "admitted_selfplay"),
    ("rollout-quarantine", "rollout_quarantine"),
    ("rollback-quarantine", "rollback_quarantine"),
    ("promotion-events", "promotion_root/events"),
    ("data-watermark", "data_watermark_path"),
    ("shuffle-watermark", "shuffle_watermark_path"),
    ("policy", "policy_path"),
    ("powered-config", "powered_config_path"),
    ("standard-config", "standard_config_path"),
    ("discovery-schedule", "discovery_schedule_path"),
    ("confirmation-schedule", "confirmation_schedule_path"),
    ("audit-schedule", "audit_schedule_path"),
    ("lead40-schedule", "lead40_schedule_path"),
    ("lead80-schedule", "lead80_schedule_path"),
    ("standard-confirmation-schedule", "standard_confirmation_schedule_path"),
    ("selfplay-config", "selfplay_config_path"),
    ("gpu-lease-config", "gpu_lease_config_path"),
    ("suite", "suites"),
)
_DYNAMIC_BASELINE_SENTINELS = frozenset({"trainer-checkpoint"})


class DrillError(RuntimeError):
    """A drill input, isolation invariant, or derived result is invalid."""


class InjectedPromotionCrash(RuntimeError):
    """The deliberate crash used at one promotion durability boundary."""


def _reject_constant(value: str) -> None:
    raise DrillError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DrillError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _decode_json(data: bytes, source: Path) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, DrillError) as exc:
        raise DrillError(f"{source}: invalid JSON: {exc}") from exc


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise DrillError(f"cannot inspect {role} {source}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DrillError(f"{role} must be a regular non-symlink file")
    if metadata.st_size > _MAX_CANONICAL_JSON_BYTES:
        raise DrillError(f"{role} exceeds the canonical JSON size limit")
    data = source.read_bytes()
    value = _decode_json(data, source)
    if not isinstance(value, dict):
        raise DrillError(f"{role} must have an object root")
    if data != canonical_json_bytes(value) + b"\n":
        raise DrillError(f"{role} must be canonical newline-terminated JSON")
    return value


def _exact_keys(value: Any, expected: Iterable[str], role: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DrillError(f"{role} must be an object")
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise DrillError(
            f"{role} keys differ; missing={sorted(expected_set - actual)}, "
            f"unknown={sorted(actual - expected_set)}"
        )
    return dict(value)


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DrillError(f"{role} must be a lowercase SHA-256")
    return value


def _require_int(
    value: Any,
    role: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise DrillError(f"{role} must be an integer in [{minimum}, {maximum}]")
    return value


def _absolute_normalized_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DrillError(f"{role} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute() or str(path) != str(path.resolve(strict=False)):
        raise DrillError(f"{role} must be an absolute normalized path")
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or _is_relative_to(left, right) or _is_relative_to(right, left)


def _existing_ancestor(path: Path) -> Path:
    current = Path(path)
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.exists():
        raise DrillError(f"path has no existing ancestor: {path}")
    return current


def _assert_no_symlink_ancestors(path: Path, *, include_leaf: bool = True) -> None:
    target = Path(path)
    parts = target.parts
    current = Path(parts[0])
    stop = len(parts) if include_leaf else max(1, len(parts) - 1)
    for part in parts[1:stop]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise DrillError(f"path traverses a symlink: {current}")


def _snapshot_path(path: Path) -> Dict[str, Any]:
    """Hash a missing path, regular file, or symlink-free directory tree."""

    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError:
        return {
            "kind": "missing",
            "sha256": canonical_sha256({"kind": "missing"}),
        }
    if stat.S_ISLNK(metadata.st_mode):
        raise DrillError(f"sentinel may not be a symlink: {source}")
    if stat.S_ISREG(metadata.st_mode):
        return {"kind": "file", "sha256": sha256_file(source)}
    if not stat.S_ISDIR(metadata.st_mode):
        raise DrillError(f"sentinel is not a regular file or directory: {source}")

    rows: List[Mapping[str, Any]] = []

    def walk(directory: Path, relative: Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda item: item.name)
        for entry in ordered:
            child = Path(entry.path)
            child_relative = relative / entry.name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode):
                raise DrillError(f"sentinel tree contains a symlink: {child}")
            if stat.S_ISDIR(child_metadata.st_mode):
                rows.append(
                    {
                        "kind": "directory",
                        "path": child_relative.as_posix(),
                    }
                )
                walk(child, child_relative)
            elif stat.S_ISREG(child_metadata.st_mode):
                rows.append(
                    {
                        "kind": "file",
                        "path": child_relative.as_posix(),
                        "size": child_metadata.st_size,
                        "sha256": sha256_file(child),
                    }
                )
            else:
                raise DrillError(f"sentinel tree contains a special file: {child}")

    walk(source, Path())
    return {
        "kind": "directory",
        "sha256": canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": "risk-score-symlink-free-tree-snapshot-v1",
                "files": rows,
            }
        ),
    }


def _sentinel_path(
    runtime_path: Path,
    runtime: RuntimeConfig,
    attribute: Optional[str],
) -> Path:
    if attribute is None:
        return runtime_path
    if "/" in attribute:
        first, *remaining = attribute.split("/")
        path = Path(getattr(runtime, first))
        for part in remaining:
            path /= part
        return path
    return Path(getattr(runtime, attribute))


def _locate_champion_model(runtime: RuntimeConfig) -> Path:
    try:
        champion = load_champion(runtime.champion_path)
    except (OSError, ValueError) as exc:
        raise DrillError(
            f"production champion projection is unavailable: {exc}"
        ) from exc
    if champion.champion_hash == runtime.controller.original_hash:
        candidate = runtime.original_model_path
    else:
        candidate = (
            runtime.accepted_models
            / "generations"
            / champion.champion_hash
            / champion.generation_id
            / "model.bin.gz"
        )
        if not candidate.is_file():
            matches = []
            if runtime.accepted_models.is_dir():
                for directory in sorted(runtime.accepted_models.iterdir()):
                    model = directory / "model.bin.gz"
                    if (
                        directory.is_dir()
                        and not directory.is_symlink()
                        and model.is_file()
                        and not model.is_symlink()
                        and sha256_file(model) == champion.champion_hash
                    ):
                        matches.append(model)
            if len(matches) != 1:
                raise DrillError(
                    "production champion has no unique accepted immutable model"
                )
            candidate = matches[0]
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or sha256_file(candidate) != champion.champion_hash
    ):
        raise DrillError("production champion model hash contradicts projection")
    return candidate


def _build_sentinel_bindings(
    runtime_path: Path,
    runtime: RuntimeConfig,
    *,
    extra_sentinels: Optional[Mapping[str, Path]] = None,
) -> List[Mapping[str, Any]]:
    bindings = []
    for role, attribute in _SENTINEL_PATHS:
        path = _sentinel_path(runtime_path, runtime, attribute)
        snapshot = _snapshot_path(path)
        bindings.append(
            {
                "role": role,
                "path": str(path),
                **snapshot,
            }
        )
    champion_model = _locate_champion_model(runtime)
    bindings.append(
        {
            "role": "champion-model",
            "path": str(champion_model),
            **_snapshot_path(champion_model),
        }
    )
    for role, raw_path in sorted((extra_sentinels or {}).items()):
        if (
            not isinstance(role, str)
            or not role.startswith("extra:")
            or "\n" in role
            or "\r" in role
        ):
            raise DrillError("extra sentinel roles must start with 'extra:'")
        path = Path(raw_path).resolve()
        bindings.append({"role": role, "path": str(path), **_snapshot_path(path)})
    bindings.sort(key=lambda item: item["role"])
    roles = [binding["role"] for binding in bindings]
    if len(roles) != len(set(roles)):
        raise DrillError("sentinel roles must be unique")
    return bindings


def build_drill_spec(
    *,
    production_runtime_path: Path,
    disposable_root: Path,
    evidence_root: Path,
    extra_sentinels: Optional[Mapping[str, Path]] = None,
    command_timeout_seconds: int = 15,
    max_evaluator_rows: int = 10000,
    max_worker_games: int = 10000,
    max_replay_attempts: int = 16,
) -> Mapping[str, Any]:
    """Build a strict spec around a RuntimeConfig emitted by build_live_runtime."""

    runtime_path = Path(production_runtime_path).resolve()
    disposable = Path(disposable_root).resolve()
    evidence = Path(evidence_root).resolve()
    if (
        runtime_path.is_symlink()
        or not runtime_path.is_file()
        or evidence.is_symlink()
        or not evidence.is_dir()
    ):
        raise DrillError("production runtime and evidence root must already exist")
    if disposable.exists():
        raise DrillError("disposable root must not exist when the spec is built")
    _assert_no_symlink_ancestors(disposable, include_leaf=False)
    runtime = RuntimeConfig.load(runtime_path)
    PromotionController(runtime, automatic=False).validate_static_inputs()
    module_path = Path(__file__).resolve()
    python_path = Path(sys.executable).resolve()
    if (
        module_path.is_symlink()
        or not module_path.is_file()
        or python_path.is_symlink()
        or not python_path.is_file()
    ):
        raise DrillError(
            "bound Python executable and drill module must be regular files"
        )
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": DRILL_SPEC_CONTRACT,
        "production_runtime": {
            "path": str(runtime_path),
            "sha256": sha256_file(runtime_path),
        },
        "disposable_root": str(disposable),
        "evidence_root": str(evidence),
        "sentinels": _build_sentinel_bindings(
            runtime_path,
            runtime,
            extra_sentinels=extra_sentinels,
        ),
        "command_bindings": {
            "python_executable": {
                "path": str(python_path),
                "sha256": sha256_file(python_path),
            },
            "drill_module": {
                "path": str(module_path),
                "sha256": sha256_file(module_path),
            },
        },
        "limits": {
            "command_timeout_seconds": _require_int(
                command_timeout_seconds,
                "command timeout",
                minimum=1,
                maximum=60,
            ),
            "max_evaluator_rows": _require_int(
                max_evaluator_rows,
                "maximum evaluator rows",
                minimum=1,
                maximum=100000,
            ),
            "max_worker_games": _require_int(
                max_worker_games,
                "maximum worker games",
                minimum=1,
                maximum=100000,
            ),
            "max_replay_attempts": _require_int(
                max_replay_attempts,
                "maximum replay attempts",
                minimum=8,
                maximum=64,
            ),
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    return value


def publish_drill_spec(
    output_path: Path,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Publish a newly built canonical drill spec without replacing one."""

    path = Path(output_path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise DrillError("drill spec output parent must be an existing directory")
    if path.exists() or path.is_symlink():
        raise DrillError(f"drill spec output already exists: {path}")
    value = build_drill_spec(**kwargs)
    _write_once(path, value)
    return value


@dataclass(frozen=True)
class DrillSpec:
    path: Path
    file_sha256: str
    spec_sha256: str
    production_runtime_path: Path
    production_runtime_sha256: str
    disposable_root: Path
    evidence_root: Path
    sentinels: Tuple[Mapping[str, Any], ...]
    python_executable: Path
    python_executable_sha256: str
    module_path: Path
    module_sha256: str
    command_timeout_seconds: int
    max_evaluator_rows: int
    max_worker_games: int
    max_replay_attempts: int

    @classmethod
    def load(cls, path: Path) -> "DrillSpec":
        source = Path(path).resolve()
        value = _load_canonical_object(source, "drill specification")
        root = _exact_keys(
            value,
            {
                "schema_version",
                "contract",
                "production_runtime",
                "disposable_root",
                "evidence_root",
                "sentinels",
                "command_bindings",
                "limits",
                "spec_sha256",
            },
            "drill specification",
        )
        payload = dict(root)
        supplied_identity = _require_sha256(
            payload.pop("spec_sha256", None), "drill specification identity"
        )
        if (
            root["schema_version"] != SCHEMA_VERSION
            or root["contract"] != DRILL_SPEC_CONTRACT
            or canonical_sha256(payload) != supplied_identity
        ):
            raise DrillError("drill specification identity is invalid")
        production = _exact_keys(
            root["production_runtime"],
            {"path", "sha256"},
            "production runtime binding",
        )
        runtime_path = _absolute_normalized_path(
            production["path"], "production runtime path"
        )
        runtime_hash = _require_sha256(production["sha256"], "production runtime hash")
        disposable = _absolute_normalized_path(
            root["disposable_root"], "disposable root"
        )
        evidence = _absolute_normalized_path(root["evidence_root"], "evidence root")
        command_bindings = _exact_keys(
            root["command_bindings"],
            {"python_executable", "drill_module"},
            "command bindings",
        )

        def command_binding(name: str) -> Tuple[Path, str]:
            binding = _exact_keys(
                command_bindings[name], {"path", "sha256"}, f"{name} binding"
            )
            return (
                _absolute_normalized_path(binding["path"], f"{name} path"),
                _require_sha256(binding["sha256"], f"{name} hash"),
            )

        python_path, python_hash = command_binding("python_executable")
        module_path, module_hash = command_binding("drill_module")
        limits = _exact_keys(
            root["limits"],
            {
                "command_timeout_seconds",
                "max_evaluator_rows",
                "max_worker_games",
                "max_replay_attempts",
            },
            "drill limits",
        )
        raw_sentinels = root["sentinels"]
        if not isinstance(raw_sentinels, list) or not raw_sentinels:
            raise DrillError("drill sentinels must be a nonempty array")
        sentinels = []
        for index, item in enumerate(raw_sentinels):
            binding = _exact_keys(
                item,
                {"role", "path", "kind", "sha256"},
                f"sentinel {index}",
            )
            role = binding["role"]
            if not isinstance(role, str) or not role or "\n" in role or "\r" in role:
                raise DrillError(f"sentinel {index} role is invalid")
            sentinel_path = _absolute_normalized_path(
                binding["path"], f"sentinel {index} path"
            )
            kind = binding["kind"]
            if kind not in {"missing", "file", "directory"}:
                raise DrillError(f"sentinel {index} kind is invalid")
            sentinels.append(
                {
                    "role": role,
                    "path": str(sentinel_path),
                    "kind": kind,
                    "sha256": _require_sha256(
                        binding["sha256"], f"sentinel {index} hash"
                    ),
                }
            )
        roles = [item["role"] for item in sentinels]
        if roles != sorted(roles) or len(roles) != len(set(roles)):
            raise DrillError("drill sentinels must have sorted unique roles")
        required_roles = {role for role, _ in _SENTINEL_PATHS} | {"champion-model"}
        if not required_roles.issubset(roles):
            raise DrillError(
                "drill sentinels omit required roles: "
                + ", ".join(sorted(required_roles.difference(roles)))
            )
        return cls(
            path=source,
            file_sha256=sha256_file(source),
            spec_sha256=supplied_identity,
            production_runtime_path=runtime_path,
            production_runtime_sha256=runtime_hash,
            disposable_root=disposable,
            evidence_root=evidence,
            sentinels=tuple(sentinels),
            python_executable=python_path,
            python_executable_sha256=python_hash,
            module_path=module_path,
            module_sha256=module_hash,
            command_timeout_seconds=_require_int(
                limits["command_timeout_seconds"],
                "command timeout",
                minimum=1,
                maximum=60,
            ),
            max_evaluator_rows=_require_int(
                limits["max_evaluator_rows"],
                "maximum evaluator rows",
                minimum=1,
                maximum=100000,
            ),
            max_worker_games=_require_int(
                limits["max_worker_games"],
                "maximum worker games",
                minimum=1,
                maximum=100000,
            ),
            max_replay_attempts=_require_int(
                limits["max_replay_attempts"],
                "maximum replay attempts",
                minimum=8,
                maximum=64,
            ),
        )


def load_drill_spec(path: Path) -> DrillSpec:
    return DrillSpec.load(path)


def _snapshot_sentinels(spec: DrillSpec) -> Tuple[Mapping[str, Any], ...]:
    values = []
    for binding in spec.sentinels:
        path = Path(binding["path"])
        values.append(
            {
                "role": binding["role"],
                "path": str(path),
                **_snapshot_path(path),
            }
        )
    return tuple(values)


def _assert_expected_sentinels(
    spec: DrillSpec,
    actual: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    expected = {
        item["role"]: (item["path"], item["kind"], item["sha256"])
        for item in spec.sentinels
    }
    observed = {
        item["role"]: (item["path"], item["kind"], item["sha256"]) for item in actual
    }
    for role in _DYNAMIC_BASELINE_SENTINELS:
        if (
            role not in expected
            or role not in observed
            or expected[role][0] != observed[role][0]
        ):
            raise DrillError(
                f"dynamic production sentinel is missing or moved during {phase}: "
                f"{role}"
            )
    fixed_expected = {
        role: value
        for role, value in expected.items()
        if role not in _DYNAMIC_BASELINE_SENTINELS
    }
    fixed_observed = {
        role: value
        for role, value in observed.items()
        if role not in _DYNAMIC_BASELINE_SENTINELS
    }
    if fixed_observed != fixed_expected:
        changed = sorted(
            role
            for role in set(fixed_expected) | set(fixed_observed)
            if fixed_expected.get(role) != fixed_observed.get(role)
        )
        raise DrillError(
            f"production sentinels changed during {phase}: {', '.join(changed)}"
        )


def _assert_command_bindings(spec: DrillSpec) -> None:
    if (
        spec.python_executable.is_symlink()
        or not spec.python_executable.is_file()
        or sha256_file(spec.python_executable) != spec.python_executable_sha256
    ):
        raise DrillError("bound Python executable changed")
    current_module = Path(__file__).resolve()
    if (
        spec.module_path != current_module
        or spec.module_path.is_symlink()
        or not spec.module_path.is_file()
        or sha256_file(spec.module_path) != spec.module_sha256
    ):
        raise DrillError("bound drill module changed")


def _validate_run_environment(spec: DrillSpec) -> RuntimeConfig:
    if (
        spec.production_runtime_path.is_symlink()
        or not spec.production_runtime_path.is_file()
        or sha256_file(spec.production_runtime_path) != spec.production_runtime_sha256
    ):
        raise DrillError("production runtime binding changed")
    _assert_command_bindings(spec)
    if spec.disposable_root.exists() or spec.disposable_root.is_symlink():
        raise DrillError("disposable root must be absent before a drill")
    if spec.evidence_root.is_symlink() or not spec.evidence_root.is_dir():
        raise DrillError("evidence root must be an existing non-symlink directory")
    _assert_no_symlink_ancestors(spec.disposable_root, include_leaf=False)
    _assert_no_symlink_ancestors(spec.evidence_root)
    if _paths_overlap(spec.disposable_root, spec.evidence_root):
        raise DrillError("evidence root may not overlap the disposable root")
    for sentinel in spec.sentinels:
        if _paths_overlap(spec.disposable_root, Path(sentinel["path"])):
            raise DrillError(
                f"disposable root overlaps production sentinel {sentinel['role']}"
            )
        if _paths_overlap(spec.evidence_root, Path(sentinel["path"])):
            raise DrillError(
                f"evidence root overlaps production sentinel {sentinel['role']}"
            )
    production = RuntimeConfig.load(spec.production_runtime_path)
    PromotionController(production, automatic=False).validate_static_inputs()
    production_anchor = _existing_ancestor(production.promotion_root)
    disposable_parent = _existing_ancestor(spec.disposable_root.parent)
    if production_anchor.stat().st_dev != disposable_parent.stat().st_dev:
        raise DrillError("disposable root is not on the production filesystem")
    return production


def _copy_regular_file(source: Path, destination: Path) -> None:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise DrillError(f"copy source is not a regular file: {source}")
    source_metadata = source.stat()
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != source_hash
        ):
            raise DrillError(f"copy destination conflicts: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.copy-",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with (
            os.fdopen(descriptor, "wb") as output_file,
            source.open("rb") as input_file,
        ):
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        if (
            sha256_file(source) != source_hash
            or source.stat().st_size != source_metadata.st_size
            or sha256_file(temporary) != source_hash
        ):
            raise DrillError(f"copy source changed while being read: {source}")
        os.chmod(temporary, stat.S_IMODE(source_metadata.st_mode))
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _assert_regular_tree(root: Path) -> None:
    source = Path(root)
    if source.is_symlink() or not source.is_dir():
        raise DrillError(f"copy source is not a non-symlink directory: {source}")
    for directory, directories, files in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            path = directory_path / name
            if path.is_symlink() or not path.is_dir():
                raise DrillError(f"copy tree contains an unsafe directory: {path}")
        for name in files:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise DrillError(f"copy tree contains an unsafe file: {path}")


def _copy_regular_tree(source: Path, destination: Path) -> None:
    _assert_regular_tree(source)
    if destination.exists() or destination.is_symlink():
        raise DrillError(f"copy destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)
    _assert_regular_tree(destination)


def _make_tree_writable(root: Path) -> None:
    path = Path(root)
    if not path.exists():
        return
    if path.is_symlink():
        raise DrillError(f"refusing to clean symlink tree: {path}")
    for directory, directories, files in os.walk(path, topdown=False):
        directory_path = Path(directory)
        for name in files:
            child = directory_path / name
            if child.is_symlink():
                raise DrillError(f"refusing to clean symlink: {child}")
            os.chmod(child, 0o600)
        for name in directories:
            child = directory_path / name
            if child.is_symlink():
                raise DrillError(f"refusing to clean symlink: {child}")
            os.chmod(child, 0o700)
        os.chmod(directory_path, 0o700)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    _make_tree_writable(root)
    shutil.rmtree(root)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_controller_evidence(path, value)
    except PromotionEvidenceError as exc:
        raise DrillError(f"immutable artifact conflicts: {path}: {exc}") from exc


def _runtime_mapping(
    spec: DrillSpec,
    production: RuntimeConfig,
    scenario_root: Path,
    *,
    mutation_enabled: bool,
) -> Tuple[Mapping[str, Any], Path, Path]:
    root = Path(scenario_root).resolve()
    promotion = root / "promotion"
    inputs = root / "inputs"
    suites = inputs / "suites"
    schedules = inputs / "schedules"
    for directory in (
        root,
        promotion,
        root / "candidate-inbox",
        root / "trainer",
        root / "admitted-selfplay",
        inputs,
        schedules,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    def copied_schedule(source: Path, fallback: str) -> Path:
        try:
            relative = Path(source).resolve().relative_to(production.suites.resolve())
        except ValueError:
            return schedules / fallback
        return suites / relative

    copied_files = {
        "policy": (production.policy_path, inputs / "policy.json"),
        "poweredConfig": (
            production.powered_config_path,
            inputs / "powered.cfg",
        ),
        "standardConfig": (
            production.standard_config_path,
            inputs / "standard.cfg",
        ),
        "discoveryOrdinarySchedule": (
            production.discovery_schedule_path,
            copied_schedule(production.discovery_schedule_path, "discovery.jsonl"),
        ),
        "confirmationOrdinarySchedule": (
            production.confirmation_schedule_path,
            copied_schedule(
                production.confirmation_schedule_path, "confirmation.jsonl"
            ),
        ),
        "auditSchedule": (
            production.audit_schedule_path,
            copied_schedule(production.audit_schedule_path, "audit.jsonl"),
        ),
        "lead40Schedule": (
            production.lead40_schedule_path,
            copied_schedule(production.lead40_schedule_path, "lead40.jsonl"),
        ),
        "lead80Schedule": (
            production.lead80_schedule_path,
            copied_schedule(production.lead80_schedule_path, "lead80.jsonl"),
        ),
        "standardConfirmationSchedule": (
            production.standard_confirmation_schedule_path,
            schedules / "standard-confirmation.jsonl",
        ),
        "selfplayConfig": (
            production.selfplay_config_path,
            inputs / "selfplay.cfg",
        ),
        "gpuLeaseConfig": (
            production.gpu_lease_config_path,
            inputs / "gpu-lease.json",
        ),
        "original": (
            production.original_model_path,
            inputs / "original.bin.gz",
        ),
        "trainerCheckpoint": (
            production.trainer_checkpoint,
            root / "trainer" / "model.ckpt",
        ),
        "dataWatermark": (
            production.data_watermark_path,
            promotion / "watermarks" / "data.json",
        ),
        "shuffleWatermark": (
            production.shuffle_watermark_path,
            promotion / "watermarks" / "shuffle.json",
        ),
    }
    _copy_regular_tree(production.suites, suites)
    for source, destination in copied_files.values():
        _copy_regular_file(source, destination)

    champion = load_champion(production.champion_path)
    champion_source = _locate_champion_model(production)
    champion_copy = inputs / "champion.bin.gz"
    _copy_regular_file(champion_source, champion_copy)
    if sha256_file(champion_copy) != champion.champion_hash:
        raise DrillError("copied champion model changed")

    policy = json.loads((inputs / "policy.json").read_text(encoding="utf-8"))
    canary_games = (
        policy.get("rollout", {}).get("canary_games")
        if isinstance(policy, Mapping)
        else None
    )
    if (
        type(canary_games) is not int
        or canary_games <= 0
        or canary_games > spec.max_worker_games
    ):
        raise DrillError("policy canary game count exceeds the bounded worker limit")

    python = str(spec.python_executable)
    module = str(spec.module_path)
    common = [
        python,
        module,
        "--internal-module-sha256",
        spec.module_sha256,
        "--internal-root",
        str(root),
    ]
    commands = {
        "trainer": [
            *common,
            "__dummy-supervisor",
            "--action",
            "trainer",
        ],
        "stage0Probe": [
            *common,
            "__dummy-supervisor",
            "--action",
            "stage0",
        ],
        "evaluator": [
            *common,
            "__dummy-supervisor",
            "--action",
            "evaluator",
        ],
        "selfplay": [
            *common,
            "__dummy-worker",
            "--generation-id",
            "{generation_id}",
            "--worker-id",
            "{worker_id}",
            "--output",
            "{worker_output_directory}",
            "--model",
            "{model}",
            "--model-sha256",
            "{model_hash}",
            "--config",
            "{selfplay_config}",
            "--config-sha256",
            "{selfplay_config_hash}",
            "--policy-sha256",
            "{policy_hash}",
            "--games",
            str(canary_games),
            "--maximum-games",
            str(spec.max_worker_games),
        ],
        "drain": [
            *common,
            "__dummy-supervisor",
            "--action",
            "drain",
            "--generation-id",
            "{generation_id}",
            "--manifest",
            "{drain_manifest}",
        ],
        "rollback": [
            *common,
            "__dummy-supervisor",
            "--action",
            "rollback",
            "--generation-id",
            "{generation_id}",
        ],
    }
    mapping: Dict[str, Any] = {
        "schemaVersion": 1,
        "mutationEnabled": mutation_enabled,
        "actor": f"{production.controller.actor}-disposable-drill",
        "hashes": {
            "controller": production.controller.controller_hash,
            "source": production.controller.source_hash,
            "original": sha256_file(copied_files["original"][1]),
            "policy": canonical_sha256(policy),
            "poweredConfig": sha256_file(copied_files["poweredConfig"][1]),
            "standardConfig": sha256_file(copied_files["standardConfig"][1]),
            "discoveryOrdinarySchedule": sha256_file(
                copied_files["discoveryOrdinarySchedule"][1]
            ),
            "confirmationOrdinarySchedule": sha256_file(
                copied_files["confirmationOrdinarySchedule"][1]
            ),
            "auditSchedule": sha256_file(copied_files["auditSchedule"][1]),
            "lead40Schedule": sha256_file(copied_files["lead40Schedule"][1]),
            "lead80Schedule": sha256_file(copied_files["lead80Schedule"][1]),
            "standardConfirmationSchedule": sha256_file(
                copied_files["standardConfirmationSchedule"][1]
            ),
            "selfplayConfig": sha256_file(copied_files["selfplayConfig"][1]),
            "gpuLeaseConfig": sha256_file(copied_files["gpuLeaseConfig"][1]),
            "suiteManifest": sha256_file(suites / "manifest.json"),
        },
        "paths": {
            "promotionRoot": str(promotion),
            "controllerLock": str(promotion / "controller.lock"),
            "champion": str(promotion / "champion.json"),
            "candidateInbox": str(root / "candidate-inbox"),
            "candidateQuarantine": str(promotion / "candidates" / "quarantined"),
            "candidateSuperseded": str(promotion / "candidates" / "superseded"),
            "candidateRejected": str(promotion / "candidates" / "rejected"),
            "candidateDeduplicated": str(promotion / "candidates" / "deduplicated"),
            "acceptedModels": str(promotion / "accepted"),
            "admittedSelfplay": str(root / "admitted-selfplay"),
            "rolloutQuarantine": str(promotion / "rollouts"),
            "rollbackQuarantine": str(promotion / "rollback"),
            "trainerCheckpoint": str(copied_files["trainerCheckpoint"][1]),
            "evaluations": str(promotion / "evaluations"),
            "reports": str(promotion / "reports"),
            "suites": str(suites),
            "policy": str(copied_files["policy"][1]),
            "poweredConfig": str(copied_files["poweredConfig"][1]),
            "standardConfig": str(copied_files["standardConfig"][1]),
            "discoveryOrdinarySchedule": str(
                copied_files["discoveryOrdinarySchedule"][1]
            ),
            "confirmationOrdinarySchedule": str(
                copied_files["confirmationOrdinarySchedule"][1]
            ),
            "auditSchedule": str(copied_files["auditSchedule"][1]),
            "lead40Schedule": str(copied_files["lead40Schedule"][1]),
            "lead80Schedule": str(copied_files["lead80Schedule"][1]),
            "standardConfirmationSchedule": str(
                copied_files["standardConfirmationSchedule"][1]
            ),
            "selfplayConfig": str(copied_files["selfplayConfig"][1]),
            "gpuLeaseConfig": str(copied_files["gpuLeaseConfig"][1]),
            "dataWatermark": str(copied_files["dataWatermark"][1]),
            "shuffleWatermark": str(copied_files["shuffleWatermark"][1]),
            "workerAckInbox": str(promotion / "ipc" / "worker-acks"),
            "rolloutReportInbox": str(promotion / "ipc" / "rollout-reports"),
            "originalModel": str(copied_files["original"][1]),
        },
        "commands": commands,
        "polling": {
            "intervalSeconds": production.controller.poll_interval_seconds,
        },
        "limits": {
            "maxActiveQueue": production.controller.max_active_queue,
            "minFreeBytes": 0,
        },
        "backlog": {
            "anchorIntervalSamples": production.controller.anchor_interval_samples,
            "anomalyNames": list(production.controller.anomaly_names),
        },
        "rollout": {
            "workerCount": production.controller.worker_count,
            "canaryWorkerCount": production.controller.canary_worker_count,
            "intermediateWorkerCount": production.controller.intermediate_worker_count,
            "threadsPerWorker": production.controller.worker_threads,
        },
    }
    return mapping, champion_copy, copied_files["trainerCheckpoint"][1]


def _assert_disposable_runtime(runtime: RuntimeConfig, root: Path) -> None:
    path_attributes = (
        "promotion_root",
        "lock_path",
        "champion_path",
        "candidate_inbox",
        "candidate_quarantine",
        "candidate_superseded",
        "candidate_rejected",
        "candidate_deduplicated",
        "accepted_models",
        "admitted_selfplay",
        "rollout_quarantine",
        "rollback_quarantine",
        "trainer_checkpoint",
        "evaluations",
        "reports",
        "suites",
        "policy_path",
        "powered_config_path",
        "standard_config_path",
        "discovery_schedule_path",
        "confirmation_schedule_path",
        "audit_schedule_path",
        "lead40_schedule_path",
        "lead80_schedule_path",
        "standard_confirmation_schedule_path",
        "selfplay_config_path",
        "gpu_lease_config_path",
        "data_watermark_path",
        "shuffle_watermark_path",
        "worker_ack_inbox",
        "rollout_report_inbox",
        "original_model_path",
    )
    resolved_root = Path(root).resolve()
    escaped = [
        name
        for name in path_attributes
        if not _is_relative_to(
            Path(getattr(runtime, name)).resolve(strict=False), resolved_root
        )
    ]
    if escaped:
        raise DrillError(
            "disposable runtime paths escape root: " + ", ".join(sorted(escaped))
        )


def _materialize_runtime(
    spec: DrillSpec,
    production: RuntimeConfig,
    scenario_root: Path,
    *,
    mutation_enabled: bool = True,
) -> Tuple[RuntimeConfig, Path]:
    mapping, champion_model, _ = _runtime_mapping(
        spec,
        production,
        scenario_root,
        mutation_enabled=mutation_enabled,
    )
    runtime_path = Path(scenario_root) / "runtime" / "promotion-runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(runtime_path, mapping)
    runtime = RuntimeConfig.load(runtime_path)
    _assert_disposable_runtime(runtime, scenario_root)
    controller = PromotionController(runtime, automatic=mutation_enabled)
    controller.validate_static_inputs()
    champion = load_champion(production.champion_path)
    if mutation_enabled:
        controller.bootstrap(
            champion.champion_hash,
            champion.generation_id,
            confirmation="BOOTSTRAP_INITIAL_CHAMPION",
        )
    initial_leaf = (
        runtime.accepted_models
        / "generations"
        / champion.champion_hash
        / champion.generation_id
        / "model.bin.gz"
    )
    _copy_regular_file(champion_model, initial_leaf)
    os.chmod(initial_leaf, 0o444)
    return runtime, runtime_path


class BoundCommands:
    """Execute only the module/interpreter pair pinned by the drill spec."""

    def __init__(self, spec: DrillSpec, root: Path) -> None:
        self.spec = spec
        self.root = Path(root).resolve()
        self.receipts = self.root / "control" / "command-receipts"
        self.receipts.mkdir(parents=True, exist_ok=True)

    def _record(
        self,
        role: str,
        argv: Sequence[str],
        child: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        command_hash = canonical_sha256(list(argv))
        existing = sorted(self.receipts.glob("*.json"))
        path = self.receipts / f"{len(existing) + 1:06d}.json"
        value = {
            "schema_version": SCHEMA_VERSION,
            "contract": COMMAND_RECEIPT_CONTRACT,
            "role": role,
            "argv_sha256": command_hash,
            "module_sha256": self.spec.module_sha256,
            "child": json.loads(canonical_json_bytes(child)),
        }
        value["receipt_sha256"] = canonical_sha256(value)
        _write_once(path, value)
        return value

    def run(self, argv: Sequence[str], *, role: str) -> Mapping[str, Any]:
        command = tuple(str(item) for item in argv)
        if (
            len(command) < 2
            or command[0] != str(self.spec.python_executable)
            or command[1] != str(self.spec.module_path)
        ):
            raise DrillError(f"{role} command is not bound to the drill module")
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        package_root = str(self.spec.module_path.parent.parent)
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            package_root
            if not inherited_pythonpath
            else package_root + os.pathsep + inherited_pythonpath
        )
        temporary_root = self.root / "control" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        environment["TMPDIR"] = str(temporary_root)
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=False,
            shell=False,
            timeout=self.spec.command_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise DrillError(
                f"{role} dummy command failed with {completed.returncode}: {stderr}"
            )
        try:
            child = _decode_json(completed.stdout, self.spec.module_path)
        except DrillError as exc:
            raise DrillError(f"{role} command returned invalid receipt: {exc}") from exc
        if not isinstance(child, Mapping):
            raise DrillError(f"{role} command receipt is not an object")
        self._record(role, command, child)
        return dict(child)

    def controller_executor(self, argv: Sequence[str]) -> Mapping[str, Any]:
        role = "worker" if "__dummy-worker" in argv else "supervisor"
        receipt = self.run(argv, role=role)
        command_hash = canonical_sha256(list(argv))
        result: Dict[str, Any] = {
            "returncode": 0,
            "process_identity": {
                "pid": receipt.get("pid", 1),
                "start_time_ticks": receipt.get("start_time_ticks", 0),
                "command_sha256": command_hash,
            },
            "process_identity_verified": True,
        }
        if receipt.get("contract") == SUPERVISOR_RECEIPT_CONTRACT:
            result.update(
                {
                    "quiescent": receipt.get("quiescent"),
                    "closed_file_manifests": receipt.get("closed_file_manifests", []),
                    "process_identities": receipt.get("process_identities", []),
                    "quiescent_roles": receipt.get("quiescent_roles", []),
                }
            )
        return result

    def process_identity_verifier(self, identity: Mapping[str, Any]) -> bool:
        command_hash = identity.get("command_sha256")
        if not isinstance(command_hash, str):
            return False
        for path in sorted(self.receipts.glob("*.json")):
            value = _load_canonical_object(path, "command receipt")
            if (
                value.get("role") == "worker"
                and value.get("argv_sha256") == command_hash
                and value.get("module_sha256") == self.spec.module_sha256
            ):
                return True
        return False

    def _job_spec(
        self,
        *,
        job_id: str,
        candidate_model: Path,
        candidate_hash: str,
        reference_model: Path,
        reference_hash: str,
        schedule_path: Path,
        schedule_hash: str,
        output_root: Path,
    ) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": EVALUATOR_JOB_CONTRACT,
            "root": str(self.root),
            "job_id": job_id,
            "candidate_model": str(Path(candidate_model).resolve()),
            "candidate_sha256": candidate_hash,
            "reference_model": str(Path(reference_model).resolve()),
            "reference_sha256": reference_hash,
            "schedule": str(Path(schedule_path).resolve()),
            "schedule_sha256": schedule_hash,
            "output": str(Path(output_root).resolve()),
            "max_rows": self.spec.max_evaluator_rows,
        }
        payload["job_sha256"] = canonical_sha256(payload)
        return payload

    def run_evaluator(
        self,
        *,
        job_id: str,
        candidate_model: Path,
        candidate_hash: str,
        reference_model: Path,
        reference_hash: str,
        schedule_path: Path,
        schedule_hash: str,
        output_root: Path,
    ) -> Mapping[str, Any]:
        output = Path(output_root).resolve()
        output.mkdir(parents=True, exist_ok=True)
        job = self._job_spec(
            job_id=job_id,
            candidate_model=candidate_model,
            candidate_hash=candidate_hash,
            reference_model=reference_model,
            reference_hash=reference_hash,
            schedule_path=schedule_path,
            schedule_hash=schedule_hash,
            output_root=output,
        )
        job_path = output / "job.json"
        _write_once(job_path, job)
        summary_path = output / "summary.json"
        if not summary_path.exists():
            argv = [
                str(self.spec.python_executable),
                str(self.spec.module_path),
                "--internal-module-sha256",
                self.spec.module_sha256,
                "--internal-root",
                str(self.root),
                "__dummy-evaluator",
                "--job",
                str(job_path),
                "--job-sha256",
                sha256_file(job_path),
            ]
            self.run(argv, role="evaluator")
        summary = _load_canonical_object(summary_path, "dummy evaluator summary")
        self._validate_evaluator_summary(job, summary)
        return summary

    def _validate_evaluator_summary(
        self,
        job: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> None:
        expected_keys = {
            "schema_version",
            "contract",
            "finalized",
            "decision",
            "job_sha256",
            "result_path",
            "result_sha256",
            "moves_path",
            "moves_sha256",
            "game_count",
            "candidate_wins",
            "true_no_results",
            "summary_sha256",
        }
        _exact_keys(summary, expected_keys, "dummy evaluator summary")
        payload = dict(summary)
        supplied = payload.pop("summary_sha256")
        result_path = Path(summary["result_path"])
        moves_path = Path(summary["moves_path"])
        if (
            summary["schema_version"] != SCHEMA_VERSION
            or summary["contract"] != EVALUATOR_SUMMARY_CONTRACT
            or summary["finalized"] is not True
            or supplied != canonical_sha256(payload)
            or summary["job_sha256"] != job["job_sha256"]
            or not _is_relative_to(result_path.resolve(), self.root)
            or not _is_relative_to(moves_path.resolve(), self.root)
            or result_path.is_symlink()
            or moves_path.is_symlink()
            or not result_path.is_file()
            or not moves_path.is_file()
            or sha256_file(result_path) != summary["result_sha256"]
            or sha256_file(moves_path) != summary["moves_sha256"]
        ):
            raise DrillError("dummy evaluator summary identity is invalid")
        rows = _jsonl_rows(result_path, self.spec.max_evaluator_rows)
        if (
            not rows
            or len(rows) != summary["game_count"]
            or summary["game_count"] > self.spec.max_evaluator_rows
            or summary["candidate_wins"] != len(rows)
            or summary["true_no_results"] != 0
            or summary["decision"] != "PASS"
            or any(
                not isinstance(row, Mapping)
                or row.get("noResult") is not False
                or row.get("winner")
                != ("B" if row.get("blackBot") == "candidate" else "W")
                for row in rows
            )
        ):
            raise DrillError("dummy evaluator PASS is not derived from its result rows")

    def evaluation_job(self, job: Any) -> EvaluationArtifacts:
        summary = self.run_evaluator(
            job_id=job.job_id,
            candidate_model=job.candidate_model_path,
            candidate_hash=job.candidate_hash,
            reference_model=job.reference_model_path,
            reference_hash=job.reference_hash,
            schedule_path=job.schedule_path,
            schedule_hash=job.schedule_hash,
            output_root=job.output_root / "bounded-command",
        )
        return EvaluationArtifacts(
            results_path=Path(summary["result_path"]),
            moves_path=Path(summary["moves_path"]),
        )


@dataclass
class Scenario:
    root: Path
    runtime: RuntimeConfig
    runtime_path: Path
    commands: BoundCommands
    artifact: Any
    report_path: Path
    report_hash: str
    generation_id: str = "generation-disposable-drill"

    def controller(
        self,
        *,
        failure_hook: Optional[Callable[[str], None]] = None,
        automatic: bool = True,
        now: Optional[Callable[[], datetime]] = None,
    ) -> PromotionController:
        return PromotionController(
            self.runtime,
            automatic=automatic,
            command_executor=self.commands.controller_executor,
            process_identity_verifier=self.commands.process_identity_verifier,
            failure_hook=failure_hook,
            now=now,
        )

    def promotion_kwargs(self) -> Mapping[str, Any]:
        return {
            "pass_report_path": self.report_path,
            "pass_report_hash": self.report_hash,
            "trainer_checkpoint_hash": sha256_file(self.runtime.trainer_checkpoint),
            "data_watermark_hash": sha256_file(self.runtime.data_watermark_path),
            "shuffle_watermark_hash": sha256_file(self.runtime.shuffle_watermark_path),
        }


def _create_candidate(runtime: RuntimeConfig) -> Any:
    name = "disposable-drill-s500000-d1000000"
    path = runtime.candidate_inbox / name
    path.mkdir(parents=True)
    artifacts = {
        "model.bin.gz": b"disposable-promotion-candidate-model-v1",
        "model.ckpt": b"disposable-promotion-candidate-checkpoint-v1",
    }
    for relative, data in artifacts.items():
        (path / relative).write_bytes(data)
    files = [
        {
            "path": relative,
            "size": len(data),
            "sha256": sha256_bytes(data),
        }
        for relative, data in sorted(artifacts.items())
    ]
    manifest = {
        "schemaVersion": 1,
        "exportContract": "katago-hardened-candidate-publication-v1",
        "requestFingerprintSha256": canonical_sha256(
            {"contract": "disposable-promotion-candidate-v1"}
        ),
        "modelProbePassed": True,
        "candidateName": name,
        "modelName": "disposable-drill-model",
        "sourceCheckpoint": {
            "name": "model.ckpt",
            "size": len(artifacts["model.ckpt"]),
            "sha256": sha256_bytes(artifacts["model.ckpt"]),
        },
        "files": files,
    }
    atomic_write_json(path / "manifest.json", manifest)
    return inspect_candidate(path)


def _gpu_handoff(runtime: RuntimeConfig, evidence_hash: str) -> Mapping[str, Any]:
    return {
        "lease_id": f"disposable-drill-{evidence_hash[:24]}",
        "expected_gpu_uuid": runtime.controller.expected_gpu_uuid,
        "handoff_checkpoint_hash": sha256_file(runtime.trainer_checkpoint),
        "clean_observations": [
            {
                "gpu_uuid": runtime.controller.expected_gpu_uuid,
                "processes": [],
            },
            {
                "gpu_uuid": runtime.controller.expected_gpu_uuid,
                "processes": [],
            },
        ],
        "trainer_restored": True,
        "restored_trainer_identity": {
            "pid": 1,
            "start_time_ticks": 0,
            "command_sha256": canonical_sha256(
                ["disposable", "trainer", runtime.controller.controller_hash]
            ),
        },
    }


def _derive_confirmation(
    controller: PromotionController,
    commands: BoundCommands,
    artifact: Any,
) -> Tuple[Any, Path, str]:
    state = controller.registry.reconstruct()
    champion_hash = state.current_champion_hash
    if champion_hash is None:
        raise DrillError("disposable controller has no bootstrap champion")
    plan = controller.build_evaluation_plan(
        artifact.model_hash,
        champion_hash,
        suite="confirmation",
        stage="stage-3",
        look="look-1",
        topology="7-workers-100-threads",
    )
    champion = load_champion(controller.runtime.champion_path)
    champion_model = (
        controller.runtime.original_model_path
        if champion_hash == controller.runtime.controller.original_hash
        else controller.runtime.accepted_models
        / "generations"
        / champion_hash
        / champion.generation_id
        / "model.bin.gz"
    )
    summaries = []
    artifacts_by_comparison = {
        value["comparison"]: value for value in plan.schedule_artifacts.values()
    }
    for index, evaluation_spec in enumerate(plan.specs):
        schedule = artifacts_by_comparison.get(evaluation_spec.comparison)
        if not isinstance(schedule, Mapping):
            raise DrillError(
                f"confirmation spec lacks schedule: {evaluation_spec.comparison}"
            )
        reference = (
            champion_model
            if evaluation_spec.reference_model_sha == champion_hash
            else controller.runtime.original_model_path
        )
        summary = commands.run_evaluator(
            job_id=f"confirmation-{index:02d}-{evaluation_spec.comparison}",
            candidate_model=artifact.path / "model.bin.gz",
            candidate_hash=artifact.model_hash,
            reference_model=reference,
            reference_hash=evaluation_spec.reference_model_sha,
            schedule_path=Path(schedule["path"]),
            schedule_hash=evaluation_spec.schedule_sha,
            output_root=(
                controller.runtime.evaluations
                / "disposable-confirmation"
                / f"{index:02d}"
            ),
        )
        summaries.append(summary)
    matrix_passed = bool(summaries) and all(
        summary.get("decision") == "PASS" for summary in summaries
    )
    promotion_evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_contract": "risk-score-disposable-promotion-evidence-v1",
        "finalized": True,
        "decision": "PASS" if matrix_passed else "FAIL",
        "candidate_hash": artifact.model_hash,
        "champion_hash": champion_hash,
        "original_hash": controller.runtime.controller.original_hash,
        "evaluation_key": plan.evaluation_key,
        "config_hash": plan.config_hash,
        "schedule_hash": plan.schedule_hash,
        "policy_hash": plan.policy_hash,
        "dummy_evaluator_summaries": summaries,
        "dummy_evaluator_summaries_sha256": canonical_sha256(summaries),
    }
    envelope = build_controller_evidence(plan.to_dict(), promotion_evidence)
    evidence_path = controller.runtime.evaluations / "evidence" / "confirmation.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    publish_controller_evidence(evidence_path, envelope)
    evidence_hash = sha256_file(evidence_path)
    published = _load_canonical_object(
        evidence_path, "published disposable controller evidence"
    )
    if (
        published != envelope
        or not matrix_passed
        or promotion_evidence["decision"] != "PASS"
    ):
        raise DrillError("disposable confirmation evaluator did not derive PASS")
    handoff = _gpu_handoff(controller.runtime, evidence_hash)
    gate = {
        "decision": "PASS",
        "finalized": True,
        "candidate_hash": artifact.model_hash,
        "tested_champion_hash": champion_hash,
        "original_hash": controller.runtime.controller.original_hash,
        "evaluation_key": plan.evaluation_key,
        "config_hash": plan.config_hash,
        "schedule_hash": plan.schedule_hash,
        "policy_hash": plan.policy_hash,
        "selfplay_config_hash": plan.selfplay_config_hash,
        "topology": plan.topology,
        "gpu_handoff_hash": canonical_sha256(handoff),
        "gpu_handoff": handoff,
        "checks": [
            {
                "name": "bounded-dummy-evaluation",
                "passed": matrix_passed,
                "evidence_path": str(evidence_path),
                "evidence_sha256": evidence_hash,
            }
        ],
    }
    provenance = controller._provenance(plan.config_hash, plan.schedule_hash)
    record = controller.registry.reconstruct().candidates.get(artifact.model_hash)
    if record is None:
        raise DrillError("candidate was not claimed before confirmation")
    candidate_path = record.candidate_path
    transitions = (
        (CandidateState.EVALUATING_INTEGRITY, "drill-integrity"),
        (CandidateState.EVALUATING_SCREEN, "drill-screen"),
        (CandidateState.EVALUATING_FINALIST, "drill-finalist"),
        (CandidateState.EVALUATING_CONFIRMATION, plan.evaluation_key),
    )
    with controller._writer_lock():
        for target, evaluation_key in transitions:
            controller.registry.transition_candidate(
                artifact.model_hash,
                candidate_path,
                target,
                provenance=provenance,
                champion_hash=champion_hash,
                evaluation_key=evaluation_key,
                reason=f"disposable evidence advanced candidate to {target.value}",
                actor=controller.runtime.controller.actor,
            )
    report_path, report_hash, _ = controller.finalize_gate_report(
        plan,
        candidate_hash=artifact.model_hash,
        tested_champion_hash=champion_hash,
        gate_result=gate,
    )
    with controller._writer_lock():
        controller.registry.transition_candidate(
            artifact.model_hash,
            candidate_path,
            CandidateState.CONFIRMED,
            provenance=provenance,
            champion_hash=champion_hash,
            evaluation_key=plan.evaluation_key,
            reason="bounded disposable evaluation derived confirmation PASS",
            actor=controller.runtime.controller.actor,
            payload={
                "report_hash": report_hash,
                "evidence_hash": evidence_hash,
            },
        )
    final_record = controller.registry.reconstruct().candidates[artifact.model_hash]
    if final_record.state != CandidateState.CONFIRMED:
        raise DrillError("candidate confirmation did not enter registry state")
    return plan, report_path, report_hash


def _prepare_scenario(
    spec: DrillSpec,
    production: RuntimeConfig,
    root: Path,
) -> Scenario:
    scenario_root = Path(root).resolve()
    if scenario_root.exists():
        raise DrillError(f"scenario root already exists: {scenario_root}")
    scenario_root.mkdir(parents=True)
    runtime, runtime_path = _materialize_runtime(
        spec, production, scenario_root, mutation_enabled=True
    )
    commands = BoundCommands(spec, scenario_root)
    controller = PromotionController(
        runtime,
        automatic=True,
        command_executor=commands.controller_executor,
        process_identity_verifier=commands.process_identity_verifier,
    )
    artifact = _create_candidate(runtime)
    scan = controller.run_once()
    state = controller.registry.reconstruct()
    record = state.candidates.get(artifact.model_hash)
    if (
        artifact.name not in scan.get("inventory", {}).get("selected", [])
        or record is None
        or record.state != CandidateState.CLAIMED
    ):
        raise DrillError("disposable candidate was not selected and claimed")
    claimed_artifact = inspect_candidate(Path(record.candidate_path))
    _, report_path, report_hash = _derive_confirmation(
        controller, commands, claimed_artifact
    )
    return Scenario(
        root=scenario_root,
        runtime=runtime,
        runtime_path=runtime_path,
        commands=commands,
        artifact=claimed_artifact,
        report_path=report_path,
        report_hash=report_hash,
    )


def _load_scenario(spec: DrillSpec, root: Path) -> Scenario:
    scenario_root = Path(root).resolve()
    runtime_path = scenario_root / "runtime" / "promotion-runtime.json"
    runtime = RuntimeConfig.load(runtime_path)
    _assert_disposable_runtime(runtime, scenario_root)
    commands = BoundCommands(spec, scenario_root)
    controller = PromotionController(
        runtime,
        automatic=True,
        command_executor=commands.controller_executor,
        process_identity_verifier=commands.process_identity_verifier,
    )
    state = controller.registry.reconstruct()
    confirmed = [
        record
        for record in state.candidates.values()
        if record.state == CandidateState.CONFIRMED
    ]
    if len(confirmed) != 1 or confirmed[0].evaluation_key is None:
        raise DrillError("scenario template has no unique confirmed candidate")
    artifact = inspect_candidate(Path(confirmed[0].candidate_path))
    report_path = runtime.reports / f"{confirmed[0].evaluation_key}.final.json"
    if not report_path.is_file():
        raise DrillError("scenario template confirmation report is missing")
    return Scenario(
        root=scenario_root,
        runtime=runtime,
        runtime_path=runtime_path,
        commands=commands,
        artifact=artifact,
        report_path=report_path,
        report_hash=sha256_file(report_path),
    )


def _acknowledge_worker(
    scenario: Scenario,
    controller: PromotionController,
    worker_id: int,
) -> None:
    generation_id = scenario.generation_id
    existing = (
        scenario.runtime.rollout_quarantine
        / generation_id
        / "acknowledgements"
        / f"worker-{worker_id:03d}.json"
    )
    if existing.is_file():
        return
    worker_root = (
        scenario.runtime.rollout_quarantine / generation_id / f"worker-{worker_id:03d}"
    )
    launch_paths = [
        path
        for path in sorted(worker_root.glob("launch-*.json"))
        if not path.name.endswith(".intent.json")
    ]
    if len(launch_paths) != 1:
        raise DrillError(f"worker {worker_id} has no unique launch marker")
    launch = _load_canonical_object(launch_paths[0], "worker launch marker")
    output = (
        scenario.runtime.rollout_quarantine
        / generation_id
        / "data"
        / f"worker-{worker_id:03d}"
    )
    manifest_hash, output_size, _ = tree_manifest(output)
    if output_size <= 0:
        raise DrillError(f"worker {worker_id} produced no closed output")
    report = {
        "schema_version": SCHEMA_VERSION,
        "finalized": True,
        "generation_id": generation_id,
        "worker_id": worker_id,
        "model_hash": scenario.artifact.model_hash,
        "selfplay_config_hash": scenario.runtime.controller.selfplay_config_hash,
        "policy_hash": scenario.runtime.controller.policy_hash,
        "threads": scenario.runtime.controller.worker_threads,
        "output_manifest_hash": manifest_hash,
        "closed_files": True,
        "process_identity": launch["process_identity"],
    }
    scenario.runtime.worker_ack_inbox.mkdir(parents=True, exist_ok=True)
    report_path = (
        scenario.runtime.worker_ack_inbox
        / f"{generation_id}.worker-{worker_id:03d}.json"
    )
    atomic_write_json(report_path, report)
    controller.record_worker_ack(
        generation_id,
        worker_id,
        scenario.artifact.model_hash,
        report_path=report_path,
        report_hash=sha256_file(report_path),
    )


def _auditor(scenario: Scenario) -> PromotionAuditor:
    lease_proof = {
        "schema_version": SCHEMA_VERSION,
        "contract": "risk-score-disposable-auditor-lease-v1",
        "trainer_restored": True,
        "exclusive": True,
    }
    return PromotionAuditor(
        AuditorRuntime.from_controller_runtime(scenario.runtime),
        evaluation_executor=scenario.commands.evaluation_job,
        lease_factory=lambda: contextlib.nullcontext(lease_proof),
        shards=1,
        max_parallel=1,
        max_attempts=1,
    )


def _mark_phase_passed(
    scenario: Scenario,
    controller: PromotionController,
    phase: str,
) -> Mapping[str, Any]:
    transaction = (
        scenario.runtime.promotion_root / "transactions" / scenario.generation_id
    )
    marker = transaction / (
        "canary-pass.json" if phase == "canary" else "intermediate-pass.json"
    )
    if marker.exists():
        return _load_canonical_object(marker, f"{phase} PASS marker")
    result = _auditor(scenario).produce_rollout_report(
        scenario.generation_id,
        phase,
        raise_on_failure=True,
    )
    if result.decision != "PASS":
        raise DrillError(f"{phase} auditor did not derive PASS")
    kwargs = {
        "report_path": result.output_path,
        "report_hash": result.output_sha256,
    }
    if phase == "canary":
        controller.mark_canary_passed(
            scenario.generation_id,
            scenario.artifact.model_hash,
            **kwargs,
        )
    else:
        controller.mark_intermediate_passed(
            scenario.generation_id,
            scenario.artifact.model_hash,
            **kwargs,
        )
    return _load_canonical_object(marker, f"{phase} PASS marker")


def _ack_canary(scenario: Scenario, controller: PromotionController) -> None:
    for worker_id in range(scenario.runtime.controller.canary_worker_count):
        _acknowledge_worker(scenario, controller, worker_id)
    _mark_phase_passed(scenario, controller, "canary")


def _ack_intermediate(scenario: Scenario, controller: PromotionController) -> None:
    for worker_id in range(
        scenario.runtime.controller.canary_worker_count,
        scenario.runtime.controller.intermediate_worker_count,
    ):
        _acknowledge_worker(scenario, controller, worker_id)
    _mark_phase_passed(scenario, controller, "intermediate")


def _ack_full(scenario: Scenario, controller: PromotionController) -> None:
    for worker_id in range(
        scenario.runtime.controller.intermediate_worker_count,
        scenario.runtime.controller.worker_count,
    ):
        _acknowledge_worker(scenario, controller, worker_id)


def _converge_promotion(
    spec: DrillSpec,
    scenario: Scenario,
    controller: PromotionController,
    kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    for _ in range(spec.max_replay_attempts):
        result = controller.promote(
            scenario.artifact.model_hash,
            scenario.generation_id,
            **kwargs,
        )
        status = result.get("status")
        if status == "ACTIVE":
            return result
        if status in {"WAITING_CANARY_ACK", "WAITING_CANARY_ADMISSION"}:
            _ack_canary(scenario, controller)
        elif status in {
            "WAITING_INTERMEDIATE_ACK",
            "WAITING_INTERMEDIATE_HEALTH",
        }:
            _ack_intermediate(scenario, controller)
        elif status == "WAITING_ROLLOUT_ACK":
            _ack_full(scenario, controller)
        elif status == "WAITING_GENERATION_DATA":
            continue
        else:
            raise DrillError(f"promotion entered unexpected status {status!r}")
    raise DrillError("promotion did not converge within the bounded replay limit")


def _active_observation(scenario: Scenario) -> Mapping[str, Any]:
    controller = scenario.controller()
    state = controller.registry.reconstruct()
    generation = state.generations.get(scenario.generation_id)
    champion = load_champion(scenario.runtime.champion_path)
    transaction = (
        scenario.runtime.promotion_root / "transactions" / scenario.generation_id
    )
    admitted = scenario.runtime.admitted_selfplay / scenario.generation_id
    if (
        generation is None
        or generation.state != GenerationState.ACTIVE
        or state.current_champion_hash != scenario.artifact.model_hash
        or champion.champion_hash != scenario.artifact.model_hash
        or not (transaction / "complete.json").is_file()
        or not admitted.is_dir()
    ):
        raise DrillError(
            "replayed promotion did not converge to committed ACTIVE state"
        )
    return {
        "generation_state": generation.state.value,
        "champion_sha256": champion.champion_hash,
        "last_sequence": state.last_sequence,
        "last_event_sha256": state.last_event_hash,
        "transaction_complete_sha256": sha256_file(transaction / "complete.json"),
        "admitted_manifest_sha256": _snapshot_path(admitted)["sha256"],
    }


def _run_canary_gate(
    spec: DrillSpec,
    production: RuntimeConfig,
    root: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenario = _prepare_scenario(spec, production, root / "scenario")
    controller = scenario.controller()
    kwargs = scenario.promotion_kwargs()
    first = controller.promote(
        scenario.artifact.model_hash,
        scenario.generation_id,
        **kwargs,
    )
    if first.get("status") != "WAITING_CANARY_ACK":
        raise DrillError("canary drill did not stop at the canary acknowledgement gate")
    _ack_canary(scenario, controller)
    second = controller.promote(
        scenario.artifact.model_hash,
        scenario.generation_id,
        **kwargs,
    )
    state = controller.registry.reconstruct()
    generation = state.generations.get(scenario.generation_id)
    report_path = (
        scenario.runtime.rollout_report_inbox / f"{scenario.generation_id}.canary.json"
    )
    report = _load_canonical_object(report_path, "canary rollout report")
    fresh_path = Path(report.get("fresh_audit_manifest_path", ""))
    fresh = _load_canonical_object(fresh_path, "canary fresh-audit manifest")
    canary_passed = bool(
        generation is not None
        and generation.state == GenerationState.ROLLOUT
        and report.get("decision") == "PASS"
        and (
            scenario.runtime.promotion_root
            / "transactions"
            / scenario.generation_id
            / "canary-admitted.json"
        ).is_file()
    )
    fresh_passed = bool(
        fresh.get("decision") == "PASS"
        and fresh.get("finalized") is True
        and report.get("fresh_audit_manifest_sha256") == sha256_file(fresh_path)
    )
    if not canary_passed or not fresh_passed:
        raise DrillError("canary state or fresh-audit artifacts did not derive PASS")
    checks = {
        "canary_passed": canary_passed,
        "fresh_audit_passed": fresh_passed,
    }
    observations = {
        "initial_promotion_status": first,
        "post_canary_status": second,
        "generation_state": generation.state.value,
        "canary_report_path": str(report_path),
        "canary_report_sha256": sha256_file(report_path),
        "fresh_audit_path": str(fresh_path),
        "fresh_audit_sha256": sha256_file(fresh_path),
        "event_log": _snapshot_path(scenario.runtime.promotion_root / "events"),
    }
    return checks, observations


def _advance_before_failure(
    failure_step: str,
    scenario: Scenario,
    controller: PromotionController,
    kwargs: Mapping[str, Any],
) -> None:
    late_steps = {
        "promotion-canary-admitted",
        "promotion-rollout-event",
        "promotion-intermediate-passed",
        "promotion-all-workers-acknowledged",
        "promotion-generation-data-admitted",
        "promotion-champion-cas",
        "promotion-active-event",
    }
    if failure_step not in late_steps:
        return
    first = controller.promote(
        scenario.artifact.model_hash,
        scenario.generation_id,
        **kwargs,
    )
    if first.get("status") != "WAITING_CANARY_ACK":
        raise DrillError("crash fixture did not reach canary acknowledgement")
    _ack_canary(scenario, controller)
    if failure_step not in {
        "promotion-canary-admitted",
        "promotion-rollout-event",
    }:
        controller.promote(
            scenario.artifact.model_hash,
            scenario.generation_id,
            **kwargs,
        )
        _ack_intermediate(scenario, controller)
    if failure_step in {
        "promotion-all-workers-acknowledged",
        "promotion-generation-data-admitted",
        "promotion-champion-cas",
        "promotion-active-event",
    }:
        controller.promote(
            scenario.artifact.model_hash,
            scenario.generation_id,
            **kwargs,
        )
        _ack_full(scenario, controller)


def _run_crash_gate(
    spec: DrillSpec,
    production: RuntimeConfig,
    root: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    case_root = root / "case"
    template_root = root / "template"
    _prepare_scenario(spec, production, case_root)
    _copy_regular_tree(case_root, template_root)
    _remove_tree(case_root)
    boundaries = []
    details = []
    for index, failure_step in enumerate(PROMOTION_FAILURE_STEPS):
        _copy_regular_tree(template_root, case_root)
        scenario = _load_scenario(spec, case_root)
        kwargs = scenario.promotion_kwargs()
        base = scenario.controller()
        _advance_before_failure(failure_step, scenario, base, kwargs)
        fired: List[str] = []

        def fail(
            step: str,
            expected_step: str = failure_step,
            fired_steps: List[str] = fired,
        ) -> None:
            if step == expected_step and not fired_steps:
                fired_steps.append(step)
                raise InjectedPromotionCrash(step)

        crashing = scenario.controller(failure_hook=fail)
        injected = False
        try:
            crashing.promote(
                scenario.artifact.model_hash,
                scenario.generation_id,
                **kwargs,
            )
        except InjectedPromotionCrash as exc:
            injected = str(exc) == failure_step and fired == [failure_step]
        if not injected:
            raise DrillError(f"failure boundary was not injected: {failure_step}")
        recovered = scenario.controller()
        reconcile_before = recovered.run_reconcile()
        _converge_promotion(spec, scenario, recovered, kwargs)
        active = _active_observation(scenario)
        event_before = recovered.registry.reconstruct().last_event_hash
        reconcile_after = recovered.run_reconcile()
        event_after = recovered.registry.reconstruct().last_event_hash
        replay_converged = bool(
            active["generation_state"] == GenerationState.ACTIVE.value
            and event_before == event_after
            and reconcile_after.get("championHash") == scenario.artifact.model_hash
        )
        boundaries.append(
            {
                "step": failure_step,
                "crash_injected": injected,
                "replay_converged": replay_converged,
            }
        )
        details.append(
            {
                "index": index,
                "step": failure_step,
                "reconcile_before": reconcile_before,
                "reconcile_after": reconcile_after,
                "active": active,
                "command_receipts": _snapshot_path(scenario.commands.receipts),
            }
        )
        _remove_tree(case_root)
    if [item["step"] for item in boundaries] != list(PROMOTION_FAILURE_STEPS):
        raise DrillError("crash boundary coverage differs from controller contract")
    if any(
        item["crash_injected"] is not True or item["replay_converged"] is not True
        for item in boundaries
    ):
        raise DrillError("one or more promotion crash boundaries did not converge")
    return {"boundaries": boundaries}, {"boundary_details": details}


def _run_rollback_before_gate(
    spec: DrillSpec,
    production: RuntimeConfig,
    root: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenario = _prepare_scenario(spec, production, root / "scenario")
    controller = scenario.controller()
    kwargs = scenario.promotion_kwargs()
    before_champion = load_champion(scenario.runtime.champion_path).champion_hash
    result = controller.promote(
        scenario.artifact.model_hash,
        scenario.generation_id,
        **kwargs,
    )
    staged = scenario.runtime.rollout_quarantine / scenario.generation_id / "data"
    staged_before = _snapshot_path(staged)
    requested = False
    refused = False
    refusal = None
    try:
        requested = True
        controller.rollback(scenario.generation_id)
    except SafetyHalt as exc:
        refusal = str(exc)
        generation = controller.registry.reconstruct().generations.get(
            scenario.generation_id
        )
        refused = bool(
            "forensic" in refusal
            and generation is not None
            and generation.state == GenerationState.CANARY
        )
    staged_after = _snapshot_path(staged)
    champion_after = load_champion(scenario.runtime.champion_path).champion_hash
    checks = {
        "rollback_requested": requested,
        "refused_without_forensic_flow": refused,
        "staged_data_preserved": staged_before == staged_after
        and staged_after["kind"] == "directory",
        "champion_unchanged": before_champion == champion_after,
    }
    if any(value is not True for value in checks.values()):
        raise DrillError("pre-admission rollback did not fail closed")
    return checks, {
        "promotion_status": result,
        "rollback_error": refusal,
        "staged_before": staged_before,
        "staged_after": staged_after,
        "generation_state": controller.registry.reconstruct()
        .generations[scenario.generation_id]
        .state.value,
    }


def _run_rollback_after_gate(
    spec: DrillSpec,
    production: RuntimeConfig,
    root: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenario = _prepare_scenario(spec, production, root / "scenario")
    controller = scenario.controller()
    derived = scenario.root / "derived" / "shuffle-generation"
    derived.mkdir(parents=True)
    (derived / "chunk.bin").write_bytes(b"candidate-derived-shuffle-v1")
    atomic_write_json(
        scenario.runtime.shuffle_watermark_path,
        {"derived_paths": [str(derived)]},
    )
    kwargs = scenario.promotion_kwargs()
    checkpoint_before = sha256_file(scenario.runtime.trainer_checkpoint)
    data_watermark_before = sha256_file(scenario.runtime.data_watermark_path)
    shuffle_watermark_before = sha256_file(scenario.runtime.shuffle_watermark_path)
    champion_before = load_champion(scenario.runtime.champion_path).champion_hash
    _converge_promotion(spec, scenario, controller, kwargs)
    admitted = scenario.runtime.admitted_selfplay / scenario.generation_id
    admitted_before = _snapshot_path(admitted)
    scenario.runtime.trainer_checkpoint.write_bytes(b"candidate-consumed-checkpoint-v1")
    rollback = controller.rollback(
        scenario.generation_id,
        trainer_consumed=True,
    )
    reconcile = controller.run_reconcile()
    quarantine = scenario.runtime.rollback_quarantine / scenario.generation_id / "data"
    quarantined_admitted = quarantine / "admitted-generation"
    quarantined_shuffle = quarantine / "shuffle-000"
    state = controller.registry.reconstruct()
    generation = state.generations.get(scenario.generation_id)
    champion_after = load_champion(scenario.runtime.champion_path).champion_hash
    checks = {
        "rollback_complete": rollback.get("status") == "ROLLED_BACK"
        and generation is not None
        and generation.state == GenerationState.ROLLED_BACK,
        "champion_restored": champion_after == champion_before
        and state.current_champion_hash == champion_before,
        "checkpoint_restored": sha256_file(scenario.runtime.trainer_checkpoint)
        == checkpoint_before,
        "admitted_data_quarantined": not admitted.exists()
        and quarantined_admitted.is_dir()
        and _snapshot_path(quarantined_admitted) == admitted_before,
        "derived_data_removed": not derived.exists()
        and quarantined_shuffle.is_dir()
        and (quarantined_shuffle / "chunk.bin").is_file(),
        "watermarks_restored": sha256_file(scenario.runtime.data_watermark_path)
        == data_watermark_before
        and sha256_file(scenario.runtime.shuffle_watermark_path)
        == shuffle_watermark_before,
    }
    if any(value is not True for value in checks.values()):
        raise DrillError("post-admission rollback did not restore durable state")
    return checks, {
        "rollback_result": rollback,
        "reconcile_result": reconcile,
        "generation_state": generation.state.value,
        "quarantine": _snapshot_path(quarantine),
        "event_log": _snapshot_path(scenario.runtime.promotion_root / "events"),
    }


def _shadow_recommendation_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove host telemetry that is not part of a controller recommendation."""

    projected = json.loads(canonical_json_bytes(value))
    backpressure = projected.get("backpressure")
    if isinstance(backpressure, dict):
        backpressure.pop("diskFreeBytes", None)
    return projected


def _run_shadow_gate(
    spec: DrillSpec,
    production: RuntimeConfig,
    root: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    scenario = _prepare_scenario(spec, production, root / "scenario")
    runtime_value = _load_canonical_object(scenario.runtime_path, "disposable runtime")
    runtime_value["mutationEnabled"] = False
    shadow_path = scenario.root / "runtime" / "shadow-runtime.json"
    atomic_write_json(shadow_path, runtime_value)
    shadow = RuntimeConfig.load(shadow_path)
    _assert_disposable_runtime(shadow, scenario.root)
    if shadow.controller.mutation_enabled is not False:
        raise DrillError("shadow runtime did not disable mutation")
    before = _snapshot_path(scenario.root)

    def frozen_now() -> datetime:
        return datetime(2099, 1, 1, tzinfo=timezone.utc)

    first_controller = PromotionController(
        shadow,
        automatic=False,
        now=frozen_now,
    )
    second_controller = PromotionController(
        shadow,
        automatic=False,
        now=frozen_now,
    )
    first = first_controller.run_once()
    first_projection = _shadow_recommendation_projection(first)
    first_hash = canonical_sha256(first_projection)
    second = second_controller.run_once()
    second_projection = _shadow_recommendation_projection(second)
    second_hash = canonical_sha256(second_projection)
    after = _snapshot_path(scenario.root)
    event_log = _snapshot_path(shadow.promotion_root / "events")
    if (
        first_controller.recommendation_only is not True
        or second_controller.recommendation_only is not True
        or first.get("mode") != "recommend-only"
        or second.get("mode") != "recommend-only"
        or first_hash != second_hash
        or before != after
    ):
        raise DrillError("independent shadow controller replays diverged or mutated")
    checks = {
        "mutation_enabled": shadow.controller.mutation_enabled,
        "first_replay_sha256": first_hash,
        "second_replay_sha256": second_hash,
        "event_log_sha256": event_log["sha256"],
    }
    return checks, {
        "first_replay": first,
        "second_replay": second,
        "first_recommendation_projection": first_projection,
        "second_recommendation_projection": second_projection,
        "disposable_tree_before": before,
        "disposable_tree_after": after,
        "event_log": event_log,
    }


_GATE_RUNNERS: Mapping[
    str,
    Callable[
        [DrillSpec, RuntimeConfig, Path],
        Tuple[Mapping[str, Any], Mapping[str, Any]],
    ],
] = {
    "disposable-canary-drill": _run_canary_gate,
    "crash-replay-drill": _run_crash_gate,
    "rollback-before-admission-drill": _run_rollback_before_gate,
    "rollback-after-admission-drill": _run_rollback_after_gate,
    "shadow-controller-replay": _run_shadow_gate,
}


def _publish_detail(
    spec: DrillSpec,
    gate_id: str,
    *,
    initial_sentinels: Sequence[Mapping[str, Any]],
    final_sentinels: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any],
    observations: Mapping[str, Any],
    root_snapshot: Mapping[str, Any],
) -> Tuple[Path, Mapping[str, Any]]:
    path = spec.evidence_root / f"{gate_id}.detail.json"
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": DRILL_DETAIL_CONTRACT,
        "gate_id": gate_id,
        "spec_sha256": spec.spec_sha256,
        "production_runtime_sha256": spec.production_runtime_sha256,
        "disposable_root": str(spec.disposable_root),
        "cleanup_state": "ready-for-atomic-removal",
        "initial_sentinels": list(initial_sentinels),
        "final_sentinels": list(final_sentinels),
        "derived_checks": json.loads(canonical_json_bytes(checks)),
        "observations": json.loads(canonical_json_bytes(observations)),
        "disposable_tree": dict(root_snapshot),
    }
    value["detail_sha256"] = canonical_sha256(value)
    _write_once(path, value)
    loaded = _load_canonical_object(path, "finalized drill detail")
    payload = dict(loaded)
    supplied = payload.pop("detail_sha256", None)
    if supplied != canonical_sha256(payload):
        raise DrillError("finalized drill detail self-hash is invalid")
    return path, loaded


def _publish_failure(
    spec: DrillSpec,
    gate_id: str,
    exc: Exception,
) -> None:
    failure_path = spec.evidence_root / f"{gate_id}.failure.json"
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": DRILL_FAILURE_CONTRACT,
        "gate_id": gate_id,
        "spec_sha256": spec.spec_sha256,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "forensic_root": str(spec.disposable_root),
        "forensic_root_present": spec.disposable_root.exists(),
    }
    value["failure_sha256"] = canonical_sha256(value)
    if not failure_path.exists():
        _write_once(failure_path, value)
    try:
        publish_gate_evidence(
            spec.evidence_root / f"{gate_id}.json",
            gate_id,
            {
                "drill_failed": True,
                "failure_evidence": str(failure_path),
                "forensic_root": str(spec.disposable_root),
            },
            decision="FAIL",
        )
    except Exception:
        pass


def _verify_gate_evidence(path: Path, gate_id: str) -> Mapping[str, Any]:
    value = _load_canonical_object(path, "bootstrap gate evidence")
    payload = dict(value)
    supplied = payload.pop("evidence_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("gate_id") != gate_id
        or value.get("decision") != "PASS"
        or supplied != canonical_sha256(payload)
    ):
        raise DrillError("published bootstrap gate evidence is not a derived PASS")
    return value


def _load_replay_detail(
    spec: DrillSpec,
    gate_id: str,
    path: Path,
) -> Mapping[str, Any]:
    value = _load_canonical_object(path, "finalized drill detail")
    expected = {
        "schema_version",
        "contract",
        "gate_id",
        "spec_sha256",
        "production_runtime_sha256",
        "disposable_root",
        "cleanup_state",
        "initial_sentinels",
        "final_sentinels",
        "derived_checks",
        "observations",
        "disposable_tree",
        "detail_sha256",
    }
    if set(value) != expected:
        raise DrillError("finalized drill detail fields differ from the schema")
    payload = dict(value)
    supplied = payload.pop("detail_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != DRILL_DETAIL_CONTRACT
        or value.get("gate_id") != gate_id
        or value.get("spec_sha256") != spec.spec_sha256
        or value.get("production_runtime_sha256")
        != spec.production_runtime_sha256
        or value.get("disposable_root") != str(spec.disposable_root)
        or value.get("cleanup_state") != "ready-for-atomic-removal"
        or supplied != canonical_sha256(payload)
        or not isinstance(value.get("initial_sentinels"), list)
        or not isinstance(value.get("final_sentinels"), list)
        or not isinstance(value.get("derived_checks"), Mapping)
    ):
        raise DrillError("finalized drill detail identity is invalid")
    return value


def _resume_finalization(
    spec: DrillSpec,
    gate_id: str,
    gate_path: Path,
    detail_path: Path,
    failure_path: Path,
) -> Optional[Mapping[str, Any]]:
    if failure_path.exists() or failure_path.is_symlink():
        raise DrillError(f"{gate_id} has preserved failure evidence")
    if not detail_path.exists() and not detail_path.is_symlink():
        if gate_path.exists() or gate_path.is_symlink():
            raise DrillError(f"{gate_id} PASS evidence exists without drill detail")
        return None
    detail = _load_replay_detail(spec, gate_id, detail_path)
    tombstone = (
        spec.disposable_root.parent
        / f".{spec.disposable_root.name}.cleanup-{detail['detail_sha256'][:16]}"
    )
    root_exists = spec.disposable_root.exists() or spec.disposable_root.is_symlink()
    tombstone_exists = tombstone.exists() or tombstone.is_symlink()
    if root_exists and tombstone_exists:
        raise DrillError("both disposable root and cleanup tombstone exist")
    if root_exists:
        if _snapshot_path(spec.disposable_root) != detail["disposable_tree"]:
            raise DrillError("disposable root changed after detail publication")
        os.replace(spec.disposable_root, tombstone)
        fsync_directory(spec.disposable_root.parent)
        tombstone_exists = True
    if tombstone_exists:
        if _snapshot_path(tombstone) != detail["disposable_tree"]:
            raise DrillError("cleanup tombstone changed after detail publication")

    current = _snapshot_sentinels(spec)
    _assert_expected_sentinels(spec, current, phase="finalization replay")
    current_values = list(current)
    if current_values != detail["final_sentinels"]:
        raise DrillError("production sentinels changed after detail publication")
    checks = dict(detail["derived_checks"])
    checks["production_unchanged"] = (
        detail["initial_sentinels"]
        == detail["final_sentinels"]
        == current_values
    )
    if gate_id == "disposable-canary-drill":
        checks["disposable_root_removed"] = not spec.disposable_root.exists()

    if gate_path.exists() or gate_path.is_symlink():
        evidence = _verify_gate_evidence(gate_path, gate_id)
        if evidence.get("checks") != checks:
            raise DrillError("existing PASS evidence contradicts finalized drill detail")
    else:
        evidence = publish_gate_evidence(
            gate_path,
            gate_id,
            checks,
            decision="PASS",
        )
        if _verify_gate_evidence(gate_path, gate_id) != evidence:
            raise DrillError("replayed gate evidence changed after publication")

    if tombstone_exists:
        _remove_tree(tombstone)
        fsync_directory(spec.disposable_root.parent)
    if spec.disposable_root.exists() or tombstone.exists():
        raise DrillError("disposable drill cleanup did not converge")
    return evidence


def run_drill(spec: DrillSpec | Path, gate_id: str) -> Mapping[str, Any]:
    """Run one gate, publish derived evidence, and remove the disposable root."""

    loaded = spec if isinstance(spec, DrillSpec) else DrillSpec.load(Path(spec))
    if gate_id not in DRILL_GATES:
        raise DrillError(f"unknown promotion drill gate: {gate_id}")
    gate_path = loaded.evidence_root / f"{gate_id}.json"
    detail_path = loaded.evidence_root / f"{gate_id}.detail.json"
    failure_path = loaded.evidence_root / f"{gate_id}.failure.json"
    replayed = _resume_finalization(
        loaded,
        gate_id,
        gate_path,
        detail_path,
        failure_path,
    )
    if replayed is not None:
        return replayed
    production = _validate_run_environment(loaded)
    initial = _snapshot_sentinels(loaded)
    _assert_expected_sentinels(loaded, initial, phase="preflight")
    loaded.disposable_root.mkdir(mode=0o700)
    fsync_directory(loaded.disposable_root.parent)
    tombstone: Optional[Path] = None
    try:
        checks, observations = _GATE_RUNNERS[gate_id](
            loaded,
            production,
            loaded.disposable_root,
        )
        production_device = _existing_ancestor(production.promotion_root).stat().st_dev
        disposable_device = loaded.disposable_root.stat().st_dev
        if production_device != disposable_device:
            raise DrillError("disposable runtime moved to a different filesystem")
        _assert_command_bindings(loaded)
        observations = {
            **dict(observations),
            "isolation": {
                "production_device": production_device,
                "disposable_device": disposable_device,
                "same_filesystem": production_device == disposable_device,
                "runtime_paths_disposable": True,
                "python_executable_sha256": loaded.python_executable_sha256,
                "drill_module_sha256": loaded.module_sha256,
            },
        }
        final = _snapshot_sentinels(loaded)
        _assert_expected_sentinels(loaded, final, phase="drill execution")
        complete_checks = dict(checks)
        complete_checks["production_unchanged"] = initial == final
        root_snapshot = _snapshot_path(loaded.disposable_root)
        _, detail = _publish_detail(
            loaded,
            gate_id,
            initial_sentinels=initial,
            final_sentinels=final,
            checks=complete_checks,
            observations=observations,
            root_snapshot=root_snapshot,
        )
        tombstone = (
            loaded.disposable_root.parent
            / f".{loaded.disposable_root.name}.cleanup-{detail['detail_sha256'][:16]}"
        )
        if tombstone.exists() or tombstone.is_symlink():
            raise DrillError(f"cleanup tombstone already exists: {tombstone}")
        os.replace(loaded.disposable_root, tombstone)
        fsync_directory(loaded.disposable_root.parent)
        if loaded.disposable_root.exists():
            raise DrillError("atomic disposable-root removal did not take effect")
        final_after_rename = _snapshot_sentinels(loaded)
        _assert_command_bindings(loaded)
        _assert_expected_sentinels(
            loaded,
            final_after_rename,
            phase="evidence finalization",
        )
        if gate_id == "disposable-canary-drill":
            complete_checks[
                "disposable_root_removed"
            ] = not loaded.disposable_root.exists()
        evidence = publish_gate_evidence(
            gate_path,
            gate_id,
            complete_checks,
            decision="PASS",
        )
        verified = _verify_gate_evidence(gate_path, gate_id)
        if verified != evidence:
            raise DrillError("published gate evidence changed after finalization")
        _remove_tree(tombstone)
        tombstone = None
        fsync_directory(loaded.disposable_root.parent)
        if loaded.disposable_root.exists():
            raise DrillError("disposable root reappeared after cleanup")
        return evidence
    except Exception as exc:
        if (
            tombstone is not None
            and tombstone.exists()
            and not loaded.disposable_root.exists()
        ):
            try:
                os.replace(tombstone, loaded.disposable_root)
                fsync_directory(loaded.disposable_root.parent)
                tombstone = None
            except OSError:
                pass
        _publish_failure(loaded, gate_id, exc)
        raise DrillError(
            f"{gate_id} failed; forensic root preserved at "
            f"{loaded.disposable_root}: {exc}"
        ) from exc


def _internal_common(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-module-sha256", required=True)
    parser.add_argument("--internal-root", required=True, type=Path)
    args, remaining = parser.parse_known_args(list(argv))
    expected = _require_sha256(args.internal_module_sha256, "internal module hash")
    module = Path(__file__).resolve()
    root = args.internal_root.resolve()
    if sha256_file(module) != expected or not root.is_dir() or root.is_symlink():
        raise DrillError("internal command module/root binding is invalid")
    return args, remaining


def _internal_path(value: str, root: Path, role: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != str(path.resolve(strict=False))
        or not _is_relative_to(path.resolve(strict=False), root)
    ):
        raise DrillError(f"{role} escapes the disposable root")
    _assert_no_symlink_ancestors(path, include_leaf=False)
    return path


def _emit_internal(value: Mapping[str, Any]) -> int:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")
    sys.stdout.buffer.flush()
    return 0


def _dummy_worker_main(common: argparse.Namespace, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="__dummy-worker")
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--games", required=True, type=int)
    parser.add_argument("--maximum-games", required=True, type=int)
    args = parser.parse_args(list(argv))
    root = common.internal_root.resolve()
    output = _internal_path(args.output, root, "worker output")
    model = _internal_path(args.model, root, "worker model")
    config = _internal_path(args.config, root, "worker config")
    model_hash = _require_sha256(args.model_sha256, "worker model hash")
    config_hash = _require_sha256(args.config_sha256, "worker config hash")
    _require_sha256(args.policy_sha256, "worker policy hash")
    if (
        args.worker_id < 0
        or not args.generation_id
        or "/" in args.generation_id
        or "\\" in args.generation_id
        or type(args.games) is not int
        or type(args.maximum_games) is not int
        or not 1 <= args.games <= args.maximum_games <= 100000
        or model.is_symlink()
        or config.is_symlink()
        or not model.is_file()
        or not config.is_file()
        or sha256_file(model) != model_hash
        or sha256_file(config) != config_hash
    ):
        raise DrillError("worker inputs do not match their bounded hash bindings")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DrillError("worker output directory is not empty")
    games_path = output / "games.sgfs"
    with games_path.open("xb") as handle:
        for index in range(args.games):
            game = (
                f"(;GM[1]FF[4]SZ[19]RE[B+{index + 1}.5]"
                f"C[{args.generation_id}-worker-{args.worker_id}-game-{index}])\n"
            ).encode("utf-8")
            handle.write(game)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(output)
    started = time.monotonic_ns()
    return _emit_internal(
        {
            "schema_version": SCHEMA_VERSION,
            "contract": WORKER_RECEIPT_CONTRACT,
            "finalized": True,
            "generation_id": args.generation_id,
            "worker_id": args.worker_id,
            "model_sha256": model_hash,
            "config_sha256": config_hash,
            "game_count": args.games,
            "output_manifest_sha256": tree_manifest(output)[0],
            "pid": os.getpid(),
            "start_time_ticks": started,
        }
    )


def _dummy_supervisor_main(common: argparse.Namespace, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="__dummy-supervisor")
    parser.add_argument(
        "--action",
        required=True,
        choices=("trainer", "stage0", "evaluator", "drain", "rollback"),
    )
    parser.add_argument("--generation-id")
    parser.add_argument("--manifest")
    args = parser.parse_args(list(argv))
    root = common.internal_root.resolve()
    closed: List[Any] = []
    identities: List[Any] = []
    if args.action == "drain":
        if args.manifest is None:
            raise DrillError("drain command requires a manifest")
        manifest_path = _internal_path(args.manifest, root, "drain manifest")
        manifest = _load_canonical_object(manifest_path, "drain manifest")
        closed_value = manifest.get("closed_file_manifests")
        identities_value = manifest.get("process_identities")
        if not isinstance(closed_value, list) or not isinstance(identities_value, list):
            raise DrillError("drain manifest lacks closed output proofs")
        closed = closed_value
        identities = identities_value
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "contract": SUPERVISOR_RECEIPT_CONTRACT,
        "finalized": True,
        "action": args.action,
        "generation_id": args.generation_id,
        "quiescent": True,
        "closed_file_manifests": closed,
        "process_identities": identities,
        "quiescent_roles": (
            list(_ALL_QUIESCENT_ROLES) if args.action == "rollback" else []
        ),
        "pid": os.getpid(),
        "start_time_ticks": time.monotonic_ns(),
    }
    return _emit_internal(receipt)


def _jsonl_rows(path: Path, maximum: int) -> List[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DrillError(f"JSONL input is not a regular file: {path}")
    rows = []
    with path.open("rb") as handle:
        for index, raw_line in enumerate(handle):
            if index >= maximum:
                raise DrillError("evaluator JSONL exceeds its bounded row limit")
            if not raw_line.endswith(b"\n") or len(raw_line) > _MAX_JSONL_LINE_BYTES:
                raise DrillError("evaluator JSONL has an unterminated or oversized row")
            line = raw_line[:-1]
            if not line:
                raise DrillError("evaluator JSONL contains a blank row")
            value = _decode_json(line, path)
            if not isinstance(value, Mapping):
                raise DrillError("evaluator JSONL row is not an object")
            rows.append(dict(value))
    if not rows:
        raise DrillError(f"JSONL input is empty: {path}")
    return rows


def _result_for_schedule(row: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate_black = row.get("blackBot") == 0
    winner = "B" if candidate_black else "W"
    result: Dict[str, Any] = {
        "schemaVersion": 1,
        "scheduleId": row["scheduleId"],
        "gameId": row["gameId"],
        "pairId": row["pairId"],
        "positionId": row["positionId"],
        "seed": row["seed"],
        "blackBot": "candidate" if candidate_black else "reference",
        "whiteBot": "reference" if candidate_black else "candidate",
        "blackBotIndex": row["blackBot"],
        "whiteBotIndex": row["whiteBot"],
        "board": {
            "xSize": row["startPosition"]["xSize"],
            "ySize": row["startPosition"]["ySize"],
        },
        "rules": {"ko": "POSITIONAL", "scoring": "AREA"},
        "komi": 7.5,
        "finalResult": f"{winner}+1",
        "finalWhiteMinusBlackScore": -1.0 if winner == "B" else 1.0,
        "winner": winner,
        "moveCount": 2,
        "blackMoveCount": 1,
        "whiteMoveCount": 1,
        "startTurnNumber": row["startPosition"]["initialTurnNumber"],
        "hitTurnLimit": False,
        "resignation": False,
        "noResult": False,
        "scored": True,
        "gameHash": "dummy-"
        + hashlib.sha256(str(row["gameId"]).encode("utf-8")).hexdigest(),
    }
    for field in (
        "suite",
        "suiteBank",
        "suiteBankSha256",
        "suiteQualifiedName",
        "suiteHoldout",
        "positionContentSha256",
        "positionSemanticSha256",
        "independentClusterId",
    ):
        if field in row:
            result[field] = row[field]
    return result


def _moves_for_schedule(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    return [
        {
            "schemaVersion": 1,
            "scheduleId": row["scheduleId"],
            "gameId": row["gameId"],
            "pairId": row["pairId"],
            "positionId": row["positionId"],
            "seed": row["seed"],
            "turnNumber": result["startTurnNumber"] + offset,
            "player": player,
            "bot": (result["blackBot"] if player == "B" else result["whiteBot"]),
            "move": "D4" if offset == 0 else "Q16",
            "scoreLead": 0.0,
            "winProbability": 0.5,
        }
        for offset, player in enumerate(("B", "W"))
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise DrillError(f"evaluator JSONL output conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, data)


def _dummy_evaluator_main(common: argparse.Namespace, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="__dummy-evaluator")
    parser.add_argument("--job", required=True)
    parser.add_argument("--job-sha256", required=True)
    args = parser.parse_args(list(argv))
    root = common.internal_root.resolve()
    job_path = _internal_path(args.job, root, "evaluator job")
    expected_file_hash = _require_sha256(args.job_sha256, "evaluator job file hash")
    if (
        job_path.is_symlink()
        or not job_path.is_file()
        or sha256_file(job_path) != expected_file_hash
    ):
        raise DrillError("evaluator job file hash changed")
    job = _load_canonical_object(job_path, "evaluator job")
    _exact_keys(
        job,
        {
            "schema_version",
            "contract",
            "root",
            "job_id",
            "candidate_model",
            "candidate_sha256",
            "reference_model",
            "reference_sha256",
            "schedule",
            "schedule_sha256",
            "output",
            "max_rows",
            "job_sha256",
        },
        "evaluator job",
    )
    payload = dict(job)
    supplied = payload.pop("job_sha256")
    candidate = _internal_path(job["candidate_model"], root, "candidate model")
    reference = _internal_path(job["reference_model"], root, "reference model")
    schedule = _internal_path(job["schedule"], root, "evaluation schedule")
    output = _internal_path(job["output"], root, "evaluation output")
    maximum = _require_int(
        job["max_rows"], "evaluator maximum rows", minimum=1, maximum=100000
    )
    if (
        job["schema_version"] != SCHEMA_VERSION
        or job["contract"] != EVALUATOR_JOB_CONTRACT
        or job["root"] != str(root)
        or supplied != canonical_sha256(payload)
        or sha256_file(candidate)
        != _require_sha256(job["candidate_sha256"], "candidate hash")
        or sha256_file(reference)
        != _require_sha256(job["reference_sha256"], "reference hash")
        or sha256_file(schedule)
        != _require_sha256(job["schedule_sha256"], "schedule hash")
    ):
        raise DrillError("evaluator job bindings are invalid")
    rows = _jsonl_rows(schedule, maximum)
    results = [_result_for_schedule(row) for row in rows]
    moves = [
        move
        for row, result in zip(rows, results, strict=True)
        for move in _moves_for_schedule(row, result)
    ]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    moves_path = output / "moves.jsonl"
    _write_jsonl(result_path, results)
    _write_jsonl(moves_path, moves)
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": EVALUATOR_SUMMARY_CONTRACT,
        "finalized": True,
        "decision": "PASS",
        "job_sha256": job["job_sha256"],
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "moves_path": str(moves_path),
        "moves_sha256": sha256_file(moves_path),
        "game_count": len(results),
        "candidate_wins": sum(
            result["winner"] == ("B" if result["blackBot"] == "candidate" else "W")
            for result in results
        ),
        "true_no_results": sum(result["noResult"] is True for result in results),
    }
    summary["decision"] = (
        "PASS"
        if summary["game_count"] > 0
        and summary["candidate_wins"] == summary["game_count"]
        and summary["true_no_results"] == 0
        else "FAIL"
    )
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_once(output / "summary.json", summary)
    return _emit_internal(
        {
            "schema_version": SCHEMA_VERSION,
            "contract": EVALUATOR_SUMMARY_CONTRACT,
            "job_sha256": job["job_sha256"],
            "summary_path": str(output / "summary.json"),
            "summary_sha256": sha256_file(output / "summary.json"),
            "pid": os.getpid(),
            "start_time_ticks": time.monotonic_ns(),
        }
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for gate_id in DRILL_GATES:
        command = subparsers.add_parser(gate_id)
        command.add_argument("--spec", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        "__dummy-worker" in arguments
        or "__dummy-supervisor" in arguments
        or "__dummy-evaluator" in arguments
    ):
        try:
            common, remaining = _internal_common(arguments)
            if not remaining:
                raise DrillError("internal command name is missing")
            command = remaining[0]
            command_argv = remaining[1:]
            if command == "__dummy-worker":
                return _dummy_worker_main(common, command_argv)
            if command == "__dummy-supervisor":
                return _dummy_supervisor_main(common, command_argv)
            if command == "__dummy-evaluator":
                return _dummy_evaluator_main(common, command_argv)
            raise DrillError(f"unknown internal command: {command}")
        except (DrillError, OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    args = parse_args(arguments)
    try:
        evidence = run_drill(args.spec, args.command)
    except (DrillError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
