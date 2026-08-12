#!/usr/bin/env python3
"""Concrete, fail-closed workers for unattended evaluation-suite rotation.

The worker is deliberately narrower than :mod:`suite_rotation_service`.  It
materializes the two existing curation specifications, drives their restartable
APIs, and performs two independent continuity shadow replays.  It never
registers or activates a suite and never imports ``service_activation``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from risk_score.build_evaluation_suites import MACHINE_MANIFEST_CONTRACT
from risk_score.build_live_runtime import verify_deployment_manifest
from risk_score.curation_pipeline import (
    PIPELINE_SPEC_CONTRACT,
    CurationPipeline,
    load_pipeline_spec,
)
from risk_score.curation_supplement import (
    SUPPLEMENT_SPEC_CONTRACT,
    CurationSupplement,
    load_supplement_spec,
)
from risk_score.paired_stats import load_policy
from risk_score.suite_rotation import (
    HOLDOUTS,
    PIPELINE_REQUEST_CONTRACT,
    POLICY_VERSION,
    ROTATION_REQUEST_CONTRACT,
    SUPPLEMENT_REQUEST_CONTRACT,
    SuiteRotationRegistry,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_registry_spec,
    publish_continuity_manifest,
    validate_suite_manifest,
)
from risk_score.suite_rotation_service import CONTINUITY_EVIDENCE_CONTRACT


SCHEMA_VERSION = 1
WORKER_SPEC_CONTRACT = "risk-score-suite-rotation-worker-spec-v1"
SPEC_CONTRACT = WORKER_SPEC_CONTRACT
CURATION_RECEIPT_CONTRACT = "risk-score-suite-rotation-curation-receipt-v1"
SHADOW_REPLAY_EVIDENCE_CONTRACT = (
    "risk-score-suite-rotation-shadow-replay-evidence-v1"
)
COMMAND_RECEIPT_CONTRACT = "risk-score-suite-rotation-command-receipt-v1"
ERROR_CONTRACT = "risk-score-suite-rotation-worker-error-v1"
MAX_JSON_BYTES = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,254})$")
_SOURCE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")

_WORKER_SPEC_KEYS = {
    "schema_version",
    "contract",
    "deployment",
    "deployment_manifest",
    "registry_spec",
    "policy",
    "katago",
    "configs",
    "original_model",
    "gpu_guardian",
    "curation_topology",
    "quarantined_source_root",
    "training_input_exclusion_roots",
    "curation_templates",
    "continuity_templates",
    "spec_sha256",
}
_SUPPLEMENT_TEMPLATE_KEYS = {
    "training_input_root",
    "selfplay_models_directory",
    "selfplay_override_args",
    "game_count",
    "consensus_reserve_fraction",
    "primary_prefilter_inventory",
    "primary_prefilter_manifests",
    "round",
    "prior_round_summaries",
    "downstream_accepted_counts",
}
_PIPELINE_TEMPLATE_KEYS = {"sources"}
_CONTINUITY_TEMPLATE_KEYS = {
    "discovery_argv",
    "confirmation_argv",
}
_TOPOLOGY_KEYS = {"supplement", "pipeline"}
_SUPPLEMENT_TOPOLOGY_KEYS = {
    "shards_per_role",
    "gpus",
    "selfplay_gpus",
    "per_gpu_parallelism",
}
_PIPELINE_TOPOLOGY_KEYS = {
    "shards_per_role",
    "gpus",
    "per_gpu_parallelism",
}
_ROTATION_REQUEST_KEYS = {
    "schema_version",
    "contract",
    "request_id",
    "registry_spec",
    "base_active_suite",
    "models",
    "policy",
    "trigger",
    "requests",
    "request_sha256",
}
_SUPPLEMENT_REQUEST_KEYS = {
    "schema_version",
    "contract",
    "requested_spec_contract",
    "request_id",
    "models",
    "policy",
    "target_counts",
    "quarantined_source_generation",
    "output_root",
    "request_sha256",
}
_PIPELINE_REQUEST_KEYS = {
    "schema_version",
    "contract",
    "requested_spec_contract",
    "request_id",
    "models",
    "policy",
    "source_quotas",
    "holdout_quotas",
    "supplement_request",
    "suite_seed",
    "output_suite_contract",
    "output_root",
    "request_sha256",
}
_SHADOW_EVIDENCE_KEYS = {
    "schema_version",
    "contract",
    "request_id",
    "role",
    "phase",
    "holdout",
    "candidate_suite_id",
    "candidate_suite_manifest_sha256",
    "model_sha256",
    "policy_identity",
    "independent_cluster_ids",
    "independent_cluster_ids_sha256",
    "command_argv_sha256",
    "decision",
    "evidence_sha256",
}
_ROLE_EVIDENCE_KEYS = {
    "schema_version",
    "contract",
    "request_id",
    "base_suite_id",
    "candidate_suite",
    "role",
    "model",
    "policy",
    "decision",
    "completed_at_utc",
    "evidence_sha256",
}
_COMMAND_TEMPLATE_FIELDS = frozenset(
    {
        "claim_id",
        "gpu_id",
        "work_id",
        "guardian_receipt",
        "log_path",
        "state_directory",
        "worker_spec",
        "worker_spec_sha256",
        "repository",
        "source_revision",
        "request_id",
        "role",
        "phase",
        "model_path",
        "model_sha256",
        "candidate_suite_id",
        "candidate_suite_manifest",
        "candidate_suite_manifest_sha256",
        "candidate_suite_manifest_identity",
        "policy_path",
        "policy_sha256",
        "policy_identity",
        "katago",
        "katago_sha256",
        "analysis_config",
        "analysis_config_sha256",
        "powered_config",
        "powered_config_sha256",
        "standard_config",
        "standard_config_sha256",
        "original_model",
        "original_model_sha256",
        "stage_evidence",
        "stage_receipt",
        "work_root",
        "independent_cluster_ids_sha256",
        "seed",
    }
)
_REQUIRED_COMMAND_FIELDS = frozenset(
    {
        "request_id",
        "role",
        "model_path",
        "model_sha256",
        "candidate_suite_id",
        "candidate_suite_manifest",
        "stage_evidence",
    }
)
_DEPLOYED_MODULES = (
    "build_evaluation_suites.py",
    "curation_orchestrator.py",
    "curation_pipeline.py",
    "curation_supplement.py",
    "promotion_evaluator.py",
    "suite_rotation.py",
    "suite_rotation_service.py",
    "suite_rotation_worker.py",
)


class SuiteRotationWorkerError(RuntimeError):
    """Base class for deterministic suite-rotation worker failures."""


class WorkerSpecError(SuiteRotationWorkerError, ValueError):
    """The canonical worker specification is malformed or stale."""


class WorkerStateError(SuiteRotationWorkerError, ValueError):
    """A request or durable artifact contradicts its frozen ancestry."""


class WorkerConflictError(SuiteRotationWorkerError):
    """An immutable output conflicts with a crash replay."""


class WorkerCommandError(SuiteRotationWorkerError):
    """A continuity shadow replay command failed or produced invalid evidence."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class IdentityBinding:
    path: Path
    sha256: str
    identity: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "identity": self.identity,
        }


@dataclass(frozen=True)
class DirectoryBinding:
    path: Path
    sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class DeploymentBinding:
    repository_path: Path
    source_revision: str
    source_sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "repository_path": str(self.repository_path),
            "source_revision": self.source_revision,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class WorkerSpec:
    path: Path
    file_sha256: str
    identity: str
    raw: Mapping[str, Any]
    deployment: DeploymentBinding
    deployment_manifest: FileBinding
    registry: IdentityBinding
    registry_value: Any
    policy: IdentityBinding
    katago: FileBinding
    configs: Mapping[str, FileBinding]
    original_model: FileBinding
    guardian_gpu_id: str
    guardian_argv_prefix: Tuple[str, ...]
    curation_topology: Mapping[str, Mapping[str, Any]]
    quarantined_source_root: Path
    training_input_exclusion_roots: Tuple[Path, ...]
    supplement_template: Mapping[str, Any]
    pipeline_template: Mapping[str, Any]
    continuity_templates: Mapping[str, Tuple[str, ...]]
    frozen_files: Tuple[FileBinding, ...]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _exact_keys(
    value: Any,
    expected: set[str],
    role: str,
    *,
    error_type: type[Exception] = WorkerSpecError,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{role} must be an object")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise error_type(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(
    value: Any,
    role: str,
    *,
    error_type: type[Exception] = WorkerSpecError,
) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise error_type(f"{role} must be a lowercase SHA-256")
    return value


def _require_id(
    value: Any,
    role: str,
    *,
    error_type: type[Exception] = WorkerStateError,
) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise error_type(f"{role} must be a safe nonempty identifier")
    return value


def _reject_symlink_ancestors(
    path: Path,
    role: str,
    *,
    error_type: type[Exception],
) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise error_type(f"{role} has a symlinked path component: {current}")
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(
    value: Any,
    role: str,
    *,
    error_type: type[Exception] = WorkerSpecError,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise error_type(f"{role} must be an absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized or "\x00" in os.fspath(path):
        raise error_type(f"{role} must be lexically normalized and absolute")
    _reject_symlink_ancestors(path, role, error_type=error_type)
    if require_file:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise error_type(f"{role} must be an existing regular file") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise error_type(f"{role} must be an existing regular file")
    if require_directory and (path.is_symlink() or not path.is_dir()):
        raise error_type(f"{role} must be an existing non-symlink directory")
    return path


def _future_file(
    value: Any,
    role: str,
    *,
    error_type: type[Exception] = WorkerStateError,
) -> Path:
    path = _absolute_path(value, role, error_type=error_type)
    if os.path.lexists(os.fspath(path)) and (
        path.is_symlink() or not path.is_file()
    ):
        raise error_type(f"{role} must be a regular non-symlink file when present")
    return path


def _load_canonical_object(
    path: Path,
    role: str,
    *,
    error_type: type[Exception] = WorkerStateError,
) -> Dict[str, Any]:
    source = _absolute_path(
        path,
        role,
        error_type=error_type,
        require_file=True,
    )
    if source.stat().st_size > MAX_JSON_BYTES:
        raise error_type(f"{role} exceeds the size limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, error_type):
            raise
        raise error_type(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"{role} root must be an object")
    if data != canonical_json_bytes(value) + b"\n":
        raise error_type(f"{role} must be canonical newline-terminated JSON")
    return value


def _file_binding(
    value: Any,
    role: str,
    *,
    error_type: type[Exception] = WorkerSpecError,
) -> FileBinding:
    checked = _exact_keys(
        value, {"path", "sha256"}, role, error_type=error_type
    )
    path = _absolute_path(
        checked["path"],
        f"{role} path",
        error_type=error_type,
        require_file=True,
    )
    digest = _require_sha256(
        checked["sha256"], f"{role} hash", error_type=error_type
    )
    if file_sha256(path) != digest:
        raise error_type(f"{role} hash changed")
    return FileBinding(path, digest)


def _identity_binding(
    value: Any,
    role: str,
    *,
    error_type: type[Exception] = WorkerSpecError,
) -> IdentityBinding:
    checked = _exact_keys(
        value, {"path", "sha256", "identity"}, role, error_type=error_type
    )
    binding = _file_binding(
        {"path": checked["path"], "sha256": checked["sha256"]},
        role,
        error_type=error_type,
    )
    identity = _require_sha256(
        checked["identity"], f"{role} identity", error_type=error_type
    )
    return IdentityBinding(binding.path, binding.sha256, identity)


def _directory_inventory(path: Path) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise WorkerSpecError(
                "bound directory may contain only regular non-symlink files"
            )
        rows.append(
            {
                "path": child.name,
                "size": metadata.st_size,
                "sha256": file_sha256(child),
            }
        )
    return tuple(rows)


def _directory_binding(value: Any, role: str) -> DirectoryBinding:
    checked = _exact_keys(value, {"path", "sha256"}, role)
    path = _absolute_path(
        checked["path"],
        f"{role} path",
        require_directory=True,
    )
    digest = _require_sha256(checked["sha256"], f"{role} hash")
    if canonical_sha256(list(_directory_inventory(path))) != digest:
        raise WorkerSpecError(f"{role} inventory hash changed")
    return DirectoryBinding(path, digest)


def _strictly_within(path: Path, root: Path) -> bool:
    if path == root:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or _strictly_within(first, second)
        or _strictly_within(second, first)
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise WorkerStateError(f"unsafe output directory: {target}")
        return
    target.mkdir(parents=True, exist_ok=False)
    _fsync_directory(target.parent)


def _atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = _absolute_path(path, "immutable output", error_type=WorkerStateError)
    _ensure_directory(target.parent)
    data = canonical_json_bytes(dict(value)) + b"\n"
    if os.path.lexists(os.fspath(target)):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise WorkerConflictError(f"immutable artifact conflicts: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(os.fspath(temporary), os.fspath(target))
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != data
            ):
                raise WorkerConflictError(f"immutable artifact conflicts: {target}")
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _deep_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _reserved_supplement_targets(
    source_quotas: Mapping[str, int],
    reserve_fraction: float,
) -> Dict[str, int]:
    return {
        label: int(source_quotas[label])
        + math.ceil(int(source_quotas[label]) * float(reserve_fraction))
        for label in ("lead-40", "lead-80")
    }


def _self_hashed(value: Mapping[str, Any], field: str) -> Dict[str, Any]:
    result = _deep_copy(dict(value))
    result[field] = canonical_sha256(result)
    return result


def _load_self_hashed(
    path: Path,
    *,
    role: str,
    contract: str,
    hash_field: str,
    expected_keys: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    value = _load_canonical_object(path, role)
    if expected_keys is not None:
        _exact_keys(value, expected_keys, role, error_type=WorkerStateError)
    payload = dict(value)
    supplied = _require_sha256(
        payload.pop(hash_field, None),
        f"{role} {hash_field}",
        error_type=WorkerStateError,
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != contract
        or supplied != canonical_sha256(payload)
    ):
        raise WorkerStateError(f"{role} contract or self-hash is invalid")
    return value, supplied


def _template_argv(
    value: Any,
    role: str,
    *,
    required: frozenset[str],
) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(part, str)
            or not part
            or "\x00" in part
            or "\n" in part
            or "\r" in part
            for part in value
        )
    ):
        raise WorkerSpecError(f"{role} must be a nonempty argv string array")
    fields = set()
    formatter = string.Formatter()
    for part in value:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise WorkerSpecError(f"{role} has invalid formatting") from exc
        for _, field, format_spec, conversion in parsed:
            if field is None:
                continue
            if (
                field not in _COMMAND_TEMPLATE_FIELDS
                or format_spec
                or conversion is not None
            ):
                raise WorkerSpecError(
                    f"{role} uses unsupported placeholder {field!r}"
                )
            fields.add(field)
    missing = sorted(required.difference(fields))
    if missing:
        raise WorkerSpecError(f"{role} omits required placeholders: {missing}")
    return tuple(value)


def _validate_gpu_tokens(value: Any, role: str, expected_gpu: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or value != [expected_gpu]:
        raise WorkerSpecError(f"{role} must be pinned exclusively to guarded GPU")
    return tuple(value)


def _validate_positive_integer(value: Any, role: str, maximum: int = 64) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise WorkerSpecError(f"{role} must be an integer between 1 and {maximum}")
    return value


def _walk_absolute_paths(value: Any) -> Tuple[Path, ...]:
    paths = []
    if isinstance(value, Mapping):
        for nested in value.values():
            paths.extend(_walk_absolute_paths(nested))
    elif isinstance(value, list):
        for nested in value:
            paths.extend(_walk_absolute_paths(nested))
    elif isinstance(value, str):
        candidates = [value]
        if "=" in value:
            candidates.append(value.rsplit("=", 1)[1])
        for candidate in candidates:
            if candidate.startswith("/"):
                with contextlib.suppress(ValueError, OSError):
                    paths.append(Path(os.path.abspath(candidate)))
    return tuple(paths)


def _load_provenance_value(path: Path, role: str) -> Any:
    """Load a canonical JSON object or canonical JSONL for path inspection."""

    source = _absolute_path(
        path,
        role,
        error_type=WorkerSpecError,
        require_file=True,
    )
    try:
        data = source.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise WorkerSpecError(f"{role} is not UTF-8 JSON provenance") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError):
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise WorkerSpecError(
                    f"{role} contains a blank JSONL row at line {line_number}"
                )
            try:
                rows.append(
                    json.loads(
                        line,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise WorkerSpecError(
                    f"{role} has invalid JSONL at line {line_number}: {exc}"
                ) from exc
        value = rows
    return value


def _assert_excluded_paths_absent(
    value: Any,
    roots: Sequence[Path],
    role: str,
    *,
    error_type: type[Exception],
) -> None:
    for path in _walk_absolute_paths(value):
        if any(_paths_overlap(path, root) for root in roots):
            raise error_type(f"{role} admits a path beneath a training-input root")


def _validate_deployment_manifest(
    binding: FileBinding,
    deployment: DeploymentBinding,
) -> None:
    try:
        manifest = verify_deployment_manifest(binding.path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkerSpecError(f"deployment manifest is invalid: {exc}") from exc
    if (
        manifest.get("source_revision") != deployment.source_revision
        or manifest.get("source_sha256") != deployment.source_sha256
    ):
        raise WorkerSpecError("deployment manifest does not bind the configured source")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise WorkerSpecError("deployment manifest file inventory is malformed")
    for module_name in _DEPLOYED_MODULES:
        path = deployment.repository_path / "python" / "risk_score" / module_name
        artifact = files.get(f"module:{module_name}")
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("path") != str(path)
            or not path.is_file()
            or path.is_symlink()
            or artifact.get("sha256") != file_sha256(path)
        ):
            raise WorkerSpecError(
                f"deployment manifest does not bind {module_name}"
            )


def _validate_guardian(value: Any) -> Tuple[str, Tuple[str, ...]]:
    checked = _exact_keys(
        value,
        {"gpu_id", "argv_prefix", "argv_prefix_sha256"},
        "GPU guardian",
    )
    gpu_id = _require_id(
        checked["gpu_id"], "guarded GPU ID", error_type=WorkerSpecError
    )
    argv = _template_argv(
        checked["argv_prefix"],
        "GPU guardian argv prefix",
        required=frozenset({"claim_id", "work_id", "guardian_receipt"}),
    )
    if argv[-1] not in {"--", "--command-json"}:
        raise WorkerSpecError("GPU guardian argv prefix must end with a command marker")
    if _require_sha256(
        checked["argv_prefix_sha256"], "GPU guardian argv hash"
    ) != canonical_sha256(list(argv)):
        raise WorkerSpecError("GPU guardian argv prefix hash is invalid")
    hash_flags = {
        "--config-sha256",
        "--expected-config-sha256",
        "--expected-spec-sha256",
        "--spec-sha256",
    }
    hash_bound = any(
        (
            part in hash_flags
            and index + 1 < len(argv)
            and _SHA256_RE.fullmatch(argv[index + 1]) is not None
        )
        or any(
            part.startswith(flag + "=")
            and _SHA256_RE.fullmatch(part[len(flag) + 1 :]) is not None
            for flag in hash_flags
        )
        for index, part in enumerate(argv)
    )
    if not hash_bound:
        raise WorkerSpecError("GPU guardian must bind a literal specification hash")
    return gpu_id, argv


def load_worker_spec(
    path: Path,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> WorkerSpec:
    """Load and fully validate the canonical, self-hashed worker spec."""

    source = _absolute_path(
        Path(path).resolve(),
        "suite-rotation worker specification",
        require_file=True,
    )
    raw = _load_canonical_object(
        source,
        "suite-rotation worker specification",
        error_type=WorkerSpecError,
    )
    _exact_keys(raw, _WORKER_SPEC_KEYS, "worker specification")
    payload = dict(raw)
    identity = _require_sha256(
        payload.pop("spec_sha256", None), "worker specification identity"
    )
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("contract") != WORKER_SPEC_CONTRACT
        or identity != canonical_sha256(payload)
    ):
        raise WorkerSpecError("worker specification contract or self-hash is invalid")
    if expected_spec_sha256 is not None and identity != _require_sha256(
        expected_spec_sha256, "expected worker specification identity"
    ):
        raise WorkerSpecError("worker specification identity is not expected")

    deployment_value = _exact_keys(
        raw["deployment"],
        {"repository_path", "source_revision", "source_sha256"},
        "deployment",
    )
    repository = _absolute_path(
        deployment_value["repository_path"],
        "deployment repository",
        require_directory=True,
    )
    revision = deployment_value["source_revision"]
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise WorkerSpecError("deployment source revision must be a lowercase Git hash")
    source_hash = _require_sha256(
        deployment_value["source_sha256"], "deployment source hash"
    )
    if source_hash != hashlib.sha256(revision.encode("utf-8")).hexdigest():
        raise WorkerSpecError("deployment source hash does not bind its revision")
    deployment = DeploymentBinding(repository, revision, source_hash)
    deployment_manifest = _file_binding(
        raw["deployment_manifest"], "deployment manifest"
    )
    _validate_deployment_manifest(deployment_manifest, deployment)

    registry = _identity_binding(raw["registry_spec"], "registry specification")
    try:
        registry_value = load_registry_spec(registry.path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkerSpecError(f"registry specification is invalid: {exc}") from exc
    if (
        registry_value.file_sha256 != registry.sha256
        or registry_value.identity != registry.identity
    ):
        raise WorkerSpecError("registry specification binding changed")

    policy_value = _exact_keys(
        raw["policy"], {"path", "sha256", "identity", "version"}, "policy"
    )
    policy_file = _file_binding(
        {"path": policy_value["path"], "sha256": policy_value["sha256"]},
        "policy",
    )
    policy_identity = _require_sha256(policy_value["identity"], "policy identity")
    try:
        policy = load_policy(policy_file.path)
    except (OSError, TypeError, ValueError) as exc:
        raise WorkerSpecError(f"policy is invalid: {exc}") from exc
    if (
        policy_value["version"] != POLICY_VERSION
        or policy.get("policy_version") != POLICY_VERSION
        or canonical_sha256(policy) != policy_identity
        or registry_value.policy_path != policy_file.path
        or registry_value.policy_file_sha256 != policy_file.sha256
        or registry_value.policy_identity != policy_identity
    ):
        raise WorkerSpecError("worker policy is not the registry's exact policy v3")
    policy_binding = IdentityBinding(
        policy_file.path, policy_file.sha256, policy_identity
    )

    katago = _file_binding(raw["katago"], "KataGo binary")
    configs_value = _exact_keys(
        raw["configs"],
        {"analysis", "selfplay", "powered", "standard"},
        "KataGo configs",
    )
    configs = {
        name: _file_binding(configs_value[name], f"{name} config")
        for name in ("analysis", "selfplay", "powered", "standard")
    }
    original_model = _file_binding(raw["original_model"], "original model")
    if (
        registry_value.original.path != original_model.path
        or registry_value.original.sha256 != original_model.sha256
    ):
        raise WorkerSpecError("worker original model differs from registry original")

    guardian_gpu_id, guardian_argv = _validate_guardian(raw["gpu_guardian"])
    topology_value = _exact_keys(
        raw["curation_topology"], _TOPOLOGY_KEYS, "curation topology"
    )
    supplement_topology = _exact_keys(
        topology_value["supplement"],
        _SUPPLEMENT_TOPOLOGY_KEYS,
        "supplement topology",
    )
    pipeline_topology = _exact_keys(
        topology_value["pipeline"],
        _PIPELINE_TOPOLOGY_KEYS,
        "pipeline topology",
    )
    for value, role in (
        (supplement_topology, "supplement"),
        (pipeline_topology, "pipeline"),
    ):
        _validate_positive_integer(value["shards_per_role"], f"{role} shards")
        _validate_positive_integer(
            value["per_gpu_parallelism"], f"{role} per-GPU parallelism"
        )
        _validate_gpu_tokens(value["gpus"], f"{role} GPUs", guardian_gpu_id)
    _validate_gpu_tokens(
        supplement_topology["selfplay_gpus"],
        "supplement self-play GPUs",
        guardian_gpu_id,
    )
    topologies = {
        "supplement": MappingProxyType(_deep_copy(supplement_topology)),
        "pipeline": MappingProxyType(_deep_copy(pipeline_topology)),
    }

    quarantine = _absolute_path(
        raw["quarantined_source_root"],
        "quarantined source root",
        require_directory=True,
    )
    roots_value = raw["training_input_exclusion_roots"]
    if not isinstance(roots_value, list) or not roots_value:
        raise WorkerSpecError(
            "training_input_exclusion_roots must be a nonempty path array"
        )
    exclusion_roots = tuple(
        _absolute_path(
            value,
            f"training-input exclusion root {index}",
            require_directory=True,
        )
        for index, value in enumerate(roots_value)
    )
    if (
        len(set(exclusion_roots)) != len(exclusion_roots)
        or list(exclusion_roots) != sorted(exclusion_roots, key=str)
    ):
        raise WorkerSpecError(
            "training-input exclusion roots must be unique and path-sorted"
        )
    if any(_paths_overlap(quarantine, root) for root in exclusion_roots):
        raise WorkerSpecError(
            "quarantined source root overlaps a training-input exclusion root"
        )

    templates = _exact_keys(
        raw["curation_templates"],
        {"supplement", "pipeline"},
        "curation templates",
    )
    supplement_template = _exact_keys(
        templates["supplement"],
        _SUPPLEMENT_TEMPLATE_KEYS,
        "supplement template",
    )
    pipeline_template = _exact_keys(
        templates["pipeline"],
        _PIPELINE_TEMPLATE_KEYS,
        "pipeline template",
    )
    training_root = _absolute_path(
        supplement_template["training_input_root"],
        "supplement training-input root",
        require_directory=True,
    )
    if training_root not in exclusion_roots:
        raise WorkerSpecError(
            "supplement training_input_root is not an exclusion root"
        )
    models_directory = _directory_binding(
        supplement_template["selfplay_models_directory"],
        "self-play models directory",
    )
    if (
        original_model.path != models_directory.path / "model.bin.gz"
        or len(_directory_inventory(models_directory.path)) != 1
    ):
        raise WorkerSpecError(
            "self-play models directory must contain only original model.bin.gz"
        )
    override_args = supplement_template["selfplay_override_args"]
    if not isinstance(override_args, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(part, str) or not part for part in pair)
        for pair in override_args
    ):
        raise WorkerSpecError("selfplay_override_args must be an argv-pair array")
    game_count = supplement_template["game_count"]
    if type(game_count) is not int or not 1 <= game_count <= 1_000_000_000:
        raise WorkerSpecError("supplement game_count is invalid")
    reserve = supplement_template["consensus_reserve_fraction"]
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not math.isfinite(float(reserve))
        or not 0 < float(reserve) <= 10
    ):
        raise WorkerSpecError("supplement consensus reserve is invalid")
    round_number = supplement_template["round"]
    if type(round_number) is not int or round_number < 1:
        raise WorkerSpecError("supplement round must be positive")

    frozen_files = [
        deployment_manifest,
        FileBinding(registry.path, registry.sha256),
        FileBinding(policy_binding.path, policy_binding.sha256),
        katago,
        *configs.values(),
        original_model,
    ]
    primary_inventory = _file_binding(
        supplement_template["primary_prefilter_inventory"],
        "primary prefilter inventory",
    )
    primary_values = supplement_template["primary_prefilter_manifests"]
    if not isinstance(primary_values, list):
        raise WorkerSpecError("primary_prefilter_manifests must be an array")
    primary = [
        _file_binding(item, f"primary prefilter manifest {index}")
        for index, item in enumerate(primary_values)
    ]
    if [str(item.path) for item in primary] != sorted(str(item.path) for item in primary):
        raise WorkerSpecError("primary prefilter manifests must be path-sorted")
    prior_values = supplement_template["prior_round_summaries"]
    if not isinstance(prior_values, list):
        raise WorkerSpecError("prior_round_summaries must be an array")
    prior = [
        _file_binding(item, f"prior round summary {index}")
        for index, item in enumerate(prior_values)
    ]
    if [str(item.path) for item in prior] != sorted(str(item.path) for item in prior):
        raise WorkerSpecError("prior round summaries must be path-sorted")
    sources_value = pipeline_template["sources"]
    if not isinstance(sources_value, list) or not sources_value:
        raise WorkerSpecError("pipeline source template must be nonempty")
    source_names = []
    source_bindings = []
    for index, item in enumerate(sources_value):
        if not isinstance(item, Mapping):
            raise WorkerSpecError(f"pipeline source {index} must be an object")
        required = {"name", "label", "selected", "prefilter_manifest"}
        allowed = required | {"supplement_summary"}
        if not required.issubset(item) or not set(item).issubset(allowed):
            raise WorkerSpecError(f"pipeline source {index} keys differ from contract")
        name = item["name"]
        if not isinstance(name, str) or _SOURCE_NAME_RE.fullmatch(name) is None:
            raise WorkerSpecError(f"pipeline source {index} name is unsafe")
        source_names.append(name)
        for field in ("selected", "prefilter_manifest"):
            source_bindings.append(
                _file_binding(item[field], f"pipeline source {name} {field}")
            )
        if "supplement_summary" in item:
            source_bindings.append(
                _file_binding(
                    item["supplement_summary"],
                    f"pipeline source {name} supplement summary",
                )
            )
    if source_names != sorted(source_names) or len(set(source_names)) != len(source_names):
        raise WorkerSpecError("pipeline sources must be uniquely name-sorted")
    quarantined_bindings = [primary_inventory, *primary, *prior, *source_bindings]
    if any(
        not _strictly_within(binding.path, quarantine)
        for binding in quarantined_bindings
    ):
        raise WorkerSpecError(
            "curation source artifacts must be strictly beneath quarantine root"
        )
    for binding in quarantined_bindings:
        _assert_excluded_paths_absent(
            _load_provenance_value(
                binding.path, f"quarantined source artifact {binding.path}"
            ),
            exclusion_roots,
            "quarantined source provenance",
            error_type=WorkerSpecError,
        )
    frozen_files.extend(quarantined_bindings)

    continuity_value = _exact_keys(
        raw["continuity_templates"],
        _CONTINUITY_TEMPLATE_KEYS,
        "continuity templates",
    )
    continuity_templates = {
        "discovery": _template_argv(
            continuity_value["discovery_argv"],
            "discovery continuity argv",
            required=_REQUIRED_COMMAND_FIELDS,
        ),
        "confirmation": _template_argv(
            continuity_value["confirmation_argv"],
            "confirmation continuity argv",
            required=_REQUIRED_COMMAND_FIELDS,
        ),
    }
    unique_frozen = {
        (binding.path, binding.sha256): binding for binding in frozen_files
    }
    return WorkerSpec(
        path=source,
        file_sha256=file_sha256(source),
        identity=identity,
        raw=MappingProxyType(raw),
        deployment=deployment,
        deployment_manifest=deployment_manifest,
        registry=registry,
        registry_value=registry_value,
        policy=policy_binding,
        katago=katago,
        configs=MappingProxyType(configs),
        original_model=original_model,
        guardian_gpu_id=guardian_gpu_id,
        guardian_argv_prefix=guardian_argv,
        curation_topology=MappingProxyType(topologies),
        quarantined_source_root=quarantine,
        training_input_exclusion_roots=exclusion_roots,
        supplement_template=MappingProxyType(_deep_copy(supplement_template)),
        pipeline_template=MappingProxyType(_deep_copy(pipeline_template)),
        continuity_templates=MappingProxyType(continuity_templates),
        frozen_files=tuple(
            unique_frozen[key]
            for key in sorted(unique_frozen, key=lambda item: str(item[0]))
        ),
    )


def publish_worker_spec(path: Path, value: Mapping[str, Any]) -> WorkerSpec:
    """Publish a caller-assembled worker spec with a canonical self-hash."""

    body = _deep_copy(dict(value))
    body.pop("spec_sha256", None)
    body.setdefault("schema_version", SCHEMA_VERSION)
    body.setdefault("contract", WORKER_SPEC_CONTRACT)
    body["spec_sha256"] = canonical_sha256(body)
    destination = Path(path).resolve()
    _atomic_immutable_json(destination, body)
    return load_worker_spec(destination)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise WorkerStateError("clock returned a naive datetime")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc(value: Any, role: str) -> datetime:
    if not isinstance(value, str):
        raise WorkerStateError(f"{role} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkerStateError(f"{role} is invalid") from exc
    if parsed.tzinfo is None or _format_utc(parsed) != value:
        raise WorkerStateError(f"{role} must be canonical UTC")
    return parsed


def _binding_matches(
    actual: Mapping[str, Any],
    expected: Union[FileBinding, IdentityBinding],
    *,
    identity: bool,
) -> bool:
    wanted = expected.to_dict()
    return dict(actual) == wanted if identity else dict(actual) == {
        "path": wanted["path"],
        "sha256": wanted["sha256"],
    }


def _request_file(
    path: Path,
    *,
    role: str,
    contract: str,
    keys: set[str],
    expected_file_sha256: Optional[str],
    expected_identity: Optional[str],
) -> Tuple[Dict[str, Any], IdentityBinding]:
    source = _absolute_path(
        path,
        role,
        error_type=WorkerStateError,
        require_file=True,
    )
    value, identity = _load_self_hashed(
        source,
        role=role,
        contract=contract,
        hash_field="request_sha256",
        expected_keys=keys,
    )
    digest = file_sha256(source)
    if expected_file_sha256 is not None and digest != _require_sha256(
        expected_file_sha256,
        f"{role} expected file hash",
        error_type=WorkerStateError,
    ):
        raise WorkerStateError(f"{role} file hash is not expected")
    if expected_identity is not None and identity != _require_sha256(
        expected_identity,
        f"{role} expected identity",
        error_type=WorkerStateError,
    ):
        raise WorkerStateError(f"{role} identity is not expected")
    return value, IdentityBinding(source, digest, identity)


def _request_model(value: Any, role: str) -> Tuple[FileBinding, str]:
    checked = _exact_keys(
        value,
        {"role", "path", "sha256"},
        f"{role} request model",
        error_type=WorkerStateError,
    )
    expected_role = "immutable_original" if role == "original" else "frozen_champion"
    if checked["role"] != expected_role:
        raise WorkerStateError(f"{role} request model role is invalid")
    binding = _file_binding(
        {"path": checked["path"], "sha256": checked["sha256"]},
        f"{role} request model",
        error_type=WorkerStateError,
    )
    return binding, expected_role


def _common_run_root(first: Path, second: Path, request_id: str) -> Path:
    try:
        common = Path(os.path.commonpath((first, second)))
    except ValueError as exc:
        raise WorkerStateError("curation output roots have no common root") from exc
    if (
        common == Path(common.anchor)
        or not _strictly_within(first, common)
        or not _strictly_within(second, common)
        or request_id not in common.parts
        or request_id not in first.parts
        or request_id not in second.parts
    ):
        raise WorkerStateError(
            "curation output roots must be isolated beneath the request ID"
        )
    return common


def _default_supplement_executor(
    path: Path, *, poll_interval: float
) -> Mapping[str, Any]:
    return CurationSupplement(path).watch(poll_interval=poll_interval)


def _default_pipeline_executor(
    path: Path, *, poll_interval: float
) -> Mapping[str, Any]:
    return CurationPipeline(path).watch(poll_interval=poll_interval)


class SuiteRotationWorker:
    """Materialize, curate, and continuity-test one frozen rotation request."""

    def __init__(
        self,
        spec: Union[WorkerSpec, Path, str],
        *,
        expected_spec_sha256: Optional[str] = None,
        command_runner: Callable[..., Any] = subprocess.run,
        supplement_loader: Callable[[Path], Any] = load_supplement_spec,
        pipeline_loader: Callable[[Path], Any] = load_pipeline_spec,
        supplement_executor: Callable[..., Mapping[str, Any]] = (
            _default_supplement_executor
        ),
        pipeline_executor: Callable[..., Mapping[str, Any]] = _default_pipeline_executor,
        suite_validator: Callable[..., Any] = validate_suite_manifest,
        registry: Optional[Any] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.spec = (
            spec
            if isinstance(spec, WorkerSpec)
            else load_worker_spec(
                Path(spec), expected_spec_sha256=expected_spec_sha256
            )
        )
        if (
            isinstance(spec, WorkerSpec)
            and expected_spec_sha256 is not None
            and spec.identity != expected_spec_sha256
        ):
            raise WorkerSpecError("worker specification identity is not expected")
        self.command_runner = command_runner
        self.supplement_loader = supplement_loader
        self.pipeline_loader = pipeline_loader
        self.supplement_executor = supplement_executor
        self.pipeline_executor = pipeline_executor
        self.suite_validator = suite_validator
        self.registry = registry or SuiteRotationRegistry(self.spec.registry.path)
        self.clock = clock
        self._assert_frozen()

    def _now(self) -> datetime:
        value = self.clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise WorkerStateError("clock returned a naive datetime")
            return value.astimezone(timezone.utc)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise WorkerStateError("clock must return datetime or Unix seconds")
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    def _assert_frozen(self) -> None:
        if (
            self.spec.path.is_symlink()
            or not self.spec.path.is_file()
            or file_sha256(self.spec.path) != self.spec.file_sha256
        ):
            raise WorkerStateError("frozen worker specification changed")
        for binding in self.spec.frozen_files:
            if (
                binding.path.is_symlink()
                or not binding.path.is_file()
                or file_sha256(binding.path) != binding.sha256
            ):
                raise WorkerStateError(f"frozen worker input changed: {binding.path}")
        _validate_deployment_manifest(
            self.spec.deployment_manifest, self.spec.deployment
        )

    def _validate_request_bundle(
        self,
        *,
        request_id: str,
        rotation_request: Path,
        supplement_request: Path,
        pipeline_request: Path,
        rotation_request_sha256: Optional[str],
        rotation_request_identity: Optional[str],
        supplement_request_sha256: Optional[str],
        supplement_request_identity: Optional[str],
        pipeline_request_sha256: Optional[str],
        pipeline_request_identity: Optional[str],
    ) -> Mapping[str, Any]:
        identifier = _require_id(request_id, "rotation request ID")
        rotation, rotation_binding = _request_file(
            rotation_request,
            role="rotation request",
            contract=ROTATION_REQUEST_CONTRACT,
            keys=_ROTATION_REQUEST_KEYS,
            expected_file_sha256=rotation_request_sha256,
            expected_identity=rotation_request_identity,
        )
        supplement, supplement_binding = _request_file(
            supplement_request,
            role="supplement child request",
            contract=SUPPLEMENT_REQUEST_CONTRACT,
            keys=_SUPPLEMENT_REQUEST_KEYS,
            expected_file_sha256=supplement_request_sha256,
            expected_identity=supplement_request_identity,
        )
        pipeline, pipeline_binding = _request_file(
            pipeline_request,
            role="pipeline child request",
            contract=PIPELINE_REQUEST_CONTRACT,
            keys=_PIPELINE_REQUEST_KEYS,
            expected_file_sha256=pipeline_request_sha256,
            expected_identity=pipeline_request_identity,
        )
        registry_request_root = (
            Path(self.spec.registry_value.root) / "requests" / identifier
        )
        expected_request_paths = {
            "rotation": registry_request_root / "manifest.json",
            "supplement": registry_request_root / "curation-supplement.json",
            "pipeline": registry_request_root / "curation-pipeline.json",
        }
        if (
            rotation_binding.path != expected_request_paths["rotation"]
            or supplement_binding.path != expected_request_paths["supplement"]
            or pipeline_binding.path != expected_request_paths["pipeline"]
        ):
            raise WorkerStateError(
                "rotation request bundle is not registry-owned"
            )
        if any(
            request.get("request_id") != identifier
            for request in (rotation, supplement, pipeline)
        ):
            raise WorkerStateError("rotation request IDs disagree")
        if not _binding_matches(
            rotation["registry_spec"], self.spec.registry, identity=True
        ):
            raise WorkerStateError("rotation request names another registry")
        child_bindings = _exact_keys(
            rotation["requests"],
            {"curation_supplement", "curation_pipeline"},
            "rotation child request inventory",
            error_type=WorkerStateError,
        )
        if (
            dict(child_bindings["curation_supplement"])
            != supplement_binding.to_dict()
            or dict(child_bindings["curation_pipeline"]) != pipeline_binding.to_dict()
        ):
            raise WorkerStateError("rotation child request bindings changed")
        if dict(pipeline["supplement_request"]) != supplement_binding.to_dict():
            raise WorkerStateError("pipeline request lost supplement ancestry")

        models = _exact_keys(
            rotation["models"],
            {"original", "champion"},
            "rotation request models",
            error_type=WorkerStateError,
        )
        original, _ = _request_model(models["original"], "original")
        champion, _ = _request_model(models["champion"], "champion")
        if original != self.spec.original_model:
            raise WorkerStateError("rotation request changed the immutable original")
        if (
            champion == original
            or supplement["models"] != rotation["models"]
            or pipeline["models"] != rotation["models"]
        ):
            raise WorkerStateError("rotation child model ancestry is invalid")
        expected_policy = {
            "path": str(self.spec.policy.path),
            "sha256": self.spec.policy.sha256,
            "identity": self.spec.policy.identity,
            "version": POLICY_VERSION,
        }
        if any(
            request.get("policy") != expected_policy
            for request in (rotation, supplement, pipeline)
        ):
            raise WorkerStateError("rotation requests changed policy v3")
        if (
            supplement.get("requested_spec_contract") != SUPPLEMENT_SPEC_CONTRACT
            or supplement.get("quarantined_source_generation") is not True
            or pipeline.get("requested_spec_contract") != PIPELINE_SPEC_CONTRACT
            or pipeline.get("output_suite_contract") != MACHINE_MANIFEST_CONTRACT
        ):
            raise WorkerStateError("rotation child request contract is invalid")
        expected_seed = "suite-rotation-" + identifier.removeprefix("rotation-")
        if pipeline.get("suite_seed") != expected_seed:
            raise WorkerStateError("pipeline request seed is not request-derived")
        if pipeline.get("source_quotas") != dict(
            self.spec.registry_value.source_quotas
        ):
            raise WorkerStateError("pipeline source quotas changed")
        expected_holdouts = {
            label: dict(self.spec.registry_value.holdout_quotas[label])
            for label in self.spec.registry_value.holdout_quotas
        }
        if pipeline.get("holdout_quotas") != expected_holdouts:
            raise WorkerStateError("pipeline holdout quotas changed")
        if supplement.get("target_counts") != {
            label: self.spec.registry_value.source_quotas[label]
            for label in ("lead-40", "lead-80")
        }:
            raise WorkerStateError("supplement target counts changed")
        base = _exact_keys(
            rotation["base_active_suite"],
            {"suite_id", "version_sha256"},
            "base active suite",
            error_type=WorkerStateError,
        )
        _require_sha256(
            base["suite_id"], "base suite ID", error_type=WorkerStateError
        )
        _require_sha256(
            base["version_sha256"],
            "base suite version",
            error_type=WorkerStateError,
        )
        return {
            "request_id": identifier,
            "rotation": rotation,
            "rotation_binding": rotation_binding,
            "supplement": supplement,
            "supplement_binding": supplement_binding,
            "pipeline": pipeline,
            "pipeline_binding": pipeline_binding,
            "original": original,
            "champion": champion,
        }

    def _assert_output_safety(
        self,
        paths: Sequence[Path],
        *,
        protected: Sequence[Path] = (),
    ) -> None:
        for path in paths:
            if any(
                _paths_overlap(path, root)
                for root in self.spec.training_input_exclusion_roots
            ):
                raise WorkerStateError(
                    "rotation output overlaps a training-input exclusion root"
                )
            if any(_paths_overlap(path, item) for item in protected):
                raise WorkerStateError("rotation output overlaps a frozen input")
        for index, path in enumerate(paths):
            if any(_paths_overlap(path, other) for other in paths[index + 1 :]):
                raise WorkerStateError("rotation output paths overlap")

    def _load_materialized_pair(
        self,
        supplement_path: Path,
        pipeline_path: Path,
    ) -> Tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        supplement_raw, supplement_identity = _load_self_hashed(
            supplement_path,
            role="materialized supplement specification",
            contract=SUPPLEMENT_SPEC_CONTRACT,
            hash_field="spec_sha256",
        )
        pipeline_raw, pipeline_identity = _load_self_hashed(
            pipeline_path,
            role="materialized pipeline specification",
            contract=PIPELINE_SPEC_CONTRACT,
            hash_field="spec_sha256",
        )
        try:
            supplement = self.supplement_loader(supplement_path)
            pipeline = self.pipeline_loader(pipeline_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WorkerStateError(
                f"materialized curation specification is invalid: {exc}"
            ) from exc
        for loaded, raw, identity, path, role in (
            (
                supplement,
                supplement_raw,
                supplement_identity,
                supplement_path,
                "supplement",
            ),
            (
                pipeline,
                pipeline_raw,
                pipeline_identity,
                pipeline_path,
                "pipeline",
            ),
        ):
            if (
                dict(getattr(loaded, "raw", raw)) != raw
                or getattr(loaded, "identity", identity) != identity
                or getattr(loaded, "file_sha256", file_sha256(path))
                != file_sha256(path)
            ):
                raise WorkerStateError(
                    f"{role} loader returned contradictory frozen coordinates"
                )
        for field in (
            "deployment",
            "deployment_manifest",
            "run_root",
            "katago",
            "analysis_config",
            "policy",
            "models",
        ):
            if supplement_raw.get(field) != pipeline_raw.get(field):
                raise WorkerStateError(
                    f"supplement and pipeline {field} bindings differ"
                )
        expected_common = {
            "deployment": self.spec.deployment.to_dict(),
            "deployment_manifest": self.spec.deployment_manifest.to_dict(),
            "katago": self.spec.katago.to_dict(),
            "analysis_config": self.spec.configs["analysis"].to_dict(),
            "policy": {
                "path": str(self.spec.policy.path),
                "sha256": self.spec.policy.sha256,
            },
        }
        for field, expected in expected_common.items():
            if supplement_raw.get(field) != expected:
                raise WorkerStateError(f"materialized {field} binding changed")
        expected_original = self.spec.original_model.to_dict()
        models = supplement_raw.get("models")
        if (
            not isinstance(models, Mapping)
            or set(models) != {"original", "champion"}
            or models.get("original") != expected_original
        ):
            raise WorkerStateError("materialized model bindings are invalid")
        if supplement_raw.get("topology") != dict(
            self.spec.curation_topology["supplement"]
        ) or pipeline_raw.get("topology") != dict(
            self.spec.curation_topology["pipeline"]
        ):
            raise WorkerStateError("materialized curation topology changed")
        supplement_template = self.spec.supplement_template
        reserved_targets = _reserved_supplement_targets(
            self.spec.registry_value.source_quotas,
            supplement_template["consensus_reserve_fraction"],
        )
        expected_supplement = {
            "selfplay_config": self.spec.configs["selfplay"].to_dict(),
            "selfplay_models_directory": _deep_copy(
                supplement_template["selfplay_models_directory"]
            ),
            "selfplay_override_args": _deep_copy(
                supplement_template["selfplay_override_args"]
            ),
            "game_count": supplement_template["game_count"],
            "consensus_reserve_fraction": supplement_template[
                "consensus_reserve_fraction"
            ],
            "primary_prefilter_inventory": _deep_copy(
                supplement_template["primary_prefilter_inventory"]
            ),
            "primary_prefilter_manifests": _deep_copy(
                supplement_template["primary_prefilter_manifests"]
            ),
            "round": supplement_template["round"],
            "prior_round_summaries": _deep_copy(
                supplement_template["prior_round_summaries"]
            ),
            "downstream_accepted_counts": _deep_copy(
                supplement_template["downstream_accepted_counts"]
            ),
            "target_counts": reserved_targets,
        }
        if any(
            supplement_raw.get(field) != expected
            for field, expected in expected_supplement.items()
        ):
            raise WorkerStateError("materialized supplement template binding changed")
        if (
            pipeline_raw.get("sources")
            != _deep_copy(self.spec.pipeline_template["sources"])
            or pipeline_raw.get("quotas")
            != dict(self.spec.registry_value.source_quotas)
        ):
            raise WorkerStateError("materialized pipeline template or quotas changed")
        if supplement_raw.get("training_input_root") not in {
            str(root) for root in self.spec.training_input_exclusion_roots
        }:
            raise WorkerStateError(
                "materialized supplement lost training-input exclusion"
            )
        # The one deliberate exclusion-root reference is the read-forbidden
        # training_input_root declaration itself.
        supplement_without_training = dict(supplement_raw)
        supplement_without_training["training_input_root"] = ""
        _assert_excluded_paths_absent(
            supplement_without_training,
            self.spec.training_input_exclusion_roots,
            "supplement specification",
            error_type=WorkerStateError,
        )
        _assert_excluded_paths_absent(
            pipeline_raw,
            self.spec.training_input_exclusion_roots,
            "pipeline specification",
            error_type=WorkerStateError,
        )
        return supplement, pipeline, supplement_raw, pipeline_raw

    def materialize(
        self,
        *,
        request_id: str,
        rotation_request: Path,
        supplement_request: Path,
        pipeline_request: Path,
        supplement_spec: Path,
        pipeline_spec: Path,
        rotation_request_sha256: Optional[str] = None,
        rotation_request_identity: Optional[str] = None,
        supplement_request_sha256: Optional[str] = None,
        supplement_request_identity: Optional[str] = None,
        pipeline_request_sha256: Optional[str] = None,
        pipeline_request_identity: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Publish the exact two curation specs requested by the registry."""

        self._assert_frozen()
        context = self._validate_request_bundle(
            request_id=request_id,
            rotation_request=rotation_request,
            supplement_request=supplement_request,
            pipeline_request=pipeline_request,
            rotation_request_sha256=rotation_request_sha256,
            rotation_request_identity=rotation_request_identity,
            supplement_request_sha256=supplement_request_sha256,
            supplement_request_identity=supplement_request_identity,
            pipeline_request_sha256=pipeline_request_sha256,
            pipeline_request_identity=pipeline_request_identity,
        )
        supplement_output = _absolute_path(
            context["supplement"]["output_root"],
            "supplement output root",
            error_type=WorkerStateError,
        )
        pipeline_output = _absolute_path(
            context["pipeline"]["output_root"],
            "pipeline output root",
            error_type=WorkerStateError,
        )
        if _paths_overlap(supplement_output, pipeline_output):
            raise WorkerStateError("supplement and pipeline output roots overlap")
        supplement_path = _future_file(supplement_spec, "supplement spec output")
        pipeline_path = _future_file(pipeline_spec, "pipeline spec output")
        if (
            not _strictly_within(supplement_path, supplement_output)
            or not _strictly_within(pipeline_path, pipeline_output)
        ):
            raise WorkerStateError(
                "materialized specification escaped its child output root"
            )
        run_root = _common_run_root(
            supplement_output, pipeline_output, context["request_id"]
        )
        protected = (
            self.spec.path,
            self.spec.registry.path,
            context["rotation_binding"].path,
            context["supplement_binding"].path,
            context["pipeline_binding"].path,
            *(binding.path for binding in self.spec.frozen_files),
        )
        supplement_work = supplement_output / "work"
        pipeline_work = pipeline_output / "work"
        reviewed_root = pipeline_output / "outputs"
        suite_directory = pipeline_output / "suite"
        self._assert_output_safety(
            (
                supplement_work,
                pipeline_work,
                reviewed_root,
                suite_directory,
            ),
            protected=protected,
        )
        if any(
            _paths_overlap(path, self.spec.quarantined_source_root)
            for path in (
                run_root,
                supplement_work,
                pipeline_work,
                reviewed_root,
                suite_directory,
            )
        ):
            raise WorkerStateError("curation output overlaps quarantined inputs")
        _ensure_directory(run_root)

        common = {
            "deployment": self.spec.deployment.to_dict(),
            "deployment_manifest": self.spec.deployment_manifest.to_dict(),
            "run_root": str(run_root),
            "policy": {
                "path": str(self.spec.policy.path),
                "sha256": self.spec.policy.sha256,
            },
            "katago": self.spec.katago.to_dict(),
            "analysis_config": self.spec.configs["analysis"].to_dict(),
            "models": {
                "original": context["original"].to_dict(),
                "champion": context["champion"].to_dict(),
            },
        }
        supplement_template = self.spec.supplement_template
        reserved_targets = _reserved_supplement_targets(
            self.spec.registry_value.source_quotas,
            supplement_template["consensus_reserve_fraction"],
        )
        supplement_value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": SUPPLEMENT_SPEC_CONTRACT,
            **common,
            "training_input_root": supplement_template["training_input_root"],
            "work_root": str(supplement_work),
            "selfplay_config": self.spec.configs["selfplay"].to_dict(),
            "selfplay_models_directory": _deep_copy(
                supplement_template["selfplay_models_directory"]
            ),
            "selfplay_override_args": _deep_copy(
                supplement_template["selfplay_override_args"]
            ),
            "game_count": supplement_template["game_count"],
            "topology": dict(self.spec.curation_topology["supplement"]),
            "consensus_reserve_fraction": supplement_template[
                "consensus_reserve_fraction"
            ],
            "target_counts": reserved_targets,
            "primary_prefilter_inventory": _deep_copy(
                supplement_template["primary_prefilter_inventory"]
            ),
            "primary_prefilter_manifests": _deep_copy(
                supplement_template["primary_prefilter_manifests"]
            ),
            "round": supplement_template["round"],
            "prior_round_summaries": _deep_copy(
                supplement_template["prior_round_summaries"]
            ),
            "downstream_accepted_counts": _deep_copy(
                supplement_template["downstream_accepted_counts"]
            ),
        }
        supplement_value["spec_sha256"] = canonical_sha256(supplement_value)
        pipeline_value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": PIPELINE_SPEC_CONTRACT,
            **common,
            "sources": _deep_copy(self.spec.pipeline_template["sources"]),
            "work_root": str(pipeline_work),
            "outputs": {
                "reviewed_bank": str(reviewed_root / "source-positions.jsonl"),
                "reviewed_manifest": str(
                    reviewed_root / "source-positions.manifest.json"
                ),
                "suite_directory": str(suite_directory),
            },
            "quotas": _deep_copy(context["pipeline"]["source_quotas"]),
            "topology": dict(self.spec.curation_topology["pipeline"]),
            "suite_seed": context["pipeline"]["suite_seed"],
        }
        pipeline_value["spec_sha256"] = canonical_sha256(pipeline_value)
        _atomic_immutable_json(supplement_path, supplement_value)
        _atomic_immutable_json(pipeline_path, pipeline_value)
        _, _, observed_supplement, observed_pipeline = self._load_materialized_pair(
            supplement_path, pipeline_path
        )
        if observed_supplement != supplement_value or observed_pipeline != pipeline_value:
            raise WorkerStateError("materialized curation specs changed after publish")
        return MappingProxyType(
            {
                "request_id": context["request_id"],
                "supplement_spec": {
                    "path": str(supplement_path),
                    "sha256": file_sha256(supplement_path),
                    "identity": supplement_value["spec_sha256"],
                },
                "pipeline_spec": {
                    "path": str(pipeline_path),
                    "sha256": file_sha256(pipeline_path),
                    "identity": pipeline_value["spec_sha256"],
                },
                "source_quotas": _deep_copy(pipeline_value["quotas"]),
                "suite_seed": pipeline_value["suite_seed"],
                "training_inputs_admitted": False,
                "activation_performed": False,
            }
        )

    @staticmethod
    def _manual_suite_provenance(
        manifest_path: Path,
        *,
        expected_policy_identity: str,
        expected_original_sha256: str,
        expected_champion_sha256: str,
    ) -> Mapping[str, Tuple[str, ...]]:
        manifest = _load_canonical_object(manifest_path, "candidate suite manifest")
        payload = dict(manifest)
        supplied = payload.pop("manifestPayloadSha256", None)
        if (
            manifest.get("schemaVersion") != 3
            or manifest.get("manifestContract") != MACHINE_MANIFEST_CONTRACT
            or manifest.get("machineReviewOnly") is not True
            or supplied != canonical_sha256(payload)
            or manifest.get("policy_hash") != expected_policy_identity
            or manifest.get("policy_version") != POLICY_VERSION
        ):
            raise WorkerStateError(
                "candidate suite is not a canonical machine-only policy-v3 suite"
            )
        sources = manifest.get("curationSources")
        if not isinstance(sources, list) or not sources:
            raise WorkerStateError(
                "candidate suite lacks machine-consensus provenance"
            )
        expected_models = {
            "original": {
                "role": "immutable_original",
                "sha256": expected_original_sha256,
            },
            "champion": {
                "role": "frozen_champion",
                "sha256": expected_champion_sha256,
            },
        }
        for source in sources:
            if (
                not isinstance(source, Mapping)
                or source.get("review_mode") != "machine-consensus"
                or source.get("consensus_rules_version") != 1
                or source.get("policy_hash") != expected_policy_identity
                or source.get("models") != expected_models
            ):
                raise WorkerStateError(
                    "candidate suite machine-consensus provenance is invalid"
                )
        by_holdout: Dict[str, list[str]] = {holdout: [] for holdout in HOLDOUTS}
        banks = manifest.get("banks")
        if not isinstance(banks, list) or not banks:
            raise WorkerStateError("candidate suite has no holdout bank inventory")
        for bank in banks:
            if not isinstance(bank, Mapping) or bank.get("holdout") not in by_holdout:
                raise WorkerStateError("candidate suite holdout bank is malformed")
            identifiers = bank.get("independentClusterIds")
            if (
                not isinstance(identifiers, list)
                or not identifiers
                or any(
                    not isinstance(item, str) or _SHA256_RE.fullmatch(item) is None
                    for item in identifiers
                )
            ):
                raise WorkerStateError(
                    "candidate suite independent-cluster inventory is invalid"
                )
            by_holdout[str(bank["holdout"])].extend(identifiers)
        seen: set[str] = set()
        result: Dict[str, Tuple[str, ...]] = {}
        for holdout in HOLDOUTS:
            identifiers = by_holdout[holdout]
            current = set(identifiers)
            if not identifiers or len(current) != len(identifiers) or seen & current:
                raise WorkerStateError(
                    "candidate suite has discovery/confirmation/audit holdout overlap"
                )
            seen.update(current)
            result[holdout] = tuple(sorted(current))
        return MappingProxyType(result)

    def _curation_receipt_path(self, pipeline_spec: Path) -> Path:
        return (
            pipeline_spec.parent
            / ".suite-rotation-worker"
            / "curation-receipt.json"
        )

    def _curation_receipt(
        self,
        *,
        request_id: str,
        supplement_spec: Path,
        pipeline_spec: Path,
        suite_manifest: Path,
        suite_id: str,
        champion_sha256: str,
    ) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": CURATION_RECEIPT_CONTRACT,
            "request_id": request_id,
            "worker_spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "supplement_spec": {
                "path": str(supplement_spec),
                "sha256": file_sha256(supplement_spec),
                "identity": _load_canonical_object(
                    supplement_spec, "supplement spec"
                )["spec_sha256"],
            },
            "pipeline_spec": {
                "path": str(pipeline_spec),
                "sha256": file_sha256(pipeline_spec),
                "identity": _load_canonical_object(
                    pipeline_spec, "pipeline spec"
                )["spec_sha256"],
            },
            "suite": {
                "suite_id": suite_id,
                "manifest_path": str(suite_manifest),
                "manifest_sha256": file_sha256(suite_manifest),
                "manifest_identity": _load_canonical_object(
                    suite_manifest, "suite manifest"
                )["manifestPayloadSha256"],
            },
            "models": {
                "original_sha256": self.spec.original_model.sha256,
                "champion_sha256": champion_sha256,
            },
            "policy_identity": self.spec.policy.identity,
            "training_inputs_admitted": False,
            "activation_performed": False,
            "service_activation_invoked": False,
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def curate(
        self,
        *,
        request_id: str,
        supplement_spec: Path,
        pipeline_spec: Path,
        suite_manifest: Path,
        poll_interval: float = 30.0,
    ) -> Mapping[str, Any]:
        """Drive supplement then pipeline to one validated v3 suite."""

        self._assert_frozen()
        identifier = _require_id(request_id, "rotation request ID")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(float(poll_interval))
            or float(poll_interval) < 0
        ):
            raise WorkerStateError("curation poll interval must be nonnegative")
        supplement_path = _absolute_path(
            supplement_spec,
            "supplement specification",
            error_type=WorkerStateError,
            require_file=True,
        )
        pipeline_path = _absolute_path(
            pipeline_spec,
            "pipeline specification",
            error_type=WorkerStateError,
            require_file=True,
        )
        _, _, supplement_raw, pipeline_raw = self._load_materialized_pair(
            supplement_path, pipeline_path
        )
        champion = _file_binding(
            pipeline_raw["models"]["champion"],
            "materialized champion",
            error_type=WorkerStateError,
        )
        suite_path = _future_file(suite_manifest, "candidate suite manifest")
        expected_suite = Path(pipeline_raw["outputs"]["suite_directory"]) / "manifest.json"
        if suite_path != expected_suite:
            raise WorkerStateError(
                "curation suite manifest differs from pipeline output binding"
            )
        if identifier not in Path(pipeline_raw["work_root"]).parts:
            raise WorkerStateError("pipeline work root is not request-isolated")
        if not suite_path.is_file():
            self.supplement_executor(
                supplement_path, poll_interval=float(poll_interval)
            )
            self._assert_frozen()
            self.pipeline_executor(pipeline_path, poll_interval=float(poll_interval))
            self._assert_frozen()
        if suite_path.is_symlink() or not suite_path.is_file():
            raise WorkerStateError(
                "curation pipeline completed without a suite manifest"
            )
        if (
            champion.path.is_symlink()
            or not champion.path.is_file()
            or file_sha256(champion.path) != champion.sha256
        ):
            raise WorkerStateError("frozen champion changed during curation")
        holdouts = self._manual_suite_provenance(
            suite_path,
            expected_policy_identity=self.spec.policy.identity,
            expected_original_sha256=self.spec.original_model.sha256,
            expected_champion_sha256=champion.sha256,
        )
        try:
            validated = self.suite_validator(
                suite_path,
                self.spec.registry.path,
                expected_champion_sha256=champion.sha256,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WorkerStateError(f"candidate suite validation failed: {exc}") from exc
        suite_id = getattr(validated, "suite_id", file_sha256(suite_path))
        if suite_id != file_sha256(suite_path):
            raise WorkerStateError("candidate suite ID is not its manifest hash")
        receipt = self._curation_receipt(
            request_id=identifier,
            supplement_spec=supplement_path,
            pipeline_spec=pipeline_path,
            suite_manifest=suite_path,
            suite_id=suite_id,
            champion_sha256=champion.sha256,
        )
        receipt_path = self._curation_receipt_path(pipeline_path)
        _atomic_immutable_json(receipt_path, receipt)
        observed, observed_identity = _load_self_hashed(
            receipt_path,
            role="curation receipt",
            contract=CURATION_RECEIPT_CONTRACT,
            hash_field="receipt_sha256",
        )
        if observed != receipt or observed_identity != receipt["receipt_sha256"]:
            raise WorkerStateError("curation receipt replay is contradictory")
        return MappingProxyType(
            {
                "request_id": identifier,
                "suite_id": suite_id,
                "suite_manifest": {
                    "path": str(suite_path),
                    "sha256": file_sha256(suite_path),
                    "identity": receipt["suite"]["manifest_identity"],
                },
                "semantic_holdouts": {
                    holdout: {
                        "count": len(holdouts[holdout]),
                        "sha256": canonical_sha256(list(holdouts[holdout])),
                    }
                    for holdout in HOLDOUTS
                },
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": file_sha256(receipt_path),
                    "identity": receipt["receipt_sha256"],
                },
                "training_inputs_admitted": False,
                "activation_performed": False,
            }
        )

    def _continuity_context(
        self,
        *,
        request_id: str,
        role: str,
        model_path: Path,
        model_sha256: str,
        candidate_suite_id: str,
        candidate_suite_manifest: Optional[Path],
    ) -> Mapping[str, Any]:
        identifier = _require_id(request_id, "rotation request ID")
        if role not in {"current_champion", "previous_champion"}:
            raise WorkerStateError("continuity role is unsupported")
        suite_id = _require_sha256(
            candidate_suite_id,
            "candidate suite ID",
            error_type=WorkerStateError,
        )
        supplied_model = FileBinding(
            _absolute_path(
                model_path,
                f"{role} model",
                error_type=WorkerStateError,
                require_file=True,
            ),
            _require_sha256(
                model_sha256, f"{role} model hash", error_type=WorkerStateError
            ),
        )
        if file_sha256(supplied_model.path) != supplied_model.sha256:
            raise WorkerStateError(f"{role} model hash changed")
        state = self.registry.reconstruct()
        request = state.requests.get(identifier)
        registration = state.registrations.get(suite_id)
        version = state.versions.get(suite_id)
        manifest_path = _absolute_path(
            (
                version.manifest_path
                if candidate_suite_manifest is None and version is not None
                else candidate_suite_manifest
            ),
            "registered candidate suite manifest",
            error_type=WorkerStateError,
            require_file=True,
        )
        if (
            request is None
            or registration is None
            or registration.get("request_id") != identifier
            or version is None
            or Path(version.manifest_path) != manifest_path
        ):
            raise WorkerStateError(
                "candidate suite is not registered to this rotation request"
            )
        if role == "current_champion":
            expected_model = state.current_champion
            expected_hash = request.get("champion_sha256")
        else:
            expected_hash = state.previous_champion_sha256
            expected_model = (
                None
                if expected_hash is None
                else state.champion_history.get(expected_hash)
            )
        if (
            expected_model is None
            or expected_hash != expected_model.sha256
            or supplied_model.path != Path(expected_model.path)
            or supplied_model.sha256 != expected_model.sha256
        ):
            raise WorkerStateError(f"{role} model is not the frozen registry model")
        try:
            validated = self.suite_validator(
                manifest_path,
                self.spec.registry.path,
                expected_champion_sha256=request.get("champion_sha256"),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise WorkerStateError(
                f"registered candidate suite validation failed: {exc}"
            ) from exc
        if (
            getattr(validated, "suite_id", suite_id) != suite_id
            or file_sha256(manifest_path) != version.manifest_sha256
        ):
            raise WorkerStateError("registered candidate suite identity changed")
        holdouts = self._manual_suite_provenance(
            manifest_path,
            expected_policy_identity=self.spec.policy.identity,
            expected_original_sha256=self.spec.original_model.sha256,
            expected_champion_sha256=request["champion_sha256"],
        )
        return {
            "request_id": identifier,
            "role": role,
            "model": supplied_model,
            "suite_id": suite_id,
            "manifest_path": manifest_path,
            "version": version,
            "request": request,
            "state": state,
            "holdouts": holdouts,
        }

    @staticmethod
    def _assert_continuity_files_frozen(context: Mapping[str, Any]) -> None:
        model = context["model"]
        version = context["version"]
        manifest = context["manifest_path"]
        if (
            model.path.is_symlink()
            or not model.path.is_file()
            or file_sha256(model.path) != model.sha256
        ):
            raise WorkerStateError("continuity model changed during shadow replay")
        if (
            manifest.is_symlink()
            or not manifest.is_file()
            or file_sha256(manifest) != version.manifest_sha256
        ):
            raise WorkerStateError(
                "candidate suite manifest changed during shadow replay"
            )

    def _continuity_work_root(
        self, evidence_path: Path, context: Mapping[str, Any]
    ) -> Path:
        identity = canonical_sha256(
            {
                "worker_spec_identity": self.spec.identity,
                "request_id": context["request_id"],
                "role": context["role"],
                "model_sha256": context["model"].sha256,
                "candidate_suite_id": context["suite_id"],
                "candidate_version_sha256": context["version"].version_sha256,
            }
        )
        root = (
            evidence_path.parent
            / ".suite-rotation-worker"
            / context["role"]
            / identity
        )
        if any(
            _paths_overlap(root, exclusion)
            for exclusion in self.spec.training_input_exclusion_roots
        ):
            raise WorkerStateError(
                "continuity work root overlaps a training-input exclusion root"
            )
        _ensure_directory(root)
        return root

    def _command_values(
        self,
        context: Mapping[str, Any],
        *,
        phase: str,
        evidence: Path,
        receipt: Path,
        work_root: Path,
    ) -> Mapping[str, str]:
        version = context["version"]
        identifiers = context["holdouts"][phase]
        return {
            "worker_spec": str(self.spec.path),
            "worker_spec_sha256": self.spec.identity,
            "repository": str(self.spec.deployment.repository_path),
            "source_revision": self.spec.deployment.source_revision,
            "request_id": context["request_id"],
            "role": context["role"],
            "phase": phase,
            "gpu_id": self.spec.guardian_gpu_id,
            "model_path": str(context["model"].path),
            "model_sha256": context["model"].sha256,
            "candidate_suite_id": context["suite_id"],
            "candidate_suite_manifest": str(context["manifest_path"]),
            "candidate_suite_manifest_sha256": version.manifest_sha256,
            "candidate_suite_manifest_identity": version.manifest_identity,
            "policy_path": str(self.spec.policy.path),
            "policy_sha256": self.spec.policy.sha256,
            "policy_identity": self.spec.policy.identity,
            "katago": str(self.spec.katago.path),
            "katago_sha256": self.spec.katago.sha256,
            "analysis_config": str(self.spec.configs["analysis"].path),
            "analysis_config_sha256": self.spec.configs["analysis"].sha256,
            "powered_config": str(self.spec.configs["powered"].path),
            "powered_config_sha256": self.spec.configs["powered"].sha256,
            "standard_config": str(self.spec.configs["standard"].path),
            "standard_config_sha256": self.spec.configs["standard"].sha256,
            "original_model": str(self.spec.original_model.path),
            "original_model_sha256": self.spec.original_model.sha256,
            "stage_evidence": str(evidence),
            "stage_receipt": str(receipt),
            "work_root": str(work_root),
            "independent_cluster_ids_sha256": canonical_sha256(list(identifiers)),
            "seed": canonical_sha256(
                {
                    "request_id": context["request_id"],
                    "role": context["role"],
                    "phase": phase,
                    "suite_id": context["suite_id"],
                }
            ),
        }

    @staticmethod
    def _expand_command(
        template: Sequence[str],
        values: Mapping[str, str],
        role: str,
    ) -> Tuple[str, ...]:
        try:
            argv = tuple(part.format_map(values) for part in template)
        except (KeyError, ValueError) as exc:
            raise WorkerStateError(f"cannot expand {role}: {exc}") from exc
        if any(
            not part
            or "\x00" in part
            or "\n" in part
            or "\r" in part
            for part in argv
        ):
            raise WorkerStateError(f"{role} expanded to an unsafe argv")
        return argv

    def _validate_shadow_evidence(
        self,
        path: Path,
        *,
        context: Mapping[str, Any],
        phase: str,
        argv: Sequence[str],
    ) -> Mapping[str, Any]:
        value, _ = _load_self_hashed(
            path,
            role=f"{phase} shadow replay evidence",
            contract=SHADOW_REPLAY_EVIDENCE_CONTRACT,
            hash_field="evidence_sha256",
            expected_keys=_SHADOW_EVIDENCE_KEYS,
        )
        expected_ids = list(context["holdouts"][phase])
        expected = {
            "request_id": context["request_id"],
            "role": context["role"],
            "phase": phase,
            "holdout": phase,
            "candidate_suite_id": context["suite_id"],
            "candidate_suite_manifest_sha256": context[
                "version"
            ].manifest_sha256,
            "model_sha256": context["model"].sha256,
            "policy_identity": self.spec.policy.identity,
            "independent_cluster_ids": expected_ids,
            "independent_cluster_ids_sha256": canonical_sha256(expected_ids),
            "command_argv_sha256": canonical_sha256(list(argv)),
            "decision": "PASS",
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise WorkerCommandError(
                f"{phase} shadow replay evidence contradicts frozen inputs or did not PASS"
            )
        return value

    def _command_receipt(
        self,
        *,
        context: Mapping[str, Any],
        phase: str,
        argv: Sequence[str],
        evidence: Path,
    ) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": COMMAND_RECEIPT_CONTRACT,
            "worker_spec_identity": self.spec.identity,
            "request_id": context["request_id"],
            "role": context["role"],
            "phase": phase,
            "argv": list(argv),
            "argv_sha256": canonical_sha256(list(argv)),
            "frozen_inputs": {
                "candidate_suite_id": context["suite_id"],
                "candidate_version_sha256": context["version"].version_sha256,
                "candidate_manifest_sha256": context[
                    "version"
                ].manifest_sha256,
                "model_sha256": context["model"].sha256,
                "policy_identity": self.spec.policy.identity,
                "holdout_cluster_ids_sha256": canonical_sha256(
                    list(context["holdouts"][phase])
                ),
            },
            "evidence": {
                "path": str(evidence),
                "sha256": file_sha256(evidence),
                "identity": _load_canonical_object(
                    evidence, f"{phase} evidence"
                )["evidence_sha256"],
            },
            "returncode": 0,
            "decision": "PASS",
            "shell": False,
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_shadow_stage(
        self,
        context: Mapping[str, Any],
        *,
        phase: str,
        work_root: Path,
    ) -> Tuple[Path, Path, Mapping[str, Any]]:
        evidence = work_root / f"{phase}-evidence.json"
        receipt = work_root / f"{phase}-command-receipt.json"
        values = self._command_values(
            context,
            phase=phase,
            evidence=evidence,
            receipt=receipt,
            work_root=work_root,
        )
        argv = self._expand_command(
            self.spec.continuity_templates[phase],
            values,
            f"{phase} continuity command",
        )
        _assert_excluded_paths_absent(
            list(argv),
            self.spec.training_input_exclusion_roots,
            f"{phase} continuity command",
            error_type=WorkerStateError,
        )
        if os.path.lexists(os.fspath(receipt)):
            observed, _ = _load_self_hashed(
                receipt,
                role=f"{phase} command receipt",
                contract=COMMAND_RECEIPT_CONTRACT,
                hash_field="receipt_sha256",
            )
            evidence_value = self._validate_shadow_evidence(
                evidence, context=context, phase=phase, argv=argv
            )
            expected = self._command_receipt(
                context=context, phase=phase, argv=argv, evidence=evidence
            )
            if observed != expected:
                raise WorkerConflictError(
                    f"{phase} continuity command receipt conflicts with replay"
                )
            return evidence, receipt, evidence_value
        if os.path.lexists(os.fspath(evidence)):
            evidence_value = self._validate_shadow_evidence(
                evidence, context=context, phase=phase, argv=argv
            )
        else:
            completed = self.command_runner(
                list(argv),
                cwd=str(self.spec.deployment.repository_path),
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
            returncode = getattr(completed, "returncode", None)
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise WorkerCommandError(
                    f"{phase} continuity runner returned no integer status"
                )
            if returncode != 0:
                stderr = str(getattr(completed, "stderr", "")).strip()
                raise WorkerCommandError(
                    f"{phase} continuity command returned {returncode}: {stderr}"
                )
            self._assert_frozen()
            self._assert_continuity_files_frozen(context)
            if not evidence.is_file() or evidence.is_symlink():
                raise WorkerCommandError(
                    f"{phase} continuity command published no canonical evidence"
                )
            evidence_value = self._validate_shadow_evidence(
                evidence, context=context, phase=phase, argv=argv
            )
        receipt_value = self._command_receipt(
            context=context, phase=phase, argv=argv, evidence=evidence
        )
        _atomic_immutable_json(receipt, receipt_value)
        return evidence, receipt, evidence_value

    def _role_evidence_value(
        self,
        context: Mapping[str, Any],
        *,
        completed_at_utc: str,
    ) -> Dict[str, Any]:
        version = context["version"]
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTINUITY_EVIDENCE_CONTRACT,
            "request_id": context["request_id"],
            "base_suite_id": context["request"]["base_suite_id"],
            "candidate_suite": {
                "suite_id": context["suite_id"],
                "version_sha256": version.version_sha256,
                "manifest_path": str(version.manifest_path),
                "manifest_sha256": version.manifest_sha256,
                "manifest_identity": version.manifest_identity,
            },
            "role": context["role"],
            "model": context["model"].to_dict(),
            "policy": {
                "path": str(self.spec.registry_value.policy_path),
                "sha256": self.spec.registry_value.policy_file_sha256,
                "identity": self.spec.registry_value.policy_identity,
            },
            "decision": "PASS",
            "completed_at_utc": completed_at_utc,
        }
        value["evidence_sha256"] = canonical_sha256(value)
        return value

    def _validate_role_evidence(
        self,
        path: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        value, _ = _load_self_hashed(
            path,
            role=f"{context['role']} continuity evidence",
            contract=CONTINUITY_EVIDENCE_CONTRACT,
            hash_field="evidence_sha256",
            expected_keys=_ROLE_EVIDENCE_KEYS,
        )
        completed = _parse_utc(
            value.get("completed_at_utc"), "continuity completion timestamp"
        )
        expected = self._role_evidence_value(
            context, completed_at_utc=_format_utc(completed)
        )
        if value != expected:
            raise WorkerStateError(
                f"{context['role']} continuity evidence contradicts registry state"
            )
        return value

    @staticmethod
    def _infer_peer_evidence(path: Path, role: str) -> Optional[Path]:
        peer = (
            "previous_champion"
            if role == "current_champion"
            else "current_champion"
        )
        text = str(path)
        if text.count(role) != 1:
            return None
        return Path(text.replace(role, peer, 1))

    def continuity(
        self,
        *,
        request_id: str,
        role: str,
        model_path: Path,
        model_sha256: str,
        candidate_suite_id: str,
        candidate_suite_manifest: Optional[Path] = None,
        continuity_evidence: Path,
        continuity_manifest: Optional[Path] = None,
        peer_evidence: Optional[Path] = None,
    ) -> Mapping[str, Any]:
        """Run discovery and confirmation replay, then publish strict PASS evidence."""

        self._assert_frozen()
        context = self._continuity_context(
            request_id=request_id,
            role=role,
            model_path=model_path,
            model_sha256=model_sha256,
            candidate_suite_id=candidate_suite_id,
            candidate_suite_manifest=candidate_suite_manifest,
        )
        output = _future_file(continuity_evidence, "continuity evidence output")
        if any(
            _paths_overlap(output, root)
            for root in self.spec.training_input_exclusion_roots
        ):
            raise WorkerStateError(
                "continuity evidence overlaps a training-input exclusion root"
            )
        work_root = self._continuity_work_root(output, context)
        discovery_path, discovery_receipt, discovery = self._ensure_shadow_stage(
            context, phase="discovery", work_root=work_root
        )
        confirmation_path, confirmation_receipt, confirmation = (
            self._ensure_shadow_stage(
                context, phase="confirmation", work_root=work_root
            )
        )
        if set(discovery["independent_cluster_ids"]).intersection(
            confirmation["independent_cluster_ids"]
        ):
            raise WorkerCommandError(
                "continuity discovery and confirmation evidence overlap"
            )
        fresh_context = self._continuity_context(
            request_id=context["request_id"],
            role=context["role"],
            model_path=context["model"].path,
            model_sha256=context["model"].sha256,
            candidate_suite_id=context["suite_id"],
            candidate_suite_manifest=context["manifest_path"],
        )
        if (
            fresh_context["request"] != context["request"]
            or fresh_context["version"].version_sha256
            != context["version"].version_sha256
        ):
            raise WorkerStateError(
                "registry continuity coordinates changed during shadow replay"
            )
        context = fresh_context
        self._assert_continuity_files_frozen(context)
        if os.path.lexists(os.fspath(output)):
            final = self._validate_role_evidence(output, context)
        else:
            final = self._role_evidence_value(
                context, completed_at_utc=_format_utc(self._now())
            )
            _atomic_immutable_json(output, final)
            final = self._validate_role_evidence(output, context)

        manifest_result: Optional[Mapping[str, Any]] = None
        if continuity_manifest is not None:
            manifest_path = _future_file(
                continuity_manifest, "continuity manifest output"
            )
            peer_path = (
                Path(peer_evidence)
                if peer_evidence is not None
                else self._infer_peer_evidence(output, context["role"])
            )
            if peer_path is not None and peer_path.is_file() and not peer_path.is_symlink():
                peer_role = (
                    "previous_champion"
                    if context["role"] == "current_champion"
                    else "current_champion"
                )
                state = context["state"]
                if peer_role == "current_champion":
                    peer_model = state.current_champion
                else:
                    peer_hash = state.previous_champion_sha256
                    peer_model = state.champion_history.get(peer_hash)
                if peer_model is None:
                    raise WorkerStateError("peer continuity model is unavailable")
                peer_context = self._continuity_context(
                    request_id=context["request_id"],
                    role=peer_role,
                    model_path=peer_model.path,
                    model_sha256=peer_model.sha256,
                    candidate_suite_id=context["suite_id"],
                    candidate_suite_manifest=context["manifest_path"],
                )
                peer = self._validate_role_evidence(peer_path, peer_context)
                current_path = output if context["role"] == "current_champion" else peer_path
                previous_path = (
                    output if context["role"] == "previous_champion" else peer_path
                )
                completed_at = max(
                    final["completed_at_utc"], peer["completed_at_utc"]
                )
                manifest_result = publish_continuity_manifest(
                    manifest_path,
                    request_id=context["request_id"],
                    candidate_suite_id=context["suite_id"],
                    base_suite_id=context["request"]["base_suite_id"],
                    policy_hash=self.spec.policy.identity,
                    current_champion_sha256=state.current_champion.sha256,
                    previous_champion_sha256=state.previous_champion_sha256,
                    current_evidence_path=current_path,
                    previous_evidence_path=previous_path,
                    completed_at_utc=completed_at,
                )
        return MappingProxyType(
            {
                "request_id": context["request_id"],
                "role": context["role"],
                "decision": "PASS",
                "evidence": {
                    "path": str(output),
                    "sha256": file_sha256(output),
                    "identity": final["evidence_sha256"],
                },
                "shadow_replays": {
                    "discovery": {
                        "evidence": {
                            "path": str(discovery_path),
                            "sha256": file_sha256(discovery_path),
                            "identity": discovery["evidence_sha256"],
                        },
                        "receipt": {
                            "path": str(discovery_receipt),
                            "sha256": file_sha256(discovery_receipt),
                        },
                    },
                    "confirmation": {
                        "evidence": {
                            "path": str(confirmation_path),
                            "sha256": file_sha256(confirmation_path),
                            "identity": confirmation["evidence_sha256"],
                        },
                        "receipt": {
                            "path": str(confirmation_receipt),
                            "sha256": file_sha256(confirmation_receipt),
                        },
                    },
                },
                "continuity_manifest": (
                    None
                    if manifest_result is None
                    else {
                        "path": str(Path(continuity_manifest)),
                        "sha256": file_sha256(Path(continuity_manifest)),
                        "identity": manifest_result["manifest_sha256"],
                    }
                ),
                "training_inputs_admitted": False,
                "activation_performed": False,
            }
        )


def publish_shadow_replay_evidence(
    path: Path,
    *,
    request_id: str,
    role: str,
    phase: str,
    candidate_suite_id: str,
    candidate_suite_manifest_sha256: str,
    model_sha256: str,
    policy_identity: str,
    independent_cluster_ids: Sequence[str],
    command_argv: Sequence[str],
    decision: str = "PASS",
) -> Mapping[str, Any]:
    """Publish the strict command evidence consumed by ``continuity``."""

    if phase not in {"discovery", "confirmation"}:
        raise ValueError("shadow replay phase is unsupported")
    if role not in {"current_champion", "previous_champion"}:
        raise ValueError("shadow replay role is unsupported")
    identifiers = sorted(independent_cluster_ids)
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(_SHA256_RE.fullmatch(item or "") is None for item in identifiers)
    ):
        raise ValueError("shadow replay independent-cluster IDs are invalid")
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SHADOW_REPLAY_EVIDENCE_CONTRACT,
        "request_id": _require_id(request_id, "rotation request ID"),
        "role": role,
        "phase": phase,
        "holdout": phase,
        "candidate_suite_id": _require_sha256(
            candidate_suite_id,
            "candidate suite ID",
            error_type=WorkerStateError,
        ),
        "candidate_suite_manifest_sha256": _require_sha256(
            candidate_suite_manifest_sha256,
            "candidate suite manifest hash",
            error_type=WorkerStateError,
        ),
        "model_sha256": _require_sha256(
            model_sha256, "continuity model hash", error_type=WorkerStateError
        ),
        "policy_identity": _require_sha256(
            policy_identity, "policy identity", error_type=WorkerStateError
        ),
        "independent_cluster_ids": identifiers,
        "independent_cluster_ids_sha256": canonical_sha256(identifiers),
        "command_argv_sha256": canonical_sha256(list(command_argv)),
        "decision": decision,
    }
    value["evidence_sha256"] = canonical_sha256(value)
    _atomic_immutable_json(Path(path).resolve(), value)
    return value


def materialize(spec_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    return SuiteRotationWorker(spec_path).materialize(**kwargs)


def curate(spec_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    return SuiteRotationWorker(spec_path).curate(**kwargs)


def continuity(spec_path: Path, **kwargs: Any) -> Mapping[str, Any]:
    return SuiteRotationWorker(spec_path).continuity(**kwargs)


load_spec = load_worker_spec
RotationWorker = SuiteRotationWorker


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        command = subparsers.add_parser(name)
        command.add_argument("--spec", required=True, type=Path)
        command.add_argument("--expected-spec-sha256")
        command.add_argument("--request-id", required=True)
        return command

    materializer = common("materialize")
    materializer.add_argument(
        "--rotation-request", "--request", dest="rotation_request", required=True, type=Path
    )
    materializer.add_argument("--rotation-request-sha256")
    materializer.add_argument("--rotation-request-identity")
    materializer.add_argument("--supplement-request", required=True, type=Path)
    materializer.add_argument("--supplement-request-sha256")
    materializer.add_argument("--supplement-request-identity")
    materializer.add_argument("--pipeline-request", required=True, type=Path)
    materializer.add_argument("--pipeline-request-sha256")
    materializer.add_argument("--pipeline-request-identity")
    materializer.add_argument("--supplement-spec", required=True, type=Path)
    materializer.add_argument("--pipeline-spec", required=True, type=Path)

    curator = common("curate")
    curator.add_argument("--supplement-spec", required=True, type=Path)
    curator.add_argument("--pipeline-spec", required=True, type=Path)
    curator.add_argument("--suite-manifest", required=True, type=Path)
    curator.add_argument("--poll-interval", type=float, default=30.0)

    continuity_parser = common("continuity")
    continuity_parser.add_argument(
        "--role",
        required=True,
        choices=("current_champion", "previous_champion"),
    )
    continuity_parser.add_argument(
        "--model", "--model-path", dest="model_path", required=True, type=Path
    )
    continuity_parser.add_argument("--model-sha256", required=True)
    continuity_parser.add_argument(
        "--suite-id",
        "--candidate-suite-id",
        dest="candidate_suite_id",
        required=True,
    )
    continuity_parser.add_argument(
        "--suite-manifest",
        "--candidate-suite-manifest",
        dest="candidate_suite_manifest",
        type=Path,
    )
    continuity_parser.add_argument(
        "--evidence",
        "--continuity-evidence",
        dest="continuity_evidence",
        required=True,
        type=Path,
    )
    continuity_parser.add_argument("--continuity-manifest", type=Path)
    continuity_parser.add_argument("--peer-evidence", type=Path)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    supplement_executor: Callable[..., Mapping[str, Any]] = (
        _default_supplement_executor
    ),
    pipeline_executor: Callable[..., Mapping[str, Any]] = _default_pipeline_executor,
) -> int:
    args = parse_args(argv)
    try:
        worker = SuiteRotationWorker(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
            command_runner=command_runner,
            supplement_executor=supplement_executor,
            pipeline_executor=pipeline_executor,
        )
        if args.mode == "materialize":
            result = worker.materialize(
                request_id=args.request_id,
                rotation_request=args.rotation_request,
                supplement_request=args.supplement_request,
                pipeline_request=args.pipeline_request,
                supplement_spec=args.supplement_spec,
                pipeline_spec=args.pipeline_spec,
                rotation_request_sha256=args.rotation_request_sha256,
                rotation_request_identity=args.rotation_request_identity,
                supplement_request_sha256=args.supplement_request_sha256,
                supplement_request_identity=args.supplement_request_identity,
                pipeline_request_sha256=args.pipeline_request_sha256,
                pipeline_request_identity=args.pipeline_request_identity,
            )
        elif args.mode == "curate":
            result = worker.curate(
                request_id=args.request_id,
                supplement_spec=args.supplement_spec,
                pipeline_spec=args.pipeline_spec,
                suite_manifest=args.suite_manifest,
                poll_interval=args.poll_interval,
            )
        else:
            result = worker.continuity(
                request_id=args.request_id,
                role=args.role,
                model_path=args.model_path,
                model_sha256=args.model_sha256,
                candidate_suite_id=args.candidate_suite_id,
                candidate_suite_manifest=args.candidate_suite_manifest,
                continuity_evidence=args.continuity_evidence,
                continuity_manifest=args.continuity_manifest,
                peer_evidence=args.peer_evidence,
            )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "contract": ERROR_CONTRACT,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        print(canonical_json(error), file=sys.stderr)
        return 2
    print(canonical_json(dict(result)))
    return 0


__all__ = [
    "COMMAND_RECEIPT_CONTRACT",
    "CURATION_RECEIPT_CONTRACT",
    "DeploymentBinding",
    "DirectoryBinding",
    "ERROR_CONTRACT",
    "FileBinding",
    "IdentityBinding",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "SHADOW_REPLAY_EVIDENCE_CONTRACT",
    "RotationWorker",
    "SuiteRotationWorker",
    "SuiteRotationWorkerError",
    "WORKER_SPEC_CONTRACT",
    "WorkerCommandError",
    "WorkerConflictError",
    "WorkerSpec",
    "WorkerSpecError",
    "WorkerStateError",
    "continuity",
    "curate",
    "load_worker_spec",
    "load_spec",
    "main",
    "materialize",
    "parse_args",
    "publish_shadow_replay_evidence",
    "publish_worker_spec",
]


if __name__ == "__main__":
    raise SystemExit(main())
