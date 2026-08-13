#!/usr/bin/env python3
"""Run the live trainer/evaluator GPU7 bootstrap lease drill.

The command accepts one canonical, self-hashed specification and never invokes
a shell.  It uses the production GPU lease state machine to drain the trainer,
validate its checkpoint handoff, launch the configured evaluator, run one
bounded probe command, release the evaluator, reconcile the lease, and prove
that the trainer was restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from risk_score.autonomy_bootstrap import publish_gate_evidence
from risk_score.gpu_lease import (
    SCHEMA_VERSION as GPU_LEASE_SCHEMA_VERSION,
)
from risk_score.gpu_lease import (
    GpuLeaseError,
    GpuLeaseManager,
    GpuObservation,
    LeaseRecord,
    ProcessIdentity,
    ProcessRunner,
    RuntimeConfig,
)

SCHEMA_VERSION = 1
SPEC_CONTRACT = "risk-score-autonomy-lease-drill-spec-v1"
ERROR_CONTRACT = "risk-score-autonomy-lease-drill-error-v1"
PROBE_RECEIPT_CONTRACT = "risk-score-autonomy-lease-probe-receipt-v1"
GATE_ID = "trainer-evaluator-lease-drill"
GPU_INDEX = 7
MAX_JSON_BYTES = 1024 * 1024
MAX_PROBE_ARGUMENTS = 256
MAX_PROBE_ARGV_BYTES = 64 * 1024
MAX_PROBE_OUTPUT_BYTES = 1024 * 1024
MAX_PROBE_TIMEOUT_SECONDS = 600.0

LEASE_CHECK_FIELDS = frozenset(
    {
        "gpu_uuid",
        "lease_schema_version",
        "trainer_drained",
        "checkpoint_handoff_verified",
        "evaluator_exclusive",
        "process_overlap_observed",
        "trainer_restored",
        "lease_clean_observations",
        "release_clean_observations",
        "safety_halt",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_COMMON_SPEC_KEYS = {
    "schema_version",
    "contract",
    "expected_gpu_uuid",
    "minimum_clean_observations",
    "work_root",
    "evidence_output",
    "spec_sha256",
}
_FLAT_CONFIG_KEYS = {"gpu_lease_config_path", "gpu_lease_config_sha256"}
_FLAT_PROBE_KEYS = {
    "evaluator_probe_argv",
    "evaluator_probe_timeout_seconds",
    "evaluator_probe_executable_sha256",
    "evaluator_probe_model_sha256",
    "evaluator_probe_config_sha256",
    "evaluator_probe_minimum_completed_work",
}
_DIRECT_SOURCE_KEY_PAIRS = (
    ("trainer_process_identity_path", "trainer_process_identity_sha256"),
    ("trainer_identity_path", "trainer_identity_sha256"),
)
_CONSUMER_SOURCE_KEY_PAIRS = (
    ("supervisor_consumer_state_path", "supervisor_consumer_state_sha256"),
    ("consumer_state_path", "consumer_state_sha256"),
)
_SNAKE_IDENTITY_KEYS = {
    "pid",
    "start_time_ticks",
    "process_group_id",
    "boot_id",
    "command_sha256",
    "cgroup",
}
_CAMEL_IDENTITY_KEYS = {
    "pid",
    "startTimeTicks",
    "processGroupId",
    "bootId",
    "commandSha256",
    "cgroup",
}


class LeaseDrillError(RuntimeError):
    """A fail-closed drill error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": ERROR_CONTRACT,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def verify(self, role: str) -> None:
        source = _normalized_absolute(self.path, role, require_file=True)
        observed = _file_sha256(source)
        if observed != self.sha256:
            raise LeaseDrillError(
                "bound_input_changed",
                f"{role} does not match its bound SHA-256",
                details={
                    "path": os.fspath(source),
                    "expected_sha256": self.sha256,
                    "observed_sha256": observed,
                },
            )


@dataclass(frozen=True)
class LeaseDrillSpec:
    source_path: Path
    source_file_sha256: str
    spec_sha256: str
    gpu_lease_config: FileBinding
    trainer_source_kind: str
    trainer_source: Optional[FileBinding]
    expected_gpu_uuid: str
    minimum_clean_observations: int
    evaluator_probe_argv: Tuple[str, ...]
    evaluator_probe_timeout_seconds: float
    evaluator_probe_executable_sha256: str
    evaluator_probe_model_sha256: str
    evaluator_probe_config_sha256: str
    evaluator_probe_minimum_completed_work: int
    work_root: Path
    evidence_output: Path

    @property
    def path(self) -> Path:
        return self.source_path

    @property
    def file_sha256(self) -> str:
        return self.source_file_sha256

    @property
    def identity(self) -> str:
        return self.spec_sha256

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_spec_sha256: Optional[str] = None,
    ) -> "LeaseDrillSpec":
        return load_drill_spec(path, expected_spec_sha256=expected_spec_sha256)

    def verify_immutable_inputs(self, *, include_trainer_source: bool) -> None:
        source = _normalized_absolute(
            self.source_path,
            "lease drill specification",
            require_file=True,
        )
        if _file_sha256(source) != self.source_file_sha256:
            raise LeaseDrillError(
                "drill_spec_changed",
                "Lease drill specification changed after validation",
            )
        self.gpu_lease_config.verify("GPU lease runtime config")
        executable = Path(self.evaluator_probe_argv[0])
        if _file_sha256(executable) != self.evaluator_probe_executable_sha256:
            raise LeaseDrillError(
                "bound_input_changed",
                "evaluator probe executable changed",
            )
        if include_trainer_source and self.trainer_source is not None:
            self.trainer_source.verify("trainer identity source")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise LeaseDrillError(
            "file_hash_failed",
            f"Could not hash {path}: {exc}",
            details={"path": os.fspath(path)},
        ) from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, role: str) -> str:
    if not _is_sha256(value):
        raise LeaseDrillError(
            "invalid_drill_spec", f"{role} must be a lowercase SHA-256"
        )
    return str(value)


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LeaseDrillError("invalid_json", f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LeaseDrillError(
        "invalid_json", f"Non-finite JSON value is forbidden: {value}"
    )


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise LeaseDrillError(
                "unsafe_path",
                f"{role} has a symlinked path component",
                details={"component": os.fspath(current)},
            )
        if current.parent == current:
            return
        current = current.parent


def _normalized_absolute(
    value: Any,
    role: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise LeaseDrillError("unsafe_path", f"{role} must be an absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise LeaseDrillError(
            "unsafe_path",
            f"{role} must be absolute and lexically normalized",
            details={"path": os.fspath(path)},
        )
    _reject_symlink_ancestors(path, role)
    if require_file:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise LeaseDrillError(
                "unsafe_path", f"{role} must be an existing regular file"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise LeaseDrillError(
                "unsafe_path", f"{role} must be an existing regular file"
            )
    if require_directory and (not path.is_dir() or path.is_symlink()):
        raise LeaseDrillError(
            "unsafe_path", f"{role} must be an existing non-symlink directory"
        )
    return path


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = _normalized_absolute(path, role, require_file=True)
    metadata = source.lstat()
    if metadata.st_size > MAX_JSON_BYTES:
        raise LeaseDrillError("invalid_json", f"{role} exceeds the size limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except LeaseDrillError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeaseDrillError(
            "invalid_json", f"Could not decode {role}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LeaseDrillError("invalid_json", f"{role} root must be an object")
    if data != (_canonical_json(value) + "\n").encode("utf-8"):
        raise LeaseDrillError(
            "noncanonical_json",
            f"{role} must be canonical newline-terminated JSON",
        )
    return value


def _strictly_within(path: Path, parent: Path) -> bool:
    if path == parent:
        return False
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _same_existing_file(left: Path, right: Path) -> bool:
    if not left.exists() or not right.exists():
        return False
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _binding(path: Any, digest: Any, role: str) -> FileBinding:
    source = _normalized_absolute(path, f"{role} path", require_file=True)
    expected = _require_sha256(digest, f"{role} hash")
    binding = FileBinding(source, expected)
    binding.verify(role)
    return binding


def _binding_object(value: Any, role: str) -> FileBinding:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise LeaseDrillError(
            "invalid_drill_spec",
            f"{role} must contain exactly path and sha256",
        )
    return _binding(value["path"], value["sha256"], role)


def _validate_probe_argv(value: Any) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_PROBE_ARGUMENTS
        or any(
            not isinstance(part, str)
            or not part
            or "\0" in part
            or "\n" in part
            or "\r" in part
            for part in value
        )
    ):
        raise LeaseDrillError(
            "invalid_drill_spec",
            "evaluator probe argv must be a bounded nonempty string array",
        )
    encoded_size = sum(len(part.encode("utf-8")) + 1 for part in value)
    if encoded_size > MAX_PROBE_ARGV_BYTES:
        raise LeaseDrillError(
            "invalid_drill_spec", "evaluator probe argv exceeds the size limit"
        )
    executable = _normalized_absolute(
        value[0], "evaluator probe executable", require_file=True
    )
    if not os.access(executable, os.X_OK):
        raise LeaseDrillError(
            "invalid_drill_spec", "evaluator probe executable is not executable"
        )
    prohibited = {"sigstop", "stop", "-stop", "-sigstop", "-19"}
    if any(part.strip().lower() in prohibited for part in value):
        raise LeaseDrillError(
            "invalid_drill_spec", "evaluator probe argv must never use SIGSTOP"
        )
    return tuple(value)


def _validate_timeout(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= MAX_PROBE_TIMEOUT_SECONDS
    ):
        raise LeaseDrillError(
            "invalid_drill_spec",
            "evaluator probe timeout must be positive and at most "
            f"{MAX_PROBE_TIMEOUT_SECONDS:g} seconds",
        )
    return float(value)


def _select_flat_source(
    raw: Mapping[str, Any],
) -> Tuple[str, FileBinding, set[str]]:
    candidates = []
    for kind, pairs in (
        ("process-identity", _DIRECT_SOURCE_KEY_PAIRS),
        ("supervisor-consumer-state", _CONSUMER_SOURCE_KEY_PAIRS),
    ):
        for path_key, hash_key in pairs:
            present = {key for key in (path_key, hash_key) if key in raw}
            if present and present != {path_key, hash_key}:
                raise LeaseDrillError(
                    "invalid_drill_spec",
                    f"{path_key} and {hash_key} must be supplied together",
                )
            if present:
                candidates.append((kind, path_key, hash_key))
    if len(candidates) != 1:
        raise LeaseDrillError(
            "invalid_drill_spec",
            "exactly one trainer identity source must be configured",
        )
    kind, path_key, hash_key = candidates[0]
    binding = _binding(raw[path_key], raw[hash_key], "trainer identity source")
    return kind, binding, {path_key, hash_key}


def _parse_flat_spec(
    raw: Mapping[str, Any],
) -> Tuple[
    FileBinding,
    str,
    Optional[FileBinding],
    Tuple[str, ...],
    float,
    str,
    str,
    str,
    int,
    set[str],
]:
    trainer_kind, trainer_source, source_keys = _select_flat_source(raw)
    expected_keys = (
        _COMMON_SPEC_KEYS | _FLAT_CONFIG_KEYS | _FLAT_PROBE_KEYS | source_keys
    )
    if set(raw) != expected_keys:
        raise LeaseDrillError(
            "invalid_drill_spec",
            "Lease drill spec fields differ from the flat schema",
            details={
                "missing": sorted(expected_keys.difference(raw)),
                "extra": sorted(set(raw).difference(expected_keys)),
            },
        )
    config = _binding(
        raw["gpu_lease_config_path"],
        raw["gpu_lease_config_sha256"],
        "GPU lease runtime config",
    )
    argv = _validate_probe_argv(raw["evaluator_probe_argv"])
    timeout = _validate_timeout(raw["evaluator_probe_timeout_seconds"])
    return (
        config,
        trainer_kind,
        trainer_source,
        argv,
        timeout,
        _require_sha256(
            raw["evaluator_probe_executable_sha256"],
            "evaluator probe executable hash",
        ),
        _require_sha256(
            raw["evaluator_probe_model_sha256"],
            "evaluator probe model hash",
        ),
        _require_sha256(
            raw["evaluator_probe_config_sha256"],
            "evaluator probe config hash",
        ),
        _strict_identity_int(
            raw["evaluator_probe_minimum_completed_work"],
            "evaluator probe minimum completed work",
            positive=True,
        ),
        expected_keys,
    )


def _parse_nested_spec(
    raw: Mapping[str, Any],
) -> Tuple[
    FileBinding,
    str,
    FileBinding,
    Tuple[str, ...],
    float,
    str,
    str,
    str,
    int,
    set[str],
]:
    expected_keys = _COMMON_SPEC_KEYS | {
        "gpu_lease_config",
        "trainer_source",
        "evaluator_probe",
    }
    if set(raw) != expected_keys:
        raise LeaseDrillError(
            "invalid_drill_spec",
            "Lease drill spec fields differ from the nested schema",
            details={
                "missing": sorted(expected_keys.difference(raw)),
                "extra": sorted(set(raw).difference(expected_keys)),
            },
        )
    config = _binding_object(raw["gpu_lease_config"], "GPU lease runtime config")
    source = raw["trainer_source"]
    if not isinstance(source, Mapping) or "kind" not in source:
        raise LeaseDrillError(
            "invalid_drill_spec",
            "trainer_source must contain a kind",
        )
    kind = source["kind"]
    if kind not in {"process-identity", "supervisor-consumer-state", "launch"}:
        raise LeaseDrillError(
            "invalid_drill_spec", "trainer_source.kind is unsupported"
        )
    if kind == "launch":
        if set(source) != {"kind"}:
            raise LeaseDrillError(
                "invalid_drill_spec",
                "launch trainer_source may contain only kind",
            )
        trainer_source = None
    else:
        if set(source) != {"kind", "path", "sha256"}:
            raise LeaseDrillError(
                "invalid_drill_spec",
                "bound trainer_source fields differ from the schema",
            )
        trainer_source = _binding(
            source["path"], source["sha256"], "trainer identity source"
        )
    probe = raw["evaluator_probe"]
    probe_keys = {
        "argv",
        "timeout_seconds",
        "executable_sha256",
        "model_sha256",
        "config_sha256",
        "minimum_completed_work",
    }
    if not isinstance(probe, Mapping) or set(probe) != probe_keys:
        raise LeaseDrillError(
            "invalid_drill_spec",
            "evaluator_probe fields differ from the schema",
        )
    argv = _validate_probe_argv(probe["argv"])
    timeout = _validate_timeout(probe["timeout_seconds"])
    return (
        config,
        str(kind),
        trainer_source,
        argv,
        timeout,
        _require_sha256(probe["executable_sha256"], "evaluator probe executable hash"),
        _require_sha256(probe["model_sha256"], "evaluator probe model hash"),
        _require_sha256(probe["config_sha256"], "evaluator probe config hash"),
        _strict_identity_int(
            probe["minimum_completed_work"],
            "evaluator probe minimum completed work",
            positive=True,
        ),
        expected_keys,
    )


def load_drill_spec(
    path: Path,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> LeaseDrillSpec:
    """Load and fully validate one canonical lease-drill specification."""

    source = _normalized_absolute(path, "lease drill specification", require_file=True)
    source_file_hash = _file_sha256(source)
    raw = _load_canonical_object(source, "lease drill specification")
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or isinstance(raw.get("schema_version"), bool)
        or raw.get("contract") != SPEC_CONTRACT
    ):
        raise LeaseDrillError(
            "invalid_drill_spec", "Lease drill specification contract is unsupported"
        )
    supplied_spec_hash = _require_sha256(
        raw.get("spec_sha256"), "lease drill specification self-hash"
    )
    body = dict(raw)
    body.pop("spec_sha256", None)
    if _canonical_sha256(body) != supplied_spec_hash:
        raise LeaseDrillError(
            "invalid_drill_spec", "Lease drill specification self-hash is invalid"
        )
    if expected_spec_sha256 is not None:
        expected = _require_sha256(
            expected_spec_sha256, "expected lease drill specification hash"
        )
        if expected not in {supplied_spec_hash, source_file_hash}:
            raise LeaseDrillError(
                "invalid_drill_spec",
                "Lease drill specification hash is not expected",
            )

    if "gpu_lease_config" in raw:
        (
            config,
            source_kind,
            trainer_source,
            argv,
            timeout,
            probe_executable_hash,
            probe_model_hash,
            probe_config_hash,
            probe_minimum_work,
            _,
        ) = _parse_nested_spec(raw)
    else:
        (
            config,
            source_kind,
            trainer_source,
            argv,
            timeout,
            probe_executable_hash,
            probe_model_hash,
            probe_config_hash,
            probe_minimum_work,
            _,
        ) = _parse_flat_spec(raw)

    if _file_sha256(Path(argv[0])) != probe_executable_hash:
        raise LeaseDrillError(
            "bound_input_changed",
            "evaluator probe executable does not match its bound SHA-256",
        )

    gpu_uuid = raw["expected_gpu_uuid"]
    if not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise LeaseDrillError("invalid_drill_spec", "expected_gpu_uuid is malformed")
    minimum = raw["minimum_clean_observations"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise LeaseDrillError(
            "invalid_drill_spec",
            "minimum_clean_observations must be an integer >= 2",
        )
    work_root = _normalized_absolute(raw["work_root"], "lease drill work root")
    if work_root.parent == work_root:
        raise LeaseDrillError(
            "unsafe_path", "lease drill work root may not be a filesystem root"
        )
    if work_root.exists() and (work_root.is_symlink() or not work_root.is_dir()):
        raise LeaseDrillError(
            "unsafe_path", "lease drill work root must be a directory"
        )
    if not work_root.exists() and (
        work_root.parent.is_symlink() or not work_root.parent.is_dir()
    ):
        raise LeaseDrillError(
            "unsafe_path",
            "lease drill work root parent must be an existing non-symlink directory",
        )
    evidence = _normalized_absolute(
        raw["evidence_output"], "lease drill evidence output"
    )
    if not _strictly_within(evidence, work_root):
        raise LeaseDrillError(
            "unsafe_path", "lease drill evidence output must be inside work root"
        )
    if evidence.exists() and (evidence.is_symlink() or not evidence.is_file()):
        raise LeaseDrillError(
            "unsafe_path", "lease drill evidence output must be a regular file"
        )
    protected_inputs = [
        (source, "drill specification"),
        (config.path, "GPU lease runtime config"),
    ]
    if trainer_source is not None:
        protected_inputs.append((trainer_source.path, "trainer identity source"))
    for protected, role in protected_inputs:
        if protected == work_root or _strictly_within(protected, work_root):
            raise LeaseDrillError(
                "unsafe_path",
                f"{role} must not be stored in the mutable drill work root",
            )
        if evidence == protected or _same_existing_file(evidence, protected):
            raise LeaseDrillError("unsafe_path", f"evidence output aliases the {role}")

    if _file_sha256(source) != source_file_hash:
        raise LeaseDrillError(
            "drill_spec_changed",
            "Lease drill specification changed while it was loaded",
        )
    return LeaseDrillSpec(
        source_path=source,
        source_file_sha256=source_file_hash,
        spec_sha256=supplied_spec_hash,
        gpu_lease_config=config,
        trainer_source_kind=source_kind,
        trainer_source=trainer_source,
        expected_gpu_uuid=gpu_uuid,
        minimum_clean_observations=minimum,
        evaluator_probe_argv=argv,
        evaluator_probe_timeout_seconds=timeout,
        evaluator_probe_executable_sha256=probe_executable_hash,
        evaluator_probe_model_sha256=probe_model_hash,
        evaluator_probe_config_sha256=probe_config_hash,
        evaluator_probe_minimum_completed_work=probe_minimum_work,
        work_root=work_root,
        evidence_output=evidence,
    )


def _load_runtime_config(spec: LeaseDrillSpec) -> RuntimeConfig:
    spec.gpu_lease_config.verify("GPU lease runtime config")
    try:
        config = RuntimeConfig.from_json_file(spec.gpu_lease_config.path)
    except GpuLeaseError as exc:
        raise LeaseDrillError(
            "invalid_runtime_config",
            str(exc),
            details={"gpu_lease_code": exc.code},
        ) from exc
    spec.gpu_lease_config.verify("GPU lease runtime config")
    return config


def _validate_runtime_separation(spec: LeaseDrillSpec, config: RuntimeConfig) -> None:
    _validate_runtime_paths_separated(spec, config)
    _validate_runtime_contract(spec, config)


def _validate_runtime_contract(spec: LeaseDrillSpec, config: RuntimeConfig) -> None:
    if not config.mutation_enabled:
        raise LeaseDrillError(
            "mutation_disabled",
            "The GPU lease runtime must enable mutation for the live drill",
        )
    if config.gpu_index != GPU_INDEX:
        raise LeaseDrillError(
            "wrong_gpu_index",
            f"The lease drill is pinned to GPU{GPU_INDEX}",
            details={"configured_gpu_index": config.gpu_index},
        )
    if config.expected_gpu_uuid != spec.expected_gpu_uuid:
        raise LeaseDrillError(
            "gpu_uuid_mismatch",
            "The drill specification and GPU lease runtime bind different UUIDs",
        )
    if config.clean_observations < spec.minimum_clean_observations:
        raise LeaseDrillError(
            "insufficient_clean_observations",
            "The GPU lease runtime cannot satisfy the drill observation minimum",
            details={
                "configured": config.clean_observations,
                "minimum": spec.minimum_clean_observations,
            },
        )


def _validate_runtime_paths_separated(
    spec: LeaseDrillSpec, config: RuntimeConfig
) -> None:
    work = spec.work_root
    roots = {
        "run root": config.run_root,
        "promotion root": config.promotion_root,
    }
    for role, root in roots.items():
        if work == root or _strictly_within(root, work):
            raise LeaseDrillError(
                "unsafe_path",
                f"mutable drill work root must not own the production {role}",
            )
    protected_files = {
        "lease state": config.lease_state_path,
        "lease lock": config.lock_path,
        "event log": config.event_log_path,
        "trainer checkpoint": config.trainer_checkpoint_path,
        "runtime config": spec.gpu_lease_config.path,
        "drill specification": spec.source_path,
    }
    if spec.trainer_source is not None:
        protected_files["trainer identity source"] = spec.trainer_source.path
    for role, protected in protected_files.items():
        if protected == work or _strictly_within(protected, work):
            raise LeaseDrillError(
                "unsafe_path",
                f"production {role} must not be inside mutable drill work root",
            )
        if spec.evidence_output == protected or _same_existing_file(
            spec.evidence_output, protected
        ):
            raise LeaseDrillError(
                "unsafe_path", f"evidence output aliases production {role}"
            )


def _strict_identity_int(value: Any, role: str, *, positive: bool) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LeaseDrillError("invalid_trainer_identity", f"{role} is malformed")
    return value


def _optional_identity_int(value: Any, role: str, *, positive: bool) -> Optional[int]:
    if value is None:
        return None
    return _strict_identity_int(value, role, positive=positive)


def _optional_identity_string(
    value: Any, role: str, *, sha256: bool = False
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise LeaseDrillError("invalid_trainer_identity", f"{role} is malformed")
    if sha256 and not _is_sha256(value):
        raise LeaseDrillError("invalid_trainer_identity", f"{role} is malformed")
    return value


def _process_identity_from_value(value: Any, role: str) -> ProcessIdentity:
    if not isinstance(value, Mapping):
        raise LeaseDrillError("invalid_trainer_identity", f"{role} must be an object")
    keys = set(value)
    if keys == _SNAKE_IDENTITY_KEYS:
        identity = ProcessIdentity(
            pid=_strict_identity_int(value["pid"], f"{role}.pid", positive=True),
            start_time_ticks=_optional_identity_int(
                value["start_time_ticks"],
                f"{role}.start_time_ticks",
                positive=False,
            ),
            process_group_id=_optional_identity_int(
                value["process_group_id"],
                f"{role}.process_group_id",
                positive=True,
            ),
            boot_id=_optional_identity_string(value["boot_id"], f"{role}.boot_id"),
            command_sha256=_optional_identity_string(
                value["command_sha256"],
                f"{role}.command_sha256",
                sha256=True,
            ),
            cgroup=_optional_identity_string(value["cgroup"], f"{role}.cgroup"),
        )
    elif keys == _CAMEL_IDENTITY_KEYS:
        identity = ProcessIdentity(
            pid=_strict_identity_int(value["pid"], f"{role}.pid", positive=True),
            start_time_ticks=_optional_identity_int(
                value["startTimeTicks"],
                f"{role}.startTimeTicks",
                positive=False,
            ),
            process_group_id=_optional_identity_int(
                value["processGroupId"],
                f"{role}.processGroupId",
                positive=True,
            ),
            boot_id=_optional_identity_string(value["bootId"], f"{role}.bootId"),
            command_sha256=_optional_identity_string(
                value["commandSha256"],
                f"{role}.commandSha256",
                sha256=True,
            ),
            cgroup=_optional_identity_string(value["cgroup"], f"{role}.cgroup"),
        )
    else:
        raise LeaseDrillError(
            "invalid_trainer_identity",
            f"{role} fields differ from ProcessIdentity",
        )
    if not identity.is_verifiable:
        raise LeaseDrillError(
            "invalid_trainer_identity",
            f"{role} cannot distinguish PID reuse",
        )
    return identity


def _load_trainer_identity(spec: LeaseDrillSpec) -> ProcessIdentity:
    if spec.trainer_source is None:
        raise LeaseDrillError(
            "invalid_trainer_identity",
            "launch trainer source has no pre-existing identity",
        )
    spec.trainer_source.verify("trainer identity source")
    value = _load_canonical_object(spec.trainer_source.path, "trainer identity source")
    if spec.trainer_source_kind == "process-identity":
        raw_identity: Any = value.get("process_identity", value)
    elif spec.trainer_source_kind == "supervisor-consumer-state":
        identities = value.get("identities")
        if not isinstance(identities, Mapping):
            raise LeaseDrillError(
                "invalid_trainer_identity",
                "Supervisor consumer state has no identities object",
            )
        trainers = identities.get("trainer")
        if not isinstance(trainers, list) or len(trainers) != 1:
            raise LeaseDrillError(
                "invalid_trainer_identity",
                "Supervisor consumer state must identify exactly one trainer",
            )
        raw_identity = trainers[0]
    else:  # Defensive against manually-created dataclass instances.
        raise LeaseDrillError(
            "invalid_drill_spec", "Trainer identity source kind is unsupported"
        )
    identity = _process_identity_from_value(raw_identity, "trainer identity")
    spec.trainer_source.verify("trainer identity source")
    return identity


def _launch_trainer(
    spec: LeaseDrillSpec,
    config: RuntimeConfig,
    manager: GpuLeaseManager,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> ProcessIdentity:
    argv = manager._expand(  # The production manager owns placeholder semantics.
        config.trainer_launch_command,
        record=None,
        identity=None,
    )
    identity = manager.runner.spawn(argv, new_process_group=True)
    deadline = clock() + config.trainer_start_timeout_seconds
    while not manager.runner.is_running(identity):
        if clock() >= deadline:
            raise LeaseDrillError(
                "trainer_start_failed",
                "Bound trainer launch command did not become live",
            )
        sleep(config.poll_interval_seconds)
    if not identity.is_verifiable:
        raise LeaseDrillError(
            "trainer_start_failed",
            "Launched trainer identity cannot distinguish PID reuse",
        )
    return identity


def _initial_checks(spec: LeaseDrillSpec) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "gpu_uuid": spec.expected_gpu_uuid,
        "lease_schema_version": GPU_LEASE_SCHEMA_VERSION,
        "trainer_drained": False,
        "checkpoint_handoff_verified": False,
        "evaluator_exclusive": False,
        "process_overlap_observed": False,
        "trainer_restored": False,
        "lease_clean_observations": 0,
        "release_clean_observations": 0,
        "safety_halt": False,
    }
    if set(checks) != LEASE_CHECK_FIELDS:
        raise AssertionError("lease drill checks do not match validator contract")
    return checks


class AutonomyLeaseDrill:
    """Execute one trainer/evaluator lease drill with injectable I/O."""

    def __init__(
        self,
        spec: LeaseDrillSpec,
        *,
        process_runner: Optional[ProcessRunner] = None,
        gpu_probe: Optional[Callable[[], GpuObservation]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        manager_factory: Callable[..., GpuLeaseManager] = GpuLeaseManager,
        evidence_publisher: Callable[..., Mapping[str, Any]] = publish_gate_evidence,
    ) -> None:
        self.spec = spec
        self.process_runner = process_runner
        self.gpu_probe = gpu_probe
        self.clock = clock
        self.sleep = sleep
        self.manager_factory = manager_factory
        self.evidence_publisher = evidence_publisher
        self._manager: Optional[GpuLeaseManager] = None
        self._lease_record: Optional[LeaseRecord] = None
        self._evaluator_identities: Tuple[ProcessIdentity, ...] = ()
        self._publication_safe = False

    def run(self) -> Mapping[str, Any]:
        checks = _initial_checks(self.spec)
        try:
            evidence = self._run_checked(checks)
        except BaseException as exc:
            error = _coerce_error(exc)
            self._refresh_final_checks(checks)
            if not self._publication_safe:
                raise error from exc
            try:
                self._publish(checks, decision="FAIL")
            except BaseException as publish_exc:
                raise LeaseDrillError(
                    "failure_evidence_publish_failed",
                    "The drill failed and FAIL evidence could not be published",
                    details={
                        "drill_error": error.to_dict()["error"],
                        "publish_error": _error_text(publish_exc),
                    },
                ) from error
            raise error from exc
        return evidence

    def _run_checked(self, checks: Dict[str, Any]) -> Mapping[str, Any]:
        self.spec.verify_immutable_inputs(include_trainer_source=True)
        config = _load_runtime_config(self.spec)
        _validate_runtime_paths_separated(self.spec, config)
        self._ensure_work_root()
        self._publication_safe = True
        _validate_runtime_contract(self.spec, config)

        kwargs: Dict[str, Any] = {
            "clock": self.clock,
            "sleep": self.sleep,
        }
        if self.process_runner is not None:
            kwargs["process_runner"] = self.process_runner
        if self.gpu_probe is not None:
            kwargs["gpu_probe"] = self.gpu_probe
        manager = self.manager_factory(config, **kwargs)
        self._manager = manager
        trainer = (
            _launch_trainer(
                self.spec,
                config,
                manager,
                clock=self.clock,
                sleep=self.sleep,
            )
            if self.spec.trainer_source_kind == "launch"
            else _load_trainer_identity(self.spec)
        )

        primary_error: Optional[BaseException] = None
        try:
            with manager.evaluator_lease(trainer) as lease:
                self._lease_record = lease
                self._evaluator_identities = tuple(lease.evaluators)
                self._validate_handoff(
                    config=config,
                    trainer=trainer,
                    record=lease,
                    checks=checks,
                )
                self._check_process_overlap(trainer, lease.evaluators, checks)
                self._run_probe(manager)
                self._check_process_overlap(trainer, lease.evaluators, checks)
                self._prove_evaluator_exclusive(
                    manager=manager,
                    trainer=trainer,
                    record=lease,
                    checks=checks,
                )
        except BaseException as exc:
            primary_error = exc

        reconcile_error: Optional[BaseException] = None
        try:
            report = manager.reconcile(mutate=True)
            if report.safety_halt:
                checks["safety_halt"] = True
        except BaseException as exc:
            reconcile_error = exc
        self._refresh_final_checks(checks)

        if primary_error is not None:
            raise primary_error
        if reconcile_error is not None:
            raise reconcile_error
        self._require_success(checks)
        self.spec.verify_immutable_inputs(include_trainer_source=False)
        return self._publish(checks, decision="PASS")

    def _validate_handoff(
        self,
        *,
        config: RuntimeConfig,
        trainer: ProcessIdentity,
        record: LeaseRecord,
        checks: Dict[str, Any],
    ) -> None:
        if (
            record.phase != "evaluating"
            or record.expected_gpu_uuid != self.spec.expected_gpu_uuid
        ):
            raise LeaseDrillError(
                "invalid_lease_handoff",
                "Evaluator lease did not enter the expected GPU handoff phase",
            )
        if manager_runner(self._manager).is_running(trainer):
            if record.evaluators:
                checks["process_overlap_observed"] = True
            raise LeaseDrillError(
                "trainer_not_drained",
                "Trainer remained live after evaluator lease acquisition",
            )
        checks["trainer_drained"] = True

        before = record.pre_drain_checkpoint
        handoff = record.handoff_checkpoint
        checkpoint_valid = (
            before is not None
            and handoff is not None
            and record.checkpoint_sha256 == handoff.sha256
            and record.checkpoint_size == handoff.size
            and (
                not config.require_checkpoint_change
                or handoff.content_changed_from(before)
            )
        )
        if not checkpoint_valid:
            raise LeaseDrillError(
                "checkpoint_handoff_mismatch",
                "Lease record lacks a valid trainer checkpoint handoff",
            )
        checks["checkpoint_handoff_verified"] = True

        checks["lease_clean_observations"] = record.lease_clean_observation_count
        if record.lease_clean_observation_count < self.spec.minimum_clean_observations:
            raise LeaseDrillError(
                "insufficient_clean_observations",
                "Lease acquisition recorded too few clean GPU observations",
            )
        if (
            len(record.evaluators) != config.evaluator_process_count
            or not record.evaluators
            or any(
                not identity.is_verifiable
                or not manager_runner(self._manager).is_running(identity)
                for identity in record.evaluators
            )
        ):
            raise LeaseDrillError(
                "evaluator_not_running",
                "Configured evaluator identities are not all live and verifiable",
            )

    def _run_probe(self, manager: GpuLeaseManager) -> None:
        try:
            result = manager.runner.run(
                self.spec.evaluator_probe_argv,
                timeout=self.spec.evaluator_probe_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise LeaseDrillError(
                "evaluator_probe_timeout",
                "Evaluator probe exceeded its configured deadline",
            ) from exc
        except OSError as exc:
            raise LeaseDrillError(
                "evaluator_probe_failed",
                f"Evaluator probe could not execute: {exc}",
            ) from exc
        returncode = getattr(result, "returncode", None)
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe returned no valid process return code",
            )
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe output was not text",
            )
        if len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")) > (
            MAX_PROBE_OUTPUT_BYTES
        ):
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe output exceeded the evidence safety limit",
            )
        if returncode != 0:
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe command failed",
                details={
                    "returncode": returncode,
                    "stderr": stderr[-4096:],
                },
            )
        try:
            encoded = stdout.encode("utf-8")
            receipt = json.loads(
                stdout,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError, LeaseDrillError) as exc:
            raise LeaseDrillError(
                "evaluator_probe_failed",
                f"Evaluator probe receipt is invalid: {exc}",
            ) from exc
        if (
            not isinstance(receipt, Mapping)
            or encoded != (_canonical_json(receipt) + "\n").encode("utf-8")
        ):
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe must emit one canonical newline-terminated receipt",
            )
        expected_fields = {
            "schema_version",
            "contract",
            "gpu_uuid",
            "evaluator_pids",
            "model_sha256",
            "config_sha256",
            "completed_work_count",
            "receipt_sha256",
        }
        if set(receipt) != expected_fields:
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe receipt fields differ from the schema",
            )
        payload = dict(receipt)
        supplied_hash = payload.pop("receipt_sha256", None)
        evaluator_pids = sorted(
            identity.pid for identity in self._evaluator_identities
        )
        completed = receipt.get("completed_work_count")
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("contract") != PROBE_RECEIPT_CONTRACT
            or supplied_hash != _canonical_sha256(payload)
            or receipt.get("gpu_uuid") != self.spec.expected_gpu_uuid
            or receipt.get("evaluator_pids") != evaluator_pids
            or receipt.get("model_sha256")
            != self.spec.evaluator_probe_model_sha256
            or receipt.get("config_sha256")
            != self.spec.evaluator_probe_config_sha256
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < self.spec.evaluator_probe_minimum_completed_work
        ):
            raise LeaseDrillError(
                "evaluator_probe_failed",
                "Evaluator probe receipt does not prove bound leased work",
            )

    def _check_process_overlap(
        self,
        trainer: ProcessIdentity,
        evaluators: Sequence[ProcessIdentity],
        checks: Dict[str, Any],
    ) -> None:
        runner = manager_runner(self._manager)
        trainer_live = runner.is_running(trainer)
        evaluator_live = any(runner.is_running(identity) for identity in evaluators)
        if trainer_live and evaluator_live:
            checks["process_overlap_observed"] = True
            raise LeaseDrillError(
                "process_overlap",
                "Trainer and evaluator process identities overlapped",
            )

    def _prove_evaluator_exclusive(
        self,
        *,
        manager: GpuLeaseManager,
        trainer: ProcessIdentity,
        record: LeaseRecord,
        checks: Dict[str, Any],
    ) -> None:
        try:
            observation = manager.gpu_probe()
        except GpuLeaseError:
            raise
        except BaseException as exc:
            raise LeaseDrillError(
                "gpu_probe_failed",
                f"Evaluator exclusivity probe failed: {exc}",
            ) from exc
        if observation.gpu_uuid != self.spec.expected_gpu_uuid:
            raise LeaseDrillError(
                "gpu_uuid_mismatch",
                "Evaluator observation came from the wrong GPU UUID",
            )
        observed_pids = {process.pid for process in observation.processes}
        expected_pids = {identity.pid for identity in record.evaluators}
        if trainer.pid in observed_pids:
            checks["process_overlap_observed"] = True
        if observed_pids != expected_pids:
            raise LeaseDrillError(
                "evaluator_not_exclusive",
                "GPU7 processes do not exactly match the leased evaluator identities",
                details={
                    "expected_pids": sorted(expected_pids),
                    "observed_pids": sorted(observed_pids),
                },
            )
        self._check_process_overlap(trainer, record.evaluators, checks)
        checks["evaluator_exclusive"] = True

    def _refresh_final_checks(self, checks: Dict[str, Any]) -> None:
        manager = self._manager
        if manager is None:
            return
        try:
            final = manager.read_record()
        except BaseException:
            return
        if final is None:
            return
        checks["release_clean_observations"] = final.release_clean_observation_count
        checks["safety_halt"] = bool(final.safety_halt)
        runner = manager.runner
        restored = final.restored_trainer or final.trainer
        evaluator_live = any(
            runner.is_running(identity) for identity in self._evaluator_identities
        )
        trainer_live = restored is not None and runner.is_running(restored)
        if trainer_live and evaluator_live:
            checks["process_overlap_observed"] = True
        checks["trainer_restored"] = bool(
            checks["trainer_drained"]
            and final.phase == "trainer_running"
            and final.restoration_status == "restored"
            and not final.safety_halt
            and not final.evaluators
            and not evaluator_live
            and trainer_live
        )

    def _require_success(self, checks: Mapping[str, Any]) -> None:
        minimum = self.spec.minimum_clean_observations
        if (
            set(checks) != LEASE_CHECK_FIELDS
            or checks["gpu_uuid"] != self.spec.expected_gpu_uuid
            or checks["lease_schema_version"] != GPU_LEASE_SCHEMA_VERSION
            or checks["trainer_drained"] is not True
            or checks["checkpoint_handoff_verified"] is not True
            or checks["evaluator_exclusive"] is not True
            or checks["process_overlap_observed"] is not False
            or checks["trainer_restored"] is not True
            or checks["safety_halt"] is not False
            or type(checks["lease_clean_observations"]) is not int
            or checks["lease_clean_observations"] < minimum
            or type(checks["release_clean_observations"]) is not int
            or checks["release_clean_observations"] < minimum
        ):
            raise LeaseDrillError(
                "lease_drill_failed",
                "Trainer/evaluator lease drill did not satisfy every PASS invariant",
            )

    def _ensure_work_root(self) -> None:
        self.spec.work_root.mkdir(mode=0o700, exist_ok=True)
        _normalized_absolute(
            self.spec.work_root,
            "lease drill work root",
            require_directory=True,
        )
        _reject_symlink_ancestors(
            self.spec.evidence_output, "lease drill evidence output"
        )

    def _publish(
        self, checks: Mapping[str, Any], *, decision: str
    ) -> Mapping[str, Any]:
        if set(checks) != LEASE_CHECK_FIELDS:
            raise LeaseDrillError(
                "invalid_evidence",
                "Lease drill attempted to publish fields outside the "
                "validator contract",
            )
        self._ensure_work_root()
        return self.evidence_publisher(
            self.spec.evidence_output,
            GATE_ID,
            dict(checks),
            decision=decision,
        )


def manager_runner(manager: Optional[GpuLeaseManager]) -> ProcessRunner:
    if manager is None:
        raise LeaseDrillError("internal_error", "GPU lease manager is not initialized")
    return manager.runner


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _coerce_error(exc: BaseException) -> LeaseDrillError:
    if isinstance(exc, LeaseDrillError):
        return exc
    if isinstance(exc, GpuLeaseError):
        return LeaseDrillError(
            "gpu_lease_failed",
            str(exc),
            details={
                "gpu_lease_code": exc.code,
                "gpu_lease_details": exc.details,
            },
        )
    if isinstance(exc, KeyboardInterrupt):
        return LeaseDrillError("interrupted", "Lease drill was interrupted")
    return LeaseDrillError(
        "unexpected_error",
        str(exc) or type(exc).__name__,
        details={"type": type(exc).__name__},
    )


def run_lease_drill(
    spec: Any,
    *,
    expected_spec_sha256: Optional[str] = None,
    process_runner: Optional[ProcessRunner] = None,
    gpu_probe: Optional[Callable[[], GpuObservation]] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    manager_factory: Callable[..., GpuLeaseManager] = GpuLeaseManager,
    evidence_publisher: Callable[..., Mapping[str, Any]] = publish_gate_evidence,
) -> Mapping[str, Any]:
    """Load, execute, and publish one lease drill."""

    if isinstance(spec, LeaseDrillSpec):
        loaded = spec
        if expected_spec_sha256 is not None:
            expected = _require_sha256(
                expected_spec_sha256,
                "expected lease drill specification hash",
            )
            if expected not in {loaded.identity, loaded.file_sha256}:
                raise LeaseDrillError(
                    "invalid_drill_spec",
                    "Loaded lease drill specification hash is not expected",
                )
    else:
        loaded = load_drill_spec(Path(spec), expected_spec_sha256=expected_spec_sha256)
    return AutonomyLeaseDrill(
        loaded,
        process_runner=process_runner,
        gpu_probe=gpu_probe,
        clock=clock,
        sleep=sleep,
        manager_factory=manager_factory,
        evidence_publisher=evidence_publisher,
    ).run()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256")
    parser.add_argument("--evidence", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        spec = load_drill_spec(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
        )
        if args.evidence is not None:
            supplied_evidence = _normalized_absolute(
                args.evidence, "CLI evidence output"
            )
            if supplied_evidence != spec.evidence_output:
                raise LeaseDrillError(
                    "evidence_path_mismatch",
                    "CLI evidence output differs from the hash-bound drill spec",
                )
        evidence = AutonomyLeaseDrill(spec).run()
        sys.stdout.write(_canonical_json(evidence) + "\n")
        return 0
    except LeaseDrillError as exc:
        sys.stderr.write(_canonical_json(exc.to_dict()) + "\n")
        return 2
    except (GpuLeaseError, OSError) as exc:
        error = _coerce_error(exc)
        sys.stderr.write(_canonical_json(error.to_dict()) + "\n")
        return 2


# Descriptive and concise aliases for callers.
TrainerEvaluatorLeaseDrill = AutonomyLeaseDrill
load_spec = load_drill_spec
run_drill = run_lease_drill


__all__ = [
    "AutonomyLeaseDrill",
    "ERROR_CONTRACT",
    "FileBinding",
    "GATE_ID",
    "GPU_INDEX",
    "LEASE_CHECK_FIELDS",
    "PROBE_RECEIPT_CONTRACT",
    "LeaseDrillError",
    "LeaseDrillSpec",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "TrainerEvaluatorLeaseDrill",
    "load_drill_spec",
    "load_spec",
    "main",
    "parse_args",
    "run_drill",
    "run_lease_drill",
]


if __name__ == "__main__":
    raise SystemExit(main())
