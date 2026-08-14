"""Run one hash-bound KataGo analysis while an evaluator GPU lease is held.

The probe is intentionally small and fail-closed.  Its canonical, self-hashed
specification binds every executable and input used by the probe. It verifies
one persistent CUDA sentinel, continuously inventories ownership while one
known KataGo child runs, and proves that only the sentinel remains afterward.

Successful CLI execution writes exactly one canonical receipt to stdout.
Failures write a canonical error to stderr and never write a partial receipt.
Only Python's standard library is used.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

SCHEMA_VERSION = 1
SPEC_CONTRACT = "risk-score-autonomy-lease-probe-spec-v1"
PROBE_RECEIPT_CONTRACT = "risk-score-autonomy-lease-probe-receipt-v1"
RECEIPT_CONTRACT = PROBE_RECEIPT_CONTRACT
ERROR_CONTRACT = "risk-score-autonomy-lease-probe-error-v1"
WORKER_READY_CONTRACT = "risk-score-autonomy-lease-worker-ready-v1"

MAX_SPEC_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_QUERY_BYTES = 256 * 1024
MAX_BOUND_FILE_BYTES = 64 * 1024 * 1024 * 1024
MAX_GPU_STDOUT_BYTES = 1024 * 1024
MAX_GPU_STDERR_BYTES = 256 * 1024
MAX_ANALYSIS_STDOUT_BYTES = 4 * 1024 * 1024
MAX_ANALYSIS_STDERR_BYTES = 1024 * 1024
MAX_TIMEOUT_SECONDS = 600.0
OUTER_DRILL_TIMEOUT_RESERVE_SECONDS = 10.0
MAX_INTERNAL_TIMEOUT_SECONDS = MAX_TIMEOUT_SECONDS - OUTER_DRILL_TIMEOUT_RESERVE_SECONDS
MAX_QUERY_VISITS = 10_000_000
MAX_READY_BYTES = 64 * 1024
DEFAULT_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_CLEANUP_MARGIN_SECONDS = 5.0
POST_ANALYSIS_CLEAN_OBSERVATIONS = 2
_PYTHON_EXECUTABLE = Path(sys.executable).resolve()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_MIG_UUID_RE = re.compile(r"^MIG-[A-Za-z0-9_./-]+$")
_SPEC_KEYS = {
    "schema_version",
    "contract",
    "nvidia_smi",
    "katago",
    "model",
    "config",
    "query",
    "probe_module",
    "sentinel_module",
    "sentinel_readiness_path",
    "expected_gpu_uuid",
    "expected_gpu_index",
    "expected_evaluator_process_count",
    "timeout_seconds",
    "cleanup_margin_seconds",
    "poll_interval_seconds",
    "minimum_completed_work",
    "spec_sha256",
}
_BINDING_KEYS = {"path", "sha256"}
_ALLOWED_QUERY_KEYS = {
    "boardXSize",
    "boardYSize",
    "id",
    "includePolicy",
    "initialPlayer",
    "initialStones",
    "komi",
    "maxVisits",
    "moves",
    "overrideSettings",
    "rules",
}
_DETERMINISTIC_CONFIG = {
    "forDeterministicTesting": "true",
    "numAnalysisThreads": "1",
    "numSearchThreadsPerAnalysisThread": "1",
    "nnRandomize": "false",
    "rootNoiseEnabled": "false",
    "rootNumSymmetriesToSample": "1",
    "useUncertainty": "false",
    "cpuctUtilityStdevScale": "0",
    "reportAnalysisWinratesAs": "sidetomove",
}
_DETERMINISTIC_QUERY_OVERRIDES = {
    "forDeterministicTesting": True,
    "numAnalysisThreads": 1,
    "nnRandomize": False,
    "rootNoiseEnabled": False,
    "rootNumSymmetriesToSample": 1,
    "useUncertainty": False,
    "cpuctUtilityStdevScale": 0,
    "reportAnalysisWinratesAs": "sidetomove",
}
_ALLOWED_QUERY_OVERRIDES = frozenset(_DETERMINISTIC_QUERY_OVERRIDES)
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        *_DETERMINISTIC_CONFIG,
        "analysisPVLen",
        "cudaDeviceToUse",
        "logAllRequests",
        "logAllResponses",
        "logErrorsAndWarnings",
        "logSearchInfo",
        "logToStderr",
        "maxVisits",
        "nnCacheSizePowerOfTwo",
        "nnMaxBatchSize",
        "nnMutexPoolSizePowerOfTwo",
        "numNNServerThreadsPerModel",
        "warnUnusedFields",
    }
)


class AutonomyLeaseProbeError(RuntimeError):
    """A fail-closed probe error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": ERROR_CONTRACT,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


ProbeError = AutonomyLeaseProbeError
ProbeSpecError = AutonomyLeaseProbeError


def canonical_json(value: Any) -> str:
    """Return the compact, sorted JSON representation used by probe contracts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError("invalid_probe_spec", f"{role} must be an object")
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise ProbeError(
            "invalid_probe_spec",
            f"{role} fields differ from the contract",
            details={"missing": missing, "extra": extra},
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProbeError(
            "invalid_probe_spec",
            f"{role} must be a lowercase 64-character SHA-256",
        )
    return value


def _positive_int(value: Any, role: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ProbeError(
            "invalid_probe_spec",
            f"{role} must be an integer between 1 and {maximum}",
        )
    return value


def _nonnegative_int(value: Any, role: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ProbeError(
            "invalid_probe_spec",
            f"{role} must be an integer between 0 and {maximum}",
        )
    return value


def _positive_timeout(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= MAX_INTERNAL_TIMEOUT_SECONDS
    ):
        raise ProbeError(
            "invalid_probe_spec",
            "timeout_seconds must be positive and leave the fixed outer drill "
            f"reserve (maximum {MAX_INTERNAL_TIMEOUT_SECONDS:g})",
        )
    return float(value)


def _positive_duration(value: Any, role: str, *, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= maximum
    ):
        raise ProbeError(
            "invalid_probe_spec",
            f"{role} must be positive and no greater than {maximum:g}",
        )
    return float(value)


def _ensure_finite_json(value: Any, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProbeError("malformed_output", f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProbeError(
                    "malformed_output", f"{role} contains a non-string object key"
                )
            _ensure_finite_json(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{role}[{index}]")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeError(
                "invalid_json", f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProbeError("invalid_json", f"non-finite JSON constant is forbidden: {value}")


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        try:
            if current.is_symlink():
                raise ProbeError(
                    "unsafe_path",
                    f"{role} has a symlinked path component",
                    details={"path": os.fspath(current)},
                )
        except OSError as exc:
            raise ProbeError(
                "unsafe_path",
                f"could not inspect {role}: {exc}",
                details={"path": os.fspath(current)},
            ) from exc
        if current.parent == current:
            return
        current = current.parent


def _canonical_absolute_path(value: Any, role: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise ProbeError("unsafe_path", f"{role} must be an absolute path") from exc
    if (
        not isinstance(raw, str)
        or not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
    ):
        raise ProbeError("unsafe_path", f"{role} must be a nonempty absolute path")
    path = Path(raw)
    normalized = Path(os.path.abspath(raw))
    if not path.is_absolute() or path != normalized:
        raise ProbeError(
            "unsafe_path", f"{role} must be absolute and lexically normalized"
        )
    _reject_symlink_ancestors(path, role)
    try:
        if path.resolve(strict=False) != path:
            raise ProbeError(
                "unsafe_path", f"{role} must contain no symlink components"
            )
    except OSError as exc:
        raise ProbeError("unsafe_path", f"could not resolve {role}: {exc}") from exc
    return path


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _open_regular_file(
    path: Path,
    role: str,
    *,
    maximum_bytes: int,
    require_single_link: bool = False,
) -> tuple[int, os.stat_result]:
    source = _canonical_absolute_path(path, role)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(source), flags)
    except OSError as exc:
        raise ProbeError(
            "unsafe_path",
            f"{role} must be an existing regular non-symlink file: {exc}",
            details={"path": os.fspath(source)},
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProbeError("unsafe_path", f"{role} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ProbeError(
                "input_too_large",
                f"{role} exceeds its {maximum_bytes}-byte safety limit",
            )
        if require_single_link and metadata.st_nlink != 1:
            raise ProbeError(
                "unsafe_path", f"{role} must have exactly one filesystem link"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, metadata


def _verify_stable_path(
    path: Path,
    role: str,
    before: os.stat_result,
    after: os.stat_result,
) -> None:
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ProbeError(
            "bound_input_changed", f"{role} disappeared while it was read"
        ) from exc
    if _metadata_identity(before) != _metadata_identity(after) or _metadata_identity(
        after
    ) != _metadata_identity(path_after):
        raise ProbeError("bound_input_changed", f"{role} changed while it was read")
    _reject_symlink_ancestors(path, role)


def _stable_file_sha256(
    path: Path,
    role: str,
    *,
    maximum_bytes: int = MAX_BOUND_FILE_BYTES,
    require_single_link: bool = False,
) -> str:
    source = _canonical_absolute_path(path, role)
    descriptor, before = _open_regular_file(
        source,
        role,
        maximum_bytes=maximum_bytes,
        require_single_link=require_single_link,
    )
    digest = hashlib.sha256()
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _verify_stable_path(source, role, before, after)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return _stable_file_sha256(Path(path), f"file {path}")


def _stable_read(
    path: Path,
    role: str,
    *,
    maximum_bytes: int,
    require_single_link: bool = False,
) -> bytes:
    source = _canonical_absolute_path(path, role)
    descriptor, before = _open_regular_file(
        source,
        role,
        maximum_bytes=maximum_bytes,
        require_single_link=require_single_link,
    )
    chunks = []
    remaining = before.st_size
    try:
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        if remaining:
            raise ProbeError(
                "bound_input_changed", f"{role} was truncated while it was read"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _verify_stable_path(source, role, before, after)
    return b"".join(chunks)


def _load_canonical_object(
    path: Path,
    role: str,
    *,
    maximum_bytes: int,
    require_single_link: bool = False,
) -> tuple[dict[str, Any], bytes]:
    data = _stable_read(
        path,
        role,
        maximum_bytes=maximum_bytes,
        require_single_link=require_single_link,
    )
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ProbeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProbeError("invalid_json", f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError("invalid_json", f"{role} root must be an object")
    _ensure_finite_json(value, role)
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ProbeError(
            "noncanonical_json",
            f"{role} must be canonical newline-terminated JSON",
        )
    return value, data


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str
    executable: bool = False

    def verify(self, role: str) -> None:
        source = _canonical_absolute_path(self.path, role)
        try:
            metadata = source.lstat()
        except OSError as exc:
            raise ProbeError(
                "bound_input_changed", f"{role} is missing: {exc}"
            ) from exc
        if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ProbeError(
                "unsafe_path", f"{role} must be a regular non-symlink file"
            )
        if self.executable and not os.access(source, os.X_OK):
            raise ProbeError("unsafe_path", f"{role} must be executable")
        observed = _stable_file_sha256(source, role)
        if observed != self.sha256:
            raise ProbeError(
                "bound_input_changed",
                f"{role} does not match its bound SHA-256",
                details={
                    "path": os.fspath(source),
                    "expected_sha256": self.sha256,
                    "observed_sha256": observed,
                },
            )

    def spec_value(self) -> Mapping[str, str]:
        return {"path": os.fspath(self.path), "sha256": self.sha256}


def _binding_from_value(value: Any, role: str, *, executable: bool) -> FileBinding:
    raw = _exact_keys(value, _BINDING_KEYS, role)
    path = _canonical_absolute_path(raw["path"], f"{role}.path")
    digest = _require_sha256(raw["sha256"], f"{role}.sha256")
    binding = FileBinding(path=path, sha256=digest, executable=executable)
    binding.verify(role)
    return binding


def _binding_for_path(path: Path, role: str, *, executable: bool) -> FileBinding:
    source = _canonical_absolute_path(path, role)
    binding = FileBinding(
        path=source,
        sha256=_stable_file_sha256(source, role),
        executable=executable,
    )
    binding.verify(role)
    return binding


def _parse_config(data: bytes) -> Mapping[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProbeError(
            "invalid_probe_spec", f"analysis config is not UTF-8: {exc}"
        ) from exc
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ProbeError(
                "invalid_probe_spec",
                "analysis config contains a non-assignment directive",
                details={"line": line_number},
            )
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if not key or key in values:
            raise ProbeError(
                "invalid_probe_spec",
                "analysis config contains an empty or duplicate key",
                details={"line": line_number, "key": key},
            )
        values[key] = raw_value.lower()
    unsupported = sorted(set(values).difference(_ALLOWED_CONFIG_KEYS))
    if unsupported:
        raise ProbeError(
            "invalid_probe_spec",
            "analysis config contains settings outside the deterministic allowlist",
            details={"settings": unsupported},
        )
    conflicts = {
        key: {"expected": expected, "observed": values.get(key)}
        for key, expected in _DETERMINISTIC_CONFIG.items()
        if values.get(key) != expected
    }
    if conflicts:
        raise ProbeError(
            "invalid_probe_spec",
            "analysis config is not deterministic and perspective-fixed",
            details={"conflicts": conflicts},
        )
    if values.get("cudaDeviceToUse") not in {None, "0"}:
        raise ProbeError(
            "invalid_probe_spec",
            "analysis config cudaDeviceToUse must be absent or zero",
        )
    return values


def _validated_query(data: bytes) -> tuple[Mapping[str, Any], str]:
    if not data or not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise ProbeError(
            "invalid_probe_spec",
            "query must contain exactly one newline-terminated JSON record",
        )
    try:
        value = json.loads(
            data[:-1].decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ProbeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProbeError("invalid_probe_spec", f"query is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError("invalid_probe_spec", "query record must be an object")
    _ensure_finite_json(value, "query")
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ProbeError(
            "invalid_probe_spec", "query must be canonical newline-terminated JSON"
        )
    query_id = value.get("id")
    if (
        not isinstance(query_id, str)
        or not query_id
        or len(query_id.encode("utf-8")) > 4096
        or "\x00" in query_id
        or "\n" in query_id
        or "\r" in query_id
    ):
        raise ProbeError(
            "invalid_probe_spec", "query id must be a safe nonempty string"
        )
    visits = value.get("maxVisits")
    if type(visits) is not int or not 1 <= visits <= MAX_QUERY_VISITS:
        raise ProbeError(
            "invalid_probe_spec",
            f"query maxVisits must be between 1 and {MAX_QUERY_VISITS}",
        )
    unsupported = sorted(set(value).difference(_ALLOWED_QUERY_KEYS))
    if unsupported:
        raise ProbeError(
            "invalid_probe_spec",
            "query contains multi-turn, streaming, priority, or unsupported fields",
            details={"fields": unsupported},
        )
    overrides = value.get("overrideSettings")
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ProbeError(
                "invalid_probe_spec", "query overrideSettings must be an object"
            )
        unsupported_overrides = sorted(
            set(overrides).difference(_ALLOWED_QUERY_OVERRIDES)
        )
        if unsupported_overrides:
            raise ProbeError(
                "invalid_probe_spec",
                "query overrideSettings contains a setting outside the "
                "deterministic allowlist",
                details={"settings": unsupported_overrides},
            )
        conflicts = {}
        for key, expected in _DETERMINISTIC_QUERY_OVERRIDES.items():
            if key not in overrides:
                continue
            observed = overrides[key]
            if isinstance(expected, str):
                matches = isinstance(observed, str) and observed.lower() == expected
            elif isinstance(expected, bool):
                matches = type(observed) is bool and observed is expected
            else:
                matches = (
                    not isinstance(observed, bool)
                    and isinstance(observed, (int, float))
                    and float(observed) == float(expected)
                )
            if not matches:
                conflicts[key] = {"expected": expected, "observed": observed}
        if conflicts:
            raise ProbeError(
                "invalid_probe_spec",
                "query overrides contradict deterministic analysis settings",
                details={"conflicts": conflicts},
            )
    for key in ("boardXSize", "boardYSize"):
        dimension = value.get(key)
        if type(dimension) is not int or not 2 <= dimension <= 25:
            raise ProbeError(
                "invalid_probe_spec", f"query {key} must be an integer from 2 to 25"
            )
    if not isinstance(value.get("moves"), list):
        raise ProbeError("invalid_probe_spec", "query moves must be an array")
    if "initialStones" in value and not isinstance(value["initialStones"], list):
        raise ProbeError("invalid_probe_spec", "query initialStones must be an array")
    for role in ("moves", "initialStones"):
        for index, move in enumerate(value.get(role, [])):
            if (
                not isinstance(move, list)
                or len(move) != 2
                or move[0] not in {"B", "W"}
                or not isinstance(move[1], str)
                or not move[1]
                or len(move[1]) > 16
                or any(character in move[1] for character in "\x00\r\n")
            ):
                raise ProbeError(
                    "invalid_probe_spec",
                    f"query {role}[{index}] is not a fixed player/location pair",
                )
    if "initialPlayer" in value and value["initialPlayer"] not in {"B", "W"}:
        raise ProbeError("invalid_probe_spec", "query initialPlayer must be B or W")
    if "includePolicy" in value and type(value["includePolicy"]) is not bool:
        raise ProbeError("invalid_probe_spec", "query includePolicy must be boolean")
    rules = value.get("rules")
    if (
        not isinstance(rules, str)
        or not rules
        or len(rules.encode()) > 4096
        or any(character in rules for character in "\x00\r\n")
    ):
        raise ProbeError(
            "invalid_probe_spec", "query rules must be a fixed safe string"
        )
    komi = value.get("komi")
    if (
        isinstance(komi, bool)
        or not isinstance(komi, (int, float))
        or not math.isfinite(float(komi))
    ):
        raise ProbeError("invalid_probe_spec", "query komi must be finite")
    return value, query_id


@dataclass(frozen=True)
class LinuxProcessIdentity:
    pid: int
    start_time_ticks: int
    process_group_id: int
    boot_id: str
    command_sha256: str
    cgroup: str

    @classmethod
    def capture(cls, pid: int) -> LinuxProcessIdentity | None:
        if sys.platform != "linux":
            raise ProbeError(
                "unsupported_platform",
                "the lease probe requires Linux /proc process identity",
            )
        proc = Path("/proc") / str(pid)
        try:
            stat_text = (proc / "stat").read_text()
            closing = stat_text.rfind(")")
            if closing < 0:
                raise ValueError("missing process-name terminator")
            tail = stat_text[closing + 2 :].split()
            if len(tail) < 20:
                raise ValueError("truncated stat record")
            command = (proc / "cmdline").read_bytes()
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            cgroup = (proc / "cgroup").read_text().strip()
            process_group_id = os.getpgid(pid)
        except FileNotFoundError:
            return None
        except ProcessLookupError:
            return None
        except (OSError, ValueError) as exc:
            raise ProbeError(
                "process_identity_unavailable",
                f"could not capture process identity for PID {pid}: {exc}",
            ) from exc
        return cls(
            pid=pid,
            start_time_ticks=int(tail[19]),
            process_group_id=process_group_id,
            boot_id=boot_id,
            command_sha256=hashlib.sha256(command).hexdigest(),
            cgroup=cgroup,
        )

    @classmethod
    def from_value(cls, value: Any, *, expected_pid: int) -> LinuxProcessIdentity:
        expected_keys = {
            "pid",
            "start_time_ticks",
            "process_group_id",
            "boot_id",
            "command_sha256",
            "cgroup",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ProbeError(
                "sentinel_readiness_invalid",
                "sentinel readiness process identity is malformed",
            )
        pid = value["pid"]
        start = value["start_time_ticks"]
        pgid = value["process_group_id"]
        boot_id = value["boot_id"]
        command_hash = value["command_sha256"]
        cgroup = value["cgroup"]
        if (
            type(pid) is not int
            or pid != expected_pid
            or type(start) is not int
            or start <= 0
            or type(pgid) is not int
            or pgid <= 0
            or not isinstance(boot_id, str)
            or not boot_id
            or not isinstance(cgroup, str)
            or not cgroup
        ):
            raise ProbeError(
                "sentinel_readiness_invalid",
                "sentinel readiness process identity fields are invalid",
            )
        return cls(
            pid=pid,
            start_time_ticks=start,
            process_group_id=pgid,
            boot_id=boot_id,
            command_sha256=_require_sha256(command_hash, "sentinel command SHA-256"),
            cgroup=cgroup,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "process_group_id": self.process_group_id,
            "boot_id": self.boot_id,
            "command_sha256": self.command_sha256,
            "cgroup": self.cgroup,
        }


class ProcessIdentityProbe(Protocol):
    def capture(self, pid: int) -> LinuxProcessIdentity | None: ...

    def descendants(self, pid: int) -> tuple[int, ...]: ...


class LinuxProcessIdentityProbe:
    def capture(self, pid: int) -> LinuxProcessIdentity | None:
        return LinuxProcessIdentity.capture(pid)

    def descendants(self, pid: int) -> tuple[int, ...]:
        if sys.platform != "linux":
            raise ProbeError(
                "unsupported_platform",
                "the lease probe requires Linux /proc descendant inspection",
            )
        pending = [pid]
        discovered = set()
        while pending:
            parent = pending.pop()
            children_path = (
                Path("/proc") / str(parent) / "task" / str(parent) / "children"
            )
            try:
                content = children_path.read_text().strip()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ProbeError(
                    "process_identity_unavailable",
                    f"could not inspect descendants of PID {parent}: {exc}",
                ) from exc
            for raw_pid in content.split():
                try:
                    child = int(raw_pid)
                except ValueError as exc:
                    raise ProbeError(
                        "process_identity_unavailable",
                        "Linux child-process record is malformed",
                    ) from exc
                if child > 0 and child not in discovered:
                    discovered.add(child)
                    pending.append(child)
        return tuple(sorted(discovered))


@dataclass(frozen=True)
class ProbeSpec(Mapping[str, Any]):
    path: Path
    file_sha256: str
    spec_sha256: str
    nvidia_smi: FileBinding
    katago: FileBinding
    model: FileBinding
    config: FileBinding
    query: FileBinding
    probe_module: FileBinding
    sentinel_module: FileBinding
    sentinel_readiness_path: Path
    expected_gpu_uuid: str
    expected_gpu_index: int
    expected_evaluator_process_count: int
    timeout_seconds: float
    cleanup_margin_seconds: float
    poll_interval_seconds: float
    minimum_completed_work: int
    raw: Mapping[str, Any]

    @property
    def identity(self) -> str:
        return self.spec_sha256

    @property
    def source_path(self) -> Path:
        return self.path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def verify_immutable_inputs(self) -> None:
        observed_spec_hash = _stable_file_sha256(
            self.path,
            "probe specification",
            maximum_bytes=MAX_SPEC_BYTES,
            require_single_link=True,
        )
        if observed_spec_hash != self.file_sha256:
            raise ProbeError(
                "probe_spec_changed",
                "probe specification changed after validation",
            )
        for binding, role in (
            (self.nvidia_smi, "nvidia-smi"),
            (self.katago, "KataGo binary"),
            (self.model, "analysis model"),
            (self.config, "analysis config"),
            (self.query, "analysis query"),
            (self.probe_module, "lease probe module"),
            (self.sentinel_module, "lease sentinel module"),
        ):
            binding.verify(role)

    @property
    def inventory_argv(self) -> tuple[str, ...]:
        return (
            os.fspath(self.nvidia_smi.path),
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        )

    @property
    def process_query_argv(self) -> tuple[str, ...]:
        return (
            os.fspath(self.nvidia_smi.path),
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        )

    @property
    def analysis_argv(self) -> tuple[str, ...]:
        return (
            os.fspath(self.katago.path),
            "analysis",
            "-config",
            os.fspath(self.config.path),
            "-model",
            os.fspath(self.model.path),
        )


def _validate_bound_paths(
    source: Path,
    bindings: Sequence[tuple[FileBinding, str]],
) -> None:
    seen_paths: dict[Path, str] = {}
    seen_files: dict[tuple[int, int], str] = {}
    try:
        source_metadata = source.lstat()
        source_identity = (source_metadata.st_dev, source_metadata.st_ino)
    except OSError as exc:
        raise ProbeError(
            "unsafe_path", f"could not stat probe specification: {exc}"
        ) from exc
    for binding, role in bindings:
        previous_role = seen_paths.get(binding.path)
        if previous_role is not None:
            raise ProbeError(
                "invalid_probe_spec",
                "probe roles must bind distinct paths",
                details={"first_role": previous_role, "second_role": role},
            )
        seen_paths[binding.path] = role
        metadata = binding.path.lstat()
        identity = (metadata.st_dev, metadata.st_ino)
        previous_file_role = seen_files.get(identity)
        if previous_file_role is not None:
            raise ProbeError(
                "invalid_probe_spec",
                "probe roles must not alias the same file",
                details={"first_role": previous_file_role, "second_role": role},
            )
        if identity == source_identity:
            raise ProbeError("unsafe_path", f"probe specification aliases the {role}")
        seen_files[identity] = role


def load_probe_spec(
    path: Path,
    *,
    expected_spec_sha256: str | None = None,
) -> ProbeSpec:
    """Load and fully verify one canonical, self-hashed probe specification."""

    source = _canonical_absolute_path(path, "probe specification")
    raw, data = _load_canonical_object(
        source,
        "probe specification",
        maximum_bytes=MAX_SPEC_BYTES,
        require_single_link=True,
    )
    _exact_keys(raw, _SPEC_KEYS, "probe specification")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != SCHEMA_VERSION
    ):
        raise ProbeError(
            "invalid_probe_spec", "probe specification schema_version is unsupported"
        )
    if raw["contract"] != SPEC_CONTRACT:
        raise ProbeError(
            "invalid_probe_spec", "probe specification contract is unsupported"
        )
    body = dict(raw)
    identity = _require_sha256(
        body.pop("spec_sha256", None), "probe specification self-hash"
    )
    if identity != canonical_sha256(body):
        raise ProbeError(
            "invalid_probe_spec", "probe specification self-hash is invalid"
        )
    file_identity = hashlib.sha256(data).hexdigest()
    if expected_spec_sha256 is not None:
        expected = _require_sha256(
            expected_spec_sha256, "expected probe specification hash"
        )
        if expected not in {identity, file_identity}:
            raise ProbeError(
                "invalid_probe_spec",
                "probe specification hash is not the expected identity",
            )

    nvidia_smi = _binding_from_value(
        raw["nvidia_smi"], "nvidia-smi binding", executable=True
    )
    katago = _binding_from_value(raw["katago"], "KataGo binding", executable=True)
    model = _binding_from_value(raw["model"], "model binding", executable=False)
    config = _binding_from_value(raw["config"], "config binding", executable=False)
    query = _binding_from_value(raw["query"], "query binding", executable=False)
    probe_module = _binding_from_value(
        raw["probe_module"], "probe module binding", executable=False
    )
    sentinel_module = _binding_from_value(
        raw["sentinel_module"], "sentinel module binding", executable=False
    )
    _validate_bound_paths(
        source,
        (
            (nvidia_smi, "nvidia-smi"),
            (katago, "KataGo binary"),
            (model, "analysis model"),
            (config, "analysis config"),
            (query, "analysis query"),
            (probe_module, "lease probe module"),
            (sentinel_module, "lease sentinel module"),
        ),
    )
    readiness_value = raw["sentinel_readiness_path"]
    if not isinstance(readiness_value, str) or not readiness_value:
        raise ProbeError(
            "invalid_probe_spec",
            "sentinel_readiness_path must be a nonempty absolute path",
        )
    readiness_path = _canonical_absolute_path(
        Path(readiness_value), "sentinel readiness path"
    )
    if readiness_path in {
        nvidia_smi.path,
        katago.path,
        model.path,
        config.path,
        query.path,
        probe_module.path,
        sentinel_module.path,
        source,
    }:
        raise ProbeError(
            "invalid_probe_spec",
            "sentinel readiness path aliases an immutable probe input",
        )

    gpu_uuid = raw["expected_gpu_uuid"]
    if not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        raise ProbeError("invalid_probe_spec", "expected_gpu_uuid is malformed")
    gpu_index = _nonnegative_int(
        raw["expected_gpu_index"], "expected_gpu_index", maximum=4096
    )
    process_count = _positive_int(
        raw["expected_evaluator_process_count"],
        "expected_evaluator_process_count",
        maximum=1,
    )
    timeout = _positive_timeout(raw["timeout_seconds"])
    cleanup_margin = _positive_duration(
        raw["cleanup_margin_seconds"],
        "cleanup_margin_seconds",
        maximum=MAX_TIMEOUT_SECONDS,
    )
    poll_interval = _positive_duration(
        raw["poll_interval_seconds"],
        "poll_interval_seconds",
        maximum=5.0,
    )
    if cleanup_margin + poll_interval >= timeout:
        raise ProbeError(
            "invalid_probe_spec",
            "timeout_seconds must leave time for work before the cleanup margin",
        )
    minimum_work = _positive_int(
        raw["minimum_completed_work"],
        "minimum_completed_work",
        maximum=1,
    )

    config_data = _stable_read(
        config.path, "analysis config", maximum_bytes=MAX_CONFIG_BYTES
    )
    if hashlib.sha256(config_data).hexdigest() != config.sha256:
        raise ProbeError(
            "bound_input_changed", "analysis config hash changed during validation"
        )
    _parse_config(config_data)
    query_data = _stable_read(
        query.path, "analysis query", maximum_bytes=MAX_QUERY_BYTES
    )
    if hashlib.sha256(query_data).hexdigest() != query.sha256:
        raise ProbeError(
            "bound_input_changed", "analysis query hash changed during validation"
        )
    _validated_query(query_data)

    return ProbeSpec(
        path=source,
        file_sha256=file_identity,
        spec_sha256=identity,
        nvidia_smi=nvidia_smi,
        katago=katago,
        model=model,
        config=config,
        query=query,
        probe_module=probe_module,
        sentinel_module=sentinel_module,
        sentinel_readiness_path=readiness_path,
        expected_gpu_uuid=gpu_uuid,
        expected_gpu_index=gpu_index,
        expected_evaluator_process_count=process_count,
        timeout_seconds=timeout,
        cleanup_margin_seconds=cleanup_margin,
        poll_interval_seconds=poll_interval,
        minimum_completed_work=minimum_work,
        raw=MappingProxyType(raw),
    )


load_spec = load_probe_spec


def build_probe_spec(
    *,
    nvidia_smi: Path,
    katago: Path,
    model: Path,
    config: Path,
    query: Path,
    expected_gpu_uuid: str,
    expected_gpu_index: int,
    sentinel_readiness_path: Path,
    expected_evaluator_process_count: int = 1,
    timeout_seconds: float = 60.0,
    cleanup_margin_seconds: float = DEFAULT_CLEANUP_MARGIN_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    minimum_completed_work: int = 1,
    probe_module: Path | None = None,
    sentinel_module: Path | None = None,
) -> Mapping[str, Any]:
    """Build a strict probe spec value from existing immutable inputs."""

    nvidia_binding = _binding_for_path(nvidia_smi, "nvidia-smi", executable=True)
    katago_binding = _binding_for_path(katago, "KataGo binary", executable=True)
    model_binding = _binding_for_path(model, "analysis model", executable=False)
    config_binding = _binding_for_path(config, "analysis config", executable=False)
    query_binding = _binding_for_path(query, "analysis query", executable=False)
    probe_binding = _binding_for_path(
        Path(os.path.abspath(__file__)) if probe_module is None else probe_module,
        "lease probe module",
        executable=False,
    )
    sentinel_binding = _binding_for_path(
        (
            Path(os.path.abspath(__file__)).with_name("autonomy_lease_worker.py")
            if sentinel_module is None
            else sentinel_module
        ),
        "lease sentinel module",
        executable=False,
    )
    readiness_path = _canonical_absolute_path(
        sentinel_readiness_path, "sentinel readiness path"
    )

    if (
        not isinstance(expected_gpu_uuid, str)
        or _GPU_UUID_RE.fullmatch(expected_gpu_uuid) is None
    ):
        raise ProbeError("invalid_probe_spec", "expected_gpu_uuid is malformed")
    gpu_index = _nonnegative_int(expected_gpu_index, "expected_gpu_index", maximum=4096)
    process_count = _positive_int(
        expected_evaluator_process_count,
        "expected_evaluator_process_count",
        maximum=1,
    )
    timeout = _positive_timeout(timeout_seconds)
    cleanup_margin = _positive_duration(
        cleanup_margin_seconds,
        "cleanup_margin_seconds",
        maximum=MAX_TIMEOUT_SECONDS,
    )
    poll_interval = _positive_duration(
        poll_interval_seconds,
        "poll_interval_seconds",
        maximum=5.0,
    )
    if cleanup_margin + poll_interval >= timeout:
        raise ProbeError(
            "invalid_probe_spec",
            "timeout_seconds must leave time for work before the cleanup margin",
        )
    minimum = _positive_int(minimum_completed_work, "minimum_completed_work", maximum=1)

    config_data = _stable_read(
        config_binding.path, "analysis config", maximum_bytes=MAX_CONFIG_BYTES
    )
    if hashlib.sha256(config_data).hexdigest() != config_binding.sha256:
        raise ProbeError("bound_input_changed", "analysis config changed")
    _parse_config(config_data)
    query_data = _stable_read(
        query_binding.path, "analysis query", maximum_bytes=MAX_QUERY_BYTES
    )
    if hashlib.sha256(query_data).hexdigest() != query_binding.sha256:
        raise ProbeError("bound_input_changed", "analysis query changed")
    _validated_query(query_data)

    binding_paths = [
        nvidia_binding.path,
        katago_binding.path,
        model_binding.path,
        config_binding.path,
        query_binding.path,
        probe_binding.path,
        sentinel_binding.path,
    ]
    if readiness_path in set(binding_paths):
        raise ProbeError(
            "invalid_probe_spec",
            "sentinel readiness path aliases an immutable probe input",
        )
    if len(set(binding_paths)) != len(binding_paths):
        raise ProbeError("invalid_probe_spec", "probe roles must bind distinct paths")
    file_identities = [
        (item.lstat().st_dev, item.lstat().st_ino) for item in binding_paths
    ]
    if len(set(file_identities)) != len(file_identities):
        raise ProbeError(
            "invalid_probe_spec", "probe roles must not alias the same file"
        )

    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SPEC_CONTRACT,
        "nvidia_smi": dict(nvidia_binding.spec_value()),
        "katago": dict(katago_binding.spec_value()),
        "model": dict(model_binding.spec_value()),
        "config": dict(config_binding.spec_value()),
        "query": dict(query_binding.spec_value()),
        "probe_module": dict(probe_binding.spec_value()),
        "sentinel_module": dict(sentinel_binding.spec_value()),
        "sentinel_readiness_path": str(readiness_path),
        "expected_gpu_uuid": expected_gpu_uuid,
        "expected_gpu_index": gpu_index,
        "expected_evaluator_process_count": process_count,
        "timeout_seconds": timeout,
        "cleanup_margin_seconds": cleanup_margin,
        "poll_interval_seconds": poll_interval,
        "minimum_completed_work": minimum,
    }
    value["spec_sha256"] = canonical_sha256(value)
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    destination = _canonical_absolute_path(path, "probe specification output")
    parent = _canonical_absolute_path(
        destination.parent, "probe specification output parent"
    )
    if parent.is_symlink() or not parent.is_dir():
        raise ProbeError(
            "unsafe_path",
            "probe specification output parent must be an existing "
            "non-symlink directory",
        )
    if os.path.lexists(os.fspath(destination)):
        raise ProbeError(
            "publication_conflict", "probe specification output already exists"
        )
    data = (canonical_json(dict(value)) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(destination), flags, 0o444)
    except OSError as exc:
        raise ProbeError(
            "publication_failed",
            f"could not create probe specification: {exc}",
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        _fsync_directory(parent)
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def publish_probe_spec(
    path: Path,
    *,
    nvidia_smi: Path,
    katago: Path,
    model: Path,
    config: Path,
    query: Path,
    expected_gpu_uuid: str,
    expected_gpu_index: int,
    sentinel_readiness_path: Path,
    expected_evaluator_process_count: int = 1,
    timeout_seconds: float = 60.0,
    cleanup_margin_seconds: float = DEFAULT_CLEANUP_MARGIN_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    minimum_completed_work: int = 1,
    probe_module: Path | None = None,
    sentinel_module: Path | None = None,
) -> ProbeSpec:
    """Publish one immutable canonical probe specification without replacement."""

    destination = _canonical_absolute_path(path, "probe specification output")
    value = build_probe_spec(
        nvidia_smi=nvidia_smi,
        katago=katago,
        model=model,
        config=config,
        query=query,
        expected_gpu_uuid=expected_gpu_uuid,
        expected_gpu_index=expected_gpu_index,
        sentinel_readiness_path=sentinel_readiness_path,
        expected_evaluator_process_count=expected_evaluator_process_count,
        timeout_seconds=timeout_seconds,
        cleanup_margin_seconds=cleanup_margin_seconds,
        poll_interval_seconds=poll_interval_seconds,
        minimum_completed_work=minimum_completed_work,
        probe_module=probe_module,
        sentinel_module=sentinel_module,
    )
    _write_once(destination, value)
    return load_probe_spec(destination, expected_spec_sha256=str(value["spec_sha256"]))


publish_spec = publish_probe_spec


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        env: Mapping[str, str] | None = None,
    ) -> Any: ...


def _validated_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or len(argv) > 256
        or any(
            not isinstance(part, str)
            or not part
            or "\x00" in part
            or "\n" in part
            or "\r" in part
            for part in argv
        )
        or sum(len(part.encode("utf-8")) + 1 for part in argv) > 64 * 1024
    ):
        raise ProbeError("invalid_command", "command argv is malformed or too large")
    return tuple(argv)


class BoundedSubprocessRunner:
    """Run argv-only subprocesses with streaming output and deadline limits."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
        try:
            process.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError):
            pass

    @staticmethod
    def _process_group_alive(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            return True
        return True

    def run(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        timeout_seconds: float,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        command = _validated_argv(argv)
        if not isinstance(input_bytes, bytes) or len(input_bytes) > MAX_QUERY_BYTES:
            raise ProbeError(
                "invalid_command", "command input is malformed or too large"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ProbeError("invalid_command", "command timeout must be positive")
        for value, role in (
            (max_stdout_bytes, "stdout"),
            (max_stderr_bytes, "stderr"),
        ):
            if type(value) is not int or value < 0:
                raise ProbeError(
                    "invalid_command", f"command {role} limit must be nonnegative"
                )
        environment: dict[str, str] | None
        if env is None:
            environment = None
        else:
            environment = {}
            for key, value in env.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or "=" in key
                    or "\x00" in key
                    or not isinstance(value, str)
                    or "\x00" in value
                ):
                    raise ProbeError(
                        "invalid_command", "command environment is malformed"
                    )
                environment[key] = value

        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            shell=False,
            start_new_session=True,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._kill(process)
            raise ProbeError("command_failed", "subprocess pipes were not created")

        selector = selectors.DefaultSelector()
        stdout = bytearray()
        stderr = bytearray()
        input_offset = 0
        streams = {
            "stdin": process.stdin,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        try:
            for stream in streams.values():
                os.set_blocking(stream.fileno(), False)
            if input_bytes:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            deadline = self._clock() + float(timeout_seconds)

            while selector.get_map():
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                events = selector.select(min(0.05, remaining))
                for key, _mask in events:
                    label = key.data
                    stream = key.fileobj
                    if label == "stdin":
                        try:
                            written = os.write(
                                stream.fileno(), input_bytes[input_offset:]
                            )
                        except BrokenPipeError:
                            selector.unregister(stream)
                            stream.close()
                            continue
                        except BlockingIOError:
                            continue
                        input_offset += written
                        if input_offset >= len(input_bytes):
                            selector.unregister(stream)
                            stream.close()
                        continue

                    target = stdout if label == "stdout" else stderr
                    limit = max_stdout_bytes if label == "stdout" else max_stderr_bytes
                    try:
                        block = os.read(stream.fileno(), min(65536, limit + 1))
                    except BlockingIOError:
                        continue
                    if not block:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    target.extend(block)
                    if len(target) > limit:
                        raise ProbeError(
                            "command_output_limit",
                            f"command {label} exceeded its {limit}-byte limit",
                        )

            remaining = deadline - self._clock()
            if remaining <= 0 and process.poll() is None:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            returncode = process.wait(timeout=max(0.001, remaining))
            if self._process_group_alive(process.pid):
                self._kill(process)
                raise ProbeError(
                    "command_process_group_not_drained",
                    "command exited while descendants remained alive",
                )
            return CommandResult(returncode, bytes(stdout), bytes(stderr))
        except BaseException:
            self._kill(process)
            raise
        finally:
            selector.close()
            for stream in streams.values():
                try:
                    stream.close()
                except OSError:
                    pass


SubprocessRunner = BoundedSubprocessRunner


class AnalysisProcess(Protocol):
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def finish(self, *, timeout_seconds: float) -> CommandResult: ...

    def kill_and_wait(self, *, timeout_seconds: float) -> None: ...


class AnalysisLauncher(Protocol):
    def start(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        env: Mapping[str, str],
    ) -> AnalysisProcess: ...


class ManagedAnalysisProcess:
    """A non-detached analysis child with capped streaming output."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        input_file: Any,
        *,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> None:
        self._process = process
        self._input_file = input_file
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._output_error: ProbeError | None = None
        self._reader_errors: list[BaseException] = []
        self._threads: list[threading.Thread] = []
        if process.stdout is None or process.stderr is None:
            self.kill_and_wait(timeout_seconds=1.0)
            raise ProbeError("probe_command_failed", "analysis pipes were not created")
        for stream, target, limit, role in (
            (process.stdout, self._stdout, max_stdout_bytes, "stdout"),
            (process.stderr, self._stderr, max_stderr_bytes, "stderr"),
        ):
            thread = threading.Thread(
                target=self._drain,
                args=(stream, target, limit, role),
                daemon=True,
                name=f"lease-probe-{role}",
            )
            thread.start()
            self._threads.append(thread)

    @property
    def pid(self) -> int:
        return self._process.pid

    def _drain(self, stream: Any, target: bytearray, limit: int, role: str) -> None:
        try:
            while block := stream.read(65536):
                remaining = limit + 1 - len(target)
                if remaining > 0:
                    target.extend(block[:remaining])
                if len(target) > limit:
                    self._output_error = ProbeError(
                        "probe_output_limit",
                        f"KataGo analysis {role} exceeded its {limit}-byte limit",
                    )
                    with contextlib.suppress(OSError):
                        self._process.kill()
                    return
        except (OSError, ValueError) as exc:
            self._reader_errors.append(exc)
            with contextlib.suppress(OSError):
                self._process.kill()
        finally:
            with contextlib.suppress(OSError):
                stream.close()

    def _raise_io_error(self) -> None:
        if self._output_error is not None:
            raise self._output_error
        if self._reader_errors:
            raise ProbeError(
                "probe_command_failed",
                f"could not read KataGo output: {self._reader_errors[0]}",
            ) from self._reader_errors[0]

    def poll(self) -> int | None:
        self._raise_io_error()
        return self._process.poll()

    def finish(self, *, timeout_seconds: float) -> CommandResult:
        try:
            returncode = self._process.wait(timeout=max(0.001, timeout_seconds))
        except subprocess.TimeoutExpired:
            self.kill_and_wait(timeout_seconds=1.0)
            raise
        for thread in self._threads:
            thread.join(timeout=max(0.001, timeout_seconds))
        self._input_file.close()
        self._raise_io_error()
        if any(thread.is_alive() for thread in self._threads):
            raise ProbeError(
                "probe_command_failed", "analysis output readers did not drain"
            )
        return CommandResult(
            returncode=returncode,
            stdout=bytes(self._stdout),
            stderr=bytes(self._stderr),
        )

    def kill_and_wait(self, *, timeout_seconds: float) -> None:
        if self._process.poll() is None:
            with contextlib.suppress(OSError):
                self._process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            self._process.wait(timeout=max(0.001, timeout_seconds))
        for thread in self._threads:
            thread.join(timeout=max(0.001, timeout_seconds))
        with contextlib.suppress(OSError):
            self._input_file.close()


class SubprocessAnalysisLauncher:
    """Start KataGo in the probe's process group with parent-death cleanup."""

    def __init__(self) -> None:
        if sys.platform != "linux":
            raise ProbeError(
                "unsupported_platform",
                "the production analysis launcher requires Linux PR_SET_PDEATHSIG",
            )
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        self._libc.prctl.restype = ctypes.c_int

    def _parent_death_hook(self, expected_parent: int) -> Callable[[], None]:
        libc = self._libc

        def arm_parent_death_signal() -> None:
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                os._exit(126)
            if os.getppid() != expected_parent:
                os.kill(os.getpid(), signal.SIGKILL)

        return arm_parent_death_signal

    def start(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes,
        max_stdout_bytes: int,
        max_stderr_bytes: int,
        env: Mapping[str, str],
    ) -> ManagedAnalysisProcess:
        command = _validated_argv(argv)
        input_file = tempfile.TemporaryFile()  # noqa: SIM115 - owned by process handle
        try:
            input_file.write(input_bytes)
            input_file.seek(0)
            parent_pid = os.getpid()
            process = subprocess.Popen(
                list(command),
                stdin=input_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                shell=False,
                start_new_session=False,
                # Parent-death cleanup is required if the outer drill SIGKILLs
                # the probe. This executes before this process starts readers.
                preexec_fn=self._parent_death_hook(parent_pid),  # noqa: PLW1509
            )
        except BaseException:
            input_file.close()
            raise
        return ManagedAnalysisProcess(
            process,
            input_file,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )


def _output_bytes(value: Any, role: str, maximum: int) -> bytes:
    if isinstance(value, str):
        try:
            data = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProbeError("command_failed", f"{role} is not valid UTF-8") from exc
    elif isinstance(value, bytes):
        data = value
    else:
        raise ProbeError("command_failed", f"{role} is not text or bytes")
    if len(data) > maximum:
        raise ProbeError(
            "command_output_limit", f"{role} exceeded its {maximum}-byte limit"
        )
    return data


def _checked_command_result(
    result: Any,
    role: str,
    *,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> CommandResult:
    returncode = getattr(result, "returncode", None)
    if type(returncode) is not int:
        raise ProbeError("command_failed", f"{role} returned no valid status")
    stdout = _output_bytes(
        getattr(result, "stdout", b"") or b"",
        f"{role} stdout",
        max_stdout_bytes,
    )
    stderr = _output_bytes(
        getattr(result, "stderr", b"") or b"",
        f"{role} stderr",
        max_stderr_bytes,
    )
    return CommandResult(returncode=returncode, stdout=stdout, stderr=stderr)


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str


@dataclass(frozen=True)
class GpuObservation:
    gpu_uuid: str
    gpu_index: int
    processes: tuple[GpuProcess, ...]


class GpuOwnershipProbe(Protocol):
    def observe(self, *, timeout_seconds: float) -> Any: ...


class NvidiaSmiComputeProbe:
    """Strictly inventory compute applications on one UUID with nvidia-smi."""

    def __init__(
        self,
        *,
        nvidia_smi: FileBinding,
        expected_gpu_uuid: str,
        expected_gpu_index: int,
        runner: CommandRunner,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._nvidia_smi = nvidia_smi
        self._expected_gpu_uuid = expected_gpu_uuid
        self._expected_gpu_index = expected_gpu_index
        self._runner = runner
        self._clock = clock

    @property
    def inventory_argv(self) -> tuple[str, ...]:
        return (
            os.fspath(self._nvidia_smi.path),
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        )

    @property
    def process_query_argv(self) -> tuple[str, ...]:
        return (
            os.fspath(self._nvidia_smi.path),
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        )

    def _run(self, argv: Sequence[str], role: str, *, deadline: float) -> CommandResult:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise ProbeError("gpu_probe_timeout", "GPU observation deadline expired")
        self._nvidia_smi.verify("nvidia-smi")
        try:
            raw = self._runner.run(
                argv,
                input_bytes=b"",
                timeout_seconds=remaining,
                max_stdout_bytes=MAX_GPU_STDOUT_BYTES,
                max_stderr_bytes=MAX_GPU_STDERR_BYTES,
                env=None,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProbeError(
                "gpu_probe_timeout", f"{role} exceeded the probe deadline"
            ) from exc
        except ProbeError as exc:
            if exc.code in {"bound_input_changed", "unsafe_path"}:
                raise
            raise ProbeError("gpu_probe_failed", f"{role} failed: {exc}") from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProbeError(
                "gpu_probe_failed", f"{role} could not execute: {exc}"
            ) from exc
        result = _checked_command_result(
            raw,
            role,
            max_stdout_bytes=MAX_GPU_STDOUT_BYTES,
            max_stderr_bytes=MAX_GPU_STDERR_BYTES,
        )
        if result.returncode != 0:
            raise ProbeError(
                "gpu_probe_failed",
                f"{role} exited unsuccessfully",
                details={
                    "returncode": result.returncode,
                    "stderr": result.stderr[-4096:].decode("utf-8", errors="replace"),
                },
            )
        self._nvidia_smi.verify("nvidia-smi")
        return result

    @staticmethod
    def _decode_lines(data: bytes, role: str) -> tuple[str, ...]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProbeError(
                "gpu_probe_malformed", f"{role} output is not UTF-8"
            ) from exc
        if "\x00" in text:
            raise ProbeError(
                "gpu_probe_malformed", f"{role} output contains a NUL byte"
            )
        return tuple(line.strip() for line in text.splitlines() if line.strip())

    def observe(self, *, timeout_seconds: float) -> GpuObservation:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise ProbeError("gpu_probe_timeout", "GPU observation has no time budget")
        deadline = self._clock() + float(timeout_seconds)
        inventory_result = self._run(
            self.inventory_argv, "GPU inventory command", deadline=deadline
        )
        inventory: dict[int, str] = {}
        seen_uuids = set()
        for line in self._decode_lines(inventory_result.stdout, "GPU inventory"):
            columns = [column.strip() for column in line.split(",")]
            if len(columns) != 2:
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU inventory row does not have index and UUID",
                    details={"line": line},
                )
            try:
                index = int(columns[0])
            except ValueError as exc:
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU inventory index is malformed",
                    details={"line": line},
                ) from exc
            uuid = columns[1]
            if (
                index < 0
                or _GPU_UUID_RE.fullmatch(uuid) is None
                or index in inventory
                or uuid in seen_uuids
            ):
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU inventory contains an invalid or duplicate row",
                    details={"line": line},
                )
            inventory[index] = uuid
            seen_uuids.add(uuid)
        observed_uuid = inventory.get(self._expected_gpu_index)
        if observed_uuid != self._expected_gpu_uuid:
            raise ProbeError(
                "gpu_uuid_mismatch",
                "expected GPU index does not map to the bound GPU UUID",
                details={
                    "expected_gpu_index": self._expected_gpu_index,
                    "expected_gpu_uuid": self._expected_gpu_uuid,
                    "observed_gpu_uuid": observed_uuid,
                },
            )

        process_result = self._run(
            self.process_query_argv, "GPU compute-process command", deadline=deadline
        )
        processes = []
        seen_pids = set()
        for line in self._decode_lines(
            process_result.stdout, "GPU compute-process query"
        ):
            columns = [column.strip() for column in line.split(",", 2)]
            if len(columns) != 3:
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU process row does not have UUID, PID, and process name",
                    details={"line": line},
                )
            uuid, raw_pid, process_name = columns
            if uuid.startswith("MIG-"):
                if _MIG_UUID_RE.fullmatch(uuid) is None:
                    raise ProbeError(
                        "gpu_probe_malformed",
                        "GPU process row contains an unsafe MIG UUID",
                        details={"line": line},
                    )
                # Physical-GPU exclusivity is bound by the GPU-* inventory row.
                # Unrelated MIG compute rows cannot be mapped to that physical
                # index by this query and are intentionally excluded.
                continue
            try:
                pid = int(raw_pid)
            except ValueError as exc:
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU process PID is malformed",
                    details={"line": line},
                ) from exc
            if (
                _GPU_UUID_RE.fullmatch(uuid) is None
                or uuid not in seen_uuids
                or pid <= 0
                or not process_name
                or "\x00" in process_name
            ):
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU process row is unsafe",
                    details={"line": line},
                )
            if uuid != self._expected_gpu_uuid:
                continue
            if pid in seen_pids:
                raise ProbeError(
                    "gpu_probe_malformed",
                    "GPU process query contains a duplicate PID",
                    details={"pid": pid},
                )
            seen_pids.add(pid)
            processes.append(GpuProcess(pid=pid, process_name=process_name))
        processes.sort(key=lambda item: item.pid)
        return GpuObservation(
            gpu_uuid=observed_uuid,
            gpu_index=self._expected_gpu_index,
            processes=tuple(processes),
        )

    __call__ = observe


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _completed_analysis_count(data: bytes, query_id: str, maximum_visits: int) -> int:
    if not data or not data.endswith(b"\n"):
        raise ProbeError(
            "malformed_probe_output",
            "KataGo output must be nonempty and newline-terminated",
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProbeError(
            "malformed_probe_output", "KataGo output is not UTF-8"
        ) from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ProbeError(
            "malformed_probe_output",
            "KataGo must return exactly one analysis record",
        )
    try:
        record = json.loads(
            lines[0],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ProbeError as exc:
        raise ProbeError(
            "malformed_probe_output", f"KataGo returned invalid JSON: {exc}"
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProbeError(
            "malformed_probe_output", f"KataGo returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(record, Mapping):
        raise ProbeError(
            "malformed_probe_output", "KataGo analysis record must be an object"
        )
    try:
        _ensure_finite_json(record, "KataGo analysis record")
    except ProbeError as exc:
        raise ProbeError("malformed_probe_output", str(exc)) from exc
    if (
        record.get("id") != query_id
        or "error" in record
        or record.get("isDuringSearch") is True
    ):
        raise ProbeError(
            "malformed_probe_output",
            "KataGo response ID is wrong or the response contains an error",
        )
    root = record.get("rootInfo")
    moves = record.get("moveInfos")
    if not isinstance(root, Mapping) or not isinstance(moves, list) or not moves:
        raise ProbeError(
            "malformed_probe_output",
            "KataGo response has no complete rootInfo and moveInfos",
        )
    for key in ("winrate", "scoreLead", "utility"):
        if not _finite_number(root.get(key)):
            raise ProbeError(
                "malformed_probe_output",
                f"KataGo rootInfo.{key} is missing or non-finite",
            )
    root_visits = root.get("visits")
    if (
        not 0.0 <= float(root["winrate"]) <= 1.0
        or type(root_visits) is not int
        or not 1 <= root_visits <= maximum_visits
    ):
        raise ProbeError(
            "malformed_probe_output",
            "KataGo rootInfo winrate or visits is out of range",
        )
    for index, move in enumerate(moves):
        if (
            not isinstance(move, Mapping)
            or not isinstance(move.get("move"), str)
            or not move["move"]
            or type(move.get("visits")) is not int
            or not 0 <= move["visits"] <= root_visits
        ):
            raise ProbeError(
                "malformed_probe_output",
                f"KataGo moveInfos[{index}] is malformed",
            )
    return 1


class AutonomyLeaseProbe:
    """Execute the bounded work and ownership observations for one spec."""

    def __init__(
        self,
        spec: ProbeSpec,
        *,
        runner: CommandRunner | None = None,
        gpu_probe: GpuOwnershipProbe | None = None,
        analysis_launcher: AnalysisLauncher | None = None,
        identity_probe: ProcessIdentityProbe | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.clock = clock
        self.sleep = sleep
        self.runner = runner or BoundedSubprocessRunner(clock=clock)
        self.analysis_launcher = analysis_launcher or SubprocessAnalysisLauncher()
        self.identity_probe = identity_probe or LinuxProcessIdentityProbe()
        self.gpu_probe = gpu_probe or NvidiaSmiComputeProbe(
            nvidia_smi=spec.nvidia_smi,
            expected_gpu_uuid=spec.expected_gpu_uuid,
            expected_gpu_index=spec.expected_gpu_index,
            runner=self.runner,
            clock=clock,
        )

    def _remaining(self, deadline: float, role: str) -> float:
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise ProbeError("probe_timeout", f"probe deadline expired before {role}")
        return remaining

    def _observe(self, deadline: float, role: str) -> Any:
        timeout = self._remaining(deadline, role)
        try:
            observer = getattr(self.gpu_probe, "observe", None)
            if callable(observer):
                observation = observer(timeout_seconds=timeout)
            elif callable(self.gpu_probe):
                observation = self.gpu_probe()
            else:
                raise TypeError("GPU probe is not callable")
        except subprocess.TimeoutExpired as exc:
            raise ProbeError("probe_timeout", f"{role} timed out") from exc
        except ProbeError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise ProbeError("gpu_probe_failed", f"{role} failed: {exc}") from exc
        self._remaining(deadline, f"completion of {role}")
        return observation

    def _validated_owners(
        self,
        observation: Any,
        role: str,
        *,
        allowed_pids: set[int],
        sentinel_pid: int,
    ) -> tuple[int, ...]:
        gpu_uuid = getattr(observation, "gpu_uuid", None)
        if gpu_uuid != self.spec.expected_gpu_uuid:
            raise ProbeError(
                "gpu_uuid_mismatch",
                f"{role} returned the wrong GPU UUID",
                details={
                    "expected": self.spec.expected_gpu_uuid,
                    "observed": gpu_uuid,
                },
            )
        observed_index = getattr(observation, "gpu_index", None)
        if observed_index is not None and (
            type(observed_index) is not int
            or observed_index != self.spec.expected_gpu_index
        ):
            raise ProbeError(
                "gpu_uuid_mismatch", f"{role} returned the wrong GPU index"
            )
        raw_processes = getattr(observation, "processes", None)
        if not isinstance(raw_processes, (list, tuple)):
            raise ProbeError("gpu_probe_malformed", f"{role} has no process inventory")
        owners = []
        seen_pids = set()
        unexpected = []
        for process in raw_processes:
            pid = getattr(process, "pid", None)
            name = getattr(process, "process_name", None)
            if (
                type(pid) is not int
                or pid <= 0
                or pid in seen_pids
                or not isinstance(name, str)
                or not name
            ):
                raise ProbeError(
                    "gpu_probe_malformed", f"{role} contains an invalid process"
                )
            seen_pids.add(pid)
            owners.append(pid)
            if pid not in allowed_pids:
                unexpected.append({"pid": pid, "process_name": name})
        if unexpected:
            raise ProbeError(
                "unexpected_gpu_process",
                f"{role} observed trainer or unknown compute ownership",
                details={"processes": unexpected},
            )
        owners.sort()
        if sentinel_pid not in owners:
            raise ProbeError(
                "evaluator_count_mismatch",
                f"{role} did not observe the persistent lease sentinel",
                details={
                    "sentinel_pid": sentinel_pid,
                    "pids": owners,
                },
            )
        return tuple(owners)

    def _load_sentinel_readiness(self) -> LinuxProcessIdentity:
        if not os.path.lexists(os.fspath(self.spec.sentinel_readiness_path)):
            raise ProbeError(
                "sentinel_not_ready", "sentinel readiness has not been published"
            )
        raw, _data = _load_canonical_object(
            self.spec.sentinel_readiness_path,
            "sentinel readiness",
            maximum_bytes=MAX_READY_BYTES,
            require_single_link=True,
        )
        expected_keys = {
            "schema_version",
            "contract",
            "gpu_uuid",
            "gpu_index",
            "pid",
            "process_identity",
            "module_sha256",
            "ready_sha256",
        }
        if set(raw) != expected_keys:
            raise ProbeError(
                "sentinel_readiness_invalid",
                "sentinel readiness fields differ from the contract",
            )
        body = dict(raw)
        supplied_hash = _require_sha256(
            body.pop("ready_sha256"), "sentinel readiness self-hash"
        )
        if supplied_hash != canonical_sha256(body):
            raise ProbeError(
                "sentinel_readiness_invalid",
                "sentinel readiness self-hash is invalid",
            )
        if (
            raw["schema_version"] != SCHEMA_VERSION
            or raw["contract"] != WORKER_READY_CONTRACT
            or raw["gpu_uuid"] != self.spec.expected_gpu_uuid
            or raw["gpu_index"] != self.spec.expected_gpu_index
            or raw["module_sha256"] != self.spec.sentinel_module.sha256
            or type(raw["pid"]) is not int
            or raw["pid"] <= 0
        ):
            raise ProbeError(
                "sentinel_readiness_invalid",
                "sentinel readiness does not match the bound lease launcher",
            )
        return LinuxProcessIdentity.from_value(
            raw["process_identity"], expected_pid=raw["pid"]
        )

    def _wait_for_sentinel(self, deadline: float) -> LinuxProcessIdentity:
        while True:
            self._remaining(deadline, "lease sentinel readiness")
            try:
                recorded = self._load_sentinel_readiness()
            except ProbeError as exc:
                if exc.code == "sentinel_not_ready":
                    self.sleep(
                        min(
                            self.spec.poll_interval_seconds,
                            self._remaining(deadline, "sentinel readiness retry"),
                        )
                    )
                    continue
                raise
            current = self.identity_probe.capture(recorded.pid)
            if current == recorded:
                observation = self._observe(
                    deadline, "initial sentinel GPU observation"
                )
                try:
                    owners = self._validated_owners(
                        observation,
                        "initial sentinel GPU observation",
                        allowed_pids={recorded.pid},
                        sentinel_pid=recorded.pid,
                    )
                except ProbeError as exc:
                    if exc.code != "evaluator_count_mismatch":
                        raise
                else:
                    if owners == (recorded.pid,):
                        return recorded
            self.sleep(
                min(
                    self.spec.poll_interval_seconds,
                    self._remaining(deadline, "sentinel GPU readiness retry"),
                )
            )

    def _require_same_identity(self, expected: LinuxProcessIdentity, role: str) -> None:
        observed = self.identity_probe.capture(expected.pid)
        if observed != expected:
            raise ProbeError(
                "process_identity_churn",
                f"{role} process identity changed",
                details={
                    "expected": expected.to_dict(),
                    "observed": (None if observed is None else observed.to_dict()),
                },
            )

    def _analysis_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        # UUID selection avoids CUDA ordinal ambiguity while the separately
        # inventoried index remains part of the immutable physical-GPU binding.
        environment["CUDA_VISIBLE_DEVICES"] = self.spec.expected_gpu_uuid
        environment["LC_ALL"] = "C"
        environment["LANG"] = "C"
        return environment

    def _run_analysis(
        self,
        query_data: bytes,
        query: Mapping[str, Any],
        query_id: str,
        sentinel: LinuxProcessIdentity,
        *,
        work_deadline: float,
        hard_deadline: float,
    ) -> int:
        process: AnalysisProcess | None = None
        child_identity: LinuxProcessIdentity | None = None
        try:
            process = self.analysis_launcher.start(
                self.spec.analysis_argv,
                input_bytes=query_data,
                max_stdout_bytes=MAX_ANALYSIS_STDOUT_BYTES,
                max_stderr_bytes=MAX_ANALYSIS_STDERR_BYTES,
                env=self._analysis_environment(),
            )
            child_identity = self.identity_probe.capture(process.pid)
            if child_identity is None:
                raise ProbeError(
                    "analysis_identity_unavailable",
                    "could not capture the KataGo child process identity",
                )
            if child_identity.process_group_id != os.getpgrp():
                raise ProbeError(
                    "analysis_kill_domain_mismatch",
                    "KataGo child is not in the probe's process group",
                )

            saw_analysis_gpu_owner = False
            while True:
                status = process.poll()
                self._require_same_identity(sentinel, "lease sentinel")
                if status is None:
                    observed_child = self.identity_probe.capture(process.pid)
                    if observed_child is None:
                        status = process.poll()
                        if status is None:
                            raise ProbeError(
                                "process_identity_churn",
                                "KataGo child disappeared while still running",
                            )
                    elif observed_child != child_identity:
                        raise ProbeError(
                            "process_identity_churn",
                            "KataGo child process identity changed",
                        )
                    descendants = self.identity_probe.descendants(process.pid)
                    if descendants:
                        raise ProbeError(
                            "analysis_descendant_process",
                            "KataGo analysis started an untracked child process",
                            details={"pids": list(descendants)},
                        )
                observation = self._observe(
                    work_deadline, "in-flight GPU ownership observation"
                )
                owners = self._validated_owners(
                    observation,
                    "in-flight GPU ownership observation",
                    allowed_pids={sentinel.pid, process.pid},
                    sentinel_pid=sentinel.pid,
                )
                saw_analysis_gpu_owner = saw_analysis_gpu_owner or process.pid in owners
                if status is not None:
                    break
                self.sleep(
                    min(
                        self.spec.poll_interval_seconds,
                        self._remaining(
                            work_deadline, "next in-flight ownership observation"
                        ),
                    )
                )

            result = process.finish(
                timeout_seconds=self._remaining(hard_deadline, "KataGo output cleanup")
            )
            try:
                result = _checked_command_result(
                    result,
                    "KataGo analysis",
                    max_stdout_bytes=MAX_ANALYSIS_STDOUT_BYTES,
                    max_stderr_bytes=MAX_ANALYSIS_STDERR_BYTES,
                )
            except ProbeError as exc:
                if exc.code == "command_output_limit":
                    raise ProbeError(
                        "probe_output_limit",
                        "KataGo analysis output exceeded its limit",
                    ) from exc
                raise
            if result.returncode != 0:
                raise ProbeError(
                    "probe_command_failed",
                    "KataGo analysis exited unsuccessfully",
                    details={
                        "returncode": result.returncode,
                        "stderr": result.stderr[-4096:].decode(
                            "utf-8", errors="replace"
                        ),
                    },
                )
            if not saw_analysis_gpu_owner:
                raise ProbeError(
                    "analysis_gpu_ownership_unobserved",
                    "KataGo child was never observed owning the expected GPU",
                )
            completed = _completed_analysis_count(
                result.stdout, query_id, int(query["maxVisits"])
            )

            clean_observations = 0
            while clean_observations < POST_ANALYSIS_CLEAN_OBSERVATIONS:
                self._require_same_identity(sentinel, "lease sentinel")
                current_child = self.identity_probe.capture(child_identity.pid)
                if current_child == child_identity:
                    raise ProbeError(
                        "analysis_cleanup_failed",
                        "KataGo child remained alive after process wait",
                    )
                if current_child is not None:
                    raise ProbeError(
                        "process_identity_churn",
                        "KataGo child PID was reused during cleanup",
                        details={"observed": current_child.to_dict()},
                    )
                observation = self._observe(
                    hard_deadline, "post-analysis GPU ownership observation"
                )
                owners = self._validated_owners(
                    observation,
                    "post-analysis GPU ownership observation",
                    allowed_pids={sentinel.pid, child_identity.pid},
                    sentinel_pid=sentinel.pid,
                )
                if child_identity.pid in owners:
                    if current_child is not None:
                        raise ProbeError(
                            "process_identity_churn",
                            "KataGo PID was reused while its GPU row remained",
                        )
                    clean_observations = 0
                elif owners == (sentinel.pid,):
                    clean_observations += 1
                else:
                    clean_observations = 0
                if clean_observations < POST_ANALYSIS_CLEAN_OBSERVATIONS:
                    self.sleep(
                        min(
                            self.spec.poll_interval_seconds,
                            self._remaining(
                                hard_deadline, "post-analysis cleanup observation"
                            ),
                        )
                    )
            return completed
        except subprocess.TimeoutExpired as exc:
            raise ProbeError(
                "probe_command_timeout",
                "KataGo analysis exceeded the internal work deadline",
            ) from exc
        except ProbeError as exc:
            if exc.code == "probe_timeout":
                raise ProbeError(
                    "probe_command_timeout",
                    "KataGo analysis exceeded the internal work deadline",
                ) from exc
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProbeError(
                "probe_command_failed", f"KataGo analysis could not execute: {exc}"
            ) from exc
        finally:
            if process is not None:
                process.kill_and_wait(
                    timeout_seconds=max(
                        0.001,
                        min(
                            self.spec.cleanup_margin_seconds,
                            hard_deadline - self.clock(),
                        ),
                    )
                )
                if (
                    child_identity is not None
                    and self.identity_probe.capture(child_identity.pid)
                    == child_identity
                ):
                    raise ProbeError(
                        "analysis_cleanup_failed",
                        "KataGo child survived bounded cleanup",
                    )

    def run(self) -> Mapping[str, Any]:
        started = self.clock()
        outer_deadline = started + self.spec.timeout_seconds
        work_deadline = outer_deadline - self.spec.cleanup_margin_seconds
        # Split the reserved interval between active cleanup and process-startup /
        # receipt overhead. The outer drill uses timeout_seconds unchanged.
        hard_deadline = outer_deadline - self.spec.cleanup_margin_seconds / 2.0
        self.spec.verify_immutable_inputs()
        config_data = _stable_read(
            self.spec.config.path,
            "analysis config",
            maximum_bytes=MAX_CONFIG_BYTES,
        )
        if hashlib.sha256(config_data).hexdigest() != self.spec.config.sha256:
            raise ProbeError("bound_input_changed", "analysis config changed")
        _parse_config(config_data)
        query_data = _stable_read(
            self.spec.query.path,
            "analysis query",
            maximum_bytes=MAX_QUERY_BYTES,
        )
        if hashlib.sha256(query_data).hexdigest() != self.spec.query.sha256:
            raise ProbeError("bound_input_changed", "analysis query changed")
        query, query_id = _validated_query(query_data)
        sentinel = self._wait_for_sentinel(work_deadline)
        self.spec.verify_immutable_inputs()

        completed = self._run_analysis(
            query_data,
            query,
            query_id,
            sentinel,
            work_deadline=work_deadline,
            hard_deadline=hard_deadline,
        )
        if completed < self.spec.minimum_completed_work:
            raise ProbeError(
                "insufficient_completed_work",
                "bounded analysis completed too few validated work items",
                details={
                    "minimum": self.spec.minimum_completed_work,
                    "completed": completed,
                },
            )

        # Close every verify/use window before claiming work or ownership.
        self.spec.verify_immutable_inputs()
        final_observation = self._observe(
            hard_deadline, "final GPU ownership observation"
        )
        final_owners = self._validated_owners(
            final_observation,
            "final GPU ownership observation",
            allowed_pids={sentinel.pid},
            sentinel_pid=sentinel.pid,
        )
        self._require_same_identity(sentinel, "lease sentinel")
        if final_owners != (sentinel.pid,):
            raise ProbeError(
                "evaluator_process_churn",
                "persistent evaluator GPU ownership changed during bounded work",
            )
        self.spec.verify_immutable_inputs()
        self._remaining(hard_deadline, "receipt publication")

        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": PROBE_RECEIPT_CONTRACT,
            "gpu_uuid": self.spec.expected_gpu_uuid,
            "evaluator_pids": [sentinel.pid],
            "model_sha256": self.spec.model.sha256,
            "config_sha256": self.spec.config.sha256,
            "completed_work_count": completed,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return receipt


EvaluatorLeaseProbe = AutonomyLeaseProbe


def run_probe(
    spec: Any,
    *,
    expected_spec_sha256: str | None = None,
    runner: CommandRunner | None = None,
    gpu_probe: GpuOwnershipProbe | None = None,
    analysis_launcher: AnalysisLauncher | None = None,
    identity_probe: ProcessIdentityProbe | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    """Load and execute a probe, with injectable command and GPU observations."""

    if isinstance(spec, ProbeSpec):
        loaded = spec
        if expected_spec_sha256 is not None:
            expected = _require_sha256(
                expected_spec_sha256, "expected probe specification hash"
            )
            if expected not in {loaded.spec_sha256, loaded.file_sha256}:
                raise ProbeError(
                    "invalid_probe_spec",
                    "loaded probe specification hash is not expected",
                )
    else:
        loaded = load_probe_spec(Path(spec), expected_spec_sha256=expected_spec_sha256)
    return AutonomyLeaseProbe(
        loaded,
        runner=runner,
        gpu_probe=gpu_probe,
        analysis_launcher=analysis_launcher,
        identity_probe=identity_probe,
        clock=clock,
        sleep=sleep,
    ).run()


def evaluator_probe_argv(
    spec: ProbeSpec,
    *,
    python_executable: Path = _PYTHON_EXECUTABLE,
) -> tuple[str, ...]:
    """Return the immutable argv intended for ``LeaseDrillSpec``."""

    python = _binding_for_path(python_executable, "Python executable", executable=True)
    spec.probe_module.verify("lease probe module")
    return (
        os.fspath(python.path),
        os.fspath(spec.probe_module.path),
        "--expected-module-sha256",
        spec.probe_module.sha256,
        "--spec",
        os.fspath(spec.path),
        "--expected-spec-sha256",
        spec.spec_sha256,
    )


def evaluator_sentinel_launch_argv(
    spec: ProbeSpec,
    *,
    python_executable: Path = _PYTHON_EXECUTABLE,
) -> tuple[str, ...]:
    """Return the exact processCount=1 persistent evaluator launch command."""

    python = _binding_for_path(python_executable, "Python executable", executable=True)
    spec.sentinel_module.verify("lease sentinel module")
    return (
        os.fspath(python.path),
        os.fspath(spec.sentinel_module.path),
        "--expected-module-sha256",
        spec.sentinel_module.sha256,
        "--expected-gpu-uuid",
        spec.expected_gpu_uuid,
        "--expected-gpu-index",
        str(spec.expected_gpu_index),
        "--ready",
        os.fspath(spec.sentinel_readiness_path),
    )


def lease_publisher_commands(
    spec: ProbeSpec,
    *,
    python_executable: Path = _PYTHON_EXECUTABLE,
) -> Mapping[str, Any]:
    """Expose values needed to build the dedicated lease-drill runtime."""

    return MappingProxyType(
        {
            "evaluator_launch_command": list(
                evaluator_sentinel_launch_argv(
                    spec, python_executable=python_executable
                )
            ),
            "evaluator_process_count": 1,
            "evaluator_readiness_path": os.fspath(spec.sentinel_readiness_path),
            "evaluator_probe_argv": list(
                evaluator_probe_argv(spec, python_executable=python_executable)
            ),
            "probe_internal_timeout_seconds": spec.timeout_seconds,
            "evaluator_probe_timeout_seconds": evaluator_probe_outer_timeout_seconds(
                spec
            ),
            "probe_module_sha256": spec.probe_module.sha256,
            "sentinel_module_sha256": spec.sentinel_module.sha256,
        }
    )


def evaluator_probe_outer_timeout_seconds(spec: ProbeSpec) -> float:
    """Return the drill deadline with fixed startup and cleanup reserve."""

    timeout = spec.timeout_seconds + OUTER_DRILL_TIMEOUT_RESERVE_SECONDS
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ProbeError(
            "invalid_probe_spec",
            "probe internal timeout leaves no valid outer drill reserve",
        )
    return timeout


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-module-sha256", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        expected_module_hash = _require_sha256(
            args.expected_module_sha256, "expected probe module SHA-256"
        )
        module_path = _canonical_absolute_path(
            Path(os.path.abspath(__file__)), "lease probe module"
        )
        observed_module_hash = _stable_file_sha256(module_path, "lease probe module")
        if observed_module_hash != expected_module_hash:
            raise ProbeError(
                "probe_module_changed",
                "lease probe module does not match its launch-command binding",
                details={
                    "expected": expected_module_hash,
                    "observed": observed_module_hash,
                },
            )
        receipt = run_probe(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
        )
    except ProbeError as exc:
        sys.stderr.write(canonical_json(exc.to_dict()) + "\n")
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        error = ProbeError(
            "unexpected_error",
            str(exc) or type(exc).__name__,
            details={"type": type(exc).__name__},
        )
        sys.stderr.write(canonical_json(error.to_dict()) + "\n")
        return 2
    sys.stdout.write(canonical_json(receipt) + "\n")
    return 0


__all__ = [
    "ERROR_CONTRACT",
    "OUTER_DRILL_TIMEOUT_RESERVE_SECONDS",
    "PROBE_RECEIPT_CONTRACT",
    "RECEIPT_CONTRACT",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "AnalysisLauncher",
    "AnalysisProcess",
    "AutonomyLeaseProbe",
    "AutonomyLeaseProbeError",
    "BoundedSubprocessRunner",
    "CommandResult",
    "CommandRunner",
    "EvaluatorLeaseProbe",
    "FileBinding",
    "GpuObservation",
    "GpuOwnershipProbe",
    "GpuProcess",
    "LinuxProcessIdentity",
    "LinuxProcessIdentityProbe",
    "ManagedAnalysisProcess",
    "NvidiaSmiComputeProbe",
    "ProbeError",
    "ProbeSpec",
    "ProbeSpecError",
    "build_probe_spec",
    "canonical_json",
    "canonical_sha256",
    "evaluator_probe_argv",
    "evaluator_probe_outer_timeout_seconds",
    "evaluator_sentinel_launch_argv",
    "file_sha256",
    "lease_publisher_commands",
    "load_probe_spec",
    "load_spec",
    "main",
    "parse_args",
    "publish_probe_spec",
    "publish_spec",
    "run_probe",
]


if __name__ == "__main__":
    raise SystemExit(main())
