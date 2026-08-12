"""Crash-safe process execution for :mod:`risk_score.cluster_scheduler`.

The scheduler remains the sole authority for selecting work and priority.  This
module only turns an existing scheduler claim into a hash-bound argv launch.
It deliberately has no priority argument and never accepts a shell command.

Executor configuration files and the per-work specifications stored in
``WorkItem.payload["executor_spec"]`` are compact, sorted-key,
newline-terminated canonical JSON with a self-hash over every field except
``spec_sha256``.  A minimal executor configuration has this shape::

    {
      "backoff_initial_seconds": 5.0,
      "backoff_max_seconds": 300.0,
      "contract": "risk-score-cluster-executor-spec-v1",
      "gpu7_id": "7",
      "gpu7_guardian_prefix": null,
      "gpu_ids": ["0", "1", "2", "3", "4", "5", "6", "7"],
      "heartbeat_interval_seconds": 5.0,
      "lease_proof_command": null,
      "lease_proof_timeout_seconds": 30.0,
      "owner_id": "cluster-executor",
      "poll_interval_seconds": 2.0,
      "retry_budget": 2,
      "scheduler_directory": "/absolute/scheduler",
      "schema_version": 1,
      "spec_sha256": "...",
      "stale_after_seconds": 30.0,
      "state_directory": "/absolute/executor-state"
    }

Every process is launched in a new process group with ``shell=False``.  The
durable start intent is written before spawning and the full Linux process
identity is written immediately afterward.  An interrupted ``starting`` state
is never replayed as another launch because it is impossible to prove that the
first launch did not occur.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import stat
import string
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production executors run on Unix.
    fcntl = None  # type: ignore[assignment]

from risk_score import gpu_lease_worker
from risk_score.cluster_scheduler import (
    Claim,
    ClusterScheduler,
    IdleReason,
    ReleaseOutcome,
    ReleaseRecord,
    SchedulerError,
    WorkKind,
    WorkRecord,
    WorkState,
    canonical_json_bytes,
    canonical_sha256,
)
from risk_score.promotion_host import (
    PROCESS_IDENTITY_FIELDS,
    HostCommandError,
    capture_process_identity,
    capture_spawned_process,
)

SCHEMA_VERSION = 1
EXECUTOR_SPEC_CONTRACT = "risk-score-cluster-executor-spec-v1"
WORK_SPEC_CONTRACT = "risk-score-cluster-work-spec-v1"
CLAIM_RECEIPT_CONTRACT = "risk-score-cluster-claim-receipt-v1"
START_INTENT_CONTRACT = "risk-score-cluster-start-intent-v1"
START_RECEIPT_CONTRACT = "risk-score-cluster-start-receipt-v1"
COMPLETION_RECEIPT_CONTRACT = "risk-score-cluster-completion-receipt-v1"
RELEASE_INTENT_CONTRACT = "risk-score-cluster-release-intent-v1"
RELEASE_RECEIPT_CONTRACT = "risk-score-cluster-release-receipt-v1"
RETRY_STATE_CONTRACT = "risk-score-cluster-retry-state-v1"
QUARANTINE_CONTRACT = "risk-score-cluster-quarantine-v1"
HEARTBEAT_CONTRACT = "risk-score-cluster-heartbeat-v1"
LEASE_GATE_CONTRACT = "risk-score-cluster-lease-gate-v1"
DRAIN_INTENT_CONTRACT = "risk-score-cluster-drain-intent-v1"
DRAIN_COMMAND_CONTRACT = "risk-score-cluster-drain-command-v1"
DRAIN_BOUNDARY_CONTRACT = "risk-score-cluster-drain-boundary-v1"
HALT_CONTRACT = "risk-score-cluster-safety-halt-v1"
STATUS_CONTRACT = "risk-score-cluster-executor-status-v1"
ONCE_CONTRACT = "risk-score-cluster-executor-once-v1"
ERROR_CONTRACT = "risk-score-cluster-executor-error-v1"
LOCK_FILENAME = ".executor.lock"
MAX_JSON_BYTES = 16 * 1024 * 1024

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_EXECUTOR_SPEC_REQUIRED_KEYS = {
    "schema_version",
    "contract",
    "scheduler_directory",
    "state_directory",
    "owner_id",
    "gpu_ids",
    "gpu7_id",
    "poll_interval_seconds",
    "heartbeat_interval_seconds",
    "stale_after_seconds",
    "retry_budget",
    "backoff_initial_seconds",
    "backoff_max_seconds",
    "lease_proof_command",
    "lease_proof_timeout_seconds",
    "spec_sha256",
}
_EXECUTOR_SPEC_OPTIONAL_KEYS = {"gpu7_guardian_prefix"}
_WORK_SPEC_KEYS = {
    "schema_version",
    "contract",
    "work_id",
    "kind",
    "eligible_gpus",
    "argv",
    "cwd",
    "environment",
    "lease_role",
    "safe_drain",
    "spec_sha256",
}
_SAFE_DRAIN_KEYS = {
    "command",
    "checkpoint_path",
    "require_checkpoint_change",
    "timeout_seconds",
}
_MAIN_PLACEHOLDERS = frozenset(
    {
        "claim_id",
        "guardian_receipt",
        "gpu_id",
        "log_path",
        "state_directory",
        "work_id",
    }
)
_DRAIN_PLACEHOLDERS = frozenset(
    set(_MAIN_PLACEHOLDERS) | set(PROCESS_IDENTITY_FIELDS) | {"checkpoint_path"}
)
_LEASE_PLACEHOLDERS = frozenset(
    {
        "claim_id",
        "executor_spec_sha256",
        "gpu_id",
        "kind",
        "lease_role",
        "work_id",
        "work_spec_sha256",
    }
)
_LEASE_ROLES = frozenset({"none", "trainer", "evaluator"})
_GUARDIAN_REQUIRED_PLACEHOLDERS = frozenset({"claim_id", "guardian_receipt", "work_id"})
_GUARDIAN_HASH_FLAGS = frozenset(
    {
        "--config-sha256",
        "--expected-config-sha256",
        "--expected-spec-sha256",
        "--spec-sha256",
    }
)
_GUARDIAN_COMMAND_MARKERS = frozenset({"--", "--command-json"})
_EVALUATOR_KINDS = frozenset(
    {
        WorkKind.PROMOTION_CONFIRMATION,
        WorkKind.PROMOTION_CANARY,
        WorkKind.SCREENING,
    }
)


class ClusterExecutorError(RuntimeError):
    """Base class for executor errors."""


class ExecutorSpecError(ClusterExecutorError, ValueError):
    """A canonical executor or work specification is malformed."""


class ExecutorStateError(ClusterExecutorError, ValueError):
    """Durable executor state is malformed or contradictory."""


class ExecutorBusyError(ClusterExecutorError):
    """Another local executor owns the nonblocking state lock."""


class LeaseProofDenied(ClusterExecutorError):
    """The external GPU-7 lease authority did not grant this launch."""


class ProcessSpawnError(ClusterExecutorError):
    """A launch failed, with an explicit indication whether retry is safe."""

    def __init__(self, message: str, *, safe_to_retry: bool) -> None:
        super().__init__(message)
        self.safe_to_retry = safe_to_retry


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class ProcessIdentity:
    """The complete PID-reuse-resistant identity required by the executor."""

    pid: int
    start_time_ticks: int
    command_sha256: str
    process_group_id: int
    boot_id: str
    cgroup: str

    @classmethod
    def from_value(cls, value: Any, role: str = "process") -> "ProcessIdentity":
        if isinstance(value, cls):
            identity = value
        else:
            if hasattr(value, "to_dict") and callable(value.to_dict):
                value = value.to_dict()
            if not isinstance(value, Mapping):
                raise ExecutorStateError(f"{role} identity must be an object")
            if set(value) != set(PROCESS_IDENTITY_FIELDS):
                raise ExecutorStateError(f"{role} identity fields differ from schema")
            try:
                identity = cls(
                    pid=value["pid"],
                    start_time_ticks=value["start_time_ticks"],
                    command_sha256=value["command_sha256"],
                    process_group_id=value["process_group_id"],
                    boot_id=value["boot_id"],
                    cgroup=value["cgroup"],
                )
            except (KeyError, TypeError) as exc:
                raise ExecutorStateError(f"{role} identity is malformed") from exc
        if (
            type(identity.pid) is not int
            or identity.pid <= 0
            or type(identity.start_time_ticks) is not int
            or identity.start_time_ticks < 0
            or type(identity.process_group_id) is not int
            or identity.process_group_id <= 0
            or not _is_sha256(identity.command_sha256)
            or not isinstance(identity.boot_id, str)
            or not identity.boot_id
            or not isinstance(identity.cgroup, str)
            or not identity.cgroup
        ):
            raise ExecutorStateError(f"{role} identity is malformed")
        return identity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "command_sha256": self.command_sha256,
            "process_group_id": self.process_group_id,
            "boot_id": self.boot_id,
            "cgroup": self.cgroup,
        }

    def same_process_as(self, other: "ProcessIdentity") -> bool:
        return self == other


class ProcessRunner(Protocol):
    """Injectable process operations used by :class:`ClusterExecutor`."""

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        log_path: Path,
    ) -> ProcessIdentity: ...

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout: float,
    ) -> CommandResult: ...

    def current_identity(self, pid: int) -> Optional[ProcessIdentity]: ...

    def returncode(self, identity: ProcessIdentity) -> Optional[int]: ...

    def process_group_alive(self, identity: ProcessIdentity) -> bool: ...


class SubprocessRunner:
    """Production runner; every command is an argv sequence with ``shell=False``."""

    def __init__(self) -> None:
        self._children: Dict[int, subprocess.Popen[Any]] = {}

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        log_path: Path,
    ) -> ProcessIdentity:
        _validate_argv(argv, "work argv", _MAIN_PLACEHOLDERS, expanded=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    list(argv),
                    cwd=os.fspath(cwd),
                    env=dict(environment),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    shell=False,
                )
        except OSError as exc:
            raise ProcessSpawnError(
                f"process could not be spawned: {exc}", safe_to_retry=True
            ) from exc
        self._children[process.pid] = process
        try:
            identity = ProcessIdentity.from_value(
                capture_spawned_process(process), "spawned process"
            )
        except Exception as exc:
            # A live child without the complete procfs identity is deliberately
            # left untouched. The durable start intent makes the executor halt
            # instead of launching a duplicate or signalling an unknown PID.
            exited = process.poll() is not None
            if exited:
                self._children.pop(process.pid, None)
            raise ProcessSpawnError(
                f"spawned process identity could not be captured: {exc}",
                safe_to_retry=exited,
            ) from exc
        return identity

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout: float,
    ) -> CommandResult:
        _validate_argv(argv, "command", _DRAIN_PLACEHOLDERS, expanded=True)
        completed = subprocess.run(
            list(argv),
            cwd=os.fspath(cwd),
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def current_identity(self, pid: int) -> Optional[ProcessIdentity]:
        try:
            return ProcessIdentity.from_value(
                capture_process_identity(pid), "observed process"
            )
        except (HostCommandError, ExecutorStateError):
            return None

    def returncode(self, identity: ProcessIdentity) -> Optional[int]:
        child = self._children.get(identity.pid)
        if child is None:
            return None
        returncode = child.poll()
        if returncode is not None:
            self._children.pop(identity.pid, None)
        return returncode

    def process_group_alive(self, identity: ProcessIdentity) -> bool:
        try:
            os.killpg(identity.process_group_id, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


@dataclass(frozen=True)
class SafeDrainSpec:
    command: Tuple[str, ...]
    checkpoint_path: Path
    require_checkpoint_change: bool
    timeout_seconds: float


@dataclass(frozen=True)
class WorkExecutionSpec:
    work_id: str
    kind: WorkKind
    eligible_gpus: Tuple[str, ...]
    argv: Tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    lease_role: str
    safe_drain: Optional[SafeDrainSpec]
    spec_sha256: str


@dataclass(frozen=True)
class ExecutorSpec:
    scheduler_directory: Path
    state_directory: Path
    owner_id: str
    gpu_ids: Tuple[str, ...]
    gpu7_id: Optional[str]
    poll_interval_seconds: float
    heartbeat_interval_seconds: float
    stale_after_seconds: float
    retry_budget: int
    backoff_initial_seconds: float
    backoff_max_seconds: float
    lease_proof_command: Optional[Tuple[str, ...]]
    lease_proof_timeout_seconds: float
    spec_sha256: str
    gpu7_guardian_prefix: Optional[Tuple[str, ...]] = None
    source_path: Optional[Path] = None


def _requires_gpu7_guardian(
    work_spec: WorkExecutionSpec, gpu7_id: Optional[str]
) -> bool:
    return (
        work_spec.lease_role == "none"
        and gpu7_id is not None
        and gpu7_id in work_spec.eligible_gpus
    )


@dataclass(frozen=True)
class LeaseRequest:
    work_id: str
    kind: str
    gpu_id: str
    lease_role: str
    claim_id: str
    executor_spec_sha256: str
    work_spec_sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "work_id": self.work_id,
            "kind": self.kind,
            "gpu_id": self.gpu_id,
            "lease_role": self.lease_role,
            "claim_id": self.claim_id,
            "executor_spec_sha256": self.executor_spec_sha256,
            "work_spec_sha256": self.work_spec_sha256,
        }


LeaseProofCallback = Callable[[LeaseRequest], Union[bool, Mapping[str, Any]]]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARACTERS)
    )


def _require_sha256(value: Any, role: str) -> str:
    if not _is_sha256(value):
        raise ExecutorSpecError(f"{role} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ExecutorSpecError(f"{role} must be a nonempty trimmed string")
    return value


def _number(value: Any, role: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutorSpecError(f"{role} must be a number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (positive and result <= 0)
        or (not positive and result < 0)
    ):
        qualifier = "positive and finite" if positive else "finite and nonnegative"
        raise ExecutorSpecError(f"{role} must be {qualifier}")
    return result


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutorStateError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ExecutorStateError(f"non-finite JSON value is forbidden: {value}")


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError as exc:
        raise ExecutorStateError(f"{role} is missing: {source}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ExecutorStateError(f"{role} must be a regular non-symlink file")
    if metadata.st_size > MAX_JSON_BYTES:
        raise ExecutorStateError(f"{role} exceeds the size limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutorStateError(f"cannot decode {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutorStateError(f"{role} root must be an object")
    if data != canonical_json_bytes(value) + b"\n":
        raise ExecutorStateError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_optional_object(path: Path, role: str) -> Optional[Dict[str, Any]]:
    if not os.path.lexists(os.fspath(path)):
        return None
    return _load_canonical_object(path, role)


def _self_hashed(value: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
    result = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    result.pop(hash_field, None)
    result[hash_field] = canonical_sha256(result)
    return result


def _validate_self_hash(value: Mapping[str, Any], hash_field: str, role: str) -> str:
    supplied = value.get(hash_field)
    if not _is_sha256(supplied):
        raise ExecutorStateError(f"{role} has no valid {hash_field}")
    body = dict(value)
    body.pop(hash_field, None)
    if canonical_sha256(body) != supplied:
        raise ExecutorStateError(f"{role} self-hash is invalid")
    return str(supplied)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ExecutorStateError(f"unsafe mutable state path: {target}")
    data = canonical_json_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(os.fspath(temporary), os.fspath(target))
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish immutable canonical JSON, accepting only an exact replay."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(value)) + b"\n"
    if os.path.lexists(os.fspath(target)):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise ExecutorStateError(f"immutable receipt conflicts: {target}")
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
                raise ExecutorStateError(f"immutable receipt conflicts: {target}")
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _normalized_absolute(
    value: Any,
    role: str,
    *,
    require_directory: bool = False,
    require_file: bool = False,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ExecutorSpecError(f"{role} must be an absolute path string")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise ExecutorSpecError(f"{role} must be lexically normalized and absolute")
    current = path
    while True:
        if current.is_symlink():
            raise ExecutorSpecError(f"{role} has a symlinked path component")
        if current.parent == current:
            break
        current = current.parent
    if require_directory and (not path.is_dir() or path.is_symlink()):
        raise ExecutorSpecError(f"{role} must be an existing non-symlink directory")
    if require_file:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise ExecutorSpecError(f"{role} must be an existing regular file") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ExecutorSpecError(f"{role} must be an existing regular file")
    return path


def _validate_argv(
    value: Any,
    role: str,
    allowed_placeholders: frozenset[str],
    *,
    allow_empty: bool = False,
    expanded: bool = False,
) -> Tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or (not value and not allow_empty)
        or any(
            not isinstance(part, str) or not part or "\x00" in part for part in value
        )
    ):
        raise ExecutorSpecError(f"{role} must be a nonempty JSON argv array")
    result = tuple(value)
    if expanded:
        return result
    formatter = string.Formatter()
    for part in result:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise ExecutorSpecError(f"{role} has invalid formatting") from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                field_name not in allowed_placeholders
                or format_spec
                or conversion is not None
            ):
                raise ExecutorSpecError(
                    f"{role} uses an unsupported placeholder: {field_name!r}"
                )
    return result


def _argv_placeholders(template: Sequence[str], role: str) -> Tuple[str, ...]:
    formatter = string.Formatter()
    placeholders: List[str] = []
    for part in template:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise ExecutorSpecError(f"{role} has invalid formatting") from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if format_spec or conversion is not None:
                raise ExecutorSpecError(f"{role} has an unsafe placeholder")
            placeholders.append(field_name)
    return tuple(placeholders)


def _validate_guardian_prefix(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    prefix = _validate_argv(
        value,
        "GPU7 guardian prefix",
        _MAIN_PLACEHOLDERS,
    )
    placeholders = frozenset(_argv_placeholders(prefix, "GPU7 guardian prefix"))
    missing = _GUARDIAN_REQUIRED_PLACEHOLDERS - placeholders
    if missing:
        raise ExecutorSpecError(
            "GPU7 guardian prefix must bind claim_id, work_id, and guardian_receipt"
        )
    if prefix[-1] not in _GUARDIAN_COMMAND_MARKERS:
        raise ExecutorSpecError(
            "GPU7 guardian prefix must end with -- or --command-json"
        )
    hash_bound = False
    for index, part in enumerate(prefix):
        if (
            part in _GUARDIAN_HASH_FLAGS
            and index + 1 < len(prefix)
            and _is_sha256(prefix[index + 1])
        ):
            hash_bound = True
            break
        for flag in _GUARDIAN_HASH_FLAGS:
            marker = flag + "="
            if part.startswith(marker) and _is_sha256(part[len(marker) :]):
                hash_bound = True
                break
        if hash_bound:
            break
    if not hash_bound:
        raise ExecutorSpecError(
            "GPU7 guardian prefix must include a literal specification hash"
        )
    return prefix


def _validate_guarded_work_argv(
    argv: Tuple[str, ...],
    guardian_prefix: Optional[Tuple[str, ...]],
) -> None:
    if guardian_prefix is None:
        raise ExecutorSpecError(
            "GPU7-eligible work without a direct lease role requires a "
            "configured guardian prefix"
        )
    prefix_length = len(guardian_prefix)
    if argv[:prefix_length] != guardian_prefix:
        raise ExecutorSpecError(
            "GPU7-eligible work argv does not begin with the configured "
            "hash-bound guardian prefix"
        )
    child_argv = argv[prefix_length:]
    if guardian_prefix[-1] == "--command-json":
        if len(child_argv) != 1:
            raise ExecutorSpecError(
                "GPU7 guardian --command-json requires exactly one JSON argv value"
            )
        try:
            decoded = json.loads(child_argv[0])
        except json.JSONDecodeError as exc:
            raise ExecutorSpecError("GPU7 guardian command JSON is malformed") from exc
        _validate_argv(decoded, "GPU7 guardian child argv", _MAIN_PLACEHOLDERS)
    elif not child_argv:
        raise ExecutorSpecError("GPU7 guardian child argv must not be empty")


def _expand_argv(
    template: Sequence[str], values: Mapping[str, str], role: str
) -> Tuple[str, ...]:
    try:
        expanded = tuple(part.format_map(values) for part in template)
    except (KeyError, ValueError) as exc:
        raise ExecutorStateError(f"cannot expand {role}: {exc}") from exc
    return _validate_argv(expanded, role, frozenset(values), expanded=True)


def load_executor_spec(
    path: Path, *, expected_spec_sha256: Optional[str] = None
) -> ExecutorSpec:
    """Load and completely validate a canonical, self-hashed executor spec."""

    source = Path(path)
    try:
        raw = _load_canonical_object(source, "cluster executor specification")
    except ExecutorStateError as exc:
        raise ExecutorSpecError(str(exc)) from exc
    raw_keys = set(raw)
    if (
        not _EXECUTOR_SPEC_REQUIRED_KEYS.issubset(raw_keys)
        or raw_keys - _EXECUTOR_SPEC_REQUIRED_KEYS - _EXECUTOR_SPEC_OPTIONAL_KEYS
    ):
        raise ExecutorSpecError("executor specification fields differ from schema")
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise ExecutorSpecError("unsupported executor specification schema")
    if raw["contract"] != EXECUTOR_SPEC_CONTRACT:
        raise ExecutorSpecError("executor specification contract is unsupported")
    body = dict(raw)
    supplied_hash = _require_sha256(
        body.pop("spec_sha256"), "executor specification identity"
    )
    if canonical_sha256(body) != supplied_hash:
        raise ExecutorSpecError("executor specification self-hash is invalid")
    if (
        expected_spec_sha256 is not None
        and _require_sha256(expected_spec_sha256, "expected specification identity")
        != supplied_hash
    ):
        raise ExecutorSpecError("executor specification identity is not expected")

    scheduler_directory = _normalized_absolute(
        raw["scheduler_directory"], "scheduler directory"
    )
    state_directory = _normalized_absolute(
        raw["state_directory"], "executor state directory"
    )
    if scheduler_directory == state_directory:
        raise ExecutorSpecError(
            "scheduler and executor state directories must be distinct"
        )
    owner_id = _identifier(raw["owner_id"], "owner_id")
    gpu_values = raw["gpu_ids"]
    if not isinstance(gpu_values, list) or not gpu_values:
        raise ExecutorSpecError("gpu_ids must be a nonempty array")
    gpu_ids = tuple(_identifier(value, "gpu_id") for value in gpu_values)
    if gpu_ids != tuple(sorted(set(gpu_ids))):
        raise ExecutorSpecError("gpu_ids must be sorted and unique")
    raw_gpu7 = raw["gpu7_id"]
    if raw_gpu7 is None:
        gpu7_id = None
    else:
        gpu7_id = _identifier(raw_gpu7, "gpu7_id")
        if gpu7_id not in gpu_ids:
            raise ExecutorSpecError("gpu7_id must belong to the fixed GPU inventory")

    poll = _number(raw["poll_interval_seconds"], "poll interval", positive=True)
    heartbeat = _number(
        raw["heartbeat_interval_seconds"], "heartbeat interval", positive=True
    )
    stale = _number(raw["stale_after_seconds"], "stale-owner interval", positive=True)
    if heartbeat > stale or poll > stale:
        raise ExecutorSpecError(
            "poll and heartbeat intervals must not exceed stale_after_seconds"
        )
    retry_budget = raw["retry_budget"]
    if (
        isinstance(retry_budget, bool)
        or not isinstance(retry_budget, int)
        or retry_budget < 0
    ):
        raise ExecutorSpecError("retry_budget must be a nonnegative integer")
    initial = _number(raw["backoff_initial_seconds"], "initial backoff", positive=True)
    maximum = _number(raw["backoff_max_seconds"], "maximum backoff", positive=True)
    if maximum < initial:
        raise ExecutorSpecError(
            "backoff_max_seconds must not be below backoff_initial_seconds"
        )
    raw_lease_command = raw["lease_proof_command"]
    lease_command = (
        None
        if raw_lease_command is None
        else _validate_argv(
            raw_lease_command, "lease proof command", _LEASE_PLACEHOLDERS
        )
    )
    lease_timeout = _number(
        raw["lease_proof_timeout_seconds"],
        "lease proof timeout",
        positive=True,
    )
    guardian_prefix = _validate_guardian_prefix(raw.get("gpu7_guardian_prefix"))
    return ExecutorSpec(
        scheduler_directory=scheduler_directory,
        state_directory=state_directory,
        owner_id=owner_id,
        gpu_ids=gpu_ids,
        gpu7_id=gpu7_id,
        poll_interval_seconds=poll,
        heartbeat_interval_seconds=heartbeat,
        stale_after_seconds=stale,
        retry_budget=retry_budget,
        backoff_initial_seconds=initial,
        backoff_max_seconds=maximum,
        lease_proof_command=lease_command,
        lease_proof_timeout_seconds=lease_timeout,
        spec_sha256=supplied_hash,
        gpu7_guardian_prefix=guardian_prefix,
        source_path=source.resolve(),
    )


def _work_execution_spec(
    record: WorkRecord,
    inventory: Sequence[str],
    gpu7_id: Optional[str],
    gpu7_guardian_prefix: Optional[Tuple[str, ...]] = None,
) -> WorkExecutionSpec:
    payload = record.payload
    raw: Any = payload.get("executor_spec")
    if raw is None:
        raw = payload.get("executor")
    if raw is None and payload.get("contract") == WORK_SPEC_CONTRACT:
        raw = payload
    if not isinstance(raw, Mapping):
        raise ExecutorSpecError(f"work {record.work_id!r} has no executor_spec object")
    if set(raw) != _WORK_SPEC_KEYS:
        raise ExecutorSpecError("work executor specification fields differ from schema")
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise ExecutorSpecError("unsupported work specification schema")
    if raw["contract"] != WORK_SPEC_CONTRACT:
        raise ExecutorSpecError("work specification contract is unsupported")
    body = dict(raw)
    supplied_hash = _require_sha256(
        body.pop("spec_sha256"), "work specification identity"
    )
    if canonical_sha256(body) != supplied_hash:
        raise ExecutorSpecError("work specification self-hash is invalid")
    work_id = _identifier(raw["work_id"], "work spec work_id")
    if work_id != record.work_id:
        raise ExecutorSpecError("work specification work_id contradicts scheduler")
    try:
        kind = WorkKind(raw["kind"])
    except (TypeError, ValueError) as exc:
        raise ExecutorSpecError("work specification kind is unknown") from exc
    if kind != record.kind:
        raise ExecutorSpecError("work specification kind contradicts scheduler")
    raw_eligible = raw["eligible_gpus"]
    if not isinstance(raw_eligible, list) or not raw_eligible:
        raise ExecutorSpecError("work eligible_gpus must be a nonempty array")
    eligible = tuple(_identifier(value, "eligible GPU") for value in raw_eligible)
    if eligible != tuple(sorted(set(eligible))):
        raise ExecutorSpecError("work eligible_gpus must be sorted and unique")
    if not set(eligible).issubset(inventory):
        raise ExecutorSpecError("work names a GPU outside the fixed inventory")
    if record.eligible_gpus is None or tuple(record.eligible_gpus) != eligible:
        raise ExecutorSpecError("work eligible_gpus contradict the scheduler work item")
    argv = _validate_argv(raw["argv"], "work argv", _MAIN_PLACEHOLDERS)
    cwd = _normalized_absolute(raw["cwd"], "work cwd", require_directory=True)
    raw_environment = raw["environment"]
    if not isinstance(raw_environment, Mapping):
        raise ExecutorSpecError("work environment must be an object")
    environment: Dict[str, str] = {}
    for key, value in raw_environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\x00" in key
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ExecutorSpecError(
                "work environment keys and values must be safe strings"
            )
        environment[key] = value
    if "CUDA_VISIBLE_DEVICES" in environment:
        raise ExecutorSpecError("CUDA_VISIBLE_DEVICES is reserved for the executor")
    lease_role = raw["lease_role"]
    if lease_role not in _LEASE_ROLES:
        raise ExecutorSpecError("work lease_role is unsupported")
    if kind == WorkKind.TRAINER and lease_role != "trainer":
        raise ExecutorSpecError("trainer work requires the trainer GPU lease role")
    if kind in _EVALUATOR_KINDS and lease_role != "evaluator":
        raise ExecutorSpecError("evaluation work requires the evaluator GPU lease role")
    if lease_role != "none":
        if gpu7_id is None or eligible != (gpu7_id,):
            raise ExecutorSpecError(
                "trainer/evaluator work must be pinned exclusively to GPU7"
            )
    elif gpu7_id is not None and gpu7_id in eligible:
        if eligible != (gpu7_id,):
            raise ExecutorSpecError(
                "GPU7 guardian work must be pinned exclusively to GPU7"
            )
        _validate_guarded_work_argv(argv, gpu7_guardian_prefix)

    safe_drain_value = raw["safe_drain"]
    if safe_drain_value is None:
        safe_drain = None
    else:
        if (
            not isinstance(safe_drain_value, Mapping)
            or set(safe_drain_value) != _SAFE_DRAIN_KEYS
        ):
            raise ExecutorSpecError("safe_drain fields differ from schema")
        command = _validate_argv(
            safe_drain_value["command"],
            "safe drain command",
            _DRAIN_PLACEHOLDERS,
        )
        checkpoint = _normalized_absolute(
            safe_drain_value["checkpoint_path"],
            "safe drain checkpoint",
            require_file=True,
        )
        require_change = safe_drain_value["require_checkpoint_change"]
        if not isinstance(require_change, bool):
            raise ExecutorSpecError(
                "safe drain require_checkpoint_change must be boolean"
            )
        timeout = _number(
            safe_drain_value["timeout_seconds"],
            "safe drain timeout",
            positive=True,
        )
        safe_drain = SafeDrainSpec(command, checkpoint, require_change, timeout)
    return WorkExecutionSpec(
        work_id=work_id,
        kind=kind,
        eligible_gpus=eligible,
        argv=argv,
        cwd=cwd,
        environment=MappingProxyType(dict(sorted(environment.items()))),
        lease_role=lease_role,
        safe_drain=safe_drain,
        spec_sha256=supplied_hash,
    )


_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.Lock] = {}


def _process_lock(path: Path) -> threading.Lock:
    key = os.fspath(path)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _state_name(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest() + ".json"


def _checkpoint_identity(path: Path) -> Optional[Dict[str, Any]]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        after = path.lstat()
    except OSError:
        return None
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_key != after_key:
        return None
    return {
        "device": after.st_dev,
        "inode": after.st_ino,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


class ClusterExecutor:
    """Execute scheduler-selected work without becoming a scheduling authority."""

    def __init__(
        self,
        spec: Union[ExecutorSpec, Path, str],
        *,
        expected_spec_sha256: Optional[str] = None,
        scheduler: Optional[ClusterScheduler] = None,
        process_runner: Optional[ProcessRunner] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        lease_proof: Optional[LeaseProofCallback] = None,
        checkpoint_reader: Callable[[Path], Optional[Mapping[str, Any]]] = (
            _checkpoint_identity
        ),
    ) -> None:
        self.spec = (
            spec
            if isinstance(spec, ExecutorSpec)
            else load_executor_spec(
                Path(spec), expected_spec_sha256=expected_spec_sha256
            )
        )
        if (
            isinstance(spec, ExecutorSpec)
            and expected_spec_sha256 is not None
            and self.spec.spec_sha256 != expected_spec_sha256
        ):
            raise ExecutorSpecError("executor specification identity is not expected")
        self._clock = clock
        self._sleep = sleep
        self.runner = process_runner or SubprocessRunner()
        self._lease_proof = lease_proof
        self._checkpoint_reader = checkpoint_reader
        self._ensure_state_directory()
        self.scheduler = scheduler or ClusterScheduler(
            self.spec.scheduler_directory,
            self.spec.gpu_ids,
            clock=clock,
        )
        if Path(self.scheduler.directory) != self.spec.scheduler_directory:
            raise ExecutorSpecError(
                "scheduler instance uses a different durable directory"
            )
        snapshot = self.scheduler.reconstruct()
        if snapshot.dynamic_gpus or snapshot.gpu_ids != self.spec.gpu_ids:
            raise ExecutorSpecError(
                "executor requires the exact fixed scheduler GPU inventory"
            )

    @property
    def state_directory(self) -> Path:
        return self.spec.state_directory

    @property
    def lock_path(self) -> Path:
        return self.state_directory / LOCK_FILENAME

    def once(self) -> Mapping[str, Any]:
        """Run one bounded reconciliation pass under the local executor lock."""

        with self._exclusive_lock():
            now = self._now()
            self._write_heartbeat(now, "reconciling")
            recovered = self._recover_release_intents(now)
            gpu_results = []
            for gpu_id in self.spec.gpu_ids:
                try:
                    gpu_results.append(self._reconcile_gpu(gpu_id, now))
                except (ClusterExecutorError, SchedulerError, OSError) as exc:
                    gpu_results.append(
                        self._halt(
                            gpu_id,
                            f"executor reconciliation failed: {exc}",
                            now=now,
                        )
                    )
            self._write_heartbeat(now, "healthy")
            return {
                "schema_version": SCHEMA_VERSION,
                "contract": ONCE_CONTRACT,
                "owner_id": self.spec.owner_id,
                "executor_spec_sha256": self.spec.spec_sha256,
                "observed_at_unix": now,
                "recovered_releases": recovered,
                "gpus": gpu_results,
            }

    reconcile_once = once

    def status(self) -> Mapping[str, Any]:
        """Return a read-only scheduler/executor status snapshot."""

        now = self._now()
        snapshot = self.scheduler.reconstruct()
        gpus = []
        for gpu_id in self.spec.gpu_ids:
            claim = snapshot.claims.get(gpu_id)
            if claim is None:
                queued = [
                    record.work_id
                    for record in snapshot.queued
                    if record.item.is_eligible(gpu_id)
                ]
                gpus.append(
                    {
                        "gpu_id": gpu_id,
                        "state": "idle" if not queued else "queued",
                        "claim": None,
                        "queued_work": queued,
                    }
                )
                continue
            start = _load_optional_object(
                self._start_receipt_path(claim.claim_id), "start receipt"
            )
            process_state = "not-started"
            if start is not None:
                try:
                    identity = self._identity_from_start(start)
                    current = self.runner.current_identity(identity.pid)
                    if current is None:
                        process_state = "absent"
                    else:
                        current_identity = ProcessIdentity.from_value(
                            current, "observed process"
                        )
                        process_state = (
                            "running"
                            if identity.same_process_as(current_identity)
                            else "identity-mismatch"
                        )
                except ClusterExecutorError:
                    process_state = "invalid-receipt"
            gpus.append(
                {
                    "gpu_id": gpu_id,
                    "state": process_state,
                    "claim": claim.to_dict(),
                    "owner_stale": self._owner_is_stale(claim, now),
                }
            )
        quarantines = []
        quarantine_root = self.state_directory / "quarantine"
        if quarantine_root.is_dir():
            for path in sorted(quarantine_root.glob("*.json")):
                value = _load_canonical_object(path, "quarantine receipt")
                _validate_self_hash(value, "receipt_sha256", "quarantine receipt")
                quarantines.append(value)
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": STATUS_CONTRACT,
            "owner_id": self.spec.owner_id,
            "executor_spec_sha256": self.spec.spec_sha256,
            "observed_at_unix": now,
            "scheduler_revision": snapshot.revision,
            "scheduler_state_sha256": snapshot.state_sha256,
            "safety_halt": snapshot.safety_halt,
            "gpu_safety_halts": dict(snapshot.gpu_safety_halts),
            "gpus": gpus,
            "quarantines": quarantines,
        }

    def watch(self) -> Iterator[Mapping[str, Any]]:
        """Yield reconciliation snapshots forever at the configured interval."""

        while True:
            yield self.once()
            self._sleep(
                min(
                    self.spec.poll_interval_seconds,
                    self.spec.heartbeat_interval_seconds,
                )
            )

    def _reconcile_gpu(self, gpu_id: str, now: float) -> Mapping[str, Any]:
        claim = self.scheduler.get_claim(gpu_id)
        if claim is not None:
            return self._reconcile_claim(claim, now)

        queued = self.scheduler.queued_work(gpu_id)
        if not queued:
            self.scheduler.claim(gpu_id, self.spec.owner_id)
            return {"gpu_id": gpu_id, "status": "idle"}
        selected = queued[0]
        try:
            work_spec = _work_execution_spec(
                selected,
                self.spec.gpu_ids,
                self.spec.gpu7_id,
                self.spec.gpu7_guardian_prefix,
            )
        except ExecutorSpecError as exc:
            return self._halt(
                gpu_id,
                f"queued work specification is invalid: {exc}",
                now=now,
            )
        quarantine = _load_optional_object(
            self._quarantine_path(selected.work_id), "quarantine receipt"
        )
        if quarantine is not None:
            return self._halt(
                gpu_id,
                f"queued work {selected.work_id!r} is quarantined",
                now=now,
            )
        retry = self._load_retry(selected.work_id)
        if (
            retry is not None
            and retry.get("status") == "backoff"
            and float(retry["not_before_unix"]) > now
        ):
            return {
                "gpu_id": gpu_id,
                "status": "backoff",
                "work_id": selected.work_id,
                "not_before_unix": retry["not_before_unix"],
            }

        claim = self.scheduler.claim(gpu_id, self.spec.owner_id)
        if claim is None:
            return {"gpu_id": gpu_id, "status": "idle"}
        selected = self.scheduler.get_work(claim.work_id)
        if selected is None:
            return self._halt(
                gpu_id, "scheduler claim names missing work", claim=claim, now=now
            )
        work_spec = _work_execution_spec(
            selected,
            self.spec.gpu_ids,
            self.spec.gpu7_id,
            self.spec.gpu7_guardian_prefix,
        )
        self._write_claim_receipt(claim, work_spec, now)
        return self._gate_and_start(claim, selected, work_spec, now)

    def _reconcile_claim(self, claim: Claim, now: float) -> Mapping[str, Any]:
        record = self.scheduler.get_work(claim.work_id)
        if (
            record is None
            or record.state != WorkState.CLAIMED
            or record.active_claim_id != claim.claim_id
        ):
            return self._halt(
                claim.gpu_id,
                "scheduler claim contradicts its work record",
                claim=claim,
                now=now,
            )
        try:
            work_spec = _work_execution_spec(
                record,
                self.spec.gpu_ids,
                self.spec.gpu7_id,
                self.spec.gpu7_guardian_prefix,
            )
        except ExecutorSpecError as exc:
            return self._halt(
                claim.gpu_id,
                f"claimed work specification is invalid: {exc}",
                claim=claim,
                now=now,
            )

        claim_receipt = _load_optional_object(
            self._claim_receipt_path(claim.claim_id), "claim receipt"
        )
        foreign_owner = claim.owner_id != self.spec.owner_id
        if foreign_owner and not self._owner_is_stale(claim, now):
            return {
                "gpu_id": claim.gpu_id,
                "status": "foreign-owner-active",
                "claim_id": claim.claim_id,
                "owner_id": claim.owner_id,
            }
        if claim_receipt is None:
            if foreign_owner:
                return self._halt(
                    claim.gpu_id,
                    "stale foreign claim has no executor claim receipt",
                    claim=claim,
                    now=now,
                )
            claim_receipt = self._write_claim_receipt(claim, work_spec, now)
        else:
            self._validate_claim_receipt(claim_receipt, claim, work_spec)

        start = _load_optional_object(
            self._start_receipt_path(claim.claim_id), "start receipt"
        )
        start_intent = _load_optional_object(
            self._start_intent_path(claim.claim_id), "start intent"
        )
        if start is None:
            if start_intent is not None:
                return self._halt(
                    claim.gpu_id,
                    "start intent has no process identity; duplicate launch refused",
                    claim=claim,
                    now=now,
                )
            if foreign_owner:
                release = self._release_claim(
                    claim,
                    work_spec,
                    ReleaseOutcome.REQUEUE,
                    "stale-owner-before-start",
                    now,
                )
                return {
                    "gpu_id": claim.gpu_id,
                    "status": "stale-owner-requeued",
                    "claim_id": claim.claim_id,
                    "release_id": release.release_id,
                }
            return self._gate_and_start(claim, record, work_spec, now)

        identity = self._identity_from_start(start)
        self._validate_start_receipt(start, claim, work_spec)
        completion = _load_optional_object(
            self._completion_path(claim.claim_id), "completion receipt"
        )
        drain_command = _load_optional_object(
            self._drain_command_path(claim.claim_id), "drain command receipt"
        )
        observed_returncode = self.runner.returncode(identity)
        if completion is None and observed_returncode is not None:
            if isinstance(observed_returncode, bool) or not isinstance(
                observed_returncode, int
            ):
                return self._halt(
                    claim.gpu_id,
                    "process runner returned a malformed return code",
                    claim=claim,
                    now=now,
                )
            completion = self._write_completion(
                claim, work_spec, start, observed_returncode, now
            )

        current_value = self.runner.current_identity(identity.pid)
        current = (
            None
            if current_value is None
            else ProcessIdentity.from_value(current_value, "observed process")
        )
        if current is not None and not identity.same_process_as(current):
            return self._halt(
                claim.gpu_id,
                "process identity mismatch; no command or signal was issued",
                claim=claim,
                now=now,
            )

        if completion is not None:
            self._validate_completion(completion, claim, work_spec, start)
            if current is not None:
                return self._halt(
                    claim.gpu_id,
                    "completion receipt exists while its exact process is live",
                    claim=claim,
                    now=now,
                )
            if self.runner.process_group_alive(identity):
                return self._halt(
                    claim.gpu_id,
                    "completed leader is absent but its process group remains live",
                    claim=claim,
                    now=now,
                )
            if _requires_gpu7_guardian(work_spec, self.spec.gpu7_id):
                try:
                    self._validate_guardian_completion(
                        claim,
                        work_spec,
                        int(completion["returncode"]),
                    )
                except (
                    ClusterExecutorError,
                    OSError,
                    gpu_lease_worker.GpuLeaseWorkerError,
                ) as exc:
                    return self._halt(
                        claim.gpu_id,
                        f"GPU7 guardian completion is unsafe: {exc}",
                        claim=claim,
                        now=now,
                    )
            if drain_command is not None:
                return self._finish_drain(
                    claim, work_spec, identity, drain_command, now
                )
            return self._finalize_completion(claim, work_spec, completion, now)

        if current is None:
            if self.runner.process_group_alive(identity):
                return self._halt(
                    claim.gpu_id,
                    "process leader disappeared while its group remains live",
                    claim=claim,
                    now=now,
                )
            if _requires_gpu7_guardian(work_spec, self.spec.gpu7_id):
                return self._halt(
                    claim.gpu_id,
                    "GPU7 guardian disappeared without a completion receipt",
                    claim=claim,
                    now=now,
                )
            if drain_command is not None:
                return self._finish_drain(
                    claim, work_spec, identity, drain_command, now
                )
            return self._failure(
                claim,
                work_spec,
                "process-disappeared-without-completion",
                now,
            )

        if work_spec.lease_role != "none":
            try:
                self._require_lease_proof(claim, record, work_spec, now)
            except LeaseProofDenied as exc:
                return self._halt(
                    claim.gpu_id,
                    f"GPU7 lease authority no longer proves the active process: {exc}",
                    claim=claim,
                    now=now,
                )
        pending = tuple(
            request
            for request in self.scheduler.pending_preemptions(claim.gpu_id)
            if request.claim_id == claim.claim_id
        )
        if drain_command is not None:
            return self._finish_drain(claim, work_spec, identity, drain_command, now)
        if pending:
            return self._begin_drain(
                claim, work_spec, identity, pending[0].request_id, now
            )
        return {
            "gpu_id": claim.gpu_id,
            "status": ("stale-owner-process-running" if foreign_owner else "running"),
            "claim_id": claim.claim_id,
            "work_id": claim.work_id,
            "process_identity": identity.to_dict(),
        }

    def _gate_and_start(
        self,
        claim: Claim,
        record: WorkRecord,
        work_spec: WorkExecutionSpec,
        now: float,
    ) -> Mapping[str, Any]:
        try:
            lease_proof = self._require_lease_proof(claim, record, work_spec, now)
        except LeaseProofDenied as exc:
            self.scheduler.record_idle(
                claim.gpu_id,
                IdleReason.LEASE_HANDOFF,
                owner_id=claim.owner_id,
                work_id=claim.work_id,
                details={"reason": str(exc), "claim_id": claim.claim_id},
            )
            return {
                "gpu_id": claim.gpu_id,
                "status": "lease-denied",
                "claim_id": claim.claim_id,
                "work_id": claim.work_id,
                "reason": str(exc),
            }

        log_path = (
            self.state_directory / "logs" / (_state_name(claim.claim_id)[:-5] + ".log")
        )
        values = {
            "claim_id": claim.claim_id,
            "guardian_receipt": os.fspath(self._guardian_receipt_path(claim.claim_id)),
            "gpu_id": claim.gpu_id,
            "log_path": os.fspath(log_path),
            "state_directory": os.fspath(self.state_directory),
            "work_id": claim.work_id,
        }
        argv = _expand_argv(work_spec.argv, values, "work argv")
        guardian_required = _requires_gpu7_guardian(work_spec, self.spec.gpu7_id)
        expanded_guardian_prefix: Optional[Tuple[str, ...]] = None
        if guardian_required:
            if self.spec.gpu7_guardian_prefix is None:
                raise ExecutorStateError(
                    "GPU7 guardian prefix disappeared before launch"
                )
            expanded_guardian_prefix = _expand_argv(
                self.spec.gpu7_guardian_prefix,
                values,
                "GPU7 guardian prefix",
            )
            if argv[: len(expanded_guardian_prefix)] != expanded_guardian_prefix:
                raise ExecutorStateError(
                    "expanded work argv contradicts the GPU7 guardian prefix"
                )
        environment = dict(work_spec.environment)
        environment["CUDA_VISIBLE_DEVICES"] = claim.gpu_id
        start_intent = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": START_INTENT_CONTRACT,
                "claim": claim.to_dict(),
                "executor_spec_sha256": self.spec.spec_sha256,
                "work_spec_sha256": work_spec.spec_sha256,
                "argv": list(argv),
                "argv_sha256": canonical_sha256(list(argv)),
                "cwd": os.fspath(work_spec.cwd),
                "environment": environment,
                "environment_sha256": canonical_sha256(environment),
                "log_path": os.fspath(log_path),
                "lease_proof": lease_proof,
                "lease_proof_sha256": (
                    None if lease_proof is None else canonical_sha256(lease_proof)
                ),
                "guardian_receipt_path": (
                    values["guardian_receipt"] if guardian_required else None
                ),
                "gpu7_guardian_prefix_sha256": (
                    None
                    if expanded_guardian_prefix is None
                    else canonical_sha256(list(expanded_guardian_prefix))
                ),
                "requested_at_unix": now,
            },
            "intent_sha256",
        )
        _atomic_write_json(self._start_intent_path(claim.claim_id), start_intent)
        try:
            identity_value = self.runner.spawn(
                argv,
                cwd=work_spec.cwd,
                environment=environment,
                log_path=log_path,
            )
            identity = ProcessIdentity.from_value(identity_value, "spawned process")
        except ProcessSpawnError as exc:
            if not exc.safe_to_retry:
                return self._halt(
                    claim.gpu_id,
                    f"ambiguous process launch: {exc}",
                    claim=claim,
                    now=now,
                )
            return self._failure(claim, work_spec, f"spawn-failed: {exc}", now)
        except OSError as exc:
            return self._failure(claim, work_spec, f"spawn-failed: {exc}", now)
        except ClusterExecutorError as exc:
            return self._halt(
                claim.gpu_id,
                f"spawned process identity is unsafe: {exc}",
                claim=claim,
                now=now,
            )
        if identity.process_group_id != identity.pid:
            return self._halt(
                claim.gpu_id,
                "spawned process is not the leader of its new process group",
                claim=claim,
                now=now,
            )
        start = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": START_RECEIPT_CONTRACT,
                "claim_id": claim.claim_id,
                "work_id": claim.work_id,
                "gpu_id": claim.gpu_id,
                "owner_id": claim.owner_id,
                "executor_spec_sha256": self.spec.spec_sha256,
                "work_spec_sha256": work_spec.spec_sha256,
                "start_intent_sha256": start_intent["intent_sha256"],
                "process_identity": identity.to_dict(),
                "process_identity_sha256": canonical_sha256(identity.to_dict()),
                "started_at_unix": now,
            },
            "receipt_sha256",
        )
        _atomic_write_json(self._start_receipt_path(claim.claim_id), start)
        current_value = self.runner.current_identity(identity.pid)
        if current_value is None or not identity.same_process_as(
            ProcessIdentity.from_value(current_value, "spawn verification")
        ):
            return self._halt(
                claim.gpu_id,
                "spawned process identity was not stable after receipt publication",
                claim=claim,
                now=now,
            )
        return {
            "gpu_id": claim.gpu_id,
            "status": "started",
            "claim_id": claim.claim_id,
            "work_id": claim.work_id,
            "process_identity": identity.to_dict(),
        }

    def _require_lease_proof(
        self,
        claim: Claim,
        record: WorkRecord,
        work_spec: WorkExecutionSpec,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if work_spec.lease_role == "none":
            return None
        if self.spec.gpu7_id is None or claim.gpu_id != self.spec.gpu7_id:
            raise LeaseProofDenied("GPU7 lease role was claimed on a different GPU")
        request = LeaseRequest(
            work_id=record.work_id,
            kind=record.kind.value,
            gpu_id=claim.gpu_id,
            lease_role=work_spec.lease_role,
            claim_id=claim.claim_id,
            executor_spec_sha256=self.spec.spec_sha256,
            work_spec_sha256=work_spec.spec_sha256,
        )
        proof_value: Union[bool, Mapping[str, Any]]
        error: Optional[str] = None
        try:
            if self._lease_proof is not None:
                proof_value = self._lease_proof(request)
            elif self.spec.lease_proof_command is not None:
                values = request.to_dict()
                command = _expand_argv(
                    self.spec.lease_proof_command,
                    values,
                    "lease proof command",
                )
                result = self.runner.run(
                    command,
                    cwd=self.state_directory,
                    environment={},
                    timeout=self.spec.lease_proof_timeout_seconds,
                )
                if result.returncode != 0:
                    raise LeaseProofDenied(
                        f"lease proof command returned {result.returncode}"
                    )
                encoded = result.stdout.encode("utf-8")
                try:
                    proof_value = json.loads(
                        result.stdout,
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                    )
                except (json.JSONDecodeError, ExecutorStateError) as exc:
                    raise LeaseProofDenied(
                        "lease proof command output is invalid JSON"
                    ) from exc
                if (
                    not isinstance(proof_value, Mapping)
                    or encoded != canonical_json_bytes(proof_value) + b"\n"
                ):
                    raise LeaseProofDenied(
                        "lease proof command output is not canonical JSON"
                    )
            else:
                raise LeaseProofDenied(
                    "trainer/evaluator launch has no GPU7 lease authority"
                )
            if proof_value is False:
                raise LeaseProofDenied("GPU7 lease authority denied the launch")
            if proof_value is True:
                raise LeaseProofDenied("GPU7 lease proof lacks renewable lease fields")
            if isinstance(proof_value, Mapping):
                proof = json.loads(
                    canonical_json_bytes(dict(proof_value)).decode("utf-8")
                )
                if proof.get("allowed") is not True:
                    raise LeaseProofDenied("GPU7 lease authority denied the launch")
            else:
                raise LeaseProofDenied("GPU7 lease proof is malformed")
            lease_id = proof.get("lease_id")
            if (
                not isinstance(lease_id, str)
                or not lease_id
                or lease_id != lease_id.strip()
                or "\x00" in lease_id
            ):
                raise LeaseProofDenied("GPU7 lease proof has no valid lease_id")
            if proof.get("claim_id") != request.claim_id:
                raise LeaseProofDenied("GPU7 lease proof contradicts claim_id")
            if proof.get("work_id") != request.work_id:
                raise LeaseProofDenied("GPU7 lease proof contradicts work_id")
            valid_until = proof.get("valid_until_unix")
            if (
                isinstance(valid_until, bool)
                or not isinstance(valid_until, (int, float))
                or not math.isfinite(float(valid_until))
                or float(valid_until) <= now
            ):
                raise LeaseProofDenied(
                    "GPU7 lease proof validity horizon is absent or expired"
                )
            next_reconcile = now + min(
                self.spec.poll_interval_seconds,
                self.spec.heartbeat_interval_seconds,
            )
            if float(valid_until) <= next_reconcile:
                raise LeaseProofDenied(
                    "GPU7 lease proof validity horizon does not cover the next "
                    "reconcile"
                )
            for key, expected in request.to_dict().items():
                if key in proof and proof[key] != expected:
                    raise LeaseProofDenied(f"GPU7 lease proof contradicts {key}")
            start_intent = _load_optional_object(
                self._start_intent_path(claim.claim_id), "start intent"
            )
            if start_intent is not None:
                _validate_self_hash(start_intent, "intent_sha256", "start intent")
                initial_proof = start_intent.get("lease_proof")
                if (
                    not isinstance(initial_proof, Mapping)
                    or initial_proof.get("lease_id") != lease_id
                ):
                    raise LeaseProofDenied(
                        "GPU7 renewable proof changed the active lease_id"
                    )
        except LeaseProofDenied as exc:
            error = str(exc)
            gate = _self_hashed(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": LEASE_GATE_CONTRACT,
                    "request": request.to_dict(),
                    "allowed": False,
                    "proof": None,
                    "error": error,
                    "checked_at_unix": now,
                },
                "receipt_sha256",
            )
            _atomic_replace_json(self._lease_gate_path(claim.claim_id), gate)
            raise
        except Exception as exc:
            error = f"lease authority failed: {exc}"
            gate = _self_hashed(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": LEASE_GATE_CONTRACT,
                    "request": request.to_dict(),
                    "allowed": False,
                    "proof": None,
                    "error": error,
                    "checked_at_unix": now,
                },
                "receipt_sha256",
            )
            _atomic_replace_json(self._lease_gate_path(claim.claim_id), gate)
            raise LeaseProofDenied(error) from exc
        gate = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": LEASE_GATE_CONTRACT,
                "request": request.to_dict(),
                "allowed": True,
                "proof": proof,
                "error": None,
                "checked_at_unix": now,
            },
            "receipt_sha256",
        )
        _atomic_replace_json(self._lease_gate_path(claim.claim_id), gate)
        return proof

    def _begin_drain(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        identity: ProcessIdentity,
        request_id: str,
        now: float,
    ) -> Mapping[str, Any]:
        safe_drain = work_spec.safe_drain
        if safe_drain is None:
            return {
                "gpu_id": claim.gpu_id,
                "status": "preemption-waiting-for-safe-boundary",
                "claim_id": claim.claim_id,
                "request_id": request_id,
            }
        before = self._checkpoint_reader(safe_drain.checkpoint_path)
        if not isinstance(before, Mapping):
            return self._halt(
                claim.gpu_id,
                "safe preemption checkpoint cannot be identified",
                claim=claim,
                now=now,
            )
        values = {
            "claim_id": claim.claim_id,
            "gpu_id": claim.gpu_id,
            "log_path": os.fspath(
                self.state_directory
                / "logs"
                / (_state_name(claim.claim_id)[:-5] + ".log")
            ),
            "state_directory": os.fspath(self.state_directory),
            "work_id": claim.work_id,
            "checkpoint_path": os.fspath(safe_drain.checkpoint_path),
            **{key: str(value) for key, value in identity.to_dict().items()},
        }
        command = _expand_argv(safe_drain.command, values, "safe drain command")
        intent = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": DRAIN_INTENT_CONTRACT,
                "claim_id": claim.claim_id,
                "work_id": claim.work_id,
                "gpu_id": claim.gpu_id,
                "request_id": request_id,
                "work_spec_sha256": work_spec.spec_sha256,
                "process_identity": identity.to_dict(),
                "process_identity_sha256": canonical_sha256(identity.to_dict()),
                "command": list(command),
                "command_sha256": canonical_sha256(list(command)),
                "checkpoint_before": dict(before),
                "checkpoint_before_sha256": canonical_sha256(before),
                "requested_at_unix": now,
                "deadline_unix": now + safe_drain.timeout_seconds,
            },
            "intent_sha256",
        )
        intent_path = self._drain_intent_path(claim.claim_id)
        command_path = self._drain_command_path(claim.claim_id)
        existing_intent = _load_optional_object(intent_path, "drain intent")
        if existing_intent is not None:
            if _load_optional_object(command_path, "drain command receipt") is None:
                return self._halt(
                    claim.gpu_id,
                    "drain intent has no command receipt; duplicate command refused",
                    claim=claim,
                    now=now,
                )
            return self._finish_drain(
                claim,
                work_spec,
                identity,
                _load_canonical_object(command_path, "drain command receipt"),
                now,
            )
        _atomic_write_json(intent_path, intent)
        current_value = self.runner.current_identity(identity.pid)
        if current_value is None or not identity.same_process_as(
            ProcessIdentity.from_value(current_value, "pre-drain process")
        ):
            return self._halt(
                claim.gpu_id,
                "process identity changed before safe drain command",
                claim=claim,
                now=now,
            )
        environment = dict(work_spec.environment)
        environment["CUDA_VISIBLE_DEVICES"] = claim.gpu_id
        try:
            result = self.runner.run(
                command,
                cwd=work_spec.cwd,
                environment=environment,
                timeout=safe_drain.timeout_seconds,
            )
            command_receipt = _self_hashed(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": DRAIN_COMMAND_CONTRACT,
                    "drain_intent_sha256": intent["intent_sha256"],
                    "returncode": result.returncode,
                    "stdout_sha256": hashlib.sha256(
                        result.stdout.encode("utf-8")
                    ).hexdigest(),
                    "stderr_sha256": hashlib.sha256(
                        result.stderr.encode("utf-8")
                    ).hexdigest(),
                    "error": None,
                    "completed_at_unix": now,
                },
                "receipt_sha256",
            )
        except Exception as exc:
            command_receipt = _self_hashed(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": DRAIN_COMMAND_CONTRACT,
                    "drain_intent_sha256": intent["intent_sha256"],
                    "returncode": None,
                    "stdout_sha256": None,
                    "stderr_sha256": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at_unix": now,
                },
                "receipt_sha256",
            )
        _atomic_write_json(command_path, command_receipt)
        if command_receipt["returncode"] != 0:
            return self._halt(
                claim.gpu_id,
                "safe drain command failed; process was not signalled by executor",
                claim=claim,
                now=now,
            )
        return self._finish_drain(claim, work_spec, identity, command_receipt, now)

    def _finish_drain(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        identity: ProcessIdentity,
        command_receipt: Mapping[str, Any],
        now: float,
    ) -> Mapping[str, Any]:
        safe_drain = work_spec.safe_drain
        if safe_drain is None:
            return self._halt(
                claim.gpu_id,
                "drain receipt exists for work without a safe drain",
                claim=claim,
                now=now,
            )
        _validate_self_hash(command_receipt, "receipt_sha256", "drain command receipt")
        if (
            command_receipt.get("contract") != DRAIN_COMMAND_CONTRACT
            or command_receipt.get("returncode") != 0
        ):
            return self._halt(
                claim.gpu_id,
                "safe drain command has no successful receipt",
                claim=claim,
                now=now,
            )
        intent = _load_canonical_object(
            self._drain_intent_path(claim.claim_id), "drain intent"
        )
        _validate_self_hash(intent, "intent_sha256", "drain intent")
        if (
            intent.get("claim_id") != claim.claim_id
            or intent.get("work_spec_sha256") != work_spec.spec_sha256
            or command_receipt.get("drain_intent_sha256") != intent.get("intent_sha256")
        ):
            return self._halt(
                claim.gpu_id,
                "safe drain receipts contradict the active claim",
                claim=claim,
                now=now,
            )
        current_value = self.runner.current_identity(identity.pid)
        if current_value is not None:
            current = ProcessIdentity.from_value(current_value, "draining process")
            if not identity.same_process_as(current):
                return self._halt(
                    claim.gpu_id,
                    "process identity changed during safe drain",
                    claim=claim,
                    now=now,
                )
            if now >= float(intent["deadline_unix"]):
                return self._halt(
                    claim.gpu_id,
                    "safe drain process did not exit before its deadline",
                    claim=claim,
                    now=now,
                )
            return {
                "gpu_id": claim.gpu_id,
                "status": "draining-at-safe-boundary",
                "claim_id": claim.claim_id,
                "deadline_unix": intent["deadline_unix"],
            }
        if self.runner.process_group_alive(identity):
            return self._halt(
                claim.gpu_id,
                "drained leader is absent but its process group remains live",
                claim=claim,
                now=now,
            )
        after_value = self._checkpoint_reader(safe_drain.checkpoint_path)
        if not isinstance(after_value, Mapping):
            if now < float(intent["deadline_unix"]):
                return {
                    "gpu_id": claim.gpu_id,
                    "status": "waiting-for-checkpoint-boundary",
                    "claim_id": claim.claim_id,
                }
            return self._halt(
                claim.gpu_id,
                "safe drain checkpoint never became identifiable",
                claim=claim,
                now=now,
            )
        before = intent["checkpoint_before"]
        if safe_drain.require_checkpoint_change and after_value.get(
            "sha256"
        ) == before.get("sha256"):
            if now < float(intent["deadline_unix"]):
                return {
                    "gpu_id": claim.gpu_id,
                    "status": "waiting-for-checkpoint-change",
                    "claim_id": claim.claim_id,
                }
            return self._halt(
                claim.gpu_id,
                "safe drain checkpoint did not change before its deadline",
                claim=claim,
                now=now,
            )
        boundary = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": DRAIN_BOUNDARY_CONTRACT,
                "claim_id": claim.claim_id,
                "work_id": claim.work_id,
                "gpu_id": claim.gpu_id,
                "drain_intent_sha256": intent["intent_sha256"],
                "drain_command_sha256": command_receipt["receipt_sha256"],
                "checkpoint_after": dict(after_value),
                "checkpoint_after_sha256": canonical_sha256(after_value),
                "boundary_at_unix": now,
            },
            "receipt_sha256",
        )
        _atomic_write_json(self._drain_boundary_path(claim.claim_id), boundary)
        release = self._release_claim(
            claim,
            work_spec,
            ReleaseOutcome.REQUEUE,
            "cooperative-preemption-checkpoint-boundary",
            now,
            idle_reason=IdleReason.CHECKPOINT_BOUNDARY,
            idle_details={
                "request_id": intent["request_id"],
                "checkpoint_sha256": after_value["sha256"],
            },
        )
        return {
            "gpu_id": claim.gpu_id,
            "status": "preempted-at-safe-boundary",
            "claim_id": claim.claim_id,
            "release_id": release.release_id,
        }

    def _write_completion(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        start: Mapping[str, Any],
        returncode: int,
        now: float,
    ) -> Dict[str, Any]:
        completion = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": COMPLETION_RECEIPT_CONTRACT,
                "claim_id": claim.claim_id,
                "work_id": claim.work_id,
                "gpu_id": claim.gpu_id,
                "owner_id": claim.owner_id,
                "work_spec_sha256": work_spec.spec_sha256,
                "start_receipt_sha256": start["receipt_sha256"],
                "process_identity_sha256": start["process_identity_sha256"],
                "returncode": returncode,
                "completed_at_unix": now,
            },
            "receipt_sha256",
        )
        _atomic_write_json(self._completion_path(claim.claim_id), completion)
        return completion

    def _finalize_completion(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        completion: Mapping[str, Any],
        now: float,
    ) -> Mapping[str, Any]:
        returncode = completion["returncode"]
        if returncode == 0:
            self._write_retry(
                claim.work_id,
                failure_count=0,
                status="completed",
                last_claim_id=claim.claim_id,
                not_before=None,
                reason=None,
                now=now,
            )
            release = self._release_claim(
                claim,
                work_spec,
                ReleaseOutcome.COMPLETED,
                "process-completed",
                now,
            )
            return {
                "gpu_id": claim.gpu_id,
                "status": "completed",
                "claim_id": claim.claim_id,
                "release_id": release.release_id,
            }
        return self._failure(
            claim,
            work_spec,
            f"process-returncode-{returncode}",
            now,
        )

    def _validate_guardian_completion(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        wrapper_returncode: int,
    ) -> None:
        receipt_path = self._guardian_receipt_path(claim.claim_id)
        paths = gpu_lease_worker.guardian_receipt_paths(receipt_path)
        ready = _load_canonical_object(paths.ready, "GPU7 guardian ready receipt")
        lifetime = _load_optional_object(
            paths.lifetime, "GPU7 guardian lifetime receipt"
        )
        completion = _load_canonical_object(
            paths.completion, "GPU7 guardian completion receipt"
        )
        for role, receipt in (
            ("GPU7 guardian ready receipt", ready),
            ("GPU7 guardian completion receipt", completion),
        ):
            _validate_self_hash(receipt, "receipt_sha256", role)
        if lifetime is not None:
            _validate_self_hash(
                lifetime,
                "receipt_sha256",
                "GPU7 guardian lifetime receipt",
            )
        start = _load_canonical_object(
            self._start_receipt_path(claim.claim_id), "start receipt"
        )
        self._validate_start_receipt(start, claim, work_spec)
        wrapper = self._identity_from_start(start)
        intent = _load_canonical_object(
            self._start_intent_path(claim.claim_id), "start intent"
        )
        _validate_self_hash(intent, "intent_sha256", "start intent")
        if (
            intent.get("guardian_receipt_path") != os.fspath(receipt_path)
            or intent.get("gpu7_guardian_prefix_sha256") is None
        ):
            raise ExecutorStateError(
                "start intent does not bind the guardian receipt and prefix"
            )
        full_argv = intent.get("argv")
        if not isinstance(full_argv, list):
            raise ExecutorStateError("start intent guardian argv is malformed")
        prefix = self.spec.gpu7_guardian_prefix
        if prefix is None or len(full_argv) <= len(prefix):
            raise ExecutorStateError("start intent has no guardian child argv")
        if prefix[-1] == "--command-json":
            if len(full_argv) != len(prefix) + 1:
                raise ExecutorStateError(
                    "start intent guardian command JSON is ambiguous"
                )
            try:
                child_value = json.loads(full_argv[-1])
            except json.JSONDecodeError as exc:
                raise ExecutorStateError(
                    "start intent guardian command JSON is malformed"
                ) from exc
            child_argv = list(
                _validate_argv(
                    child_value,
                    "guardian receipt child argv",
                    _MAIN_PLACEHOLDERS,
                    expanded=True,
                )
            )
        else:
            child_argv = full_argv[len(prefix) :]
            _validate_argv(
                child_argv,
                "guardian receipt child argv",
                _MAIN_PLACEHOLDERS,
                expanded=True,
            )
        common = {
            "claim_id": claim.claim_id,
            "work_id": claim.work_id,
            "receipt_path": os.fspath(receipt_path),
            "argv": child_argv,
            "argv_sha256": canonical_sha256(child_argv),
            "wrapper_process_identity": wrapper.to_dict(),
            "wrapper_process_identity_sha256": canonical_sha256(wrapper.to_dict()),
        }
        for role, receipt in (
            ("ready", ready),
            ("lifetime", lifetime),
            ("completion", completion),
        ):
            if receipt is None:
                continue
            for key, expected in common.items():
                if receipt.get(key) != expected:
                    raise ExecutorStateError(
                        f"guardian {role} receipt contradicts {key}"
                    )
            if (
                not isinstance(receipt.get("lease_id"), str)
                or not receipt["lease_id"]
                or not isinstance(receipt.get("expected_gpu_uuid"), str)
                or not receipt["expected_gpu_uuid"]
            ):
                raise ExecutorStateError(
                    f"guardian {role} receipt lacks lease identity"
                )
        if (
            ready.get("contract") != gpu_lease_worker.READY_RECEIPT_CONTRACT
            or ready.get("phase") != "ready"
            or completion.get("contract")
            != gpu_lease_worker.COMPLETION_RECEIPT_CONTRACT
            or completion.get("phase") != "completion"
            or completion.get("ready_receipt_sha256") != ready.get("receipt_sha256")
            or completion.get("trainer_restored") is not True
            or not isinstance(completion.get("restoration"), Mapping)
            or completion.get("returncode") != wrapper_returncode
        ):
            raise ExecutorStateError(
                "guardian completion does not prove a restored lifetime"
            )
        lease_id = completion["lease_id"]
        expected_gpu_uuid = completion["expected_gpu_uuid"]
        if (
            ready.get("lease_id") != lease_id
            or ready.get("expected_gpu_uuid") != expected_gpu_uuid
        ):
            raise ExecutorStateError(
                "guardian ready and completion lease identities differ"
            )
        status = completion.get("status")
        lifetime_hash = completion.get("lifetime_receipt_sha256")
        if status == "interrupted-before-spawn":
            if lifetime is not None or lifetime_hash is not None:
                raise ExecutorStateError(
                    "guardian no-spawn completion has a lifetime receipt"
                )
            return
        if status not in {"completed", "child-failed", "interrupted"}:
            raise ExecutorStateError("guardian completion status is unsafe")
        if (
            lifetime is None
            or lifetime.get("contract") != gpu_lease_worker.LIFETIME_RECEIPT_CONTRACT
            or lifetime.get("phase") != "lifetime"
            or lifetime.get("lease_id") != lease_id
            or lifetime.get("expected_gpu_uuid") != expected_gpu_uuid
            or lifetime.get("ready_receipt_sha256") != ready.get("receipt_sha256")
            or lifetime.get("receipt_sha256") != lifetime_hash
            or completion.get("child_process_identity_sha256")
            != lifetime.get("child_process_identity_sha256")
        ):
            raise ExecutorStateError("guardian lifetime receipt chain is incomplete")
        child = ProcessIdentity.from_value(
            lifetime.get("child_process_identity"), "guardian child process"
        )
        if (
            child.pid == wrapper.pid
            or child.process_group_id != wrapper.process_group_id
            or lifetime.get("child_process_identity_sha256")
            != canonical_sha256(child.to_dict())
        ):
            raise ExecutorStateError(
                "guardian child did not share the wrapper process group"
            )

    def _failure(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        reason: str,
        now: float,
    ) -> Mapping[str, Any]:
        previous = self._load_retry(claim.work_id)
        if previous is not None and previous.get("last_claim_id") == claim.claim_id:
            failure_count = int(previous["failure_count"])
            status = str(previous["status"])
            not_before = previous["not_before_unix"]
        else:
            previous_count = 0 if previous is None else int(previous["failure_count"])
            failure_count = previous_count + 1
            exhausted = failure_count > self.spec.retry_budget
            delay = min(
                self.spec.backoff_max_seconds,
                self.spec.backoff_initial_seconds * (2.0 ** min(failure_count - 1, 62)),
            )
            status = "quarantined" if exhausted else "backoff"
            not_before = None if exhausted else now + delay
            self._write_retry(
                claim.work_id,
                failure_count=failure_count,
                status=status,
                last_claim_id=claim.claim_id,
                not_before=not_before,
                reason=reason,
                now=now,
            )
        if status == "quarantined":
            quarantine = _self_hashed(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": QUARANTINE_CONTRACT,
                    "work_id": claim.work_id,
                    "work_spec_sha256": work_spec.spec_sha256,
                    "claim_id": claim.claim_id,
                    "failure_count": failure_count,
                    "retry_budget": self.spec.retry_budget,
                    "reason": reason,
                    "quarantined_at_unix": now,
                },
                "receipt_sha256",
            )
            _atomic_write_json(self._quarantine_path(claim.work_id), quarantine)
            release = self._release_claim(
                claim,
                work_spec,
                ReleaseOutcome.FAILED,
                "retry-budget-exhausted",
                now,
            )
            return {
                "gpu_id": claim.gpu_id,
                "status": "quarantined",
                "work_id": claim.work_id,
                "claim_id": claim.claim_id,
                "failure_count": failure_count,
                "release_id": release.release_id,
            }
        release = self._release_claim(
            claim,
            work_spec,
            ReleaseOutcome.REQUEUE,
            reason,
            now,
        )
        return {
            "gpu_id": claim.gpu_id,
            "status": "retry-backoff",
            "work_id": claim.work_id,
            "claim_id": claim.claim_id,
            "failure_count": failure_count,
            "not_before_unix": not_before,
            "release_id": release.release_id,
        }

    def _release_claim(
        self,
        claim: Claim,
        work_spec: WorkExecutionSpec,
        outcome: ReleaseOutcome,
        reason: str,
        now: float,
        *,
        idle_reason: Optional[IdleReason] = None,
        idle_details: Optional[Mapping[str, Any]] = None,
    ) -> ReleaseRecord:
        intent = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": RELEASE_INTENT_CONTRACT,
                "claim": claim.to_dict(),
                "work_spec_sha256": work_spec.spec_sha256,
                "outcome": outcome.value,
                "reason": reason,
                "idle_reason": None if idle_reason is None else idle_reason.value,
                "idle_details": dict(idle_details or {}),
                "requested_at_unix": now,
            },
            "intent_sha256",
        )
        _atomic_write_json(self._release_intent_path(claim.claim_id), intent)
        existing = self.scheduler.get_work(claim.work_id)
        release = None if existing is None else existing.last_release
        if (
            release is None
            or release.claim_id != claim.claim_id
            or release.outcome != outcome
        ):
            release = self.scheduler.release(
                claim,
                outcome=outcome,
                idle_reason=idle_reason,
                idle_details=idle_details,
            )
        receipt = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": RELEASE_RECEIPT_CONTRACT,
                "release_intent_sha256": intent["intent_sha256"],
                "release": release.to_dict(),
                "finalized_at_unix": now,
            },
            "receipt_sha256",
        )
        _atomic_write_json(self._release_receipt_path(claim.claim_id), receipt)
        return release

    def _recover_release_intents(self, now: float) -> List[str]:
        recovered = []
        root = self.state_directory / "release-intents"
        if not root.is_dir():
            return recovered
        for path in sorted(root.glob("*.json")):
            intent = _load_canonical_object(path, "release intent")
            _validate_self_hash(intent, "intent_sha256", "release intent")
            if intent.get("contract") != RELEASE_INTENT_CONTRACT:
                raise ExecutorStateError("release intent contract is unsupported")
            claim = Claim.from_dict(intent["claim"])
            receipt_path = self._release_receipt_path(claim.claim_id)
            if receipt_path.is_file():
                continue
            try:
                outcome = ReleaseOutcome(intent["outcome"])
                idle = (
                    None
                    if intent["idle_reason"] is None
                    else IdleReason(intent["idle_reason"])
                )
            except ValueError as exc:
                raise ExecutorStateError("release intent outcome is invalid") from exc
            record = self.scheduler.get_work(claim.work_id)
            release = None if record is None else record.last_release
            if (
                release is None
                or release.claim_id != claim.claim_id
                or release.outcome != outcome
            ):
                release = self.scheduler.release(
                    claim,
                    outcome=outcome,
                    idle_reason=idle,
                    idle_details=intent["idle_details"],
                )
            receipt = _self_hashed(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": RELEASE_RECEIPT_CONTRACT,
                    "release_intent_sha256": intent["intent_sha256"],
                    "release": release.to_dict(),
                    "finalized_at_unix": now,
                },
                "receipt_sha256",
            )
            _atomic_write_json(receipt_path, receipt)
            recovered.append(claim.claim_id)
        return recovered

    def _write_claim_receipt(
        self, claim: Claim, work_spec: WorkExecutionSpec, now: float
    ) -> Dict[str, Any]:
        receipt = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": CLAIM_RECEIPT_CONTRACT,
                "claim": claim.to_dict(),
                "executor_spec_sha256": self.spec.spec_sha256,
                "work_spec_sha256": work_spec.spec_sha256,
                "recorded_at_unix": now,
            },
            "receipt_sha256",
        )
        _atomic_write_json(self._claim_receipt_path(claim.claim_id), receipt)
        return receipt

    def _validate_claim_receipt(
        self,
        receipt: Mapping[str, Any],
        claim: Claim,
        work_spec: WorkExecutionSpec,
    ) -> None:
        _validate_self_hash(receipt, "receipt_sha256", "claim receipt")
        if (
            receipt.get("contract") != CLAIM_RECEIPT_CONTRACT
            or receipt.get("claim") != claim.to_dict()
            or receipt.get("work_spec_sha256") != work_spec.spec_sha256
        ):
            raise ExecutorStateError("claim receipt contradicts active claim")

    def _identity_from_start(self, start: Mapping[str, Any]) -> ProcessIdentity:
        _validate_self_hash(start, "receipt_sha256", "start receipt")
        identity = ProcessIdentity.from_value(
            start.get("process_identity"), "start receipt process"
        )
        if start.get("process_identity_sha256") != canonical_sha256(identity.to_dict()):
            raise ExecutorStateError("start receipt process hash is invalid")
        return identity

    def _validate_start_receipt(
        self,
        start: Mapping[str, Any],
        claim: Claim,
        work_spec: WorkExecutionSpec,
    ) -> None:
        if (
            start.get("contract") != START_RECEIPT_CONTRACT
            or start.get("claim_id") != claim.claim_id
            or start.get("work_id") != claim.work_id
            or start.get("gpu_id") != claim.gpu_id
            or start.get("owner_id") != claim.owner_id
            or start.get("work_spec_sha256") != work_spec.spec_sha256
        ):
            raise ExecutorStateError("start receipt contradicts active claim")
        intent = _load_canonical_object(
            self._start_intent_path(claim.claim_id), "start intent"
        )
        _validate_self_hash(intent, "intent_sha256", "start intent")
        if start.get("start_intent_sha256") != intent.get("intent_sha256"):
            raise ExecutorStateError("start receipt does not bind its intent")

    def _validate_completion(
        self,
        completion: Mapping[str, Any],
        claim: Claim,
        work_spec: WorkExecutionSpec,
        start: Mapping[str, Any],
    ) -> None:
        _validate_self_hash(completion, "receipt_sha256", "completion receipt")
        returncode = completion.get("returncode")
        if (
            completion.get("contract") != COMPLETION_RECEIPT_CONTRACT
            or completion.get("claim_id") != claim.claim_id
            or completion.get("work_id") != claim.work_id
            or completion.get("gpu_id") != claim.gpu_id
            or completion.get("work_spec_sha256") != work_spec.spec_sha256
            or completion.get("start_receipt_sha256") != start.get("receipt_sha256")
            or completion.get("process_identity_sha256")
            != start.get("process_identity_sha256")
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
        ):
            raise ExecutorStateError("completion receipt contradicts active process")

    def _load_retry(self, work_id: str) -> Optional[Dict[str, Any]]:
        retry = _load_optional_object(self._retry_path(work_id), "retry state")
        if retry is None:
            return None
        _validate_self_hash(retry, "state_sha256", "retry state")
        if (
            retry.get("contract") != RETRY_STATE_CONTRACT
            or retry.get("work_id") != work_id
            or retry.get("status") not in {"backoff", "quarantined", "completed"}
            or isinstance(retry.get("failure_count"), bool)
            or not isinstance(retry.get("failure_count"), int)
            or retry["failure_count"] < 0
        ):
            raise ExecutorStateError("retry state is malformed")
        not_before = retry.get("not_before_unix")
        if retry["status"] == "backoff":
            if (
                isinstance(not_before, bool)
                or not isinstance(not_before, (int, float))
                or not math.isfinite(float(not_before))
            ):
                raise ExecutorStateError("retry backoff timestamp is malformed")
        elif not_before is not None:
            raise ExecutorStateError(
                "non-backoff retry state has a not-before timestamp"
            )
        return retry

    def _write_retry(
        self,
        work_id: str,
        *,
        failure_count: int,
        status: str,
        last_claim_id: str,
        not_before: Optional[float],
        reason: Optional[str],
        now: float,
    ) -> Dict[str, Any]:
        value = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": RETRY_STATE_CONTRACT,
                "work_id": work_id,
                "failure_count": failure_count,
                "retry_budget": self.spec.retry_budget,
                "status": status,
                "last_claim_id": last_claim_id,
                "not_before_unix": not_before,
                "reason": reason,
                "updated_at_unix": now,
            },
            "state_sha256",
        )
        _atomic_replace_json(self._retry_path(work_id), value)
        return value

    def _write_heartbeat(self, now: float, state: str) -> None:
        value = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": HEARTBEAT_CONTRACT,
                "owner_id": self.spec.owner_id,
                "executor_spec_sha256": self.spec.spec_sha256,
                "state": state,
                "updated_at_unix": now,
            },
            "state_sha256",
        )
        _atomic_replace_json(self._heartbeat_path(self.spec.owner_id), value)

    def _owner_is_stale(self, claim: Claim, now: float) -> bool:
        heartbeat = _load_optional_object(
            self._heartbeat_path(claim.owner_id), "owner heartbeat"
        )
        if heartbeat is None:
            return now - claim.claimed_at > self.spec.stale_after_seconds
        _validate_self_hash(heartbeat, "state_sha256", "owner heartbeat")
        updated = heartbeat.get("updated_at_unix")
        if (
            heartbeat.get("contract") != HEARTBEAT_CONTRACT
            or heartbeat.get("owner_id") != claim.owner_id
            or isinstance(updated, bool)
            or not isinstance(updated, (int, float))
            or not math.isfinite(float(updated))
        ):
            raise ExecutorStateError("owner heartbeat is malformed")
        if float(updated) > now:
            return False
        return now - float(updated) > self.spec.stale_after_seconds

    def _halt(
        self,
        gpu_id: str,
        reason: str,
        *,
        claim: Optional[Claim] = None,
        now: Optional[float] = None,
    ) -> Mapping[str, Any]:
        observed_at = self._now() if now is None else now
        self.scheduler.set_safety_halt(reason, gpu_id=gpu_id)
        value = _self_hashed(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": HALT_CONTRACT,
                "gpu_id": gpu_id,
                "claim_id": None if claim is None else claim.claim_id,
                "work_id": None if claim is None else claim.work_id,
                "reason": reason,
                "halted_at_unix": observed_at,
            },
            "state_sha256",
        )
        _atomic_replace_json(self._halt_path(gpu_id), value)
        return {
            "gpu_id": gpu_id,
            "status": "safety-halt",
            "claim_id": None if claim is None else claim.claim_id,
            "reason": reason,
        }

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise ClusterExecutorError("clock returned a non-finite value")
        return value

    def _ensure_state_directory(self) -> None:
        directory = self.state_directory
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise ExecutorStateError(
                    "executor state path is not a regular directory"
                )
        else:
            directory.mkdir(parents=True, exist_ok=False)
            _fsync_directory(directory.parent)
        for name in (
            "claim-receipts",
            "start-intents",
            "start-receipts",
            "completions",
            "release-intents",
            "release-receipts",
            "retry",
            "quarantine",
            "heartbeats",
            "lease-gates",
            "guardian-receipts",
            "drain-intents",
            "drain-commands",
            "drain-boundaries",
            "halts",
            "logs",
        ):
            path = directory / name
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise ExecutorStateError(f"unsafe executor state directory: {path}")
            else:
                path.mkdir()
                _fsync_directory(directory)

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise ExecutorBusyError("POSIX advisory locking is unavailable")
        process_lock = _process_lock(self.lock_path)
        if not process_lock.acquire(blocking=False):
            raise ExecutorBusyError("another local executor is active")
        descriptor = -1
        locked = False
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(os.fspath(self.lock_path), flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ExecutorBusyError("executor lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise ExecutorBusyError(
                        "another local executor holds the state lock"
                    ) from exc
                raise
            locked = True
            yield
        finally:
            if descriptor >= 0:
                if locked:
                    with contextlib.suppress(OSError):
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            process_lock.release()

    def _path(self, directory: str, identifier: str) -> Path:
        return self.state_directory / directory / _state_name(identifier)

    def _claim_receipt_path(self, claim_id: str) -> Path:
        return self._path("claim-receipts", claim_id)

    def _start_intent_path(self, claim_id: str) -> Path:
        return self._path("start-intents", claim_id)

    def _start_receipt_path(self, claim_id: str) -> Path:
        return self._path("start-receipts", claim_id)

    def _completion_path(self, claim_id: str) -> Path:
        return self._path("completions", claim_id)

    def _release_intent_path(self, claim_id: str) -> Path:
        return self._path("release-intents", claim_id)

    def _release_receipt_path(self, claim_id: str) -> Path:
        return self._path("release-receipts", claim_id)

    def _retry_path(self, work_id: str) -> Path:
        return self._path("retry", work_id)

    def _quarantine_path(self, work_id: str) -> Path:
        return self._path("quarantine", work_id)

    def _heartbeat_path(self, owner_id: str) -> Path:
        return self._path("heartbeats", owner_id)

    def _lease_gate_path(self, claim_id: str) -> Path:
        return self._path("lease-gates", claim_id)

    def _guardian_receipt_path(self, claim_id: str) -> Path:
        return self._path("guardian-receipts", claim_id)

    def _drain_intent_path(self, claim_id: str) -> Path:
        return self._path("drain-intents", claim_id)

    def _drain_command_path(self, claim_id: str) -> Path:
        return self._path("drain-commands", claim_id)

    def _drain_boundary_path(self, claim_id: str) -> Path:
        return self._path("drain-boundaries", claim_id)

    def _halt_path(self, gpu_id: str) -> Path:
        return self._path("halts", gpu_id)


Executor = ClusterExecutor


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", dest="global_spec", type=Path)
    parser.add_argument("--expected-spec-sha256", dest="global_expected_spec_sha256")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "once", "watch"):
        child = subparsers.add_parser(command)
        child.add_argument("--spec", type=Path)
        child.add_argument("--expected-spec-sha256")
        if command == "watch":
            child.add_argument("--interval", type=float)
    args = parser.parse_args(argv)
    if (
        args.spec is not None
        and args.global_spec is not None
        and args.spec != args.global_spec
    ):
        parser.error("global and command --spec values differ")
    args.spec = args.spec or args.global_spec
    if args.spec is None:
        parser.error("--spec is required")
    if (
        args.expected_spec_sha256 is not None
        and args.global_expected_spec_sha256 is not None
        and args.expected_spec_sha256 != args.global_expected_spec_sha256
    ):
        parser.error("global and command expected specification hashes differ")
    args.expected_spec_sha256 = (
        args.expected_spec_sha256 or args.global_expected_spec_sha256
    )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        executor = ClusterExecutor(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
        )
        if args.command == "status":
            payloads: Iterator[Mapping[str, Any]] = iter((executor.status(),))
        elif args.command == "once":
            payloads = iter((executor.once(),))
        else:
            interval = (
                executor.spec.poll_interval_seconds
                if args.interval is None
                else float(args.interval)
            )
            if not math.isfinite(interval) or interval <= 0:
                raise ExecutorSpecError("watch interval must be positive and finite")

            def watch() -> Iterator[Mapping[str, Any]]:
                while True:
                    yield executor.once()
                    time.sleep(min(interval, executor.spec.heartbeat_interval_seconds))

            payloads = watch()
        for payload in payloads:
            sys.stdout.buffer.write(canonical_json_bytes(payload) + b"\n")
            sys.stdout.buffer.flush()
        return 0
    except (ClusterExecutorError, SchedulerError, OSError) as exc:
        error = {
            "schema_version": SCHEMA_VERSION,
            "contract": ERROR_CONTRACT,
            "error": type(exc).__name__,
            "message": str(exc),
        }
        sys.stderr.buffer.write(canonical_json_bytes(error) + b"\n")
        return 2


__all__ = [
    "CLAIM_RECEIPT_CONTRACT",
    "COMPLETION_RECEIPT_CONTRACT",
    "ClusterExecutor",
    "ClusterExecutorError",
    "CommandResult",
    "EXECUTOR_SPEC_CONTRACT",
    "Executor",
    "ExecutorBusyError",
    "ExecutorSpec",
    "ExecutorSpecError",
    "ExecutorStateError",
    "LeaseProofCallback",
    "LeaseProofDenied",
    "LeaseRequest",
    "ProcessIdentity",
    "ProcessRunner",
    "ProcessSpawnError",
    "SCHEMA_VERSION",
    "SafeDrainSpec",
    "SubprocessRunner",
    "WORK_SPEC_CONTRACT",
    "WorkExecutionSpec",
    "load_executor_spec",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
