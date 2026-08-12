#!/usr/bin/env python3
"""Run one argv command while owning the GPU lease for its full lifetime.

The wrapper is intentionally an argv-only process guardian.  It enters
``GpuLeaseManager.exclusive_handoff`` before spawning the child, keeps the
handoff context open until that exact child exits, and only publishes a
successful completion after the manager proves that the trainer was restored.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from risk_score.cluster_scheduler import canonical_json_bytes, canonical_sha256
from risk_score.gpu_lease import (
    GpuLeaseError,
    GpuLeaseManager,
    LeaseRecord,
    ProcessIdentity,
    RuntimeConfig,
)
from risk_score.promotion_host import (
    PROCESS_IDENTITY_FIELDS,
    capture_process_identity,
    capture_spawned_process,
)

SCHEMA_VERSION = 1
WORKER_SPEC_CONTRACT = "risk-score-gpu-lease-worker-spec-v1"
READY_RECEIPT_CONTRACT = "risk-score-gpu-lease-worker-ready-v1"
LIFETIME_RECEIPT_CONTRACT = "risk-score-gpu-lease-worker-lifetime-v1"
COMPLETION_RECEIPT_CONTRACT = "risk-score-gpu-lease-worker-completion-v1"
ERROR_CONTRACT = "risk-score-gpu-lease-worker-error-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_WORKER_SPEC_KEYS = {
    "schema_version",
    "contract",
    "runtime_config",
    "runtime_config_sha256",
    "receipt_directory",
    "spec_sha256",
}
_REPLAYABLE_COMPLETION_STATUSES = frozenset(
    {"child-failed", "completed", "interrupted", "interrupted-before-spawn"}
)


class GpuLeaseWorkerError(RuntimeError):
    """A fail-closed worker error with a stable machine-readable code."""

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
class WorkerSpec:
    runtime_config: Path
    runtime_config_sha256: str
    receipt_directory: Path
    spec_sha256: str
    source_path: Path


@dataclass(frozen=True)
class GuardianReceiptPaths:
    ready: Path
    lifetime: Path
    completion: Path


class GuardianProcessRunner(Protocol):
    """Process operations kept separate from the GPU lease manager runner."""

    def wrapper_identity(self) -> Mapping[str, Any]: ...

    def spawn(
        self, argv: Sequence[str], *, process_group_id: int
    ) -> Mapping[str, Any]: ...

    def wait(self, identity: Mapping[str, Any]) -> int: ...

    def send_signal(self, identity: Mapping[str, Any], sig: int) -> None: ...


class SubprocessGuardianRunner:
    """Production child runner; it never invokes a shell."""

    def __init__(self) -> None:
        self._children: Dict[int, subprocess.Popen[Any]] = {}

    def wrapper_identity(self) -> Mapping[str, Any]:
        return capture_process_identity(os.getpid())

    def spawn(self, argv: Sequence[str], *, process_group_id: int) -> Mapping[str, Any]:
        command = _validate_command(argv)
        process = subprocess.Popen(
            list(command),
            start_new_session=False,
            shell=False,
        )
        self._children[process.pid] = process
        try:
            identity = _validated_identity(
                capture_spawned_process(process), "spawned guardian child"
            )
        except BaseException:
            with contextlib.suppress(OSError):
                process.terminate()
            self._children.pop(process.pid, None)
            raise
        if identity["process_group_id"] != process_group_id:
            with contextlib.suppress(OSError):
                process.terminate()
            self._children.pop(process.pid, None)
            raise GpuLeaseWorkerError(
                "child_process_group_mismatch",
                "Guardian child did not join the wrapper process group",
                details={
                    "expected_process_group_id": process_group_id,
                    "observed_process_group_id": identity["process_group_id"],
                },
            )
        return identity

    def wait(self, identity: Mapping[str, Any]) -> int:
        expected = _validated_identity(identity, "guardian child")
        process = self._children.get(expected["pid"])
        if process is None:
            raise GpuLeaseWorkerError(
                "unknown_child",
                "Guardian runner has no handle for the recorded child",
            )
        returncode = process.wait()
        self._children.pop(expected["pid"], None)
        remaining = _process_group_members(
            expected["process_group_id"], exclude=(os.getpid(),)
        )
        if remaining:
            raise GpuLeaseWorkerError(
                "child_process_group_not_drained",
                "Guardian child exited while descendants remain in its process group",
                details={"remaining_pids": list(remaining)},
            )
        return int(returncode)

    def send_signal(self, identity: Mapping[str, Any], sig: int) -> None:
        expected = _validated_identity(identity, "guardian child")
        try:
            current = _validated_identity(
                capture_process_identity(expected["pid"]),
                "observed guardian child",
            )
        except BaseException as exc:
            raise GpuLeaseWorkerError(
                "child_identity_unverifiable",
                "Refusing to signal an absent or unverifiable guardian child",
            ) from exc
        if current != expected:
            raise GpuLeaseWorkerError(
                "child_identity_changed",
                "Refusing to signal a reused guardian child PID",
            )
        os.killpg(expected["process_group_id"], sig)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARACTERS)
    )


def _identifier(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise GpuLeaseWorkerError(
            "invalid_binding", f"{role} must be a nonempty trimmed string"
        )
    return value


def _validate_command(argv: Sequence[str]) -> Tuple[str, ...]:
    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or any(not isinstance(part, str) or not part or "\x00" in part for part in argv)
    ):
        raise GpuLeaseWorkerError(
            "invalid_command", "Child command must be a nonempty argv array"
        )
    return tuple(argv)


def _process_group_members(
    process_group_id: int, *, exclude: Sequence[int] = ()
) -> Tuple[int, ...]:
    proc = Path("/proc")
    if not proc.is_dir():
        raise GpuLeaseWorkerError(
            "process_group_unverifiable",
            "Linux procfs is required to verify the guardian process group",
        )
    excluded = frozenset(exclude)
    members = []
    try:
        entries = tuple(proc.iterdir())
    except OSError as exc:
        raise GpuLeaseWorkerError(
            "process_group_unverifiable",
            "Could not enumerate procfs for guardian process-group verification",
        ) from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8")
            close_paren = text.rfind(")")
            fields = text[close_paren + 1 :].split()
            observed_group = int(fields[2])
        except (IndexError, OSError, ValueError):
            continue
        if observed_group == process_group_id:
            members.append(pid)
    return tuple(sorted(members))


def _normalized_absolute(
    value: Any,
    role: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise GpuLeaseWorkerError("unsafe_path", f"{role} must be an absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise GpuLeaseWorkerError(
            "unsafe_path", f"{role} must be lexically normalized and absolute"
        )
    current = path
    while True:
        if current.is_symlink():
            raise GpuLeaseWorkerError(
                "unsafe_path",
                f"{role} has a symlinked path component",
                details={"component": os.fspath(current)},
            )
        if current.parent == current:
            break
        current = current.parent
    if require_file:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise GpuLeaseWorkerError(
                "unsafe_path", f"{role} must be an existing regular file"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise GpuLeaseWorkerError(
                "unsafe_path", f"{role} must be an existing regular file"
            )
    if require_directory and (not path.is_dir() or path.is_symlink()):
        raise GpuLeaseWorkerError(
            "unsafe_path", f"{role} must be an existing non-symlink directory"
        )
    return path


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
        raise GpuLeaseWorkerError(
            "file_hash_failed",
            f"Could not hash {path}: {exc}",
        ) from exc
    return digest.hexdigest()


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GpuLeaseWorkerError(
                "invalid_json", f"Duplicate JSON object key: {key}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise GpuLeaseWorkerError(
        "invalid_json", f"Non-finite JSON value is forbidden: {value}"
    )


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = _normalized_absolute(path, role, require_file=True)
    metadata = source.lstat()
    if metadata.st_size > MAX_JSON_BYTES:
        raise GpuLeaseWorkerError("invalid_json", f"{role} exceeds the size limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except GpuLeaseWorkerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GpuLeaseWorkerError(
            "invalid_json", f"Could not decode {role}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GpuLeaseWorkerError("invalid_json", f"{role} root must be an object")
    if data != canonical_json_bytes(value) + b"\n":
        raise GpuLeaseWorkerError(
            "noncanonical_json",
            f"{role} must be canonical newline-terminated JSON",
        )
    return value


def _load_optional_canonical_object(path: Path, role: str) -> Optional[Dict[str, Any]]:
    if not os.path.lexists(os.fspath(path)):
        return None
    return _load_canonical_object(path, role)


def _self_hashed(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _validate_receipt(value: Mapping[str, Any], role: str) -> str:
    supplied = value.get("receipt_sha256")
    if not _is_sha256(supplied):
        raise GpuLeaseWorkerError(
            "invalid_receipt", f"{role} has no valid receipt hash"
        )
    body = dict(value)
    body.pop("receipt_sha256", None)
    if canonical_sha256(body) != supplied:
        raise GpuLeaseWorkerError("invalid_receipt", f"{role} receipt hash is invalid")
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


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    target = _normalized_absolute(path, "guardian receipt")
    target.parent.mkdir(parents=True, exist_ok=True)
    _normalized_absolute(
        target.parent, "guardian receipt directory", require_directory=True
    )
    data = canonical_json_bytes(dict(value)) + b"\n"
    if os.path.lexists(os.fspath(target)):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise GpuLeaseWorkerError(
                "receipt_conflict",
                "Immutable guardian receipt conflicts with an existing file",
                details={"path": os.fspath(target)},
            )
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
                raise GpuLeaseWorkerError(
                    "receipt_conflict",
                    "Immutable guardian receipt conflicts with an existing file",
                    details={"path": os.fspath(target)},
                )
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _validated_identity(value: Any, role: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(PROCESS_IDENTITY_FIELDS):
        raise GpuLeaseWorkerError(
            "invalid_process_identity", f"{role} identity fields are malformed"
        )
    identity = {key: value[key] for key in PROCESS_IDENTITY_FIELDS}
    if (
        type(identity["pid"]) is not int
        or identity["pid"] <= 0
        or type(identity["start_time_ticks"]) is not int
        or identity["start_time_ticks"] < 0
        or type(identity["process_group_id"]) is not int
        or identity["process_group_id"] <= 0
        or not _is_sha256(identity["command_sha256"])
        or not isinstance(identity["boot_id"], str)
        or not identity["boot_id"]
        or not isinstance(identity["cgroup"], str)
        or not identity["cgroup"]
    ):
        raise GpuLeaseWorkerError(
            "invalid_process_identity", f"{role} identity is malformed"
        )
    return identity


def _lease_identity(identity: Mapping[str, Any]) -> ProcessIdentity:
    value = _validated_identity(identity, "supervised trainer")
    return ProcessIdentity(
        pid=value["pid"],
        start_time_ticks=value["start_time_ticks"],
        process_group_id=value["process_group_id"],
        boot_id=value["boot_id"],
        command_sha256=value["command_sha256"],
        cgroup=value["cgroup"],
    )


def _lease_identity_dict(identity: ProcessIdentity) -> Dict[str, Any]:
    value = {
        "pid": identity.pid,
        "start_time_ticks": identity.start_time_ticks,
        "process_group_id": identity.process_group_id,
        "boot_id": identity.boot_id,
        "command_sha256": identity.command_sha256,
        "cgroup": identity.cgroup,
    }
    return _validated_identity(value, "GPU lease process")


def load_worker_spec(
    path: Path, *, expected_spec_sha256: Optional[str] = None
) -> WorkerSpec:
    source = _normalized_absolute(path, "GPU lease worker spec", require_file=True)
    raw = _load_canonical_object(source, "GPU lease worker spec")
    if set(raw) != _WORKER_SPEC_KEYS:
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "GPU lease worker spec fields differ from schema"
        )
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "GPU lease worker spec schema is unsupported"
        )
    if raw["contract"] != WORKER_SPEC_CONTRACT:
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "GPU lease worker spec contract is unsupported"
        )
    supplied_hash = raw["spec_sha256"]
    if not _is_sha256(supplied_hash):
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "GPU lease worker spec hash is malformed"
        )
    body = dict(raw)
    body.pop("spec_sha256")
    if canonical_sha256(body) != supplied_hash:
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "GPU lease worker spec self-hash is invalid"
        )
    if expected_spec_sha256 is not None and expected_spec_sha256 != supplied_hash:
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "GPU lease worker spec hash is not expected"
        )
    runtime_hash = raw["runtime_config_sha256"]
    if not _is_sha256(runtime_hash):
        raise GpuLeaseWorkerError(
            "invalid_worker_spec", "Runtime config hash is malformed"
        )
    runtime = _normalized_absolute(
        raw["runtime_config"], "GPU lease runtime config", require_file=True
    )
    if _file_sha256(runtime) != runtime_hash:
        raise GpuLeaseWorkerError(
            "runtime_config_changed",
            "GPU lease runtime config does not match the worker spec",
        )
    receipt_directory = _normalized_absolute(
        raw["receipt_directory"], "guardian receipt directory"
    )
    if receipt_directory.exists() and (
        receipt_directory.is_symlink() or not receipt_directory.is_dir()
    ):
        raise GpuLeaseWorkerError("unsafe_path", "Guardian receipt directory is unsafe")
    return WorkerSpec(
        runtime_config=runtime,
        runtime_config_sha256=runtime_hash,
        receipt_directory=receipt_directory,
        spec_sha256=str(supplied_hash),
        source_path=source,
    )


def load_runtime_config(path: Path, *, expected_sha256: str) -> RuntimeConfig:
    if not _is_sha256(expected_sha256):
        raise GpuLeaseWorkerError(
            "invalid_runtime_hash", "Expected runtime config hash is malformed"
        )
    source = _normalized_absolute(path, "GPU lease runtime config", require_file=True)
    before = _file_sha256(source)
    if before != expected_sha256:
        raise GpuLeaseWorkerError(
            "runtime_config_changed",
            "GPU lease runtime config hash is not expected",
        )
    try:
        config = RuntimeConfig.from_json_file(source)
    except GpuLeaseError as exc:
        raise GpuLeaseWorkerError(
            "invalid_runtime_config", str(exc), details={"gpu_lease_code": exc.code}
        ) from exc
    after = _file_sha256(source)
    if after != before:
        raise GpuLeaseWorkerError(
            "runtime_config_changed",
            "GPU lease runtime config changed while it was loaded",
        )
    return config


def guardian_receipt_paths(path: Path) -> GuardianReceiptPaths:
    completion = _normalized_absolute(path, "guardian completion receipt")
    if completion.suffix == ".json":
        base = completion.name[: -len(".json")]
        ready = completion.with_name(base + ".ready.json")
        lifetime = completion.with_name(base + ".lifetime.json")
    else:
        ready = completion.with_name(completion.name + ".ready.json")
        lifetime = completion.with_name(completion.name + ".lifetime.json")
    return GuardianReceiptPaths(ready=ready, lifetime=lifetime, completion=completion)


def _trainer_binding_path(config: RuntimeConfig) -> Path:
    return Path(
        os.path.abspath(
            os.fspath(config.promotion_root / "supervisor" / "trainer.json")
        )
    )


def load_supervised_trainer_identity(
    config: RuntimeConfig,
    manager: GpuLeaseManager,
    *,
    binding_path: Optional[Path] = None,
) -> Tuple[ProcessIdentity, Path, str]:
    canonical_path = _trainer_binding_path(config)
    supplied = canonical_path if binding_path is None else Path(binding_path)
    supplied = _normalized_absolute(
        supplied, "supervised trainer binding", require_file=True
    )
    if supplied != canonical_path:
        raise GpuLeaseWorkerError(
            "trainer_binding_not_canonical",
            "Trainer identity must come from promotion/supervisor/trainer.json",
            details={
                "expected": os.fspath(canonical_path),
                "supplied": os.fspath(supplied),
            },
        )
    binding_hash = _file_sha256(supplied)
    value = _load_canonical_object(supplied, "supervised trainer binding")
    if (
        value.get("schema_version") != 1
        or value.get("role") != "trainer"
        or ("launch_status" in value and value.get("launch_status") != "running")
    ):
        raise GpuLeaseWorkerError(
            "invalid_trainer_binding",
            "Supervised trainer binding is not a live trainer record",
        )
    trainer = _lease_identity(value.get("process_identity"))
    current = manager.runner.current_identity(trainer.pid)
    if current is None or not trainer.same_process_as(current):
        raise GpuLeaseWorkerError(
            "stale_trainer_binding",
            "Supervised trainer binding does not identify the live trainer",
        )
    if _file_sha256(supplied) != binding_hash:
        raise GpuLeaseWorkerError(
            "trainer_binding_changed",
            "Supervised trainer binding changed during verification",
        )
    return trainer, supplied, binding_hash


class GpuLeaseWorker:
    """Own a manager handoff around one exact child argv."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        runtime_config_path: Path,
        runtime_config_sha256: str,
        claim_id: str,
        work_id: str,
        receipt_path: Path,
        argv: Sequence[str],
        worker_spec_sha256: Optional[str] = None,
        receipt_directory: Optional[Path] = None,
        trainer_binding_path: Optional[Path] = None,
        manager: Optional[GpuLeaseManager] = None,
        process_runner: Optional[GuardianProcessRunner] = None,
        clock: Callable[[], float] = time.time,
        install_signal_handlers: bool = True,
    ) -> None:
        self.config = config
        self.runtime_config_path = _normalized_absolute(
            runtime_config_path, "GPU lease runtime config", require_file=True
        )
        if not _is_sha256(runtime_config_sha256):
            raise GpuLeaseWorkerError(
                "invalid_runtime_hash", "Runtime config hash is malformed"
            )
        self.runtime_config_sha256 = runtime_config_sha256
        if _file_sha256(self.runtime_config_path) != runtime_config_sha256:
            raise GpuLeaseWorkerError(
                "runtime_config_changed",
                "Runtime config changed before guarded execution",
            )
        self.claim_id = _identifier(claim_id, "claim_id")
        self.work_id = _identifier(work_id, "work_id")
        self.receipts = guardian_receipt_paths(receipt_path)
        self.argv = _validate_command(argv)
        self.argv_sha256 = canonical_sha256(list(self.argv))
        if worker_spec_sha256 is not None and not _is_sha256(worker_spec_sha256):
            raise GpuLeaseWorkerError(
                "invalid_worker_spec", "Worker spec hash is malformed"
            )
        self.worker_spec_sha256 = worker_spec_sha256
        if receipt_directory is not None:
            expected_directory = _normalized_absolute(
                receipt_directory, "guardian receipt directory"
            )
            if self.receipts.completion.parent != expected_directory:
                raise GpuLeaseWorkerError(
                    "receipt_outside_spec",
                    "Guardian receipt is outside the hash-bound receipt directory",
                )
        self.trainer_binding_path = trainer_binding_path
        self.manager = manager or GpuLeaseManager(config)
        self.process_runner = process_runner or SubprocessGuardianRunner()
        self.clock = clock
        self.install_signal_handlers = install_signal_handlers
        self._interrupted = False
        self._child_identity: Optional[Dict[str, Any]] = None
        self._signal_error: Optional[BaseException] = None

    def run(self) -> int:
        replay = self._replay_or_refuse_incomplete()
        if replay is not None:
            return replay

        wrapper = _validated_identity(
            self.process_runner.wrapper_identity(), "GPU lease wrapper"
        )
        if wrapper["process_group_id"] != wrapper["pid"]:
            raise GpuLeaseWorkerError(
                "wrapper_not_group_leader",
                "GPU lease wrapper must be the leader of its process group",
            )
        trainer, binding_path, binding_sha256 = load_supervised_trainer_identity(
            self.config,
            self.manager,
            binding_path=self.trainer_binding_path,
        )

        ready: Optional[Dict[str, Any]] = None
        lifetime: Optional[Dict[str, Any]] = None
        lease: Optional[LeaseRecord] = None
        child_returncode: Optional[int] = None
        body_error: Optional[BaseException] = None

        with self._sigint_handler():
            try:
                with self.manager.exclusive_handoff(trainer) as lease_value:
                    lease = self._validated_lease(lease_value)
                    common = self._receipt_binding(
                        lease,
                        wrapper,
                        binding_path=binding_path,
                        binding_sha256=binding_sha256,
                    )
                    ready = _self_hashed(
                        {
                            **common,
                            "schema_version": SCHEMA_VERSION,
                            "contract": READY_RECEIPT_CONTRACT,
                            "phase": "ready",
                            "ready_at_unix": self._now(),
                        }
                    )
                    _atomic_create_json(self.receipts.ready, ready)
                    if not self._interrupted:
                        child = _validated_identity(
                            self.process_runner.spawn(
                                self.argv,
                                process_group_id=wrapper["process_group_id"],
                            ),
                            "GPU lease child",
                        )
                        if (
                            child["pid"] == wrapper["pid"]
                            or child["process_group_id"] != wrapper["process_group_id"]
                        ):
                            raise GpuLeaseWorkerError(
                                "child_process_group_mismatch",
                                "Child identity is not inside the wrapper process group",
                            )
                        self._child_identity = child
                        lifetime = _self_hashed(
                            {
                                **common,
                                "schema_version": SCHEMA_VERSION,
                                "contract": LIFETIME_RECEIPT_CONTRACT,
                                "phase": "lifetime",
                                "ready_receipt_sha256": ready["receipt_sha256"],
                                "child_process_identity": child,
                                "child_process_identity_sha256": canonical_sha256(
                                    child
                                ),
                                "started_at_unix": self._now(),
                            }
                        )
                        _atomic_create_json(self.receipts.lifetime, lifetime)
                        child_returncode = self.process_runner.wait(child)
                        if isinstance(child_returncode, bool) or not isinstance(
                            child_returncode, int
                        ):
                            raise GpuLeaseWorkerError(
                                "invalid_child_returncode",
                                "Guardian runner returned a malformed child status",
                            )
                    if self._signal_error is not None:
                        raise GpuLeaseWorkerError(
                            "child_interrupt_failed",
                            f"Could not forward SIGINT to the child: "
                            f"{self._signal_error}",
                        ) from self._signal_error
            except BaseException as exc:
                body_error = exc
            finally:
                self._child_identity = None

        restoration_error: Optional[BaseException] = None
        restoration: Optional[Dict[str, Any]] = None
        if lease is not None:
            try:
                restoration = self._restoration_evidence(lease)
            except BaseException as exc:
                restoration_error = exc

        if ready is not None and lease is not None:
            status: str
            effective_returncode: Optional[int]
            if restoration_error is not None:
                status = "cleanup-failed"
                effective_returncode = None
            elif body_error is not None:
                status = "worker-failed"
                effective_returncode = None
            elif self._interrupted and lifetime is None:
                status = "interrupted-before-spawn"
                effective_returncode = 130
            elif self._interrupted:
                status = "interrupted"
                effective_returncode = 130
            elif child_returncode == 0:
                status = "completed"
                effective_returncode = 0
            else:
                status = "child-failed"
                effective_returncode = _shell_returncode(child_returncode)
            completion = _self_hashed(
                {
                    **self._receipt_binding(
                        lease,
                        wrapper,
                        binding_path=binding_path,
                        binding_sha256=binding_sha256,
                    ),
                    "schema_version": SCHEMA_VERSION,
                    "contract": COMPLETION_RECEIPT_CONTRACT,
                    "phase": "completion",
                    "status": status,
                    "ready_receipt_sha256": ready["receipt_sha256"],
                    "lifetime_receipt_sha256": (
                        None if lifetime is None else lifetime["receipt_sha256"]
                    ),
                    "child_process_identity_sha256": (
                        None
                        if lifetime is None
                        else lifetime["child_process_identity_sha256"]
                    ),
                    "child_returncode": child_returncode,
                    "returncode": effective_returncode,
                    "interrupted": self._interrupted,
                    "trainer_restored": restoration is not None,
                    "restoration": restoration,
                    "error": (
                        None
                        if body_error is None and restoration_error is None
                        else _error_text(restoration_error or body_error)
                    ),
                    "completed_at_unix": self._now(),
                }
            )
            _atomic_create_json(self.receipts.completion, completion)

        if restoration_error is not None:
            raise GpuLeaseWorkerError(
                "trainer_restoration_failed",
                f"GPU lease cleanup did not prove trainer restoration: "
                f"{restoration_error}",
            ) from restoration_error
        if body_error is not None:
            if isinstance(body_error, GpuLeaseWorkerError):
                raise body_error
            if isinstance(body_error, GpuLeaseError):
                raise GpuLeaseWorkerError(
                    "gpu_lease_failed",
                    str(body_error),
                    details={"gpu_lease_code": body_error.code},
                ) from body_error
            raise GpuLeaseWorkerError(
                "guarded_execution_failed", str(body_error)
            ) from body_error
        if lease is None or ready is None or restoration is None:
            raise GpuLeaseWorkerError(
                "handoff_incomplete",
                "GPU lease handoff completed without durable lifetime evidence",
            )
        if self._interrupted:
            return 130
        return _shell_returncode(child_returncode)

    def _receipt_request_binding(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "work_id": self.work_id,
            "receipt_path": os.fspath(self.receipts.completion),
            "runtime_config_path": os.fspath(self.runtime_config_path),
            "runtime_config_sha256": self.runtime_config_sha256,
            "worker_spec_sha256": self.worker_spec_sha256,
            "argv": list(self.argv),
            "argv_sha256": self.argv_sha256,
        }

    def _receipt_binding(
        self,
        lease: LeaseRecord,
        wrapper: Mapping[str, Any],
        *,
        binding_path: Path,
        binding_sha256: str,
    ) -> Dict[str, Any]:
        wrapper_identity = _validated_identity(wrapper, "GPU lease wrapper")
        return {
            **self._receipt_request_binding(),
            "lease_id": lease.lease_id,
            "expected_gpu_uuid": lease.expected_gpu_uuid,
            "trainer_binding_path": os.fspath(binding_path),
            "trainer_binding_sha256": binding_sha256,
            "wrapper_process_identity": wrapper_identity,
            "wrapper_process_identity_sha256": canonical_sha256(wrapper_identity),
        }

    def _validated_lease(self, value: Any) -> LeaseRecord:
        if not isinstance(value, LeaseRecord):
            required = ("lease_id", "expected_gpu_uuid")
            if any(not hasattr(value, field) for field in required):
                raise GpuLeaseWorkerError(
                    "invalid_lease", "GPU lease manager yielded a malformed lease"
                )
        lease = value
        _identifier(lease.lease_id, "lease_id")
        if lease.expected_gpu_uuid != self.config.expected_gpu_uuid:
            raise GpuLeaseWorkerError(
                "gpu_uuid_mismatch",
                "GPU lease UUID contradicts the runtime configuration",
            )
        if getattr(lease, "safety_halt", False):
            raise GpuLeaseWorkerError(
                "lease_safety_halt", "GPU lease manager yielded a safety halt"
            )
        return lease

    def _restoration_evidence(self, lease: LeaseRecord) -> Dict[str, Any]:
        final = self.manager.read_record()
        if final is None:
            raise GpuLeaseWorkerError(
                "missing_lease_state",
                "GPU lease state disappeared during trainer restoration",
            )
        if (
            final.lease_id != lease.lease_id
            or final.expected_gpu_uuid != self.config.expected_gpu_uuid
        ):
            raise GpuLeaseWorkerError(
                "lease_state_changed",
                "Final GPU lease state contradicts the guarded lifetime",
            )
        if final.safety_halt:
            raise GpuLeaseWorkerError(
                "lease_safety_halt",
                "GPU lease cleanup entered a safety halt",
                details={"reason": final.safety_reason},
            )
        if final.phase != "trainer_running" or final.restoration_status not in {
            "restored",
            "not_needed",
        }:
            raise GpuLeaseWorkerError(
                "trainer_not_restored",
                "GPU lease cleanup did not reach trainer_running",
                details={
                    "phase": final.phase,
                    "restoration_status": final.restoration_status,
                },
            )
        restored = final.restored_trainer or final.trainer
        if restored is None:
            raise GpuLeaseWorkerError(
                "trainer_not_restored",
                "GPU lease cleanup has no restored trainer identity",
            )
        current = self.manager.runner.current_identity(restored.pid)
        if current is None or not restored.same_process_as(current):
            raise GpuLeaseWorkerError(
                "trainer_not_restored",
                "Restored trainer identity is not live",
            )
        restored_identity = _lease_identity_dict(restored)
        return {
            "phase": final.phase,
            "restoration_status": final.restoration_status,
            "restored_trainer_identity": restored_identity,
            "restored_trainer_identity_sha256": canonical_sha256(restored_identity),
            "release_clean_observation_count": (final.release_clean_observation_count),
        }

    def _replay_or_refuse_incomplete(self) -> Optional[int]:
        request = self._receipt_request_binding()
        completion = _load_optional_canonical_object(
            self.receipts.completion, "guardian completion receipt"
        )
        ready = _load_optional_canonical_object(
            self.receipts.ready, "guardian ready receipt"
        )
        lifetime = _load_optional_canonical_object(
            self.receipts.lifetime, "guardian lifetime receipt"
        )
        for role, receipt in (
            ("guardian completion receipt", completion),
            ("guardian ready receipt", ready),
            ("guardian lifetime receipt", lifetime),
        ):
            if receipt is None:
                continue
            _validate_receipt(receipt, role)
            for key, expected in request.items():
                if receipt.get(key) != expected:
                    raise GpuLeaseWorkerError(
                        "receipt_binding_changed",
                        f"{role} contradicts {key}",
                    )
        if completion is not None:
            if (
                completion.get("contract") != COMPLETION_RECEIPT_CONTRACT
                or completion.get("phase") != "completion"
            ):
                raise GpuLeaseWorkerError(
                    "invalid_receipt",
                    "Guardian completion receipt contract is unsupported",
                )
            if ready is None:
                raise GpuLeaseWorkerError(
                    "invalid_receipt",
                    "Guardian completion has no ready receipt",
                )
            if ready.get("contract") != READY_RECEIPT_CONTRACT or completion.get(
                "ready_receipt_sha256"
            ) != ready.get("receipt_sha256"):
                raise GpuLeaseWorkerError(
                    "invalid_receipt",
                    "Guardian completion does not bind its ready receipt",
                )
            evidence_keys = (
                "lease_id",
                "expected_gpu_uuid",
                "trainer_binding_path",
                "trainer_binding_sha256",
                "wrapper_process_identity",
                "wrapper_process_identity_sha256",
            )
            for key in evidence_keys:
                if ready.get(key) != completion.get(key):
                    raise GpuLeaseWorkerError(
                        "invalid_receipt",
                        f"Guardian completion contradicts ready receipt {key}",
                    )
            wrapper = _validated_identity(
                completion.get("wrapper_process_identity"),
                "guardian receipt wrapper",
            )
            if wrapper["process_group_id"] != wrapper["pid"] or completion.get(
                "wrapper_process_identity_sha256"
            ) != canonical_sha256(wrapper):
                raise GpuLeaseWorkerError(
                    "invalid_receipt",
                    "Guardian receipt wrapper identity is invalid",
                )
            lifetime_hash = completion.get("lifetime_receipt_sha256")
            if lifetime_hash is None:
                if lifetime is not None:
                    raise GpuLeaseWorkerError(
                        "invalid_receipt",
                        "Guardian completion omits an existing lifetime receipt",
                    )
            elif (
                lifetime is None
                or lifetime.get("contract") != LIFETIME_RECEIPT_CONTRACT
                or lifetime.get("ready_receipt_sha256") != ready.get("receipt_sha256")
                or lifetime.get("receipt_sha256") != lifetime_hash
            ):
                raise GpuLeaseWorkerError(
                    "invalid_receipt",
                    "Guardian completion does not bind its lifetime receipt",
                )
            if lifetime is not None:
                for key in evidence_keys:
                    if lifetime.get(key) != completion.get(key):
                        raise GpuLeaseWorkerError(
                            "invalid_receipt",
                            f"Guardian lifetime contradicts completion {key}",
                        )
                child = _validated_identity(
                    lifetime.get("child_process_identity"),
                    "guardian receipt child",
                )
                if (
                    child["pid"] == wrapper["pid"]
                    or child["process_group_id"] != wrapper["process_group_id"]
                    or lifetime.get("child_process_identity_sha256")
                    != canonical_sha256(child)
                    or completion.get("child_process_identity_sha256")
                    != lifetime.get("child_process_identity_sha256")
                ):
                    raise GpuLeaseWorkerError(
                        "invalid_receipt",
                        "Guardian receipt child lifetime identity is invalid",
                    )
            status = completion.get("status")
            if (
                status not in _REPLAYABLE_COMPLETION_STATUSES
                or completion.get("trainer_restored") is not True
                or not isinstance(completion.get("restoration"), Mapping)
            ):
                raise GpuLeaseWorkerError(
                    "unsafe_completion",
                    "Guardian completion does not prove safe trainer restoration",
                )
            returncode = completion.get("returncode")
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise GpuLeaseWorkerError(
                    "invalid_receipt",
                    "Guardian completion return code is malformed",
                )
            return returncode
        if ready is not None or lifetime is not None:
            try:
                self.manager.reconcile(mutate=True)
            except BaseException as exc:
                raise GpuLeaseWorkerError(
                    "incomplete_guardian_receipt",
                    "A previous guardian lifetime is incomplete and reconciliation "
                    "failed",
                    details={"reconcile_error": _error_text(exc)},
                ) from exc
            raise GpuLeaseWorkerError(
                "incomplete_guardian_receipt",
                "A previous guardian lifetime is incomplete; duplicate spawn refused",
            )
        return None

    @contextlib.contextmanager
    def _sigint_handler(self) -> Iterator[None]:
        if not self.install_signal_handlers:
            yield
            return
        try:
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
        except (ValueError, OSError) as exc:
            raise GpuLeaseWorkerError(
                "signal_handler_unavailable",
                "GPU lease worker must install a SIGINT handler",
            ) from exc
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)

    def _handle_sigint(self, _signum: int, _frame: Optional[FrameType]) -> None:
        first_interrupt = not self._interrupted
        self._interrupted = True
        child = self._child_identity
        if first_interrupt and child is not None:
            try:
                self.process_runner.send_signal(child, signal.SIGINT)
            except BaseException as exc:
                self._signal_error = exc

    def _now(self) -> float:
        value = float(self.clock())
        if not math.isfinite(value):
            raise GpuLeaseWorkerError(
                "invalid_clock", "Worker clock returned a non-finite value"
            )
        return value


def _error_text(exc: Optional[BaseException]) -> Optional[str]:
    if exc is None:
        return None
    return f"{type(exc).__name__}: {exc}"


def _shell_returncode(returncode: Optional[int]) -> int:
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise GpuLeaseWorkerError(
            "invalid_child_returncode", "Child process has no valid return code"
        )
    return returncode if returncode >= 0 else 128 + min(abs(returncode), 127)


def _parse_command_json(value: str) -> Tuple[str, ...]:
    try:
        decoded = json.loads(value, parse_constant=_reject_constant)
    except (json.JSONDecodeError, GpuLeaseWorkerError) as exc:
        raise GpuLeaseWorkerError(
            "invalid_command", "--command-json must encode one argv array"
        ) from exc
    return _validate_command(decoded)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    remainder: Tuple[str, ...] = ()
    if "--" in raw:
        marker = raw.index("--")
        remainder = _validate_command(raw[marker + 1 :])
        raw = raw[:marker]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path)
    source.add_argument("--config", "--runtime-config", dest="config", type=Path)
    parser.add_argument("--expected-spec-sha256")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--trainer-binding", type=Path)
    parser.add_argument(
        "--receipt", "--guardian-receipt", dest="receipt", required=True, type=Path
    )
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--command-json")
    args = parser.parse_args(raw)
    if args.spec is not None:
        if args.expected_spec_sha256 is None:
            parser.error("--spec requires --expected-spec-sha256")
        if args.expected_config_sha256 is not None:
            parser.error("--expected-config-sha256 is only valid with --config")
    else:
        if args.expected_config_sha256 is None:
            parser.error("--config requires --expected-config-sha256")
        if args.expected_spec_sha256 is not None:
            parser.error("--expected-spec-sha256 is only valid with --spec")
    if args.command_json is not None and remainder:
        parser.error("use exactly one of --command-json or -- argv")
    if args.command_json is None and not remainder:
        parser.error("a child argv is required via --command-json or --")
    args.child_argv = (
        _parse_command_json(args.command_json)
        if args.command_json is not None
        else remainder
    )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        if args.spec is not None:
            spec = load_worker_spec(
                args.spec,
                expected_spec_sha256=args.expected_spec_sha256,
            )
            runtime_path = spec.runtime_config
            runtime_hash = spec.runtime_config_sha256
            worker_spec_hash: Optional[str] = spec.spec_sha256
            receipt_directory: Optional[Path] = spec.receipt_directory
        else:
            runtime_path = _normalized_absolute(
                args.config, "GPU lease runtime config", require_file=True
            )
            runtime_hash = args.expected_config_sha256
            worker_spec_hash = None
            receipt_directory = None
        config = load_runtime_config(runtime_path, expected_sha256=runtime_hash)
        worker = GpuLeaseWorker(
            config,
            runtime_config_path=runtime_path,
            runtime_config_sha256=runtime_hash,
            claim_id=args.claim_id,
            work_id=args.work_id,
            receipt_path=args.receipt,
            argv=args.child_argv,
            worker_spec_sha256=worker_spec_hash,
            receipt_directory=receipt_directory,
            trainer_binding_path=args.trainer_binding,
        )
        return worker.run()
    except GpuLeaseWorkerError as exc:
        sys.stderr.buffer.write(canonical_json_bytes(exc.to_dict()) + b"\n")
        return 2
    except GpuLeaseError as exc:
        error = GpuLeaseWorkerError(
            "gpu_lease_failed",
            str(exc),
            details={"gpu_lease_code": exc.code},
        )
        sys.stderr.buffer.write(canonical_json_bytes(error.to_dict()) + b"\n")
        return 2
    except OSError as exc:
        error = GpuLeaseWorkerError("os_error", str(exc))
        sys.stderr.buffer.write(canonical_json_bytes(error.to_dict()) + b"\n")
        return 2


__all__ = [
    "COMPLETION_RECEIPT_CONTRACT",
    "GpuLeaseWorker",
    "GpuLeaseWorkerError",
    "GuardianProcessRunner",
    "GuardianReceiptPaths",
    "LIFETIME_RECEIPT_CONTRACT",
    "READY_RECEIPT_CONTRACT",
    "SCHEMA_VERSION",
    "SubprocessGuardianRunner",
    "WORKER_SPEC_CONTRACT",
    "WorkerSpec",
    "guardian_receipt_paths",
    "load_runtime_config",
    "load_supervised_trainer_identity",
    "load_worker_spec",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
