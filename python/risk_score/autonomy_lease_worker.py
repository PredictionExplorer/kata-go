"""Hold one verified CUDA context for an autonomy evaluator lease.

The worker is the sole persistent evaluator process used by the dedicated
trainer/evaluator lease drill. It loads the CUDA driver directly with
``ctypes``, creates a context under its own PID, publishes a canonical
readiness record, and waits for SIGINT or SIGTERM. It never starts a child
process and uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import signal
import stat
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol

SCHEMA_VERSION = 1
READY_CONTRACT = "risk-score-autonomy-lease-worker-ready-v1"
ERROR_CONTRACT = "risk-score-autonomy-lease-worker-error-v1"
MAX_READY_BYTES = 64 * 1024

_SHA256_LENGTH = 64
_PR_SET_PDEATHSIG = 1


class LeaseWorkerError(RuntimeError):
    """Fail-closed worker error with a stable machine-readable code."""

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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise LeaseWorkerError(
            "module_verification_failed",
            f"could not hash worker module: {exc}",
        ) from exc
    return digest.hexdigest()


def _required_sha256(value: str, role: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LeaseWorkerError(
            "invalid_arguments", f"{role} must be a lowercase SHA-256"
        )
    return value


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        try:
            if current.is_symlink():
                raise LeaseWorkerError(
                    "unsafe_path",
                    f"{role} has a symlinked path component",
                    details={"path": os.fspath(current)},
                )
        except OSError as exc:
            raise LeaseWorkerError(
                "unsafe_path",
                f"could not inspect {role}: {exc}",
            ) from exc
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(path: Path, role: str) -> Path:
    source = Path(path)
    normalized = Path(os.path.abspath(os.fspath(source)))
    if not source.is_absolute() or source != normalized:
        raise LeaseWorkerError(
            "unsafe_path", f"{role} must be absolute and lexically normalized"
        )
    _reject_symlink_ancestors(source, role)
    return source


@dataclass(frozen=True)
class LinuxProcessIdentity:
    pid: int
    start_time_ticks: int
    process_group_id: int
    boot_id: str
    command_sha256: str
    cgroup: str

    @classmethod
    def capture(cls, pid: int) -> LinuxProcessIdentity:
        if sys.platform != "linux":
            raise LeaseWorkerError(
                "unsupported_platform",
                "the CUDA lease worker requires Linux /proc process identity",
            )
        proc = Path("/proc") / str(pid)
        try:
            stat_text = (proc / "stat").read_text()
            closing = stat_text.rfind(")")
            if closing < 0:
                raise ValueError("missing process-name terminator")
            stat_fields = stat_text[closing + 2 :].split()
            command = (proc / "cmdline").read_bytes()
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            cgroup = (proc / "cgroup").read_text().strip()
        except (OSError, ValueError) as exc:
            raise LeaseWorkerError(
                "process_identity_unavailable",
                f"could not capture worker process identity: {exc}",
            ) from exc
        if len(stat_fields) < 20:
            raise LeaseWorkerError(
                "process_identity_unavailable",
                "worker /proc stat record is truncated",
            )
        return cls(
            pid=pid,
            start_time_ticks=int(stat_fields[19]),
            process_group_id=os.getpgid(pid),
            boot_id=boot_id,
            command_sha256=hashlib.sha256(command).hexdigest(),
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


class CudaContextDriver(Protocol):
    def acquire(self, expected_gpu_uuid: str) -> object: ...

    def synchronize(self) -> None: ...

    def release(self, context: object) -> None: ...


class _CuUuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


class ParentDeathGuard(Protocol):
    def arm(self) -> int: ...


class LinuxParentDeathGuard:
    """Arm an irrevocable kernel signal if the launching controller dies."""

    def __init__(
        self,
        *,
        library_loader: Callable[..., Any] = ctypes.CDLL,
        parent_pid: Callable[[], int] = os.getppid,
        terminate_self: Callable[[], NoReturn] | None = None,
        platform: str = sys.platform,
    ) -> None:
        if platform != "linux":
            raise LeaseWorkerError(
                "unsupported_platform",
                "the CUDA lease worker requires Linux PR_SET_PDEATHSIG",
            )
        self._parent_pid = parent_pid
        self._terminate_self = terminate_self or self._kill_self
        try:
            self._libc = library_loader(None, use_errno=True)
        except OSError as exc:
            raise LeaseWorkerError(
                "parent_death_guard_unavailable",
                f"could not load libc for PR_SET_PDEATHSIG: {exc}",
            ) from exc
        self._libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        self._libc.prctl.restype = ctypes.c_int

    @staticmethod
    def _kill_self() -> NoReturn:
        os.kill(os.getpid(), signal.SIGKILL)
        os._exit(128 + signal.SIGKILL)

    def arm(self) -> int:
        parent_before = self._parent_pid()
        if parent_before <= 1:
            raise LeaseWorkerError(
                "controller_parent_unavailable",
                "lease worker has no live controller parent",
                details={"parent_pid": parent_before},
            )
        if self._libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise LeaseWorkerError(
                "parent_death_guard_failed",
                "could not arm PR_SET_PDEATHSIG",
                details={"errno": error_number},
            )
        parent_after = self._parent_pid()
        if parent_after != parent_before or parent_after <= 1:
            self._terminate_self()
            raise LeaseWorkerError(
                "controller_parent_changed",
                "controller parent changed while PR_SET_PDEATHSIG was armed",
                details={"before": parent_before, "after": parent_after},
            )
        return parent_before


class CtypesCudaDriver:
    """Minimal CUDA driver API wrapper for one directly owned context."""

    def __init__(
        self,
        library_loader: Callable[..., Any] = ctypes.CDLL,
    ) -> None:
        try:
            self._cuda = library_loader("libcuda.so.1", use_errno=True)
        except OSError as exc:
            raise LeaseWorkerError(
                "cuda_driver_unavailable", f"could not load libcuda.so.1: {exc}"
            ) from exc
        self._bind()

    def _function(
        self,
        *names: str,
        argtypes: Sequence[Any],
        restype: Any = ctypes.c_int,
    ) -> Any:
        for name in names:
            function = getattr(self._cuda, name, None)
            if function is not None:
                function.argtypes = list(argtypes)
                function.restype = restype
                return function
        raise LeaseWorkerError(
            "cuda_driver_incompatible",
            f"CUDA driver does not export any of {', '.join(names)}",
        )

    def _bind(self) -> None:
        self._cu_init = self._function("cuInit", argtypes=(ctypes.c_uint,))
        self._cu_device_count = self._function(
            "cuDeviceGetCount", argtypes=(ctypes.POINTER(ctypes.c_int),)
        )
        self._cu_device_get = self._function(
            "cuDeviceGet",
            argtypes=(ctypes.POINTER(ctypes.c_int), ctypes.c_int),
        )
        self._cu_device_uuid = self._function(
            "cuDeviceGetUuid_v2",
            "cuDeviceGetUuid",
            argtypes=(ctypes.POINTER(_CuUuid), ctypes.c_int),
        )
        self._cu_context_create = self._function(
            "cuCtxCreate_v2",
            "cuCtxCreate",
            argtypes=(
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
                ctypes.c_int,
            ),
        )
        self._cu_context_destroy = self._function(
            "cuCtxDestroy_v2",
            "cuCtxDestroy",
            argtypes=(ctypes.c_void_p,),
        )
        self._cu_context_synchronize = self._function("cuCtxSynchronize", argtypes=())
        self._cu_error_name = getattr(self._cuda, "cuGetErrorName", None)
        if self._cu_error_name is not None:
            self._cu_error_name.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char_p),
            ]
            self._cu_error_name.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        name = None
        if self._cu_error_name is not None:
            value = ctypes.c_char_p()
            if self._cu_error_name(result, ctypes.byref(value)) == 0 and value.value:
                name = value.value.decode("ascii", errors="replace")
        raise LeaseWorkerError(
            "cuda_operation_failed",
            f"{operation} failed with CUDA result {result}"
            + ("" if name is None else f" ({name})"),
            details={"operation": operation, "cuda_result": result},
        )

    @staticmethod
    def _expected_uuid_bytes(expected_gpu_uuid: str) -> bytes:
        if not expected_gpu_uuid.startswith("GPU-"):
            raise LeaseWorkerError(
                "invalid_arguments",
                "expected GPU UUID must use the canonical GPU- UUID form",
            )
        try:
            return uuid.UUID(expected_gpu_uuid.removeprefix("GPU-")).bytes
        except ValueError as exc:
            raise LeaseWorkerError(
                "invalid_arguments",
                "expected GPU UUID must contain a canonical hexadecimal UUID",
            ) from exc

    def acquire(self, expected_gpu_uuid: str) -> object:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible not in {"", expected_gpu_uuid}:
            raise LeaseWorkerError(
                "ambiguous_cuda_inventory",
                "CUDA_VISIBLE_DEVICES must be unset or exactly the expected GPU UUID",
                details={"CUDA_VISIBLE_DEVICES": visible},
            )
        expected_uuid = self._expected_uuid_bytes(expected_gpu_uuid)
        self._check(self._cu_init(0), "cuInit")
        count = ctypes.c_int()
        self._check(self._cu_device_count(ctypes.byref(count)), "cuDeviceGetCount")
        selected_device = None
        matching_ordinals = []
        for ordinal in range(count.value):
            device = ctypes.c_int()
            self._check(
                self._cu_device_get(ctypes.byref(device), ordinal),
                "cuDeviceGet",
            )
            observed_uuid = _CuUuid()
            self._check(
                self._cu_device_uuid(ctypes.byref(observed_uuid), device.value),
                "cuDeviceGetUuid",
            )
            if bytes(observed_uuid.bytes) == expected_uuid:
                matching_ordinals.append(ordinal)
                selected_device = device.value
        if len(matching_ordinals) != 1 or selected_device is None:
            raise LeaseWorkerError(
                "gpu_uuid_mismatch",
                "expected GPU UUID was not uniquely present in CUDA visibility",
                details={
                    "expected_gpu_uuid": expected_gpu_uuid,
                    "visible_device_count": count.value,
                    "matching_ordinals": matching_ordinals,
                },
            )
        context = ctypes.c_void_p()
        self._check(
            self._cu_context_create(ctypes.byref(context), 0, selected_device),
            "cuCtxCreate",
        )
        return context

    def synchronize(self) -> None:
        self._check(self._cu_context_synchronize(), "cuCtxSynchronize")

    def release(self, context: object) -> None:
        pointer = context if isinstance(context, ctypes.c_void_p) else ctypes.c_void_p()
        self._check(self._cu_context_destroy(pointer), "cuCtxDestroy")


def _write_ready(path: Path, value: Mapping[str, Any]) -> bytes:
    destination = _absolute_path(path, "readiness path")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise LeaseWorkerError(
            "unsafe_path", "readiness parent must be an existing non-symlink directory"
        )
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise LeaseWorkerError(
            "unsafe_path", "existing readiness path must be a regular file"
        )
    data = (canonical_json(value) + "\n").encode()
    if len(data) > MAX_READY_BYTES:
        raise LeaseWorkerError(
            "readiness_publish_failed", "readiness record is too large"
        )
    temporary = parent / f".{destination.name}.tmp-{os.getpid()}"
    _reject_symlink_ancestors(temporary, "temporary readiness path")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise LeaseWorkerError(
            "readiness_publish_failed", f"could not publish readiness: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink()
    return data


class LeaseSentinel:
    """Own a CUDA context and a canonical readiness record until stopped."""

    def __init__(
        self,
        *,
        expected_gpu_uuid: str,
        gpu_index: int,
        ready_path: Path,
        module_sha256: str,
        cuda: CudaContextDriver | None = None,
        parent_death_guard: ParentDeathGuard | None = None,
        identity_capture: Callable[[int], LinuxProcessIdentity] = (
            LinuxProcessIdentity.capture
        ),
    ) -> None:
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
        ):
            raise LeaseWorkerError(
                "invalid_arguments", "GPU index must be a nonnegative integer"
            )
        self.expected_gpu_uuid = expected_gpu_uuid
        self.gpu_index = gpu_index
        self.ready_path = _absolute_path(ready_path, "readiness path")
        self.module_sha256 = _required_sha256(module_sha256, "module SHA-256")
        self.cuda = CtypesCudaDriver() if cuda is None else cuda
        self.parent_death_guard = (
            LinuxParentDeathGuard()
            if parent_death_guard is None
            else parent_death_guard
        )
        self.identity_capture = identity_capture
        self._stopped = threading.Event()

    def request_stop(self, _signum: int | None = None, _frame: object = None) -> None:
        self._stopped.set()

    def _wait(self) -> None:
        while not self._stopped.wait(60.0):
            continue

    def run(self, *, waiter: Callable[[], None] | None = None) -> None:
        context: object | None = None
        published_data: bytes | None = None
        try:
            # Arm before any CUDA call so controller death can never leave a
            # context-owning sentinel behind.
            self.parent_death_guard.arm()
            context = self.cuda.acquire(self.expected_gpu_uuid)
            self.cuda.synchronize()
            identity = self.identity_capture(os.getpid())
            ready: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "contract": READY_CONTRACT,
                "gpu_uuid": self.expected_gpu_uuid,
                "gpu_index": self.gpu_index,
                "pid": os.getpid(),
                "process_identity": identity.to_dict(),
                "module_sha256": self.module_sha256,
            }
            ready["ready_sha256"] = canonical_sha256(ready)
            published_data = _write_ready(self.ready_path, ready)
            (self._wait if waiter is None else waiter)()
        finally:
            if published_data is not None:
                try:
                    if self.ready_path.read_bytes() == published_data:
                        self.ready_path.unlink()
                except OSError:
                    pass
            if context is not None:
                self.cuda.release(context)


def lease_worker_module_path() -> Path:
    path = Path(__file__)
    if not path.is_absolute():
        path = Path(os.path.abspath(os.fspath(path)))
    return _absolute_path(path, "worker module")


def evaluator_sentinel_argv(
    *,
    python_executable: Path,
    expected_gpu_uuid: str,
    expected_gpu_index: int,
    ready_path: Path,
    worker_module: Path | None = None,
) -> tuple[str, ...]:
    """Return the exact processCount=1 evaluator launch command."""

    python = _absolute_path(python_executable, "Python executable")
    module = (
        lease_worker_module_path()
        if worker_module is None
        else _absolute_path(worker_module, "worker module")
    )
    for path, role in ((python, "Python executable"), (module, "worker module")):
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise LeaseWorkerError(
                "unsafe_path", f"could not inspect {role}: {exc}"
            ) from exc
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise LeaseWorkerError("unsafe_path", f"{role} must be a regular file")
    readiness = _absolute_path(ready_path, "readiness path")
    return (
        str(python),
        str(module),
        "--expected-module-sha256",
        file_sha256(module),
        "--expected-gpu-uuid",
        expected_gpu_uuid,
        "--expected-gpu-index",
        str(expected_gpu_index),
        "--ready",
        str(readiness),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-module-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-gpu-index", required=True, type=int)
    parser.add_argument("--ready", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_args(argv)
        expected_module_hash = _required_sha256(
            arguments.expected_module_sha256, "expected module SHA-256"
        )
        module = lease_worker_module_path()
        observed_module_hash = file_sha256(module)
        if observed_module_hash != expected_module_hash:
            raise LeaseWorkerError(
                "module_verification_failed",
                "worker module does not match its launch-command binding",
                details={
                    "expected": expected_module_hash,
                    "observed": observed_module_hash,
                },
            )
        sentinel = LeaseSentinel(
            expected_gpu_uuid=arguments.expected_gpu_uuid,
            gpu_index=arguments.expected_gpu_index,
            ready_path=arguments.ready,
            module_sha256=observed_module_hash,
        )
        previous_handlers: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, sentinel.request_stop)
        try:
            sentinel.run()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        return 0
    except (LeaseWorkerError, OSError, ValueError) as exc:
        error = (
            exc
            if isinstance(exc, LeaseWorkerError)
            else LeaseWorkerError("worker_failed", str(exc))
        )
        sys.stderr.write(canonical_json(error.to_dict()) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
