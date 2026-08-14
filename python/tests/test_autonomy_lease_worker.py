import contextlib
import hashlib
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import pytest

from risk_score import autonomy_lease_worker as worker
from risk_score.autonomy_lease_worker import (
    READY_CONTRACT,
    CtypesCudaDriver,
    LeaseSentinel,
    LeaseWorkerError,
    LinuxParentDeathGuard,
    LinuxProcessIdentity,
    canonical_sha256,
    evaluator_sentinel_argv,
    main,
)

GPU_UUID = "GPU-12345678-1234-5678-9abc-def012345678"
GPU_INDEX = 2
MODULE_HASH = "a" * 64


def process_identity(pid):
    return LinuxProcessIdentity(
        pid=pid,
        start_time_ticks=12345,
        process_group_id=pid,
        boot_id="boot-worker-test",
        command_sha256=hashlib.sha256(b"worker").hexdigest(),
        cgroup="/worker/test",
    )


class FakeCuda:
    def __init__(self, *, error=None, events=None):
        self.error = error
        self.calls = []
        self.events = events
        self.context = object()

    def acquire(self, expected_gpu_uuid):
        self.calls.append(("acquire", expected_gpu_uuid))
        if self.events is not None:
            self.events.append("cuda")
        if self.error is not None:
            raise self.error
        return self.context

    def synchronize(self):
        self.calls.append(("synchronize",))

    def release(self, context):
        self.calls.append(("release", context))


class FakeParentDeathGuard:
    def __init__(self, events=None):
        self.calls = 0
        self.events = events

    def arm(self):
        self.calls += 1
        if self.events is not None:
            self.events.append("guard")
        return 77


class FakeCudaFunction:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        self.calls.append(arguments)
        return self.callback(*arguments)


class FakeCudaLibrary:
    def __init__(self, gpu_uuids):
        raw_uuids = [
            uuid.UUID(gpu_uuid.removeprefix("GPU-")).bytes for gpu_uuid in gpu_uuids
        ]
        self.cuInit = FakeCudaFunction(lambda _flags: 0)

        def device_count(pointer):
            pointer._obj.value = len(raw_uuids)
            return 0

        def device_get(pointer, index):
            pointer._obj.value = index
            return 0

        def device_uuid(pointer, device):
            for index, value in enumerate(raw_uuids[device]):
                pointer._obj.bytes[index] = value
            return 0

        def context_create(pointer, _flags, _device):
            pointer._obj.value = 123
            return 0

        self.cuDeviceGetCount = FakeCudaFunction(device_count)
        self.cuDeviceGet = FakeCudaFunction(device_get)
        self.cuDeviceGetUuid_v2 = FakeCudaFunction(device_uuid)
        self.cuCtxCreate_v2 = FakeCudaFunction(context_create)
        self.cuCtxDestroy_v2 = FakeCudaFunction(lambda _context: 0)
        self.cuCtxSynchronize = FakeCudaFunction(lambda: 0)


def test_ctypes_driver_selects_uuid_without_assuming_cuda_ordinal(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    library = FakeCudaLibrary(
        [
            "GPU-00000000-0000-0000-0000-000000000000",
            GPU_UUID,
            "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
        ]
    )
    driver = CtypesCudaDriver(library_loader=lambda _name, **_kwargs: library)

    context = driver.acquire(GPU_UUID)
    driver.synchronize()
    driver.release(context)

    assert context.value == 123
    assert [call[1] for call in library.cuDeviceGet.calls] == [0, 1, 2]
    assert library.cuCtxCreate_v2.calls[0][2] == 1
    assert len(library.cuCtxCreate_v2.calls) == 1
    assert len(library.cuCtxDestroy_v2.calls) == 1


def test_ctypes_driver_handles_cuda_visibility_only_by_exact_uuid(monkeypatch):
    library = FakeCudaLibrary([GPU_UUID])
    driver = CtypesCudaDriver(library_loader=lambda _name, **_kwargs: library)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    context = driver.acquire(GPU_UUID)
    driver.release(context)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(GPU_INDEX))
    with pytest.raises(LeaseWorkerError) as raised:
        driver.acquire(GPU_UUID)
    assert raised.value.code == "ambiguous_cuda_inventory"


def test_sentinel_owns_context_publishes_identity_and_cleans_up(tmp_path):
    ready = tmp_path / "ready.json"
    cuda = FakeCuda()
    guard = FakeParentDeathGuard()
    observed = {}
    sentinel = LeaseSentinel(
        expected_gpu_uuid=GPU_UUID,
        gpu_index=GPU_INDEX,
        ready_path=ready,
        module_sha256=MODULE_HASH,
        cuda=cuda,
        parent_death_guard=guard,
        identity_capture=process_identity,
    )

    def inspect_ready():
        observed.update(json.loads(ready.read_bytes()))
        sentinel.request_stop()

    sentinel.run(waiter=inspect_ready)

    assert observed["contract"] == READY_CONTRACT
    assert observed["gpu_uuid"] == GPU_UUID
    assert observed["gpu_index"] == GPU_INDEX
    assert observed["pid"] == os.getpid()
    assert observed["process_identity"] == process_identity(os.getpid()).to_dict()
    supplied = observed.pop("ready_sha256")
    assert supplied == canonical_sha256(observed)
    assert not ready.exists()
    assert guard.calls == 1
    assert cuda.calls == [
        ("acquire", GPU_UUID),
        ("synchronize",),
        ("release", cuda.context),
    ]


def test_sentinel_never_publishes_readiness_when_cuda_acquisition_fails(tmp_path):
    ready = tmp_path / "ready.json"
    failure = LeaseWorkerError("cuda_operation_failed", "no context")
    sentinel = LeaseSentinel(
        expected_gpu_uuid=GPU_UUID,
        gpu_index=GPU_INDEX,
        ready_path=ready,
        module_sha256=MODULE_HASH,
        cuda=FakeCuda(error=failure),
        parent_death_guard=FakeParentDeathGuard(),
        identity_capture=process_identity,
    )

    with pytest.raises(LeaseWorkerError) as raised:
        sentinel.run(waiter=lambda: None)

    assert raised.value is failure
    assert not ready.exists()


def test_parent_death_guard_is_armed_before_cuda_acquisition(tmp_path):
    events = []
    sentinel = LeaseSentinel(
        expected_gpu_uuid=GPU_UUID,
        gpu_index=GPU_INDEX,
        ready_path=tmp_path / "ready.json",
        module_sha256=MODULE_HASH,
        cuda=FakeCuda(events=events),
        parent_death_guard=FakeParentDeathGuard(events),
        identity_capture=process_identity,
    )

    sentinel.run(waiter=lambda: None)

    assert events == ["guard", "cuda"]


def test_parent_death_guard_fails_if_parent_changes_while_arming():
    libc = type(
        "FakeLibc",
        (),
        {"prctl": FakeCudaFunction(lambda *_arguments: 0)},
    )()
    parents = iter((700, 1))

    def terminate():
        raise RuntimeError("terminated")

    guard = LinuxParentDeathGuard(
        library_loader=lambda _name, **_kwargs: libc,
        parent_pid=lambda: next(parents),
        terminate_self=terminate,
        platform="linux",
    )

    with pytest.raises(RuntimeError, match="terminated"):
        guard.arm()

    assert len(libc.prctl.calls) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux PR_SET_PDEATHSIG")
def test_controller_death_kills_context_owning_sentinel(tmp_path):
    ready = tmp_path / "fork-ready.json"
    read_descriptor, write_descriptor = os.pipe()
    controller_pid = os.fork()
    if controller_pid == 0:
        os.close(read_descriptor)
        sentinel_pid = os.fork()
        if sentinel_pid == 0:
            os.close(write_descriptor)
            try:
                LeaseSentinel(
                    expected_gpu_uuid=GPU_UUID,
                    gpu_index=GPU_INDEX,
                    ready_path=ready,
                    module_sha256=MODULE_HASH,
                    cuda=FakeCuda(),
                    identity_capture=LinuxProcessIdentity.capture,
                ).run()
            finally:
                os._exit(90)
        os.write(write_descriptor, str(sentinel_pid).encode())
        os.close(write_descriptor)
        while True:
            signal.pause()

    os.close(write_descriptor)
    sentinel_pid = int(os.read(read_descriptor, 32))
    os.close(read_descriptor)
    try:
        deadline = time.monotonic() + 5.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "sentinel never acquired fake CUDA and became ready"

        os.kill(controller_pid, signal.SIGKILL)
        os.waitpid(controller_pid, 0)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            proc_stat = Path(f"/proc/{sentinel_pid}/stat")
            if not proc_stat.exists():
                break
            stat_text = proc_stat.read_text()
            closing = stat_text.rfind(")")
            if closing >= 0 and stat_text[closing + 2 :].startswith("Z"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("sentinel retained its CUDA-owning process after parent death")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(controller_pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            os.kill(sentinel_pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(controller_pid, 0)
        ready.unlink(missing_ok=True)


def test_sentinel_launch_argv_binds_absolute_worker_module_and_hash(tmp_path):
    ready = tmp_path / "ready.json"
    argv = evaluator_sentinel_argv(
        python_executable=Path(sys.executable).resolve(),
        expected_gpu_uuid=GPU_UUID,
        expected_gpu_index=GPU_INDEX,
        ready_path=ready,
    )

    assert argv[0] == str(Path(sys.executable).resolve())
    assert Path(argv[1]).is_absolute()
    assert argv[2:4] == (
        "--expected-module-sha256",
        worker.file_sha256(Path(argv[1])),
    )
    assert argv[-2:] == ("--ready", str(ready))
    assert "-m" not in argv


def test_worker_cli_rejects_module_hash_drift_without_loading_cuda(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(worker, "file_sha256", lambda _path: "b" * 64)

    result = main(
        [
            "--expected-module-sha256",
            "c" * 64,
            "--expected-gpu-uuid",
            GPU_UUID,
            "--expected-gpu-index",
            str(GPU_INDEX),
            "--ready",
            str(tmp_path / "ready.json"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"]["code"] == "module_verification_failed"


def test_readiness_path_symlink_is_rejected(tmp_path):
    real = tmp_path / "real-ready.json"
    real.write_text("{}\n")
    linked = tmp_path / "linked-ready.json"
    linked.symlink_to(real)

    with pytest.raises(LeaseWorkerError) as raised:
        LeaseSentinel(
            expected_gpu_uuid=GPU_UUID,
            gpu_index=GPU_INDEX,
            ready_path=linked,
            module_sha256=MODULE_HASH,
            cuda=FakeCuda(),
            parent_death_guard=FakeParentDeathGuard(),
            identity_capture=process_identity,
        )

    assert raised.value.code == "unsafe_path"
