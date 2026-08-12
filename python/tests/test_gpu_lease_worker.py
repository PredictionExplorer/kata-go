import contextlib
import hashlib
import json

import pytest

from risk_score.cluster_scheduler import canonical_json_bytes, canonical_sha256
from risk_score.gpu_lease import LeaseRecord, ProcessIdentity, RuntimeConfig
from risk_score.gpu_lease_worker import (
    COMPLETION_RECEIPT_CONTRACT,
    LIFETIME_RECEIPT_CONTRACT,
    READY_RECEIPT_CONTRACT,
    WORKER_SPEC_CONTRACT,
    GpuLeaseWorker,
    GpuLeaseWorkerError,
    guardian_receipt_paths,
    load_worker_spec,
    parse_args,
)


def identity_dict(pid, *, role="process", process_group_id=None):
    return {
        "pid": pid,
        "start_time_ticks": pid * 10,
        "command_sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
        "process_group_id": pid if process_group_id is None else process_group_id,
        "boot_id": "boot-test",
        "cgroup": f"0::/test/{role}",
    }


def lease_identity(pid, *, role="process", process_group_id=None):
    value = identity_dict(pid, role=role, process_group_id=process_group_id)
    return ProcessIdentity(
        pid=value["pid"],
        start_time_ticks=value["start_time_ticks"],
        process_group_id=value["process_group_id"],
        boot_id=value["boot_id"],
        command_sha256=value["command_sha256"],
        cgroup=value["cgroup"],
    )


class FakeLeaseRunner:
    def __init__(self):
        self.processes = {}

    def add(self, identity):
        self.processes[identity.pid] = identity
        return identity

    def current_identity(self, pid):
        return self.processes.get(pid)


class FakeManager:
    def __init__(self, config, events, *, restore=True):
        self.config = config
        self.events = events
        self.restore = restore
        self.runner = FakeLeaseRunner()
        self.in_handoff = False
        self.record = None
        self.reconcile_calls = 0
        self.restored = lease_identity(701, role="restored-trainer")

    @contextlib.contextmanager
    def exclusive_handoff(self, trainer):
        self.events.append("handoff-enter")
        self.in_handoff = True
        self.runner.processes.pop(trainer.pid, None)
        record = LeaseRecord(
            lease_id="lease-test",
            owner_id="guardian-test",
            phase="leased",
            expected_gpu_uuid=self.config.expected_gpu_uuid,
            trainer=trainer,
            evaluators=(),
            checkpoint_sha256="c" * 64,
            checkpoint_size=1,
            safety_halt=False,
            safety_reason=None,
            created_at=1.0,
            updated_at=2.0,
            restoration_status="pending",
        )
        self.record = record
        try:
            yield record
        finally:
            if self.restore:
                self.runner.add(self.restored)
                self.record = LeaseRecord(
                    lease_id=record.lease_id,
                    owner_id=record.owner_id,
                    phase="trainer_running",
                    expected_gpu_uuid=record.expected_gpu_uuid,
                    trainer=self.restored,
                    evaluators=(),
                    checkpoint_sha256=record.checkpoint_sha256,
                    checkpoint_size=record.checkpoint_size,
                    safety_halt=False,
                    safety_reason=None,
                    created_at=record.created_at,
                    updated_at=3.0,
                    release_clean_observation_times=(2.5, 2.75),
                    restoration_status="restored",
                    restored_trainer=self.restored,
                )
            else:
                self.record = LeaseRecord(
                    lease_id=record.lease_id,
                    owner_id=record.owner_id,
                    phase="safety_halt",
                    expected_gpu_uuid=record.expected_gpu_uuid,
                    trainer=trainer,
                    evaluators=(),
                    checkpoint_sha256=record.checkpoint_sha256,
                    checkpoint_size=record.checkpoint_size,
                    safety_halt=True,
                    safety_reason="injected restoration failure",
                    created_at=record.created_at,
                    updated_at=3.0,
                    restoration_status="safety_halt",
                )
            self.in_handoff = False
            self.events.append("handoff-exit")

    def read_record(self):
        return self.record

    def reconcile(self, *, mutate):
        assert mutate is True
        self.reconcile_calls += 1
        self.events.append("reconcile")


class FakeGuardianRunner:
    def __init__(self, manager, events, *, returncode=0, wrapper_pid=500):
        self.manager = manager
        self.events = events
        self.returncode = returncode
        self.wrapper = identity_dict(wrapper_pid, role="guardian")
        self.child = identity_dict(
            wrapper_pid + 1,
            role="child",
            process_group_id=wrapper_pid,
        )
        self.spawns = []
        self.signals = []
        self.on_wait = None

    def wrapper_identity(self):
        return dict(self.wrapper)

    def spawn(self, argv, *, process_group_id):
        assert self.manager.in_handoff is True
        assert process_group_id == self.wrapper["pid"]
        self.events.append("spawn")
        self.spawns.append(list(argv))
        return dict(self.child)

    def wait(self, identity):
        assert self.manager.in_handoff is True
        assert dict(identity) == self.child
        self.events.append("wait")
        if self.on_wait is not None:
            self.on_wait()
        return self.returncode

    def send_signal(self, identity, sig):
        assert dict(identity) == self.child
        self.events.append("signal")
        self.signals.append(sig)


def runtime_config(tmp_path):
    checkpoint = tmp_path / "trainer" / "checkpoint.ckpt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    promotion = tmp_path / "promotion"
    return RuntimeConfig(
        mutation_enabled=True,
        run_root=tmp_path,
        promotion_root=promotion,
        lease_state_path=promotion / "gpu-lease.json",
        event_log_path=promotion / "gpu-events.jsonl",
        expected_gpu_uuid="GPU-test",
        gpu_index=7,
        gpu_inventory_command=("nvidia-smi", "inventory"),
        gpu_process_query_command=("nvidia-smi", "processes"),
        clean_observations=2,
        clean_observation_interval_seconds=0.1,
        poll_interval_seconds=0.1,
        trainer_launch_command=("trainer", "{checkpoint_path}"),
        trainer_graceful_command=("drain", "{pid}"),
        trainer_checkpoint_path=checkpoint,
        trainer_drain_timeout_seconds=1.0,
        trainer_checkpoint_timeout_seconds=1.0,
        trainer_checkpoint_stable_seconds=0.0,
        require_checkpoint_change=False,
        trainer_start_timeout_seconds=1.0,
        evaluator_launch_command=("unused-evaluator",),
        evaluator_drain_command=(),
        evaluator_process_count=1,
        evaluator_drain_timeout_seconds=1.0,
        owner_id="guardian-test",
    )


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def setup_worker(
    tmp_path,
    *,
    returncode=0,
    restore=True,
    wrapper_pid=500,
):
    config = runtime_config(tmp_path)
    runtime_path = tmp_path / "gpu-runtime.json"
    runtime_path.write_bytes(b"runtime-config-for-test\n")
    runtime_hash = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    trainer = lease_identity(700, role="trainer")
    binding = {
        "schema_version": 1,
        "role": "trainer",
        "launch_status": "running",
        "process_identity": identity_dict(700, role="trainer"),
    }
    write_canonical(
        config.promotion_root / "supervisor" / "trainer.json",
        binding,
    )
    events = []
    manager = FakeManager(config, events, restore=restore)
    manager.runner.add(trainer)
    runner = FakeGuardianRunner(
        manager,
        events,
        returncode=returncode,
        wrapper_pid=wrapper_pid,
    )
    receipt = tmp_path / "receipts" / "claim.json"
    worker = GpuLeaseWorker(
        config,
        runtime_config_path=runtime_path,
        runtime_config_sha256=runtime_hash,
        claim_id="claim-a",
        work_id="work-a",
        receipt_path=receipt,
        argv=("trial", "--id", "work-a"),
        manager=manager,
        process_runner=runner,
    )
    return worker, manager, runner, events, receipt


def load_receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_guardian_owns_handoff_for_entire_child_lifetime(tmp_path):
    worker, manager, runner, events, receipt = setup_worker(tmp_path)

    assert worker.run() == 0

    assert events == ["handoff-enter", "spawn", "wait", "handoff-exit"]
    assert runner.spawns == [["trial", "--id", "work-a"]]
    paths = guardian_receipt_paths(receipt)
    ready = load_receipt(paths.ready)
    lifetime = load_receipt(paths.lifetime)
    completion = load_receipt(paths.completion)
    assert ready["contract"] == READY_RECEIPT_CONTRACT
    assert lifetime["contract"] == LIFETIME_RECEIPT_CONTRACT
    assert completion["contract"] == COMPLETION_RECEIPT_CONTRACT
    assert lifetime["ready_receipt_sha256"] == ready["receipt_sha256"]
    assert completion["lifetime_receipt_sha256"] == lifetime["receipt_sha256"]
    assert completion["lease_id"] == "lease-test"
    assert completion["expected_gpu_uuid"] == "GPU-test"
    assert completion["argv_sha256"] == canonical_sha256(["trial", "--id", "work-a"])
    assert completion["wrapper_process_identity"] == runner.wrapper
    assert completion["trainer_restored"] is True
    assert manager.read_record().phase == "trainer_running"


def test_child_failure_still_restores_trainer(tmp_path):
    worker, manager, _runner, events, receipt = setup_worker(tmp_path, returncode=9)

    assert worker.run() == 9

    assert events[-1] == "handoff-exit"
    assert manager.read_record().phase == "trainer_running"
    completion = load_receipt(receipt)
    assert completion["status"] == "child-failed"
    assert completion["returncode"] == 9
    assert completion["trainer_restored"] is True


def test_sigint_drains_child_then_restores_trainer(tmp_path):
    worker, manager, runner, events, receipt = setup_worker(tmp_path, returncode=-2)
    runner.on_wait = lambda: worker._handle_sigint(2, None)

    assert worker.run() == 130

    assert runner.signals == [2]
    assert events.index("signal") < events.index("handoff-exit")
    assert manager.read_record().phase == "trainer_running"
    completion = load_receipt(receipt)
    assert completion["status"] == "interrupted"
    assert completion["returncode"] == 130
    assert completion["trainer_restored"] is True


def test_restoration_failure_is_a_fail_closed_completion(tmp_path):
    worker, manager, _runner, _events, receipt = setup_worker(tmp_path, restore=False)

    with pytest.raises(GpuLeaseWorkerError, match="did not prove trainer restoration"):
        worker.run()

    assert manager.read_record().safety_halt is True
    completion = load_receipt(receipt)
    assert completion["status"] == "cleanup-failed"
    assert completion["trainer_restored"] is False
    assert completion["returncode"] is None


def test_completed_receipt_replays_without_handoff_or_spawn(tmp_path):
    worker, _manager, _runner, _events, receipt = setup_worker(tmp_path, returncode=6)
    assert worker.run() == 6

    replay, manager, runner, events, _ = setup_worker(
        tmp_path,
        returncode=99,
        wrapper_pid=800,
    )

    assert replay.run() == 6
    assert manager.record is None
    assert runner.spawns == []
    assert events == []
    assert load_receipt(receipt)["status"] == "child-failed"


def test_incomplete_lifetime_reconciles_and_refuses_duplicate_spawn(tmp_path):
    worker, _manager, _runner, _events, receipt = setup_worker(tmp_path)
    assert worker.run() == 0
    receipt.unlink()

    replay, manager, runner, events, _ = setup_worker(tmp_path, wrapper_pid=900)
    with pytest.raises(GpuLeaseWorkerError, match="duplicate spawn refused") as raised:
        replay.run()

    assert raised.value.code == "incomplete_guardian_receipt"
    assert manager.reconcile_calls == 1
    assert runner.spawns == []
    assert events == ["reconcile"]


def test_worker_spec_is_canonical_hash_bound_and_command_json_is_argv(tmp_path):
    config = runtime_config(tmp_path)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_bytes(b"runtime\n")
    runtime_hash = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    receipts = tmp_path / "receipts"
    value = {
        "schema_version": 1,
        "contract": WORKER_SPEC_CONTRACT,
        "runtime_config": str(runtime_path),
        "runtime_config_sha256": runtime_hash,
        "receipt_directory": str(receipts),
    }
    value["spec_sha256"] = canonical_sha256(value)
    spec_path = tmp_path / "worker-spec.json"
    write_canonical(spec_path, value)

    loaded = load_worker_spec(spec_path, expected_spec_sha256=value["spec_sha256"])
    args = parse_args(
        [
            "--spec",
            str(spec_path),
            "--expected-spec-sha256",
            value["spec_sha256"],
            "--claim-id",
            "claim-a",
            "--work-id",
            "work-a",
            "--receipt",
            str(receipts / "claim.json"),
            "--command-json",
            json.dumps(["python", "-c", "print('safe')"]),
        ]
    )

    assert loaded.runtime_config == runtime_path
    assert args.child_argv == ("python", "-c", "print('safe')")
    assert not hasattr(args, "shell")
    assert config.gpu_index == 7
