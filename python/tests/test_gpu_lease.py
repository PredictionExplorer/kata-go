import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from risk_score.gpu_lease import (
    CommandResult,
    CheckpointIdentity,
    GpuLeaseError,
    GpuLeaseManager,
    GpuObservation,
    GpuProcess,
    LeaseRecord,
    ProcessIdentity,
    RuntimeConfig,
)


class FakeTime:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeRunner:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.processes = {}
        self.next_pid = 200
        self.commands = []
        self.signals = []
        self.fail_evaluator_launch = False
        self.drain_trainer = True
        self.record_at_graceful_command = None
        self.probe_count = lambda: 0
        self.evaluator_launch_probe_counts = []

    @staticmethod
    def identity(pid, start_time=None, role="process"):
        if start_time is None:
            start_time = pid * 10
        return ProcessIdentity(
            pid=pid,
            start_time_ticks=start_time,
            process_group_id=pid,
            boot_id="boot-a",
            command_sha256=f"sha-{role}",
            cgroup=f"/test/{role}",
        )

    def add(self, identity):
        self.processes[identity.pid] = identity
        return identity

    def run(self, argv, *, timeout=None):
        argv = list(argv)
        self.commands.append(argv)
        if argv[0] == "graceful-trainer":
            lease_state = (
                self.checkpoint_path.parents[1] / "promotion" / "gpu-lease.json"
            )
            if lease_state.exists():
                self.record_at_graceful_command = json.loads(
                    lease_state.read_text(encoding="utf-8")
                )
            if self.drain_trainer:
                pid = int(argv[1])
                self.processes.pop(pid, None)
                self.checkpoint_path.write_bytes(b"checkpoint-after-drain")
            return CommandResult(0)
        return CommandResult(0)

    def spawn(self, argv, *, new_process_group=True):
        argv = list(argv)
        self.commands.append(argv)
        role = argv[0]
        if role == "evaluator" and self.fail_evaluator_launch:
            raise RuntimeError("injected evaluator launch failure")
        if role == "evaluator":
            self.evaluator_launch_probe_counts.append(self.probe_count())
        pid = self.next_pid
        self.next_pid += 1
        return self.add(self.identity(pid, role=role))

    def current_identity(self, pid):
        return self.processes.get(pid)

    def is_running(self, identity):
        current = self.current_identity(identity.pid)
        return current is not None and identity.same_process_as(current)

    def process_group_alive(self, identity):
        return any(
            process.process_group_id == identity.process_group_id
            for process in self.processes.values()
        )

    def signal_process_group(self, identity, sig):
        current = self.current_identity(identity.pid)
        if current is None or not identity.same_process_as(current):
            raise AssertionError("attempted to signal a reused process identity")
        self.signals.append((identity, sig))
        group_id = identity.process_group_id
        for pid in [
            pid
            for pid, process in self.processes.items()
            if process.process_group_id == group_id
        ]:
            self.processes.pop(pid)


class SequenceProbe:
    def __init__(self, observations):
        self.observations = list(observations)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if not self.observations:
            raise AssertionError("GPU probe called more often than expected")
        if len(self.observations) == 1:
            return self.observations[0]
        return self.observations.pop(0)


def runtime_config(
    tmp_path,
    *,
    mutation_enabled=True,
    clean_observations=2,
    require_checkpoint_change=False,
):
    checkpoint_path = tmp_path / "trainer" / "model.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint-before-drain")
    promotion_root = tmp_path / "promotion"
    return RuntimeConfig(
        mutation_enabled=mutation_enabled,
        run_root=tmp_path,
        promotion_root=promotion_root,
        lease_state_path=promotion_root / "gpu-lease.json",
        event_log_path=promotion_root / "gpu-events.jsonl",
        expected_gpu_uuid="GPU-test",
        gpu_index=7,
        gpu_inventory_command=("nvidia-smi", "inventory"),
        gpu_process_query_command=("nvidia-smi", "processes"),
        clean_observations=clean_observations,
        clean_observation_interval_seconds=0.25,
        poll_interval_seconds=0.1,
        trainer_launch_command=("trainer", "{checkpoint_path}"),
        trainer_graceful_command=("graceful-trainer", "{pid}"),
        trainer_checkpoint_path=checkpoint_path,
        trainer_drain_timeout_seconds=0.3,
        trainer_checkpoint_timeout_seconds=0.3,
        trainer_checkpoint_stable_seconds=0.0,
        require_checkpoint_change=require_checkpoint_change,
        trainer_start_timeout_seconds=0.3,
        evaluator_launch_command=("evaluator", "{worker_index}", "{gpu_uuid}"),
        evaluator_drain_command=(),
        evaluator_process_count=1,
        evaluator_drain_timeout_seconds=0.3,
        owner_id="test-controller",
    )


def checkpoint_identity(path):
    stat_result = path.stat()
    return CheckpointIdentity(
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def manager_for(config, runner, probe, fake_time):
    runner.probe_count = lambda: probe.calls
    return GpuLeaseManager(
        config,
        process_runner=runner,
        gpu_probe=probe,
        clock=fake_time,
        sleep=fake_time.sleep,
        event_sink=lambda event: None,
        lease_id_factory=lambda: "lease-test",
    )


def test_foreign_gpu_process_prevents_evaluator_launch_and_restores_trainer(
    tmp_path,
):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    trainer = runner.add(runner.identity(100, role="trainer"))
    foreign = GpuObservation("GPU-test", (GpuProcess(999, "foreign-python", 4096),))
    clean = GpuObservation("GPU-test")
    probe = SequenceProbe([foreign, clean, clean])
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    with pytest.raises(GpuLeaseError, match="not clean") as raised:
        with manager.evaluator_lease(trainer):
            pytest.fail("foreign process must prevent lease entry")

    assert raised.value.code == "foreign_gpu_process"
    assert not any(command[0] == "evaluator" for command in runner.commands)
    assert any(command[0] == "trainer" for command in runner.commands)
    assert manager.read_record().phase == "trainer_running"


def test_requires_repeated_clean_observations_before_launch(tmp_path):
    config = runtime_config(tmp_path, clean_observations=3)
    runner = FakeRunner(config.trainer_checkpoint_path)
    trainer = runner.add(runner.identity(100, role="trainer"))
    clean = GpuObservation("GPU-test")
    probe = SequenceProbe([clean] * 6)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    with manager.evaluator_lease(trainer) as lease:
        assert lease.phase == "evaluating"

    assert runner.evaluator_launch_probe_counts == [3]
    assert probe.calls == 6
    assert fake_time.value >= 1.0


def test_trainer_drain_failure_does_not_launch_duplicate_or_evaluator(tmp_path):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    runner.drain_trainer = False
    trainer = runner.add(runner.identity(100, role="trainer"))
    probe = SequenceProbe([GpuObservation("GPU-test")])
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    with pytest.raises(GpuLeaseError) as raised:
        with manager.evaluator_lease(trainer):
            pass

    assert raised.value.code == "trainer_drain_timeout"
    assert [command[0] for command in runner.commands] == ["graceful-trainer"]
    assert runner.is_running(trainer)
    assert probe.calls == 0
    assert manager.read_record().phase == "trainer_running"


def test_evaluator_launch_failure_restores_trainer(tmp_path):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    runner.fail_evaluator_launch = True
    trainer = runner.add(runner.identity(100, role="trainer"))
    probe = SequenceProbe([GpuObservation("GPU-test")] * 4)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    with pytest.raises(RuntimeError, match="injected evaluator"):
        with manager.evaluator_lease(trainer):
            pass

    restored = manager.read_record()
    assert restored.phase == "trainer_running"
    assert restored.trainer.pid != trainer.pid
    assert runner.is_running(restored.trainer)


def test_reconcile_never_signals_reused_evaluator_pid(tmp_path):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    stale_evaluator = runner.identity(150, start_time=1500, role="evaluator")
    reused_process = runner.identity(150, start_time=9999, role="unrelated")
    runner.add(reused_process)
    handoff = checkpoint_identity(config.trainer_checkpoint_path)
    now = 1.0
    record = LeaseRecord(
        lease_id="lease-crashed",
        owner_id="dead-controller",
        phase="evaluating",
        expected_gpu_uuid="GPU-test",
        trainer=None,
        evaluators=(stale_evaluator,),
        checkpoint_sha256=None,
        checkpoint_size=None,
        safety_halt=False,
        safety_reason=None,
        created_at=now,
        updated_at=now,
        pre_drain_checkpoint=handoff,
        handoff_checkpoint=handoff,
        restoration_status="pending",
    )
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")] * 2)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.current_phase == "trainer_running"
    assert runner.signals == []
    assert runner.current_identity(150) == reused_process


def test_mutation_disabled_blocks_external_changes(tmp_path):
    config = runtime_config(tmp_path, mutation_enabled=False)
    runner = FakeRunner(config.trainer_checkpoint_path)
    trainer = runner.add(runner.identity(100, role="trainer"))
    probe = SequenceProbe([GpuObservation("GPU-test")])
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    with pytest.raises(GpuLeaseError) as raised:
        with manager.evaluator_lease(trainer):
            pass

    assert raised.value.code == "mutation_disabled"
    assert runner.commands == []
    assert probe.calls == 0
    assert not config.lease_state_path.exists()
    assert not Path(str(config.lease_state_path) + ".lock").exists()


def test_crash_reconcile_drains_evaluator_and_restores_trainer(tmp_path):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    evaluator = runner.add(runner.identity(150, role="evaluator"))
    handoff = checkpoint_identity(config.trainer_checkpoint_path)
    record = LeaseRecord(
        lease_id="lease-crashed",
        owner_id="dead-controller",
        phase="evaluating",
        expected_gpu_uuid="GPU-test",
        trainer=None,
        evaluators=(evaluator,),
        checkpoint_sha256=handoff.sha256,
        checkpoint_size=handoff.size,
        safety_halt=False,
        safety_reason=None,
        created_at=1.0,
        updated_at=2.0,
        pre_drain_checkpoint=handoff,
        handoff_checkpoint=handoff,
        restoration_status="pending",
    )
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")] * 2)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.current_phase == "trainer_running"
    assert report.actions == ("drained_evaluators", "restored_trainer")
    assert not runner.is_running(evaluator)
    restored = manager.read_record()
    assert runner.is_running(restored.trainer)
    assert restored.evaluators == ()


def test_controller_handoff_persists_proofs_without_spawning_evaluator(
    tmp_path,
):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    trainer = runner.add(runner.identity(100, role="trainer"))
    probe = SequenceProbe([GpuObservation("GPU-test")] * 4)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    with manager.exclusive_handoff(trainer) as lease:
        assert lease.phase == "leased"
        assert lease.pre_drain_checkpoint is not None
        assert lease.handoff_checkpoint is not None
        assert lease.checkpoint_sha256 == lease.handoff_checkpoint.sha256
        assert len(lease.lease_clean_observation_times) == 2
        assert lease.lease_clean_observation_count == 2
        assert not any(command[0] == "evaluator" for command in runner.commands)
        persisted_before_signal = runner.record_at_graceful_command
        assert persisted_before_signal["phase"] == "draining_trainer"
        assert persisted_before_signal["preDrainCheckpoint"]["sha256"]
        assert persisted_before_signal["handoffCheckpoint"] is None

    restored = manager.read_record()
    assert restored.phase == "trainer_running"
    assert restored.restoration_status == "restored"
    assert restored.restored_trainer == restored.trainer
    assert len(restored.release_clean_observation_times) == 2
    assert restored.release_clean_observation_count == 2
    assert [command[0] for command in runner.commands] == [
        "graceful-trainer",
        "trainer",
    ]


def test_reconcile_validates_checkpoint_after_crash_immediately_after_exit(
    tmp_path,
):
    config = runtime_config(tmp_path, require_checkpoint_change=True)
    runner = FakeRunner(config.trainer_checkpoint_path)
    original_trainer = runner.identity(100, role="trainer")
    before = checkpoint_identity(config.trainer_checkpoint_path)
    record = LeaseRecord(
        lease_id="lease-crash-after-exit",
        owner_id="dead-controller",
        phase="draining_trainer",
        expected_gpu_uuid="GPU-test",
        trainer=original_trainer,
        evaluators=(),
        checkpoint_sha256=None,
        checkpoint_size=None,
        safety_halt=False,
        safety_reason=None,
        created_at=1.0,
        updated_at=2.0,
        pre_drain_checkpoint=before,
        restoration_status="pending",
    )
    config.trainer_checkpoint_path.write_bytes(b"new-complete-checkpoint")
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")] * 2)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.current_phase == "trainer_running"
    assert report.actions == (
        "validated_trainer_checkpoint_handoff",
        "restored_trainer",
    )
    restored = manager.read_record()
    assert restored.handoff_checkpoint.sha256 != before.sha256
    assert restored.restoration_status == "restored"


@pytest.mark.parametrize("checkpoint_state", ["unchanged", "missing"])
def test_reconcile_halts_if_crash_checkpoint_is_not_valid(tmp_path, checkpoint_state):
    config = runtime_config(tmp_path, require_checkpoint_change=True)
    runner = FakeRunner(config.trainer_checkpoint_path)
    original_trainer = runner.identity(100, role="trainer")
    before = checkpoint_identity(config.trainer_checkpoint_path)
    record = LeaseRecord(
        lease_id=f"lease-{checkpoint_state}",
        owner_id="dead-controller",
        phase="draining_trainer",
        expected_gpu_uuid="GPU-test",
        trainer=original_trainer,
        evaluators=(),
        checkpoint_sha256=None,
        checkpoint_size=None,
        safety_halt=False,
        safety_reason=None,
        created_at=1.0,
        updated_at=2.0,
        pre_drain_checkpoint=before,
        restoration_status="pending",
    )
    if checkpoint_state == "missing":
        config.trainer_checkpoint_path.unlink()
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")])
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.current_phase == "safety_halt"
    assert report.actions == ("safety_halt_checkpoint_handoff",)
    assert probe.calls == 0
    assert not any(command[0] == "trainer" for command in runner.commands)
    assert manager.read_record().restoration_status == "safety_halt"


def test_reconcile_does_not_signal_reused_trainer_pid(tmp_path):
    config = runtime_config(tmp_path, require_checkpoint_change=True)
    runner = FakeRunner(config.trainer_checkpoint_path)
    stale_trainer = runner.identity(100, start_time=1000, role="trainer")
    unrelated = runner.identity(100, start_time=9999, role="unrelated")
    runner.add(unrelated)
    before = checkpoint_identity(config.trainer_checkpoint_path)
    config.trainer_checkpoint_path.write_bytes(b"new-complete-checkpoint")
    record = LeaseRecord(
        lease_id="lease-reused-trainer",
        owner_id="dead-controller",
        phase="draining_trainer",
        expected_gpu_uuid="GPU-test",
        trainer=stale_trainer,
        evaluators=(),
        checkpoint_sha256=None,
        checkpoint_size=None,
        safety_halt=False,
        safety_reason=None,
        created_at=1.0,
        updated_at=2.0,
        pre_drain_checkpoint=before,
        restoration_status="pending",
    )
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")] * 2)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.current_phase == "trainer_running"
    assert runner.signals == []
    assert runner.current_identity(100) == unrelated


@pytest.mark.parametrize(
    "phase",
    [
        "trainer_drained",
        "leased",
        "evaluator_starting",
        "evaluating",
        "draining_evaluators",
        "evaluator_drained",
        "releasing",
        "release_gpu_verified",
        "restoring_trainer",
    ],
)
def test_reconcile_restores_every_post_drain_handoff_phase(tmp_path, phase):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    original_trainer = runner.identity(100, role="trainer")
    handoff = checkpoint_identity(config.trainer_checkpoint_path)
    record = LeaseRecord(
        lease_id=f"lease-{phase}",
        owner_id="dead-controller",
        phase=phase,
        expected_gpu_uuid="GPU-test",
        trainer=original_trainer,
        evaluators=(),
        checkpoint_sha256=handoff.sha256,
        checkpoint_size=handoff.size,
        safety_halt=False,
        safety_reason=None,
        created_at=1.0,
        updated_at=2.0,
        pre_drain_checkpoint=handoff,
        handoff_checkpoint=handoff,
        restoration_status="pending",
    )
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")] * 2)
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.current_phase == "trainer_running"
    assert report.actions == ("restored_trainer",)
    assert manager.read_record().restoration_status == "restored"


def test_reconcile_completes_recorded_live_restoration(tmp_path):
    config = runtime_config(tmp_path)
    runner = FakeRunner(config.trainer_checkpoint_path)
    original_trainer = runner.identity(100, role="trainer")
    restored_trainer = runner.add(runner.identity(200, role="trainer"))
    handoff = checkpoint_identity(config.trainer_checkpoint_path)
    record = LeaseRecord(
        lease_id="lease-restoring-live",
        owner_id="dead-controller",
        phase="restoring_trainer",
        expected_gpu_uuid="GPU-test",
        trainer=original_trainer,
        evaluators=(),
        checkpoint_sha256=handoff.sha256,
        checkpoint_size=handoff.size,
        safety_halt=False,
        safety_reason=None,
        created_at=1.0,
        updated_at=2.0,
        pre_drain_checkpoint=handoff,
        handoff_checkpoint=handoff,
        restoration_status="started",
        restored_trainer=restored_trainer,
    )
    config.lease_state_path.parent.mkdir(parents=True)
    config.lease_state_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    probe = SequenceProbe([GpuObservation("GPU-test")])
    fake_time = FakeTime()
    manager = manager_for(config, runner, probe, fake_time)

    report = manager.reconcile()

    assert report.actions == ("completed_trainer_restoration",)
    assert report.current_phase == "trainer_running"
    assert probe.calls == 0
    assert manager.read_record().trainer == restored_trainer


def test_runtime_rejects_relative_and_out_of_scope_paths(tmp_path):
    config = runtime_config(tmp_path)
    with pytest.raises(GpuLeaseError, match="absolute"):
        dataclasses.replace(config, run_root=Path("relative-run"))
    with pytest.raises(GpuLeaseError, match="promotionRoot"):
        dataclasses.replace(
            config,
            promotion_root=tmp_path.parent / "outside-promotion",
        )
    with pytest.raises(GpuLeaseError, match="outside promotionRoot"):
        dataclasses.replace(
            config,
            trainer_checkpoint_path=config.promotion_root / "checkpoint.ckpt",
        )


def test_runtime_rejects_symlinked_ancestors_and_lock_alias(tmp_path):
    config = runtime_config(tmp_path)
    config.promotion_root.mkdir()
    symlink_target = tmp_path / "symlink-target"
    symlink_target.mkdir()
    symlink = config.promotion_root / "linked"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(GpuLeaseError, match="symlinked"):
        dataclasses.replace(config, lease_state_path=symlink / "lease.json")

    config.lease_state_path.write_text("{}", encoding="utf-8")
    os.link(config.lease_state_path, config.lock_path)
    with pytest.raises(GpuLeaseError, match="hard links"):
        dataclasses.replace(config)


def test_checked_in_runtime_example_is_safe_and_parseable():
    example = (
        Path(__file__).resolve().parents[1]
        / "risk_score"
        / "gpu_lease_runtime.example.json"
    )
    config = RuntimeConfig.from_json_file(example)
    assert config.mutation_enabled is False
    assert config.gpu_index == 7
    assert config.clean_observations >= 2
    assert config.run_root.is_absolute()
    assert config.promotion_root.is_absolute()
    assert config.lease_state_path != config.lock_path
