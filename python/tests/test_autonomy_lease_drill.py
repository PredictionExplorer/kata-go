import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score import autonomy_bootstrap
from risk_score.autonomy_lease_drill import (
    GATE_ID,
    LEASE_CHECK_FIELDS,
    PROBE_RECEIPT_CONTRACT,
    SPEC_CONTRACT,
    AutonomyLeaseDrill,
    LeaseDrillError,
    load_drill_spec,
    parse_args,
)
from risk_score.gpu_lease import (
    SCHEMA_VERSION as GPU_LEASE_SCHEMA_VERSION,
)
from risk_score.gpu_lease import (
    CommandResult,
    GpuObservation,
    GpuProcess,
    ProcessIdentity,
)

GPU_UUID = "GPU-test-uuid"
PROBE_MODEL_HASH = hashlib.sha256(b"probe-model").hexdigest()
PROBE_CONFIG_HASH = hashlib.sha256(b"probe-config").hexdigest()


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def identity(pid, role):
    return ProcessIdentity(
        pid=pid,
        start_time_ticks=pid * 10,
        process_group_id=pid,
        boot_id="boot-test",
        command_sha256=hashlib.sha256(role.encode("utf-8")).hexdigest(),
        cgroup=f"/test/{role}",
    )


def snake_identity(value):
    return {
        "pid": value.pid,
        "start_time_ticks": value.start_time_ticks,
        "process_group_id": value.process_group_id,
        "boot_id": value.boot_id,
        "command_sha256": value.command_sha256,
        "cgroup": value.cgroup,
    }


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeRunner:
    def __init__(
        self,
        checkpoint_path,
        probe_argv,
        *,
        probe_returncode=0,
        fail_restore=False,
        checkpoint_mismatch=False,
    ):
        self.checkpoint_path = checkpoint_path
        self.probe_argv = tuple(probe_argv)
        self.probe_returncode = probe_returncode
        self.fail_restore = fail_restore
        self.checkpoint_mismatch = checkpoint_mismatch
        self.processes = {}
        self.roles = {}
        self.commands = []
        self.next_pid = 200

    def add(self, process, role):
        self.processes[process.pid] = process
        self.roles[process.pid] = role
        return process

    def run(self, argv, *, timeout=None):
        command = tuple(argv)
        self.commands.append((command, timeout))
        if command[0] == "graceful-trainer":
            trainer_pid = int(command[1])
            self.processes.pop(trainer_pid, None)
            self.checkpoint_path.write_bytes(b"checkpoint-after-drain")
            return CommandResult(0)
        if command == self.probe_argv:
            if self.checkpoint_mismatch:
                self.checkpoint_path.write_bytes(b"tampered-after-handoff")
            evaluator_pids = sorted(
                pid
                for pid, role in self.roles.items()
                if role == "evaluator" and pid in self.processes
            )
            receipt = {
                "schema_version": 1,
                "contract": PROBE_RECEIPT_CONTRACT,
                "gpu_uuid": GPU_UUID,
                "evaluator_pids": evaluator_pids,
                "model_sha256": PROBE_MODEL_HASH,
                "config_sha256": PROBE_CONFIG_HASH,
                "completed_work_count": 1,
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            return CommandResult(
                self.probe_returncode,
                stdout=(
                    canonical_json(receipt) + "\n"
                    if self.probe_returncode == 0
                    else ""
                ),
                stderr="" if self.probe_returncode == 0 else "probe failed",
            )
        return CommandResult(0)

    def spawn(self, argv, *, new_process_group=True):
        command = tuple(argv)
        self.commands.append((command, None))
        role = command[0]
        process = identity(self.next_pid, role)
        self.next_pid += 1
        if role == "trainer" and self.fail_restore:
            return process
        return self.add(process, role)

    def current_identity(self, pid):
        return self.processes.get(pid)

    def is_running(self, process):
        current = self.current_identity(process.pid)
        return current is not None and process.same_process_as(current)

    def process_group_alive(self, process):
        return any(
            current.process_group_id == process.process_group_id
            for current in self.processes.values()
        )

    def signal_process_group(self, process, _signal):
        group = process.process_group_id
        for pid in [
            pid
            for pid, current in self.processes.items()
            if current.process_group_id == group
        ]:
            self.processes.pop(pid)


class FakeGpuProbe:
    def __init__(self, runner, *, overlap=False):
        self.runner = runner
        self.overlap = overlap
        self.calls = 0

    def __call__(self):
        self.calls += 1
        evaluator_pids = sorted(
            pid
            for pid in self.runner.processes
            if self.runner.roles.get(pid) == "evaluator"
        )
        processes = [GpuProcess(pid, "evaluator", 128) for pid in evaluator_pids]
        if evaluator_pids and self.overlap:
            processes.append(GpuProcess(100, "trainer", 256))
        return GpuObservation(GPU_UUID, tuple(processes))


def runtime_value(root, *, clean_observations=3):
    production = root / "production"
    checkpoint = production / "trainer" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint-before-drain")
    promotion = production / "promotion"
    return (
        {
            "mutationEnabled": True,
            "paths": {
                "runRoot": str(production),
                "promotionRoot": str(promotion),
                "leaseState": str(promotion / "gpu-lease.json"),
                "eventLog": str(promotion / "gpu-events.jsonl"),
            },
            "gpu": {
                "expectedUuid": GPU_UUID,
                "index": 7,
                "inventoryCommand": ["nvidia-smi", "inventory"],
                "processQueryCommand": ["nvidia-smi", "processes"],
                "cleanObservations": clean_observations,
                "cleanObservationIntervalSeconds": 0.25,
            },
            "pollIntervalSeconds": 0.1,
            "trainer": {
                "launchCommand": ["trainer", "{checkpoint_path}"],
                "gracefulCommand": ["graceful-trainer", "{pid}"],
                "checkpointPath": str(checkpoint),
                "drainTimeoutSeconds": 0.3,
                "checkpointTimeoutSeconds": 0.3,
                "checkpointStableSeconds": 0.0,
                "requireCheckpointChange": True,
                "startTimeoutSeconds": 0.3,
            },
            "evaluator": {
                "launchCommand": ["evaluator", "{gpu_uuid}"],
                "drainCommand": [],
                "processCount": 1,
                "drainTimeoutSeconds": 0.3,
            },
            "ownerId": "lease-drill-test",
        },
        checkpoint,
    )


def make_fixture(
    root,
    *,
    clean_observations=3,
    minimum_clean_observations=3,
    source_kind="process-identity",
):
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    runtime, checkpoint = runtime_value(root, clean_observations=clean_observations)
    runtime_path = inputs / "gpu-lease.json"
    write_canonical(runtime_path, runtime)

    trainer = identity(100, "trainer")
    source_path = inputs / "trainer-source.json"
    if source_kind == "process-identity":
        source = {
            "schema_version": 1,
            "role": "trainer",
            "process_identity": snake_identity(trainer),
        }
        source_keys = {
            "trainer_process_identity_path": str(source_path),
            "trainer_process_identity_sha256": None,
        }
    else:
        source = {
            "schema_version": 1,
            "identities": {
                "selfplay": [],
                "shuffler": [],
                "trainer": [snake_identity(trainer)],
                "exporter": [],
                "evaluator": [],
            },
        }
        source_keys = {
            "supervisor_consumer_state_path": str(source_path),
            "supervisor_consumer_state_sha256": None,
        }
    write_canonical(source_path, source)
    for key in tuple(source_keys):
        if key.endswith("_sha256"):
            source_keys[key] = file_sha256(source_path)

    work_root = root / "drill-work"
    evidence = work_root / "lease-evidence.json"
    probe_argv = (str(Path(sys.executable).resolve()), "-c", "probe")
    spec_value = {
        "schema_version": 1,
        "contract": SPEC_CONTRACT,
        "gpu_lease_config_path": str(runtime_path),
        "gpu_lease_config_sha256": file_sha256(runtime_path),
        **source_keys,
        "expected_gpu_uuid": GPU_UUID,
        "minimum_clean_observations": minimum_clean_observations,
        "evaluator_probe_argv": list(probe_argv),
        "evaluator_probe_timeout_seconds": 5.0,
        "evaluator_probe_executable_sha256": file_sha256(
            Path(sys.executable).resolve()
        ),
        "evaluator_probe_model_sha256": PROBE_MODEL_HASH,
        "evaluator_probe_config_sha256": PROBE_CONFIG_HASH,
        "evaluator_probe_minimum_completed_work": 1,
        "work_root": str(work_root),
        "evidence_output": str(evidence),
    }
    spec_value["spec_sha256"] = canonical_sha256(spec_value)
    spec_path = inputs / "lease-drill.json"
    write_canonical(spec_path, spec_value)
    spec = load_drill_spec(spec_path, expected_spec_sha256=spec_value["spec_sha256"])
    return SimpleNamespace(
        root=root,
        runtime=runtime,
        runtime_path=runtime_path,
        checkpoint=checkpoint,
        trainer=trainer,
        source_path=source_path,
        work_root=work_root,
        evidence=evidence,
        probe_argv=probe_argv,
        spec_path=spec_path,
        spec_value=spec_value,
        spec=spec,
    )


def execute(fixture, **runner_options):
    clock = FakeClock()
    overlap = runner_options.pop("overlap", False)
    runner = FakeRunner(
        fixture.checkpoint,
        fixture.probe_argv,
        **runner_options,
    )
    runner.add(fixture.trainer, "trainer")
    probe = FakeGpuProbe(runner, overlap=overlap)
    drill = AutonomyLeaseDrill(
        fixture.spec,
        process_runner=runner,
        gpu_probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )
    return drill, runner, probe


def read_evidence(fixture):
    return json.loads(fixture.evidence.read_text(encoding="utf-8"))


def test_cli_accepts_bootstrap_evidence_binding(tmp_path):
    fixture = make_fixture(tmp_path)
    rebound = load_drill_spec(
        fixture.spec_path,
        expected_spec_sha256=file_sha256(fixture.spec_path),
    )

    args = parse_args(
        [
            "--spec",
            str(fixture.spec_path),
            "--evidence",
            str(fixture.evidence),
        ]
    )

    assert args.spec == fixture.spec_path
    assert args.evidence == fixture.evidence
    assert args.expected_spec_sha256 is None
    assert rebound.identity == fixture.spec.spec_sha256


@pytest.mark.parametrize(
    "source_kind", ["process-identity", "supervisor-consumer-state"]
)
def test_happy_path_runs_probe_and_publishes_validator_fields(tmp_path, source_kind):
    fixture = make_fixture(tmp_path, source_kind=source_kind)
    drill, runner, probe = execute(fixture)

    evidence = drill.run()

    assert evidence["decision"] == "PASS"
    assert set(evidence["checks"]) == LEASE_CHECK_FIELDS
    assert evidence["checks"] == {
        "gpu_uuid": GPU_UUID,
        "lease_schema_version": GPU_LEASE_SCHEMA_VERSION,
        "trainer_drained": True,
        "checkpoint_handoff_verified": True,
        "evaluator_exclusive": True,
        "process_overlap_observed": False,
        "trainer_restored": True,
        "lease_clean_observations": 3,
        "release_clean_observations": 3,
        "safety_halt": False,
    }
    assert (fixture.probe_argv, 5.0) in runner.commands
    assert probe.calls == 7
    gate = autonomy_bootstrap.GateSpec(
        gate_id=GATE_ID,
        argv=(),
        evidence=fixture.evidence,
        inputs=(),
        outputs=(fixture.evidence,),
        requirements={
            "expected_gpu_uuid": GPU_UUID,
            "minimum_clean_observations": 3,
        },
    )
    assert autonomy_bootstrap._validate_lease_drill(gate) == {"verified": True}


def test_launch_source_starts_trainer_before_handoff(tmp_path):
    fixture = make_fixture(tmp_path)
    flat = dict(fixture.spec_value)
    for key in (
        "gpu_lease_config_path",
        "gpu_lease_config_sha256",
        "trainer_process_identity_path",
        "trainer_process_identity_sha256",
        "evaluator_probe_argv",
        "evaluator_probe_timeout_seconds",
        "evaluator_probe_executable_sha256",
        "evaluator_probe_model_sha256",
        "evaluator_probe_config_sha256",
        "evaluator_probe_minimum_completed_work",
    ):
        flat.pop(key)
    flat["gpu_lease_config"] = {
        "path": str(fixture.runtime_path),
        "sha256": file_sha256(fixture.runtime_path),
    }
    flat["trainer_source"] = {"kind": "launch"}
    flat["evaluator_probe"] = {
        "argv": list(fixture.probe_argv),
        "timeout_seconds": 5.0,
        "executable_sha256": file_sha256(Path(sys.executable).resolve()),
        "model_sha256": PROBE_MODEL_HASH,
        "config_sha256": PROBE_CONFIG_HASH,
        "minimum_completed_work": 1,
    }
    flat.pop("spec_sha256")
    flat["spec_sha256"] = canonical_sha256(flat)
    write_canonical(fixture.spec_path, flat)
    launched_spec = load_drill_spec(fixture.spec_path)
    clock = FakeClock()
    runner = FakeRunner(fixture.checkpoint, fixture.probe_argv)
    drill = AutonomyLeaseDrill(
        launched_spec,
        process_runner=runner,
        gpu_probe=FakeGpuProbe(runner),
        clock=clock,
        sleep=clock.sleep,
    )

    evidence = drill.run()

    assert evidence["decision"] == "PASS"
    assert any(command[0][0] == "trainer" for command in runner.commands)


def test_gpu_process_overlap_writes_fail_without_pass_claim(tmp_path):
    fixture = make_fixture(tmp_path)
    drill, _runner, _probe = execute(fixture, overlap=True)

    with pytest.raises(LeaseDrillError) as raised:
        drill.run()

    assert raised.value.code == "evaluator_not_exclusive"
    evidence = read_evidence(fixture)
    assert evidence["decision"] == "FAIL"
    assert set(evidence["checks"]) == LEASE_CHECK_FIELDS
    assert evidence["checks"]["process_overlap_observed"] is True
    assert evidence["checks"]["evaluator_exclusive"] is False


def test_failed_trainer_restoration_sets_safety_halt_and_fails(tmp_path):
    fixture = make_fixture(tmp_path)
    drill, _runner, _probe = execute(fixture, fail_restore=True)

    with pytest.raises(LeaseDrillError) as raised:
        drill.run()

    assert raised.value.code == "gpu_lease_failed"
    assert raised.value.details["gpu_lease_code"] == "trainer_restart_failed"
    evidence = read_evidence(fixture)
    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["trainer_restored"] is False
    assert evidence["checks"]["safety_halt"] is True


def test_checkpoint_mismatch_refuses_restore_and_writes_fail(tmp_path):
    fixture = make_fixture(tmp_path)
    drill, _runner, _probe = execute(fixture, checkpoint_mismatch=True)

    with pytest.raises(LeaseDrillError) as raised:
        drill.run()

    assert raised.value.code == "gpu_lease_failed"
    assert raised.value.details["gpu_lease_code"] == "handoff_checkpoint_changed"
    evidence = read_evidence(fixture)
    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["checkpoint_handoff_verified"] is True
    assert evidence["checks"]["trainer_restored"] is False
    assert evidence["checks"]["safety_halt"] is True


def test_insufficient_observation_configuration_fails_before_drain(tmp_path):
    fixture = make_fixture(
        tmp_path,
        clean_observations=2,
        minimum_clean_observations=3,
    )
    drill, runner, probe = execute(fixture)

    with pytest.raises(LeaseDrillError) as raised:
        drill.run()

    assert raised.value.code == "insufficient_clean_observations"
    assert runner.commands == []
    assert probe.calls == 0
    evidence = read_evidence(fixture)
    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["lease_clean_observations"] == 0


def test_evaluator_probe_command_failure_restores_then_writes_fail(tmp_path):
    fixture = make_fixture(tmp_path)
    drill, runner, _probe = execute(fixture, probe_returncode=9)

    with pytest.raises(LeaseDrillError) as raised:
        drill.run()

    assert raised.value.code == "evaluator_probe_failed"
    evidence = read_evidence(fixture)
    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["trainer_restored"] is True
    assert evidence["checks"]["safety_halt"] is False
    assert any(command[0][0] == "trainer" for command in runner.commands)


def test_canonical_hash_and_path_safety_fail_closed(tmp_path):
    noncanonical = make_fixture(tmp_path / "noncanonical")
    noncanonical.spec_path.write_text(
        json.dumps(noncanonical.spec_value, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LeaseDrillError) as raised:
        load_drill_spec(noncanonical.spec_path)
    assert raised.value.code == "noncanonical_json"

    changed = make_fixture(tmp_path / "changed")
    changed.runtime_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(LeaseDrillError) as raised:
        load_drill_spec(changed.spec_path)
    assert raised.value.code == "bound_input_changed"

    escaped = make_fixture(tmp_path / "escaped")
    escaped_value = dict(escaped.spec_value)
    escaped_value["evidence_output"] = str(escaped.root / "outside.json")
    escaped_value.pop("spec_sha256")
    escaped_value["spec_sha256"] = canonical_sha256(escaped_value)
    write_canonical(escaped.spec_path, escaped_value)
    with pytest.raises(LeaseDrillError) as raised:
        load_drill_spec(escaped.spec_path)
    assert raised.value.code == "unsafe_path"

    confused = make_fixture(tmp_path / "confused")
    confused_value = dict(confused.spec_value)
    production = Path(confused.runtime["paths"]["runRoot"])
    confused_value["work_root"] = str(production)
    confused_value["evidence_output"] = str(production / "lease-evidence.json")
    confused_value.pop("spec_sha256")
    confused_value["spec_sha256"] = canonical_sha256(confused_value)
    write_canonical(confused.spec_path, confused_value)
    confused_spec = load_drill_spec(confused.spec_path)
    runner = FakeRunner(confused.checkpoint, confused.probe_argv)
    runner.add(confused.trainer, "trainer")
    clock = FakeClock()
    drill = AutonomyLeaseDrill(
        confused_spec,
        process_runner=runner,
        gpu_probe=FakeGpuProbe(runner),
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(LeaseDrillError) as raised:
        drill.run()
    assert raised.value.code == "unsafe_path"
    assert not (production / "lease-evidence.json").exists()
    assert runner.commands == []


def test_spec_rejects_aliasing_production_input_into_work_root(tmp_path):
    fixture = make_fixture(tmp_path)
    work_root = fixture.root / "unsafe-work"
    runtime_path = work_root / "gpu-lease.json"
    work_root.mkdir()
    runtime_path.write_bytes(fixture.runtime_path.read_bytes())
    value = dict(fixture.spec_value)
    value["gpu_lease_config_path"] = str(runtime_path)
    value["gpu_lease_config_sha256"] = file_sha256(runtime_path)
    value["work_root"] = str(work_root)
    value["evidence_output"] = str(work_root / "evidence.json")
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(fixture.spec_path, value)

    with pytest.raises(LeaseDrillError) as raised:
        load_drill_spec(fixture.spec_path)

    assert raised.value.code == "unsafe_path"
