import dataclasses
import json
from pathlib import Path

import pytest

from risk_score import gpu_lease_worker
from risk_score.cluster_executor import (
    EXECUTOR_SPEC_CONTRACT,
    WORK_SPEC_CONTRACT,
    ClusterExecutor,
    CommandResult,
    ExecutorSpecError,
    ProcessIdentity,
    load_executor_spec,
    parse_args,
)
from risk_score.cluster_scheduler import (
    ClusterScheduler,
    IdleReason,
    PreemptionStatus,
    WorkKind,
    WorkState,
    canonical_json_bytes,
    canonical_sha256,
)


class FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds

    def advance(self, seconds):
        self.value += seconds


class FakeRunner:
    def __init__(self):
        self.next_pid = 1000
        self.processes = {}
        self.returncodes = {}
        self.spawns = []
        self.commands = []
        self.on_run = None

    @staticmethod
    def make_identity(pid, *, start_time_ticks=None):
        return ProcessIdentity(
            pid=pid,
            start_time_ticks=(
                pid * 10 if start_time_ticks is None else start_time_ticks
            ),
            command_sha256=f"{pid:064x}"[-64:],
            process_group_id=pid,
            boot_id="boot-test",
            cgroup="0::/test",
        )

    def spawn(self, argv, *, cwd, environment, log_path):
        pid = self.next_pid
        self.next_pid += 1
        identity = self.make_identity(pid)
        self.processes[pid] = identity
        self.spawns.append(
            {
                "argv": list(argv),
                "cwd": Path(cwd),
                "environment": dict(environment),
                "log_path": Path(log_path),
                "identity": identity,
            }
        )
        return identity

    def run(self, argv, *, cwd, environment, timeout):
        call = {
            "argv": list(argv),
            "cwd": Path(cwd),
            "environment": dict(environment),
            "timeout": timeout,
        }
        self.commands.append(call)
        if self.on_run is not None:
            return self.on_run(call)
        return CommandResult(0)

    def current_identity(self, pid):
        return self.processes.get(pid)

    def returncode(self, identity):
        return self.returncodes.get(identity.pid)

    def process_group_alive(self, identity):
        return any(
            current.process_group_id == identity.process_group_id
            for current in self.processes.values()
        )

    def finish(self, identity, returncode):
        self.processes.pop(identity.pid, None)
        self.returncodes[identity.pid] = returncode

    def disappear(self, identity):
        self.processes.pop(identity.pid, None)
        self.returncodes.pop(identity.pid, None)


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def self_hashed(value):
    result = dict(value)
    result["spec_sha256"] = canonical_sha256(result)
    return result


def receipt_hashed(value):
    result = dict(value)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def executor_spec_value(
    tmp_path,
    gpu_ids=("0",),
    *,
    owner_id="executor-a",
    retry_budget=2,
    stale_after_seconds=30.0,
    gpu7_guardian_prefix=None,
):
    value = {
        "schema_version": 1,
        "contract": EXECUTOR_SPEC_CONTRACT,
        "scheduler_directory": str((tmp_path / "scheduler").resolve()),
        "state_directory": str((tmp_path / "executor-state").resolve()),
        "owner_id": owner_id,
        "gpu_ids": list(gpu_ids),
        "gpu7_id": "7" if "7" in gpu_ids else None,
        "poll_interval_seconds": 1.0,
        "heartbeat_interval_seconds": 2.0,
        "stale_after_seconds": stale_after_seconds,
        "retry_budget": retry_budget,
        "backoff_initial_seconds": 5.0,
        "backoff_max_seconds": 60.0,
        "lease_proof_command": None,
        "lease_proof_timeout_seconds": 10.0,
    }
    if gpu7_guardian_prefix is not None:
        value["gpu7_guardian_prefix"] = list(gpu7_guardian_prefix)
    return self_hashed(value)


def write_executor_spec(tmp_path, **kwargs):
    value = executor_spec_value(tmp_path, **kwargs)
    path = tmp_path / f"executor-{value['owner_id']}.json"
    write_canonical(path, value)
    return path


def work_payload(
    tmp_path,
    work_id,
    kind,
    eligible_gpus,
    *,
    argv=None,
    lease_role="none",
    safe_drain=None,
):
    value = {
        "schema_version": 1,
        "contract": WORK_SPEC_CONTRACT,
        "work_id": work_id,
        "kind": kind.value,
        "eligible_gpus": list(eligible_gpus),
        "argv": list(argv or ("worker", work_id, "{gpu_id}")),
        "cwd": str(tmp_path.resolve()),
        "environment": {"WORK_ID": work_id},
        "lease_role": lease_role,
        "safe_drain": safe_drain,
    }
    return {"executor_spec": self_hashed(value)}


def make_scheduler(tmp_path, clock, gpu_ids=("0",)):
    return ClusterScheduler(tmp_path / "scheduler", gpu_ids, clock=clock)


def make_executor(
    tmp_path,
    scheduler,
    runner,
    clock,
    *,
    gpu_ids=("0",),
    owner_id="executor-a",
    retry_budget=2,
    lease_proof=None,
    gpu7_guardian_prefix=None,
):
    spec = write_executor_spec(
        tmp_path,
        gpu_ids=gpu_ids,
        owner_id=owner_id,
        retry_budget=retry_budget,
        gpu7_guardian_prefix=gpu7_guardian_prefix,
    )
    return ClusterExecutor(
        spec,
        scheduler=scheduler,
        process_runner=runner,
        clock=clock,
        sleep=clock.sleep,
        lease_proof=lease_proof,
    )


def enqueue(
    scheduler,
    tmp_path,
    work_id,
    kind=WorkKind.BACKFILL,
    *,
    eligible_gpus=("0",),
    preemptible=False,
    lease_role="none",
    safe_drain=None,
    argv=None,
):
    return scheduler.enqueue(
        work_id,
        kind,
        eligible_gpus=eligible_gpus,
        preemptible=preemptible,
        payload=work_payload(
            tmp_path,
            work_id,
            kind,
            eligible_gpus,
            argv=argv,
            lease_role=lease_role,
            safe_drain=safe_drain,
        ),
    )


def test_starts_scheduler_selected_argv_with_fixed_gpu_binding(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    enqueue(scheduler, tmp_path, "low", WorkKind.BACKFILL)
    enqueue(scheduler, tmp_path, "recovery", WorkKind.RECOVERY)
    executor = make_executor(tmp_path, scheduler, runner, clock)

    result = executor.once()

    assert result["gpus"][0]["status"] == "started"
    assert scheduler.get_claim("0").work_id == "recovery"
    assert runner.spawns[0]["argv"] == ["worker", "recovery", "0"]
    assert runner.spawns[0]["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    identity = runner.spawns[0]["identity"].to_dict()
    assert set(identity) == {
        "pid",
        "start_time_ticks",
        "command_sha256",
        "process_group_id",
        "boot_id",
        "cgroup",
    }
    assert list((tmp_path / "executor-state" / "claim-receipts").glob("*.json"))
    assert list((tmp_path / "executor-state" / "start-receipts").glob("*.json"))


def test_replay_adopts_exact_process_without_duplicate_start(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    enqueue(scheduler, tmp_path, "job")
    spec_path = write_executor_spec(tmp_path)
    first = ClusterExecutor(
        spec_path,
        scheduler=scheduler,
        process_runner=runner,
        clock=clock,
    )
    first.once()

    restarted = ClusterExecutor(
        spec_path,
        scheduler=scheduler,
        process_runner=runner,
        clock=clock,
    )
    replay = restarted.once()

    assert replay["gpus"][0]["status"] == "running"
    assert len(runner.spawns) == 1
    assert scheduler.get_claim("0").claim_id == replay["gpus"][0]["claim_id"]


def test_completion_receipt_and_scheduler_release_are_replay_safe(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    enqueue(scheduler, tmp_path, "job")
    executor = make_executor(tmp_path, scheduler, runner, clock)
    executor.once()
    identity = runner.spawns[0]["identity"]
    runner.finish(identity, 0)

    completed = executor.once()
    revision = scheduler.reconstruct().revision
    replay = executor.once()

    assert completed["gpus"][0]["status"] == "completed"
    assert scheduler.get_work("job").state == WorkState.COMPLETED
    assert scheduler.get_claim("0") is None
    assert replay["gpus"][0]["status"] == "idle"
    assert scheduler.reconstruct().revision == revision + 1  # idle telemetry only
    receipts = list((tmp_path / "executor-state" / "completions").glob("*.json"))
    assert len(receipts) == 1


def test_failures_use_exponential_backoff_then_quarantine(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    enqueue(scheduler, tmp_path, "flaky")
    executor = make_executor(tmp_path, scheduler, runner, clock, retry_budget=2)
    executor.once()
    runner.finish(runner.spawns[-1]["identity"], 3)

    first = executor.once()["gpus"][0]
    assert first["status"] == "retry-backoff"
    assert first["failure_count"] == 1
    assert first["not_before_unix"] == 105.0
    assert executor.once()["gpus"][0]["status"] == "backoff"

    clock.advance(5)
    executor.once()
    runner.finish(runner.spawns[-1]["identity"], 4)
    second = executor.once()["gpus"][0]
    assert second["failure_count"] == 2
    assert second["not_before_unix"] == 115.0

    clock.advance(10)
    executor.once()
    runner.finish(runner.spawns[-1]["identity"], 5)
    third = executor.once()["gpus"][0]
    assert third["status"] == "quarantined"
    assert third["failure_count"] == 3
    assert scheduler.get_work("flaky").state == WorkState.FAILED
    assert len(runner.spawns) == 3
    marker = next((tmp_path / "executor-state" / "quarantine").glob("*.json"))
    assert json.loads(marker.read_text(encoding="utf-8"))["reason"] == (
        "process-returncode-5"
    )


def test_preemption_runs_configured_drain_and_requeues_at_checkpoint(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"before")
    safe_drain = {
        "command": ["drain", "{pid}", "{checkpoint_path}"],
        "checkpoint_path": str(checkpoint.resolve()),
        "require_checkpoint_change": True,
        "timeout_seconds": 20.0,
    }
    enqueue(
        scheduler,
        tmp_path,
        "backfill",
        WorkKind.BACKFILL,
        preemptible=True,
        safe_drain=safe_drain,
    )
    executor = make_executor(tmp_path, scheduler, runner, clock)
    executor.once()
    running_identity = runner.spawns[0]["identity"]
    enqueue(scheduler, tmp_path, "recovery", WorkKind.RECOVERY)
    request = scheduler.request_preemption(
        "0", for_work_id="recovery", requested_by="test"
    )

    def drain(call):
        assert call["argv"][0] == "drain"
        assert int(call["argv"][1]) == running_identity.pid
        checkpoint.write_bytes(b"after")
        return CommandResult(0)

    runner.on_run = drain
    draining = executor.once()["gpus"][0]
    assert draining["status"] == "draining-at-safe-boundary"
    runner.finish(running_identity, -2)
    result = executor.once()["gpus"][0]

    assert result["status"] == "preempted-at-safe-boundary"
    assert scheduler.get_work("backfill").state == WorkState.QUEUED
    assert scheduler.idle_status("0").reason == IdleReason.CHECKPOINT_BOUNDARY
    assert scheduler.idle_status("0").details["request_id"] == request.request_id
    assert scheduler.reconstruct().preemption_requests[0].status == (
        PreemptionStatus.BOUNDARY_REACHED
    )
    executor.once()
    assert scheduler.get_claim("0").work_id == "recovery"
    assert len(runner.commands) == 1


def test_identity_mismatch_halts_without_drain_or_signal(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"before")
    safe_drain = {
        "command": ["drain", "{pid}"],
        "checkpoint_path": str(checkpoint.resolve()),
        "require_checkpoint_change": False,
        "timeout_seconds": 20.0,
    }
    enqueue(
        scheduler,
        tmp_path,
        "job",
        preemptible=True,
        safe_drain=safe_drain,
    )
    executor = make_executor(tmp_path, scheduler, runner, clock)
    executor.once()
    expected = runner.spawns[0]["identity"]
    runner.processes[expected.pid] = dataclasses.replace(
        expected, start_time_ticks=expected.start_time_ticks + 1
    )
    enqueue(scheduler, tmp_path, "recovery", WorkKind.RECOVERY)
    scheduler.request_preemption("0", for_work_id="recovery")

    result = executor.once()["gpus"][0]

    assert result["status"] == "safety-halt"
    assert "identity mismatch" in result["reason"]
    assert runner.commands == []
    assert scheduler.get_claim("0").work_id == "job"
    assert scheduler.reconstruct().gpu_safety_halts["0"] == result["reason"]


def test_stale_owner_with_proven_dead_process_is_requeued(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock)
    runner = FakeRunner()
    enqueue(scheduler, tmp_path, "job")
    old = make_executor(tmp_path, scheduler, runner, clock, owner_id="executor-old")
    old.once()
    identity = runner.spawns[0]["identity"]
    runner.disappear(identity)
    clock.advance(31)

    new = make_executor(tmp_path, scheduler, runner, clock, owner_id="executor-new")
    result = new.once()["gpus"][0]

    assert result["status"] == "retry-backoff"
    assert scheduler.get_claim("0") is None
    assert scheduler.get_work("job").state == WorkState.QUEUED
    assert len(runner.spawns) == 1


def test_gpu7_trainer_never_starts_when_external_lease_denies(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    enqueue(
        scheduler,
        tmp_path,
        "trainer",
        WorkKind.TRAINER,
        eligible_gpus=("7",),
        lease_role="trainer",
    )
    requests = []

    def deny(request):
        requests.append(request)
        return {"allowed": False, "reason": "evaluator owns GPU7"}

    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        lease_proof=deny,
    )
    first = executor.once()["gpus"][0]
    second = executor.once()["gpus"][0]

    assert first["status"] == second["status"] == "lease-denied"
    assert runner.spawns == []
    assert len(requests) == 2
    assert requests[0].lease_role == "trainer"
    assert scheduler.get_claim("7").work_id == "trainer"
    assert scheduler.idle_status("7").reason == IdleReason.LEASE_HANDOFF


def guardian_prefix():
    return (
        "python",
        "-m",
        "risk_score.gpu_lease_worker",
        "--spec",
        "/opt/risk-score/gpu-worker.json",
        "--expected-spec-sha256",
        "a" * 64,
        "--claim-id",
        "{claim_id}",
        "--work-id",
        "{work_id}",
        "--receipt",
        "{guardian_receipt}",
        "--command-json",
    )


def renewable_proof(request, *, valid_until=200.0, lease_id="lease-a"):
    return {
        "allowed": True,
        "lease_id": lease_id,
        "claim_id": request.claim_id,
        "work_id": request.work_id,
        "valid_until_unix": valid_until,
    }


def test_gpu7_none_role_rejects_argv_without_exact_guardian_prefix(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    enqueue(
        scheduler,
        tmp_path,
        "unsafe",
        eligible_gpus=("7",),
        argv=("trial", "unsafe"),
    )
    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        gpu7_guardian_prefix=guardian_prefix(),
    )

    result = executor.once()["gpus"][0]

    assert result["status"] == "safety-halt"
    assert "guardian prefix" in result["reason"]
    assert runner.spawns == []
    assert scheduler.get_claim("7") is None


def test_gpu7_guardian_prefix_binds_claim_work_and_receipt(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    prefix = guardian_prefix()
    child_json = json.dumps(
        ["trial", "{work_id}"], sort_keys=True, separators=(",", ":")
    )
    enqueue(
        scheduler,
        tmp_path,
        "guarded",
        eligible_gpus=("7",),
        argv=prefix + (child_json,),
    )
    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        gpu7_guardian_prefix=prefix,
    )

    result = executor.once()["gpus"][0]

    assert result["status"] == "started"
    claim_id = result["claim_id"]
    argv = runner.spawns[0]["argv"]
    assert argv[:3] == ["python", "-m", "risk_score.gpu_lease_worker"]
    assert argv[argv.index("--claim-id") + 1] == claim_id
    assert argv[argv.index("--work-id") + 1] == "guarded"
    receipt = Path(argv[argv.index("--receipt") + 1])
    assert receipt.parent == tmp_path / "executor-state" / "guardian-receipts"
    assert json.loads(argv[-1]) == ["trial", "guarded"]
    intent = json.loads(
        next((tmp_path / "executor-state" / "start-intents").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert intent["guardian_receipt_path"] == str(receipt)
    assert intent["gpu7_guardian_prefix_sha256"]


def test_gpu7_guardian_completion_proves_lifetime_before_release(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    prefix = guardian_prefix()
    child_argv = ["trial", "guarded"]
    enqueue(
        scheduler,
        tmp_path,
        "guarded",
        eligible_gpus=("7",),
        argv=prefix + (json.dumps(child_argv, sort_keys=True, separators=(",", ":")),),
    )
    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        gpu7_guardian_prefix=prefix,
    )
    started = executor.once()["gpus"][0]
    wrapper = runner.spawns[0]["identity"]
    paths = gpu_lease_worker.guardian_receipt_paths(
        executor._guardian_receipt_path(started["claim_id"])
    )
    common = {
        "schema_version": 1,
        "claim_id": started["claim_id"],
        "work_id": "guarded",
        "receipt_path": str(paths.completion),
        "lease_id": "lease-guarded",
        "expected_gpu_uuid": "GPU-test",
        "argv": child_argv,
        "argv_sha256": canonical_sha256(child_argv),
        "wrapper_process_identity": wrapper.to_dict(),
        "wrapper_process_identity_sha256": canonical_sha256(wrapper.to_dict()),
    }
    ready = receipt_hashed(
        {
            **common,
            "contract": gpu_lease_worker.READY_RECEIPT_CONTRACT,
            "phase": "ready",
        }
    )
    child = dataclasses.replace(
        FakeRunner.make_identity(2000),
        process_group_id=wrapper.process_group_id,
    )
    lifetime = receipt_hashed(
        {
            **common,
            "contract": gpu_lease_worker.LIFETIME_RECEIPT_CONTRACT,
            "phase": "lifetime",
            "ready_receipt_sha256": ready["receipt_sha256"],
            "child_process_identity": child.to_dict(),
            "child_process_identity_sha256": canonical_sha256(child.to_dict()),
        }
    )
    completion = receipt_hashed(
        {
            **common,
            "contract": gpu_lease_worker.COMPLETION_RECEIPT_CONTRACT,
            "phase": "completion",
            "status": "completed",
            "ready_receipt_sha256": ready["receipt_sha256"],
            "lifetime_receipt_sha256": lifetime["receipt_sha256"],
            "child_process_identity_sha256": lifetime["child_process_identity_sha256"],
            "returncode": 0,
            "trainer_restored": True,
            "restoration": {"phase": "trainer_running"},
        }
    )
    write_canonical(paths.ready, ready)
    write_canonical(paths.lifetime, lifetime)
    write_canonical(paths.completion, completion)
    runner.finish(wrapper, 0)

    result = executor.once()["gpus"][0]

    assert result["status"] == "completed"
    assert scheduler.get_work("guarded").state == WorkState.COMPLETED


def test_gpu7_guardian_exit_without_receipt_halts_closed(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    prefix = guardian_prefix()
    enqueue(
        scheduler,
        tmp_path,
        "guarded",
        eligible_gpus=("7",),
        argv=prefix + ('["trial"]',),
    )
    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        gpu7_guardian_prefix=prefix,
    )
    assert executor.once()["gpus"][0]["status"] == "started"
    runner.finish(runner.spawns[0]["identity"], 0)

    result = executor.once()["gpus"][0]

    assert result["status"] == "safety-halt"
    assert "guardian completion is unsafe" in result["reason"]
    assert scheduler.get_work("guarded").state == WorkState.CLAIMED


def test_direct_gpu7_role_rejects_expired_renewable_proof(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    enqueue(
        scheduler,
        tmp_path,
        "trainer",
        WorkKind.TRAINER,
        eligible_gpus=("7",),
        lease_role="trainer",
    )
    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        lease_proof=lambda request: renewable_proof(request, valid_until=105.0),
    )
    assert executor.once()["gpus"][0]["status"] == "started"

    clock.advance(6)
    result = executor.once()["gpus"][0]

    assert result["status"] == "safety-halt"
    assert "expired" in result["reason"]
    assert len(runner.spawns) == 1


@pytest.mark.parametrize(
    "contradiction",
    [
        {"claim_id": "different-claim"},
        {"work_id": "different-work"},
        {"lease_id": "different-lease"},
    ],
)
def test_direct_gpu7_role_rejects_contradictory_renewal(tmp_path, contradiction):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock, ("7",))
    runner = FakeRunner()
    enqueue(
        scheduler,
        tmp_path,
        "trainer",
        WorkKind.TRAINER,
        eligible_gpus=("7",),
        lease_role="trainer",
    )
    calls = []

    def prove(request):
        proof = renewable_proof(request)
        if calls:
            proof.update(contradiction)
        calls.append(proof)
        return proof

    executor = make_executor(
        tmp_path,
        scheduler,
        runner,
        clock,
        gpu_ids=("7",),
        lease_proof=prove,
    )
    assert executor.once()["gpus"][0]["status"] == "started"

    result = executor.once()["gpus"][0]

    assert result["status"] == "safety-halt"
    assert "proof" in result["reason"]
    assert len(runner.spawns) == 1


def test_spec_is_strict_canonical_self_hashed_and_cli_is_bounded(tmp_path):
    path = write_executor_spec(tmp_path)
    spec = load_executor_spec(path)
    assert spec.gpu_ids == ("0",)
    assert parse_args(["status", "--spec", str(path)]).command == "status"
    assert parse_args(["once", "--spec", str(path)]).command == "once"
    assert parse_args(["watch", "--spec", str(path)]).command == "watch"
    assert parse_args(["--spec", str(path), "status"]).spec == path

    value = json.loads(path.read_text(encoding="utf-8"))
    value["retry_budget"] = 99
    write_canonical(path, value)
    with pytest.raises(ExecutorSpecError, match="self-hash"):
        load_executor_spec(path)

    value["spec_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "spec_sha256"}
    )
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ExecutorSpecError, match="canonical"):
        load_executor_spec(path)
