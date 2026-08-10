import json

import pytest

from risk_score.cluster_scheduler import (
    ClusterScheduler,
    GpuBusyError,
    IdleReason,
    PRIORITY_RANK,
    PreemptionStatus,
    SchedulerConflictError,
    SchedulerStateError,
    WorkItem,
    WorkKind,
    WorkState,
    canonical_json_bytes,
    canonical_sha256,
)


class FakeClock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        self.value += 0.25
        return self.value


def make_scheduler(tmp_path, gpu_ids=("gpu-a", "gpu-b"), clock=None):
    return ClusterScheduler(
        tmp_path / "scheduler",
        gpu_ids,
        clock=clock or FakeClock(),
    )


def test_fixed_priority_order_is_deterministic(tmp_path):
    scheduler = make_scheduler(tmp_path, ("gpu-a",))
    enqueue_order = (
        WorkKind.BACKFILL,
        WorkKind.SELF_PLAY,
        WorkKind.TRAINER,
        WorkKind.CURATION,
        WorkKind.SCREENING,
        WorkKind.PROMOTION_CANARY,
        WorkKind.PROMOTION_CONFIRMATION,
        WorkKind.RECOVERY,
    )
    for index, kind in enumerate(enqueue_order):
        scheduler.enqueue(f"work-{index}", kind)

    claimed_kinds = []
    for index in range(len(enqueue_order)):
        claim = scheduler.claim("gpu-a", f"owner-{index}")
        claimed_kinds.append(scheduler.get_work(claim.work_id).kind)
        scheduler.release(claim)

    assert claimed_kinds == [
        WorkKind.RECOVERY,
        WorkKind.PROMOTION_CONFIRMATION,
        WorkKind.PROMOTION_CANARY,
        WorkKind.SCREENING,
        WorkKind.CURATION,
        WorkKind.TRAINER,
        WorkKind.SELF_PLAY,
        WorkKind.BACKFILL,
    ]
    assert PRIORITY_RANK[WorkKind.PROMOTION_CONFIRMATION] == PRIORITY_RANK[
        WorkKind.PROMOTION_CANARY
    ]


def test_priority_ties_use_fifo_then_work_id(tmp_path):
    scheduler = make_scheduler(tmp_path, ("gpu-a",))
    scheduler.enqueue("screen-z", WorkKind.SCREENING)
    scheduler.enqueue("screen-a", WorkKind.SCREENING)

    first = scheduler.claim("gpu-a", "worker")
    assert first.work_id == "screen-z"
    scheduler.release(first)
    assert scheduler.claim("gpu-a", "worker").work_id == "screen-a"


def test_gpu_eligibility_and_work_stealing(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.enqueue(
        "gpu-a-only",
        WorkKind.RECOVERY,
        eligible_gpus=("gpu-a",),
    )
    scheduler.enqueue(
        "stealable",
        WorkKind.CURATION,
        eligible_gpus=("gpu-a", "gpu-b"),
        preferred_gpu="gpu-a",
    )

    stolen = scheduler.claim("gpu-b", "worker-b")
    assert stolen.work_id == "stealable"
    assert stolen.stolen is True
    assert scheduler.claim("gpu-a", "worker-a").work_id == "gpu-a-only"


def test_duplicate_enqueue_is_idempotent_and_conflicts_are_rejected(tmp_path):
    clock = FakeClock()
    scheduler = make_scheduler(tmp_path, clock=clock)
    item = WorkItem(
        "same-id",
        WorkKind.SELF_PLAY,
        eligible_gpus=("gpu-b", "gpu-a"),
        preemptible=True,
        payload={"model": "candidate", "visits": 128},
    )
    first = scheduler.enqueue(item)
    revision = scheduler.reconstruct().revision
    retry = scheduler.enqueue(
        work_id="same-id",
        kind="self_play",
        eligible_gpus=("gpu-a", "gpu-b"),
        preemptible=True,
        payload={"visits": 128, "model": "candidate"},
    )

    assert retry == first
    assert scheduler.reconstruct().revision == revision
    assert len(scheduler.reconstruct().work) == 1
    with pytest.raises(SchedulerConflictError, match="different metadata"):
        scheduler.enqueue(
            "same-id",
            WorkKind.SELF_PLAY,
            eligible_gpus=("gpu-a",),
            preemptible=True,
            payload={"model": "candidate", "visits": 128},
        )


def test_one_owner_per_gpu_survives_multiple_scheduler_instances(tmp_path):
    root = tmp_path / "scheduler"
    first_scheduler = ClusterScheduler(root, ("gpu-a",), clock=FakeClock())
    first_scheduler.enqueue("trainer", WorkKind.TRAINER)
    first_claim = first_scheduler.claim("gpu-a", "owner-one")

    restarted = ClusterScheduler(root, ("gpu-a",), clock=FakeClock())
    assert restarted.claim("gpu-a", "owner-one") == first_claim
    with pytest.raises(GpuBusyError, match="owner-one"):
        restarted.claim("gpu-a", "owner-two")

    snapshot = restarted.reconstruct()
    assert snapshot.active_owners == {"gpu-a": "owner-one"}
    assert len(snapshot.claims) == 1


def test_preemption_is_cooperative_idempotent_and_preemptible_only(tmp_path):
    scheduler = make_scheduler(tmp_path, ("gpu-a",))
    scheduler.enqueue(
        "backfill",
        WorkKind.BACKFILL,
        preemptible=True,
    )
    backfill_claim = scheduler.claim("gpu-a", "backfill-executor")
    scheduler.enqueue("recovery", WorkKind.RECOVERY)

    request = scheduler.request_preemption(
        "gpu-a",
        "control-plane",
        for_work_id="recovery",
    )
    retry = scheduler.request_preemption(
        "gpu-a",
        "control-plane",
        for_work_id="recovery",
    )

    assert retry == request
    assert request.running_work_id == "backfill"
    assert request.status == PreemptionStatus.PENDING
    assert scheduler.get_claim("gpu-a") == backfill_claim
    assert scheduler.get_work("backfill").state == WorkState.CLAIMED

    release = scheduler.release(backfill_claim, requeue=True)
    assert release.outcome.value == "requeue"
    assert scheduler.get_work("backfill").state == WorkState.QUEUED
    assert scheduler.idle_status("gpu-a").reason == IdleReason.CHECKPOINT_BOUNDARY
    assert scheduler.reconstruct().preemption_requests[0].status == (
        PreemptionStatus.BOUNDARY_REACHED
    )
    assert scheduler.claim("gpu-a", "recovery-executor").work_id == "recovery"

    second = make_scheduler(tmp_path / "second", ("gpu-a",))
    second.enqueue("trainer", WorkKind.TRAINER, preemptible=False)
    second.claim("gpu-a", "trainer-executor")
    second.enqueue("urgent", WorkKind.RECOVERY)
    assert second.request_preemption("gpu-a", for_work_id="urgent") is None
    assert second.pending_preemptions() == ()


def test_backfill_keeps_an_idle_eligible_gpu_busy(tmp_path):
    scheduler = make_scheduler(tmp_path)
    scheduler.enqueue(
        "urgent-on-b",
        WorkKind.RECOVERY,
        eligible_gpus=("gpu-b",),
    )
    scheduler.enqueue(
        "backfill-on-a",
        WorkKind.BACKFILL,
        eligible_gpus=("gpu-a",),
        preemptible=True,
    )

    # Ineligible high-priority work does not leave gpu-a idle.
    assert scheduler.claim("gpu-a", "worker-a").work_id == "backfill-on-a"
    assert scheduler.claim("gpu-b", "worker-b").work_id == "urgent-on-b"


def test_restart_reconstructs_claim_queue_and_idempotent_release(tmp_path):
    root = tmp_path / "scheduler"
    scheduler = ClusterScheduler(root, ("gpu-a", "gpu-b"), clock=FakeClock())
    scheduler.enqueue("screen", WorkKind.SCREENING)
    scheduler.enqueue("trainer", WorkKind.TRAINER)
    screen_claim = scheduler.claim("gpu-a", "worker-a")

    restarted = ClusterScheduler(root, clock=FakeClock())
    snapshot = restarted.reconstruct()
    assert snapshot.claims["gpu-a"] == screen_claim
    assert snapshot.work["screen"].state == WorkState.CLAIMED
    assert [record.work_id for record in snapshot.queued] == ["trainer"]
    assert restarted.claim("gpu-a", "worker-a") == screen_claim

    release = restarted.release(screen_claim)
    revision = restarted.reconstruct().revision
    release_retry = ClusterScheduler(root, clock=FakeClock()).release(screen_claim)
    assert release_retry == release
    assert restarted.reconstruct().revision == revision
    assert restarted.get_work("screen").state == WorkState.COMPLETED
    assert restarted.claim("gpu-b", "worker-b").work_id == "trainer"


def test_idle_reason_telemetry_covers_all_scheduler_reasons(tmp_path):
    scheduler = make_scheduler(tmp_path, ("gpu-a",))
    assert scheduler.claim("gpu-a", "worker") is None
    first = scheduler.idle_status("gpu-a")
    revision = scheduler.reconstruct().revision
    assert first.reason == IdleReason.NO_RUNNABLE_WORK

    # Polling an unchanged idle state is idempotent.
    assert scheduler.claim("gpu-a", "worker") is None
    assert scheduler.reconstruct().revision == revision
    scheduler.record_idle(
        "gpu-a",
        IdleReason.CHECKPOINT_BOUNDARY,
        details={"checkpoint": "model-42"},
    )
    scheduler.record_idle(
        "gpu-a",
        IdleReason.LEASE_HANDOFF,
        details={"lease": "promotion-evaluation"},
    )
    scheduler.set_safety_halt("temperature threshold exceeded", gpu_id="gpu-a")
    assert scheduler.claim("gpu-a", "worker") is None

    reasons = [event.reason for event in scheduler.idle_events("gpu-a")]
    assert reasons == [
        IdleReason.NO_RUNNABLE_WORK,
        IdleReason.CHECKPOINT_BOUNDARY,
        IdleReason.LEASE_HANDOFF,
        IdleReason.SAFETY_HALT,
    ]
    assert scheduler.idle_status("gpu-a").details == {
        "reason": "temperature threshold exceeded"
    }


def test_state_is_canonical_hashed_and_atomically_replaced(tmp_path):
    scheduler = make_scheduler(tmp_path, ("gpu-a",))
    scheduler.enqueue(
        "unicode",
        WorkKind.CURATION,
        payload={"position": "é", "nested": {"z": 1, "a": True}},
    )

    data = scheduler.state_path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    body = dict(value)
    state_hash = body.pop("state_sha256")
    assert data == canonical_json_bytes(value) + b"\n"
    assert state_hash == canonical_sha256(body)
    assert not list(scheduler.directory.glob(f".{scheduler.state_path.name}.*.tmp"))

    value["revision"] += 1
    scheduler.state_path.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(SchedulerStateError, match="hash mismatch"):
        ClusterScheduler(scheduler.directory, ("gpu-a",))
