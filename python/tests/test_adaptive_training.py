import copy
import json
import time
from pathlib import Path

import pytest

from risk_score.adaptive_training import (
    DEFAULT_POLICY_PATH,
    OBSERVATION_CONTRACT,
    POLICY_HASH,
    SERVICE_SPEC_CONTRACT,
    TRIAL_RESULT_CONTRACT,
    AdaptiveTrainingService,
    AdaptiveTrainingError,
    AdaptiveTrainingStore,
    BudgetExceededError,
    EvidenceRejectedError,
    GpuInterval,
    ObservationValidationError,
    PolicyValidationError,
    RecipeConflictError,
    ServiceSpecError,
    TrialConflictError,
    bootstrap_recipe_binding,
    canonical_json_bytes,
    canonical_sha256,
    compare_and_swap_recipe_binding,
    deterministic_rank,
    deterministic_successive_halving,
    evaluate_trigger,
    gpu_budget_status,
    load_policy,
    load_candidate_handoff,
    load_adaptive_service_spec,
    load_recipe_binding,
    main,
    publish_adaptive_observation,
    publish_adaptive_service_spec,
    publish_trial_result,
    rollback_recipe_binding,
    rolling_gpu_seconds,
    validate_evidence,
    validate_policy,
    validate_recipe,
)
from risk_score.cluster_executor import WORK_SPEC_CONTRACT
from risk_score.cluster_scheduler import (
    ClusterScheduler,
    ReleaseOutcome,
    WorkKind,
    WorkState,
)


def digest(label):
    return canonical_sha256({"label": label})


def evidence(trial_id, source, value, *, round_index=0, suffix=""):
    metric = (
        "discovery_powered_terminal_utility"
        if source == "discovery"
        else "fixed_validation_loss"
    )
    return {
        "artifact_sha256": digest(
            f"{trial_id}:{source}:{round_index}:{suffix}:{value}"
        ),
        "finalized": True,
        "metrics": {metric: value},
        "round_index": round_index,
        "sample_count": 100,
        "schema_version": 1,
        "source": source,
        "trial_id": trial_id,
    }


def make_store(tmp_path):
    checkpoint = tmp_path / "champion.ckpt"
    checkpoint.write_bytes(b"immutable champion")
    admitted = tmp_path / "admitted-data.json"
    admitted.write_bytes(b'{"snapshot":"immutable"}\n')
    store = AdaptiveTrainingStore(tmp_path / "adaptive")
    plan = store.plan_epoch(
        admitted_samples=3_000_000,
        last_promotion_admitted_samples=0,
        candidate_queue_depth=3,
        parent_champion_model_sha256=digest("champion-model"),
        champion_checkpoint_path=checkpoint,
        admitted_data_manifest_path=admitted,
        now=1_000_000,
        timestamp_utc="2026-08-11T12:00:00.000000Z",
    )
    assert plan["planned"] is True
    return store, plan, checkpoint, admitted


def complete_round_trial(
    store,
    trial_id,
    *,
    round_index,
    utility,
    validation_loss,
    clock,
):
    started = store.start_trial(
        trial_id,
        now=clock,
        timestamp_utc=f"2026-08-11T12:{clock % 60:02d}:00.000000Z",
    )
    assert started.event_type == "trial.started"
    store.record_gpu_usage(
        trial_id,
        started_at=clock,
        ended_at=clock + 1,
        timestamp_utc=f"2026-08-11T13:{clock % 60:02d}:00.000000Z",
    )
    store.record_evidence(
        trial_id,
        evidence(
            trial_id,
            "discovery",
            utility,
            round_index=round_index,
        ),
    )
    store.record_evidence(
        trial_id,
        evidence(
            trial_id,
            "fixed_validation",
            validation_loss,
            round_index=round_index,
        ),
    )
    store.complete_trial(trial_id)


def run_epoch_to_winner(store, plan):
    survivors = list(plan["trial_ids"])
    clock = 1_000_100
    round_index = 0
    while True:
        for rank, trial_id in enumerate(survivors):
            complete_round_trial(
                store,
                trial_id,
                round_index=round_index,
                utility=float(len(survivors) - rank),
                validation_loss=float(rank),
                clock=clock,
            )
            clock += 2
        store.halve_round(plan["epoch_id"], round_index=round_index)
        status = store.status()
        epoch = status["epochs"][plan["epoch_id"]]
        if epoch["winner_trial_id"] is not None:
            return epoch["winner_trial_id"]
        survivors = epoch["survivor_trial_ids"]
        round_index += 1


class ServiceClock:
    def __init__(self, value=1_800_000_000.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def make_service(tmp_path, *, poll_interval_seconds=10.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock = ServiceClock()
    scheduler = ClusterScheduler(
        (tmp_path / "scheduler").resolve(),
        tuple(str(index) for index in range(8)),
        clock=clock,
    )
    champion = tmp_path / "service-champion.ckpt"
    champion.write_bytes(b"resumable champion")
    admitted = tmp_path / "service-admitted-data.json"
    admitted.write_bytes(b'{"snapshot":"service"}\n')
    observation_path = (tmp_path / "observation.json").resolve()
    observation = publish_adaptive_observation(
        observation_path,
        admitted_samples=3_000_000,
        last_promotion_admitted_samples=0,
        candidate_queue_depth=0,
        current_champion_model_sha256=digest("service-champion-model"),
        champion_checkpoint_path=champion.resolve(),
        admitted_data_manifest_path=admitted.resolve(),
        updated_at_unix=clock(),
    )
    spec_path = (tmp_path / "adaptive-service.json").resolve()
    spec = publish_adaptive_service_spec(
        spec_path,
        root=(tmp_path / "adaptive-service-root").resolve(),
        autonomy_policy_path=DEFAULT_POLICY_PATH.resolve(),
        scheduler_directory=(tmp_path / "scheduler").resolve(),
        observation_path=observation_path,
        trial_command_argv_template=[
            "reviewed-trial",
            "--manifest",
            "{trial_manifest_path}",
            "--result",
            "{trial_result_path}",
            "--work-id",
            "{work_id}",
        ],
        gpu_lease_guardian_argv_prefix=[
            "gpu-lease-worker",
            "--spec",
            str((tmp_path / "gpu-lease-worker.json").resolve()),
            "--expected-spec-sha256",
            digest("gpu-lease-worker-spec"),
            "--receipt",
            "{guardian_receipt}",
            "--claim-id",
            "{claim_id}",
            "--work-id",
            "{work_id}",
            "--",
        ],
        poll_interval_seconds=poll_interval_seconds,
        actor="adaptive-service-test",
    )
    service = AdaptiveTrainingService(
        spec,
        scheduler=scheduler,
        clock=clock,
        sleeper=lambda _: None,
    )
    return {
        "admitted": admitted,
        "champion": champion,
        "clock": clock,
        "observation": observation,
        "observation_path": observation_path,
        "scheduler": scheduler,
        "service": service,
        "spec": spec,
        "spec_path": spec_path,
    }


def finish_service_trial(fixture, *, utility, validation_loss, candidate=False):
    service = fixture["service"]
    scheduler = fixture["scheduler"]
    clock = fixture["clock"]
    state = service.store.status()
    trial_id = state["active_trial_id"]
    assert trial_id is not None
    trial = state["trials"][trial_id]
    round_index = trial["round_index"]
    payload = service.build_trial_work_payload(trial_id)
    work_id = payload["executor_spec"]["work_id"]
    record = scheduler.get_work(work_id)
    assert record is not None
    claim = scheduler.get_claim("7")
    if claim is None:
        claim = scheduler.claim("7", "executor-test")
    assert claim is not None and claim.work_id == work_id
    started_at = clock()
    clock.advance(1)
    candidate_model = None
    candidate_checkpoint = None
    if candidate:
        candidate_model = (
            Path(trial["manifest_path"]).parent
            / f"candidate-round-{round_index}.bin.gz"
        )
        candidate_checkpoint = (
            Path(trial["manifest_path"]).parent
            / f"candidate-round-{round_index}.ckpt"
        )
        candidate_model.write_bytes(f"candidate:{trial_id}".encode())
        candidate_checkpoint.write_bytes(f"checkpoint:{trial_id}".encode())
    result = publish_trial_result(
        service.trial_result_path(trial_id, round_index),
        trial_manifest_path=trial["manifest_path"],
        work_id=work_id,
        round_index=round_index,
        gpu_id="7",
        started_at_unix=started_at,
        ended_at_unix=clock(),
        status="completed",
        evidence=[
            evidence(
                trial_id,
                "discovery",
                utility,
                round_index=round_index,
            ),
            evidence(
                trial_id,
                "fixed_validation",
                validation_loss,
                round_index=round_index,
            ),
        ],
        candidate_model_path=candidate_model,
        candidate_checkpoint_path=candidate_checkpoint,
    )
    scheduler.release(claim, outcome=ReleaseOutcome.COMPLETED)
    status = service.once()
    return trial_id, result, status


def test_frozen_policy_is_canonical_pinned_and_strict(tmp_path):
    raw = DEFAULT_POLICY_PATH.read_bytes()
    policy = load_policy()

    assert raw == canonical_json_bytes(policy) + b"\n"
    assert canonical_sha256(policy) == POLICY_HASH
    assert policy["status"] == "frozen"
    assert policy["trigger"]["minimum_admitted_samples_without_promotion"] == 3_000_000
    assert policy["trials"]["maximum_active"] == 1
    assert policy["gpu_budget"] == {
        "host_gpu_count": 8,
        "maximum_fraction": 0.1,
        "rolling_window_seconds": 604_800,
    }

    modified = copy.deepcopy(policy)
    modified["trigger"]["minimum_admitted_samples_without_promotion"] += 1
    with pytest.raises(PolicyValidationError, match="content hash"):
        validate_policy(modified)

    noncanonical = tmp_path / "policy.json"
    noncanonical.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    with pytest.raises(PolicyValidationError, match="canonical"):
        load_policy(noncanonical)


def test_recipe_allowlist_rejects_every_protected_surface():
    policy = load_policy()
    recipe = {
        key: values[0]
        for key, values in policy["allowed_recipe_knobs"].items()
    }
    assert validate_recipe(recipe, policy) == recipe

    for forbidden in (
        "objective",
        "game_rules",
        "architecture",
        "promotion_thresholds",
        "confirmation_inputs",
        "audit_inputs",
    ):
        changed = dict(recipe)
        changed[forbidden] = {}
        with pytest.raises(AdaptiveTrainingError) as caught:
            validate_recipe(changed, policy)
        assert caught.value.code == "recipe_surface_forbidden"

    changed = dict(recipe)
    changed["learning_rate_scale"] = 1000
    with pytest.raises(AdaptiveTrainingError) as caught:
        validate_recipe(changed, policy)
    assert caught.value.code == "recipe_value_forbidden"


def test_rolling_seven_day_gpu_budget_exact_edges():
    policy = load_policy()
    maximum = 604_800 * 8 * 0.1
    now = 1_000_000.0
    exact = [GpuInterval(now - maximum, now, 1, "trial-exact")]

    assert rolling_gpu_seconds(exact, now=now) == maximum
    status = gpu_budget_status(exact, now=now, policy=policy)
    assert status.allowed is True
    assert status.remaining_gpu_seconds == 0

    over = gpu_budget_status(
        exact,
        now=now,
        requested_gpu_seconds=0.0001,
        policy=policy,
    )
    assert over.allowed is False
    assert over.projected_gpu_seconds > over.maximum_gpu_seconds

    at_left_boundary = [
        GpuInterval(now - 700_000, now - 604_800, 8, "expired")
    ]
    assert rolling_gpu_seconds(at_left_boundary, now=now) == 0

    clipped = [GpuInterval(now - 604_810, now - 604_790, 2, "clipped")]
    assert rolling_gpu_seconds(clipped, now=now) == 20


def test_trigger_requires_all_sample_queue_concurrency_and_budget_predicates():
    base = {
        "admitted_samples": 2_999_999,
        "last_promotion_admitted_samples": 0,
        "candidate_queue_depth": 3,
        "now": 1_000_000,
    }
    decision = evaluate_trigger(**base)
    assert decision.eligible is False
    assert decision.reason_codes == ("INSUFFICIENT_ADMITTED_SAMPLES",)

    eligible = evaluate_trigger(**{**base, "admitted_samples": 3_000_000})
    assert eligible.eligible is True

    queue = evaluate_trigger(
        **{**base, "admitted_samples": 3_000_000, "candidate_queue_depth": 4}
    )
    assert queue.reason_codes == ("CANDIDATE_QUEUE_UNBOUNDED",)

    active = evaluate_trigger(
        **{**base, "admitted_samples": 3_000_000, "active_trial_count": 1}
    )
    assert active.reason_codes == ("ACTIVE_TRIAL_EXISTS",)

    second_epoch_too_soon = evaluate_trigger(
        **{
            **base,
            "admitted_samples": 5_999_999,
            "last_trial_epoch_admitted_samples": 3_000_000,
        }
    )
    assert second_epoch_too_soon.eligible is False

    maximum = 604_800 * 8 * 0.1
    no_room = evaluate_trigger(
        **{
            **base,
            "admitted_samples": 3_000_000,
            "gpu_intervals": [
                GpuInterval(1_000_000 - maximum, 1_000_000, 1, "full")
            ],
        }
    )
    assert no_room.reason_codes == ("GPU_BUDGET_EXHAUSTED",)


def test_deterministic_ranking_and_successive_halving_are_input_order_stable():
    trial_ids = ["trial-c", "trial-a", "trial-d", "trial-b"]
    rows = []
    for trial_id in trial_ids:
        rows.extend(
            [
                evidence(trial_id, "discovery", 1.0),
                evidence(trial_id, "fixed_validation", 0.25),
            ]
        )

    expected = ("trial-a", "trial-b", "trial-c", "trial-d")
    assert deterministic_rank(rows, trial_ids=trial_ids, round_index=0) == expected
    assert (
        deterministic_rank(
            list(reversed(rows)),
            trial_ids=list(reversed(trial_ids)),
            round_index=0,
        )
        == expected
    )

    ranking, survivors = deterministic_successive_halving(
        rows,
        trial_ids=trial_ids,
        round_index=0,
    )
    assert ranking == expected
    assert survivors == ("trial-a", "trial-b")


def test_confirmation_audit_and_nested_holdout_references_are_rejected():
    trial_id = "trial-safe"
    for source in ("confirmation", "audit"):
        row = evidence(trial_id, "discovery", 1.0)
        row["source"] = source
        with pytest.raises(EvidenceRejectedError) as caught:
            validate_evidence(row)
        assert caught.value.code == "holdout_evidence_forbidden"

    row = evidence(trial_id, "discovery", 1.0)
    row["confirmation_schedule_hash"] = digest("secret")
    with pytest.raises(EvidenceRejectedError) as caught:
        validate_evidence(row)
    assert caught.value.code == "holdout_reference_forbidden"

    row = evidence(trial_id, "discovery", 1.0)
    row["artifact_sha256"] = "audit/" + row["artifact_sha256"]
    with pytest.raises(EvidenceRejectedError) as caught:
        validate_evidence(row)
    assert caught.value.code == "holdout_reference_forbidden"


def test_trial_planning_is_content_addressed_and_isolated(tmp_path):
    store, plan, checkpoint, admitted = make_store(tmp_path)
    assert len(plan["trial_ids"]) == 8
    assert len(set(plan["trial_ids"])) == 8

    recipe_hashes = set()
    for trial_id in plan["trial_ids"]:
        trial_dir = store.trials_dir / trial_id
        manifest = json.loads((trial_dir / "trial.json").read_text())
        assert manifest["trial_id"] == trial_id
        assert manifest["isolation_root"] == str(trial_dir)
        assert Path(manifest["recipe_path"]).parent == store.recipes_dir
        assert (trial_dir / "evidence").is_dir()
        recipe_hashes.add(manifest["recipe_sha256"])
    assert len(recipe_hashes) == 8

    replay = store.plan_epoch(
        admitted_samples=3_000_000,
        last_promotion_admitted_samples=0,
        candidate_queue_depth=3,
        parent_champion_model_sha256=digest("champion-model"),
        champion_checkpoint_path=checkpoint,
        admitted_data_manifest_path=admitted,
        now=1_000_000,
    )
    assert replay["planned"] is True
    assert replay["reused"] is True
    assert replay["epoch_id"] == plan["epoch_id"]
    assert len(store.events()) == 1

    first = plan["trial_ids"][0]
    store.start_trial(first, now=1_000_100)
    with pytest.raises(TrialConflictError) as caught:
        store.start_trial(plan["trial_ids"][1], now=1_000_101)
    assert caught.value.code == "active_trial_exists"


def test_immutable_checkpoint_and_data_bindings_are_rechecked(tmp_path):
    store, plan, checkpoint, admitted = make_store(tmp_path)
    checkpoint.write_bytes(b"mutated champion")
    with pytest.raises(TrialConflictError) as caught:
        store.start_trial(plan["trial_ids"][0], now=1_000_100)
    assert caught.value.code == "immutable_trial_binding_changed"

    checkpoint.write_bytes(b"immutable champion")
    admitted.write_bytes(b'{"snapshot":"changed"}\n')
    with pytest.raises(TrialConflictError) as caught:
        store.start_trial(plan["trial_ids"][0], now=1_000_100)
    assert caught.value.code == "immutable_trial_binding_changed"


def test_event_replay_repairs_status_and_exact_retries_are_idempotent(tmp_path):
    store, plan, _, _ = make_store(tmp_path)
    trial_id = plan["trial_ids"][0]
    first_start = store.start_trial(trial_id, now=1_000_100)
    retry_start = store.start_trial(trial_id, now=1_000_500)
    assert retry_start.event_hash == first_start.event_hash

    row = evidence(trial_id, "discovery", 1.5)
    first_evidence = store.record_evidence(trial_id, row)
    retry_evidence = store.record_evidence(trial_id, copy.deepcopy(row))
    assert retry_evidence.event_hash == first_evidence.event_hash
    assert len(store.events()) == 3

    store.status_path.write_text('{"torn":', encoding="utf-8")
    recovered = AdaptiveTrainingStore(store.root).reconcile()
    assert recovered["active_trial_id"] == trial_id
    assert recovered["last_sequence"] == 3
    assert len(recovered["trials"][trial_id]["evidence"]) == 1
    assert json.loads(store.status_path.read_text())["status_sha256"] == recovered[
        "status_sha256"
    ]

    temporary = store.events_dir / ".crash-window.tmp"
    temporary.write_text("partial", encoding="utf-8")
    assert AdaptiveTrainingStore(store.root).status()["last_sequence"] == 3


def test_usage_cannot_exceed_round_reservation(tmp_path):
    store, plan, _, _ = make_store(tmp_path)
    trial_id = plan["trial_ids"][0]
    store.start_trial(trial_id, now=1_000_100)
    with pytest.raises(BudgetExceededError) as caught:
        store.record_gpu_usage(
            trial_id,
            started_at=1_000_100,
            ended_at=1_000_100 + 14_401,
        )
    assert caught.value.code == "trial_reservation_exceeded"


def test_winner_handoff_has_no_direct_promotion_and_only_discovery_evidence(
    tmp_path,
):
    store, plan, _, _ = make_store(tmp_path)
    winner = run_epoch_to_winner(store, plan)
    candidate = tmp_path / "candidate.bin.gz"
    candidate.write_bytes(b"winning candidate")
    trial_checkpoint = tmp_path / "trial.ckpt"
    trial_checkpoint.write_bytes(b"winning checkpoint")

    result = store.create_handoff(
        winner,
        candidate_path=candidate,
        candidate_checkpoint_path=trial_checkpoint,
    )
    handoff = result["handoff"]
    assert handoff["direct_promotion_permitted"] is False
    assert handoff["promotion_path"]["policy_version"].endswith("-v3")
    assert handoff["promotion_path"]["required_stages"] == [
        "confirmation",
        "canary",
        "audit",
    ]
    assert {item["source"] for item in handoff["evidence"]} == {
        "discovery",
        "fixed_validation",
    }
    assert handoff["parent_champion_model_sha256"] == digest(
        "champion-model"
    )
    assert Path(result["handoff_path"]).is_file()
    assert (
        store.candidate_handoffs_dir
        / f"{handoff['candidate']['sha256']}.json"
    ).is_file()
    loaded_handoff = load_candidate_handoff(
        result["handoff_path"],
        expected_candidate_sha256=handoff["candidate"]["sha256"],
    )
    assert loaded_handoff["handoff_id"] == handoff["handoff_id"]
    assert store.status()["trials"][winner]["state"] == "handed_off"

    replay = store.create_handoff(
        winner,
        candidate_path=candidate,
        candidate_checkpoint_path=trial_checkpoint,
    )
    assert replay["handoff"]["handoff_id"] == handoff["handoff_id"]


def test_recipe_cas_and_rollback_restore_model_checkpoint_and_watermarks(tmp_path):
    binding_path = tmp_path / "active-recipe.json"
    first = bootstrap_recipe_binding(
        binding_path,
        recipe_sha256=digest("recipe-0"),
        recipe_path="/recipes/recipe-0.json",
        champion_model_sha256=digest("model-0"),
        champion_checkpoint_sha256=digest("checkpoint-0"),
        admitted_data_manifest_sha256=digest("data-0"),
        data_watermark_sha256s={
            "data": digest("data-watermark-0"),
            "shuffle": digest("shuffle-watermark-0"),
        },
        generation_id="generation-0",
        activated_at_utc="2026-08-11T12:00:00.000000Z",
    )
    assert load_recipe_binding(binding_path) == first

    second = compare_and_swap_recipe_binding(
        binding_path,
        expected_record_sha256=first["record_sha256"],
        recipe_sha256=digest("recipe-1"),
        recipe_path="/recipes/recipe-1.json",
        champion_model_sha256=digest("model-1"),
        champion_checkpoint_sha256=digest("checkpoint-1"),
        admitted_data_manifest_sha256=digest("data-1"),
        data_watermark_sha256s={
            "data": digest("data-watermark-1"),
            "shuffle": digest("shuffle-watermark-1"),
        },
        generation_id="generation-1",
        activated_at_utc="2026-08-11T13:00:00.000000Z",
    )
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert second["rollback"]["restore_recipe_sha256"] == first["recipe_sha256"]

    replay = compare_and_swap_recipe_binding(
        binding_path,
        expected_record_sha256=first["record_sha256"],
        recipe_sha256=digest("recipe-1"),
        recipe_path="/recipes/recipe-1.json",
        champion_model_sha256=digest("model-1"),
        champion_checkpoint_sha256=digest("checkpoint-1"),
        admitted_data_manifest_sha256=digest("data-1"),
        data_watermark_sha256s={
            "data": digest("data-watermark-1"),
            "shuffle": digest("shuffle-watermark-1"),
        },
        generation_id="generation-1",
    )
    assert replay["record_sha256"] == second["record_sha256"]

    with pytest.raises(RecipeConflictError) as caught:
        compare_and_swap_recipe_binding(
            binding_path,
            expected_record_sha256=digest("stale"),
            recipe_sha256=digest("recipe-other"),
            recipe_path="/recipes/other.json",
            champion_model_sha256=digest("model-other"),
            champion_checkpoint_sha256=digest("checkpoint-other"),
            admitted_data_manifest_sha256=digest("data-other"),
            data_watermark_sha256s={"data": digest("watermark-other")},
            generation_id="generation-other",
        )
    assert caught.value.code == "stale_recipe_binding"

    restored = rollback_recipe_binding(
        binding_path,
        expected_record_sha256=second["record_sha256"],
        rollback=second["rollback"],
        activated_at_utc="2026-08-11T14:00:00.000000Z",
    )
    assert restored["recipe_sha256"] == first["recipe_sha256"]
    assert restored["champion_model_sha256"] == first["champion_model_sha256"]
    assert (
        restored["champion_checkpoint_sha256"]
        == first["champion_checkpoint_sha256"]
    )
    assert restored["data_watermark_sha256s"] == first[
        "data_watermark_sha256s"
    ]


def test_service_spec_is_canonical_self_hashed_and_gpu7_fixed(tmp_path):
    fixture = make_service(tmp_path)
    spec = load_adaptive_service_spec(fixture["spec_path"])
    raw = json.loads(fixture["spec_path"].read_text(encoding="utf-8"))

    assert raw["contract"] == SERVICE_SPEC_CONTRACT
    assert raw["spec_sha256"] == spec.spec_sha256
    assert raw["autonomy_policy_sha256"] == POLICY_HASH
    assert spec.gpu7_id == "7"

    changed = dict(raw)
    changed["gpu7_id"] = "6"
    changed.pop("spec_sha256")
    changed["spec_sha256"] = canonical_sha256(changed)
    invalid = (tmp_path / "invalid-service.json").resolve()
    invalid.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(ServiceSpecError, match="GPU ID 7"):
        load_adaptive_service_spec(invalid)

    noncanonical = (tmp_path / "noncanonical-service.json").resolve()
    noncanonical.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(ServiceSpecError, match="canonical"):
        load_adaptive_service_spec(noncanonical)


def test_service_rejects_stale_or_rebound_observation(tmp_path):
    stale = make_service(tmp_path / "stale")
    stale["clock"].advance(21)
    with pytest.raises(ObservationValidationError) as caught:
        stale["service"].once()
    assert caught.value.code == "stale_observation"
    assert stale["service"].store.events() == ()

    rebound = make_service(tmp_path / "rebound")
    assert rebound["observation"]["contract"] == OBSERVATION_CONTRACT
    rebound["champion"].write_bytes(b"changed champion checkpoint")
    with pytest.raises(ObservationValidationError) as caught:
        rebound["service"].once()
    assert caught.value.code == "invalid_observation_binding"
    assert rebound["service"].store.events() == ()


def test_service_enqueues_exact_guarded_gpu7_backfill_and_only_one_active(
    tmp_path,
):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    scheduler = fixture["scheduler"]

    first = service.once()
    state = service.store.status()
    trial_id = state["active_trial_id"]
    assert trial_id is not None
    assert len(scheduler.reconstruct().work) == 1
    record = next(iter(scheduler.reconstruct().work.values()))
    work_spec = record.payload["executor_spec"]

    assert record.kind == WorkKind.BACKFILL
    assert record.state == WorkState.QUEUED
    assert record.eligible_gpus == ("7",)
    assert record.preferred_gpu == "7"
    assert record.preemptible is True
    assert work_spec["contract"] == WORK_SPEC_CONTRACT
    assert work_spec["kind"] == WorkKind.BACKFILL.value
    assert work_spec["eligible_gpus"] == ["7"]
    assert work_spec["lease_role"] == "none"
    prefix = list(fixture["spec"].gpu_lease_guardian_argv_prefix)
    assert work_spec["argv"][: len(prefix)] == prefix
    assert state["trials"][trial_id]["manifest_path"] in work_spec["argv"]
    assert str(service.trial_result_path(trial_id, 0)) in work_spec["argv"]
    assert work_spec["work_id"] in work_spec["argv"]
    body = dict(work_spec)
    supplied_hash = body.pop("spec_sha256")
    assert supplied_hash == canonical_sha256(body)
    assert first["active_work_id"] == work_spec["work_id"]

    replay = service.once()
    assert replay["active_trial_id"] == trial_id
    assert len(scheduler.reconstruct().work) == 1
    assert len(
        [
            event
            for event in service.store.events()
            if event.event_type == "trial.started"
        ]
    ) == 1


def test_scheduler_completion_ingests_usage_evidence_and_advances(tmp_path):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    service.once()
    completed, result, status_value = finish_service_trial(
        fixture,
        utility=3.0,
        validation_loss=0.25,
    )
    state = service.store.status()
    trial = state["trials"][completed]

    assert result["contract"] == TRIAL_RESULT_CONTRACT
    assert trial["state"] == "complete"
    assert len(trial["gpu_usage"]) == 1
    assert {item["source"] for item in trial["evidence"]} == {
        "discovery",
        "fixed_validation",
    }
    assert status_value["active_trial_id"] != completed
    completed_action = next(
        item
        for item in status_value["actions"]
        if item["action"] == "trial-completed"
    )
    assert completed_action["result_sha256"] == result["result_sha256"]


def test_service_halves_closed_round_deterministically(tmp_path):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    service.once()
    initial_epoch = service.store.status()["active_epoch_id"]

    for index in range(8):
        finish_service_trial(
            fixture,
            utility=float(8 - index),
            validation_loss=float(index),
        )

    state = service.store.status()
    epoch = state["epochs"][initial_epoch]
    assert len(epoch["survivor_trial_ids"]) == 4
    assert state["active_trial_id"] in epoch["survivor_trial_ids"]
    assert {
        state["trials"][trial_id]["round_index"]
        for trial_id in epoch["survivor_trial_ids"]
    } == {1}
    halving_events = [
        event
        for event in service.store.events()
        if event.event_type == "round.halved"
    ]
    assert len(halving_events) == 1
    assert halving_events[0].payload["survivors"] == epoch[
        "survivor_trial_ids"
    ]


def test_service_emits_only_final_winner_candidate_indexed_handoff(tmp_path):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    service.once()
    final_results = {}

    for index in range(20):
        state = service.store.status()
        if state["active_epoch_id"] is None:
            break
        trial_id = state["active_trial_id"]
        assert trial_id is not None
        round_index = state["trials"][trial_id]["round_index"]
        completed, result, _ = finish_service_trial(
            fixture,
            utility=float(100 - index),
            validation_loss=float(index),
            candidate=round_index == 2,
        )
        if round_index == 2:
            final_results[completed] = result
    else:
        pytest.fail("adaptive service did not finish its bounded epoch")

    state = service.store.status()
    epoch = next(iter(state["epochs"].values()))
    winner = epoch["winner_trial_id"]
    assert epoch["state"] == "handed_off"
    assert winner in final_results
    handoff_path = Path(state["trials"][winner]["handoff_path"])
    handoff = load_candidate_handoff(
        handoff_path,
        expected_candidate_sha256=final_results[winner]["candidate_model"][
            "sha256"
        ],
    )
    assert handoff["direct_promotion_permitted"] is False
    assert handoff["candidate_checkpoint"]["sha256"] == final_results[winner][
        "candidate_checkpoint"
    ]["sha256"]
    indexed = list(service.store.candidate_handoffs_dir.glob("*.json"))
    assert indexed == [
        service.store.candidate_handoffs_dir
        / f"{handoff['candidate']['sha256']}.json"
    ]


def test_service_replays_crash_between_start_event_and_scheduler_enqueue(tmp_path):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    observation = fixture["observation"]
    plan = service.store.plan_epoch(
        admitted_samples=observation["admitted_samples"],
        last_promotion_admitted_samples=observation[
            "last_promotion_admitted_samples"
        ],
        candidate_queue_depth=observation["candidate_queue_depth"],
        parent_champion_model_sha256=observation[
            "current_champion_model_sha256"
        ],
        champion_checkpoint_path=observation["champion_checkpoint"]["path"],
        champion_checkpoint_sha256=observation["champion_checkpoint"]["sha256"],
        admitted_data_manifest_path=observation[
            "admitted_data_manifest"
        ]["path"],
        admitted_data_manifest_sha256=observation[
            "admitted_data_manifest"
        ]["sha256"],
        now=fixture["clock"](),
    )
    trial_id = plan["trial_ids"][0]
    service.store.start_trial(trial_id, now=fixture["clock"]())
    assert fixture["scheduler"].reconstruct().work == {}

    replay = service.once()
    assert replay["active_trial_id"] == trial_id
    assert len(fixture["scheduler"].reconstruct().work) == 1
    enqueued = next(
        action
        for action in replay["actions"]
        if action["action"] == "trial-enqueued"
    )
    assert enqueued["replay"] is True
    second = service.once()
    assert second["active_trial_id"] == trial_id
    assert len(fixture["scheduler"].reconstruct().work) == 1


def test_service_reports_real_rolling_budget_exhaustion(tmp_path):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    store = service.store
    clock = fixture["clock"]
    observation = fixture["observation"]

    def plan(admitted_samples, candidate_queue_depth):
        return store.plan_epoch(
            admitted_samples=admitted_samples,
            last_promotion_admitted_samples=0,
            candidate_queue_depth=candidate_queue_depth,
            parent_champion_model_sha256=observation[
                "current_champion_model_sha256"
            ],
            champion_checkpoint_path=observation["champion_checkpoint"]["path"],
            champion_checkpoint_sha256=observation["champion_checkpoint"][
                "sha256"
            ],
            admitted_data_manifest_path=observation[
                "admitted_data_manifest"
            ]["path"],
            admitted_data_manifest_sha256=observation[
                "admitted_data_manifest"
            ]["sha256"],
            now=clock(),
        )

    def complete_round(epoch_id, round_index, reservation):
        state = store.status()
        survivors = list(state["epochs"][epoch_id]["survivor_trial_ids"])
        for rank, trial_id in enumerate(survivors):
            started_at = clock()
            store.start_trial(trial_id, now=started_at)
            clock.advance(reservation)
            store.record_gpu_usage(
                trial_id,
                started_at=started_at,
                ended_at=clock(),
            )
            store.record_evidence(
                trial_id,
                evidence(
                    trial_id,
                    "discovery",
                    float(len(survivors) - rank),
                    round_index=round_index,
                ),
            )
            store.record_evidence(
                trial_id,
                evidence(
                    trial_id,
                    "fixed_validation",
                    float(rank),
                    round_index=round_index,
                ),
            )
            store.complete_trial(trial_id)
        store.halve_round(epoch_id, round_index=round_index)

    first = plan(3_000_000, 0)
    for round_index, reservation in enumerate((14_400, 28_800, 57_600)):
        complete_round(first["epoch_id"], round_index, reservation)
    winner = store.status()["epochs"][first["epoch_id"]]["winner_trial_id"]
    candidate = Path(store.trials_dir / winner / "budget-candidate.bin.gz")
    checkpoint = Path(store.trials_dir / winner / "budget-candidate.ckpt")
    candidate.write_bytes(b"budget candidate")
    checkpoint.write_bytes(b"budget checkpoint")
    store.create_handoff(
        winner,
        candidate_path=candidate,
        candidate_checkpoint_path=checkpoint,
    )

    second = plan(6_000_000, 1)
    complete_round(second["epoch_id"], 0, 14_400)
    publish_adaptive_observation(
        fixture["observation_path"],
        admitted_samples=6_000_000,
        last_promotion_admitted_samples=0,
        candidate_queue_depth=1,
        current_champion_model_sha256=observation[
            "current_champion_model_sha256"
        ],
        champion_checkpoint_path=fixture["champion"].resolve(),
        admitted_data_manifest_path=fixture["admitted"].resolve(),
        updated_at_unix=clock(),
    )

    status_value = service.once()
    assert status_value["blocked_reason"] == "GPU_BUDGET_EXHAUSTED"
    assert store.status()["active_trial_id"] is None
    assert fixture["scheduler"].reconstruct().work == {}


def test_trial_result_rejects_forbidden_holdout_evidence(tmp_path):
    fixture = make_service(tmp_path)
    service = fixture["service"]
    service.once()
    state = service.store.status()
    trial_id = state["active_trial_id"]
    trial = state["trials"][trial_id]
    payload = service.build_trial_work_payload(trial_id)
    forbidden = evidence(trial_id, "discovery", 1.0)
    forbidden["source"] = "confirmation"

    with pytest.raises(EvidenceRejectedError) as caught:
        publish_trial_result(
            service.trial_result_path(trial_id, 0),
            trial_manifest_path=trial["manifest_path"],
            work_id=payload["executor_spec"]["work_id"],
            round_index=0,
            gpu_id="7",
            started_at_unix=fixture["clock"](),
            ended_at_unix=fixture["clock"](),
            status="completed",
            evidence=[
                forbidden,
                evidence(trial_id, "fixed_validation", 0.2),
            ],
        )
    assert caught.value.code == "holdout_evidence_forbidden"
    assert service.store.status()["trials"][trial_id]["evidence"] == []


def test_watch_reconciles_and_persists_status_without_stdout(tmp_path, capsys):
    fixture = make_service(tmp_path)
    calls = []

    def stop_after_first(interval):
        calls.append(interval)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fixture["service"].watch(sleeper=stop_after_first)

    assert calls == [fixture["spec"].poll_interval_seconds]
    written = json.loads(
        fixture["service"].service_status_path.read_text(encoding="utf-8")
    )
    assert written["contract"].endswith("service-status-v1")
    assert capsys.readouterr().out == ""


def test_cli_supports_service_spec_and_legacy_status(tmp_path, capsys):
    fixture = make_service(tmp_path)
    publish_adaptive_observation(
        fixture["observation_path"],
        admitted_samples=3_000_000,
        last_promotion_admitted_samples=0,
        candidate_queue_depth=0,
        current_champion_model_sha256=digest("service-champion-model"),
        champion_checkpoint_path=fixture["champion"].resolve(),
        admitted_data_manifest_path=fixture["admitted"].resolve(),
        updated_at_unix=time.time(),
    )

    assert main(["--spec", str(fixture["spec_path"]), "status"]) == 0
    service_status = json.loads(capsys.readouterr().out)
    assert service_status["contract"].endswith("service-status-v1")

    assert main(["--spec", str(fixture["spec_path"]), "once"]) == 0
    once_status = json.loads(capsys.readouterr().out)
    assert once_status["active_trial_id"] is not None

    legacy_root = (tmp_path / "legacy-root").resolve()
    assert main(["--root", str(legacy_root), "status"]) == 0
    legacy_status = json.loads(capsys.readouterr().out)
    assert legacy_status["contract"].endswith("training-status-v1")
