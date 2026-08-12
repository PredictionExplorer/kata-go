import datetime
import hashlib
import sys
from pathlib import Path

import pytest

from katago.utils.shuffle_input_gate import run_if_changed
from risk_score.adaptive_training import (
    DEFAULT_POLICY_PATH,
    load_adaptive_observation,
    publish_adaptive_service_spec,
)
from risk_score.promotion_feedback import (
    DATA_WATERMARK_CONTRACT,
    SHUFFLE_MANIFEST_FILENAME,
    SHUFFLE_WATERMARK_CONTRACT,
    PromotionFeedbackError,
    PromotionFeedbackWatcher,
    TrainerProvenanceRecorder,
    _resolve_cli_paths,
    atomic_create_json,
    atomic_replace_json,
    canonical_sha256,
    file_sha256,
    load_canonical_json,
    load_data_watermark,
    load_shuffle_manifest,
    load_shuffle_watermark,
    parse_args,
)
from risk_score.promotion_state import (
    ChampionRecord,
    EventProvenance,
    bootstrap_champion,
)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_json(path, value)


def add_generation(layout, generation, candidate, *, with_game=True):
    transaction = layout["promotion"] / "transactions" / generation
    canonical_write(
        transaction / "intent.json",
        {
            "schema_version": 1,
            "generation_id": generation,
            "candidate_hash": candidate,
        },
    )
    root = layout["admitted"] / generation / "worker-000"
    tdata = root / "tdata" / f"{generation}.npz"
    tdata.parent.mkdir(parents=True)
    tdata.write_bytes(f"tdata:{generation}".encode("utf-8"))
    if with_game:
        sgfs = root / "sgfs" / f"{generation}.sgfs"
        sgfs.parent.mkdir()
        sgfs.write_text(f"(;GM[1]C[{generation}])\n", encoding="utf-8")
    canonical_write(
        transaction / "generation-data-admitted.json",
        {
            "schema_version": 1,
            "model_hash": candidate,
            "manifest_hash": digest(f"manifest:{generation}"),
        },
    )
    canonical_write(
        transaction / "complete.json",
        {
            "schema_version": 1,
            "champion_hash": candidate,
        },
    )
    return tdata


@pytest.fixture
def layout(tmp_path):
    values = {
        "promotion": (tmp_path / "promotion").resolve(),
        "admitted": (tmp_path / "selfplay").resolve(),
        "shuffles": (tmp_path / "shuffleddata").resolve(),
        "trainer": (tmp_path / "trainer-provenance").resolve(),
        "state": (tmp_path / "feedback-state").resolve(),
        "data_watermark": (tmp_path / "promotion" / "watermarks" / "data.json").resolve(),
        "shuffle_watermark": (
            tmp_path / "promotion" / "watermarks" / "shuffle.json"
        ).resolve(),
    }
    for key in ("promotion", "admitted", "shuffles", "trainer", "state"):
        values[key].mkdir(parents=True, exist_ok=True)
    return values


def watcher(layout, **kwargs):
    return PromotionFeedbackWatcher(
        promotion_root=layout["promotion"],
        admitted_root=layout["admitted"],
        shuffle_root=layout["shuffles"],
        trainer_provenance_root=layout["trainer"],
        state_root=layout["state"],
        data_watermark_path=layout["data_watermark"],
        shuffle_watermark_path=layout["shuffle_watermark"],
        strict=True,
        **kwargs,
    )


def output_command(output_root, name="20260808-210000"):
    output = output_root / name
    script = (
        "from pathlib import Path;"
        f"p=Path({str(output)!r});"
        "(p/'train').mkdir(parents=True);"
        "(p/'train'/'data.npz').write_bytes(b'shuffled');"
        "(p/'train.json').write_text('{\"range\":[0,8]}')"
    )
    return [sys.executable, "-c", script]


def run_strict_gate(layout, command):
    return run_if_changed(
        input_root=layout["admitted"],
        state_file=(layout["state"] / "shuffle-gate.json").resolve(),
        lock_file=(layout["state"] / "shuffle-gate.lock").resolve(),
        output_root=layout["shuffles"],
        force_after_seconds=0.0,
        command=command,
        data_watermark=layout["data_watermark"],
        strict_provenance=True,
    )


FIXED_NOW = datetime.datetime(
    2026, 8, 11, 12, 0, 0, tzinfo=datetime.timezone.utc
)


def adaptive_provenance():
    return EventProvenance(
        controller_hash=digest("adaptive-controller"),
        source_hash=digest("adaptive-source"),
        original_hash=digest("adaptive-original"),
        config_hash=digest("adaptive-config"),
        schedule_hash=digest("adaptive-schedule"),
        policy_hash=digest("adaptive-promotion-policy"),
    )


def configure_adaptive_observation(layout, *, with_trainer_receipt=True):
    generation = "generation-adaptive"
    champion_hash = digest("adaptive-champion")
    add_generation(layout, generation, champion_hash)
    watcher(layout).scan_once()
    gate = run_strict_gate(
        layout,
        output_command(layout["shuffles"], "20260811-120000"),
    )
    shuffle_path = layout["shuffles"] / gate["last_successful_output"]
    checkpoint = (layout["trainer"] / "checkpoint.ckpt").resolve()
    checkpoint.write_bytes(b"adaptive-checkpoint-v1")
    recorder = TrainerProvenanceRecorder(layout["trainer"], strict=True)
    recorder.bind_shuffle(shuffle_path)
    if with_trainer_receipt:
        recorder.record_consumption(
            [shuffle_path / "train" / "data.npz"],
            samples_before=0,
            samples_after=8,
        )
        recorder.record_checkpoint(checkpoint, sample_count=8)

    champion_path = layout["promotion"] / "champion.json"
    champion = bootstrap_champion(
        champion_path,
        champion_hash=champion_hash,
        generation_id=generation,
        provenance=adaptive_provenance(),
        actor="adaptive-feedback-test",
        timestamp_utc="2026-08-11T11:00:00.000000Z",
    )
    candidate_inbox = layout["promotion"].parent / "modelstobetested"
    candidate_inbox.mkdir()
    scheduler = layout["promotion"] / "scheduler"
    scheduler.mkdir()
    adaptive_root = layout["promotion"] / "adaptive"
    configs = layout["promotion"].parent / "configs"
    configs.mkdir()
    executor_spec_path = (configs / "cluster-executor.json").resolve()
    suite_spec_path = (configs / "suite-registry.json").resolve()
    canonical_write(executor_spec_path, {"name": "cluster-executor"})
    canonical_write(suite_spec_path, {"name": "suite-registry"})
    adaptive_spec_path = (configs / "adaptive-training.json").resolve()
    adaptive_spec = publish_adaptive_service_spec(
        adaptive_spec_path,
        root=adaptive_root.resolve(),
        autonomy_policy_path=DEFAULT_POLICY_PATH.resolve(),
        scheduler_directory=scheduler.resolve(),
        observation_path=(adaptive_root / "observation.json").resolve(),
        trial_command_argv_template=[
            "adaptive-trial",
            "{trial_manifest_path}",
            "{trial_result_path}",
            "{work_id}",
        ],
        gpu_lease_guardian_argv_prefix=["gpu-lease-guardian"],
        poll_interval_seconds=15.0,
        actor="adaptive-feedback-test",
    )
    runtime_path = (configs / "promotion-runtime.json").resolve()
    canonical_write(
        runtime_path,
        {
            "paths": {
                "promotionRoot": str(layout["promotion"]),
                "admittedSelfplay": str(layout["admitted"]),
                "dataWatermark": str(layout["data_watermark"]),
                "shuffleWatermark": str(layout["shuffle_watermark"]),
                "rolloutQuarantine": str(
                    layout["promotion"] / "rollouts"
                ),
                "champion": str(champion_path),
                "trainerCheckpoint": str(checkpoint),
                "candidateInbox": str(candidate_inbox),
            }
        },
    )
    services_path = (configs / "promotion-services.json").resolve()
    canonical_write(
        services_path,
        {
            "schema_version": 3,
            "contract": "risk-score-host-services-v3",
            "mutation_enabled": True,
            "full_autonomy": True,
            "service_inputs": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path in {
                    "autonomy_policy": DEFAULT_POLICY_PATH.resolve(),
                    "executor_spec": executor_spec_path,
                    "adaptive_spec": adaptive_spec_path,
                    "suite_registry_spec": suite_spec_path,
                }.items()
            },
        },
    )

    def service():
        return watcher(
            layout,
            runtime_config=runtime_path,
            feedback_recorder=lambda *_: {"recorded": True},
            now=lambda: FIXED_NOW,
        )

    return {
        "adaptive_root": adaptive_root,
        "adaptive_spec": adaptive_spec,
        "adaptive_spec_path": adaptive_spec_path,
        "champion": champion,
        "champion_path": champion_path,
        "checkpoint": checkpoint,
        "recorder": recorder,
        "runtime_path": runtime_path,
        "service": service,
        "services_path": services_path,
        "shuffle_path": shuffle_path,
    }


def test_strict_shuffle_manifest_detects_mixed_generations(layout):
    candidates = {
        "generation-one": digest("candidate-one"),
        "generation-two": digest("candidate-two"),
    }
    for generation, candidate in candidates.items():
        add_generation(layout, generation, candidate)
    watcher(layout).scan_once()

    result = run_strict_gate(layout, output_command(layout["shuffles"]))
    manifest_path = Path(result["provenance_manifest"])
    manifest = load_shuffle_manifest(manifest_path)

    assert manifest_path.name == SHUFFLE_MANIFEST_FILENAME
    assert manifest["generation_ids"] == sorted(candidates)
    assert manifest["candidate_hashes"] == candidates
    assert manifest["mixed_generation"] is True
    assert manifest["unbound_source_count"] == 0
    assert all(
        row["lineage_status"] == "admitted"
        for row in manifest["source_inventory"]
    )


def test_shuffle_and_trainer_receipts_are_idempotent_and_hash_bound(layout):
    generation = "generation-one"
    candidate = digest("candidate-one")
    add_generation(layout, generation, candidate)
    watcher(layout).scan_once()
    result = run_strict_gate(layout, output_command(layout["shuffles"]))
    shuffle_path = Path(result["last_successful_output"])
    shuffle_path = layout["shuffles"] / shuffle_path

    recorder = TrainerProvenanceRecorder(layout["trainer"], strict=True)
    binding = recorder.bind_shuffle(shuffle_path)
    train_file = shuffle_path / "train" / "data.npz"
    first = recorder.record_consumption(
        [train_file], samples_before=100, samples_after=108
    )
    restarted_recorder = TrainerProvenanceRecorder(
        layout["trainer"], strict=True
    )
    restarted_recorder.bind_shuffle(shuffle_path)
    second = restarted_recorder.record_consumption(
        [train_file], samples_before=100, samples_after=108
    )
    assert first == second
    assert first.read_bytes() == second.read_bytes()

    checkpoint = (layout["trainer"] / "checkpoint.ckpt").resolve()
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_receipt = recorder.record_checkpoint(
        checkpoint, sample_count=108
    )
    export = (layout["trainer"] / "export-s108").resolve()
    export.mkdir()
    (export / "model.ckpt").write_bytes(b"export")
    export_receipt = recorder.record_export(export, sample_count=108)

    consumption = load_canonical_json(first)
    checkpoint_value = load_canonical_json(checkpoint_receipt)
    export_value = load_canonical_json(export_receipt)
    assert consumption["generation_ids"] == [generation]
    assert checkpoint_value["consumption_receipt_sha256"] == consumption[
        "receipt_sha256"
    ]
    assert export_value["consumption_receipt_sha256"] == consumption[
        "receipt_sha256"
    ]
    assert checkpoint_value["shuffle_manifest_file_sha256"] == binding[
        "manifest_file_sha256"
    ]
    assert checkpoint_value["sample_count"] == 108
    assert export_value["sample_count"] == 108
    assert checkpoint_value["consumption_lineage"][0][
        "receipt_sha256"
    ] == consumption["receipt_sha256"]
    assert export_value["consumption_lineage_sha256"] == checkpoint_value[
        "consumption_lineage_sha256"
    ]


def test_shuffle_output_hash_drift_fails_closed(layout):
    add_generation(layout, "generation-one", digest("candidate-one"))
    watcher(layout).scan_once()
    command = output_command(layout["shuffles"])
    result = run_strict_gate(layout, command)
    output = layout["shuffles"] / result["last_successful_output"]
    (output / "train" / "data.npz").write_bytes(b"tampered")

    with pytest.raises(PromotionFeedbackError, match="output inventory"):
        run_strict_gate(layout, command)


def test_data_watermark_rejects_in_place_source_hash_drift(layout):
    source = add_generation(
        layout, "generation-one", digest("candidate-one")
    )
    service = watcher(layout)
    service.scan_once()
    original_size = source.stat().st_size
    source.write_bytes(b"x" * original_size)

    with pytest.raises(PromotionFeedbackError, match="data hash drift"):
        service.scan_once()


def test_watcher_recovers_after_restart_and_delivers_each_feedback_once(layout):
    generation = "generation-one"
    candidate = digest("candidate-one")
    add_generation(layout, generation, candidate)

    # The initial scan establishes historical-directory baselines and the
    # admitted-source watermark before the shuffler is allowed to run.
    watcher(layout).scan_once()
    result = run_strict_gate(layout, output_command(layout["shuffles"]))
    shuffle_path = layout["shuffles"] / result["last_successful_output"]
    recorder = TrainerProvenanceRecorder(layout["trainer"], strict=True)
    recorder.bind_shuffle(shuffle_path)
    recorder.record_consumption(
        [shuffle_path / "train" / "data.npz"],
        samples_before=0,
        samples_after=8,
    )

    delivered = []

    def record(generation_id, kind, evidence_path):
        delivered.append((generation_id, kind, file_sha256(evidence_path)))
        return {"recorded": True}

    first = watcher(layout, feedback_recorder=record).scan_once()
    assert {kind for _, kind, _ in delivered} == {
        "first-game",
        "first-tdata",
        "first-shuffle",
        "first-training-consumption",
    }
    assert any(
        row["kind"] == "admission" and row["delivered"] is False
        for row in first["milestones"]
    )
    data_bytes = layout["data_watermark"].read_bytes()
    shuffle_bytes = layout["shuffle_watermark"].read_bytes()

    replayed = []
    restarted = watcher(
        layout,
        feedback_recorder=lambda generation_id, kind, evidence_path: replayed.append(
            (generation_id, kind)
        ),
    )
    restarted.scan_once()

    assert replayed == []
    assert layout["data_watermark"].read_bytes() == data_bytes
    assert layout["shuffle_watermark"].read_bytes() == shuffle_bytes
    assert load_data_watermark(layout["data_watermark"])[
        "contract"
    ] == DATA_WATERMARK_CONTRACT
    assert load_shuffle_watermark(layout["shuffle_watermark"])[
        "contract"
    ] == SHUFFLE_WATERMARK_CONTRACT
    assert load_shuffle_watermark(layout["shuffle_watermark"])[
        "trainer_consumed_by_generation"
    ][generation] is True


def test_strict_watcher_preserves_historical_shuffle_directories(layout):
    historical = layout["shuffles"] / "20200101-000000"
    (historical / "train").mkdir(parents=True)
    (historical / "train" / "data.npz").write_bytes(b"legacy")
    service = watcher(layout)
    service.scan_once()
    watermark = load_shuffle_watermark(layout["shuffle_watermark"])
    assert watermark["historical_paths"][0]["path"] == str(historical)
    assert (
        TrainerProvenanceRecorder(
            layout["trainer"], strict=False
        ).bind_shuffle(historical)
        is None
    )

    unbound_new = layout["shuffles"] / "20260808-220000"
    (unbound_new / "train").mkdir(parents=True)
    (unbound_new / "train" / "data.npz").write_bytes(b"new-unbound")
    with pytest.raises(PromotionFeedbackError, match="lacks provenance"):
        service.scan_once()


def test_pending_promotion_freezes_rollback_watermark_bytes(layout):
    service = watcher(layout)
    service.scan_once()
    data_before = layout["data_watermark"].read_bytes()
    shuffle_before = layout["shuffle_watermark"].read_bytes()
    generation = "generation-pending"
    transaction = layout["promotion"] / "transactions" / generation
    canonical_write(
        transaction / "intent.json",
        {
            "schema_version": 1,
            "generation_id": generation,
            "candidate_hash": digest("candidate-pending"),
        },
    )

    result = service.scan_once()

    assert result["watermarks_frozen"] is True
    assert layout["data_watermark"].read_bytes() == data_before
    assert layout["shuffle_watermark"].read_bytes() == shuffle_before


def test_bootstrap_generation_data_is_provenance_bound_without_transaction(layout):
    generation = "generation-bootstrap"
    candidate = digest("bootstrap-candidate")
    event_body = {
        "schema_version": 1,
        "sequence": 1,
        "transition": "champion.bootstrapped",
        "champion_hash": candidate,
        "payload": {"generation_id": generation},
    }
    canonical_write(
        layout["promotion"] / "events" / "00000000000000000001.json",
        {**event_body, "event_hash": canonical_sha256(event_body)},
    )
    source = (
        layout["admitted"]
        / "continuous"
        / generation
        / "worker-000"
        / "tdata"
        / "bootstrap.npz"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"bootstrap")

    result = watcher(layout).scan_once()
    watermark = load_data_watermark(layout["data_watermark"])

    assert result["watermarks_frozen"] is False
    assert watermark["generations"][0]["generation_id"] == generation
    assert watermark["generations"][0]["candidate_hash"] == candidate


def test_strict_lineage_preserves_frozen_legacy_selfplay_sources(layout):
    legacy = layout["admitted"] / "legacy-model" / "tdata" / "legacy.npz"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy-selfplay")
    service = watcher(layout)
    service.scan_once()

    result = run_strict_gate(layout, output_command(layout["shuffles"]))
    manifest = load_shuffle_manifest(Path(result["provenance_manifest"]))

    assert manifest["generation_ids"] == []
    assert manifest["historical_source_count"] == 1
    assert manifest["unbound_source_count"] == 0
    assert manifest["source_inventory"][0][
        "lineage_status"
    ] == "historical-baseline"

    (legacy.parent / "unexpected.npz").write_bytes(b"new-unattributed")
    with pytest.raises(
        PromotionFeedbackError, match="historical source baseline changed"
    ):
        service.scan_once()


def test_service_cli_resolves_runtime_and_run_root_shorthand(layout, tmp_path):
    runtime = (tmp_path / "promotion-runtime.json").resolve()
    canonical_write(
        runtime,
        {
            "paths": {
                "promotionRoot": str(layout["promotion"]),
                "admittedSelfplay": str(layout["admitted"]),
                "dataWatermark": str(layout["data_watermark"]),
                "shuffleWatermark": str(layout["shuffle_watermark"]),
                "rolloutQuarantine": str(
                    layout["promotion"] / "rollouts"
                ),
            }
        },
    )
    run_root = layout["admitted"].parent
    args = parse_args(
        [
            "--runtime-config",
            str(runtime),
            "--run-root",
            str(run_root),
            "--mode",
            "watch",
            "--interval",
            "15",
        ]
    )

    paths = _resolve_cli_paths(args)

    assert paths["shuffle_root"] == run_root / "shuffleddata"
    assert paths["trainer_provenance_root"] == (
        layout["promotion"] / "provenance" / "trainer"
    )
    assert args.interval_seconds == 15


def test_trainer_rejects_selected_file_not_in_manifest(layout):
    add_generation(layout, "generation-one", digest("candidate-one"))
    watcher(layout).scan_once()
    result = run_strict_gate(layout, output_command(layout["shuffles"]))
    shuffle_path = layout["shuffles"] / result["last_successful_output"]
    recorder = TrainerProvenanceRecorder(layout["trainer"], strict=True)
    recorder.bind_shuffle(shuffle_path)
    outside = (layout["trainer"] / "outside.npz").resolve()
    outside.write_bytes(b"outside")

    with pytest.raises(PromotionFeedbackError, match="outside"):
        recorder.record_consumption(
            [outside], samples_before=0, samples_after=8
        )


def test_adaptive_observation_is_legacy_v2_noop(layout):
    configs = layout["promotion"].parent / "configs"
    canonical_write(
        configs / "promotion-services.json",
        {
            "schema_version": 2,
            "contract": "risk-score-host-services-v2",
            "mutation_enabled": True,
            "full_autonomy": False,
        },
    )

    result = watcher(layout).scan_once()

    assert "adaptive_observation" not in result
    assert not (layout["promotion"] / "adaptive").exists()


def test_adaptive_observation_publishes_hash_bound_snapshots(layout):
    fixture = configure_adaptive_observation(layout)

    result = fixture["service"]().scan_once()
    adaptive = result["adaptive_observation"]
    observation = load_adaptive_observation(
        fixture["adaptive_spec"].observation_path
    )

    assert adaptive["status"] == "PUBLISHED"
    assert observation["admitted_samples"] == 8
    assert observation["last_promotion_admitted_samples"] == 8
    assert observation["current_champion_model_sha256"] == fixture[
        "champion"
    ].champion_hash
    checkpoint_snapshot = Path(
        observation["champion_checkpoint"]["path"]
    )
    assert checkpoint_snapshot.parent == (
        fixture["adaptive_root"] / "parents" / "checkpoints"
    )
    assert checkpoint_snapshot != fixture["checkpoint"]
    assert checkpoint_snapshot.read_bytes() == fixture["checkpoint"].read_bytes()
    admitted_snapshot = Path(
        observation["admitted_data_manifest"]["path"]
    )
    assert admitted_snapshot.parent == (
        fixture["adaptive_root"] / "parents" / "admitted-data"
    )
    admitted_value = load_canonical_json(admitted_snapshot)
    assert admitted_value["source_inventory"]
    assert admitted_value["provenance_receipts"]


def test_adaptive_observation_updates_baseline_on_champion_change(layout):
    fixture = configure_adaptive_observation(layout)
    fixture["service"]().scan_once()
    train_file = fixture["shuffle_path"] / "train" / "data.npz"
    fixture["recorder"].record_consumption(
        [train_file], samples_before=8, samples_after=16
    )
    fixture["checkpoint"].write_bytes(b"adaptive-checkpoint-v2")
    fixture["recorder"].record_checkpoint(
        fixture["checkpoint"], sample_count=16
    )
    promoted_hash = digest("adaptive-promoted-champion")
    promoted = ChampionRecord.build(
        champion_hash=promoted_hash,
        generation_id="generation-promoted",
        previous_champion_hash=fixture["champion"].champion_hash,
        provenance=adaptive_provenance(),
        activated_at_utc="2026-08-11T11:30:00.000000Z",
        evaluation_key="adaptive-confirmation",
        pass_report_path=str(
            layout["promotion"] / "reports" / "adaptive.json"
        ),
        pass_report_hash=digest("adaptive-pass-report"),
        actor="adaptive-feedback-test",
        bootstrap=False,
    )
    atomic_replace_json(fixture["champion_path"], promoted.to_dict())

    result = fixture["service"]().scan_once()
    observation = load_adaptive_observation(
        fixture["adaptive_spec"].observation_path
    )

    assert result["adaptive_observation"]["status"] == "PUBLISHED"
    assert observation["admitted_samples"] == 16
    assert observation["last_promotion_admitted_samples"] == 16
    assert observation["current_champion_model_sha256"] == promoted_hash
    baselines = sorted(
        (
            fixture["adaptive_root"] / "parents" / "baselines"
        ).glob("*.json")
    )
    assert len(baselines) == 2


def test_adaptive_observation_checkpoint_tamper_is_blocked(layout):
    fixture = configure_adaptive_observation(layout)
    fixture["service"]().scan_once()
    observation_path = fixture["adaptive_spec"].observation_path
    before = observation_path.read_bytes()
    fixture["checkpoint"].write_bytes(b"adaptive-checkpoint-X1")

    result = fixture["service"]().scan_once()

    assert result["adaptive_observation"]["status"] == "BLOCKED"
    assert "contradicts its immutable receipt" in result[
        "adaptive_observation"
    ]["blocked_reason"]
    assert observation_path.read_bytes() == before


def test_adaptive_observation_checkpoint_race_is_blocked(
    layout, monkeypatch
):
    fixture = configure_adaptive_observation(layout)
    import risk_score.promotion_feedback as promotion_feedback

    original_copy = promotion_feedback.shutil.copyfileobj

    def racing_copy(source, destination, length):
        original_copy(source, destination, length)
        fixture["checkpoint"].write_bytes(b"adaptive-checkpoint-X1")

    monkeypatch.setattr(
        promotion_feedback.shutil, "copyfileobj", racing_copy
    )

    result = fixture["service"]().scan_once()

    assert result["adaptive_observation"]["status"] == "BLOCKED"
    assert "changed while snapshotting" in result[
        "adaptive_observation"
    ]["blocked_reason"]
    assert not fixture["adaptive_spec"].observation_path.exists()


def test_adaptive_observation_missing_provenance_retains_prior(layout):
    fixture = configure_adaptive_observation(layout)
    fixture["service"]().scan_once()
    observation_path = fixture["adaptive_spec"].observation_path
    before = observation_path.read_bytes()
    for directory in ("checkpoints", "consumption"):
        for path in (layout["trainer"] / directory).glob("*.json"):
            path.unlink()

    result = fixture["service"]().scan_once()
    producer_status = load_canonical_json(
        fixture["adaptive_root"] / "observation-producer-status.json"
    )

    assert result["adaptive_observation"]["status"] == "BLOCKED"
    assert producer_status["state"] == "BLOCKED"
    assert result["adaptive_observation"]["observation_retained"] is True
    assert observation_path.read_bytes() == before


def test_adaptive_observation_scan_is_snapshot_idempotent(layout):
    fixture = configure_adaptive_observation(layout)
    first = fixture["service"]().scan_once()
    observation_path = fixture["adaptive_spec"].observation_path
    observation_bytes = observation_path.read_bytes()
    immutable_before = {
        str(path.relative_to(fixture["adaptive_root"])): path.read_bytes()
        for path in (
            fixture["adaptive_root"] / "parents"
        ).rglob("*")
        if path.is_file()
    }

    second = fixture["service"]().scan_once()
    immutable_after = {
        str(path.relative_to(fixture["adaptive_root"])): path.read_bytes()
        for path in (
            fixture["adaptive_root"] / "parents"
        ).rglob("*")
        if path.is_file()
    }

    assert first["adaptive_observation"]["status"] == "PUBLISHED"
    assert second["adaptive_observation"]["status"] == "PUBLISHED"
    assert observation_path.read_bytes() == observation_bytes
    assert immutable_after == immutable_before


def test_adaptive_checkpoint_snapshots_only_advance_when_trial_is_due(layout):
    fixture = configure_adaptive_observation(layout)
    fixture["service"]().scan_once()
    observation_path = fixture["adaptive_spec"].observation_path
    first = load_adaptive_observation(observation_path)
    train_file = fixture["shuffle_path"] / "train" / "data.npz"

    fixture["recorder"].record_consumption(
        [train_file], samples_before=8, samples_after=16
    )
    fixture["checkpoint"].write_bytes(b"adaptive-checkpoint-v2")
    fixture["recorder"].record_checkpoint(
        fixture["checkpoint"], sample_count=16
    )
    fixture["service"]().scan_once()
    second = load_adaptive_observation(observation_path)

    assert second["admitted_samples"] == 16
    assert (
        second["champion_checkpoint"]["path"]
        == first["champion_checkpoint"]["path"]
    )
    assert len(
        list(
            (
                fixture["adaptive_root"] / "parents" / "checkpoints"
            ).glob("*.ckpt")
        )
    ) == 1

    due_samples = 3_000_008
    fixture["recorder"].record_consumption(
        [train_file], samples_before=16, samples_after=due_samples
    )
    fixture["checkpoint"].write_bytes(b"adaptive-checkpoint-due")
    fixture["recorder"].record_checkpoint(
        fixture["checkpoint"], sample_count=due_samples
    )
    fixture["service"]().scan_once()
    due = load_adaptive_observation(observation_path)

    assert due["admitted_samples"] == due_samples
    assert (
        due["champion_checkpoint"]["path"]
        != first["champion_checkpoint"]["path"]
    )
    assert len(
        list(
            (
                fixture["adaptive_root"] / "parents" / "checkpoints"
            ).glob("*.ckpt")
        )
    ) == 2

    later_samples = due_samples + 8
    fixture["recorder"].record_consumption(
        [train_file], samples_before=due_samples, samples_after=later_samples
    )
    fixture["checkpoint"].write_bytes(b"adaptive-checkpoint-after-due")
    fixture["recorder"].record_checkpoint(
        fixture["checkpoint"], sample_count=later_samples
    )
    fixture["service"]().scan_once()
    later = load_adaptive_observation(observation_path)
    assert (
        later["champion_checkpoint"]["path"]
        == due["champion_checkpoint"]["path"]
    )
    assert len(
        list(
            (
                fixture["adaptive_root"] / "parents" / "checkpoints"
            ).glob("*.ckpt")
        )
    ) == 2


def test_adaptive_observation_service_spec_hash_change_is_blocked(layout):
    fixture = configure_adaptive_observation(layout)
    fixture["service"]().scan_once()
    observation_path = fixture["adaptive_spec"].observation_path
    before = observation_path.read_bytes()
    changed = load_canonical_json(fixture["adaptive_spec_path"])
    changed["actor"] = "changed-adaptive-service"
    body = dict(changed)
    body.pop("spec_sha256")
    changed["spec_sha256"] = canonical_sha256(body)
    atomic_replace_json(fixture["adaptive_spec_path"], changed)

    result = fixture["service"]().scan_once()

    assert result["adaptive_observation"]["status"] == "BLOCKED"
    assert "service_inputs hash" in result["adaptive_observation"][
        "blocked_reason"
    ]
    assert result["adaptive_observation"]["observation_retained"] is True
    assert observation_path.read_bytes() == before
