import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import risk_score.promotion_status as promotion_status
from risk_score.cluster_scheduler import ClusterScheduler, WorkKind
from risk_score.position_samples import canonical_sha256, file_sha256
from risk_score.promotion_state import atomic_write_json
from risk_score.promotion_status import StatusError, collect_status


def _utc_timestamp(epoch_seconds):
    return (
        datetime.datetime.fromtimestamp(epoch_seconds, datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_runtime(
    root,
    *,
    min_free_bytes=0,
    gpu_config=None,
    checkpoint=None,
    candidate_inbox=None,
):
    paths = {}
    if gpu_config is not None:
        paths["gpuLeaseConfig"] = str(gpu_config)
    if checkpoint is not None:
        paths["trainerCheckpoint"] = str(checkpoint)
    if candidate_inbox is not None:
        paths["candidateInbox"] = str(candidate_inbox)
    runtime = root / "configs" / "promotion-runtime.json"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        runtime,
        {
            "limits": {"minFreeBytes": min_free_bytes},
            "paths": paths,
        },
    )
    return runtime


def _self_hashed(value, field):
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _write_curation_pipeline(root, *, binding_overrides=None):
    spec_path = root / "configs" / "curation-pipeline.json"
    work_root = root / "evaluation" / "curation" / "pipeline"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    spec = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-curation-pipeline-spec-v1",
            "run_root": str(root),
            "work_root": str(work_root),
        },
        "spec_sha256",
    )
    atomic_write_json(spec_path, spec)
    binding = {
        "path": str(spec_path),
        "sha256": file_sha256(spec_path),
        "identity": spec["spec_sha256"],
    }
    binding.update(binding_overrides or {})
    status = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-curation-pipeline-status-v1",
            "spec": binding,
            "work_root": str(work_root),
            "state": "run_consensus",
            "next_stage": {"kind": "run_consensus", "source": "ordinary"},
            "accepted_counts": None,
            "deficits": {"ordinary": 10},
            "error": None,
        },
        "status_sha256",
    )
    status_path = work_root / "status.json"
    atomic_write_json(status_path, status)
    return spec_path, status_path


def test_status_summarizes_controller_and_detects_stale_supervisor(tmp_path):
    root = tmp_path.resolve()
    promotion = root / "promotion"
    (promotion / "supervisor").mkdir(parents=True)
    (promotion / "operations").mkdir()
    (root / "train" / "network").mkdir(parents=True)
    (promotion / "reports").mkdir()
    atomic_write_json(
        promotion / "status.json",
        {
            "schema_version": 1,
            "contract": "risk-score-controller-status-v1",
            "observed_at_utc": "2026-01-01T00:00:00Z",
            "controller_actor": "test",
            "source_revision_hash": "1" * 64,
            "policy_hash": "2" * 64,
            "result": {
                "mode": "automatic",
                "championHash": "3" * 64,
                "currentGenerationId": "generation-test",
                "queueDepth": 2,
                "activeStage": "screen",
                "activeLook": "automatic",
                "leaseOwner": "evaluator",
                "warnings": [],
            },
        },
    )
    atomic_write_json(
        promotion / "supervisor" / "service.json",
        {
            "schema_version": 1,
            "process_identity": {"pid": 10},
            "updated_at_unix": 900.0,
            "runtime_config": str(root / "configs" / "promotion-runtime.json"),
            "mutation_enabled": True,
        },
    )
    atomic_write_json(
        promotion / "operations" / "backpressure.json",
        {"allowExport": True, "allowEvaluation": True},
    )
    atomic_write_json(
        promotion / "champion.json",
        {"championHash": "3" * 64, "generationId": "generation-test"},
    )
    (root / "selfplay.summary.json").write_text("{}\n", encoding="utf-8")
    (root / "shuffle-input-state.json").write_text("{}\n", encoding="utf-8")
    (root / "train" / "network" / "checkpoint.ckpt").write_bytes(b"checkpoint")

    status = collect_status(root, now=1000.0)
    assert status["controller"]["queue_depth"] == 2
    assert status["controller"]["active_stage"] == "screen"
    assert status["champion"]["generationId"] == "generation-test"
    assert "supervisor-heartbeat-stale" in status["warnings"]
    assert not status["healthy"]


def test_status_rejects_noncanonical_control_file(tmp_path):
    root = tmp_path.resolve()
    (root / "promotion").mkdir()
    (root / "promotion" / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "risk-score-controller-status-v1",
                "result": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(StatusError, match="canonical"):
        collect_status(root)


def test_status_requires_absolute_run_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("relative").mkdir()
    with pytest.raises(StatusError, match="absolute"):
        collect_status(Path("relative"))


def test_status_reports_scheduler_owners_idle_work_and_pipeline_backlogs(tmp_path):
    root = tmp_path.resolve()
    promotion = root / "promotion"
    promotion.mkdir()
    scheduler = ClusterScheduler(
        promotion / "scheduler", ("0", "1"), clock=lambda: 1000.0
    )
    scheduler.enqueue(
        "full-consensus",
        WorkKind.CURATION,
        eligible_gpus=("0", "1"),
        preemptible=True,
    )
    for directory, names in (
        (root / "torchmodels_toexport", ("raw-1", "raw-2", ".partial")),
        (root / "modelstobetested", ("candidate-1",)),
        (root / "models", ("original",)),
    ):
        directory.mkdir()
        for name in names:
            (directory / name).mkdir()

    waiting = collect_status(root, now=1000.0)
    assert waiting["scheduler"]["active_claims"] == 0
    assert waiting["scheduler"]["work_by_state"] == {"queued": 1}
    assert waiting["pipeline"] == {
        "raw_checkpoint_backlog": 2,
        "candidate_inbox_depth": 1,
        "accepted_model_count": 1,
        "reviewed_position_bank_ready": False,
        "v3_suite_ready": False,
    }
    assert "scheduler-runnable-work-unclaimed" in waiting["warnings"]

    claim = scheduler.claim("0", "curation-worker")
    running = collect_status(root, now=1000.0)
    assert running["scheduler"]["active_claims"] == 1
    assert running["scheduler"]["owners"] == {"0": "curation-worker"}
    assert "scheduler-runnable-work-unclaimed" not in running["warnings"]
    scheduler.release(claim)


def test_status_chooses_newest_nested_curation_status_of_either_name(tmp_path):
    root = tmp_path.resolve()
    curation = root / "evaluation" / "curation" / "machine-consensus-v3"
    regular = curation / "older" / "status.json"
    original = curation / "newer" / "nested" / "original-status.json"
    regular.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    atomic_write_json(
        regular,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-machine-consensus-curation-status-v1",
                "state": "older",
            },
            "status_sha256",
        ),
    )
    atomic_write_json(
        original,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-machine-consensus-curation-status-v1",
                "state": "newest",
                "ready_for_labeling": True,
            },
            "status_sha256",
        ),
    )
    os.utime(regular, (800.0, 800.0))
    os.utime(original, (950.0, 950.0))

    status = collect_status(root, now=1000.0)

    assert status["curation"]["path"] == str(original)
    assert status["curation"]["state"] == "newest"
    assert status["artifacts"]["curation"]["age_seconds"] == 50.0


def test_status_prefers_spec_bound_curation_pipeline_status(tmp_path):
    root = tmp_path.resolve()
    _, pipeline_status = _write_curation_pipeline(root)
    legacy = (
        root
        / "evaluation"
        / "curation"
        / "machine-consensus-v3"
        / "newer"
        / "status.json"
    )
    legacy.parent.mkdir(parents=True)
    atomic_write_json(
        legacy,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-machine-consensus-curation-status-v1",
                "state": "complete",
            },
            "status_sha256",
        ),
    )
    os.utime(pipeline_status, (800.0, 800.0))
    os.utime(legacy, (999.0, 999.0))

    status = collect_status(root, now=1000.0)

    assert status["curation"]["path"] == str(pipeline_status)
    assert status["curation"]["source"] == "pipeline"
    assert status["curation"]["state"] == "run_consensus"
    assert status["curation"]["next_stage"] == {
        "kind": "run_consensus",
        "source": "ordinary",
    }


@pytest.mark.parametrize(
    ("binding_key", "bad_value"),
    (
        ("path", "/wrong/curation-pipeline.json"),
        ("sha256", "0" * 64),
        ("identity", "1" * 64),
    ),
)
def test_status_rejects_curation_pipeline_status_binding_contradictions(
    tmp_path, binding_key, bad_value
):
    root = tmp_path.resolve()
    _write_curation_pipeline(root, binding_overrides={binding_key: bad_value})

    with pytest.raises(StatusError, match="contradicts"):
        collect_status(root, now=1000.0)


def test_status_observes_real_selfplay_training_data_not_fresh_summary(tmp_path):
    root = tmp_path.resolve()
    training_data = root / "selfplay" / "model" / "tdata"
    training_data.mkdir(parents=True)
    older = training_data / "older.npz"
    newest = training_data / "newest.npz"
    older.write_bytes(b"old")
    newest.write_bytes(b"new")
    summary = root / "selfplay.summary.json"
    os.utime(older, (500.0, 500.0))
    os.utime(newest, (600.0, 600.0))
    atomic_write_json(
        summary,
        {
            str(training_data): {
                "dir_mtime": 600.0,
                "filename_mtime_num_rowss": [
                    ["older.npz", 500.0, 128],
                    ["newest.npz", 600.0, 256],
                ],
            }
        },
    )
    os.utime(summary, (999.0, 999.0))

    status = collect_status(root, now=1000.0)

    assert status["artifacts"]["selfplay"]["age_seconds"] == 1.0
    assert status["artifacts"]["selfplay_training_data"]["path"] == str(newest)
    assert status["artifacts"]["selfplay_training_data"]["age_seconds"] == 400.0
    assert (
        status["artifacts"]["selfplay_training_data"]["source"]
        == "selfplay.summary.json"
    )
    assert "selfplay-summary-stale" not in status["warnings"]
    assert "selfplay-training-data-stale" in status["warnings"]


def test_status_uses_summary_without_recursive_npz_scan_on_large_tree(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    training_data = root / "selfplay" / "model" / "tdata"
    training_data.mkdir(parents=True)
    newest = training_data / "newest.npz"
    newest.write_bytes(b"new")
    os.utime(newest, (990.0, 990.0))
    for index in range(40):
        nested = root / "selfplay" / f"decoy-{index}" / "deep" / "tree"
        nested.mkdir(parents=True)
        for file_index in range(10):
            (nested / f"{file_index}.npz").write_bytes(b"decoy")
    atomic_write_json(
        root / "selfplay.summary.json",
        {
            str(training_data): {
                "dir_mtime": 990.0,
                "filename_mtime_num_rowss": [["newest.npz", 990.0, 256]],
            }
        },
    )
    original_glob = Path.glob

    def guarded_glob(path, pattern):
        if pattern == "**/*.npz":
            raise AssertionError("recursive NPZ scan is forbidden")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)

    status = collect_status(root, now=1000.0)

    assert status["artifacts"]["selfplay_training_data"]["path"] == str(newest)
    assert (
        status["artifacts"]["selfplay_training_data"]["source"]
        == "selfplay.summary.json"
    )


def test_status_uses_generation_watermark_without_tree_scan(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    tdata = root / "selfplay" / "generation-1" / "tdata"
    tdata.mkdir(parents=True)
    data = tdata / "data.npz"
    data.write_bytes(b"watermarked")
    os.utime(data, ns=(990_000_000_000, 990_000_000_000))
    record = {
        "path": "data.npz",
        "mtime_ns": data.stat().st_mtime_ns,
        "size": data.stat().st_size,
    }
    inventory = [record]
    watermark = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-generation-data-watermark-v1",
            "generations": [
                {
                    "generation_id": "generation-1",
                    "roots": [
                        {
                            "path": str(tdata),
                            "inventory": inventory,
                            "inventory_sha256": canonical_sha256(inventory),
                        }
                    ],
                }
            ],
            "historical_sources": [],
        },
        "watermark_sha256",
    )
    watermark_path = root / "promotion" / "watermarks" / "data.json"
    watermark_path.parent.mkdir(parents=True)
    atomic_write_json(watermark_path, watermark)
    original_glob = Path.glob

    def guarded_glob(path, pattern):
        if pattern == "**/*.npz":
            raise AssertionError("recursive NPZ scan is forbidden")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)

    status = collect_status(root, now=1000.0)

    assert status["artifacts"]["selfplay_training_data"]["path"] == str(data)
    assert (
        status["artifacts"]["selfplay_training_data"]["source"]
        == "generation-data-watermark"
    )


def test_status_reports_trainer_decision_duration_and_abnormal_exit(tmp_path):
    root = tmp_path.resolve()
    supervisor = root / "promotion" / "supervisor"
    supervisor.mkdir(parents=True)
    atomic_write_json(
        supervisor / "service.json",
        {
            "schema_version": 1,
            "process_identity": {"pid": 10},
            "updated_at_unix": 999.0,
            "mutation_enabled": True,
        },
    )
    observation = {
        "schema_version": 1,
        "contract": "risk-score-host-trainer-observation-v1",
        "role": "trainer",
        "observation": "short-clean-bucket-limited-exit",
        "decision": "restart-backoff",
        "decision_since_unix": 900.0,
        "updated_at_unix": 999.0,
        "restart_not_before_unix": 1030.0,
        "consecutive_short_clean_exits": 2,
    }
    atomic_write_json(supervisor / "trainer-observation.json", observation)

    status = collect_status(root, now=1000.0)

    assert status["trainer"]["observation"]["decision"] == "restart-backoff"
    assert status["trainer"]["observation"]["decision_duration_seconds"] == 100.0
    assert "trainer-observation-missing" not in status["warnings"]

    observation.update(
        {
            "observation": "completed-trainer",
            "decision": "abnormal-exit",
            "decision_since_unix": 1000.0,
            "updated_at_unix": 1000.0,
        }
    )
    atomic_write_json(supervisor / "trainer-observation.json", observation)
    failed = collect_status(root, now=1000.0)
    assert "trainer-abnormal-exit" in failed["warnings"]


@pytest.mark.parametrize("allow_export", (True, False))
def test_status_warns_on_stale_backpressure_timestamp_even_for_denial(
    tmp_path, allow_export
):
    root = tmp_path.resolve()
    path = root / "promotion" / "operations" / "backpressure.json"
    path.parent.mkdir(parents=True)
    atomic_write_json(
        path,
        {
            "allowEvaluation": allow_export,
            "allowExport": allow_export,
            "updated_at_utc": _utc_timestamp(879.0),
        },
    )
    os.utime(path, (999.0, 999.0))

    status = collect_status(root, now=1000.0)

    assert status["backpressure_freshness"] == {
        "freshness_source": "updated_at_utc",
        "freshness_age_seconds": 121.0,
        "maximum_age_seconds": 120.0,
        "future_dated": False,
        "stale": True,
    }
    assert "backpressure-status-stale" in status["warnings"]


def test_status_uses_backpressure_mtime_with_120_second_boundary(tmp_path):
    root = tmp_path.resolve()
    path = root / "promotion" / "operations" / "backpressure.json"
    path.parent.mkdir(parents=True)
    atomic_write_json(
        path,
        {"allowEvaluation": True, "allowExport": True},
    )
    os.utime(path, (880.0, 880.0))

    fresh = collect_status(root, now=1000.0)
    assert fresh["backpressure_freshness"]["freshness_source"] == "mtime"
    assert fresh["backpressure_freshness"]["stale"] is False
    assert "backpressure-status-stale" not in fresh["warnings"]

    os.utime(path, (879.0, 879.0))
    stale = collect_status(root, now=1000.0)
    assert stale["backpressure_freshness"]["stale"] is True
    assert "backpressure-status-stale" in stale["warnings"]


def test_status_keeps_corroborated_long_active_evaluation_lease_fresh(tmp_path):
    root = tmp_path.resolve()
    promotion = root / "promotion"
    promotion.mkdir()
    atomic_write_json(
        promotion / "status.json",
        {
            "schema_version": 1,
            "contract": "risk-score-controller-status-v1",
            "observed_at_utc": _utc_timestamp(9999.0),
            "result": {
                "activeEvaluations": [
                    {
                        "stage": "confirmation",
                        "startedAtUtc": _utc_timestamp(100.0),
                    }
                ],
                "activeStage": "confirmation",
                "leaseOwner": "controller-test",
                "warnings": [],
            },
        },
    )
    atomic_write_json(
        promotion / "gpu-lease.json",
        {
            "schemaVersion": 1,
            "leaseId": "lease-test",
            "ownerId": "controller-test",
            "phase": "evaluating",
            "safetyHalt": False,
            "safetyReason": None,
            "updatedAt": 100.0,
        },
    )

    status = collect_status(root, now=10000.0)

    assert status["gpu_lease"]["age_seconds"] == 9900.0
    assert status["gpu_lease"]["active_evaluation"]["corroborated"] is True
    assert status["gpu_lease"]["active_evaluation"]["stage"] == "confirmation"
    assert status["gpu_lease"]["stale_after_seconds"] == 21600.0
    assert status["gpu_lease"]["stale_non_trainer_phase"] is False
    assert "gpu-lease-non-trainer-phase-stale" not in status["warnings"]

    controller_path = promotion / "status.json"
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["result"]["activeEvaluations"][0]["startedAtUtc"] = _utc_timestamp(
        -12000.0
    )
    atomic_write_json(controller_path, controller)
    overdue = collect_status(root, now=10000.0)
    assert overdue["gpu_lease"]["active_evaluation"]["deadline_exceeded"] is True
    assert overdue["gpu_lease"]["stale_non_trainer_phase"] is True
    assert "gpu-lease-non-trainer-phase-stale" in overdue["warnings"]


def test_status_warns_on_stale_orphan_non_trainer_gpu_lease(tmp_path):
    root = tmp_path.resolve()
    gpu_config = root / "runtime" / "gpu.json"
    lease = root / "runtime" / "state" / "lease.json"
    gpu_config.parent.mkdir(parents=True)
    lease.parent.mkdir(parents=True)
    atomic_write_json(gpu_config, {"paths": {"leaseState": str(lease)}})
    _write_runtime(root, gpu_config=gpu_config)
    atomic_write_json(
        lease,
        {
            "schemaVersion": 1,
            "leaseId": "lease-test",
            "ownerId": "controller-test",
            "phase": "evaluating",
            "safetyHalt": False,
            "safetyReason": None,
            "updatedAt": 879.0,
        },
    )

    status = collect_status(root, now=1000.0)

    assert status["gpu_lease"]["path"] == str(lease)
    assert status["gpu_lease"]["phase"] == "evaluating"
    assert status["gpu_lease"]["age_seconds"] == 121.0
    assert status["gpu_lease"]["active_evaluation"] == {
        "corroborated": False,
        "reason": "controller-status-missing",
    }
    assert status["gpu_lease"]["stale_non_trainer_phase"] is True
    assert "gpu-lease-non-trainer-phase-stale" in status["warnings"]


def test_status_reports_gpu_safety_halt_and_rejects_noncanonical_state(tmp_path):
    root = tmp_path.resolve()
    lease = root / "promotion" / "gpu-lease.json"
    lease.parent.mkdir(parents=True)
    value = {
        "schemaVersion": 1,
        "leaseId": "lease-test",
        "ownerId": "controller-test",
        "phase": "safety_halt",
        "safetyHalt": True,
        "safetyReason": "trainer restore failed",
        "updatedAt": 999.0,
    }
    atomic_write_json(lease, value)

    status = collect_status(root, now=1000.0)
    assert status["gpu_lease"]["safety_reason"] == "trainer restore failed"
    assert "gpu-lease-safety-halt" in status["warnings"]

    lease.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(StatusError, match="canonical"):
        collect_status(root, now=1000.0)


def test_status_reports_disk_checkpoint_and_raw_candidate_backlogs(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    checkpoint = root / "trainer-state" / "active.snapshot"
    candidate_inbox = root / "custom-candidate-inbox"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    os.utime(checkpoint, (900.0, 900.0))
    candidate_inbox.mkdir()
    (candidate_inbox / "candidate-1").mkdir()
    raw = root / "torchmodels_toexport"
    raw.mkdir()
    (raw / "raw-1").mkdir()
    (raw / "raw-2").mkdir()
    _write_runtime(
        root,
        min_free_bytes=500,
        checkpoint=checkpoint,
        candidate_inbox=candidate_inbox,
    )
    monkeypatch.setattr(
        promotion_status.shutil,
        "disk_usage",
        lambda unused: SimpleNamespace(free=499),
    )

    status = collect_status(root, now=1000.0)

    assert status["disk"]["free_bytes"] == 499
    assert status["disk"]["runtime_min_free_bytes"] == 500
    assert status["disk"]["below_runtime_minimum"] is True
    assert "disk-free-below-runtime-minimum" in status["warnings"]
    assert status["trainer"] == {
        "checkpoint_present": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_age_seconds": 100.0,
        "observation": {},
    }
    assert status["backlogs"] == {
        "raw_checkpoint_depth": 2,
        "candidate_inbox_depth": 1,
    }
