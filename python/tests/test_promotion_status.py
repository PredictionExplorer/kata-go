import json
from pathlib import Path

import pytest

from risk_score.cluster_scheduler import ClusterScheduler, WorkKind
from risk_score.promotion_state import atomic_write_json
from risk_score.promotion_status import StatusError, collect_status


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
