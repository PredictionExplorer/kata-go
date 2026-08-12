import datetime
import hashlib
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


def _rewrite_self_hashed(path, field, updates):
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(updates)
    value.pop(field, None)
    value[field] = canonical_sha256(value)
    atomic_write_json(path, value)
    return value


def _write_full_autonomy_status_tree(root, *, now=1000.0):
    promotion = root / "promotion"
    runtime_root = promotion / "autonomy-bootstrap" / "runtime"
    bootstrap_root = promotion / "autonomy-bootstrap"
    executor_root = promotion / "cluster-executor"
    adaptive_root = promotion / "adaptive"
    registry_root = promotion / "suite-registry"
    runtime_root.mkdir(parents=True)
    (bootstrap_root / "receipts").mkdir()
    (executor_root / "heartbeats").mkdir(parents=True)
    adaptive_root.mkdir()
    checkpoint = root / "train" / "network" / "checkpoint.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    os.utime(checkpoint, (now - 1, now - 1))

    champion_hash = "3" * 64
    generation_id = "generation-full-autonomy"
    (promotion / "supervisor").mkdir(parents=True)
    (promotion / "operations").mkdir()
    atomic_write_json(
        promotion / "status.json",
        {
            "schema_version": 1,
            "contract": "risk-score-controller-status-v1",
            "observed_at_utc": _utc_timestamp(now - 1),
            "result": {
                "mode": "automatic",
                "championHash": champion_hash,
                "currentGenerationId": generation_id,
                "queueDepth": 0,
                "warnings": [],
            },
        },
    )
    os.utime(promotion / "status.json", (now - 1, now - 1))
    atomic_write_json(
        promotion / "champion.json",
        {
            "championHash": champion_hash,
            "generationId": generation_id,
        },
    )
    atomic_write_json(
        promotion / "supervisor" / "trainer-observation.json",
        {
            "schema_version": 1,
            "contract": "risk-score-host-trainer-observation-v1",
            "role": "trainer",
            "observation": "running",
            "decision": "continue",
            "decision_since_unix": now - 10,
            "updated_at_unix": now - 1,
            "restart_not_before_unix": None,
            "consecutive_short_clean_exits": 0,
        },
    )
    atomic_write_json(
        promotion / "operations" / "backpressure.json",
        {
            "allowEvaluation": True,
            "allowExport": True,
            "updated_at_utc": _utc_timestamp(now - 1),
        },
    )
    atomic_write_json(
        promotion / "gpu-lease.json",
        {
            "schemaVersion": 1,
            "leaseId": None,
            "ownerId": "trainer",
            "phase": "trainer_running",
            "safetyHalt": False,
            "safetyReason": None,
            "updatedAt": now - 1,
        },
    )
    shuffle = root / "shuffle-input-state.json"
    atomic_write_json(shuffle, {})
    os.utime(shuffle, (now - 1, now - 1))
    training_data = root / "selfplay" / "continuous" / "tdata"
    training_data.mkdir(parents=True)
    npz = training_data / "fresh.npz"
    npz.write_bytes(b"training-data")
    os.utime(npz, (now - 1, now - 1))
    summary_path = root / "selfplay.summary.json"
    atomic_write_json(
        summary_path,
        {
            str(training_data): {
                "filename_mtime_num_rowss": [["fresh.npz", now - 1, 128]]
            }
        },
    )
    os.utime(summary_path, (now - 1, now - 1))

    ClusterScheduler(promotion / "scheduler", ("0",), clock=lambda: now - 1)
    scheduler_value = json.loads(
        (promotion / "scheduler" / "state.json").read_text(encoding="utf-8")
    )

    executor_spec_path = root / "configs" / "cluster-executor.json"
    executor_spec_path.parent.mkdir(parents=True)
    executor_spec = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-cluster-executor-spec-v1",
            "scheduler_directory": str(promotion / "scheduler"),
            "state_directory": str(executor_root),
            "owner_id": "executor-a",
            "gpu_ids": ["0"],
            "gpu7_id": None,
            "poll_interval_seconds": 1.0,
            "heartbeat_interval_seconds": 2.0,
            "stale_after_seconds": 30.0,
            "retry_budget": 2,
            "backoff_initial_seconds": 5.0,
            "backoff_max_seconds": 60.0,
            "lease_proof_command": None,
            "lease_proof_timeout_seconds": 10.0,
        },
        "spec_sha256",
    )
    atomic_write_json(executor_spec_path, executor_spec)
    heartbeat_path = (
        executor_root
        / "heartbeats"
        / (hashlib.sha256(b"executor-a").hexdigest() + ".json")
    )
    atomic_write_json(
        heartbeat_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-cluster-heartbeat-v1",
                "owner_id": "executor-a",
                "executor_spec_sha256": executor_spec["spec_sha256"],
                "state": "healthy",
                "updated_at_unix": now - 1,
            },
            "state_sha256",
        ),
    )
    executor_status_path = executor_root / "status.json"
    atomic_write_json(
        executor_status_path,
        {
            "schema_version": 1,
            "contract": "risk-score-cluster-executor-status-v1",
            "owner_id": "executor-a",
            "executor_spec_sha256": executor_spec["spec_sha256"],
            "observed_at_unix": now - 1,
            "scheduler_revision": scheduler_value["revision"],
            "scheduler_state_sha256": scheduler_value["state_sha256"],
            "safety_halt": None,
            "gpu_safety_halts": {},
            "gpus": [
                {
                    "gpu_id": "0",
                    "state": "idle",
                    "claim": None,
                    "queued_work": [],
                }
            ],
            "quarantines": [],
        },
    )

    policy_path = root / "configs" / "autonomy-policy.json"
    policy_value = {
        "schema_version": 1,
        "gpu_budget": {
            "host_gpu_count": 8,
            "maximum_fraction": 0.1,
            "rolling_window_seconds": 604800,
        },
        "queue": {"maximum_candidate_queue_depth": 3},
        "successive_halving": {"round_gpu_seconds": [14400, 28800, 57600]},
        "trials": {"maximum_active": 1},
        "trigger": {"minimum_admitted_samples_without_promotion": 3000000},
    }
    atomic_write_json(policy_path, policy_value)
    adaptive_spec_path = root / "configs" / "adaptive-training.json"
    atomic_write_json(
        adaptive_spec_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": ("risk-score-adaptive-training-service-spec-v1"),
                "root": str(adaptive_root),
                "autonomy_policy_path": str(policy_path),
                "autonomy_policy_sha256": canonical_sha256(policy_value),
                "scheduler_directory": str(promotion / "scheduler"),
                "gpu7_id": "7",
                "observation_path": str(adaptive_root / "observation.json"),
                "trial_command_argv_template": [
                    "/bin/true",
                    "{trial_manifest_path}",
                    "{trial_result_path}",
                ],
                "gpu_lease_guardian_argv_prefix": ["/bin/true"],
                "poll_interval_seconds": 5.0,
                "actor": "adaptive-test",
            },
            "spec_sha256",
        ),
    )
    adaptive_status_path = adaptive_root / "status.json"
    atomic_write_json(
        adaptive_status_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-adaptive-training-status-v1",
                "active_epoch_id": None,
                "active_trial_id": None,
                "epochs": {},
                "gpu_usage": [],
                "last_epoch_admitted_samples": None,
                "last_event_hash": "0" * 64,
                "last_sequence": 0,
                "policy_hash": canonical_sha256(policy_value),
                "trials": {},
            },
            "status_sha256",
        ),
    )
    os.utime(adaptive_status_path, (now - 1, now - 1))
    adaptive_status_value = json.loads(adaptive_status_path.read_text(encoding="utf-8"))
    admitted_manifest = adaptive_root / "admitted.json"
    atomic_write_json(admitted_manifest, {"samples": 3000000})
    adaptive_observation = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-adaptive-training-observation-v1",
            "admitted_data_manifest": {
                "path": str(admitted_manifest),
                "sha256": file_sha256(admitted_manifest),
            },
            "admitted_samples": 3000000,
            "candidate_queue_depth": 0,
            "champion_checkpoint": {
                "path": str(checkpoint),
                "resumable": True,
                "sha256": file_sha256(checkpoint),
            },
            "current_champion_model_sha256": champion_hash,
            "last_promotion_admitted_samples": 0,
            "updated_at_unix": now - 1,
        },
        "observation_sha256",
    )
    adaptive_service_status_path = adaptive_root / "service-status.json"
    adaptive_spec_value = json.loads(adaptive_spec_path.read_text(encoding="utf-8"))
    atomic_write_json(
        adaptive_service_status_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": ("risk-score-adaptive-training-service-status-v1"),
                "actions": [],
                "active_trial_id": None,
                "active_work_id": None,
                "actor": "adaptive-test",
                "adaptive_status": adaptive_status_value,
                "blocked_reason": None,
                "error": None,
                "observation": adaptive_observation,
                "observed_at_unix": now - 1,
                "scheduler": {
                    "dynamic_gpus": False,
                    "gpu_ids": ["0"],
                    "revision": scheduler_value["revision"],
                    "service_work": [],
                    "state_sha256": scheduler_value["state_sha256"],
                },
                "service_spec_sha256": adaptive_spec_value["spec_sha256"],
            },
            "status_sha256",
        ),
    )
    active_recipe_path = adaptive_root / "active-recipe.json"
    atomic_write_json(
        active_recipe_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-active-training-recipe-v1",
                "activated_at_utc": _utc_timestamp(now - 1),
                "admitted_data_manifest_sha256": "4" * 64,
                "champion_checkpoint_sha256": "5" * 64,
                "champion_model_sha256": champion_hash,
                "data_watermark_sha256s": {"data": "6" * 64},
                "generation_id": generation_id,
                "previous_record_sha256": None,
                "recipe_path": "/recipes/baseline.json",
                "recipe_sha256": "7" * 64,
                "rollback": None,
            },
            "record_sha256",
        ),
    )

    suite_spec_path = root / "configs" / "suite-registry.json"
    suite_spec = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-evaluation-suite-registry-spec-v1",
            "registry_root": str(registry_root),
        },
        "spec_sha256",
    )
    atomic_write_json(suite_spec_path, suite_spec)
    suite_version = "9" * 64
    suite_root = registry_root / "suites" / "initial"
    suite_root.mkdir(parents=True)
    suite_manifest = suite_root / "manifest.json"
    atomic_write_json(suite_manifest, {"suite": "initial"})
    suite_id = file_sha256(suite_manifest)
    active_suite_path = registry_root / "active-suite.json"
    atomic_write_json(
        active_suite_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-active-evaluation-suite-v1",
                "spec_sha256": suite_spec["spec_sha256"],
                "suite_id": suite_id,
                "version_sha256": suite_version,
                "manifest_path": str(suite_manifest),
                "manifest_sha256": file_sha256(suite_manifest),
                "manifest_identity": "a" * 64,
                "activated_at_utc": _utc_timestamp(now - 10),
                "activation_champion_sha256": champion_hash,
                "activation_generation_id": generation_id,
                "event_sequence": 1,
                "event_sha256": "b" * 64,
            },
            "record_sha256",
        ),
    )
    suite_status_path = registry_root / "status.json"
    atomic_write_json(
        suite_status_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": ("risk-score-evaluation-suite-rotation-status-v1"),
                "generated_at_utc": _utc_timestamp(now - 1),
                "spec": {
                    "path": str(suite_spec_path),
                    "sha256": file_sha256(suite_spec_path),
                    "identity": suite_spec["spec_sha256"],
                },
                "state": "active",
                "next_action": "wait-for-cadence",
                "active_suite": {
                    "suite_id": suite_id,
                    "version_sha256": suite_version,
                    "activated_at_utc": _utc_timestamp(now - 10),
                },
                "current_champion": {
                    "sha256": champion_hash,
                    "generation_id": generation_id,
                    "previous_sha256": "c" * 64,
                },
                "cadence": {
                    "eligible": False,
                    "reason_codes": [],
                    "accepted_champion_count": 0,
                },
                "current_request_id": None,
                "candidate_suite_id": None,
                "in_flight_evaluations": [],
                "retained_suites": [
                    {
                        "suite_id": suite_id,
                        "version_sha256": suite_version,
                        "active": True,
                        "immutable": True,
                        "retained": True,
                    }
                ],
                "active_projection_consistent": True,
                "last_event_sequence": 1,
                "last_event_sha256": "b" * 64,
            },
            "status_sha256",
        ),
    )

    runtime_path = runtime_root / "promotion-runtime.json"
    atomic_write_json(
        runtime_path,
        {
            "limits": {"minFreeBytes": 0},
            "paths": {
                "candidateInbox": str(root / "modelstobetested"),
                "trainerCheckpoint": str(checkpoint),
                "suites": str(suite_root),
            },
            "hashes": {
                "suiteManifest": file_sha256(suite_manifest),
            },
        },
    )
    atomic_write_json(
        promotion / "supervisor" / "service.json",
        {
            "schema_version": 1,
            "process_identity": {"pid": 10},
            "updated_at_unix": now - 1,
            "runtime_config": str(runtime_path),
            "mutation_enabled": True,
        },
    )

    service_inputs = {
        "autonomy_policy": policy_path,
        "executor_spec": executor_spec_path,
        "adaptive_spec": adaptive_spec_path,
        "suite_registry_spec": suite_spec_path,
    }
    service_path = runtime_root / "promotion-services.json"
    atomic_write_json(
        service_path,
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
                for name, path in sorted(service_inputs.items())
            },
        },
    )
    activation_path = bootstrap_root / "activation.json"
    installed_target = root / "systemd" / "katago-risk-training.target"
    installed_target.parent.mkdir()
    installed_target.write_text("[Unit]\n", encoding="utf-8")
    atomic_write_json(
        activation_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-systemd-activation-receipt-v1",
                "service_spec_sha256": file_sha256(service_path),
                "target_unit": "katago-risk-training.target",
                "unit_inventory": ["katago-risk-training.target"],
                "removed_units": {},
                "restart_occurred": False,
                "installed_units": {
                    "katago-risk-training.target": {
                        "path": str(installed_target),
                        "sha256": file_sha256(installed_target),
                    }
                },
                "active": {
                    "katago-risk-training.target": "active",
                },
            },
            "receipt_sha256",
        ),
    )
    bootstrap_spec_path = root / "configs" / "autonomy-bootstrap.json"
    bootstrap_spec = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-autonomy-bootstrap-spec-v1",
            "state_root": str(bootstrap_root),
            "runtime": {"output_dir": str(runtime_root)},
            "activation": {"receipt": str(activation_path)},
        },
        "spec_sha256",
    )
    atomic_write_json(bootstrap_spec_path, bootstrap_spec)
    runtime_receipt_path = bootstrap_root / "receipts" / "runtime.json"
    runtime_receipt = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-autonomy-runtime-receipt-v1",
            "decision": "PASS",
            "result": {
                "promotion_runtime": str(runtime_path),
                "promotion_runtime_sha256": file_sha256(runtime_path),
                "service_spec": str(service_path),
                "service_spec_sha256": file_sha256(service_path),
                "mutation_enabled": True,
                "full_autonomy": True,
            },
        },
        "receipt_sha256",
    )
    atomic_write_json(runtime_receipt_path, runtime_receipt)
    activation_verification_path = (
        bootstrap_root / "receipts" / "activation-verification.json"
    )
    atomic_write_json(
        activation_verification_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": ("risk-score-autonomy-activation-verification-v1"),
                "decision": "PASS",
                "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
                "activation_receipt": {
                    "path": str(activation_path),
                    "sha256": file_sha256(activation_path),
                    "identity": json.loads(activation_path.read_text(encoding="utf-8"))[
                        "receipt_sha256"
                    ],
                },
                "active": {
                    "katago-risk-training.target": "active",
                },
            },
            "receipt_sha256",
        ),
    )
    bootstrap_status_path = bootstrap_root / "status.json"
    atomic_write_json(
        bootstrap_status_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-autonomy-bootstrap-status-v1",
                "spec": {
                    "path": str(bootstrap_spec_path),
                    "file_sha256": file_sha256(bootstrap_spec_path),
                    "identity": bootstrap_spec["spec_sha256"],
                },
                "state": "active",
                "completed_gates": ["gate-a", "gate-b"],
                "total_gates": 2,
                "next_gate": None,
                "waiting_gate": None,
                "selected_evaluator_processes": 8,
                "runtime_ready": True,
                "activation_verified": True,
                "safety_halt": None,
            },
            "status_sha256",
        ),
    )
    return {
        "activation_receipt": activation_path,
        "activation_verification": activation_verification_path,
        "active_recipe": active_recipe_path,
        "active_suite": active_suite_path,
        "adaptive_service_status": adaptive_service_status_path,
        "adaptive_status": adaptive_status_path,
        "bootstrap_status": bootstrap_status_path,
        "executor_heartbeat": heartbeat_path,
        "executor_status": executor_status_path,
        "runtime": runtime_path,
        "runtime_receipt": runtime_receipt_path,
        "service_spec": service_path,
        "suite_manifest": suite_manifest,
        "suite_id": suite_id,
        "suite_status": suite_status_path,
    }


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


def test_status_reports_healthy_full_autonomy_state(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)

    status = collect_status(root, now=1000.0)

    assert status["healthy"] is True
    assert status["warnings"] == []
    autonomy = status["autonomy"]
    assert autonomy["full_autonomy"] is True
    assert autonomy["bootstrap"]["state"] == "active"
    assert autonomy["bootstrap"]["completed_gate_count"] == 2
    assert autonomy["runtime"]["activation_verified"] is True
    assert autonomy["executor"]["heartbeat"]["age_seconds"] == 1.0
    assert autonomy["executor"]["active_claims"] == 0
    assert autonomy["executor"]["quarantine_count"] == 0
    assert autonomy["adaptive"]["active_epoch_id"] is None
    assert autonomy["adaptive"]["gpu_budget"]["allowed"] is True
    assert autonomy["adaptive"]["trigger"]["eligible"] is True
    assert autonomy["adaptive"]["active_recipe"]["champion_matches"] is True
    assert autonomy["suite_rotation"]["desired_suite_id"] == paths["suite_id"]
    assert autonomy["suite_rotation"]["pin_count"] == 0
    assert autonomy["suite_rotation"]["runtime_divergence"] is False
    assert autonomy["terminal"]["halted"] is False
    assert autonomy["terminal_remediation_reason"] is None


def test_status_does_not_require_autonomy_artifacts_without_v3_runtime(tmp_path):
    root = tmp_path.resolve()
    adaptive = root / "promotion" / "adaptive"
    adaptive.mkdir(parents=True)
    status_path = adaptive / "status.json"
    atomic_write_json(
        status_path,
        _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-adaptive-training-status-v1",
                "active_epoch_id": None,
                "active_trial_id": None,
                "epochs": {},
                "gpu_usage": [],
                "last_epoch_admitted_samples": None,
                "last_event_hash": "0" * 64,
                "last_sequence": 0,
                "policy_hash": "1" * 64,
                "trials": {},
            },
            "status_sha256",
        ),
    )
    os.utime(status_path, (1.0, 1.0))

    status = collect_status(root, now=1000.0)

    assert status["autonomy"]["full_autonomy"] is False
    assert not any(
        warning.startswith(
            (
                "autonomy-bootstrap-",
                "cluster-executor-",
                "adaptive-training-",
                "suite-rotation-",
            )
        )
        for warning in status["warnings"]
    )


def test_status_rejects_changed_hash_bound_autonomy_service_input(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    service = json.loads(paths["service_spec"].read_text(encoding="utf-8"))
    adaptive_spec = Path(service["service_inputs"]["adaptive_spec"]["path"])
    value = json.loads(adaptive_spec.read_text(encoding="utf-8"))
    value["actor"] = "changed-after-runtime-build"
    atomic_write_json(adaptive_spec, value)

    with pytest.raises(StatusError, match="binding changed"):
        collect_status(root, now=1000.0)


def test_status_reports_executor_claims_and_quarantine_receipts(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    scheduler = ClusterScheduler(
        root / "promotion" / "scheduler",
        ("0",),
        clock=lambda: 999.0,
    )
    scheduler.enqueue(
        "work-1",
        WorkKind.BACKFILL,
        eligible_gpus=("0",),
        preemptible=True,
    )
    claim = scheduler.claim("0", "executor-a")
    assert claim is not None
    quarantine = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-cluster-quarantine-v1",
            "work_id": "failed-work",
            "work_spec_sha256": "a" * 64,
            "claim_id": "claim-failed",
            "failure_count": 3,
            "retry_budget": 2,
            "reason": "retry-budget-exhausted",
            "quarantined_at_unix": 998.0,
        },
        "receipt_sha256",
    )
    quarantine_root = root / "promotion" / "cluster-executor" / "quarantine"
    quarantine_root.mkdir()
    quarantine_path = quarantine_root / "failed-work.json"
    atomic_write_json(quarantine_path, quarantine)
    scheduler_value = json.loads(
        (root / "promotion" / "scheduler" / "state.json").read_text(encoding="utf-8")
    )
    executor_status = json.loads(paths["executor_status"].read_text(encoding="utf-8"))
    executor_status.update(
        {
            "scheduler_revision": scheduler_value["revision"],
            "scheduler_state_sha256": scheduler_value["state_sha256"],
            "gpus": [
                {
                    "gpu_id": "0",
                    "state": "running",
                    "claim": claim.to_dict(),
                    "owner_stale": False,
                }
            ],
            "quarantines": [quarantine],
        }
    )
    atomic_write_json(paths["executor_status"], executor_status)

    status = collect_status(root, now=1000.0)

    executor = status["autonomy"]["executor"]
    assert executor["active_claims"] == 1
    assert executor["claims"][0]["work_id"] == "work-1"
    assert executor["quarantine_count"] == 1
    assert executor["quarantines"][0]["path"] == str(quarantine_path)
    assert "scheduler-executor-disagreement" not in status["warnings"]


def test_status_warns_when_active_recipe_champion_binding_diverges(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    _rewrite_self_hashed(
        paths["active_recipe"],
        "record_sha256",
        {"champion_model_sha256": "e" * 64},
    )

    status = collect_status(root, now=1000.0)

    assert "adaptive-recipe-champion-mismatch" in status["warnings"]
    assert status["autonomy"]["adaptive"]["active_recipe"]["champion_matches"] is False


def test_status_warns_when_active_suite_differs_from_frozen_runtime(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    replacement = root / "replacement-suite" / "manifest.json"
    replacement.parent.mkdir()
    atomic_write_json(replacement, {"suite": "replacement"})
    _rewrite_self_hashed(
        paths["active_suite"],
        "record_sha256",
        {
            "suite_id": file_sha256(replacement),
            "manifest_path": str(replacement),
            "manifest_sha256": file_sha256(replacement),
        },
    )

    status = collect_status(root, now=1000.0)

    assert "suite-active-pointer-runtime-divergence" in status["warnings"]
    assert status["autonomy"]["suite_rotation"]["runtime_divergence"] is True


def test_status_warns_on_scheduler_executor_snapshot_disagreement(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    value = json.loads(paths["executor_status"].read_text(encoding="utf-8"))
    value["scheduler_state_sha256"] = "f" * 64
    atomic_write_json(paths["executor_status"], value)

    status = collect_status(root, now=1000.0)

    assert "scheduler-executor-disagreement" in status["warnings"]
    assert status["autonomy"]["executor"]["scheduler_disagreement"] is True


def test_status_derives_executor_snapshot_when_no_projection_file(tmp_path):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    paths["executor_status"].unlink()

    status = collect_status(root, now=1000.0)

    assert "cluster-executor-status-missing" not in status["warnings"]
    assert status["autonomy"]["executor"]["status_source"] == "durable-state"
    assert status["autonomy"]["executor"]["active_claims"] == 0


@pytest.mark.parametrize(
    ("artifact", "warning"),
    (
        ("bootstrap_status", "autonomy-bootstrap-status-missing"),
        ("executor_heartbeat", "cluster-executor-heartbeat-missing"),
        ("adaptive_status", "adaptive-training-status-missing"),
        ("suite_status", "suite-rotation-status-missing"),
    ),
)
def test_status_warns_on_missing_required_full_autonomy_status(
    tmp_path, artifact, warning
):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    paths[artifact].unlink()

    status = collect_status(root, now=1000.0)

    assert warning in status["warnings"]


@pytest.mark.parametrize(
    ("artifact", "hash_field", "updates", "warning"),
    (
        (
            "bootstrap_status",
            "status_sha256",
            {"state": "waiting", "waiting_gate": "gate-c"},
            "autonomy-bootstrap-status-stale",
        ),
        (
            "executor_heartbeat",
            "state_sha256",
            {"updated_at_unix": 900.0},
            "cluster-executor-heartbeat-stale",
        ),
        (
            "executor_status",
            None,
            {"observed_at_unix": 900.0},
            "cluster-executor-status-stale",
        ),
        (
            "adaptive_service_status",
            "status_sha256",
            {"observed_at_unix": 900.0},
            "adaptive-training-status-stale",
        ),
        (
            "suite_status",
            "status_sha256",
            {"generated_at_utc": _utc_timestamp(900.0)},
            "suite-rotation-status-stale",
        ),
    ),
)
def test_status_warns_on_stale_required_full_autonomy_status(
    tmp_path, artifact, hash_field, updates, warning
):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    path = paths[artifact]
    if hash_field is None:
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(updates)
        atomic_write_json(path, value)
    else:
        _rewrite_self_hashed(path, hash_field, updates)
    if artifact == "bootstrap_status":
        os.utime(path, (900.0, 900.0))

    status = collect_status(root, now=1000.0)

    assert warning in status["warnings"]


@pytest.mark.parametrize("source", ("bootstrap", "executor"))
def test_status_reports_terminal_halt_and_remediation_reason(tmp_path, source):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    reason = f"{source} requires operator remediation"
    if source == "bootstrap":
        status_value = json.loads(paths["bootstrap_status"].read_text(encoding="utf-8"))
        halt = _self_hashed(
            {
                "schema_version": 1,
                "contract": "risk-score-autonomy-safety-halt-v1",
                "state": "safety-halt",
                "spec": status_value["spec"],
                "failed_stage": "activation",
                "completed_gates": status_value["completed_gates"],
                "activation_invoked": True,
                "error": {
                    "type": "BootstrapSafetyHalt",
                    "message": reason,
                },
            },
            "halt_sha256",
        )
        _rewrite_self_hashed(
            paths["bootstrap_status"],
            "status_sha256",
            {"state": "safety-halt", "safety_halt": halt},
        )
    else:
        halt_path = (
            root
            / "promotion"
            / "cluster-executor"
            / "halts"
            / (hashlib.sha256(b"0").hexdigest() + ".json")
        )
        halt_path.parent.mkdir()
        atomic_write_json(
            halt_path,
            _self_hashed(
                {
                    "schema_version": 1,
                    "contract": "risk-score-cluster-safety-halt-v1",
                    "gpu_id": "0",
                    "claim_id": None,
                    "work_id": None,
                    "reason": reason,
                    "halted_at_unix": 999.0,
                },
                "state_sha256",
            ),
        )

    status = collect_status(root, now=1000.0)

    assert "autonomy-terminal-halt" in status["warnings"]
    assert status["autonomy"]["terminal"]["halted"] is True
    assert reason in {
        item["reason"] for item in status["autonomy"]["terminal"]["reasons"]
    }
    assert status["autonomy"]["terminal_remediation_reason"] == reason


@pytest.mark.parametrize(
    ("artifact", "hash_field"),
    (
        ("bootstrap_status", "status_sha256"),
        ("runtime_receipt", "receipt_sha256"),
        ("activation_verification", "receipt_sha256"),
        ("activation_receipt", "receipt_sha256"),
        ("executor_heartbeat", "state_sha256"),
        ("adaptive_status", "status_sha256"),
        ("adaptive_service_status", "status_sha256"),
        ("active_recipe", "record_sha256"),
        ("suite_status", "status_sha256"),
        ("active_suite", "record_sha256"),
    ),
)
def test_status_rejects_invalid_autonomy_self_hash(tmp_path, artifact, hash_field):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    value = json.loads(paths[artifact].read_text(encoding="utf-8"))
    value[hash_field] = "0" * 64
    atomic_write_json(paths[artifact], value)

    with pytest.raises(StatusError, match="self-hash"):
        collect_status(root, now=1000.0)


def test_suite_rotation_service_status_is_discovered_through_v3_input(
    tmp_path,
):
    root = tmp_path.resolve()
    paths = _write_full_autonomy_status_tree(root)
    promotion_services = json.loads(
        paths["service_spec"].read_text(encoding="utf-8")
    )
    registry_binding = promotion_services["service_inputs"][
        "suite_registry_spec"
    ]
    executor_spec = json.loads(
        Path(
            promotion_services["service_inputs"]["executor_spec"]["path"]
        ).read_text(encoding="utf-8")
    )
    service_status_path = (
        root / "promotion" / "suite-rotation-service" / "status.json"
    )
    service_status_path.parent.mkdir(parents=True)
    suite_service_path = root / "configs" / "suite-rotation-service.json"
    suite_service = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-suite-rotation-service-spec-v1",
            "root": str(service_status_path.parent),
            "registry_spec": registry_binding,
            "scheduler_directory": executor_spec["scheduler_directory"],
            "gpu7_id": "7",
            "guardian_argv_prefix": ["/bin/true"],
            "materializer_argv_template": ["/bin/true"],
            "curation_argv_template": ["/bin/true"],
            "continuity_argv_template": ["/bin/true"],
            "results": {
                "status": {
                    "path": str(service_status_path),
                    "contract": (
                        "risk-score-suite-rotation-service-status-v1"
                    ),
                }
            },
            "poll_interval_seconds": 5.0,
            "actor": "suite-service-test",
        },
        "spec_sha256",
    )
    atomic_write_json(suite_service_path, suite_service)
    promotion_services["service_inputs"]["suite_registry_spec"] = {
        "path": str(suite_service_path),
        "sha256": file_sha256(suite_service_path),
    }
    inputs = promotion_status._validate_autonomy_service_inputs(
        root,
        promotion_services,
    )
    service_status = _self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-suite-rotation-service-status-v1",
            "generated_at_utc": _utc_timestamp(999.0),
            "service_spec": {
                "path": str(suite_service_path),
                "sha256": file_sha256(suite_service_path),
                "identity": suite_service["spec_sha256"],
            },
            "actor": "suite-service-test",
            "state": "deployment-requested",
            "next_action": "await-privileged-clean-boundary-deployment",
            "registry": {},
            "candidate_suite_id": "f" * 64,
            "work": {},
            "outputs": {},
            "activation_performed": False,
            "active_pointer_mutated": False,
            "error": None,
        },
        "status_sha256",
    )
    atomic_write_json(service_status_path, service_status)

    suite = promotion_status._suite_rotation_observability(
        root,
        1000.0,
        inputs,
        json.loads(paths["runtime"].read_text(encoding="utf-8")),
    )

    assert suite["summary"]["service"]["state"] == "deployment-requested"
    assert suite["summary"]["service"]["stale"] is False
    assert suite["service_status"]["active_pointer_mutated"] is False
