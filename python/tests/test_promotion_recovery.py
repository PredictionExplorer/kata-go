import contextlib
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.promotion_controller import (
    ConfigurationError,
    IncompleteCandidate,
    InsufficientDiskError,
    PROMOTION_FAILURE_STEPS,
    PromotionController,
    RuntimeConfig,
    SafetyHalt,
    configured_gate_evaluator,
    inspect_candidate,
    inventory_candidates,
    parse_args,
    parse_candidate_counters,
    select_backlog,
)
from risk_score.promotion_state import (
    CandidateState,
    ControllerLock,
    ControllerLockError,
    GenerationState,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)


def digest(label):
    return sha256_bytes(label.encode("utf-8"))


def gate_ranking_summary(
    candidate_hash,
    sample_count,
    candidate_manifest_hash,
    *,
    utility_lcb=0.2,
    final50_upper=0.01,
):
    return {
        "schema_version": 1,
        "source_bound": True,
        "source_cell": "powered_candidate_vs_champion",
        "candidate_hash": candidate_hash,
        "look_number": 1,
        "statistics_artifact_hash": digest(
            f"{candidate_hash}:statistics-artifact"
        ),
        "statistics_manifest_hash": digest(
            f"{candidate_hash}:statistics-manifest"
        ),
        "candidate_manifest_hash": candidate_manifest_hash,
        "realized_powered_utility_lower_bound": utility_lcb,
        "final50_risk_upper_bound": final50_upper,
        "final_50_risk_upper_bound": final50_upper,
        "sample_count": sample_count,
    }


def gpu_handoff_factory(runtime):
    @contextlib.contextmanager
    def factory(config_path, plan, candidate):
        assert config_path == runtime.gpu_lease_config_path
        proof = {
            "lease_id": f"lease-{plan.evaluation_key}",
            "expected_gpu_uuid": runtime.controller.expected_gpu_uuid,
            "handoff_checkpoint_hash": sha256_file(runtime.trainer_checkpoint),
            "clean_observations": [
                {
                    "gpu_uuid": runtime.controller.expected_gpu_uuid,
                    "processes": [],
                },
                {
                    "gpu_uuid": runtime.controller.expected_gpu_uuid,
                    "processes": [],
                },
            ],
            "trainer_restored": True,
            "restored_trainer_identity": {
                "pid": 100,
                "start_time_ticks": 200,
                "command_sha256": digest("trainer-command"),
            },
        }
        yield proof

    return factory


def successful_command(argv):
    command_hash = canonical_sha256(list(argv))
    drain_plan = None
    if "--manifest" in argv:
        drain_plan = json.loads(
            Path(argv[list(argv).index("--manifest") + 1]).read_text(
                encoding="utf-8"
            )
        )
    return {
        "returncode": 0,
        "process_identity": {
            "pid": int(command_hash[:8], 16) % 100000 + 1,
            "start_time_ticks": int(command_hash[8:16], 16),
            "command_sha256": command_hash,
        },
        "process_identity_verified": True,
        "quiescent": True,
        "closed_file_manifests": (
            []
            if drain_plan is None
            else drain_plan["closed_file_manifests"]
        ),
        "process_identities": (
            []
            if drain_plan is None
            else drain_plan["process_identities"]
        ),
        "quiescent_roles": (
            ["selfplay", "shuffler", "trainer", "exporter", "evaluator"]
            if "stop-all" in argv
            else []
        ),
    }


def runtime_mapping(root, *, mutation_enabled=True, max_queue=4, min_free=0):
    root = Path(root).resolve()
    promotion = root / "promotion"
    return {
        "schemaVersion": 1,
        "mutationEnabled": mutation_enabled,
        "actor": "test-controller",
        "hashes": {
            "controller": digest("controller"),
            "source": digest("source"),
            "original": digest("original"),
            "policy": digest("policy"),
            "poweredConfig": digest("powered-config"),
            "standardConfig": digest("standard-config"),
            "discoveryOrdinarySchedule": digest("discovery-schedule"),
            "confirmationOrdinarySchedule": digest("confirmation-schedule"),
            "auditSchedule": digest("audit-schedule"),
            "lead40Schedule": digest("lead40-schedule"),
            "lead80Schedule": digest("lead80-schedule"),
            "standardConfirmationSchedule": digest("confirmation-schedule"),
            "selfplayConfig": digest("selfplay-config"),
            "gpuLeaseConfig": digest("gpu-lease-config"),
            "suiteManifest": digest("suite-manifest"),
        },
        "paths": {
            "promotionRoot": str(promotion),
            "controllerLock": str(promotion / "controller.lock"),
            "champion": str(promotion / "champion.json"),
            "candidateInbox": str(root / "inbox"),
            "candidateQuarantine": str(promotion / "candidates" / "quarantined"),
            "candidateSuperseded": str(promotion / "candidates" / "superseded"),
            "candidateRejected": str(promotion / "candidates" / "rejected"),
            "candidateDeduplicated": str(
                promotion / "candidates" / "deduplicated"
            ),
            "acceptedModels": str(promotion / "accepted"),
            "admittedSelfplay": str(promotion / "admitted"),
            "rolloutQuarantine": str(promotion / "rollout"),
            "rollbackQuarantine": str(promotion / "rollback"),
            "trainerCheckpoint": str(root / "train" / "model.ckpt"),
            "evaluations": str(promotion / "evaluations"),
            "reports": str(promotion / "reports"),
            "suites": str(root / "suites"),
            "policy": str(root / "policy.json"),
            "poweredConfig": str(root / "powered.cfg"),
            "standardConfig": str(root / "standard.cfg"),
            "discoveryOrdinarySchedule": str(root / "discovery.jsonl"),
            "confirmationOrdinarySchedule": str(root / "confirmation.jsonl"),
            "auditSchedule": str(root / "audit.jsonl"),
            "lead40Schedule": str(root / "lead40.jsonl"),
            "lead80Schedule": str(root / "lead80.jsonl"),
            "standardConfirmationSchedule": str(
                root / "standard-confirmation.jsonl"
            ),
            "selfplayConfig": str(root / "selfplay.cfg"),
            "gpuLeaseConfig": str(root / "gpu-lease.json"),
            "dataWatermark": str(root / "data-watermark.json"),
            "shuffleWatermark": str(root / "shuffle-watermark.json"),
            "workerAckInbox": str(promotion / "ipc" / "worker-acks"),
            "rolloutReportInbox": str(promotion / "ipc" / "rollout-reports"),
            "originalModel": str(root / "original.bin.gz"),
        },
        "commands": {
            "trainer": ["python3", "train.py", "--checkpoint", "{checkpoint}"],
            "stage0Probe": [
                "fake-stage0-probe",
                "--request",
                "{stage0_request}",
                "--request-sha256",
                "{stage0_request_sha256}",
                "--candidate",
                "{candidate_model}",
                "--champion",
                "{champion_model}",
                "--output",
                "{stage0_probe}",
            ],
            "evaluator": [
                "fake-evaluator",
                "--plan",
                "{plan}",
                "--candidate",
                "{candidate_model}",
                "--evidence",
                "{evidence_output}",
            ],
            "selfplay": ["katago", "selfplay", "--model", "{model}"],
            "drain": [
                "supervisor",
                "drain",
                "--generation",
                "{generation_id}",
                "--manifest",
                "{drain_manifest}",
            ],
            "rollback": [
                "supervisor",
                "stop-all",
                "--generation",
                "{generation_id}",
            ],
        },
        "polling": {"intervalSeconds": 0.25},
        "limits": {"maxActiveQueue": max_queue, "minFreeBytes": min_free},
        "backlog": {
            "anchorIntervalSamples": 500000,
            "anomalyNames": [],
        },
        "rollout": {
            "workerCount": 7,
            "canaryWorkerCount": 1,
            "intermediateWorkerCount": 3,
            "threadsPerWorker": 100,
        },
    }


def prepare_runtime(tmp_path, *, mutation_enabled=True, min_free=0, max_queue=4):
    root = Path(tmp_path).resolve()
    for directory in (
        root,
        root / "promotion",
        root / "inbox",
        root / "admitted",
        root / "train",
        root / "suites",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "train" / "model.ckpt").write_bytes(b"checkpoint-before-promotion")
    (root / "original.bin.gz").write_bytes(b"original")
    (root / "policy.json").write_text(
        json.dumps(
            {
                "policy_version": "test-v1",
                "threshold": 0.5,
                "frozen_plan": {"source_revision": "a" * 40},
                "evaluation_stages": {
                    "stage_0_integrity_and_fixed_probes": {
                        "fixed_analysis_positions": 8,
                        "fixed_analysis_visits": 16,
                        "exploitability_sentinel_positions": 2,
                        "exploitability_sentinel_visits": 32,
                        "required_checks": ["model_hash", "finite_outputs"],
                    },
                    "stage_2_finalist_selection": {
                        "utility_tie_width": 0.1,
                    },
                    "stage_3_promotion_confirmation": {
                        "looks": [
                            {"look_number": 1},
                            {"look_number": 2},
                        ],
                    },
                    "deep_audit": {
                        "promotion_interval": 5,
                        "near_boundary_fraction": 0.1,
                    },
                },
                "queue": {
                    "maximum_active_evaluator_entries": max_queue,
                    "important_queue_warning_depth": max_queue + 1,
                },
                "retention": {"trash_grace_period_days": 30},
                "rollout": {
                    "worker_count": 7,
                    "canary_workers": 1,
                    "intermediate_workers": 3,
                    "full_workers": 7,
                    "games_per_worker_initial_threads": 100,
                    "canary_games": 2000,
                    "canary_fresh_audit_color_pairs": 1024,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in (
        "powered.cfg",
        "standard.cfg",
        "discovery.jsonl",
        "confirmation.jsonl",
        "audit.jsonl",
        "lead40.jsonl",
        "lead80.jsonl",
        "standard-confirmation.jsonl",
        "selfplay.cfg",
    ):
        (root / name).write_text(name, encoding="utf-8")
    (root / "standard-confirmation.jsonl").write_bytes(
        (root / "confirmation.jsonl").read_bytes()
    )
    (root / "gpu-lease.json").write_text(
        json.dumps({"gpu": {"expectedUuid": "GPU-production-0"}}),
        encoding="utf-8",
    )
    (root / "data-watermark.json").write_text('{"watermark":"data"}')
    (root / "shuffle-watermark.json").write_text('{"watermark":"shuffle"}')
    policy = json.loads((root / "policy.json").read_text(encoding="utf-8"))
    bank_files = {
        "discovery": root / "discovery.jsonl",
        "confirmation": root / "confirmation.jsonl",
        "audit": root / "audit.jsonl",
        "lead-40": root / "lead40.jsonl",
        "lead-80": root / "lead80.jsonl",
    }
    suite_payload = {
        "schemaVersion": 1,
        "policy_hash": canonical_sha256(policy),
        "source_revision": policy["frozen_plan"]["source_revision"],
        "banks": [
            {
                "name": name,
                "positions": {"sha256": digest(f"{name}-positions")},
                "schedule": {
                    "sha256": sha256_file(path),
                    "scheduleId": f"{name}-schedule",
                    "rowCount": 2,
                    "pairCount": 1,
                },
            }
            for name, path in bank_files.items()
        ],
    }
    suite_manifest = {
        **suite_payload,
        "manifestPayloadSha256": canonical_sha256(suite_payload),
    }
    (root / "suites" / "manifest.json").write_bytes(
        canonical_json_bytes(suite_manifest) + b"\n"
    )
    mapping = runtime_mapping(
        root,
        mutation_enabled=mutation_enabled,
        min_free=min_free,
        max_queue=max_queue,
    )
    mapping["hashes"].update(
        {
            "original": sha256_file(root / "original.bin.gz"),
            "policy": canonical_sha256(
                json.loads((root / "policy.json").read_text(encoding="utf-8"))
            ),
            "poweredConfig": sha256_file(root / "powered.cfg"),
            "standardConfig": sha256_file(root / "standard.cfg"),
            "discoveryOrdinarySchedule": sha256_file(root / "discovery.jsonl"),
            "confirmationOrdinarySchedule": sha256_file(
                root / "confirmation.jsonl"
            ),
            "auditSchedule": sha256_file(root / "audit.jsonl"),
            "lead40Schedule": sha256_file(root / "lead40.jsonl"),
            "lead80Schedule": sha256_file(root / "lead80.jsonl"),
            "standardConfirmationSchedule": sha256_file(
                root / "standard-confirmation.jsonl"
            ),
            "selfplayConfig": sha256_file(root / "selfplay.cfg"),
            "gpuLeaseConfig": sha256_file(root / "gpu-lease.json"),
            "suiteManifest": sha256_file(root / "suites" / "manifest.json"),
        }
    )
    return RuntimeConfig.from_mapping(mapping)


def create_candidate(inbox, name="candidate-s500000-d1000000", model=b"candidate"):
    path = Path(inbox) / name
    path.mkdir(parents=True)
    (path / "model.bin.gz").write_bytes(model)
    (path / "model.ckpt").write_bytes(b"checkpoint-" + model)
    return inspect_candidate(path)


def create_hardened_candidate(
    inbox, name="hardened-s500000-d1000000", model=b"hardened-model"
):
    path = Path(inbox) / name
    path.mkdir(parents=True)
    artifacts = {
        "model.bin.gz": model,
        "model.ckpt": b"checkpoint-" + model,
    }
    for relative, content in artifacts.items():
        (path / relative).write_bytes(content)
    files = [
        {
            "path": relative,
            "size": len(content),
            "sha256": sha256_bytes(content),
        }
        for relative, content in sorted(artifacts.items())
    ]
    manifest = {
        "schemaVersion": 1,
        "exportContract": "katago-hardened-candidate-publication-v1",
        "requestFingerprintSha256": digest("export-request"),
        "modelProbePassed": True,
        "candidateName": name,
        "modelName": "test-model",
        "sourceCheckpoint": {
            "name": "model.ckpt",
            "size": len(artifacts["model.ckpt"]),
            "sha256": sha256_bytes(artifacts["model.ckpt"]),
        },
        "files": files,
    }
    (path / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return inspect_candidate(path)


def bootstrap_and_claim(tmp_path, *, max_queue=4):
    runtime = prepare_runtime(tmp_path, max_queue=max_queue)
    controller = PromotionController(
        runtime, automatic=True, command_executor=successful_command
    )
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    artifact = create_candidate(runtime.candidate_inbox)
    status = controller.run_once()
    assert status["inventory"]["selected"] == [artifact.name]
    state = controller.registry.reconstruct()
    assert state.candidates[artifact.model_hash].state == CandidateState.CLAIMED
    return runtime, controller, inspect_candidate(
        Path(state.candidates[artifact.model_hash].candidate_path)
    )


def confirm_and_report(controller, artifact):
    state = controller.registry.reconstruct()
    champion = state.current_champion_hash
    plan = controller.build_evaluation_plan(
        artifact.model_hash,
        champion,
        suite="confirmation",
        stage="stage-3",
        look="final",
        topology="7-workers-100-threads",
    )
    provenance = controller._provenance(plan.config_hash, plan.schedule_hash)
    handoff = {
        "lease_id": "test-confirmation-lease",
        "expected_gpu_uuid": controller.runtime.controller.expected_gpu_uuid,
        "handoff_checkpoint_hash": sha256_file(
            controller.runtime.trainer_checkpoint
        ),
        "clean_observations": [
            {
                "gpu_uuid": controller.runtime.controller.expected_gpu_uuid,
                "processes": [],
            },
            {
                "gpu_uuid": controller.runtime.controller.expected_gpu_uuid,
                "processes": [],
            },
        ],
        "trainer_restored": True,
        "restored_trainer_identity": {
            "pid": 100,
            "start_time_ticks": 200,
            "command_sha256": digest("trainer-command"),
        },
    }
    path = str(artifact.path)
    for target, key in (
        (CandidateState.EVALUATING_INTEGRITY, "integrity"),
        (CandidateState.EVALUATING_SCREEN, "screen"),
        (CandidateState.EVALUATING_FINALIST, "finalist"),
        (CandidateState.EVALUATING_CONFIRMATION, plan.evaluation_key),
        (CandidateState.CONFIRMED, plan.evaluation_key),
    ):
        controller.registry.transition_candidate(
            artifact.model_hash,
            path,
            target,
            provenance=provenance,
            champion_hash=champion,
            evaluation_key=key,
            reason=f"advance to {target.value}",
            actor="test-controller",
        )
    report_path, report_hash, _ = controller.finalize_gate_report(
        plan,
        candidate_hash=artifact.model_hash,
        tested_champion_hash=champion,
        gate_result={
            "decision": "PASS",
            "finalized": True,
            "candidate_hash": artifact.model_hash,
            "tested_champion_hash": champion,
            "original_hash": controller.runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
            "selfplay_config_hash": plan.selfplay_config_hash,
            "topology": plan.topology,
            "gpu_handoff_hash": canonical_sha256(handoff),
            "gpu_handoff": handoff,
            "checks": [],
        },
    )
    return plan, report_path, report_hash


def promotion_kwargs(runtime, report_path, report_hash):
    return {
        "pass_report_path": report_path,
        "pass_report_hash": report_hash,
        "trainer_checkpoint_hash": sha256_file(runtime.trainer_checkpoint),
        "data_watermark_hash": sha256_file(runtime.data_watermark_path),
        "shuffle_watermark_hash": sha256_file(runtime.shuffle_watermark_path),
    }


def acknowledge_worker(controller, runtime, artifact, generation_id, worker_id):
    data = (
        runtime.rollout_quarantine
        / generation_id
        / "data"
        / f"worker-{worker_id:03d}"
    )
    data.mkdir(parents=True, exist_ok=True)
    (data / "games.bin").write_bytes(f"worker-{worker_id}".encode())
    launch_path = next(
        path
        for path in (
            runtime.rollout_quarantine
            / generation_id
            / f"worker-{worker_id:03d}"
        ).glob("launch-*.json")
        if not path.name.endswith(".intent.json")
    )
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "finalized": True,
        "generation_id": generation_id,
        "worker_id": worker_id,
        "model_hash": artifact.model_hash,
        "selfplay_config_hash": runtime.controller.selfplay_config_hash,
        "policy_hash": runtime.controller.policy_hash,
        "threads": runtime.controller.worker_threads,
        "output_manifest_hash": promotion_controller_tree_manifest(data)[0],
        "closed_files": True,
        "process_identity": launch["process_identity"],
    }
    runtime.worker_ack_inbox.mkdir(parents=True, exist_ok=True)
    report_path = (
        runtime.worker_ack_inbox
        / f"{generation_id}.worker-{worker_id:03d}.json"
    )
    atomic_write_json(report_path, report)
    controller.record_worker_ack(
        generation_id,
        worker_id,
        artifact.model_hash,
        report_path=report_path,
        report_hash=sha256_file(report_path),
    )


def promotion_controller_tree_manifest(path):
    rows = []
    total = 0
    for item in sorted(Path(path).iterdir()):
        rows.append(
            {
                "path": item.name,
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
        total += item.stat().st_size
    return canonical_sha256({"schemaVersion": 1, "files": rows}), total


def mark_health_pass(controller, runtime, artifact, generation_id, phase):
    worker_count = 1 if phase == "canary" else 3
    report = {
        "schema_version": 1,
        "finalized": True,
        "decision": "PASS",
        "phase": phase,
        "generation_id": generation_id,
        "candidate_hash": artifact.model_hash,
        "policy_hash": runtime.controller.policy_hash,
        "promotion_config_hash": load_json(
            runtime.promotion_root
            / "transactions"
            / generation_id
            / "intent.json"
        )["config_hash"],
        "selfplay_config_hash": runtime.controller.selfplay_config_hash,
        "audit_schedule_hash": runtime.controller.audit_schedule_hash,
        "topology": "7-workers-100-threads",
        "worker_count": worker_count,
        "model_purity_pass": True,
        "output_schema_pass": True,
        "throughput_pass": True,
        "crash_error_pass": True,
        "behavior_pass": True,
        "tactical_pass": True,
        "exploitability_pass": True,
        "catastrophe_pass": True,
        "game_count": 2000 if phase == "canary" else 6000,
        "fresh_audit_pairs": 1024 if phase == "canary" else 0,
    }
    runtime.rollout_report_inbox.mkdir(parents=True, exist_ok=True)
    report_path = runtime.rollout_report_inbox / f"{generation_id}.{phase}.json"
    atomic_write_json(report_path, report)
    kwargs = {
        "report_path": report_path,
        "report_hash": sha256_file(report_path),
    }
    if phase == "canary":
        controller.mark_canary_passed(
            generation_id, artifact.model_hash, **kwargs
        )
    else:
        controller.mark_intermediate_passed(
            generation_id, artifact.model_hash, **kwargs
        )


def acknowledge_canary(controller, runtime, artifact, generation_id):
    acknowledge_worker(controller, runtime, artifact, generation_id, 0)
    mark_health_pass(
        controller, runtime, artifact, generation_id, "canary"
    )


def acknowledge_remaining(controller, artifact, generation_id):
    for worker_id in (1, 2):
        acknowledge_worker(
            controller, controller.runtime, artifact, generation_id, worker_id
        )
    mark_health_pass(
        controller,
        controller.runtime,
        artifact,
        generation_id,
        "intermediate",
    )


def acknowledge_full(controller, artifact, generation_id):
    for worker_id in range(3, controller.runtime.controller.worker_count):
        acknowledge_worker(
            controller, controller.runtime, artifact, generation_id, worker_id
        )


def converge_promotion(controller, runtime, artifact, generation_id, kwargs):
    for _ in range(8):
        result = controller.promote(
            artifact.model_hash, generation_id, **kwargs
        )
        if result["status"] == "ACTIVE":
            return result
        if result["status"] in {
            "WAITING_CANARY_ACK",
            "WAITING_CANARY_ADMISSION",
        }:
            acknowledge_canary(controller, runtime, artifact, generation_id)
        elif result["status"] in {
            "WAITING_INTERMEDIATE_ACK",
            "WAITING_INTERMEDIATE_HEALTH",
        }:
            acknowledge_remaining(controller, artifact, generation_id)
        elif result["status"] == "WAITING_ROLLOUT_ACK":
            acknowledge_full(controller, artifact, generation_id)
    raise AssertionError("promotion did not converge")


def test_runtime_config_is_strict_absolute_and_safe_by_default(tmp_path):
    prepared = prepare_runtime(tmp_path, mutation_enabled=False)
    mapping = runtime_mapping(tmp_path, mutation_enabled=False)
    mapping["hashes"].update(
        {
            "policy": prepared.controller.policy_hash,
            "gpuLeaseConfig": prepared.controller.gpu_lease_config_hash,
        }
    )
    runtime = RuntimeConfig.from_mapping(mapping)
    assert runtime.controller.mutation_enabled is False
    assert runtime.promotion_root.is_absolute()
    assert runtime.commands["trainer"][0] == "python3"
    assert runtime.controller.worker_count == 7
    assert runtime.controller.worker_threads == 100

    bad = json.loads(json.dumps(mapping))
    bad["unknown"] = True
    with pytest.raises(ConfigurationError, match="unknown"):
        RuntimeConfig.from_mapping(bad)
    bad = json.loads(json.dumps(mapping))
    del bad["paths"]["champion"]
    with pytest.raises(ConfigurationError, match="missing"):
        RuntimeConfig.from_mapping(bad)
    bad = json.loads(json.dumps(mapping))
    bad["paths"]["candidateInbox"] = "relative/inbox"
    with pytest.raises(ConfigurationError, match="absolute"):
        RuntimeConfig.from_mapping(bad)
    bad = json.loads(json.dumps(mapping))
    bad["commands"]["trainer"] = "python3 train.py"
    with pytest.raises(ConfigurationError, match="argv"):
        RuntimeConfig.from_mapping(bad)
    bad = json.loads(json.dumps(mapping))
    bad["paths"]["reports"] = bad["paths"]["evaluations"]
    with pytest.raises(ConfigurationError, match="aliases"):
        RuntimeConfig.from_mapping(bad)
    symlink = tmp_path / "unsafe-link"
    symlink.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    bad = json.loads(json.dumps(mapping))
    bad["paths"]["candidateInbox"] = str(symlink / "inbox")
    with pytest.raises(ConfigurationError, match="symlink ancestor"):
        RuntimeConfig.from_mapping(bad)


def test_runtime_example_is_safe_and_uses_repository_evidence_cli():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "risk_score"
        / "promotion_runtime.example.json"
    )
    example = load_json(example_path)
    assert example["mutationEnabled"] is False
    assert "risk_score.promotion_evaluator" in example["commands"]["evaluator"]
    assert example["commands"]["stage0Probe"][0].endswith(
        "run-risk-score-stage0-probe"
    )
    assert example["paths"]["policy"].endswith(
        "risk_score/promotion_policy_v2.json"
    )
    assert example["paths"]["suites"].endswith("promotion-suites-v2")
    assert example["paths"]["lead40Schedule"].endswith(
        "promotion-suites-v2/schedules/lead-40-confirmation.jsonl"
    )
    assert example["paths"]["lead80Schedule"].endswith(
        "promotion-suites-v2/schedules/lead-80-confirmation.jsonl"
    )
    assert example["paths"]["standardConfirmationSchedule"].endswith(
        "promotion-suites-v2/schedules/prefixes/confirmation-pairs-128.jsonl"
    )


def test_cli_defaults_to_recommendation_and_requires_explicit_automatic():
    args = parse_args(["--runtime-config", "/tmp/runtime.json"])
    assert not args.automatic
    assert not args.recommend_only
    assert args.mode == "once"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--runtime-config",
                "/tmp/runtime.json",
                "--automatic",
                "--recommend-only",
            ]
        )


def test_candidate_validation_counters_and_incomplete_exports(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    complete = create_candidate(inbox, "net-s500000-d900000")
    assert complete.sample_count == 500000
    assert complete.data_count == 900000
    assert parse_candidate_counters(complete.name) == (500000, 900000)

    partial = inbox / "net-s600000-d1000000.partial"
    partial.mkdir()
    (partial / "model.bin.gz").write_bytes(b"partial")
    unfinished = inbox / "net-s700000-d1000000"
    unfinished.mkdir()
    (unfinished / "model.bin.gz").write_bytes(b"incomplete")
    candidates, ignored = inventory_candidates(inbox)
    assert [item.name for item in candidates] == [complete.name]
    assert set(ignored) == {partial.name, unfinished.name}

    (complete.path / "leftover.tmp").write_bytes(b"bad")
    with pytest.raises(IncompleteCandidate, match="temporary"):
        inspect_candidate(complete.path)


def test_hardened_export_manifest_is_strictly_compatible(tmp_path):
    artifact = create_hardened_candidate(tmp_path)
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert artifact.model_hash == manifest["files"][0]["sha256"]
    assert artifact.directory_manifest_hash == sha256_file(manifest_path)

    (artifact.path / "unmanifested.bin").write_bytes(b"extra")
    with pytest.raises(SafetyHalt, match="unmanifested"):
        inspect_candidate(artifact.path)
    (artifact.path / "unmanifested.bin").unlink()

    unprobed = dict(manifest)
    unprobed["modelProbePassed"] = False
    atomic_write_json(manifest_path, unprobed)
    with pytest.raises(SafetyHalt, match="identity"):
        inspect_candidate(artifact.path)
    atomic_write_json(manifest_path, manifest)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(SafetyHalt, match="canonical"):
        inspect_candidate(artifact.path)


def test_policy_hash_is_canonical_object_hash(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.validate_static_inputs()
    policy = json.loads(runtime.policy_path.read_text(encoding="utf-8"))
    runtime.policy_path.write_text(
        json.dumps(policy, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    controller.validate_static_inputs()
    policy["threshold"] = 0.6
    runtime.policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(SafetyHalt, match="canonical hash mismatch"):
        controller.validate_static_inputs()


def test_backlog_selection_preserves_edges_anchors_anomalies_and_started(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    artifacts = [
        create_candidate(
            inbox,
            f"net-{index}-s{samples}-d{samples * 2}",
            model=f"model-{index}".encode(),
        )
        for index, samples in enumerate((0, 450000, 900000, 1400000, 1900000))
    ]
    selection = select_backlog(
        artifacts,
        original_hash=artifacts[0].model_hash,
        anomaly_names=(artifacts[2].name,),
        evaluation_started_hashes=(artifacts[1].model_hash,),
        max_active_queue=5,
    )
    selected_hashes = {item.model_hash for item in selection.selected}
    assert artifacts[0].model_hash in selected_hashes
    assert artifacts[-1].model_hash in selected_hashes
    assert artifacts[2].model_hash in selected_hashes
    assert artifacts[1].model_hash in selected_hashes


def test_duplicate_hash_is_idempotent_and_duplicate_name_conflict_halts(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = create_candidate(first_root, "same-s500000-d1000000", b"same-model")
    duplicate_hash = create_candidate(
        second_root, "later-s1000000-d2000000", b"same-model"
    )
    selection = select_backlog(
        (first, duplicate_hash),
        original_hash=digest("original"),
        max_active_queue=2,
    )
    assert len(selection.selected) == 1
    assert selection.selected[0].model_hash == first.model_hash

    conflicting_name = create_candidate(
        second_root, "same-s500000-d1000000", b"different-model"
    )
    with pytest.raises(SafetyHalt, match="duplicate candidate name"):
        select_backlog(
            (first, conflicting_name),
            original_hash=digest("original"),
            max_active_queue=2,
        )


def test_recommendation_scan_does_not_create_lock_or_move_candidates(tmp_path):
    runtime, mutating, _ = bootstrap_and_claim(tmp_path)
    candidate = create_candidate(
        runtime.candidate_inbox,
        "second-s1000000-d2000000",
        b"second",
    )
    runtime.lock_path.unlink()
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    controller = PromotionController(runtime, automatic=False)
    status = controller.run_once()
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert before == after
    assert candidate.path.exists()
    assert not runtime.lock_path.exists()
    assert status["mode"] == "recommend-only"


def test_claim_rename_crash_recovers_exact_destination(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    artifact = create_candidate(runtime.candidate_inbox)

    def fail(step):
        if step == "candidate-renamed":
            raise RuntimeError("injected crash")

    crashing = PromotionController(runtime, automatic=True, failure_hook=fail)
    with pytest.raises(RuntimeError, match="injected"):
        crashing.run_once()
    destination = runtime.promotion_root / "candidates" / "claimed" / artifact.name
    assert not artifact.path.exists()
    assert destination.exists()
    assert crashing.registry.reconstruct().last_sequence == 1

    recovered = PromotionController(runtime, automatic=True)
    status = recovered.run_reconcile()
    assert status["candidates"][0]["state"] == CandidateState.CLAIMED.value
    assert recovered.registry.reconstruct().candidates[artifact.model_hash].candidate_path == str(
        destination
    )


def test_claim_recovery_halts_on_destination_hash_contradiction(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    artifact = create_candidate(runtime.candidate_inbox)

    def fail(step):
        if step == "candidate-renamed":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        PromotionController(runtime, automatic=True, failure_hook=fail).run_once()
    destination = runtime.promotion_root / "candidates" / "claimed" / artifact.name
    (destination / "model.bin.gz").write_bytes(b"contradictory")
    with pytest.raises(SafetyHalt, match="candidate name|contradict"):
        PromotionController(runtime, automatic=True).run_reconcile()


def test_claim_intent_freezes_parent_champion(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    artifact = create_candidate(runtime.candidate_inbox)

    def crash(step):
        if step == "candidate-renamed":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        PromotionController(runtime, automatic=True, failure_hook=crash).run_once()
    intent = (
        runtime.promotion_root
        / "candidates"
        / "claim-intents"
        / f"{artifact.name}.json"
    )
    value = json.loads(intent.read_text(encoding="utf-8"))
    value["parent_champion_hash"] = digest("different-champion")
    os.chmod(intent, 0o644)
    intent.write_bytes(canonical_json_bytes(value))
    with pytest.raises(SafetyHalt, match="champion changed"):
        PromotionController(runtime, automatic=True).run_reconcile()


def test_duplicate_model_hash_is_durably_archived(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    duplicate = create_candidate(
        runtime.candidate_inbox,
        "duplicate-s1000000-d2000000",
        b"candidate",
    )
    status = controller.run_once()
    archived = runtime.candidate_deduplicated / duplicate.name
    assert status["inventory"]["deduplicated"] == [duplicate.name]
    assert archived.is_dir()
    assert not duplicate.path.exists()
    assert inspect_candidate(archived).model_hash == artifact.model_hash
    assert (
        runtime.promotion_root
        / "candidates"
        / "dedup-intents"
        / f"{duplicate.name}.complete.json"
    ).is_file()
    assert len(controller.registry.reconstruct().candidates) == 1


def test_lock_conflict_and_insufficient_disk_halt_before_intake(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    create_candidate(runtime.candidate_inbox)
    with ControllerLock(runtime.lock_path, owner="other-controller"):
        with pytest.raises(ControllerLockError):
            controller.run_once()

    low_disk = PromotionController(
        runtime,
        automatic=True,
        disk_usage=lambda _path: SimpleNamespace(free=0),
    )
    with pytest.raises(InsufficientDiskError):
        low_disk.run_once()


def test_automatic_controller_can_hold_single_writer_lock_for_process_lifetime(
    tmp_path,
):
    runtime = prepare_runtime(tmp_path)
    held = ControllerLock(runtime.lock_path, owner="persistent-controller").acquire()
    try:
        controller = PromotionController(
            runtime,
            automatic=True,
            held_controller_lock=held,
        )
        controller.bootstrap(
            digest("champion-0"),
            "generation-0",
            confirmation="BOOTSTRAP_INITIAL_CHAMPION",
        )
        controller.run_once()
        with pytest.raises(ControllerLockError):
            ControllerLock(runtime.lock_path, owner="second-controller").acquire()
    finally:
        held.release()

    with ControllerLock(runtime.lock_path, owner="replacement-controller"):
        pass


def test_evaluation_plan_and_missing_executor_are_recommendations(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    plan = controller.build_evaluation_plan(
        artifact.model_hash,
        digest("champion-0"),
        suite="confirmation",
        stage="stage-3",
        look="final",
        topology="7-workers-100-threads",
    )
    assert len(plan.specs) == 5
    assert plan.evaluation_key.startswith("matrix-")
    recommendation = PromotionController(runtime, automatic=False).evaluate_or_recommend(
        plan, artifact
    )
    assert recommendation["decision"] == "RECOMMEND"
    inconclusive = PromotionController(runtime, automatic=True).evaluate_or_recommend(
        plan, artifact
    )
    assert inconclusive["decision"] == "INCONCLUSIVE"


def test_confirmation_plan_selects_exact_authoritative_look_cells(tmp_path):
    runtime = prepare_runtime(tmp_path)
    manifest_path = runtime.suites / "manifest.json"
    base = load_json(manifest_path)
    comparisons = {
        "powered_candidate_vs_champion":
            "candidate-vs-champion-powered",
        "powered_candidate_vs_original":
            "candidate-vs-original-powered",
        "standard_candidate_vs_original":
            "candidate-vs-original-standard",
        "lead_40": "candidate-vs-champion-powered-lead-40",
        "lead_80": "candidate-vs-champion-powered-lead-80",
    }
    pair_counts = {
        "look-1": {
            "powered_candidate_vs_champion": 4,
            "powered_candidate_vs_original": 4,
            "standard_candidate_vs_original": 2,
            "lead_40": 3,
            "lead_80": 5,
        },
        "look-2": {
            "powered_candidate_vs_champion": 8,
            "powered_candidate_vs_original": 8,
            "standard_candidate_vs_original": 2,
            "lead_40": 6,
            "lead_80": 10,
        },
    }
    cells = []
    artifacts = {}
    for look, counts in pair_counts.items():
        for cell_name, comparison in comparisons.items():
            schedule_role = (
                "powered-ordinary"
                if cell_name.startswith("powered_")
                else "standard-ordinary"
                if cell_name.startswith("standard_")
                else cell_name
            )
            key = (look, schedule_role)
            if key not in artifacts:
                path = runtime.suites / "schedules" / f"{look}-{schedule_role}.jsonl"
                path.parent.mkdir(exist_ok=True)
                path.write_text(f"{look}:{schedule_role}\n", encoding="utf-8")
                artifacts[key] = (path, sha256_file(path))
            path, schedule_hash = artifacts[key]
            count = counts[cell_name]
            bank_role = (
                "confirmation"
                if cell_name.startswith(("powered_", "standard_"))
                else schedule_role
            )
            cells.append(
                {
                    "cell_name": cell_name,
                    "stage": "stage-3",
                    "look": look,
                    "comparison": comparison,
                    "suite": (
                        "lead-40"
                        if cell_name == "lead_40"
                        else "lead-80"
                        if cell_name == "lead_80"
                        else "confirmation"
                    ),
                    "visits": (
                        800
                        if cell_name == "standard_candidate_vs_original"
                        else 2000
                    ),
                    "color_pairs": count,
                    "independent_cluster_ids_hash": digest(
                        f"{look}:{schedule_role}:clusters"
                    ),
                    "bank_hash": digest(f"{bank_role}:bank"),
                    "schedule_path": str(path.relative_to(runtime.suites)),
                    "schedule_hash": schedule_hash,
                    "schedule_id": f"{look}-{schedule_role}",
                }
            )
    for stage, stage_cells in (
        (
            "stage-1",
            (
                (
                    "powered_candidate_vs_champion",
                    "candidate-vs-champion-powered",
                    "discovery",
                    400,
                ),
            ),
        ),
        (
            "stage-2",
            (
                (
                    "powered_candidate_vs_champion",
                    "candidate-vs-champion-powered",
                    "discovery",
                    800,
                ),
                (
                    "powered_candidate_vs_original",
                    "candidate-vs-original-powered",
                    "discovery",
                    800,
                ),
                (
                    "lead_40",
                    "candidate-vs-champion-powered-lead-40",
                    "lead-40",
                    800,
                ),
                (
                    "lead_80",
                    "candidate-vs-champion-powered-lead-80",
                    "lead-80",
                    800,
                ),
            ),
        ),
    ):
        for cell_name, comparison, suite, visits in stage_cells:
            schedule_role = (
                "ordinary" if suite == "discovery" else suite
            )
            key = (stage, schedule_role)
            if key not in artifacts:
                path = (
                    runtime.suites
                    / "schedules"
                    / f"{stage}-{schedule_role}.jsonl"
                )
                path.write_text(f"{stage}:{schedule_role}\n", encoding="utf-8")
                artifacts[key] = (path, sha256_file(path))
            path, schedule_hash = artifacts[key]
            cells.append(
                {
                    "cell_name": cell_name,
                    "stage": stage,
                    "look": "automatic",
                    "comparison": comparison,
                    "suite": suite,
                    "visits": visits,
                    "color_pairs": 1,
                    "independent_cluster_ids_hash": digest(
                        f"{stage}:{schedule_role}:clusters"
                    ),
                    "bank_hash": digest(f"{stage}:{suite}:bank"),
                    "schedule_path": str(path.relative_to(runtime.suites)),
                    "schedule_hash": schedule_hash,
                    "schedule_id": f"{stage}-{schedule_role}",
                }
            )
    payload = {
        **{
            key: value
            for key, value in base.items()
            if key != "manifestPayloadSha256"
        },
        "schemaVersion": 2,
        "manifestContract":
            "risk-score-authoritative-evaluation-manifest-v2",
        "cells": cells,
    }
    manifest = {**payload, "manifestPayloadSha256": canonical_sha256(payload)}
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    runtime = replace(
        runtime,
        controller=replace(
            runtime.controller,
            suite_manifest_hash=sha256_file(manifest_path),
        ),
    )
    controller = PromotionController(runtime, automatic=False)
    look1 = controller.build_evaluation_plan(
        digest("candidate"),
        digest("champion"),
        suite="confirmation",
        stage="confirmation",
        look="look-1",
        topology="7-workers-100-threads",
    )
    look2 = controller.build_evaluation_plan(
        digest("candidate"),
        digest("champion"),
        suite="confirmation",
        stage="confirmation",
        look="look-2",
        topology="7-workers-100-threads",
    )
    assert look1.look == "look-1"
    assert look2.look == "look-2"
    assert look1.evaluation_key != look2.evaluation_key
    for plan, look in ((look1, "look-1"), (look2, "look-2")):
        assert set(plan.schedule_artifacts) == set(comparisons)
        for spec in plan.specs:
            selected = next(
                cell
                for cell in cells
                if cell["look"] == look
                and cell["comparison"] == spec.comparison
            )
            assert spec.schedule_sha == selected["schedule_hash"]
            assert (
                plan.schedule_artifacts[
                    selected["cell_name"]
                ]["path"]
                == str(runtime.suites / selected["schedule_path"])
            )

    stage0 = controller.build_evaluation_plan(
        digest("candidate"),
        digest("champion"),
        suite="integrity",
        stage="integrity",
        look="automatic",
        topology="7-workers-100-threads",
    )
    assert stage0.specs == ()
    assert stage0.schedule_artifacts == {}
    assert stage0.evaluation_key.startswith("probe-")

    stage1 = controller.build_evaluation_plan(
        digest("candidate"),
        digest("champion"),
        suite="screen",
        stage="screen",
        look="automatic",
        topology="7-workers-100-threads",
    )
    assert [spec.comparison for spec in stage1.specs] == [
        "candidate-vs-champion-powered"
    ]
    assert stage1.specs[0].max_visits == 400
    assert stage1.schedule_artifacts[
        "powered_candidate_vs_champion"
    ]["pairCount"] == 1

    stage2 = controller.build_evaluation_plan(
        digest("candidate"),
        digest("champion"),
        suite="finalist",
        stage="finalist",
        look="automatic",
        topology="7-workers-100-threads",
    )
    assert [spec.comparison for spec in stage2.specs] == [
        "candidate-vs-champion-powered",
        "candidate-vs-original-powered",
        "candidate-vs-champion-powered-lead-40",
        "candidate-vs-champion-powered-lead-80",
    ]
    assert {spec.max_visits for spec in stage2.specs} == {800}

    stage2_original_champion = controller.build_evaluation_plan(
        digest("candidate"),
        runtime.controller.original_hash,
        suite="finalist",
        stage="finalist",
        look="automatic",
        topology="7-workers-100-threads",
    )
    assert "candidate-vs-original-powered" not in {
        spec.comparison for spec in stage2_original_champion.specs
    }


def test_finalize_gate_report_rejects_bare_and_stale_pass(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    champion = controller.registry.reconstruct().current_champion_hash
    plan = controller.build_evaluation_plan(
        artifact.model_hash,
        champion,
        suite="confirmation",
        stage="stage-3",
        look="fresh",
        topology="7-workers-100-threads",
    )
    with pytest.raises(SafetyHalt, match="not finalized"):
        controller.finalize_gate_report(
            plan,
            candidate_hash=artifact.model_hash,
            tested_champion_hash=champion,
            gate_result={"decision": "PASS"},
        )
    stale = {
        "decision": "PASS",
        "finalized": True,
        "candidate_hash": artifact.model_hash,
        "tested_champion_hash": digest("stale"),
        "original_hash": runtime.controller.original_hash,
        "evaluation_key": plan.evaluation_key,
        "config_hash": plan.config_hash,
        "schedule_hash": plan.schedule_hash,
        "policy_hash": plan.policy_hash,
    }
    with pytest.raises(SafetyHalt, match="tested_champion_hash"):
        controller.finalize_gate_report(
            plan,
            candidate_hash=artifact.model_hash,
            tested_champion_hash=champion,
            gate_result=stale,
        )


def test_injected_evaluation_and_gate_adapters_drive_confirmation(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    calls = []

    def execute(plan, candidate):
        calls.append((plan.evaluation_key, candidate.model_hash))
        return {
            "evaluation_key": plan.evaluation_key,
            "candidate_hash": candidate.model_hash,
            "tested_champion_hash": controller.registry.reconstruct().current_champion_hash,
            "original_hash": runtime.controller.original_hash,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
            "complete": True,
        }

    def gate(evidence):
        return {
            "decision": "PASS",
            "finalized": True,
            "evaluation_key": evidence["evaluation_key"],
            "candidate_hash": evidence["candidate_hash"],
            "tested_champion_hash": evidence["tested_champion_hash"],
            "original_hash": evidence["original_hash"],
            "config_hash": evidence["config_hash"],
            "schedule_hash": evidence["schedule_hash"],
            "policy_hash": evidence["policy_hash"],
            "selfplay_config_hash": evidence["selfplay_config_hash"],
            "topology": evidence["topology"],
            "gpu_handoff_hash": evidence["gpu_handoff_hash"],
            "ranking_summary": gate_ranking_summary(
                evidence["candidate_hash"],
                artifact.sample_count,
                evidence["candidate_manifest_hash"],
            ),
        }

    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    for stage in ("integrity", "screen", "finalist", "confirmation"):
        result = controller.process_evaluation_stage(
            artifact.model_hash,
            stage=stage,
            suite=stage,
            look="final",
            topology="7-workers-100-threads",
        )
        assert result["decision"] == "PASS"
    state = controller.registry.reconstruct()
    assert state.candidates[artifact.model_hash].state == CandidateState.CONFIRMED
    assert len(calls) == 4
    assert Path(result["reportPath"]).is_file()
    retry = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="final",
        topology="7-workers-100-threads",
    )
    assert retry["reused"] is True
    assert len(calls) == 4


def sequential_adapters(runtime, controller_holder, calls, *, final_decision="PASS"):
    def execute(plan, candidate):
        calls.append((plan.stage, plan.look, plan.evaluation_key))
        return {
            "evaluation_key": plan.evaluation_key,
            "candidate_hash": candidate.model_hash,
            "tested_champion_hash":
                controller_holder["controller"]
                .registry.reconstruct()
                .current_champion_hash,
            "original_hash": runtime.controller.original_hash,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
        }

    def gate(evidence):
        decision = "PASS"
        next_action = None
        if evidence["controller_stage"] == "confirmation":
            if evidence["look"] == "look-1":
                decision = "INCONCLUSIVE"
                next_action = "CONTINUE_TO_LOOK_2"
            else:
                decision = final_decision
                next_action = (
                    "PROMOTE"
                    if final_decision == "PASS"
                    else "STOP_MAXIMUM_INCONCLUSIVE"
                )
        result = {
            "decision": decision,
            "finalized": True,
            "ranking_summary": gate_ranking_summary(
                evidence["candidate_hash"],
                500000,
                evidence["candidate_manifest_hash"],
            ),
            **{
                key: evidence[key]
                for key in (
                    "candidate_hash",
                    "tested_champion_hash",
                    "original_hash",
                    "evaluation_key",
                    "config_hash",
                    "schedule_hash",
                    "policy_hash",
                    "selfplay_config_hash",
                    "topology",
                    "gpu_handoff_hash",
                )
            },
        }
        if next_action is not None:
            result["next_action"] = next_action
        return result

    return execute, gate


def advance_to_first_confirmation_look(
    controller, artifact, *, final_decision="PASS"
):
    for stage in ("integrity", "screen", "finalist"):
        result = controller.process_evaluation_stage(
            artifact.model_hash,
            stage=stage,
            suite=stage,
            look="automatic",
            topology="7-workers-100-threads",
        )
        assert result["decision"] == "PASS"
    return controller.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="look-1",
        topology="7-workers-100-threads",
    )


def test_confirmation_look1_continues_to_look2_exactly_once(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    holder = {}
    calls = []
    execute, gate = sequential_adapters(runtime, holder, calls)
    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = controller
    first = advance_to_first_confirmation_look(controller, artifact)
    assert first["decision"] == "INCONCLUSIVE"
    assert first["nextAction"] == "CONTINUE_TO_LOOK_2"

    second = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="look-2",
        topology="7-workers-100-threads",
    )
    retry = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="look-2",
        topology="7-workers-100-threads",
    )
    confirmation_events = [
        event
        for event in controller.registry.reconstruct().events
        if event.transition.value == "evaluation.confirmation_started"
    ]
    assert second["decision"] == retry["decision"] == "PASS"
    assert retry["reused"] is True
    assert [event.payload["look"] for event in confirmation_events] == [
        "look-1",
        "look-2",
    ]
    assert [look for stage, look, _ in calls if stage == "stage-3"] == [
        "look-1",
        "look-2",
    ]


def test_stage0_request_is_deterministic_and_event_bound(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    holder = {}
    calls = []
    execute, gate = sequential_adapters(runtime, holder, calls)
    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = controller
    result = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="integrity",
        suite="integrity",
        look="automatic",
        topology="7-workers-100-threads",
    )
    event = controller.registry.reconstruct().events[-1]
    request_path = Path(event.payload["stage0_request_path"])
    request = load_json(request_path)
    assert result["decision"] == "PASS"
    assert event.payload["stage0_request_hash"] == sha256_file(request_path)
    assert request["contract"] == "risk-score-stage-0-request-v1"
    assert request["candidate_hash"] == artifact.model_hash
    assert request["probe_contract"] == runtime.frozen_policy[
        "evaluation_stages"
    ]["stage_0_integrity_and_fixed_probes"]

    before = request_path.read_bytes()
    retry = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="integrity",
        suite="integrity",
        look="automatic",
        topology="7-workers-100-threads",
    )
    assert retry["reused"] is True
    assert request_path.read_bytes() == before


def test_confirmation_recovers_after_crash_starting_look2(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    holder = {}
    calls = []
    execute, gate = sequential_adapters(runtime, holder, calls)
    base = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = base
    advance_to_first_confirmation_look(base, artifact)
    fired = []

    def crash(step):
        if step == "confirmation-look-2-started" and not fired:
            fired.append(step)
            raise RuntimeError("crash between confirmation looks")

    crashing = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
        failure_hook=crash,
    )
    holder["controller"] = crashing
    with pytest.raises(RuntimeError, match="between confirmation looks"):
        crashing.process_evaluation_stage(
            artifact.model_hash,
            stage="confirmation",
            suite="confirmation",
            look="look-2",
            topology="7-workers-100-threads",
        )

    recovered = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = recovered
    result = recovered.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="look-2",
        topology="7-workers-100-threads",
    )
    events = [
        event
        for event in recovered.registry.reconstruct().events
        if event.transition.value == "evaluation.confirmation_started"
    ]
    assert result["decision"] == "PASS"
    assert len(events) == 2
    assert [look for stage, look, _ in calls if stage == "stage-3"] == [
        "look-1",
        "look-2",
    ]


def test_confirmation_recovers_orphaned_final_report(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    holder = {}
    calls = []
    execute, gate = sequential_adapters(runtime, holder, calls)
    crashing = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = crashing
    for stage in ("integrity", "screen", "finalist"):
        crashing.process_evaluation_stage(
            artifact.model_hash,
            stage=stage,
            suite=stage,
            look="automatic",
            topology="7-workers-100-threads",
        )
    original_finalize = crashing.finalize_gate_report

    def finalize_then_crash(*args, **kwargs):
        original_finalize(*args, **kwargs)
        raise RuntimeError("crash after final report")

    crashing.finalize_gate_report = finalize_then_crash
    with pytest.raises(RuntimeError, match="after final report"):
        crashing.process_evaluation_stage(
            artifact.model_hash,
            stage="confirmation",
            suite="confirmation",
            look="look-1",
            topology="7-workers-100-threads",
        )

    recovered = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = recovered
    result = recovered.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="look-1",
        topology="7-workers-100-threads",
    )
    assert result["decision"] == "INCONCLUSIVE"
    assert result["recovered"] is True
    assert [look for stage, look, _ in calls if stage == "stage-3"] == [
        "look-1"
    ]


def test_final_confirmation_inconclusive_stops_without_promotion(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    holder = {}
    calls = []
    execute, gate = sequential_adapters(
        runtime, holder, calls, final_decision="INCONCLUSIVE"
    )
    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = controller
    advance_to_first_confirmation_look(
        controller, artifact, final_decision="INCONCLUSIVE"
    )
    final = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="confirmation",
        suite="confirmation",
        look="look-2",
        topology="7-workers-100-threads",
    )
    record = controller.registry.reconstruct().candidates[
        artifact.model_hash
    ]
    assert final["decision"] == "INCONCLUSIVE"
    assert final["nextAction"] == "STOP_MAXIMUM_INCONCLUSIVE"
    assert record.state == CandidateState.REJECTED
    assert not any(
        generation.candidate_hash == artifact.model_hash
        for generation in controller.registry.reconstruct().generations.values()
    )


def test_second_look_rejects_stale_champion(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(tmp_path)
    holder = {}
    calls = []
    execute, gate = sequential_adapters(runtime, holder, calls)
    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = controller
    advance_to_first_confirmation_look(controller, artifact)
    old_champion = controller.registry.reconstruct().current_champion_hash

    replacement = create_candidate(
        runtime.candidate_inbox,
        "replacement-s1000000-d2000000",
        b"replacement",
    )
    claimed = controller._claim_candidate(
        replacement, controller.registry.reconstruct()
    )
    provenance = controller._provenance(
        runtime.controller.powered_config_hash,
        runtime.controller.discovery_schedule_hash,
    )
    for target, key in (
        (CandidateState.EVALUATING_INTEGRITY, "replacement-integrity"),
        (CandidateState.EVALUATING_SCREEN, "replacement-screen"),
        (CandidateState.EVALUATING_FINALIST, "replacement-finalist"),
        (CandidateState.EVALUATING_CONFIRMATION, "replacement-confirmation"),
        (CandidateState.CONFIRMED, "replacement-confirmation"),
    ):
        controller.registry.transition_candidate(
            claimed.model_hash,
            str(claimed.path),
            target,
            provenance=provenance,
            champion_hash=old_champion,
            evaluation_key=key,
            reason=f"replacement {target.value}",
            actor="test-controller",
        )
    for target in (
        GenerationState.PROMOTION_INTENT,
        GenerationState.CANARY,
        GenerationState.ROLLOUT,
        GenerationState.ACTIVE,
    ):
        controller.registry.transition_generation(
            "generation-replacement",
            claimed.model_hash,
            str(claimed.path),
            target,
            provenance=provenance,
            tested_champion_hash=old_champion,
            evaluation_key=(
                "replacement-confirmation"
                if target == GenerationState.PROMOTION_INTENT
                else None
            ),
            reason=f"replacement {target.value}",
            actor="test-controller",
        )

    with pytest.raises(SafetyHalt, match="stale champion"):
        controller.process_evaluation_stage(
            artifact.model_hash,
            stage="confirmation",
            suite="confirmation",
            look="look-2",
            topology="7-workers-100-threads",
        )


def test_run_once_orchestrates_all_stages_one_poll_at_a_time(tmp_path):
    runtime = prepare_runtime(tmp_path)
    holder = {}

    def execute(plan, candidate):
        controller = holder["controller"]
        return {
            "evaluation_key": plan.evaluation_key,
            "candidate_hash": candidate.model_hash,
            "tested_champion_hash": controller.registry.reconstruct().current_champion_hash,
            "original_hash": runtime.controller.original_hash,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
        }

    def gate(evidence):
        return {
            "decision": "PASS",
            "finalized": True,
            "ranking_summary": gate_ranking_summary(
                evidence["candidate_hash"],
                artifact.sample_count,
                evidence["candidate_manifest_hash"],
            ),
            **evidence,
        }

    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = controller
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    artifact = create_candidate(runtime.candidate_inbox)
    stages = []
    for _ in range(4):
        status = controller.run_once()
        stages.extend(item["stage"] for item in status["orchestration"])
    assert stages == ["integrity", "screen", "finalist", "confirmation"]
    assert (
        controller.registry.reconstruct().candidates[artifact.model_hash].state
        == CandidateState.CONFIRMED
    )


def test_only_one_confirmation_attempt_is_allocated_per_champion(tmp_path):
    runtime = prepare_runtime(tmp_path)
    holder = {}

    def execute(plan, candidate):
        controller = holder["controller"]
        return {
            "controller_stage": plan.stage,
            "candidate_hash": candidate.model_hash,
            "tested_champion_hash": controller.registry.reconstruct().current_champion_hash,
            "original_hash": runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
        }

    def gate(evidence):
        return {
            "decision": (
                "FAIL"
                if evidence["controller_stage"] == "confirmation"
                else "PASS"
            ),
            "finalized": True,
            "ranking_summary": gate_ranking_summary(
                evidence["candidate_hash"],
                parse_candidate_counters(
                    Path(
                        holder["controller"]
                        .registry.reconstruct()
                        .candidates[evidence["candidate_hash"]]
                        .candidate_path
                    ).name
                )[0],
                evidence["candidate_manifest_hash"],
            ),
            **{
                key: evidence[key]
                for key in (
                    "candidate_hash",
                    "tested_champion_hash",
                    "original_hash",
                    "evaluation_key",
                    "config_hash",
                    "schedule_hash",
                    "policy_hash",
                    "selfplay_config_hash",
                    "topology",
                    "gpu_handoff_hash",
                )
            },
        }

    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=gate,
        gpu_lease_factory=gpu_handoff_factory(runtime),
        command_executor=successful_command,
    )
    holder["controller"] = controller
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    first = create_candidate(
        runtime.candidate_inbox, "a-s500000-d1000000", b"first"
    )
    second = create_candidate(
        runtime.candidate_inbox, "b-s1000000-d2000000", b"second"
    )

    statuses = [controller.run_once() for _ in range(5)]
    confirmation_events = [
        event
        for event in controller.registry.reconstruct().events
        if event.transition.value == "evaluation.confirmation_started"
    ]
    assert [event.candidate_hash for event in confirmation_events] == [
        second.model_hash
    ]
    ranking_path = (
        runtime.evaluations
        / "rankings"
        / f"generation-0.{digest('champion-0')}.json"
    )
    assert load_json(ranking_path)["selected_candidate_hash"] == second.model_hash
    assert (
        controller.registry.reconstruct().candidates[first.model_hash].state
        == CandidateState.EVALUATING_FINALIST
    )
    assert (
        controller.registry.reconstruct().candidates[second.model_hash].state
        == CandidateState.REJECTED
    )
    assert any(
        item.get("reason") == "lower-ranked-safe-finalist"
        for status in statuses
        for item in status["orchestration"]
    )


def test_missing_orchestration_adapters_do_not_advance_candidate(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    artifact = create_candidate(runtime.candidate_inbox)
    status = controller.run_once()
    assert status["orchestration"][0]["decision"] == "INCONCLUSIVE"
    assert (
        controller.registry.reconstruct().candidates[artifact.model_hash].state
        == CandidateState.CLAIMED
    )


def test_configured_evaluator_adapter_is_shell_free_and_identity_checked(tmp_path):
    runtime = prepare_runtime(tmp_path)
    bootstrap = PromotionController(
        runtime, automatic=True, command_executor=successful_command
    )
    bootstrap.bootstrap(
        runtime.controller.original_hash,
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    discovered = create_candidate(runtime.candidate_inbox)
    bootstrap.run_once()
    record = bootstrap.registry.reconstruct().candidates[discovered.model_hash]
    artifact = inspect_candidate(Path(record.candidate_path))
    calls = []

    def command(argv):
        argv = list(argv)
        calls.append(argv)
        if argv[0] == "fake-stage0-probe":
            request_path = Path(argv[argv.index("--request") + 1])
            output_path = Path(argv[argv.index("--output") + 1])
            request = load_json(request_path)
            stage_policy = runtime.frozen_policy["evaluation_stages"][
                "stage_0_integrity_and_fixed_probes"
            ]
            atomic_write_json(
                output_path,
                {
                    "schema_version": 1,
                    "contract": "risk-score-stage-0-probe-output-v1",
                    "finalized": True,
                    "candidate_hash": request["candidate_hash"],
                    "tested_champion_hash": request["tested_champion_hash"],
                    "original_hash": request["original_hash"],
                    "policy_hash": request["policy_hash"],
                    "request_hash": sha256_file(request_path),
                    "checks": {
                        name: True for name in stage_policy["required_checks"]
                    },
                    "measurements": {
                        "fixed_analysis_positions":
                            stage_policy["fixed_analysis_positions"],
                        "fixed_analysis_visits":
                            stage_policy["fixed_analysis_visits"],
                        "exploitability_sentinel_positions":
                            stage_policy["exploitability_sentinel_positions"],
                        "exploitability_sentinel_visits":
                            stage_policy["exploitability_sentinel_visits"],
                        "hard_tactical_failures": 0,
                        "hard_exploitability_failures": 0,
                        "unresolved_failures": 0,
                        "model_runtime_errors": 0,
                        "perspective_violations": 0,
                        "clamp_violations": 0,
                        "endpoint_violations": 0,
                        "nonfinite_violations": 0,
                        "decomposition_violations": 0,
                        "selected_move_endpoint_mass_dominated": False,
                        "visit_stability_acceptable": True,
                    },
                },
            )
            return SimpleNamespace(returncode=0)
        plan_path = Path(argv[argv.index("--plan") + 1])
        evidence_path = Path(argv[argv.index("--evidence") + 1])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        champion = plan["championModelSha256"]
        metadata = {
            "finalized": True,
            "controller_stage": "integrity",
            "candidate_hash": artifact.model_hash,
            "tested_champion_hash": champion,
            "original_hash": runtime.controller.original_hash,
            "evaluation_key": plan["evaluationKey"],
            "config_hash": plan["configHash"],
            "schedule_hash": plan["scheduleHash"],
            "policy_hash": plan["policyHash"],
            "policy_path": plan["policyPath"],
            "policy_version": plan["policyVersion"],
            "suite_manifest_path": plan["suiteManifestPath"],
            "suite_manifest_hash": plan["suiteManifestHash"],
            "look": plan["look"],
            "selfplay_config_hash": plan["selfplayConfigHash"],
            "topology": plan["topology"],
        }
        stage_evidence = {
            "schema_version": 1,
            **metadata,
            "decision": "PASS",
            "source_artifact_hashes": {
                "runner": digest("validated-stage-runner"),
                "statistics": digest("validated-stage-statistics"),
            },
        }
        request_path = next(
            (runtime.evaluations / "stage-0" / "requests").glob("*.json")
        )
        probe_path = next(
            (runtime.evaluations / "stage-0" / "probes").glob("*.json")
        )
        runner_map_path = evidence_path.parent / "runner-manifests.json"
        runner_map = {}
        for name in plan["scheduleArtifacts"]:
            manifest_path = evidence_path.parent / f"{name}.manifest.json"
            atomic_write_json(manifest_path, {"cell": name})
            runner_map[name] = str(manifest_path)
        atomic_write_json(runner_map_path, runner_map)
        atomic_write_json(
            evidence_path,
            {
                "schema_version": 1,
                "controller_stage": "integrity",
                **metadata,
                "schedule_artifacts": plan["scheduleArtifacts"],
                "stage0_request_path": str(request_path),
                "stage0_request_hash": sha256_file(request_path),
                "stage0_probe_path": str(probe_path),
                "stage0_probe_hash": sha256_file(probe_path),
                "stage0_probe_sha256": sha256_file(probe_path),
                "runner_manifests_path": str(runner_map_path),
                "runner_manifests_hash": sha256_file(runner_map_path),
                "stage_evidence": stage_evidence,
                "stage_evidence_hash": canonical_sha256(stage_evidence),
                "stage_gate": {
                    "decision": "PASS",
                    "stage_evidence_hash": canonical_sha256(stage_evidence),
                },
            },
        )
        return SimpleNamespace(returncode=0)

    controller = PromotionController(
        runtime,
        automatic=True,
        gate_evaluator=configured_gate_evaluator,
        command_executor=command,
        gpu_lease_factory=gpu_handoff_factory(runtime),
    )
    controller.evaluation_executor = controller.configured_evaluation_executor
    result = controller.process_evaluation_stage(
        artifact.model_hash,
        stage="integrity",
        suite="integrity",
        look="automatic",
        topology="7-workers-100-threads",
    )
    assert result["decision"] == "PASS"
    assert [call[0] for call in calls] == [
        "fake-stage0-probe",
        "fake-evaluator",
    ]
    assert (
        controller.registry.reconstruct().candidates[artifact.model_hash].state
        == CandidateState.EVALUATING_INTEGRITY
    )

    evidence_path = (
        runtime.evaluations
        / "controller-adapter"
        / result["evaluationKey"]
        / "evidence.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    untrusted = dict(evidence)
    untrusted.pop("stage_evidence")
    untrusted.pop("stage_evidence_hash")
    untrusted["stage_gate"] = {"decision": "PASS"}
    untrusted.update(
        {
            "gpu_handoff_hash": digest("handoff"),
        }
    )
    with pytest.raises(SafetyHalt, match="derived stage evidence"):
        configured_gate_evaluator(untrusted)
    evidence["candidate_hash"] = digest("wrong-candidate")
    atomic_write_json(evidence_path, evidence)
    with pytest.raises(SafetyHalt, match="contradicts"):
        controller.configured_evaluation_executor(
            controller.build_evaluation_plan(
                artifact.model_hash,
                runtime.controller.original_hash,
                suite="integrity",
                stage="integrity",
                look="automatic",
                topology="7-workers-100-threads",
            ),
            artifact,
        )


def test_configured_stage0_probe_missing_output_fails_closed(tmp_path):
    runtime = prepare_runtime(tmp_path)
    bootstrap = PromotionController(
        runtime, automatic=True, command_executor=successful_command
    )
    bootstrap.bootstrap(
        runtime.controller.original_hash,
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    discovered = create_candidate(runtime.candidate_inbox)
    bootstrap.run_once()
    record = bootstrap.registry.reconstruct().candidates[discovered.model_hash]
    artifact = inspect_candidate(Path(record.candidate_path))
    calls = []

    def command(argv):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0)

    controller = PromotionController(
        runtime,
        automatic=True,
        gate_evaluator=configured_gate_evaluator,
        command_executor=command,
        gpu_lease_factory=gpu_handoff_factory(runtime),
    )
    controller.evaluation_executor = controller.configured_evaluation_executor
    with pytest.raises(SafetyHalt, match="did not publish"):
        controller.process_evaluation_stage(
            artifact.model_hash,
            stage="integrity",
            suite="integrity",
            look="automatic",
            topology="7-workers-100-threads",
        )
    assert [call[0] for call in calls] == ["fake-stage0-probe"]
    assert not list(
        (runtime.evaluations / "controller-adapter").glob("*/evidence.json")
    )


def test_superseded_and_rejected_candidates_move_durably(tmp_path):
    runtime = prepare_runtime(tmp_path / "superseded", max_queue=2)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    first = create_candidate(
        runtime.candidate_inbox, "first-s100000-d100000", b"first"
    )
    second = create_candidate(
        runtime.candidate_inbox, "second-s200000-d200000", b"second"
    )
    third = create_candidate(
        runtime.candidate_inbox, "third-s300000-d300000", b"third"
    )
    controller.run_once()
    state = controller.registry.reconstruct()
    superseded = next(
        item
        for item in (first, second, third)
        if state.candidates[item.model_hash].state == CandidateState.SUPERSEDED
    )
    superseded_record = state.candidates[superseded.model_hash]
    assert Path(superseded_record.candidate_path).parent == runtime.candidate_superseded
    assert Path(superseded_record.candidate_path).is_dir()

    runtime2, base, candidate = bootstrap_and_claim(tmp_path / "rejected")

    def execute(plan, artifact):
        return {"evaluation_key": plan.evaluation_key}

    def crash(step):
        if step == "candidate-rejected-renamed":
            raise RuntimeError("crash after rejected rename")

    rejecting = PromotionController(
        runtime2,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=lambda evidence: {
            "decision": "FAIL",
            "gpu_handoff_hash": evidence["gpu_handoff_hash"],
        },
        gpu_lease_factory=gpu_handoff_factory(runtime2),
        failure_hook=crash,
    )
    with pytest.raises(RuntimeError, match="rejected rename"):
        rejecting.process_evaluation_stage(
            candidate.model_hash,
            stage="integrity",
            suite="integrity",
            look="automatic",
            topology="7-workers-100-threads",
        )
    PromotionController(runtime2, automatic=True).run_reconcile()
    rejected = PromotionController(runtime2, automatic=True).registry.reconstruct().candidates[
        candidate.model_hash
    ]
    assert rejected.state == CandidateState.REJECTED
    assert Path(rejected.candidate_path).parent == runtime2.candidate_rejected


def test_partial_worker_ack_canary_admission_and_idempotent_promotion(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)

    first = controller.promote(artifact.model_hash, "generation-1", **kwargs)
    assert first["status"] == "WAITING_CANARY_ACK"
    assert (
        controller.registry.reconstruct().generations["generation-1"].state
        == GenerationState.CANARY
    )
    acknowledge_canary(controller, runtime, artifact, "generation-1")
    second = controller.promote(artifact.model_hash, "generation-1", **kwargs)
    assert second["status"] == "WAITING_INTERMEDIATE_ACK"
    assert (
        runtime.rollout_quarantine / "generation-1" / "data" / "worker-000"
    ).exists()
    assert not (runtime.admitted_selfplay / "generation-1").exists()

    acknowledge_remaining(controller, artifact, "generation-1")
    waiting = controller.promote(artifact.model_hash, "generation-1", **kwargs)
    assert waiting["status"] == "WAITING_ROLLOUT_ACK"
    acknowledge_full(controller, artifact, "generation-1")
    active = controller.promote(artifact.model_hash, "generation-1", **kwargs)
    retry = controller.promote(artifact.model_hash, "generation-1", **kwargs)
    assert active["status"] == retry["status"] == "ACTIVE"
    assert (runtime.admitted_selfplay / "generation-1" / "worker-000").exists()
    assert controller.registry.reconstruct().current_champion_hash == artifact.model_hash


def test_rollout_launches_only_phase_deltas_and_admits_full_generation(tmp_path):
    runtime, base, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(base, artifact)
    runtime = replace(
        runtime,
        commands={
            **runtime.commands,
            "selfplay": (
                "selfplay",
                "--phase",
                "{phase}",
                "--worker",
                "{worker_id}",
            ),
        },
    )
    invocations = []

    def command(argv):
        invocations.append(tuple(argv))
        return successful_command(argv)

    controller = PromotionController(
        runtime, automatic=True, command_executor=command
    )
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    assert (
        controller.promote(artifact.model_hash, "generation-phases", **kwargs)[
            "status"
        ]
        == "WAITING_CANARY_ACK"
    )
    acknowledge_canary(controller, runtime, artifact, "generation-phases")
    assert (
        controller.promote(artifact.model_hash, "generation-phases", **kwargs)[
            "status"
        ]
        == "WAITING_INTERMEDIATE_ACK"
    )
    acknowledge_remaining(controller, artifact, "generation-phases")
    assert (
        controller.promote(artifact.model_hash, "generation-phases", **kwargs)[
            "status"
        ]
        == "WAITING_ROLLOUT_ACK"
    )
    acknowledge_full(controller, artifact, "generation-phases")
    assert (
        controller.promote(artifact.model_hash, "generation-phases", **kwargs)[
            "status"
        ]
        == "ACTIVE"
    )
    assert [
        (item[2], item[4]) for item in invocations if item[0] == "selfplay"
    ] == [
        ("canary", "0"),
        ("intermediate", "1"),
        ("intermediate", "2"),
        ("full", "3"),
        ("full", "4"),
        ("full", "5"),
        ("full", "6"),
    ]
    assert any("drain" in item for item in invocations)
    admitted = runtime.admitted_selfplay / "generation-phases"
    assert {path.name for path in admitted.iterdir()} == {
        "worker-000",
        "worker-001",
        "worker-002",
        "worker-003",
        "worker-004",
        "worker-005",
        "worker-006",
    }
    assert not (
        runtime.rollout_quarantine / "generation-phases" / "data"
    ).exists()


def test_command_executor_nonzero_status_is_a_safety_halt(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(
        runtime,
        automatic=True,
        command_executor=lambda argv: SimpleNamespace(returncode=9),
    )
    with pytest.raises(SafetyHalt, match="status 9"):
        controller.execute_argv(
            "selfplay",
            {
                "model": "/immutable/model.bin.gz",
                "phase": "canary",
                "worker_id": 0,
            },
        )


def test_stale_report_never_creates_promotion_intent(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    plan, _, _ = confirm_and_report(controller, artifact)
    bad_report = runtime.reports / "stale.final.json"
    atomic_write_json(
        bad_report,
        {
            "schema_version": 1,
            "decision": "PASS",
            "finalized": True,
            "candidate_hash": artifact.model_hash,
            "tested_champion_hash": digest("different-champion"),
            "original_hash": runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
        },
    )
    with pytest.raises(SafetyHalt, match="stale"):
        controller.promote(
            artifact.model_hash,
            "generation-stale",
            **promotion_kwargs(runtime, bad_report, sha256_file(bad_report)),
        )
    assert "generation-stale" not in controller.registry.reconstruct().generations


def activate_registry_generation(controller, runtime, index):
    champion = controller.registry.reconstruct().current_champion_hash
    artifact = create_candidate(
        runtime.candidate_inbox,
        f"audit-{index}-s{index * 500000}-d{index * 1000000}",
        f"audit-candidate-{index}".encode(),
    )
    claimed = controller._claim_candidate(
        artifact, controller.registry.reconstruct()
    )
    provenance = controller._provenance(
        runtime.controller.powered_config_hash,
        runtime.controller.discovery_schedule_hash,
    )
    confirmation_key = f"audit-confirmation-{index}"
    for target, key in (
        (CandidateState.EVALUATING_INTEGRITY, f"audit-integrity-{index}"),
        (CandidateState.EVALUATING_SCREEN, f"audit-screen-{index}"),
        (CandidateState.EVALUATING_FINALIST, f"audit-finalist-{index}"),
        (CandidateState.EVALUATING_CONFIRMATION, confirmation_key),
        (CandidateState.CONFIRMED, confirmation_key),
    ):
        controller.registry.transition_candidate(
            claimed.model_hash,
            str(claimed.path),
            target,
            provenance=provenance,
            champion_hash=champion,
            evaluation_key=key,
            reason=f"audit fixture {target.value}",
            actor="test-controller",
        )
    generation_id = f"generation-audit-{index}"
    for target in (
        GenerationState.PROMOTION_INTENT,
        GenerationState.CANARY,
        GenerationState.ROLLOUT,
        GenerationState.ACTIVE,
    ):
        controller.registry.transition_generation(
            generation_id,
            claimed.model_hash,
            str(claimed.path),
            target,
            provenance=provenance,
            tested_champion_hash=champion,
            evaluation_key=(
                confirmation_key
                if target == GenerationState.PROMOTION_INTENT
                else None
            ),
            reason=f"audit fixture {target.value}",
            actor="test-controller",
        )
    return generation_id, claimed


def test_every_fifth_promotion_enqueues_one_deep_audit(tmp_path):
    runtime = prepare_runtime(tmp_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        digest("champion-0"),
        "generation-0",
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    fifth = None
    for index in range(1, 6):
        fifth = activate_registry_generation(controller, runtime, index)
        if index == 1:
            near_generation, near_artifact = fifth
            near = controller._schedule_deep_audit_if_needed(
                near_generation,
                near_artifact.model_hash,
                {
                    "gate": {
                        "ranking_summary": {
                            "near_safety_boundary": True,
                        }
                    }
                },
            )
            assert near["reasons"] == ["near-safety-boundary"]
    generation_id, artifact = fifth
    first = controller._schedule_deep_audit_if_needed(
        generation_id,
        artifact.model_hash,
        {"gate": {}},
    )
    retry = controller._schedule_deep_audit_if_needed(
        generation_id,
        artifact.model_hash,
        {"gate": {}},
    )
    assert first == retry
    assert first["reasons"] == ["every-5-promotions"]
    assert len(
        list(
            (
                runtime.promotion_root / "audits" / "queue"
            ).glob("*.json")
        )
    ) == 2


def test_trash_grace_reference_protection_and_backpressure_status(tmp_path):
    runtime, _, artifact = bootstrap_and_claim(
        tmp_path, max_queue=2
    )
    clock = [datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)]
    controller = PromotionController(
        runtime,
        automatic=True,
        now=lambda: clock[0],
    )
    state = controller.registry.reconstruct()
    provenance = controller._provenance(
        runtime.controller.powered_config_hash,
        runtime.controller.discovery_schedule_hash,
    )
    controller.registry.pin_reference(
        "test-trash-hold",
        artifact.model_hash,
        kind="test-hold",
        owner="test",
        provenance=provenance,
        champion_hash=state.current_champion_hash,
        reason="protect trash fixture",
        actor="test-controller",
    )
    controller._move_candidate_terminal(
        artifact,
        CandidateState.SUPERSEDED,
        provenance=provenance,
        champion_hash=state.current_champion_hash,
        evaluation_key=None,
        reason="trash fixture superseded",
    )
    blocked = controller.reconcile_trash(mutate=True)
    assert blocked[0]["status"] == "BLOCKED_REFERENCES"
    assert Path(
        controller.registry.reconstruct()
        .candidates[artifact.model_hash]
        .candidate_path
    ).is_dir()

    controller.registry.unpin_reference(
        "test-trash-hold",
        provenance=provenance,
        champion_hash=state.current_champion_hash,
        reason="release trash fixture",
        actor="test-controller",
    )
    grace = controller.reconcile_trash(mutate=True)
    object_path = Path(grace[0]["objectPath"])
    assert grace[0]["status"] == "GRACE_PERIOD"
    assert object_path.is_dir()

    for index in range(3):
        create_candidate(
            runtime.candidate_inbox,
            f"queued-{index}-s{(index + 1) * 500000}-d{index + 1}",
            f"queued-{index}".encode(),
        )
    status = controller.reconcile(mutate=False)
    assert status["queue"]["depth"] == 3
    assert status["backpressure"]["exportPaused"] is True
    assert status["backpressure"]["allowExport"] is False
    assert status["lease"]["owner"] is None
    assert status["retention"]["retainedBytes"] >= artifact.size_bytes
    assert status["trash"][0]["status"] == "GRACE_PERIOD"

    clock[0] += timedelta(days=31)
    deleted = controller.reconcile_trash(mutate=True)
    assert deleted[0]["status"] == "DELETED"
    assert not object_path.exists()
    assert controller.reconcile_trash(mutate=True)[0]["reused"] is True


def test_reconcile_status_reports_worker_acks_and_feedback_timestamps(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    converge_promotion(
        controller,
        runtime,
        artifact,
        "generation-feedback",
        promotion_kwargs(runtime, report_path, report_hash),
    )
    for kind in (
        "first-game",
        "first-tdata",
        "first-shuffle",
        "first-training-consumption",
    ):
        controller.record_promotion_feedback(
            "generation-feedback",
            kind,
            evidence={"path": f"/evidence/{kind}"},
        )
    status = controller.reconcile(mutate=False)
    acknowledgements = next(
        row
        for row in status["workerAcknowledgements"]
        if row["generationId"] == "generation-feedback"
    )
    feedback = next(
        row
        for row in status["promotionFeedback"]
        if row["generationId"] == "generation-feedback"
    )
    assert acknowledgements["acknowledgedCount"] == 7
    assert feedback["promotedAtUtc"] is not None
    assert feedback["first_game_at_utc"] is not None
    assert feedback["first_training_consumption_at_utc"] is not None
    assert status["deepAuditQueue"]["pendingDepth"] == 0


def test_deep_audit_failure_triggers_replay_safe_rollback(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    converge_promotion(
        controller, runtime, artifact, "generation-audit-rollback", kwargs
    )
    scheduled = controller.schedule_deep_audit(
        "generation-audit-rollback",
        artifact.model_hash,
        reasons=["near-safety-boundary"],
    )
    external = tmp_path / "deep-audit-fail.json"
    atomic_write_json(
        external,
        {
            "schema_version": 1,
            "finalized": True,
            "decision": "FAIL",
            "rollback_required": True,
            "generation_id": "generation-audit-rollback",
            "candidate_hash": artifact.model_hash,
            "policy_hash": runtime.controller.policy_hash,
            "audit_request_hash": scheduled["request_hash"],
        },
    )
    controller.record_deep_audit_report(
        "generation-audit-rollback",
        report_path=external,
        report_hash=sha256_file(external),
    )
    controller.run_reconcile()
    assert (
        controller.registry.reconstruct()
        .generations["generation-audit-rollback"]
        .state
        == GenerationState.ROLLED_BACK
    )


def test_promotion_failure_hooks_converge_after_rename_and_cas(tmp_path):
    runtime, base, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(base, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    failures = []

    def fail_after_accept(step):
        if step == "promotion-candidate-accepted" and not failures:
            failures.append(step)
            raise RuntimeError("kill after accepted rename")

    controller = PromotionController(
        runtime,
        automatic=True,
        failure_hook=fail_after_accept,
        command_executor=successful_command,
    )
    with pytest.raises(RuntimeError, match="accepted"):
        controller.promote(artifact.model_hash, "generation-crash", **kwargs)
    assert not artifact.path.exists()
    assert (runtime.accepted_models / artifact.name).exists()

    controller = PromotionController(
        runtime, automatic=True, command_executor=successful_command
    )
    controller.run_reconcile()
    assert (
        controller.registry.reconstruct().generations["generation-crash"].state
        == GenerationState.CANARY
    )
    acknowledge_canary(controller, runtime, artifact, "generation-crash")
    controller.promote(artifact.model_hash, "generation-crash", **kwargs)
    acknowledge_remaining(controller, artifact, "generation-crash")
    controller.promote(artifact.model_hash, "generation-crash", **kwargs)
    acknowledge_full(controller, artifact, "generation-crash")

    cas_failures = []

    def fail_after_cas(step):
        if step == "promotion-champion-cas" and not cas_failures:
            cas_failures.append(step)
            raise RuntimeError("kill after champion CAS")

    controller = PromotionController(
        runtime,
        automatic=True,
        failure_hook=fail_after_cas,
        command_executor=successful_command,
    )
    with pytest.raises(RuntimeError, match="champion CAS"):
        controller.promote(artifact.model_hash, "generation-crash", **kwargs)
    assert load_json(runtime.champion_path)["champion_hash"] == artifact.model_hash

    recovered = PromotionController(
        runtime,
        automatic=True,
        command_executor=successful_command,
        process_identity_verifier=lambda identity: True,
    )
    recovered.run_reconcile()
    assert (
        recovered.registry.reconstruct().generations["generation-crash"].state
        == GenerationState.ACTIVE
    )


@pytest.mark.parametrize("failure_step", PROMOTION_FAILURE_STEPS)
def test_every_promotion_failure_boundary_converges(tmp_path, failure_step):
    runtime, base, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(base, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    generation_id = "generation-boundary"

    late_steps = {
        "promotion-canary-admitted",
        "promotion-rollout-event",
        "promotion-intermediate-passed",
        "promotion-all-workers-acknowledged",
        "promotion-generation-data-admitted",
        "promotion-champion-cas",
        "promotion-active-event",
    }
    if failure_step in late_steps:
        assert (
            base.promote(artifact.model_hash, generation_id, **kwargs)["status"]
            == "WAITING_CANARY_ACK"
        )
        acknowledge_canary(base, runtime, artifact, generation_id)
        if failure_step not in {
            "promotion-canary-admitted",
            "promotion-rollout-event",
        }:
            base.promote(artifact.model_hash, generation_id, **kwargs)
            acknowledge_remaining(base, artifact, generation_id)
        if failure_step in {
            "promotion-all-workers-acknowledged",
            "promotion-generation-data-admitted",
            "promotion-champion-cas",
            "promotion-active-event",
        }:
            base.promote(artifact.model_hash, generation_id, **kwargs)
            acknowledge_full(base, artifact, generation_id)

    fired = []

    def fail(step):
        if step == failure_step and not fired:
            fired.append(step)
            raise RuntimeError(f"injected failure at {step}")

    crashing = PromotionController(
        runtime,
        automatic=True,
        failure_hook=fail,
        command_executor=successful_command,
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        crashing.promote(artifact.model_hash, generation_id, **kwargs)
    assert fired == [failure_step]

    recovered = PromotionController(
        runtime,
        automatic=True,
        command_executor=successful_command,
        process_identity_verifier=lambda identity: True,
    )
    converge_promotion(
        recovered, runtime, artifact, generation_id, kwargs
    )
    assert (
        recovered.registry.reconstruct().generations[generation_id].state
        == GenerationState.ACTIVE
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_rollback_before_admission_requires_forensic_flow(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    controller.promote(artifact.model_hash, "generation-rollback", **kwargs)
    staged = (
        runtime.rollout_quarantine
        / "generation-rollback"
        / "data"
        / "worker-000"
    )
    (staged / "game.bin").write_bytes(b"staged")

    with pytest.raises(SafetyHalt, match="forensic"):
        controller.rollback("generation-rollback")
    assert staged.exists()
    assert controller.registry.reconstruct().current_champion_hash == digest("champion-0")


def test_rollback_after_admission_and_consumption_restores_checkpoint(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    shuffle = tmp_path / "derived-shuffle"
    shuffle.mkdir()
    (shuffle / "chunk.bin").write_bytes(b"candidate-derived")
    runtime.shuffle_watermark_path.write_text(
        json.dumps({"derived_paths": [str(shuffle)]}),
        encoding="utf-8",
    )
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    converge_promotion(
        controller, runtime, artifact, "generation-active", kwargs
    )
    original_checkpoint = runtime.trainer_checkpoint.read_bytes()
    runtime.trainer_checkpoint.write_bytes(b"candidate-derived-checkpoint")

    result = controller.rollback(
        "generation-active",
        trainer_consumed=True,
    )
    assert result["status"] == "ROLLED_BACK"
    assert runtime.trainer_checkpoint.read_bytes() == original_checkpoint
    assert not shuffle.exists()
    quarantined = (
        runtime.rollback_quarantine / "generation-active" / "data"
    )
    assert {
        path.name
        for path in (quarantined / "admitted-generation").iterdir()
    } == {f"worker-{worker_id:03d}" for worker_id in range(7)}
    assert (
        quarantined
        / "staged-rollout"
        / "acknowledgements"
        / "worker-002.json"
    ).is_file()
    assert controller.registry.reconstruct().current_champion_hash == digest("champion-0")
    assert load_json(runtime.champion_path)["champion_hash"] == digest("champion-0")


def test_production_topology_and_schedule_banks_are_fail_closed(tmp_path):
    prepared = prepare_runtime(tmp_path)
    mapping = runtime_mapping(tmp_path)
    mapping["hashes"].update(
        {
            "policy": prepared.controller.policy_hash,
            "gpuLeaseConfig": prepared.controller.gpu_lease_config_hash,
        }
    )
    reduced = json.loads(json.dumps(mapping))
    reduced["rollout"]["workerCount"] = 6
    with pytest.raises(ConfigurationError, match="exactly 7"):
        RuntimeConfig.from_mapping(reduced)
    overlap = json.loads(json.dumps(mapping))
    overlap["hashes"]["confirmationOrdinarySchedule"] = overlap["hashes"][
        "discoveryOrdinarySchedule"
    ]
    with pytest.raises(ConfigurationError, match="pairwise distinct"):
        RuntimeConfig.from_mapping(overlap)


def test_gpu_handoff_is_mandatory_and_restoration_is_verified(tmp_path):
    runtime, base, artifact = bootstrap_and_claim(tmp_path)

    def execute(plan, candidate):
        return {}

    controller = PromotionController(
        runtime,
        automatic=True,
        evaluation_executor=execute,
        gate_evaluator=lambda evidence: {
            "decision": "PASS",
            "gpu_handoff_hash": evidence["gpu_handoff_hash"],
        },
    )
    with pytest.raises(SafetyHalt, match="exclusive GPU handoff"):
        controller.process_evaluation_stage(
            artifact.model_hash,
            stage="integrity",
            suite="integrity",
            look="automatic",
            topology="7-workers-100-threads",
        )

    # A complete-looking proof that does not restore the trainer still fails.
    @contextlib.contextmanager
    def bad_factory(config_path, plan, candidate):
        with gpu_handoff_factory(runtime)(config_path, plan, candidate) as proof:
            proof["trainer_restored"] = False
            yield proof

    controller.gpu_lease_factory = bad_factory
    with pytest.raises(SafetyHalt, match="restore trainer"):
        controller.process_evaluation_stage(
            artifact.model_hash,
            stage="integrity",
            suite="integrity",
            look="automatic",
            topology="7-workers-100-threads",
        )


def test_bare_health_markers_and_unsafe_generation_ids_are_rejected(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    controller.promote(artifact.model_hash, "generation-evidence", **kwargs)
    acknowledge_worker(controller, runtime, artifact, "generation-evidence", 0)
    with pytest.raises(SafetyHalt, match="bare canary"):
        controller.mark_canary_passed(
            "generation-evidence", artifact.model_hash
        )
    with pytest.raises(SafetyHalt, match="generation_id"):
        controller.promote(artifact.model_hash, "../escape", **kwargs)


def test_generation_leaf_is_copied_readonly_and_not_hardlinked(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    controller.promote(
        artifact.model_hash,
        "generation-immutable",
        **promotion_kwargs(runtime, report_path, report_hash),
    )
    accepted = runtime.accepted_models / artifact.name / "model.bin.gz"
    leaf = (
        runtime.accepted_models
        / "generations"
        / artifact.model_hash
        / "generation-immutable"
        / "model.bin.gz"
    )
    assert accepted.stat().st_ino != leaf.stat().st_ino
    assert leaf.stat().st_mode & 0o222 == 0
    assert sha256_file(leaf) == artifact.model_hash


def test_projection_mismatch_and_missing_automatic_executor_halt(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    champion = load_json(runtime.champion_path)
    champion["champion_hash"] = digest("foreign-champion")
    champion["record_hash"] = canonical_sha256(
        {key: value for key, value in champion.items() if key != "record_hash"}
    )
    runtime.champion_path.write_bytes(canonical_json_bytes(champion) + b"\n")
    with pytest.raises(SafetyHalt, match="projection differs"):
        controller.reconcile(mutate=True)

    runtime2, base, artifact2 = bootstrap_and_claim(tmp_path / "other")
    _, report_path, report_hash = confirm_and_report(base, artifact2)
    without_executor = PromotionController(runtime2, automatic=True)
    with pytest.raises(SafetyHalt, match="no executor"):
        without_executor.promote(
            artifact2.model_hash,
            "generation-no-executor",
            **promotion_kwargs(runtime2, report_path, report_hash),
        )


def test_rollback_pending_reconcile_resumes_complete_intent(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    _, report_path, report_hash = confirm_and_report(controller, artifact)
    kwargs = promotion_kwargs(runtime, report_path, report_hash)
    converge_promotion(
        controller, runtime, artifact, "generation-rollback-resume", kwargs
    )
    fired = []

    def crash(step):
        if step == "rollback-data-quarantined" and not fired:
            fired.append(step)
            raise RuntimeError("rollback crash")

    crashing = PromotionController(
        runtime,
        automatic=True,
        command_executor=successful_command,
        failure_hook=crash,
    )
    with pytest.raises(RuntimeError, match="rollback crash"):
        crashing.rollback("generation-rollback-resume")
    recovered = PromotionController(
        runtime, automatic=True, command_executor=successful_command
    )
    recovered.run_reconcile()
    assert (
        recovered.registry.reconstruct()
        .generations["generation-rollback-resume"]
        .state
        == GenerationState.ROLLED_BACK
    )


def test_registry_rebuild_ignores_non_authoritative_index(tmp_path):
    runtime, controller, artifact = bootstrap_and_claim(tmp_path)
    index = runtime.promotion_root / "registry.sqlite"
    index.write_bytes(b"lost mutable index")
    before = controller.registry.reconstruct()
    index.unlink()
    rebuilt = controller.registry.reconstruct()
    assert rebuilt.last_event_hash == before.last_event_hash
    assert rebuilt.candidates[artifact.model_hash] == before.candidates[artifact.model_hash]
