import json
from pathlib import Path

import pytest

from risk_score import autonomy_promotion_drills as drills
from risk_score.autonomy_bootstrap import GateSpec, _validate_gate_evidence
from risk_score.generate_schedule import build_schedule
from risk_score.position_samples import semantic_position_sha256
from risk_score.promotion_controller import (
    PROMOTION_FAILURE_STEPS,
    PromotionController,
    RuntimeConfig,
)
from risk_score.promotion_state import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
)


def digest(label):
    return sha256_bytes(label.encode("utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def position(index, label):
    return {
        "xSize": 19,
        "ySize": 19,
        "board": "/".join(["." * 19] * 19),
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": index,
        "hintLoc": "null",
        "metadata": label,
    }


def make_schedule(path, *, seed, index, label):
    rows = build_schedule(
        [position(index, label)],
        bot_a_index=0,
        bot_b_index=1,
        pairs_per_position=1,
        base_seed=seed,
        schedule_id=f"schedule-{label}",
    )
    write_jsonl(path, rows)
    return rows


def production_mapping(root, hashes):
    promotion = root / "promotion"
    return {
        "schemaVersion": 1,
        "mutationEnabled": True,
        "actor": "production-test-controller",
        "hashes": hashes,
        "paths": {
            "promotionRoot": str(promotion),
            "controllerLock": str(promotion / "controller.lock"),
            "champion": str(promotion / "champion.json"),
            "candidateInbox": str(root / "candidate-inbox"),
            "candidateQuarantine": str(promotion / "candidates" / "quarantined"),
            "candidateSuperseded": str(promotion / "candidates" / "superseded"),
            "candidateRejected": str(promotion / "candidates" / "rejected"),
            "candidateDeduplicated": str(promotion / "candidates" / "deduplicated"),
            "acceptedModels": str(promotion / "accepted"),
            "admittedSelfplay": str(root / "admitted-selfplay"),
            "rolloutQuarantine": str(promotion / "rollouts"),
            "rollbackQuarantine": str(promotion / "rollback"),
            "trainerCheckpoint": str(root / "trainer" / "model.ckpt"),
            "evaluations": str(promotion / "evaluations"),
            "reports": str(promotion / "reports"),
            "suites": str(root / "suites"),
            "policy": str(root / "policy.json"),
            "poweredConfig": str(root / "powered.cfg"),
            "standardConfig": str(root / "standard.cfg"),
            "discoveryOrdinarySchedule": str(
                root / "suites" / "schedules" / "discovery.jsonl"
            ),
            "confirmationOrdinarySchedule": str(
                root / "suites" / "schedules" / "confirmation.jsonl"
            ),
            "auditSchedule": str(root / "suites" / "schedules" / "audit.jsonl"),
            "lead40Schedule": str(root / "suites" / "schedules" / "lead-40.jsonl"),
            "lead80Schedule": str(root / "suites" / "schedules" / "lead-80.jsonl"),
            "standardConfirmationSchedule": str(root / "standard-confirmation.jsonl"),
            "selfplayConfig": str(root / "selfplay.cfg"),
            "gpuLeaseConfig": str(root / "gpu-lease.json"),
            "dataWatermark": str(promotion / "watermarks" / "data.json"),
            "shuffleWatermark": str(promotion / "watermarks" / "shuffle.json"),
            "workerAckInbox": str(promotion / "ipc" / "worker-acks"),
            "rolloutReportInbox": str(promotion / "ipc" / "rollout-reports"),
            "originalModel": str(root / "original.bin.gz"),
        },
        "commands": {
            "trainer": ["unused-trainer"],
            "stage0Probe": ["unused-stage0"],
            "evaluator": ["unused-evaluator"],
            "selfplay": ["unused-selfplay"],
            "drain": ["unused-drain"],
            "rollback": ["unused-rollback"],
        },
        "polling": {"intervalSeconds": 0.01},
        "limits": {"maxActiveQueue": 4, "minFreeBytes": 0},
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


@pytest.fixture
def production_runtime(tmp_path):
    root = tmp_path / "production"
    promotion = root / "promotion"
    for path in (
        root,
        promotion,
        root / "candidate-inbox",
        root / "admitted-selfplay",
        root / "trainer",
        root / "suites" / "schedules",
        promotion / "accepted",
        promotion / "rollouts",
        promotion / "rollback",
        promotion / "watermarks",
        promotion / "ipc" / "worker-acks",
        promotion / "ipc" / "rollout-reports",
    ):
        path.mkdir(parents=True, exist_ok=True)

    original = root / "original.bin.gz"
    original.write_bytes(b"immutable-production-original")
    champion_bytes = b"immutable-production-champion"
    champion_hash = sha256_bytes(champion_bytes)
    generation_id = "generation-production-zero"
    champion_leaf = (
        promotion
        / "accepted"
        / "generations"
        / champion_hash
        / generation_id
        / "model.bin.gz"
    )
    champion_leaf.parent.mkdir(parents=True)
    champion_leaf.write_bytes(champion_bytes)
    (root / "trainer" / "model.ckpt").write_bytes(b"production-trainer-checkpoint")
    (root / "powered.cfg").write_text("maxVisits = 2000\n", encoding="utf-8")
    (root / "standard.cfg").write_text("maxVisits = 800\n", encoding="utf-8")
    (root / "selfplay.cfg").write_text("switchNetsMidGame = false\n", encoding="utf-8")
    write_json(
        root / "gpu-lease.json",
        {"gpu": {"expectedUuid": "GPU-production-drill-test"}},
    )
    write_json(promotion / "watermarks" / "data.json", {"watermark": "data"})
    write_json(
        promotion / "watermarks" / "shuffle.json",
        {"watermark": "shuffle", "derived_paths": []},
    )

    policy = {
        "schema_version": 1,
        "policy_version": "risk-seeking-checkpoint-promotion-v3",
        "threshold": 0.5,
        "frozen_plan": {"source_revision": "a" * 40},
        "machine_curation_contract": {
            "final_contract": "risk-score-reviewed-position-bank-v2",
            "review_mode": "machine-consensus",
            "consensus_rules_version": 1,
            "stability_margin": 5.0,
            "allowed_labels": ["ordinary", "lead-40", "lead-80"],
            "model_roles": ["immutable_original", "frozen_champion"],
            "search_modes": ["standard", "powered"],
            "visits": [2000, 8000],
            "symmetry_semantics": "katago-shape-preserving-d4-v1",
            "automatic_promotion_requires_transitive_suite_provenance": True,
        },
        "evaluation_stages": {
            "stage_0_integrity_and_fixed_probes": {
                "required_checks": ["model_hash", "finite_outputs"],
            },
            "stage_2_finalist_selection": {"utility_tie_width": 0.1},
            "stage_3_promotion_confirmation": {
                "powered_visits": 2000,
                "standard_visits": 800,
                "looks": [{"look_number": 1}, {"look_number": 2}],
            },
            "deep_audit": {
                "promotion_interval": 5,
                "near_boundary_fraction": 0.1,
                "ordinary_color_pairs": 1,
                "lead_40_color_pairs": 1,
                "lead_80_color_pairs": 1,
                "visits": [2000, 8000],
                "controls": ["candidate", "champion", "original", "b28"],
            },
        },
        "promotion_thresholds": {
            "powered_win_rate_vs_champion_lower_bound_strictly_above": 0.47,
            "true_no_result_rate_strictly_below": 0.001,
        },
        "queue": {
            "maximum_active_evaluator_entries": 4,
            "important_queue_warning_depth": 5,
        },
        "retention": {"trash_grace_period_days": 30},
        "rollout": {
            "worker_count": 7,
            "canary_workers": 1,
            "intermediate_workers": 3,
            "full_workers": 7,
            "games_per_worker_initial_threads": 100,
            "canary_games": 2,
            "canary_fresh_audit_color_pairs": 1,
        },
    }
    write_json(root / "policy.json", policy)

    schedule_specs = (
        ("discovery", 10, "ordinary-discovery"),
        ("confirmation", 20, "ordinary-confirmation"),
        ("audit", 30, "ordinary-audit"),
        ("lead-40", 40, "lead-40"),
        ("lead-80", 50, "lead-80"),
    )
    banks = []
    for name, index, label in schedule_specs:
        schedule_path = root / "suites" / "schedules" / f"{name}.jsonl"
        rows = make_schedule(
            schedule_path,
            seed=f"seed-{name}",
            index=index,
            label=label,
        )
        bank_hash = digest(f"{name}-position-bank")
        for row in rows:
            semantic_hash = semantic_position_sha256(row["startPosition"])
            row.update(
                {
                    "suite": name,
                    "suiteBank": name,
                    "suiteBankSha256": bank_hash,
                    "suiteQualifiedName": name,
                    "suiteHoldout": ("audit" if name == "audit" else name),
                    "positionContentSha256": canonical_sha256(row["startPosition"]),
                    "positionSemanticSha256": semantic_hash,
                    "independentClusterId": semantic_hash,
                }
            )
        write_jsonl(schedule_path, rows)
        banks.append(
            {
                "name": name,
                "qualifiedName": name,
                "positions": {"sha256": bank_hash},
                "schedule": {
                    "path": f"schedules/{name}.jsonl",
                    "sha256": sha256_file(schedule_path),
                    "scheduleId": rows[0]["scheduleId"],
                    "rowCount": len(rows),
                    "pairCount": 1,
                },
            }
        )
    (root / "standard-confirmation.jsonl").write_bytes(
        (root / "suites" / "schedules" / "confirmation.jsonl").read_bytes()
    )
    source_hash = digest("machine-reviewed-source")
    suite_payload = {
        "schemaVersion": 3,
        "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
        "policy_hash": canonical_sha256(policy),
        "source_revision": policy["frozen_plan"]["source_revision"],
        "machineReviewOnly": True,
        "acceptedLabels": ["lead-40", "lead-80", "ordinary"],
        "sources": [
            {
                "name": "source-positions.jsonl",
                "sha256": source_hash,
                "rowCount": 3,
                "blankLineCount": 0,
            }
        ],
        "curationSources": [
            {
                "source_name": "source-positions.jsonl",
                "contract": "risk-score-reviewed-position-bank-v2",
                "review_mode": "machine-consensus",
                "consensus_rules_version": 1,
                "policy_hash": canonical_sha256(policy),
                "allowed_labels": ["lead-40", "lead-80", "ordinary"],
                "output_sha256": source_hash,
                "manifest_sha256": digest("curation-manifest"),
                "rejected_count": 0,
                "rejected_sha256": digest("curation-rejected"),
                "models": {
                    "original": {
                        "role": "immutable_original",
                        "sha256": sha256_file(original),
                    },
                    "champion": {
                        "role": "frozen_champion",
                        "sha256": champion_hash,
                    },
                },
            }
        ],
        "banks": banks,
    }
    suite = {
        **suite_payload,
        "manifestPayloadSha256": canonical_sha256(suite_payload),
    }
    write_json(root / "suites" / "manifest.json", suite)

    hashes = {
        "controller": digest("controller"),
        "source": digest("source"),
        "original": sha256_file(original),
        "policy": canonical_sha256(policy),
        "poweredConfig": sha256_file(root / "powered.cfg"),
        "standardConfig": sha256_file(root / "standard.cfg"),
        "discoveryOrdinarySchedule": sha256_file(
            root / "suites" / "schedules" / "discovery.jsonl"
        ),
        "confirmationOrdinarySchedule": sha256_file(
            root / "suites" / "schedules" / "confirmation.jsonl"
        ),
        "auditSchedule": sha256_file(root / "suites" / "schedules" / "audit.jsonl"),
        "lead40Schedule": sha256_file(root / "suites" / "schedules" / "lead-40.jsonl"),
        "lead80Schedule": sha256_file(root / "suites" / "schedules" / "lead-80.jsonl"),
        "standardConfirmationSchedule": sha256_file(
            root / "standard-confirmation.jsonl"
        ),
        "selfplayConfig": sha256_file(root / "selfplay.cfg"),
        "gpuLeaseConfig": sha256_file(root / "gpu-lease.json"),
        "suiteManifest": sha256_file(root / "suites" / "manifest.json"),
    }
    runtime_path = root / "promotion-runtime.json"
    write_json(runtime_path, production_mapping(root, hashes))
    runtime = RuntimeConfig.load(runtime_path)
    controller = PromotionController(runtime, automatic=True)
    controller.bootstrap(
        champion_hash,
        generation_id,
        confirmation="BOOTSTRAP_INITIAL_CHAMPION",
    )
    return runtime_path


def make_spec(tmp_path, production_runtime, **limits):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    spec_path = tmp_path / "drill-spec.json"
    drills.publish_drill_spec(
        spec_path,
        production_runtime_path=production_runtime,
        disposable_root=tmp_path / "disposable",
        evidence_root=evidence,
        **limits,
    )
    return spec_path, evidence, tmp_path / "disposable"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_bootstrap_accepts(evidence_root, gate_id):
    gate = GateSpec(
        gate_id=gate_id,
        argv=("drill",),
        evidence=evidence_root / f"{gate_id}.json",
        inputs=(),
        outputs=(),
        requirements={},
    )
    assert _validate_gate_evidence(gate)["verified"] is True


def test_strict_canonical_spec_binds_runtime_commands_and_sentinels(
    tmp_path, production_runtime
):
    spec_path, _, disposable = make_spec(tmp_path, production_runtime)
    spec = drills.load_drill_spec(spec_path)

    assert spec.disposable_root == disposable
    assert spec.module_path == Path(drills.__file__).resolve()
    assert spec.python_executable.is_file()
    assert {
        "champion-projection",
        "champion-model",
        "trainer-checkpoint",
        "admitted-selfplay",
        "promotion-events",
    }.issubset({item["role"] for item in spec.sentinels})

    value = load_json(spec_path)
    value["limits"]["command_timeout_seconds"] += 1
    write_json(spec_path, value)
    with pytest.raises(drills.DrillError, match="identity"):
        drills.load_drill_spec(spec_path)


def test_cli_has_exact_public_gate_subcommands(tmp_path):
    for gate in drills.DRILL_GATES:
        parsed = drills.parse_args([gate, "--spec", str(tmp_path / "spec.json")])
        assert parsed.command == gate
    with pytest.raises(SystemExit):
        drills.parse_args(["not-a-gate", "--spec", str(tmp_path / "spec.json")])


def test_disposable_canary_uses_auditor_evidence_then_cleans_root(
    tmp_path, production_runtime
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)
    production_before = drills._snapshot_path(production_runtime.parent)

    result = drills.run_drill(spec_path, "disposable-canary-drill")

    assert result["decision"] == "PASS"
    assert result["checks"] == {
        "canary_passed": True,
        "fresh_audit_passed": True,
        "disposable_root_removed": True,
        "production_unchanged": True,
    }
    assert not disposable.exists()
    detail = load_json(evidence_root / "disposable-canary-drill.detail.json")
    assert detail["contract"] == drills.DRILL_DETAIL_CONTRACT
    assert detail["observations"]["canary_report_sha256"]
    assert drills._snapshot_path(production_runtime.parent) == production_before
    assert_bootstrap_accepts(evidence_root, "disposable-canary-drill")


def test_crash_replay_injects_every_controller_boundary_and_converges(
    tmp_path, production_runtime
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)

    result = drills.run_drill(spec_path, "crash-replay-drill")

    assert result["decision"] == "PASS"
    assert result["checks"]["production_unchanged"] is True
    assert [item["step"] for item in result["checks"]["boundaries"]] == list(
        PROMOTION_FAILURE_STEPS
    )
    assert all(
        item["crash_injected"] is True and item["replay_converged"] is True
        for item in result["checks"]["boundaries"]
    )
    detail = load_json(evidence_root / "crash-replay-drill.detail.json")
    assert len(detail["observations"]["boundary_details"]) == len(
        PROMOTION_FAILURE_STEPS
    )
    assert not disposable.exists()
    assert_bootstrap_accepts(evidence_root, "crash-replay-drill")


def test_rollback_before_admission_refuses_and_preserves_staged_data(
    tmp_path, production_runtime
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)

    result = drills.run_drill(spec_path, "rollback-before-admission-drill")

    assert result["decision"] == "PASS"
    assert result["checks"] == {
        "rollback_requested": True,
        "refused_without_forensic_flow": True,
        "staged_data_preserved": True,
        "champion_unchanged": True,
        "production_unchanged": True,
    }
    assert not disposable.exists()
    assert_bootstrap_accepts(evidence_root, "rollback-before-admission-drill")


def test_rollback_after_admission_restores_and_quarantines_every_lineage(
    tmp_path, production_runtime
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)

    result = drills.run_drill(spec_path, "rollback-after-admission-drill")

    assert result["decision"] == "PASS"
    assert all(result["checks"].values())
    assert set(result["checks"]) == {
        "rollback_complete",
        "champion_restored",
        "checkpoint_restored",
        "admitted_data_quarantined",
        "derived_data_removed",
        "watermarks_restored",
        "production_unchanged",
    }
    detail = load_json(evidence_root / "rollback-after-admission-drill.detail.json")
    assert detail["observations"]["generation_state"] == "rolled_back"
    assert not disposable.exists()
    assert_bootstrap_accepts(evidence_root, "rollback-after-admission-drill")


def test_shadow_replays_are_independent_identical_and_read_only(
    tmp_path, production_runtime
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)

    result = drills.run_drill(spec_path, "shadow-controller-replay")

    checks = result["checks"]
    assert checks["mutation_enabled"] is False
    assert checks["first_replay_sha256"] == checks["second_replay_sha256"]
    assert len(checks["event_log_sha256"]) == 64
    assert checks["production_unchanged"] is True
    assert not disposable.exists()
    assert_bootstrap_accepts(evidence_root, "shadow-controller-replay")


def test_lease_updated_checkpoint_becomes_dynamic_drill_baseline(
    tmp_path, production_runtime
):
    spec_path, evidence_root, _ = make_spec(tmp_path, production_runtime)
    runtime = RuntimeConfig.load(production_runtime)
    runtime.trainer_checkpoint.write_bytes(b"checkpoint-updated-by-lease-drill")

    result = drills.run_drill(spec_path, "shadow-controller-replay")

    assert result["decision"] == "PASS"
    assert result["checks"]["production_unchanged"] is True
    assert_bootstrap_accepts(evidence_root, "shadow-controller-replay")


def test_failure_never_publishes_pass_and_preserves_forensic_root(
    tmp_path, production_runtime
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)
    value = load_json(spec_path)
    value["limits"]["max_worker_games"] = 1
    payload = dict(value)
    payload.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(payload)
    write_json(spec_path, value)

    with pytest.raises(drills.DrillError, match="forensic root preserved"):
        drills.run_drill(spec_path, "disposable-canary-drill")

    assert disposable.is_dir()
    gate = load_json(evidence_root / "disposable-canary-drill.json")
    failure = load_json(evidence_root / "disposable-canary-drill.failure.json")
    assert gate["decision"] == "FAIL"
    assert failure["forensic_root_present"] is True
    assert not (evidence_root / "disposable-canary-drill.detail.json").exists()


def test_finalization_replays_after_hard_crash(
    tmp_path, production_runtime, monkeypatch
):
    spec_path, evidence_root, disposable = make_spec(tmp_path, production_runtime)
    original_publish = drills.publish_gate_evidence

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(drills, "publish_gate_evidence", interrupt)
    with pytest.raises(KeyboardInterrupt):
        drills.run_drill(spec_path, "shadow-controller-replay")

    detail = load_json(evidence_root / "shadow-controller-replay.detail.json")
    tombstone = (
        disposable.parent
        / f".{disposable.name}.cleanup-{detail['detail_sha256'][:16]}"
    )
    assert not disposable.exists()
    assert tombstone.is_dir()
    assert not (evidence_root / "shadow-controller-replay.json").exists()

    monkeypatch.setattr(drills, "publish_gate_evidence", original_publish)
    evidence = drills.run_drill(spec_path, "shadow-controller-replay")

    assert evidence["decision"] == "PASS"
    assert not disposable.exists()
    assert not tombstone.exists()
