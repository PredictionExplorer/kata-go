import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.build_evaluation_suites import (
    build_evaluation_suites,
    canonical_json as suite_json,
    canonical_sha256 as suite_hash,
)
from risk_score.evaluation_runner import (
    EvaluationRunner,
    EvaluationSpec,
    RUNNER_CONTRACT_V2,
    canonical_json,
    canonical_sha256,
    file_sha256,
    resolve_manifest_cell,
)
from risk_score.paired_stats import compute_paired_statistics
from risk_score.promotion_evidence import (
    CELL_COMPARISONS,
    CELL_ORDER,
    CELL_SUITES,
    DISCOVERY_CONTRACT,
    PromotionEvidenceError,
    build_controller_evidence,
    build_nonconfirmation_controller_evidence,
    build_promotion_evidence,
    derive_discovery_evidence,
    main,
    publish_controller_evidence,
    _load_generic_runner_cell,
)

DEFAULT_V2 = Path(__file__).parents[1] / "risk_score" / "promotion_policy_v2.json"


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


def write_json(path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def small_policy(path):
    policy = json.loads(DEFAULT_V2.read_text(encoding="utf-8"))
    policy["policy_version"] = "risk-seeking-checkpoint-promotion-v2-test"
    stages = policy["evaluation_stages"]
    stages["stage_1_cheap_paired_screen"]["ordinary_color_pairs"] = 2
    stages["stage_2_finalist_selection"].update(
        {
            "ordinary_color_pairs": 2,
            "lead_40_color_pairs": 2,
            "lead_80_color_pairs": 2,
        }
    )
    stages["deep_audit"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    quotas = (
        {
            "powered_candidate_vs_champion": 2,
            "powered_candidate_vs_original": 2,
            "standard_candidate_vs_original": 2,
            "lead_40": 2,
            "lead_80": 2,
        },
        {
            "powered_candidate_vs_champion": 3,
            "powered_candidate_vs_original": 3,
            "standard_candidate_vs_original": 2,
            "lead_40": 3,
            "lead_80": 4,
        },
    )
    for look, counts in zip(stages["stage_3_promotion_confirmation"]["looks"], quotas):
        look.update(
            {
                "powered_ordinary_color_pairs_per_matchup": counts[
                    "powered_candidate_vs_champion"
                ],
                "standard_ordinary_color_pairs": counts[
                    "standard_candidate_vs_original"
                ],
                "lead_40_color_pairs": counts["lead_40"],
                "lead_80_color_pairs": counts["lead_80"],
                "minimum_independent_position_clusters": dict(counts),
            }
        )
    write_json(path, policy)
    return policy


def result_for(row):
    candidate_black = row["blackBot"] == 0
    winner = "B" if candidate_black else "W"
    score = -1.0 if winner == "B" else 1.0
    value = {
        "schemaVersion": 1,
        "scheduleId": row["scheduleId"],
        "gameId": row["gameId"],
        "pairId": row["pairId"],
        "positionId": row["positionId"],
        "seed": row["seed"],
        "blackBot": "candidate" if candidate_black else "reference",
        "whiteBot": "reference" if candidate_black else "candidate",
        "blackBotIndex": row["blackBot"],
        "whiteBotIndex": row["whiteBot"],
        "board": {"xSize": 19, "ySize": 19},
        "rules": {"ko": "POSITIONAL", "scoring": "AREA"},
        "komi": 7.5,
        "finalResult": f"{winner}+1",
        "finalWhiteMinusBlackScore": score,
        "winner": winner,
        "moveCount": 2,
        "blackMoveCount": 1,
        "whiteMoveCount": 1,
        "startTurnNumber": row["startPosition"]["initialTurnNumber"],
        "hitTurnLimit": False,
        "resignation": False,
        "noResult": False,
        "scored": True,
        "gameHash": "hash-" + row["gameId"],
    }
    return value


def moves_for(row, result):
    values = []
    for offset, player in enumerate(("B", "W")):
        values.append(
            {
                "schemaVersion": 1,
                "scheduleId": row["scheduleId"],
                "gameId": row["gameId"],
                "pairId": row["pairId"],
                "positionId": row["positionId"],
                "seed": row["seed"],
                "turnNumber": result["startTurnNumber"] + offset,
                "player": player,
                "bot": result["blackBot"] if player == "B" else result["whiteBot"],
                "move": "D4" if offset == 0 else "Q16",
                "scoreLead": 0.0,
                "winProbability": 0.5,
            }
        )
    return values


class FakeMatch:
    def __call__(self, argv, **kwargs):
        override = argv[argv.index("-override-config") + 1]
        values = {
            item.split("=", 1)[0]: item.split("=", 1)[1] for item in override.split(",")
        }
        schedule_path = Path(values["deterministicScheduleFile"])
        result_path = Path(values["matchResultJsonlFile"])
        move_path = Path(values["matchMoveJsonlFile"])
        schedule = [
            json.loads(line)
            for line in schedule_path.read_text(encoding="utf-8").splitlines()
        ]
        results = [result_for(row) for row in schedule]
        write_jsonl(result_path, results)
        write_jsonl(
            move_path,
            [
                move
                for row, result in zip(schedule, results)
                for move in moves_for(row, result)
            ],
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def add_schedule_bindings(results, schedule):
    by_game = {row["gameId"]: row for row in schedule}
    for result in results:
        source = by_game[result["gameId"]]
        for field in (
            "suite",
            "suiteBank",
            "suiteBankSha256",
            "suiteQualifiedName",
            "suiteHoldout",
            "positionContentSha256",
            "positionSemanticSha256",
            "independentClusterId",
        ):
            result[field] = source[field]
    return results


def build_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy_path = tmp_path / "policy.json"
    policy = small_policy(policy_path)
    source = tmp_path / "positions.jsonl"
    rows = (
        [position(index, "ordinary") for index in range(6)]
        + [position(100 + index, "lead-40") for index in range(6)]
        + [position(200 + index, "lead-80") for index in range(7)]
        + [position(300, "tactical"), position(301, "exploitability")]
    )
    source.write_text("".join(suite_json(row) + "\n" for row in rows), encoding="utf-8")
    suites = build_evaluation_suites(
        [source], tmp_path / "suites", seed="evidence", policy_path=policy_path
    )

    candidate = tmp_path / "candidate.bin"
    champion = tmp_path / "champion.bin"
    original = tmp_path / "original.bin"
    binary = tmp_path / "katago"
    powered = tmp_path / "powered.cfg"
    standard = tmp_path / "standard.cfg"
    for path, data in (
        (candidate, b"candidate"),
        (champion, b"champion"),
        (original, b"original"),
        (binary, b"katago"),
        (powered, b"powered"),
        (standard, b"standard"),
    ):
        path.write_bytes(data)
    models = {"champion": champion, "original": original}
    runner_manifests = {}
    runner_specs = []
    schedule_artifacts = {}
    for name in CELL_ORDER:
        cell = resolve_manifest_cell(
            suites.manifest,
            stage="stage-3",
            look="look-1",
            comparison=CELL_COMPARISONS[name],
            suite=CELL_SUITES[name],
        )
        reference_role = policy["required_confirmation_matrix"][name]["reference"]
        config = standard if "standard" in name else powered
        spec = EvaluationSpec(
            candidate_model_sha=file_sha256(candidate),
            reference_model_sha=file_sha256(models[reference_role]),
            original_model_sha=file_sha256(original),
            config_sha=file_sha256(config),
            schedule_sha=cell["schedule_hash"],
            policy_sha=suite_hash(policy),
            comparison=cell["comparison"],
            suite=cell["suite"],
            stage=cell["stage"],
            look=cell["look"],
            topology="7-workers-100-threads",
            max_visits=cell["visits"],
            suite_manifest_sha=suites.manifest_sha256,
            suite_bank_sha=cell["bank_hash"],
            schedule_id=cell["schedule_id"],
        )
        runner = EvaluationRunner(
            katago_binary=binary,
            config_path=config,
            output_root=tmp_path / "evaluations",
            shard_count=2,
            max_parallel=2,
            max_attempts=1,
            include_move_traces=True,
            subprocess_runner=FakeMatch(),
        )
        outcome = runner.run(
            spec,
            suites.output_dir / cell["schedule_path"],
            candidate,
            models[reference_role],
            original_model_path=original,
            policy_path=policy_path,
            suite_manifest_path=suites.manifest_path,
        )
        runner_manifests[name] = outcome.manifest_path
        runner_specs.append(spec.to_dict())
        schedule_artifacts[name] = {
            "cell": name,
            "comparison": cell["comparison"],
            "stage": cell["stage"],
            "look": cell["look"],
            "path": str(suites.output_dir / cell["schedule_path"]),
            "sha256": cell["schedule_hash"],
            "scheduleId": cell["schedule_id"],
            "pairCount": cell["color_pairs"],
            "suiteBankSha256": cell["bank_hash"],
            "independentClusterIdsSha256": cell["independent_cluster_ids_hash"],
            "visits": cell["visits"],
        }

    discovery_bank = next(
        bank
        for bank in suites.manifest["banks"]
        if bank["qualifiedName"] == "discovery"
    )
    discovery_schedule = [
        json.loads(line)
        for line in (suites.output_dir / discovery_bank["schedule"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    discovery_results = add_schedule_bindings(
        [result_for(row) for row in discovery_schedule], discovery_schedule
    )
    discovery_moves = [
        move
        for row, result in zip(discovery_schedule, discovery_results)
        for move in moves_for(row, result)
    ]
    discovery_binding = {
        "candidate_hash": file_sha256(candidate),
        "reference_hash": file_sha256(champion),
        "comparison": "candidate-vs-champion-powered",
        "suite": "discovery",
        "suite_hash": discovery_bank["positions"]["sha256"],
        "schedule_id": discovery_bank["schedule"]["scheduleId"],
        "schedule_hash": discovery_bank["schedule"]["sha256"],
        "config_hash": file_sha256(powered),
        "runner_manifest_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "katago_binary_hash": file_sha256(binary),
        "independent_cluster_ids": discovery_bank["independentClusterIds"],
        "independent_cluster_ids_hash": discovery_bank["independentClusterIdsSha256"],
    }
    discovery_statistics = compute_paired_statistics(
        discovery_results,
        candidate_bot="candidate",
        move_records=discovery_moves,
        policy=policy,
        look_number=1,
        bootstrap_replications=9,
        data_binding=discovery_binding,
        finalized=True,
    )
    confirmation_clusters = [
        cell["independent_cluster_ids"]
        for cell in suites.manifest["cells"]
        if cell["look"] == "look-1"
    ]
    discovery = derive_discovery_evidence(
        discovery_statistics,
        [discovery_statistics],
        candidate_hash=file_sha256(candidate),
        policy=policy,
        confirmation_cluster_ids=[
            item for group in confirmation_clusters for item in group
        ],
    )

    probe = {
        "schema_version": 1,
        "contract": "risk-score-stage-0-probe-output-v1",
        "finalized": True,
        "candidate_hash": file_sha256(candidate),
        "tested_champion_hash": file_sha256(champion),
        "original_hash": file_sha256(original),
        "policy_hash": suite_hash(policy),
        "request_hash": "3" * 64,
        "checks": {
            name: True
            for name in policy["evaluation_stages"][
                "stage_0_integrity_and_fixed_probes"
            ]["required_checks"]
        },
        "measurements": {
            "fixed_analysis_positions": 256,
            "fixed_analysis_visits": 200,
            "exploitability_sentinel_positions": 16,
            "exploitability_sentinel_visits": 2000,
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
    }
    probe_path = tmp_path / "stage0.json"
    write_json(probe_path, probe)
    attempt = {
        "generation_id": "generation-test",
        "attempt_number": 1,
        "promotions_for_generation": 0,
    }
    config_hash = canonical_sha256(
        sorted({spec["config_sha"] for spec in runner_specs})
    )
    schedule_hash = canonical_sha256(
        sorted({spec["schedule_sha"] for spec in runner_specs})
    )
    plan = {
        "planContract": "risk-score-evaluation-plan-v2",
        "candidateModelSha256": file_sha256(candidate),
        "championModelSha256": file_sha256(champion),
        "originalModelSha256": file_sha256(original),
        "evaluationKey": "matrix-" + canonical_sha256(runner_specs),
        "configHash": config_hash,
        "scheduleHash": schedule_hash,
        "policyHash": suite_hash(policy),
        "policyPath": str(policy_path),
        "policyVersion": policy["policy_version"],
        "selfplayConfigHash": "4" * 64,
        "topology": "7-workers-100-threads",
        "stage": "stage-3",
        "look": "look-1",
        "suiteManifestPath": str(suites.manifest_path),
        "suiteManifestHash": suites.manifest_sha256,
        "scheduleArtifacts": schedule_artifacts,
        "cellOrder": list(CELL_ORDER),
        "specs": runner_specs,
    }
    return {
        "policy": policy,
        "policy_path": policy_path,
        "suites": suites,
        "runner_manifests": runner_manifests,
        "discovery": discovery,
        "probe": probe,
        "probe_path": probe_path,
        "attempt": attempt,
        "plan": plan,
        "candidate": candidate,
        "champion": champion,
        "original": original,
        "binary": binary,
        "powered": powered,
        "standard": standard,
    }


def assemble(fixture):
    promotion = build_promotion_evidence(
        fixture["runner_manifests"],
        suite_manifest_path=fixture["suites"].manifest_path,
        policy_path=fixture["policy_path"],
        stage0_probe_path=fixture["probe_path"],
        stage0_probe_sha256=file_sha256(fixture["probe_path"]),
        discovery_evidence=fixture["discovery"],
        attempt=fixture["attempt"],
        bootstrap_replications=9,
    )
    return promotion, build_controller_evidence(fixture["plan"], promotion)


def test_runner_statistics_to_promotion_evidence_assembly(tmp_path):
    fixture = build_fixture(tmp_path)
    promotion, envelope = assemble(fixture)

    assert list(promotion["confirmation_matrix"]) == list(CELL_ORDER)
    assert promotion["confirmation_finalized"] is True
    assert promotion["discovery"]["stage_1_passed"] is True
    assert promotion["discovery"]["stage_2_passed"] is True
    assert promotion["exploitability"]["stage_0_passed"] is True
    assert promotion["validity"]["promotion_valid"] is True
    assert promotion["combined_lead_artifact"]["finalized"] is True
    assert set(promotion["risk_differences"]) == {
        "final_20",
        "final_50",
        "high_confidence_loss",
        "lead_40_loss",
        "lead_80_loss",
        "targeted_lead_40_suite_loss",
        "targeted_lead_80_suite_loss",
    }
    assert envelope["controller_stage"] == "confirmation"
    assert envelope["promotion_evidence"] == promotion
    assert "stage_gate" not in envelope


def test_missing_cell_tamper_and_arbitrary_discovery_marker_fail(tmp_path):
    fixture = build_fixture(tmp_path)
    missing = dict(fixture["runner_manifests"])
    missing.pop("lead_80")
    with pytest.raises(PromotionEvidenceError, match="exactly the five"):
        build_promotion_evidence(
            missing,
            suite_manifest_path=fixture["suites"].manifest_path,
            policy_path=fixture["policy_path"],
            stage0_probe_path=fixture["probe_path"],
            stage0_probe_sha256=file_sha256(fixture["probe_path"]),
            discovery_evidence=fixture["discovery"],
            attempt=fixture["attempt"],
            bootstrap_replications=9,
        )

    target = fixture["runner_manifests"]["lead_40"].parent / "results.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PromotionEvidenceError, match="hash contradicts"):
        assemble(fixture)

    fixture = build_fixture(tmp_path / "arbitrary")
    arbitrary = {
        "schema_version": 1,
        "contract": DISCOVERY_CONTRACT,
        "finalized": True,
        "summary": {"stage_1_passed": True, "stage_2_passed": True},
        "artifact_hash": "0" * 64,
    }
    with pytest.raises(PromotionEvidenceError, match="artifact_hash|derivable"):
        build_promotion_evidence(
            fixture["runner_manifests"],
            suite_manifest_path=fixture["suites"].manifest_path,
            policy_path=fixture["policy_path"],
            stage0_probe_path=fixture["probe_path"],
            stage0_probe_sha256=file_sha256(fixture["probe_path"]),
            discovery_evidence=arbitrary,
            attempt=fixture["attempt"],
            bootstrap_replications=9,
        )


def test_missing_or_malformed_stage0_probe_fails_closed(tmp_path):
    fixture = build_fixture(tmp_path)
    missing = tmp_path / "missing-stage0.json"
    with pytest.raises((OSError, ValueError)):
        build_promotion_evidence(
            fixture["runner_manifests"],
            suite_manifest_path=fixture["suites"].manifest_path,
            policy_path=fixture["policy_path"],
            stage0_probe_path=missing,
            stage0_probe_sha256="0" * 64,
            discovery_evidence=fixture["discovery"],
            attempt=fixture["attempt"],
            bootstrap_replications=9,
        )

    probe = copy.deepcopy(fixture["probe"])
    probe["stage_0_passed"] = True
    write_json(fixture["probe_path"], probe)
    with pytest.raises(PromotionEvidenceError, match="PASS marker"):
        assemble(fixture)


def test_nonconfirmation_stage_gate_is_derived_from_runner_statistics(tmp_path):
    fixture = build_fixture(tmp_path)
    cell = resolve_manifest_cell(
        fixture["suites"].manifest,
        stage="stage-1",
        look="automatic",
        comparison="candidate-vs-champion-powered",
        suite="discovery",
    )
    schedule_path = fixture["suites"].output_dir / cell["schedule_path"]
    spec = EvaluationSpec(
        candidate_model_sha=file_sha256(fixture["candidate"]),
        reference_model_sha=file_sha256(fixture["champion"]),
        original_model_sha=file_sha256(fixture["original"]),
        config_sha=file_sha256(fixture["powered"]),
        schedule_sha=cell["schedule_hash"],
        policy_sha=suite_hash(fixture["policy"]),
        comparison=cell["comparison"],
        suite=cell["suite"],
        stage="stage-1",
        look="automatic",
        topology="7-workers-100-threads",
        max_visits=cell["visits"],
        suite_manifest_sha=fixture["suites"].manifest_sha256,
        suite_bank_sha=cell["bank_hash"],
        schedule_id=cell["schedule_id"],
    )
    outcome = EvaluationRunner(
        katago_binary=fixture["binary"],
        config_path=fixture["powered"],
        output_root=tmp_path / "stage-1-evaluations",
        shard_count=1,
        max_parallel=1,
        max_attempts=1,
        include_move_traces=True,
        subprocess_runner=FakeMatch(),
    ).run(
        spec,
        schedule_path,
        fixture["candidate"],
        fixture["champion"],
        original_model_path=fixture["original"],
        policy_path=fixture["policy_path"],
        suite_manifest_path=fixture["suites"].manifest_path,
    )
    manifests = {"powered_candidate_vs_champion": outcome.manifest_path}
    specs = [spec.to_dict()]
    plan = {
        "planContract": "risk-score-evaluation-plan-v2",
        "candidateModelSha256": file_sha256(fixture["candidate"]),
        "championModelSha256": file_sha256(fixture["champion"]),
        "originalModelSha256": file_sha256(fixture["original"]),
        "evaluationKey": "matrix-" + canonical_sha256(specs),
        "configHash": canonical_sha256(sorted({spec["config_sha"] for spec in specs})),
        "scheduleHash": canonical_sha256(
            sorted({spec["schedule_sha"] for spec in specs})
        ),
        "policyHash": suite_hash(fixture["policy"]),
        "policyPath": str(fixture["policy_path"]),
        "policyVersion": fixture["policy"]["policy_version"],
        "selfplayConfigHash": "4" * 64,
        "topology": "7-workers-100-threads",
        "stage": "stage-1",
        "look": "automatic",
        "suiteManifestPath": str(fixture["suites"].manifest_path),
        "suiteManifestHash": fixture["suites"].manifest_sha256,
        "scheduleArtifacts": {
            "powered_candidate_vs_champion": {
                "cell": "powered_candidate_vs_champion",
                "comparison": cell["comparison"],
                "stage": "stage-1",
                "look": "automatic",
                "path": str(schedule_path),
                "sha256": cell["schedule_hash"],
                "scheduleId": cell["schedule_id"],
                "pairCount": cell["color_pairs"],
                "suiteBankSha256": cell["bank_hash"],
                "independentClusterIdsSha256":
                    cell["independent_cluster_ids_hash"],
                "minimumIndependentPositionClusters":
                    cell["minimum_independent_position_clusters"],
                "visits": cell["visits"],
            }
        },
        "cellOrder": ["powered_candidate_vs_champion"],
        "specs": specs,
    }
    evidence = build_nonconfirmation_controller_evidence(
        plan,
        runner_manifests=manifests,
        policy_path=fixture["policy_path"],
        bootstrap_replications=9,
    )
    assert evidence["controller_stage"] == "screen"
    assert evidence["stage_gate"]["decision"] == "PASS"
    assert evidence["stage_gate"]["derivation_hash"] == canonical_sha256(
        {
            key: value
            for key, value in evidence["stage_gate"].items()
            if key != "derivation_hash"
        }
    )
    assert "derived_artifacts" in evidence["stage_gate"]

    incomplete_stage_2 = copy.deepcopy(plan)
    incomplete_stage_2["stage"] = "stage-2"
    with pytest.raises(PromotionEvidenceError, match="cellOrder"):
        build_nonconfirmation_controller_evidence(
            incomplete_stage_2,
            runner_manifests=manifests,
            policy_path=fixture["policy_path"],
            bootstrap_replications=9,
        )


def test_historical_runner_v2_bundle_remains_replayable(tmp_path):
    fixture = build_fixture(tmp_path)
    current_path = fixture["runner_manifests"][
        "powered_candidate_vs_champion"
    ]
    legacy = json.loads(current_path.read_text(encoding="utf-8"))
    legacy["runnerContract"] = RUNNER_CONTRACT_V2
    legacy["evaluationSpec"].pop("max_visits")
    legacy["cell"].pop("maxVisits")
    payload = dict(legacy)
    payload.pop("manifestPayloadSha256")
    legacy["manifestPayloadSha256"] = canonical_sha256(payload)
    legacy_path = current_path.parent / "legacy-v2-manifest.json"
    write_json(legacy_path, legacy)

    loaded = _load_generic_runner_cell(
        legacy_path,
        cell_name="powered_candidate_vs_champion",
        policy_hash=suite_hash(fixture["policy"]),
        runner_contract=RUNNER_CONTRACT_V2,
    )
    assert loaded.manifest["runnerContract"] == RUNNER_CONTRACT_V2
    assert "max_visits" not in loaded.manifest["evaluationSpec"]


def test_atomic_cli_publication_is_canonical_and_idempotent(tmp_path):
    fixture = build_fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    map_path = tmp_path / "runner-map.json"
    discovery_path = tmp_path / "discovery.json"
    attempt_path = tmp_path / "attempt.json"
    output_path = tmp_path / "evidence.json"
    write_json(plan_path, fixture["plan"])
    write_json(
        map_path,
        {name: str(path) for name, path in fixture["runner_manifests"].items()},
    )
    write_json(discovery_path, fixture["discovery"])
    write_json(attempt_path, fixture["attempt"])
    argv = [
        "--plan",
        str(plan_path),
        "--runner-manifests",
        str(map_path),
        "--stage0-probe",
        str(fixture["probe_path"]),
        "--stage0-probe-sha256",
        file_sha256(fixture["probe_path"]),
        "--discovery-evidence",
        str(discovery_path),
        "--discovery-evidence-sha256",
        file_sha256(discovery_path),
        "--attempt",
        str(attempt_path),
        "--bootstrap-replications",
        "9",
        "--output",
        str(output_path),
    ]
    assert main(argv) == 0
    first = output_path.read_bytes()
    value = json.loads(first)
    assert first == (canonical_json(value) + "\n").encode("utf-8")
    assert value["promotion_evidence"]["confirmation_finalized"] is True
    assert main(argv) == 0
    assert output_path.read_bytes() == first
    assert not list(tmp_path.glob(".evidence.json.partial-*"))

    conflicting = dict(value, finalized=False)
    with pytest.raises(PromotionEvidenceError, match="contradicts"):
        publish_controller_evidence(output_path, conflicting)
