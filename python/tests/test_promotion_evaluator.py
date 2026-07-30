import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.evaluation_runner import (
    EvaluationSpec,
    canonical_json,
    canonical_sha256,
    file_sha256,
    resolve_manifest_cell,
)
from risk_score.promotion_evaluator import main

from test_promotion_evidence import FakeMatch, build_fixture, write_json


COMPARISONS = (
    (
        "powered_candidate_vs_champion",
        "candidate-vs-champion-powered",
        "champion",
        "powered",
    ),
    (
        "powered_candidate_vs_original",
        "candidate-vs-original-powered",
        "original",
        "powered",
    ),
    (
        "standard_candidate_vs_original",
        "candidate-vs-original-standard",
        "original",
        "standard",
    ),
)
SCHEDULE_OPTIONS = {
    "powered_candidate_vs_champion": "powered-champion",
    "powered_candidate_vs_original": "powered-original",
    "standard_candidate_vs_original": "standard-original",
    "lead_40": "lead40",
    "lead_80": "lead80",
}


class CountingMatch(FakeMatch):
    def __init__(self, *, fail_suite=None):
        self.calls = 0
        self.fail_suite = fail_suite

    def __call__(self, argv, **kwargs):
        self.calls += 1
        override = argv[argv.index("-override-config") + 1]
        values = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in override.split(",")
        }
        schedule_path = Path(values["deterministicScheduleFile"])
        rows = [
            json.loads(line)
            for line in schedule_path.read_text(encoding="utf-8").splitlines()
        ]
        if self.fail_suite is not None and rows[0]["suite"] == self.fail_suite:
            return SimpleNamespace(returncode=17, stdout="", stderr="injected failure")
        return super().__call__(argv, **kwargs)


def make_nonconfirmation_plan(fixture, *, stage, look):
    candidate_hash = file_sha256(fixture["candidate"])
    champion_hash = file_sha256(fixture["champion"])
    original_hash = file_sha256(fixture["original"])
    policy_hash = canonical_sha256(fixture["policy"])
    if stage == "stage-0":
        specs = []
        artifacts = {}
        config_hash = canonical_sha256(
            sorted(
                {
                    file_sha256(fixture["powered"]),
                    file_sha256(fixture["standard"]),
                }
            )
        )
        schedule_hash = canonical_sha256([])
        evaluation_key = "probe-" + canonical_sha256(
            {
                "planContract": "risk-score-evaluation-plan-v2",
                "candidateModelSha256": candidate_hash,
                "championModelSha256": champion_hash,
                "originalModelSha256": original_hash,
                "configHash": config_hash,
                "scheduleHash": schedule_hash,
                "policyHash": policy_hash,
                "suiteManifestHash": fixture["suites"].manifest_sha256,
                "stage": stage,
                "look": look,
                "topology": "7-workers-100-threads",
            }
        )
        cell_order = []
    else:
        cell_order = (
            ["powered_candidate_vs_champion"]
            if stage == "stage-1"
            else [
                "powered_candidate_vs_champion",
                "powered_candidate_vs_original",
                "lead_40",
                "lead_80",
            ]
        )
        definitions = {
            "powered_candidate_vs_champion": (
                "candidate-vs-champion-powered",
                "discovery",
                "champion",
            ),
            "powered_candidate_vs_original": (
                "candidate-vs-original-powered",
                "discovery",
                "original",
            ),
            "lead_40": (
                "candidate-vs-champion-powered-lead-40",
                "lead-40",
                "champion",
            ),
            "lead_80": (
                "candidate-vs-champion-powered-lead-80",
                "lead-80",
                "champion",
            ),
        }
        specs = []
        artifacts = {}
        models = {
            "champion": fixture["champion"],
            "original": fixture["original"],
        }
        for cell_name in cell_order:
            comparison, suite, reference_role = definitions[cell_name]
            cell = resolve_manifest_cell(
                fixture["suites"].manifest,
                stage=stage,
                look=look,
                comparison=comparison,
                suite=suite,
            )
            spec = EvaluationSpec(
                candidate_model_sha=candidate_hash,
                reference_model_sha=file_sha256(models[reference_role]),
                original_model_sha=original_hash,
                config_sha=file_sha256(fixture["powered"]),
                schedule_sha=cell["schedule_hash"],
                policy_sha=policy_hash,
                comparison=comparison,
                suite=suite,
                stage=stage,
                look=look,
                topology="7-workers-100-threads",
                max_visits=cell["visits"],
                suite_manifest_sha=fixture["suites"].manifest_sha256,
                suite_bank_sha=cell["bank_hash"],
                schedule_id=cell["schedule_id"],
            )
            specs.append(spec.to_dict())
            artifacts[cell_name] = {
                "cell": cell_name,
                "comparison": comparison,
                "stage": stage,
                "look": look,
                "path": str(
                    fixture["suites"].output_dir / cell["schedule_path"]
                ),
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
        config_hash = canonical_sha256(
            sorted({spec["config_sha"] for spec in specs})
        )
        schedule_hash = canonical_sha256(
            sorted({spec["schedule_sha"] for spec in specs})
        )
        evaluation_key = "matrix-" + canonical_sha256(specs)
    return {
        "planContract": "risk-score-evaluation-plan-v2",
        "candidateModelSha256": candidate_hash,
        "championModelSha256": champion_hash,
        "originalModelSha256": original_hash,
        "evaluationKey": evaluation_key,
        "configHash": config_hash,
        "scheduleHash": schedule_hash,
        "policyHash": policy_hash,
        "policyPath": str(fixture["policy_path"]),
        "policyVersion": fixture["policy"]["policy_version"],
        "selfplayConfigHash": "4" * 64,
        "topology": "7-workers-100-threads",
        "stage": stage,
        "look": look,
        "suiteManifestPath": str(fixture["suites"].manifest_path),
        "suiteManifestHash": fixture["suites"].manifest_sha256,
        "scheduleArtifacts": artifacts,
        "cellOrder": list(cell_order),
        "specs": specs,
    }


def make_stage0_artifacts(fixture, root):
    stage0_plan = make_nonconfirmation_plan(
        fixture, stage="stage-0", look="automatic"
    )
    request = {
        "schema_version": 1,
        "contract": "risk-score-stage-0-request-v1",
        "candidate_hash": file_sha256(fixture["candidate"]),
        "checkpoint_hash": "a" * 64,
        "candidate_manifest_hash": "b" * 64,
        "tested_champion_hash": file_sha256(fixture["champion"]),
        "original_hash": file_sha256(fixture["original"]),
        "policy_path": str(fixture["policy_path"]),
        "policy_hash": canonical_sha256(fixture["policy"]),
        "policy_version": fixture["policy"]["policy_version"],
        "suite_manifest_path": str(fixture["suites"].manifest_path),
        "suite_manifest_hash": fixture["suites"].manifest_sha256,
        "config_hash": stage0_plan["configHash"],
        "powered_config_path": str(fixture["powered"]),
        "powered_config_hash": file_sha256(fixture["powered"]),
        "standard_config_path": str(fixture["standard"]),
        "standard_config_hash": file_sha256(fixture["standard"]),
        "evaluation_key": stage0_plan["evaluationKey"],
        "stage": "stage-0",
        "look": "automatic",
        "probe_contract": fixture["policy"]["evaluation_stages"][
            "stage_0_integrity_and_fixed_probes"
        ],
        "schedule_artifacts": stage0_plan["scheduleArtifacts"],
    }
    request_path = root / "stage0-request.json"
    write_json(request_path, request)
    probe = copy.deepcopy(fixture["probe"])
    probe["request_hash"] = file_sha256(request_path)
    probe_path = root / "stage0-probe.json"
    write_json(probe_path, probe)
    return request_path, probe_path


def evaluator_argv(
    fixture,
    root,
    *,
    plan=None,
    stage1_evidence=None,
    discovery=None,
):
    root.mkdir(parents=True, exist_ok=True)
    plan = fixture["plan"] if plan is None else plan
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    request_path, probe_path = make_stage0_artifacts(fixture, root)
    runner_map = root / "runner-manifests.json"
    evidence = root / "evidence.json"
    argv = [
        "--plan",
        str(plan_path),
        "--plan-sha256",
        file_sha256(plan_path),
        "--katago",
        str(fixture["binary"]),
        "--candidate-model",
        str(fixture["candidate"]),
        "--candidate-model-sha256",
        file_sha256(fixture["candidate"]),
        "--champion-model",
        str(fixture["champion"]),
        "--champion-model-sha256",
        file_sha256(fixture["champion"]),
        "--original-model",
        str(fixture["original"]),
        "--original-model-sha256",
        file_sha256(fixture["original"]),
        "--powered-config",
        str(fixture["powered"]),
        "--powered-config-sha256",
        file_sha256(fixture["powered"]),
        "--standard-config",
        str(fixture["standard"]),
        "--standard-config-sha256",
        file_sha256(fixture["standard"]),
        "--suite-manifest",
        str(fixture["suites"].manifest_path),
        "--suite-manifest-sha256",
        fixture["suites"].manifest_sha256,
        "--policy",
        str(fixture["policy_path"]),
        "--policy-sha256",
        canonical_sha256(fixture["policy"]),
        "--stage0-request",
        str(request_path),
        "--stage0-request-sha256",
        file_sha256(request_path),
        "--stage0-probe",
        str(probe_path),
        "--stage0-probe-sha256",
        file_sha256(probe_path),
        "--evaluation-root",
        str(root / "evaluations"),
        "--runner-manifests",
        str(runner_map),
        "--shards",
        "2",
        "--max-parallel",
        "2",
        "--max-attempts",
        "1",
        "--bootstrap-replications",
        "9",
        "--output",
        str(evidence),
    ]
    for cell, artifact in plan["scheduleArtifacts"].items():
        option = SCHEDULE_OPTIONS[cell]
        argv.extend(
            [
                f"--{option}-schedule",
                artifact["path"],
                f"--{option}-schedule-sha256",
                artifact["sha256"],
                f"--{option}-schedule-id",
                artifact["scheduleId"],
            ]
        )
    if plan["stage"] == "stage-2":
        assert stage1_evidence is not None
        argv.extend(
            [
                "--stage1-evidence",
                str(stage1_evidence),
                "--stage1-evidence-sha256",
                file_sha256(stage1_evidence),
                "--discovery-output",
                str(root / "discovery.json"),
            ]
        )
    if plan["stage"] == "stage-3":
        discovery_path = root / "discovery.json"
        write_json(
            discovery_path,
            fixture["discovery"] if discovery is None else discovery,
        )
        attempt_path = root / "attempt.json"
        write_json(attempt_path, fixture["attempt"])
        argv.extend(
            [
                "--discovery-evidence",
                str(discovery_path),
                "--discovery-evidence-sha256",
                file_sha256(discovery_path),
                "--attempt",
                str(attempt_path),
                "--attempt-sha256",
                file_sha256(attempt_path),
            ]
        )
    return argv, runner_map, evidence


def test_cli_executes_confirmation_matrix_and_reuses_finalized_cells(tmp_path):
    fixture = build_fixture(tmp_path / "fixture")
    argv, runner_map_path, evidence_path = evaluator_argv(
        fixture, tmp_path / "configured-command"
    )
    fake = CountingMatch()

    assert not runner_map_path.exists()
    assert main(argv, subprocess_runner=fake) == 0
    first_call_count = fake.calls
    runner_map = json.loads(runner_map_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert set(runner_map) == set(fixture["plan"]["scheduleArtifacts"])
    assert all(Path(path).is_file() for path in runner_map.values())
    assert evidence["promotion_evidence"]["confirmation_finalized"] is True
    assert evidence["promotion_evidence"]["validity"]["promotion_valid"] is True
    assert evidence["runner_manifests_hash"] == file_sha256(runner_map_path)
    assert first_call_count > 0

    assert main(argv, subprocess_runner=fake) == 0
    assert fake.calls == first_call_count


def test_nonconfirmation_stages_publish_real_statistics_and_discovery(tmp_path):
    fixture = build_fixture(tmp_path / "fixture")
    stage1 = make_nonconfirmation_plan(
        fixture, stage="stage-1", look="automatic"
    )
    stage1_argv, stage1_map, stage1_evidence = evaluator_argv(
        fixture, tmp_path / "stage1", plan=stage1
    )
    fake = CountingMatch()
    assert main(stage1_argv, subprocess_runner=fake) == 0
    stage1_value = json.loads(stage1_evidence.read_text(encoding="utf-8"))
    assert set(json.loads(stage1_map.read_text(encoding="utf-8"))) == {
        "powered_candidate_vs_champion"
    }
    assert stage1_value["stage_gate"]["decision"] == "PASS"
    assert stage1_value["stage_gate"]["derived_artifacts"]["statistics"]

    stage2 = make_nonconfirmation_plan(
        fixture, stage="stage-2", look="automatic"
    )
    stage2_argv, _, stage2_evidence = evaluator_argv(
        fixture,
        tmp_path / "stage2",
        plan=stage2,
        stage1_evidence=stage1_evidence,
    )
    assert main(stage2_argv, subprocess_runner=fake) == 0
    stage2_value = json.loads(stage2_evidence.read_text(encoding="utf-8"))
    discovery_path = Path(stage2_value["discovery_evidence_path"])
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    assert stage2_value["stage_gate"]["decision"] == "PASS"
    assert discovery["summary"]["stage_1_passed"] is True
    assert discovery["summary"]["stage_2_passed"] is True


def test_stage0_uses_probe_measurements_without_running_matches(tmp_path):
    fixture = build_fixture(tmp_path / "fixture")
    plan = make_nonconfirmation_plan(
        fixture, stage="stage-0", look="automatic"
    )
    argv, runner_map, evidence_path = evaluator_argv(
        fixture, tmp_path / "stage0", plan=plan
    )
    fake = CountingMatch()
    assert main(argv, subprocess_runner=fake) == 0
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert runner_map.is_file()
    assert json.loads(runner_map.read_text(encoding="utf-8")) == {}
    assert fake.calls == 0
    assert evidence["stage_gate"]["decision"] == "PASS"
    assert evidence["stage_gate"]["derived_artifacts"]["stage_0"][
        "stage_0_passed"
    ] is True
    assert "decision" not in json.loads(
        Path(evidence["stage0_probe_path"]).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "mode, expected_decision, expected_reason",
    [
        ("invalid", "FAIL", "MATCH_VALIDATION_FAILED_LEAD_40"),
        (
            "incomplete",
            "INCONCLUSIVE",
            "UTILITY_INFERENCE_UNAVAILABLE_LEAD_40",
        ),
    ],
)
def test_stage2_checks_every_required_cell(
    tmp_path, monkeypatch, mode, expected_decision, expected_reason
):
    fixture = build_fixture(tmp_path / "fixture")
    stage1 = make_nonconfirmation_plan(
        fixture, stage="stage-1", look="automatic"
    )
    stage1_argv, _, stage1_evidence = evaluator_argv(
        fixture, tmp_path / "stage1-invalid-lead", plan=stage1
    )
    assert main(stage1_argv, subprocess_runner=CountingMatch()) == 0

    import risk_score.promotion_evidence as evidence_module

    original = evidence_module.compute_paired_statistics

    def invalidate_lead(*args, **kwargs):
        report = original(*args, **kwargs)
        if kwargs.get("data_binding", {}).get("suite") == "lead-40":
            report = copy.deepcopy(report)
            if mode == "invalid":
                report["validation"]["promotion_valid"] = False
            else:
                report["metrics"]["realized_utility"]["available"] = False
        return report

    monkeypatch.setattr(
        evidence_module, "compute_paired_statistics", invalidate_lead
    )
    stage2 = make_nonconfirmation_plan(
        fixture, stage="stage-2", look="automatic"
    )
    stage2_argv, _, stage2_evidence = evaluator_argv(
        fixture,
        tmp_path / "stage2-invalid-lead",
        plan=stage2,
        stage1_evidence=stage1_evidence,
    )
    assert main(stage2_argv, subprocess_runner=CountingMatch()) == 0
    value = json.loads(stage2_evidence.read_text(encoding="utf-8"))
    assert value["stage_gate"]["decision"] == expected_decision
    assert expected_reason in value["stage_gate"]["reason_codes"]


def test_tampered_champion_path_or_hash_fails_before_publication(tmp_path):
    fixture = build_fixture(tmp_path / "fixture")
    argv, runner_map, evidence = evaluator_argv(
        fixture, tmp_path / "tampered-path"
    )
    champion_index = argv.index("--champion-model") + 1
    argv[champion_index] = str(fixture["original"])
    assert main(argv, subprocess_runner=CountingMatch()) == 2
    assert not runner_map.exists()
    assert not evidence.exists()

    argv, runner_map, evidence = evaluator_argv(
        fixture, tmp_path / "tampered-hash"
    )
    hash_index = argv.index("--champion-model-sha256") + 1
    argv[hash_index] = "f" * 64
    assert main(argv, subprocess_runner=CountingMatch()) == 2
    assert not runner_map.exists()
    assert not evidence.exists()


def test_missing_stage0_and_partial_cell_failure_fail_closed(tmp_path):
    fixture = build_fixture(tmp_path / "fixture")
    argv, runner_map, evidence = evaluator_argv(
        fixture, tmp_path / "missing-probe"
    )
    probe_path = Path(argv[argv.index("--stage0-probe") + 1])
    probe_path.unlink()
    assert main(argv, subprocess_runner=CountingMatch()) == 2
    assert not runner_map.exists()
    assert not evidence.exists()

    argv, runner_map, evidence = evaluator_argv(
        fixture, tmp_path / "partial-cell"
    )
    assert (
        main(
            argv,
            subprocess_runner=CountingMatch(fail_suite="lead-80"),
        )
        == 2
    )
    assert not runner_map.exists()
    assert not evidence.exists()
