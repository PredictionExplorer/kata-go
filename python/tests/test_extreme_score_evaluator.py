import copy
import json
from pathlib import Path

import pytest
from risk_score.extreme_score_evaluator import (
    DEFAULT_POLICY_PATH,
    PLAN_REQUEST_CONTRACT,
    ExtremeScoreEvaluatorError,
    ExtremeScoreIntegrityError,
    build_plan,
    build_runner_jobs,
    canonical_json,
    canonical_sha256,
    evaluate_expected_max,
    evaluate_plan_file,
    evaluation_status,
    file_sha256,
    load_plan,
    load_report,
    main,
    publish_immutable_json,
)

CANDIDATE_HASH = "a" * 64
CONFIG_HASH = "c" * 64
OPPONENT_HASH = "d" * 64


def write_policy(tmp_path, *, minimum_clusters=8):
    policy = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    policy["inference"]["minimum_clusters"] = minimum_clusters
    path = tmp_path / "expected-max-policy.json"
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return path


def plan_request(
    tmp_path,
    *,
    clusters=8,
    cohort_size=2,
    legal_score_bounds=None,
):
    reference_artifact = tmp_path / "reference-model.bin"
    checkpoint_artifact = tmp_path / "reference-checkpoint.bin"
    if not reference_artifact.exists():
        reference_artifact.write_bytes(b"frozen reference model\n")
    if not checkpoint_artifact.exists():
        checkpoint_artifact.write_bytes(b"frozen trainer checkpoint\n")
    reference_hash = file_sha256(reference_artifact)
    checkpoint_hash = file_sha256(checkpoint_artifact)
    cohorts = []
    for cohort_index in range(2 * clusters):
        cluster_index = cohort_index // 2
        cohorts.append(
            {
                "cohort_id": f"cohort-{cohort_index:02d}",
                "cluster_id": f"cluster-{cluster_index:02d}",
                "league_cell": f"league-cell-{cluster_index:02d}",
                "opponent_snapshot_id": f"opponent-{cluster_index:02d}",
                "opponent_model_sha256": OPPONENT_HASH,
                "focal_color": "B" if cohort_index % 2 == 0 else "W",
                "seeds": [
                    f"seed-{cohort_index:02d}-{trial_index:02d}"
                    for trial_index in range(cohort_size)
                ],
            }
        )
    return {
        "schema_version": 1,
        "contract": PLAN_REQUEST_CONTRACT,
        "candidate_model": {
            "model_id": "candidate-model",
            "sha256": CANDIDATE_HASH,
        },
        "reference_model": {
            "model_id": "reference-model",
            "sha256": reference_hash,
        },
        "config": {
            "config_id": "frozen-extreme-score-config",
            "sha256": CONFIG_HASH,
        },
        "cohort_size": cohort_size,
        "legal_score_bounds": legal_score_bounds
        or {"minimum": -200.0, "maximum": 200.0},
        "cohorts": cohorts,
        "rollback_recommendation": {
            "action": "retain_reference",
            "reference_model": {
                "model_id": "reference-model",
                "sha256": reference_hash,
            },
            "reference_model_artifact": {
                "path": str(reference_artifact.resolve()),
                "file_sha256": reference_hash,
            },
            "trainer_checkpoint_artifact": {
                "path": str(checkpoint_artifact.resolve()),
                "file_sha256": checkpoint_hash,
            },
            "quarantine_candidate_on_failure": True,
        },
    }


def make_plan(tmp_path, *, clusters=8, cohort_size=2, minimum_clusters=8):
    policy_path = write_policy(
        tmp_path,
        minimum_clusters=minimum_clusters,
    )
    return build_plan(
        plan_request(tmp_path, clusters=clusters, cohort_size=cohort_size),
        policy_path=policy_path,
    )


def completed_rows(plan, arm, maxima):
    if isinstance(maxima, (int, float)):
        maxima = {cohort["cohort_id"]: float(maxima) for cohort in plan["cohorts"]}
    rows = []
    for job in build_runner_jobs(plan, arm):
        maximum = float(maxima[job["cohort_id"]])
        rows.append(
            {
                **job,
                "score": maximum - job["trial_index"],
                "no_result": False,
                "hit_turn_limit": False,
            }
        )
    return rows


def evaluate_maxima(plan, candidate_maxima, reference_maxima, **kwargs):
    return evaluate_expected_max(
        plan,
        completed_rows(plan, "candidate", candidate_maxima),
        completed_rows(plan, "reference", reference_maxima),
        **kwargs,
    )


def write_canonical(path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_raw_lifetime_record_cannot_trigger_promotion(tmp_path):
    plan = make_plan(tmp_path)
    candidate_maxima = {
        cohort["cohort_id"]: 100.0 if index == 0 else 0.0
        for index, cohort in enumerate(plan["cohorts"])
    }
    reference_maxima = {cohort["cohort_id"]: 20.0 for cohort in plan["cohorts"]}

    report = evaluate_maxima(
        plan,
        candidate_maxima,
        reference_maxima,
        raw_lifetime_records={
            "candidate": {"maximum": 10_000.0},
            "reference": {"maximum": 50.0},
        },
    )

    assert report["statistics"]["overall"]["candidate_J_N"] == pytest.approx(
        100.0 / len(plan["cohorts"])
    )
    assert report["statistics"]["overall"]["reference_J_N"] == 20.0
    assert report["decision"] == "FAIL"
    assert report["promotion_recommended"] is False
    assert report["non_gating_diagnostics"] == {
        "raw_lifetime_records": {
            "candidate": {"maximum": 10_000.0},
            "reference": {"maximum": 50.0},
        },
        "used_for_decision": False,
    }


def test_candidate_reference_maxima_are_paired_by_precommitted_cohort(tmp_path):
    plan = make_plan(tmp_path)
    cohort_ids = [cohort["cohort_id"] for cohort in plan["cohorts"]]
    candidate_maxima = {
        cohort_id: 10.0 + index for index, cohort_id in enumerate(cohort_ids)
    }
    reference_maxima = {
        cohort_id: 9.0 + 2 * index for index, cohort_id in enumerate(cohort_ids)
    }
    candidate_rows = completed_rows(plan, "candidate", candidate_maxima)
    reference_rows = list(reversed(completed_rows(plan, "reference", reference_maxima)))

    report = evaluate_expected_max(plan, candidate_rows, reference_rows)

    assert [row["cohort_id"] for row in report["cohort_maxima"]] == cohort_ids
    assert [row["paired_delta"] for row in report["cohort_maxima"]] == [
        candidate_maxima[cohort_id] - reference_maxima[cohort_id]
        for cohort_id in cohort_ids
    ]
    assert report["statistics"]["overall"]["paired_delta_estimate"] == pytest.approx(
        sum(
            candidate_maxima[cohort_id] - reference_maxima[cohort_id]
            for cohort_id in cohort_ids
        )
        / len(cohort_ids)
    )


def test_plan_requires_fixed_n_alternating_balanced_color_clusters(tmp_path):
    policy = write_policy(tmp_path)
    short = plan_request(tmp_path)
    short["cohorts"][0]["seeds"].pop()
    with pytest.raises(ExtremeScoreEvaluatorError, match="exactly 2 seeds"):
        build_plan(short, policy_path=policy)

    unbalanced = plan_request(tmp_path)
    unbalanced["cohorts"][1]["focal_color"] = "B"
    with pytest.raises(ExtremeScoreEvaluatorError, match="alternate|balanced"):
        build_plan(unbalanced, policy_path=policy)

    changed_opponent = plan_request(tmp_path)
    changed_opponent["cohorts"][1]["opponent_snapshot_id"] = "other-opponent"
    with pytest.raises(ExtremeScoreEvaluatorError, match="frozen opponent"):
        build_plan(changed_opponent, policy_path=policy)


@pytest.mark.parametrize(
    "malformation, expected_code",
    [
        ("missing", "MISSING_RESULT"),
        ("duplicate", "DUPLICATE_RESULT"),
        ("seed", "IDENTITY_MISMATCH"),
        ("config", "IDENTITY_MISMATCH"),
        ("opponent", "IDENTITY_MISMATCH"),
        ("model", "IDENTITY_MISMATCH"),
        ("cohort", "UNPLANNED_RESULT"),
        ("no-result", "NO_RESULT"),
        ("turn-limit", "TURN_LIMIT"),
        ("nonfinite", "INVALID_SCORE"),
        ("illegal", "ILLEGAL_SCORE"),
    ],
)
def test_malformed_or_incomplete_cohorts_fail_integrity(
    tmp_path, malformation, expected_code
):
    plan = make_plan(tmp_path)
    candidate_rows = completed_rows(plan, "candidate", 20.0)
    reference_rows = completed_rows(plan, "reference", 10.0)
    if malformation == "missing":
        candidate_rows.pop()
    elif malformation == "duplicate":
        candidate_rows.append(copy.deepcopy(candidate_rows[0]))
    elif malformation == "seed":
        candidate_rows[0]["seed"] = "wrong-seed"
    elif malformation == "config":
        candidate_rows[0]["config_sha256"] = "0" * 64
    elif malformation == "opponent":
        candidate_rows[0]["opponent_snapshot_id"] = "wrong-opponent"
    elif malformation == "model":
        candidate_rows[0]["model_sha256"] = plan["reference_model"]["sha256"]
    elif malformation == "cohort":
        candidate_rows[0]["cohort_id"] = "unplanned-cohort"
    elif malformation == "no-result":
        candidate_rows[0]["no_result"] = True
    elif malformation == "turn-limit":
        candidate_rows[0]["hit_turn_limit"] = True
    elif malformation == "nonfinite":
        candidate_rows[0]["score"] = float("nan")
    elif malformation == "illegal":
        candidate_rows[0]["score"] = 201.0

    report = evaluate_expected_max(plan, candidate_rows, reference_rows)

    assert report["decision"] == "FAIL"
    assert report["promotion_recommended"] is False
    assert report["integrity"]["valid"] is False
    assert expected_code in report["integrity"]["issue_codes"]
    assert report["statistics"] == {}
    assert report["cohort_maxima"] == []


def test_exact_cluster_sign_test_is_deterministic_and_order_independent(tmp_path):
    plan = make_plan(tmp_path)
    cluster_deltas = [-6.0, -6.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0]
    deltas = [delta for delta in cluster_deltas for _color in ("B", "W")]
    reference_maxima = {cohort["cohort_id"]: 20.0 for cohort in plan["cohorts"]}
    candidate_maxima = {
        cohort["cohort_id"]: 20.0 + deltas[index]
        for index, cohort in enumerate(plan["cohorts"])
    }
    candidate_rows = completed_rows(plan, "candidate", candidate_maxima)
    reference_rows = completed_rows(plan, "reference", reference_maxima)

    first = evaluate_expected_max(plan, candidate_rows, reference_rows)
    second = evaluate_expected_max(
        plan, reversed(candidate_rows), reversed(reference_rows)
    )

    assert first["statistics"] == second["statistics"]
    assert (
        first["statistics"]["overall"]["inference"]["cluster_effects_sha256"]
        == second["statistics"]["overall"]["inference"]["cluster_effects_sha256"]
    )
    assert first["statistics"]["overall"]["inference"]["inference_unit"] == (
        "precommitted_black_white_cluster"
    )
    assert first["statistics"]["overall"]["inference"]["method"] == (
        "exact_paired_cluster_sign_test"
    )


@pytest.mark.parametrize(
    "cluster_deltas, expected_decision",
    [
        ([2.0] * 8, "PASS"),
        ([-1.0] * 8, "FAIL"),
        ([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "INCONCLUSIVE"),
    ],
)
def test_decision_is_pass_fail_or_inconclusive(
    tmp_path, cluster_deltas, expected_decision
):
    plan = make_plan(tmp_path)
    reference_maxima = {cohort["cohort_id"]: 20.0 for cohort in plan["cohorts"]}
    candidate_maxima = {}
    for index, cohort in enumerate(plan["cohorts"]):
        candidate_maxima[cohort["cohort_id"]] = 20.0 + cluster_deltas[index // 2]

    report = evaluate_maxima(plan, candidate_maxima, reference_maxima)

    assert report["decision"] == expected_decision
    assert report["promotion_recommended"] is (expected_decision == "PASS")
    if expected_decision == "PASS":
        assert all(
            report["statistics"][name]["one_sided_p_value"]
            <= report["statistics"][name]["one_sided_alpha"]
            for name in ("overall", "B", "W")
        )
    elif expected_decision == "INCONCLUSIVE":
        assert report["statistics"]["overall"]["paired_delta_estimate"] > 0.0
        assert (
            report["statistics"]["overall"]["one_sided_p_value"]
            > report["statistics"]["overall"]["one_sided_alpha"]
        )


def test_too_few_clusters_is_inconclusive_not_promotable(tmp_path):
    plan = make_plan(tmp_path, clusters=4)
    report = evaluate_maxima(plan, 20.0, 10.0)

    assert report["decision"] == "INCONCLUSIVE"
    assert report["promotion_recommended"] is False
    assert all(
        not report["statistics"][name]["inference_available"]
        for name in ("overall", "B", "W")
    )


def test_exact_sign_gate_has_valid_resolution_and_rejects_four_cluster_policy(
    tmp_path,
):
    underpowered_policy = write_policy(tmp_path, minimum_clusters=4)
    with pytest.raises(ExtremeScoreEvaluatorError, match="at least 8"):
        build_plan(plan_request(tmp_path), policy_path=underpowered_policy)

    plan = make_plan(tmp_path)
    report = evaluate_maxima(plan, 20.0, 10.0)

    assert report["decision"] == "PASS"
    for name in ("overall", "B", "W"):
        statistic = report["statistics"][name]
        assert statistic["informative_clusters"] == 8
        assert statistic["positive_clusters"] == 8
        assert statistic["one_sided_p_value"] == pytest.approx(1.0 / 256.0)


def test_legal_score_bounds_are_transformed_to_focal_color(tmp_path):
    policy_path = write_policy(tmp_path)
    request = plan_request(
        tmp_path,
        legal_score_bounds={"minimum": -10.0, "maximum": 100.0},
    )
    plan = build_plan(request, policy_path=policy_path)
    legal_maxima = {
        cohort["cohort_id"]: -99.0 if cohort["focal_color"] == "B" else 100.0
        for cohort in plan["cohorts"]
    }

    legal = evaluate_maxima(plan, legal_maxima, legal_maxima)
    assert legal["integrity"]["valid"] is True

    candidate_rows = completed_rows(plan, "candidate", 0.0)
    reference_rows = completed_rows(plan, "reference", 0.0)
    black_row = next(row for row in candidate_rows if row["focal_color"] == "B")
    white_row = next(row for row in candidate_rows if row["focal_color"] == "W")
    black_row["score"] = 11.0
    white_row["score"] = -11.0

    illegal = evaluate_expected_max(plan, candidate_rows, reference_rows)

    illegal_issues = [
        issue
        for issue in illegal["integrity"]["issues"]
        if issue["code"] == "ILLEGAL_SCORE"
    ]
    assert len(illegal_issues) == 2
    assert any("B-focal" in issue["detail"] for issue in illegal_issues)
    assert any("W-focal" in issue["detail"] for issue in illegal_issues)


def test_rollback_recommendation_is_strict_and_artifact_bound(tmp_path):
    policy_path = write_policy(tmp_path)
    request = plan_request(tmp_path)

    missing_key = copy.deepcopy(request)
    missing_key["rollback_recommendation"].pop("reference_model_artifact")
    with pytest.raises(ExtremeScoreEvaluatorError, match="keys differ"):
        build_plan(missing_key, policy_path=policy_path)

    wrong_reference = copy.deepcopy(request)
    wrong_reference["rollback_recommendation"]["reference_model"]["sha256"] = (
        CANDIDATE_HASH
    )
    with pytest.raises(ExtremeScoreEvaluatorError, match="exactly bind"):
        build_plan(wrong_reference, policy_path=policy_path)

    missing_checkpoint = copy.deepcopy(request)
    missing_checkpoint["rollback_recommendation"]["trainer_checkpoint_artifact"][
        "path"
    ] = str((tmp_path / "missing-checkpoint.bin").resolve())
    with pytest.raises(ExtremeScoreEvaluatorError, match="regular non-symlink"):
        build_plan(missing_checkpoint, policy_path=policy_path)

    plan = build_plan(request, policy_path=policy_path)
    plan_path = tmp_path / "artifact-bound-plan.json"
    write_canonical(plan_path, plan)
    checkpoint_path = Path(
        plan["rollback_recommendation"]["trainer_checkpoint_artifact"]["path"]
    )
    checkpoint_path.write_bytes(b"changed checkpoint\n")
    with pytest.raises(ExtremeScoreEvaluatorError, match="missing or changed"):
        load_plan(plan_path)


def test_transient_runner_failure_does_not_finalize_canonical_report(tmp_path):
    plan = make_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    write_canonical(plan_path, plan)
    failed_once = False

    def flaky_runner(arm, jobs):
        nonlocal failed_once
        if arm == "candidate" and not failed_once:
            failed_once = True
            raise OSError("temporary runner outage")
        maximum = 30.0 if arm == "candidate" else 20.0
        return [
            {
                **job,
                "score": maximum - job["trial_index"],
                "no_result": False,
                "hit_turn_limit": False,
            }
            for job in jobs
        ]

    with pytest.raises(ExtremeScoreIntegrityError, match="RUNNER_ERROR"):
        evaluate_plan_file(plan_path, report_path, runner=flaky_runner)
    assert not report_path.exists()
    assert not list(tmp_path.glob("report.json.*.results.jsonl"))

    report = evaluate_plan_file(plan_path, report_path, runner=flaky_runner)

    assert report["decision"] == "PASS"
    assert load_report(report_path) == report


def test_rehashed_forged_pass_is_rejected_by_full_recomputation(tmp_path):
    plan = make_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    forged_path = tmp_path / "forged-report.json"
    write_canonical(plan_path, plan)

    def losing_runner(arm, jobs):
        maximum = 10.0 if arm == "candidate" else 20.0
        return [
            {
                **job,
                "score": maximum - job["trial_index"],
                "no_result": False,
                "hit_turn_limit": False,
            }
            for job in jobs
        ]

    genuine = evaluate_plan_file(plan_path, report_path, runner=losing_runner)
    assert genuine["decision"] == "FAIL"
    forged = copy.deepcopy(genuine)
    fabricated_pass = evaluate_maxima(plan, 30.0, 20.0)
    for key in (
        "cohort_maxima",
        "statistics",
        "decision",
        "reason_codes",
        "promotion_recommended",
    ):
        forged[key] = copy.deepcopy(fabricated_pass[key])
    forged["report_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "report_sha256"}
    )
    write_canonical(forged_path, forged)
    forged_path.chmod(0o444)

    with pytest.raises(
        ExtremeScoreEvaluatorError, match="deterministic recomputation"
    ):
        load_report(forged_path)


def test_immutable_publication_requires_identical_read_only_file(tmp_path):
    artifact = tmp_path / "immutable.json"
    value = {"answer": 42}

    publish_immutable_json(artifact, value)

    assert artifact.stat().st_mode & 0o222 == 0
    publish_immutable_json(artifact, value)
    artifact.chmod(0o644)
    with pytest.raises(ExtremeScoreEvaluatorError, match="read-only"):
        publish_immutable_json(artifact, value)
    artifact.chmod(0o444)
    with pytest.raises(ExtremeScoreEvaluatorError, match="bytes or SHA-256"):
        publish_immutable_json(artifact, {"answer": 43})


def test_cli_plan_evaluate_status_and_injected_runner_are_hash_bound(tmp_path, capsys):
    policy_path = write_policy(tmp_path)
    request_path = tmp_path / "request.json"
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    request_path.write_text(
        json.dumps(plan_request(tmp_path), indent=2) + "\n", encoding="utf-8"
    )

    assert (
        main(
            [
                "plan",
                "--spec",
                str(request_path),
                "--policy",
                str(policy_path),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    plan = load_plan(plan_path)
    calls = []

    def runner(arm, jobs):
        calls.append((arm, len(jobs)))
        maximum = 30.0 if arm == "candidate" else 20.0
        return [
            {
                **job,
                "score": maximum - job["trial_index"],
                "no_result": False,
                "hit_turn_limit": False,
            }
            for job in jobs
        ]

    assert (
        main(
            [
                "evaluate",
                "--plan",
                str(plan_path),
                "--output",
                str(report_path),
            ],
            runner=runner,
        )
        == 0
    )
    capsys.readouterr()
    report = load_report(report_path)
    assert calls == [("candidate", 32), ("reference", 32)]
    assert report["decision"] == "PASS"
    assert plan_path.stat().st_mode & 0o222 == 0
    assert report_path.stat().st_mode & 0o222 == 0
    assert report["plan_binding"]["file_sha256"] == file_sha256(plan_path)
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert report["rollback_recommendation"] == plan["rollback_recommendation"]
    assert report["rollback_recommendation_sha256"] == canonical_sha256(
        plan["rollback_recommendation"]
    )

    assert (
        main(
            [
                "status",
                "--plan",
                str(plan_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output == evaluation_status(plan_path, report_path=report_path)
    assert output["state"] == "EVALUATED"
    assert output["decision"] == "PASS"
    assert output["status_sha256"] == canonical_sha256(
        {key: value for key, value in output.items() if key != "status_sha256"}
    )


def test_jsonl_cli_runner_needs_no_gpu_and_binds_input_files(tmp_path, capsys):
    policy_path = write_policy(tmp_path)
    request_path = tmp_path / "request.json"
    plan_path = tmp_path / "plan.json"
    candidate_path = tmp_path / "candidate.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    report_path = tmp_path / "report.json"
    request_path.write_text(
        json.dumps(plan_request(tmp_path), indent=2) + "\n", encoding="utf-8"
    )
    assert (
        main(
            [
                "plan",
                "--spec",
                str(request_path),
                "--policy",
                str(policy_path),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    plan = load_plan(plan_path)
    write_jsonl(candidate_path, completed_rows(plan, "candidate", 30.0))
    write_jsonl(reference_path, completed_rows(plan, "reference", 20.0))

    assert (
        main(
            [
                "evaluate",
                "--plan",
                str(plan_path),
                "--candidate-results",
                str(candidate_path),
                "--reference-results",
                str(reference_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    report = load_report(report_path)
    for arm, original_path in (
        ("candidate", candidate_path),
        ("reference", reference_path),
    ):
        source = report["result_bindings"][arm]["source"]
        snapshot_path = Path(source["path"])
        assert snapshot_path != original_path.resolve()
        assert source["file_sha256"] == file_sha256(snapshot_path)
        assert snapshot_path.stat().st_mode & 0o222 == 0
