import copy
import hashlib
import math

import pytest

from risk_score.paired_stats import (
    DEFAULT_POLICY_PATH,
    MatchValidationError,
    V1_POLICY_PATH,
    V2_POLICY_PATH,
    canonical_sha256,
    compute_paired_statistics,
    exact_zero_event_upper_bound,
    finalize_statistics_artifact,
    load_policy,
    realized_utility,
    validate_match_rows,
)


CANDIDATE = "candidate"
REFERENCE = "reference"


def result_row(
    pair_id,
    position_id,
    member,
    *,
    candidate_margin=0.0,
    outcome=0,
    schedule_id="confirmation",
    suite=None,
    metadata=None,
    no_result=False,
    resignation=False,
    turn_limit=False,
    score_missing=False,
):
    candidate_black = member == "candidate-black"
    black = CANDIDATE if candidate_black else REFERENCE
    white = REFERENCE if candidate_black else CANDIDATE
    if no_result:
        winner = None
        final_result = "Void"
        score = None
    else:
        candidate_color = "B" if candidate_black else "W"
        if outcome > 0:
            winner = candidate_color
        elif outcome < 0:
            winner = "W" if candidate_color == "B" else "B"
        else:
            winner = "draw"
        final_result = "0" if winner == "draw" else f"{winner}+{abs(candidate_margin):g}"
        score = candidate_margin if not candidate_black else -candidate_margin
        if score_missing:
            score = None
    row = {
        "schemaVersion": 1,
        "scheduleId": schedule_id,
        "gameId": f"{pair_id}:{member}",
        "seed": f"seed:{pair_id}:{member}",
        "pairId": pair_id,
        "positionId": position_id,
        "blackBot": black,
        "whiteBot": white,
        "winner": winner,
        "finalResult": final_result,
        "finalWhiteMinusBlackScore": score,
        "resignation": resignation,
        "hitTurnLimit": turn_limit,
        "noResult": no_result,
    }
    if suite is not None:
        row.update(
            {
                "suite": suite,
                "suiteBank": suite,
                "suiteBankSha256": hashlib.sha256(
                    f"suite:{suite}".encode("utf-8")
                ).hexdigest(),
                "positionContentSha256": hashlib.sha256(
                    f"content:{position_id}".encode("utf-8")
                ).hexdigest(),
                "positionSemanticSha256": hashlib.sha256(
                    f"semantic:{position_id}".encode("utf-8")
                ).hexdigest(),
            }
        )
        if metadata is None:
            metadata = suite
    if metadata is not None:
        row["metadata"] = metadata
    return row


def pair(pair_id, position_id, *, outcomes=(0, 0), margins=(0.0, 0.0), **kwargs):
    return [
        result_row(
            pair_id,
            position_id,
            "candidate-black",
            outcome=outcomes[0],
            candidate_margin=margins[0],
            **kwargs,
        ),
        result_row(
            pair_id,
            position_id,
            "candidate-white",
            outcome=outcomes[1],
            candidate_margin=margins[1],
            **kwargs,
        ),
    ]


def test_realized_utility_uses_frozen_candidate_perspective_formula():
    expected_score_term = 2.0**1.5 - 1.0
    assert realized_utility(
        1.0,
        20.0,
        win_weight=4.0,
        score_scale=20.0,
        score_power=1.5,
    ) == pytest.approx(4.0 + expected_score_term)
    assert realized_utility(
        -1.0,
        -20.0,
        win_weight=4.0,
        score_scale=20.0,
        score_power=1.5,
    ) == pytest.approx(-4.0 - expected_score_term)
    assert realized_utility(
        1.0,
        None,
        win_weight=4.0,
        score_scale=20.0,
        score_power=1.5,
    ) == 4.0


def test_pair_then_position_averaging_does_not_double_pair_utility():
    rows = []
    rows += pair("a-1", "position-a", outcomes=(1, 1), score_missing=True)
    rows += pair("a-2", "position-a", outcomes=(-1, -1), score_missing=True)
    rows += pair("b-1", "position-b", outcomes=(1, 1), score_missing=True)

    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=99,
        bootstrap_seed=7,
    )
    metric = report["metrics"]["realized_utility"]

    assert [row["value"] for row in metric["pair_values"]] == [4.0, -4.0, 4.0]
    assert [row["value"] for row in metric["position_values"]] == [0.0, 4.0]
    assert metric["estimate"] == 2.0
    assert metric["color_pairs"] == 3
    assert metric["position_clusters"] == 2
    assert report["aggregation_order"][0] == "game_to_color_pair_arithmetic_mean"


def test_repeated_pairs_do_not_increase_exact_zero_event_cluster_count():
    rows = []
    rows += pair("a-1", "position-a")
    rows += pair("a-2", "position-a")
    rows += pair("b-1", "position-b")

    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=19,
    )
    risk = report["risk_differences"]["final_50"]
    alpha = report["look"]["catastrophe_one_sided_alpha"]

    assert risk["color_pairs"] == 3
    assert risk["position_clusters"] == 2
    assert risk["zero_event_independent_position_clusters"] == 2
    assert risk["zero_event_uncertainty_upper_bound"] == pytest.approx(
        exact_zero_event_upper_bound(alpha, 2)
    )
    assert risk["zero_event_uncertainty_upper_bound"] > (
        exact_zero_event_upper_bound(alpha, 3)
    )


def test_resolved_missing_score_retains_outcome_and_reports_zero_score_term():
    rows = []
    rows += pair("p-1", "position-1", outcomes=(1, -1), score_missing=True)
    rows += pair("p-2", "position-2", outcomes=(1, 1), score_missing=True)

    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=49,
    )

    assert report["counts"]["resolved_missing_numeric_scores"] == 4
    assert report["metrics"]["realized_utility"]["pair_values"][0]["value"] == 0.0
    assert report["metrics"]["realized_utility"]["pair_values"][1]["value"] == 4.0
    assert report["metrics"]["win_rate"]["estimate"] == pytest.approx(0.75)
    assert report["risk_differences"]["final_20"]["complete"] is False
    assert report["risk_differences"]["final_20"]["classification_missing"] == 4


def test_true_no_results_are_neutral_counted_rows_not_dropped():
    rows = []
    for pair_id, position_id in (("p-1", "position-1"), ("p-2", "position-2")):
        rows.extend(
            [
                result_row(
                    pair_id,
                    position_id,
                    "candidate-black",
                    no_result=True,
                ),
                result_row(
                    pair_id,
                    position_id,
                    "candidate-white",
                    no_result=True,
                ),
            ]
        )

    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=49,
    )

    assert report["counts"]["true_no_results"] == 4
    assert report["counts"]["true_no_result_rate"] == 1.0
    assert report["metrics"]["realized_utility"]["estimate"] == 0.0
    assert report["metrics"]["win_rate"]["estimate"] == 0.5
    assert report["metrics"]["realized_utility"]["color_pairs"] == 2


@pytest.mark.parametrize(
    "mutate, expected_code",
    [
        (lambda rows: rows.pop(), "MISSING_PAIR_MEMBER"),
        (
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            "DUPLICATE_GAME_ID",
        ),
        (
            lambda rows: rows[1].update(
                {
                    "gameId": "different-id",
                    "blackBot": CANDIDATE,
                    "whiteBot": REFERENCE,
                }
            ),
            "DUPLICATE_PAIR_MEMBER",
        ),
        (
            lambda rows: rows[1].update({"positionId": "other-position"}),
            "PAIR_POSITION_MISMATCH",
        ),
        (
            lambda rows: rows[1].update({"scheduleId": "other-schedule"}),
            "PAIR_SCHEDULE_MISMATCH",
        ),
        (
            lambda rows: rows[0].pop("seed"),
            "MISSING_SEED",
        ),
        (
            lambda rows: rows[0].update({"schemaVersion": 2}),
            "UNSUPPORTED_SCHEMA_VERSION",
        ),
        (
            lambda rows: rows[1].update({"seed": rows[0]["seed"]}),
            "PAIR_SEED_COLLISION",
        ),
    ],
)
def test_structural_pair_validation_rejects_incomplete_or_duplicate_members(
    mutate, expected_code
):
    rows = pair("pair", "position")
    mutate(rows)
    validation = validate_match_rows(rows, candidate_bot=CANDIDATE)
    assert validation["promotion_valid"] is False
    assert expected_code in validation["error_codes"]
    with pytest.raises(MatchValidationError):
        compute_paired_statistics(
            rows,
            candidate_bot=CANDIDATE,
            bootstrap_replications=9,
        )


@pytest.mark.parametrize(
    "changes, expected_code",
    [
        ({"resignation": True}, "RESIGNATION"),
        ({"hitTurnLimit": True}, "TURN_LIMIT"),
        (
            {"winner": None, "finalResult": "unknown", "finalWhiteMinusBlackScore": None},
            "UNRESOLVED_ROW",
        ),
        (
            {"blackBot": REFERENCE, "whiteBot": REFERENCE},
            "TARGET_BOT_NOT_EXACTLY_ONCE",
        ),
    ],
)
def test_promotion_validity_rejects_bad_terminal_rows(changes, expected_code):
    rows = pair("pair", "position")
    rows[0].update(changes)
    validation = validate_match_rows(rows, candidate_bot=CANDIDATE)
    assert expected_code in validation["error_codes"]
    assert validation["promotion_valid"] is False


def test_terminal_validity_flags_must_be_explicit():
    rows = pair("pair", "position")
    del rows[0]["noResult"]
    validation = validate_match_rows(rows, candidate_bot=CANDIDATE)
    assert validation["promotion_valid"] is False
    assert "MISSING_TERMINAL_FLAG" in validation["error_codes"]


def test_empty_results_are_not_promotion_valid():
    validation = validate_match_rows([], candidate_bot=CANDIDATE)
    assert validation["promotion_valid"] is False
    assert validation["error_codes"] == ["NO_MATCH_ROWS"]
    with pytest.raises(MatchValidationError):
        compute_paired_statistics([], candidate_bot=CANDIDATE)


def test_position_cluster_bounds_and_bootstrap_are_seed_deterministic():
    rows = []
    for index, outcomes in enumerate(((1, 1), (1, -1), (-1, -1), (1, 0))):
        rows += pair(f"pair-{index}", f"position-{index}", outcomes=outcomes)

    first = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=199,
        bootstrap_seed=12345,
    )
    second = compute_paired_statistics(
        copy.deepcopy(rows),
        candidate_bot=CANDIDATE,
        bootstrap_replications=199,
        bootstrap_seed=12345,
    )
    first_metric = first["metrics"]["realized_utility"]
    second_metric = second["metrics"]["realized_utility"]

    assert first_metric["standard_error"] > 0.0
    assert first_metric["lower_bound"] < first_metric["estimate"]
    assert first_metric["upper_bound"] > first_metric["estimate"]
    assert first_metric["bootstrap"] == second_metric["bootstrap"]
    assert first_metric["bootstrap"]["method"] == (
        "stratified_wild_position_cluster_rademacher_"
        "centered_within_declared_strata"
    )
    assert first_metric["bootstrap"]["stratum_dimensions"] == [
        "schedule_id",
        "suite",
    ]
    assert first_metric["bootstrap"]["stratum_cluster_counts"] == [
        {
            "schedule_id": "confirmation",
            "suite": "ordinary",
            "position_clusters": 4,
        }
    ]
    assert first_metric["small_cluster_correction"]["variance_multiplier"] == pytest.approx(
        4.0 / 3.0
    )
    assert first["look"]["number"] == 1
    assert first["look"]["routine_one_sided_alpha"] == 0.01
    assert first["look"]["data_dependent_thresholds_allowed"] is False


def test_zero_catastrophe_events_have_nonzero_analytic_and_bootstrap_upper_bounds():
    rows = []
    for index in range(5):
        rows += pair(f"pair-{index}", f"position-{index}", outcomes=(0, 0), margins=(0, 0))

    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=99,
        bootstrap_seed=5,
    )
    risk = report["risk_differences"]["final_50"]

    assert risk["estimate"] == 0.0
    assert risk["candidate_events"] == 0
    assert risk["reference_events"] == 0
    expected = exact_zero_event_upper_bound(
        report["look"]["catastrophe_one_sided_alpha"],
        5,
    )
    assert risk["zero_event_uncertainty_upper_bound"] == pytest.approx(expected)
    assert risk["zero_event_independent_position_clusters"] == 5
    assert risk["upper_bound"] > 0.0
    assert risk["bootstrap"]["upper_bound"] > 0.0


def test_matched_final_trace_and_targeted_suite_risk_differences():
    rows = []
    rows += pair(
        "lead-40-a",
        "position-a",
        outcomes=(-1, -1),
        margins=(-30.0, -55.0),
        suite="lead-40",
    )
    rows += pair(
        "lead-40-b",
        "position-b",
        outcomes=(1, 1),
        margins=(30.0, 55.0),
        suite="lead-40",
    )
    rows += pair(
        "lead-80-c",
        "position-c",
        outcomes=(-1, 1),
        margins=(-90.0, 90.0),
        suite="lead-80",
    )

    moves = []
    for row in rows:
        candidate_lost = (
            row["winner"] == ("W" if row["blackBot"] == CANDIDATE else "B")
        )
        reference_lost = (
            row["winner"] == ("W" if row["blackBot"] == REFERENCE else "B")
        )
        moves.append(
            {
                "gameId": row["gameId"],
                "bot": CANDIDATE,
                "scoreLead": 85.0 if candidate_lost else 0.0,
                "winProbability": 0.97 if candidate_lost else 0.5,
            }
        )
        moves.append(
            {
                "gameId": row["gameId"],
                "bot": REFERENCE,
                "scoreLead": 85.0 if reference_lost else 0.0,
                "winProbability": 0.97 if reference_lost else 0.5,
            }
        )

    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        move_records=moves,
        bootstrap_replications=99,
    )

    final_20 = report["risk_differences"]["final_20"]
    assert final_20["candidate_events"] == 3
    assert final_20["reference_events"] == 3
    assert final_20["estimate"] == 0.0
    assert report["risk_differences"]["lead_80_loss"]["estimate"] == 0.0
    assert report["risk_differences"]["high_confidence_loss"]["estimate"] == 0.0
    assert (
        report["risk_differences"]["targeted_lead_40_suite_loss"]["estimate"]
        == 0.0
    )
    assert report["suite_metrics"]["combined_lead"]["color_pairs"] == 3
    assert report["move_trace_coverage"]["complete"] is True


def test_explicit_trace_flag_cannot_hide_a_derived_loss():
    rows = pair(
        "lead-loss",
        "position-loss",
        outcomes=(-1, -1),
        margins=(-10.0, -10.0),
    )
    rows[0]["riskFlags"] = {
        "candidate": {"lead_40_loss": False},
        "reference": {"lead_40_loss": False},
    }
    moves = []
    for row in rows:
        moves.extend(
            [
                {
                    "gameId": row["gameId"],
                    "bot": CANDIDATE,
                    "scoreLead": 45.0,
                    "winProbability": 0.6,
                },
                {
                    "gameId": row["gameId"],
                    "bot": REFERENCE,
                    "scoreLead": 0.0,
                    "winProbability": 0.4,
                },
            ]
        )
    with pytest.raises(ValueError, match="conflicts with trace"):
        compute_paired_statistics(
            rows,
            candidate_bot=CANDIDATE,
            move_records=moves,
            bootstrap_replications=9,
        )


def test_explicit_targeted_flag_requires_frozen_suite_and_matches_outcome():
    ordinary = pair("ordinary", "position-ordinary", outcomes=(-1, -1))
    ordinary[0]["riskFlags"] = {
        "candidate": {"targeted_lead_40_suite_loss": True},
        "reference": {"targeted_lead_40_suite_loss": False},
    }
    with pytest.raises(ValueError, match="not backed by frozen schedule"):
        compute_paired_statistics(
            ordinary,
            candidate_bot=CANDIDATE,
            bootstrap_replications=9,
        )

    targeted = pair(
        "targeted",
        "position-targeted",
        outcomes=(-1, -1),
        suite="lead-40",
    )
    targeted[0]["riskFlags"] = {
        "candidate": {"targeted_lead_40_suite_loss": False},
        "reference": {"targeted_lead_40_suite_loss": False},
    }
    with pytest.raises(ValueError, match="conflicts with outcome"):
        compute_paired_statistics(
            targeted,
            candidate_bot=CANDIDATE,
            bootstrap_replications=9,
        )


def test_finalized_statistics_emit_hash_bound_artifact_manifest():
    rows = []
    for index in range(3):
        rows += pair(
            f"pair-{index}",
            f"position-{index}",
            outcomes=(1, 0),
            score_missing=True,
        )
    binding = {
        "candidate_hash": "1" * 64,
        "reference_hash": "2" * 64,
        "comparison": "candidate-vs-champion-powered",
        "suite": "confirmation",
        "suite_hash": "3" * 64,
        "schedule_id": "confirmation-look-1",
        "schedule_hash": "4" * 64,
        "config_hash": "5" * 64,
        "runner_manifest_hash": "6" * 64,
        "execution_hash": "7" * 64,
        "katago_binary_hash": "8" * 64,
    }
    report = compute_paired_statistics(
        rows,
        candidate_bot=CANDIDATE,
        bootstrap_replications=9,
        finalized=True,
        data_binding=binding,
    )
    finalized = finalize_statistics_artifact(
        report,
        cell_name="powered_candidate_vs_champion",
        metric_names=["realized_utility", "win_rate"],
    )
    assert finalized["statistics_artifact_hash"] == canonical_sha256(
        finalized["statistics_artifact"]
    )
    assert finalized["statistics_manifest_hash"] == canonical_sha256(
        finalized["statistics_manifest"]
    )
    assert finalized["statistics_manifest"]["position_ids"] == [
        "position-0",
        "position-1",
        "position-2",
    ]
    assert finalized["statistics_manifest"]["metric_names"] == [
        "realized_utility",
        "win_rate",
    ]


def test_policy_objective_and_sequential_look_values_are_loaded_from_json():
    policy = load_policy()
    assert DEFAULT_POLICY_PATH == V2_POLICY_PATH
    assert V1_POLICY_PATH != V2_POLICY_PATH
    assert policy["schema_version"] == 2
    assert policy["policy_version"] == "risk-seeking-checkpoint-promotion-v2"
    assert policy["objective"] == {
        "name": "candidate_realized_powered_terminal_utility",
        "candidate_perspective_outcomes": {
            "win": 1.0,
            "draw": 0.0,
            "loss": -1.0,
            "true_no_result": 0.0,
        },
        "ordinary_win_score": {
            "win": 1.0,
            "draw": 0.5,
            "loss": 0.0,
            "true_no_result": 0.5,
        },
        "win_weight": 4.0,
        "score_power": 1.5,
        "score_scale": 20.0,
        "missing_numeric_score_term": 0.0,
        "color_pair_aggregation": "arithmetic_mean_of_two_games",
        "repeated_pair_aggregation": "arithmetic_mean_within_position_id",
    }
    assert policy["confidence"]["sequential_testing"]["look_count"] == 2
    assert math.isclose(
        sum(
            look["catastrophe_one_sided_alpha"]
            for look in policy["confidence"]["sequential_testing"]["looks"]
        ),
        0.01,
    )
