import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def assignments(path):
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            assert key not in values
            values[key] = value
    return values


def test_extreme_selfplay_is_score_only_focal_and_cohort_bound():
    values = assignments(
        REPO / "cpp/configs/risk_score/extreme_score_selfplay_19x19.cfg"
    )

    assert values["useExpectedMaxScoreUtility"] == "true"
    assert values["useScoreMaximizingUtility"] == "false"
    assert values["winWeight"] == "0"
    assert values["extremeCohortSize"] == values["extremeScoreGroupSize"] == "1"
    assert values["extremeCohortFocalColor"] == values["expectedMaxFocalColor"]
    assert values["switchNetsMidGame"] == "false"
    assert values["bSizes"] == "19"
    assert values["komiMean"] == "7.5"
    assert values["handicapProb"] == "0"
    assert values["rootNoiseEnabled"] == "true"
    assert values["nnRandomize"] == "false"
    assert values["policyOptimism"] == values["rootPolicyOptimism"] == "0"


def test_extreme_match_uses_identical_stochastic_attempt_policy():
    selfplay = assignments(
        REPO / "cpp/configs/risk_score/extreme_score_selfplay_19x19.cfg"
    )
    match = assignments(
        REPO / "cpp/configs/risk_score/extreme_score_match_19x19.cfg"
    )

    shared = (
        "bSizes",
        "komiMean",
        "handicapProb",
        "chosenMoveTemperatureEarly",
        "chosenMoveTemperatureHalflife",
        "chosenMoveTemperature",
        "useLcbForSelection",
        "rootNoiseEnabled",
        "rootDirichletNoiseTotalConcentration",
        "rootDirichletNoiseWeight",
        "rootPolicyTemperatureEarly",
        "rootPolicyTemperature",
        "rootNumSymmetriesToSample",
        "nnRandomize",
    )
    assert {key: selfplay[key] for key in shared} == {
        key: match[key] for key in shared
    }
    assert match["allowResignation"] == "false"
    assert match["useExpectedMaxScoreUtility"] == "true"
    assert match["winWeight"] == "0"


def test_extreme_training_policy_has_no_win_objective_and_caps_n_at_eight():
    policy = json.loads(
        (
            REPO / "python/risk_score/extreme_score_training_policy_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert policy["status"] == "frozen"
    assert policy["objective"]["win_loss_weight"] == 0
    assert policy["objective"]["cohort_sizes"] == [1, 2, 4, 8]
    assert policy["objective"]["production_cohort_size"] == 8
    assert policy["objective"]["maximum_supported_cohort_size"] == 8
    assert policy["league"]["opponent_gradient"] == "stopped"
    assert policy["league"]["focal_colors"] == ["B", "W"]
    assert sum(policy["league"]["opponent_weights"].values()) == 1.0
    assert policy["training"]["extreme_score_only"] is True
    assert policy["training"]["allow_mixed_legacy_rows"] is False
    assert policy["evaluation"]["raw_lifetime_record_is_gating"] is False
