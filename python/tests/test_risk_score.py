import json

import pytest

from risk_score.compare_moves import compare_analyses
from risk_score.generate_schedule import build_schedule, load_positions, write_schedule
from risk_score.summarize_matches import summarize_matches


def position(metadata="ordinary"):
    return {
        "xSize": 19,
        "ySize": 19,
        "board": "/".join(["." * 19] * 19),
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": 0,
        "hintLoc": "null",
        "weight": 1.0,
        "metadata": metadata,
    }


def result(
    game_id,
    *,
    black,
    white,
    winner,
    score,
    final_result,
    resignation=False,
    no_result=False,
    turn_limit=False,
):
    return {
        "schemaVersion": 1,
        "scheduleId": "schedule",
        "gameId": game_id,
        "seed": f"seed-{game_id}",
        "blackBot": black,
        "whiteBot": white,
        "winner": winner,
        "finalWhiteMinusBlackScore": score,
        "finalResult": final_result,
        "resignation": resignation,
        "noResult": no_result,
        "hitTurnLimit": turn_limit,
    }


def test_schedule_is_stable_and_color_paired(tmp_path):
    positions = [position("ordinary"), position("lead-40")]
    first = build_schedule(
        positions,
        bot_a_index=2,
        bot_b_index=5,
        pairs_per_position=2,
        base_seed="experiment-a",
    )
    second = build_schedule(
        positions,
        bot_a_index=2,
        bot_b_index=5,
        pairs_per_position=2,
        base_seed="experiment-a",
    )

    assert first == second
    assert len(first) == 8
    assert len({row["gameId"] for row in first}) == 8
    assert len({row["scheduleId"] for row in first}) == 1

    for index in range(0, len(first), 2):
        bot_a_black, bot_a_white = first[index : index + 2]
        assert (bot_a_black["blackBot"], bot_a_black["whiteBot"]) == (2, 5)
        assert (bot_a_white["blackBot"], bot_a_white["whiteBot"]) == (5, 2)
        assert bot_a_black["pairId"] == bot_a_white["pairId"]
        assert bot_a_black["seed"] != bot_a_white["seed"]
        assert bot_a_black["startPosition"] == bot_a_white["startPosition"]

    output = tmp_path / "schedule.jsonl"
    assert write_schedule(first, str(output)) == len(first)
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert written == first


def test_schedule_changes_when_seed_changes():
    positions = [position()]
    first = build_schedule(positions, base_seed="a")
    second = build_schedule(positions, base_seed="b")
    assert first[0]["scheduleId"] != second[0]["scheduleId"]
    assert first[0]["seed"] != second[0]["seed"]


def test_load_positions_rejects_non_position(tmp_path):
    path = tmp_path / "positions.jsonl"
    path.write_text('{"xSize":19}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing PositionSample keys"):
        load_positions([path])


def test_summarize_match_metrics_and_terminal_utility():
    records = [
        result(
            "win-white",
            black="baseline",
            white="target",
            winner="W",
            score=50.0,
            final_result="W+50",
        ),
        result(
            "loss-black",
            black="target",
            white="baseline",
            winner="W",
            score=45.0,
            final_result="W+45",
        ),
        result(
            "win-black",
            black="target",
            white="baseline",
            winner="B",
            score=-10.0,
            final_result="B+10",
        ),
        result(
            "draw",
            black="baseline",
            white="target",
            winner="draw",
            score=0.0,
            final_result="0",
        ),
        result(
            "resign-loss",
            black="target",
            white="baseline",
            winner="W",
            score=None,
            final_result="W+R",
            resignation=True,
        ),
        result(
            "no-result",
            black="baseline",
            white="target",
            winner=None,
            score=None,
            final_result="Void",
            no_result=True,
        ),
        result(
            "turn-limit",
            black="target",
            white="baseline",
            winner=None,
            score=None,
            final_result="turn_limit",
            turn_limit=True,
        ),
    ]

    summary = summarize_matches(
        records,
        target_bot="target",
        catastrophe_thresholds=[40.0, 80.0],
        probabilities=[0.5],
        score_power=2.0,
        score_scale=20.0,
        win_weight=1.0,
        top_n=2,
    )

    assert summary["games"] == {
        "total": 7,
        "resolved_for_win_rate": 5,
        "wins": 2,
        "losses": 2,
        "draws": 1,
        "no_results": 1,
        "turn_limits": 1,
        "unresolved": 0,
        "resignations": 1,
    }
    assert summary["ordinary_win_rate"] == pytest.approx(0.5)
    assert summary["ordinary_win_rate_interval_95"][0] < 0.5
    assert summary["ordinary_win_rate_interval_95"][1] > 0.5
    assert summary["final_margin"]["count"] == 4
    assert summary["final_margin"]["mean"] == pytest.approx(3.75)
    assert summary["final_margin"]["median"] == pytest.approx(5.0)
    assert summary["final_margin"]["largest_wins"][0]["game_id"] == "win-white"
    assert summary["final_margin"]["largest_losses"][0]["game_id"] == "loss-black"

    catastrophes = summary["final_margin_catastrophes"]["thresholds"]
    assert catastrophes[0] == {
        "threshold_points": 40.0,
        "count": 1,
        "rate": pytest.approx(0.25),
    }
    assert catastrophes[1] == {
        "threshold_points": 80.0,
        "count": 0,
        "rate": pytest.approx(0.0),
    }
    assert summary["led_then_lost"]["available"] is False
    assert summary["terminal_custom_utility"]["count"] == 4
    assert summary["terminal_custom_utility"]["mean"] == pytest.approx(0.984375)
    assert summary["terminal_custom_utility"]["excluded_without_numeric_margin"] == 3


def test_summarizer_requires_target_exactly_once():
    records = [
        result(
            "bad",
            black="baseline",
            white="baseline",
            winner="B",
            score=-1.0,
            final_result="B+1",
        )
    ]
    with pytest.raises(ValueError, match="exactly once"):
        summarize_matches(records, target_bot="target")


def test_summarizer_reports_led_then_lost_from_move_traces():
    records = [
        result(
            "lost-after-lead",
            black="target",
            white="baseline",
            winner="W",
            score=5.0,
            final_result="W+5",
        ),
        result(
            "ordinary-win",
            black="baseline",
            white="target",
            winner="W",
            score=3.0,
            final_result="W+3",
        ),
    ]
    moves = [
        {
            "gameId": "lost-after-lead",
            "bot": "target",
            "scoreLead": 82.0,
            "winProbability": 0.97,
        },
        {
            "gameId": "ordinary-win",
            "bot": "target",
            "scoreLead": 4.0,
            "winProbability": 0.70,
        },
    ]

    summary = summarize_matches(records, target_bot="target", move_records=moves)
    led_then_lost = summary["led_then_lost"]
    assert led_then_lost["available"] is True
    assert led_then_lost["traced_resolved_games"] == 2
    assert led_then_lost["traced_losses"] == 1
    assert led_then_lost["lead_thresholds"] == [
        {
            "threshold_points": 40.0,
            "count": 1,
            "rate_per_traced_resolved_game": 0.5,
            "rate_per_traced_loss": 1.0,
        },
        {
            "threshold_points": 80.0,
            "count": 1,
            "rate_per_traced_resolved_game": 0.5,
            "rate_per_traced_loss": 1.0,
        },
    ]
    assert led_then_lost["high_confidence_loss"]["count"] == 1


def test_compare_analyses_reports_move_disagreement_and_upside():
    baseline = {
        "position-1": {
            "id": "position-1",
            "moveInfos": [
                {"move": "D4", "order": 0, "visits": 100, "utility": 1.0, "scoreSelfplay": 40.0},
                {"move": "Q16", "order": 1, "visits": 50, "utility": 0.8, "scoreSelfplay": 45.0},
            ],
        }
    }
    custom = {
        "position-1": {
            "id": "position-1",
            "moveInfos": [
                {"move": "Q16", "order": 0, "visits": 120, "utility": 12.0, "scoreSelfplay": 55.0},
                {"move": "D4", "order": 1, "visits": 30, "utility": 10.0, "scoreSelfplay": 42.0},
            ],
        }
    }

    summary = compare_analyses(baseline, custom)
    assert summary["positions"] == 1
    assert summary["disagreements"] == 1
    assert summary["disagreement_rate"] == 1.0
    assert summary["mean_custom_utility_delta"] == 2.0
    assert summary["mean_custom_score_upside"] == 13.0
    assert summary["comparisons"][0]["baseline_utility_delta"] == pytest.approx(-0.2)
