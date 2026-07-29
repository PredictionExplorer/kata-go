#!/usr/bin/env python3
"""Summarize deterministic KataGo match-result JSONL files."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple


DEFAULT_SCORE_POWER = 1.5
DEFAULT_SCORE_SCALE = 20.0
DEFAULT_WIN_WEIGHT = 4.0
DEFAULT_CATASTROPHE_THRESHOLDS = (20.0, 50.0)
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_results(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: result row must be a JSON object")
                records.append(record)
    if not records:
        raise ValueError("no match-result rows were loaded")
    return records


def quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantiles must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _winner(record: Dict[str, Any]) -> Optional[str]:
    winner = record.get("winner")
    if winner in ("B", "W", "draw"):
        return winner
    final_result = record.get("finalResult")
    if isinstance(final_result, str):
        if final_result.startswith("B+"):
            return "B"
        if final_result.startswith("W+"):
            return "W"
        if final_result == "0":
            return "draw"
    return None


def _stats(values: Sequence[float], probabilities: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "quantiles": {f"{p:g}": None for p in probabilities},
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "quantiles": {f"{p:g}": quantile(values, p) for p in probabilities},
        "minimum": min(values),
        "maximum": max(values),
    }


def wilson_interval(successes: float, total: int, z: float = 1.959963984540054) -> Optional[List[float]]:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _compact_game(record: Dict[str, Any], target_margin: float) -> Dict[str, Any]:
    return {
        "schedule_id": record.get("scheduleId"),
        "game_id": record.get("gameId"),
        "pair_id": record.get("pairId"),
        "position_id": record.get("positionId"),
        "seed": record.get("seed"),
        "black_bot": record.get("blackBot"),
        "white_bot": record.get("whiteBot"),
        "final_result": record.get("finalResult"),
        "target_margin": target_margin,
    }


def realized_terminal_utility(
    target_margin: float,
    outcome: float,
    *,
    score_power: float,
    score_scale: float,
    win_weight: float,
) -> float:
    score_term = math.copysign(
        (1.0 + abs(target_margin) / score_scale) ** score_power - 1.0,
        target_margin,
    ) if target_margin != 0.0 else 0.0
    return win_weight * outcome + score_term


def summarize_matches(
    records: Iterable[Dict[str, Any]],
    *,
    target_bot: str,
    move_records: Optional[Iterable[Dict[str, Any]]] = None,
    catastrophe_thresholds: Sequence[float] = DEFAULT_CATASTROPHE_THRESHOLDS,
    probabilities: Sequence[float] = DEFAULT_QUANTILES,
    score_power: float = DEFAULT_SCORE_POWER,
    score_scale: float = DEFAULT_SCORE_SCALE,
    win_weight: float = DEFAULT_WIN_WEIGHT,
    top_n: int = 5,
) -> Dict[str, Any]:
    if not target_bot:
        raise ValueError("target_bot must not be empty")
    if not math.isfinite(score_power) or score_power <= 0.0:
        raise ValueError("score_power must be finite and positive")
    if not math.isfinite(score_scale) or score_scale <= 0.0:
        raise ValueError("score_scale must be finite and positive")
    if not math.isfinite(win_weight):
        raise ValueError("win_weight must be finite")
    if top_n < 0:
        raise ValueError("top_n must be nonnegative")

    thresholds = sorted(set(float(value) for value in catastrophe_thresholds))
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("catastrophe thresholds must be finite and positive")
    quantiles = sorted(set(float(value) for value in probabilities))
    if any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("quantiles must be between 0 and 1")

    materialized = list(records)
    wins = 0
    losses = 0
    draws = 0
    no_results = 0
    turn_limits = 0
    unresolved = 0
    resignations = 0
    margins: List[float] = []
    utilities: List[float] = []
    margin_games: List[Tuple[float, Dict[str, Any]]] = []
    outcomes_by_game_id: Dict[str, Optional[float]] = {}

    for index, record in enumerate(materialized):
        black_is_target = record.get("blackBot") == target_bot
        white_is_target = record.get("whiteBot") == target_bot
        if black_is_target == white_is_target:
            raise ValueError(
                f"record {index} gameId={record.get('gameId')!r} must contain target bot "
                "exactly once"
            )
        target_color = "B" if black_is_target else "W"
        hit_turn_limit = bool(record.get("hitTurnLimit", False))
        is_no_result = bool(record.get("noResult", False))
        is_resignation = bool(record.get("resignation", False))
        if is_resignation:
            resignations += 1

        winner = _winner(record)
        outcome: Optional[float]
        if hit_turn_limit:
            turn_limits += 1
            outcome = None
        elif is_no_result:
            no_results += 1
            outcome = None
        elif winner == target_color:
            wins += 1
            outcome = 1.0
        elif winner in ("B", "W"):
            losses += 1
            outcome = -1.0
        elif winner == "draw":
            draws += 1
            outcome = 0.0
        else:
            unresolved += 1
            outcome = None
        game_id = record.get("gameId")
        if isinstance(game_id, str) and game_id:
            outcomes_by_game_id[game_id] = outcome

        white_minus_black = _number(record.get("finalWhiteMinusBlackScore"))
        if white_minus_black is None or hit_turn_limit or is_no_result:
            continue
        target_margin = white_minus_black if white_is_target else -white_minus_black
        margins.append(target_margin)
        margin_games.append((target_margin, record))

        utility_outcome = outcome
        if utility_outcome is None:
            utility_outcome = 1.0 if target_margin > 0 else -1.0 if target_margin < 0 else 0.0
        utility = realized_terminal_utility(
            target_margin,
            utility_outcome,
            score_power=score_power,
            score_scale=score_scale,
            win_weight=win_weight,
        )
        if not math.isfinite(utility):
            raise ValueError(f"non-finite terminal utility for gameId={record.get('gameId')!r}")
        utilities.append(utility)

    resolved_games = wins + losses + draws
    ordinary_win_rate = (
        (wins + 0.5 * draws) / resolved_games if resolved_games > 0 else None
    )
    ordinary_win_rate_interval = wilson_interval(
        wins + 0.5 * draws,
        resolved_games,
    )
    largest_wins = [
        _compact_game(record, margin)
        for margin, record in sorted(
            (item for item in margin_games if item[0] > 0.0),
            key=lambda item: item[0],
            reverse=True,
        )[:top_n]
    ]
    largest_losses = [
        _compact_game(record, margin)
        for margin, record in sorted(
            (item for item in margin_games if item[0] < 0.0),
            key=lambda item: item[0],
        )[:top_n]
    ]
    catastrophe_rows = []
    for threshold in thresholds:
        count = sum(1 for margin in margins if margin <= -threshold)
        catastrophe_rows.append(
            {
                "threshold_points": threshold,
                "count": count,
                "rate": count / len(margins) if margins else None,
            }
        )

    margin_stats = _stats(margins, quantiles)
    margin_stats["largest_wins"] = largest_wins
    margin_stats["largest_losses"] = largest_losses
    utility_stats = _stats(utilities, quantiles)
    utility_stats.update(
        {
            "formula": (
                "win_weight*outcome + sign(target_margin)*"
                "((1+abs(target_margin)/score_scale)**score_power-1)"
            ),
            "outcome_convention": "target win=1, draw=0, target loss=-1",
            "score_power": score_power,
            "score_scale": score_scale,
            "win_weight": win_weight,
            "excluded_without_numeric_margin": len(materialized) - len(utilities),
        }
    )

    led_then_lost: Dict[str, Any]
    if move_records is None:
        led_then_lost = {
            "available": False,
            "reason": (
                "No per-move JSONL was supplied, so 'led by 40/80 then lost' "
                "and high-confidence-loss metrics are not computed."
            ),
        }
    else:
        traces_by_game: Dict[str, List[Dict[str, Any]]] = {}
        for index, move_record in enumerate(move_records):
            if not isinstance(move_record, dict):
                raise ValueError(f"move record {index} must be a JSON object")
            if move_record.get("bot") != target_bot:
                continue
            game_id = move_record.get("gameId")
            if not isinstance(game_id, str) or not game_id:
                raise ValueError(f"move record {index} has no nonempty gameId")
            traces_by_game.setdefault(game_id, []).append(move_record)

        eligible_game_ids = sorted(
            game_id
            for game_id, outcome in outcomes_by_game_id.items()
            if outcome is not None and game_id in traces_by_game
        )
        lost_game_ids = {
            game_id
            for game_id in eligible_game_ids
            if outcomes_by_game_id[game_id] == -1.0
        }

        lead_thresholds = []
        for threshold in (40.0, 80.0):
            count = sum(
                1
                for game_id in lost_game_ids
                if any(
                    (_number(move.get("scoreLead")) or -math.inf) >= threshold
                    for move in traces_by_game[game_id]
                )
            )
            lead_thresholds.append(
                {
                    "threshold_points": threshold,
                    "count": count,
                    "rate_per_traced_resolved_game": (
                        count / len(eligible_game_ids) if eligible_game_ids else None
                    ),
                    "rate_per_traced_loss": (
                        count / len(lost_game_ids) if lost_game_ids else None
                    ),
                }
            )

        high_confidence_count = sum(
            1
            for game_id in lost_game_ids
            if any(
                (_number(move.get("winProbability")) or -math.inf) >= 0.95
                for move in traces_by_game[game_id]
            )
        )
        led_then_lost = {
            "available": True,
            "traced_resolved_games": len(eligible_game_ids),
            "traced_losses": len(lost_game_ids),
            "lead_thresholds": lead_thresholds,
            "high_confidence_loss": {
                "win_probability_threshold": 0.95,
                "count": high_confidence_count,
                "rate_per_traced_resolved_game": (
                    high_confidence_count / len(eligible_game_ids)
                    if eligible_game_ids
                    else None
                ),
                "rate_per_traced_loss": (
                    high_confidence_count / len(lost_game_ids)
                    if lost_game_ids
                    else None
                ),
            },
        }

    return {
        "schema_version": 1,
        "target_bot": target_bot,
        "games": {
            "total": len(materialized),
            "resolved_for_win_rate": resolved_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "no_results": no_results,
            "turn_limits": turn_limits,
            "unresolved": unresolved,
            "resignations": resignations,
        },
        "ordinary_win_rate": ordinary_win_rate,
        "ordinary_win_rate_interval_95": ordinary_win_rate_interval,
        "final_margin": margin_stats,
        "final_margin_catastrophes": {
            "metric": "target final margin <= -threshold_points",
            "denominator": "games with a numeric finalWhiteMinusBlackScore",
            "thresholds": catastrophe_rows,
        },
        "led_then_lost": led_then_lost,
        "terminal_custom_utility": utility_stats,
    }


def write_text_summary(summary: Dict[str, Any], output: TextIO = sys.stderr) -> None:
    games = summary["games"]
    margin = summary["final_margin"]
    utility = summary["terminal_custom_utility"]
    win_rate = summary["ordinary_win_rate"]
    win_rate_text = "n/a" if win_rate is None else f"{100.0 * win_rate:.2f}%"
    win_rate_interval = summary["ordinary_win_rate_interval_95"]
    win_rate_interval_text = (
        ""
        if win_rate_interval is None
        else (
            f" (95% Wilson {100.0 * win_rate_interval[0]:.2f}%"
            f"–{100.0 * win_rate_interval[1]:.2f}%)"
        )
    )
    print(f"Target bot: {summary['target_bot']}", file=output)
    print(
        f"Games: {games['total']} total; {games['wins']} wins, {games['losses']} losses, "
        f"{games['draws']} draws, {games['no_results']} no-results, "
        f"{games['turn_limits']} turn limits",
        file=output,
    )
    print(
        f"Ordinary win rate (draw=0.5): {win_rate_text}{win_rate_interval_text}",
        file=output,
    )
    if margin["count"]:
        print(
            f"Final margin: n={margin['count']}, mean={margin['mean']:.3f}, "
            f"median={margin['median']:.3f}, min={margin['minimum']:.3f}, "
            f"max={margin['maximum']:.3f}",
            file=output,
        )
    else:
        print("Final margin: no games with numeric terminal scores", file=output)
    for row in summary["final_margin_catastrophes"]["thresholds"]:
        rate = "n/a" if row["rate"] is None else f"{100.0 * row['rate']:.2f}%"
        print(
            f"Final-margin catastrophe <= -{row['threshold_points']:g}: "
            f"{row['count']} ({rate})",
            file=output,
        )
    led_then_lost = summary["led_then_lost"]
    if led_then_lost["available"]:
        for row in led_then_lost["lead_thresholds"]:
            rate = row["rate_per_traced_resolved_game"]
            rate_text = "n/a" if rate is None else f"{100.0 * rate:.2f}%"
            print(
                f"Led by {row['threshold_points']:g}, then lost: "
                f"{row['count']} ({rate_text} of traced resolved games)",
                file=output,
            )
        high_confidence = led_then_lost["high_confidence_loss"]
        high_rate = high_confidence["rate_per_traced_resolved_game"]
        high_rate_text = "n/a" if high_rate is None else f"{100.0 * high_rate:.2f}%"
        print(
            f"Reached {100.0 * high_confidence['win_probability_threshold']:.0f}% "
            f"win probability, then lost: {high_confidence['count']} "
            f"({high_rate_text} of traced resolved games)",
            file=output,
        )
    else:
        print(f"Led-then-lost metrics unavailable: {led_then_lost['reason']}", file=output)
    if utility["count"]:
        print(
            f"Terminal custom utility: n={utility['count']}, "
            f"mean={utility['mean']:.6g}, median={utility['median']:.6g}",
            file=output,
        )
    else:
        print("Terminal custom utility: no scored games", file=output)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate KataGo match result JSONL from one or more files."
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--moves",
        nargs="+",
        type=Path,
        help="Optional per-move JSONL files for lead-then-loss metrics",
    )
    parser.add_argument("--target-bot", required=True)
    parser.add_argument(
        "--catastrophe-thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_CATASTROPHE_THRESHOLDS),
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_QUANTILES),
    )
    parser.add_argument("--score-power", type=float, default=DEFAULT_SCORE_POWER)
    parser.add_argument("--score-scale", type=float, default=DEFAULT_SCORE_SCALE)
    parser.add_argument("--win-weight", type=float, default=DEFAULT_WIN_WEIGHT)
    parser.add_argument("--top", type=int, default=5, help="Largest wins/losses to retain")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Suppress the readable summary on stderr",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        records = load_results(args.results)
        move_records = load_results(args.moves) if args.moves else None
        summary = summarize_matches(
            records,
            target_bot=args.target_bot,
            move_records=move_records,
            catastrophe_thresholds=args.catastrophe_thresholds,
            probabilities=args.quantiles,
            score_power=args.score_power,
            score_scale=args.score_scale,
            win_weight=args.win_weight,
            top_n=args.top,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary["input_files"] = [str(path) for path in args.results]
    summary["move_input_files"] = [str(path) for path in args.moves] if args.moves else []
    if not args.no_text:
        write_text_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
