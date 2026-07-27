#!/usr/bin/env python3
"""Compare aligned standard and score-maximizing KataGo analysis JSONL."""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_analysis(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: missing nonempty id")
            if record_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            records[record_id] = record
    if not records:
        raise ValueError(f"{path}: no analysis records")
    return records


def _ordered_moves(record: Dict[str, Any], record_id: str) -> List[Dict[str, Any]]:
    moves = record.get("moveInfos")
    if not isinstance(moves, list) or not moves:
        raise ValueError(f"analysis id {record_id!r} has no moveInfos")
    if not all(isinstance(move, dict) for move in moves):
        raise ValueError(f"analysis id {record_id!r} has a non-object moveInfo")
    return sorted(moves, key=lambda move: (move.get("order", 1 << 30), -move.get("visits", 0)))


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _delta(
    move_by_name: Dict[str, Dict[str, Any]],
    first_move: str,
    second_move: str,
    field: str,
) -> Optional[float]:
    first = _finite_number(move_by_name.get(first_move, {}).get(field))
    second = _finite_number(move_by_name.get(second_move, {}).get(field))
    return first - second if first is not None and second is not None else None


def compare_analyses(
    baseline_records: Dict[str, Dict[str, Any]],
    custom_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if set(baseline_records) != set(custom_records):
        missing_custom = sorted(set(baseline_records).difference(custom_records))
        missing_baseline = sorted(set(custom_records).difference(baseline_records))
        raise ValueError(
            f"analysis IDs differ; missing custom={missing_custom}, "
            f"missing baseline={missing_baseline}"
        )

    rows: List[Dict[str, Any]] = []
    custom_utility_deltas: List[float] = []
    custom_score_upsides: List[float] = []
    disagreements = 0

    for record_id in sorted(baseline_records):
        baseline_moves = _ordered_moves(baseline_records[record_id], record_id)
        custom_moves = _ordered_moves(custom_records[record_id], record_id)
        baseline_move = str(baseline_moves[0].get("move"))
        custom_move = str(custom_moves[0].get("move"))
        disagrees = baseline_move != custom_move
        disagreements += int(disagrees)

        baseline_by_name = {str(move.get("move")): move for move in baseline_moves}
        custom_by_name = {str(move.get("move")): move for move in custom_moves}
        custom_utility_delta = _delta(
            custom_by_name, custom_move, baseline_move, "utility"
        )
        custom_score_upside = _delta(
            custom_by_name, custom_move, baseline_move, "scoreSelfplay"
        )
        baseline_utility_delta = _delta(
            baseline_by_name, custom_move, baseline_move, "utility"
        )
        if custom_utility_delta is not None:
            custom_utility_deltas.append(custom_utility_delta)
        if custom_score_upside is not None:
            custom_score_upsides.append(custom_score_upside)

        rows.append(
            {
                "id": record_id,
                "baseline_move": baseline_move,
                "custom_move": custom_move,
                "disagrees": disagrees,
                "custom_utility_delta": custom_utility_delta,
                "custom_score_upside": custom_score_upside,
                "baseline_utility_delta": baseline_utility_delta,
                "baseline_move_in_custom_candidates": baseline_move in custom_by_name,
                "custom_move_in_baseline_candidates": custom_move in baseline_by_name,
            }
        )

    return {
        "schema_version": 1,
        "positions": len(rows),
        "disagreements": disagreements,
        "disagreement_rate": disagreements / len(rows) if rows else None,
        "mean_custom_utility_delta": (
            statistics.fmean(custom_utility_deltas) if custom_utility_deltas else None
        ),
        "mean_custom_score_upside": (
            statistics.fmean(custom_score_upsides) if custom_score_upsides else None
        ),
        "comparisons": rows,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare aligned baseline and score-maximizing analysis JSONL."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("custom", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = compare_analyses(
            load_analysis(args.baseline),
            load_analysis(args.custom),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
