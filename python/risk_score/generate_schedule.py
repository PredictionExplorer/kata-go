#!/usr/bin/env python3
"""Generate deterministic, color-reversed KataGo match schedules."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = 1
GENERATOR_CONTRACT = "paired-color-reversal-with-per-game-seeds-v1"
REQUIRED_POSITION_KEYS = {
    "xSize",
    "ySize",
    "board",
    "nextPla",
    "moveLocs",
    "movePlas",
    "initialTurnNumber",
    "hintLoc",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def validate_position(position: Any, source: str) -> Dict[str, Any]:
    if not isinstance(position, dict):
        raise ValueError(f"{source}: position must be a JSON object")
    missing = sorted(REQUIRED_POSITION_KEYS.difference(position))
    if missing:
        raise ValueError(f"{source}: missing PositionSample keys: {', '.join(missing)}")
    if not isinstance(position["xSize"], int) or not isinstance(position["ySize"], int):
        raise ValueError(f"{source}: xSize and ySize must be integers")
    if position["xSize"] <= 0 or position["ySize"] <= 0:
        raise ValueError(f"{source}: xSize and ySize must be positive")
    if not isinstance(position["moveLocs"], list) or not isinstance(position["movePlas"], list):
        raise ValueError(f"{source}: moveLocs and movePlas must be arrays")
    if len(position["moveLocs"]) != len(position["movePlas"]):
        raise ValueError(f"{source}: moveLocs and movePlas must have equal length")
    return position


def load_positions(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    positions: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                source = f"{path}:{line_number}"
                try:
                    position = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}: invalid JSON: {exc}") from exc
                positions.append(validate_position(position, source))
    if not positions:
        raise ValueError("no PositionSample rows were loaded")
    return positions


def build_schedule(
    positions: Sequence[Dict[str, Any]],
    *,
    bot_a_index: int = 0,
    bot_b_index: int = 1,
    pairs_per_position: int = 1,
    base_seed: str = "risk-score-phase1-v1",
    schedule_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if bot_a_index < 0 or bot_b_index < 0:
        raise ValueError("bot indices must be nonnegative")
    if bot_a_index == bot_b_index:
        raise ValueError("bot A and bot B must have different indices")
    if pairs_per_position <= 0:
        raise ValueError("pairs_per_position must be positive")
    if not base_seed:
        raise ValueError("base_seed must not be empty")
    if not positions:
        raise ValueError("at least one position is required")

    checked_positions = [
        validate_position(position, f"position {index}")
        for index, position in enumerate(positions)
    ]
    schedule_basis = {
        "schemaVersion": SCHEMA_VERSION,
        "generatorContract": GENERATOR_CONTRACT,
        "botA": bot_a_index,
        "botB": bot_b_index,
        "pairsPerPosition": pairs_per_position,
        "baseSeed": base_seed,
        "positions": checked_positions,
    }
    if schedule_id is None:
        schedule_id = f"risk-score-v1-{_sha256(schedule_basis)[:20]}"
    elif not schedule_id or "\n" in schedule_id or "\r" in schedule_id:
        raise ValueError("schedule_id must be a nonempty single-line string")

    schedule: List[Dict[str, Any]] = []
    pair_index = 0
    for source_index, position in enumerate(checked_positions):
        position_hash = _sha256(position)
        position_id = f"pos-{position_hash[:20]}"
        for repetition in range(pairs_per_position):
            pair_id = (
                f"{schedule_id}:pair-{pair_index:06d}:"
                f"src-{source_index:06d}:rep-{repetition:03d}"
            )
            seed_basis = {
                "scheduleId": schedule_id,
                "baseSeed": base_seed,
                "positionHash": position_hash,
                "sourceIndex": source_index,
                "repetition": repetition,
            }

            common = {
                "schemaVersion": SCHEMA_VERSION,
                "generatorContract": GENERATOR_CONTRACT,
                "scheduleId": schedule_id,
                "pairId": pair_id,
                "positionId": position_id,
                "startPosition": position,
                "sourcePositionIndex": source_index,
                "pairRepetition": repetition,
            }
            first_seed_basis = dict(seed_basis)
            first_seed_basis["colorRole"] = "a-black"
            second_seed_basis = dict(seed_basis)
            second_seed_basis["colorRole"] = "a-white"
            first = dict(common)
            first.update(
                {
                    "gameId": f"{pair_id}:a-black",
                    "seed": f"risk-score-v1-{_sha256(first_seed_basis)[:32]}",
                    "blackBot": bot_a_index,
                    "whiteBot": bot_b_index,
                }
            )
            second = dict(common)
            second.update(
                {
                    "gameId": f"{pair_id}:a-white",
                    "seed": f"risk-score-v1-{_sha256(second_seed_basis)[:32]}",
                    "blackBot": bot_b_index,
                    "whiteBot": bot_a_index,
                }
            )
            schedule.extend((first, second))
            pair_index += 1
    return schedule


def write_schedule(rows: Iterable[Dict[str, Any]], output: str) -> int:
    count = 0
    if output == "-":
        handle = sys.stdout
        should_close = False
    else:
        handle = Path(output).open("w", encoding="utf-8")
        should_close = True
    try:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")
            count += 1
    finally:
        if should_close:
            handle.close()
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stable two-game color pair for every PositionSample."
    )
    parser.add_argument("positions", nargs="+", type=Path, help="PositionSample JSONL input")
    parser.add_argument("-o", "--output", required=True, help="Schedule JSONL output, or -")
    parser.add_argument("--bot-a-index", type=int, default=0)
    parser.add_argument("--bot-b-index", type=int, default=1)
    parser.add_argument("--pairs-per-position", type=int, default=1)
    parser.add_argument("--base-seed", default="risk-score-phase1-v1")
    parser.add_argument("--schedule-id", help="Explicit schedule ID instead of a content hash")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        positions = load_positions(args.positions)
        schedule = build_schedule(
            positions,
            bot_a_index=args.bot_a_index,
            bot_b_index=args.bot_b_index,
            pairs_per_position=args.pairs_per_position,
            base_seed=args.base_seed,
            schedule_id=args.schedule_id,
        )
        count = write_schedule(schedule, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {count} games ({count // 2} color pairs) with schedule ID "
        f"{schedule[0]['scheduleId']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
