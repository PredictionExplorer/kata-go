#!/usr/bin/env python3
"""Shared validation and identity helpers for KataGo PositionSample JSON."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from risk_score.generate_schedule import REQUIRED_POSITION_KEYS, validate_position


SEMANTIC_POSITION_FIELDS = (
    "xSize",
    "ySize",
    "board",
    "nextPla",
    "moveLocs",
    "movePlas",
    "initialTurnNumber",
)
OPTIONAL_POSITION_FIELDS = ("weight", "trainingWeight", "metadata")
_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
_MACHINE_LOCATION = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_GTP_LOCATION = re.compile(r"^([A-HJ-Z])(\d+)$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_location(
    value: Any,
    *,
    x_size: int,
    y_size: int,
    allow_null: bool,
    source: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source}: location must be a string")
    text = value.strip()
    lowered = text.lower()
    if lowered in {"pass", "pss"}:
        return "pass"
    if lowered in {"", "null", "''", '""', "'null'", '"null"'}:
        if allow_null:
            return "null"
        raise ValueError(f"{source}: null is not a legal move")
    machine = _MACHINE_LOCATION.fullmatch(text)
    if machine is not None:
        x = int(machine.group(1))
        y = int(machine.group(2))
        if not (0 <= x < x_size and 0 <= y < y_size):
            raise ValueError(f"{source}: machine location is off board")
        return f"{_GTP_COLUMNS[x]}{y_size-y}"
    match = _GTP_LOCATION.fullmatch(text.upper())
    if match is None:
        raise ValueError(f"{source}: malformed GTP location {value!r}")
    x = _GTP_COLUMNS.index(match.group(1))
    row = int(match.group(2))
    if x >= x_size or not 1 <= row <= y_size:
        raise ValueError(f"{source}: GTP location is off board")
    return f"{_GTP_COLUMNS[x]}{row}"


def normalize_position_sample(value: Mapping[str, Any], source: str) -> Dict[str, Any]:
    """Return a canonical, C++-readable PositionSample without curation metadata."""

    raw = value.get("position") if isinstance(value, Mapping) else None
    position = raw if isinstance(raw, Mapping) else value
    checked = validate_position(dict(position), source)
    x_size = checked["xSize"]
    y_size = checked["ySize"]
    if isinstance(x_size, bool) or isinstance(y_size, bool):
        raise ValueError(f"{source}: board dimensions may not be booleans")
    board = checked["board"]
    if not isinstance(board, str):
        raise ValueError(f"{source}: board must be a string")
    rows = board.rstrip("/").split("/")
    if (
        len(rows) != y_size
        or any(len(row) != x_size for row in rows)
        or any(character not in ".xXoO" for row in rows for character in row)
    ):
        raise ValueError(f"{source}: board string does not match its dimensions")
    next_player = (
        checked["nextPla"].upper()
        if isinstance(checked["nextPla"], str)
        else checked["nextPla"]
    )
    if next_player not in {"B", "W"}:
        raise ValueError(f"{source}: nextPla must be B or W")
    move_players = [
        player.upper() if isinstance(player, str) else player
        for player in checked["movePlas"]
    ]
    if any(player not in {"B", "W"} for player in move_players):
        raise ValueError(f"{source}: movePlas must contain only B or W")
    if (
        isinstance(checked["initialTurnNumber"], bool)
        or not isinstance(checked["initialTurnNumber"], int)
        or checked["initialTurnNumber"] < 0
    ):
        raise ValueError(f"{source}: initialTurnNumber must be nonnegative")
    normalized = {key: checked[key] for key in REQUIRED_POSITION_KEYS}
    normalized["board"] = "/".join(row.upper() for row in rows)
    normalized["nextPla"] = next_player
    normalized["moveLocs"] = [
        _canonical_location(
            move,
            x_size=x_size,
            y_size=y_size,
            allow_null=False,
            source=f"{source}: moveLocs[{index}]",
        )
        for index, move in enumerate(checked["moveLocs"])
    ]
    normalized["movePlas"] = move_players
    normalized["hintLoc"] = _canonical_location(
        checked["hintLoc"],
        x_size=x_size,
        y_size=y_size,
        allow_null=True,
        source=f"{source}: hintLoc",
    )
    for key in OPTIONAL_POSITION_FIELDS:
        if key not in checked:
            continue
        item = checked[key]
        if key == "metadata":
            if not isinstance(item, str):
                raise ValueError(
                    f"{source}: metadata must remain a string for C++ compatibility"
                )
        elif (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{source}: {key} must be finite")
        normalized[key] = item
    return normalized


def semantic_position(position: Mapping[str, Any]) -> Dict[str, Any]:
    raw = position.get("position") if isinstance(position, Mapping) else None
    source = raw if isinstance(raw, Mapping) else position
    core = {key: source[key] for key in REQUIRED_POSITION_KEYS if key in source}
    normalized = normalize_position_sample(core, "semantic position")
    return {key: normalized[key] for key in SEMANTIC_POSITION_FIELDS}


def semantic_position_sha256(position: Mapping[str, Any]) -> str:
    return canonical_sha256(semantic_position(position))


def _gtp_location(x: int, y: int, y_size: int) -> str:
    if x >= len(_GTP_COLUMNS):
        raise ValueError("analysis query generation supports boards up to 25 columns")
    return f"{_GTP_COLUMNS[x]}{y_size - y}"


def initial_stones(position: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    normalized = normalize_position_sample(position, "analysis position")
    rows = normalized["board"].rstrip("/").split("/")
    stones = []
    for y, row in enumerate(rows):
        for x, character in enumerate(row):
            if character in "xX":
                stones.append(("B", _gtp_location(x, y, normalized["ySize"])))
            elif character in "oO":
                stones.append(("W", _gtp_location(x, y, normalized["ySize"])))
    return tuple(stones)


def build_analysis_query(
    position: Mapping[str, Any],
    *,
    query_id: str,
    max_visits: int,
    powered: bool,
    komi: float = 7.5,
) -> Dict[str, Any]:
    """Build one deterministic final-turn KataGo analysis query."""

    normalized = normalize_position_sample(position, f"analysis query {query_id}")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("analysis query id must be nonempty")
    if type(max_visits) is not int or max_visits <= 0:
        raise ValueError("analysis max_visits must be a positive integer")
    if isinstance(komi, bool) or not isinstance(komi, (int, float)):
        raise ValueError("analysis komi must be numeric")
    return {
        "id": query_id,
        "moves": [
            [player, location]
            for player, location in zip(
                normalized["movePlas"], normalized["moveLocs"]
            )
        ],
        "initialStones": [list(stone) for stone in initial_stones(normalized)],
        "initialPlayer": normalized["nextPla"],
        "rules": "tromp-taylor",
        "komi": float(komi),
        "boardXSize": normalized["xSize"],
        "boardYSize": normalized["ySize"],
        "includePolicy": True,
        "maxVisits": max_visits,
        "overrideSettings": {
            "useScoreMaximizingUtility": powered,
            "scorePower": 1.5,
            "scoreScale": 20.0,
            "winWeight": 4.0,
            "rootNoiseEnabled": False,
            "rootNumSymmetriesToSample": 1,
        },
    }
