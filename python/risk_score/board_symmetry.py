"""KataGo-compatible symmetries for PositionSample dictionaries."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Mapping, Tuple

from risk_score.position_samples import canonical_sha256, semantic_position_sha256

NUM_SYMMETRIES = 8
NUM_SYMMETRIES_WITHOUT_TRANSPOSE = 4
SQUARE_SYMMETRIES = tuple(range(NUM_SYMMETRIES))
RECTANGULAR_SYMMETRIES = tuple(range(NUM_SYMMETRIES_WITHOUT_TRANSPOSE))

_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
_GTP_LOCATION = re.compile(r"^([A-HJ-Z])(\d+)$", re.IGNORECASE)
_UNCHANGED_LOCATIONS = {"pass", "pss", "null"}

SymmetryOrbitEntry = Tuple[int, Dict[str, Any]]


def _validate_dimensions(x_size: int, y_size: int) -> None:
    if (
        isinstance(x_size, bool)
        or isinstance(y_size, bool)
        or not isinstance(x_size, int)
        or not isinstance(y_size, int)
        or x_size <= 0
        or y_size <= 0
    ):
        raise ValueError("xSize and ySize must be positive integers")
    if x_size > len(_GTP_COLUMNS):
        raise ValueError("GTP symmetry mapping supports boards up to 25 columns")


def _validate_symmetry(symmetry: int, x_size: int, y_size: int) -> None:
    if isinstance(symmetry, bool) or not isinstance(symmetry, int):
        raise ValueError("symmetry must be an integer")
    if symmetry not in shape_preserving_symmetries(x_size, y_size):
        raise ValueError(
            f"symmetry {symmetry} does not preserve a {x_size}x{y_size} board"
        )


def shape_preserving_symmetries(x_size: int, y_size: int) -> Tuple[int, ...]:
    """Return KataGo symmetry IDs valid without changing the board shape."""

    _validate_dimensions(x_size, y_size)
    return SQUARE_SYMMETRIES if x_size == y_size else RECTANGULAR_SYMMETRIES


def inverse_symmetry(symmetry: int) -> int:
    """Return the inverse of a KataGo symmetry ID."""

    if isinstance(symmetry, bool) or not isinstance(symmetry, int):
        raise ValueError("symmetry must be an integer")
    if not 0 <= symmetry < NUM_SYMMETRIES:
        raise ValueError(f"symmetry must be between 0 and {NUM_SYMMETRIES - 1}")
    if symmetry == 5:
        return 6
    if symmetry == 6:
        return 5
    return symmetry


def transform_xy(
    x: int,
    y: int,
    x_size: int,
    y_size: int,
    symmetry: int,
) -> Tuple[int, int]:
    """Map a zero-based KataGo board coordinate through ``symmetry``."""

    _validate_dimensions(x_size, y_size)
    _validate_symmetry(symmetry, x_size, y_size)
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, int)
        or not isinstance(y, int)
        or not 0 <= x < x_size
        or not 0 <= y < y_size
    ):
        raise ValueError("coordinate is off board")

    if symmetry & 0x2:
        x = x_size - x - 1
    if symmetry & 0x1:
        y = y_size - y - 1
    if symmetry & 0x4:
        x, y = y, x
    return x, y


def inverse_transform_xy(
    x: int,
    y: int,
    x_size: int,
    y_size: int,
    symmetry: int,
) -> Tuple[int, int]:
    """Map a transformed zero-based coordinate back to the source board."""

    _validate_dimensions(x_size, y_size)
    _validate_symmetry(symmetry, x_size, y_size)
    return transform_xy(x, y, x_size, y_size, inverse_symmetry(symmetry))


def _parse_gtp_location(location: str, x_size: int, y_size: int) -> Tuple[int, int]:
    if not isinstance(location, str):
        raise ValueError("location must be a string")
    text = location.strip().upper()
    match = _GTP_LOCATION.fullmatch(text)
    if match is None:
        raise ValueError(f"malformed GTP location {location!r}")
    x = _GTP_COLUMNS.index(match.group(1))
    row = int(match.group(2))
    if x >= x_size or not 1 <= row <= y_size:
        raise ValueError(f"GTP location {location!r} is off board")
    return x, y_size - row


def _format_gtp_location(x: int, y: int, y_size: int) -> str:
    return f"{_GTP_COLUMNS[x]}{y_size - y}"


def transform_gtp_location(
    location: str,
    x_size: int,
    y_size: int,
    symmetry: int,
) -> str:
    """Map a GTP location through ``symmetry``, preserving pass and null."""

    _validate_dimensions(x_size, y_size)
    _validate_symmetry(symmetry, x_size, y_size)
    if not isinstance(location, str):
        raise ValueError("location must be a string")
    if location.strip().lower() in _UNCHANGED_LOCATIONS:
        return location
    x, y = _parse_gtp_location(location, x_size, y_size)
    transformed_x, transformed_y = transform_xy(x, y, x_size, y_size, symmetry)
    return _format_gtp_location(transformed_x, transformed_y, y_size)


def inverse_transform_gtp_location(
    location: str,
    x_size: int,
    y_size: int,
    symmetry: int,
) -> str:
    """Map a transformed GTP location back to its source coordinate."""

    _validate_dimensions(x_size, y_size)
    _validate_symmetry(symmetry, x_size, y_size)
    if not isinstance(location, str):
        raise ValueError("location must be a string")
    if location.strip().lower() in _UNCHANGED_LOCATIONS:
        return location
    x, y = _parse_gtp_location(location, x_size, y_size)
    source_x, source_y = inverse_transform_xy(x, y, x_size, y_size, symmetry)
    return _format_gtp_location(source_x, source_y, y_size)


def _position_shape(position: Mapping[str, Any]) -> Tuple[int, int]:
    if not isinstance(position, Mapping):
        raise ValueError("position must be a mapping")
    try:
        x_size = position["xSize"]
        y_size = position["ySize"]
    except KeyError as exc:
        raise ValueError(f"position is missing {exc.args[0]}") from exc
    _validate_dimensions(x_size, y_size)
    return x_size, y_size


def _board_rows(position: Mapping[str, Any], x_size: int, y_size: int) -> List[str]:
    board = position.get("board")
    if not isinstance(board, str):
        raise ValueError("position board must be a string")
    rows = board.rstrip("/").split("/")
    if len(rows) != y_size or any(len(row) != x_size for row in rows):
        raise ValueError("position board does not match xSize and ySize")
    if any(character not in ".xXoO" for row in rows for character in row):
        raise ValueError("position board contains an invalid character")
    return rows


def apply_symmetry(
    position: Mapping[str, Any],
    symmetry: int,
) -> Dict[str, Any]:
    """Return a copy of ``position`` transformed like SymmetryHelpers."""

    x_size, y_size = _position_shape(position)
    _validate_symmetry(symmetry, x_size, y_size)
    rows = _board_rows(position, x_size, y_size)

    transformed_rows = [[""] * x_size for _ in range(y_size)]
    for y, row in enumerate(rows):
        for x, character in enumerate(row):
            transformed_x, transformed_y = transform_xy(x, y, x_size, y_size, symmetry)
            transformed_rows[transformed_y][transformed_x] = character

    transformed = copy.deepcopy(dict(position))
    transformed["xSize"] = x_size
    transformed["ySize"] = y_size
    transformed["board"] = "/".join("".join(row) for row in transformed_rows)

    move_locations = position.get("moveLocs")
    if not isinstance(move_locations, list):
        raise ValueError("position moveLocs must be an array")
    move_players = position.get("movePlas")
    if not isinstance(move_players, list):
        raise ValueError("position movePlas must be an array")
    if len(move_locations) != len(move_players):
        raise ValueError("moveLocs and movePlas must have equal length")
    transformed["moveLocs"] = [
        transform_gtp_location(location, x_size, y_size, symmetry)
        for location in move_locations
    ]
    if "hintLoc" in position:
        transformed["hintLoc"] = transform_gtp_location(
            position["hintLoc"], x_size, y_size, symmetry
        )
    return transformed


def invert_symmetry(
    position: Mapping[str, Any],
    symmetry: int,
) -> Dict[str, Any]:
    """Undo ``symmetry`` on an already transformed PositionSample."""

    return apply_symmetry(position, inverse_symmetry(symmetry))


def symmetry_orbit(position: Mapping[str, Any]) -> List[SymmetryOrbitEntry]:
    """Return distinct ``(symmetry, position)`` pairs sorted by semantic hash."""

    x_size, y_size = _position_shape(position)
    by_semantic_hash: Dict[str, SymmetryOrbitEntry] = {}
    for symmetry in shape_preserving_symmetries(x_size, y_size):
        transformed = apply_symmetry(position, symmetry)
        hashable = (
            transformed
            if "hintLoc" in transformed
            else {**transformed, "hintLoc": "null"}
        )
        digest = semantic_position_sha256(hashable)
        by_semantic_hash.setdefault(digest, (symmetry, transformed))
    return [by_semantic_hash[digest] for digest in sorted(by_semantic_hash)]


def symmetry_orbit_sha256(position: Mapping[str, Any]) -> str:
    """Return the orientation-invariant identity of a position's orbit."""

    return canonical_sha256(
        sorted(
            semantic_position_sha256(transformed)
            for _, transformed in symmetry_orbit(position)
        )
    )
