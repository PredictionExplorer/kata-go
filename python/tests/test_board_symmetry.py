import pytest

from risk_score.board_symmetry import (
    RECTANGULAR_SYMMETRIES,
    SQUARE_SYMMETRIES,
    apply_symmetry,
    inverse_symmetry,
    inverse_transform_gtp_location,
    invert_symmetry,
    shape_preserving_symmetries,
    symmetry_orbit,
    transform_gtp_location,
)
from risk_score.position_samples import semantic_position_sha256


def position(*, x_size=3, y_size=3, board="XO./..X/O.."):
    return {
        "xSize": x_size,
        "ySize": y_size,
        "board": board,
        "nextPla": "B",
        "moveLocs": ["A3", "pass"],
        "movePlas": ["B", "W"],
        "initialTurnNumber": 2,
        "hintLoc": "C1",
        "weight": 1.25,
        "metadata": "symmetry-test",
    }


@pytest.mark.parametrize(
    ("symmetry", "expected_board", "expected_corner"),
    [
        (0, "XO./..X/O..", "A3"),
        (1, "O../..X/XO.", "A1"),
        (2, ".OX/X../..O", "C3"),
        (3, "..O/X../.OX", "C1"),
        (4, "X.O/O../.X.", "A3"),
        (5, "O.X/..O/.X.", "C3"),
        (6, ".X./O../X.O", "A1"),
        (7, ".X./..O/O.X", "C1"),
    ],
)
def test_square_symmetries_match_katago_bit_order(
    symmetry, expected_board, expected_corner
):
    transformed = apply_symmetry(position(), symmetry)

    assert transformed["board"] == expected_board
    assert transform_gtp_location("A3", 3, 3, symmetry) == expected_corner
    assert (transformed["xSize"], transformed["ySize"]) == (3, 3)


@pytest.mark.parametrize(
    ("symmetry", "expected_board", "expected_corner"),
    [
        (0, "X.O./.OX.", "A2"),
        (1, ".OX./X.O.", "A1"),
        (2, ".O.X/.XO.", "D2"),
        (3, ".XO./.O.X", "D1"),
    ],
)
def test_rectangular_boards_have_only_shape_preserving_reflections(
    symmetry, expected_board, expected_corner
):
    sample = position(
        x_size=4,
        y_size=2,
        board="X.O./.OX.",
    )
    sample["moveLocs"] = ["A2", "pass"]
    sample["hintLoc"] = "D1"

    transformed = apply_symmetry(sample, symmetry)

    assert transformed["board"] == expected_board
    assert transformed["moveLocs"][0] == expected_corner
    assert (transformed["xSize"], transformed["ySize"]) == (4, 2)


def test_symmetry_sets_reject_rectangular_transpose():
    assert shape_preserving_symmetries(3, 3) == SQUARE_SYMMETRIES
    assert shape_preserving_symmetries(4, 2) == RECTANGULAR_SYMMETRIES

    with pytest.raises(ValueError, match="does not preserve"):
        apply_symmetry(
            position(x_size=4, y_size=2, board="X.O./.OX."),
            4,
        )


def test_gtp_mapping_skips_i_and_preserves_pass_and_null():
    assert transform_gtp_location("B3", 10, 3, 2) == "J3"
    assert inverse_transform_gtp_location("J3", 10, 3, 2) == "B3"
    assert transform_gtp_location("pass", 3, 3, 7) == "pass"
    assert transform_gtp_location("null", 3, 3, 7) == "null"
    assert inverse_transform_gtp_location("PASS", 3, 3, 7) == "PASS"

    with pytest.raises(ValueError, match="malformed GTP"):
        transform_gtp_location("I3", 10, 3, 0)


def test_move_history_hint_and_other_fields_are_transformed_selectively():
    sample = position()
    transformed = apply_symmetry(sample, 5)

    assert transformed["moveLocs"] == ["C3", "pass"]
    assert transformed["movePlas"] == sample["movePlas"]
    assert transformed["hintLoc"] == "A1"
    assert transformed["nextPla"] == sample["nextPla"]
    assert transformed["initialTurnNumber"] == sample["initialTurnNumber"]
    assert transformed["weight"] == sample["weight"]
    assert transformed["metadata"] == sample["metadata"]
    assert sample["moveLocs"] == ["A3", "pass"]

    without_hint = dict(sample)
    without_hint.pop("hintLoc")
    assert "hintLoc" not in apply_symmetry(without_hint, 3)


@pytest.mark.parametrize(
    "sample,symmetries",
    [
        (position(), SQUARE_SYMMETRIES),
        (
            {
                **position(x_size=4, y_size=2, board="X.O./.OX."),
                "moveLocs": ["A2", "pass"],
                "hintLoc": "D1",
            },
            RECTANGULAR_SYMMETRIES,
        ),
    ],
)
def test_position_and_coordinate_inverse_round_trips(sample, symmetries):
    for symmetry in symmetries:
        transformed = apply_symmetry(sample, symmetry)
        assert invert_symmetry(transformed, symmetry) == sample

        for location in sample["moveLocs"] + [sample["hintLoc"]]:
            mapped = transform_gtp_location(
                location,
                sample["xSize"],
                sample["ySize"],
                symmetry,
            )
            assert (
                inverse_transform_gtp_location(
                    mapped,
                    sample["xSize"],
                    sample["ySize"],
                    symmetry,
                )
                == location
            )

    assert inverse_symmetry(5) == 6
    assert inverse_symmetry(6) == 5
    assert all(inverse_symmetry(value) == value for value in (0, 1, 2, 3, 4, 7))


def test_symmetry_orbit_is_semantically_deduplicated_and_hash_sorted():
    asymmetric = position()
    first = symmetry_orbit(asymmetric)
    second = symmetry_orbit(asymmetric)
    hashes = [semantic_position_sha256(transformed) for _, transformed in first]

    assert first == second
    assert len(first) == 8
    assert len(set(hashes)) == len(hashes)
    assert hashes == sorted(hashes)

    symmetric = position(board=".../.../...")
    symmetric["moveLocs"] = []
    symmetric["movePlas"] = []
    symmetric["hintLoc"] = "null"
    deduplicated = symmetry_orbit(symmetric)

    assert deduplicated == [(0, symmetric)]

    symmetric.pop("hintLoc")
    assert symmetry_orbit(symmetric) == [(0, symmetric)]
