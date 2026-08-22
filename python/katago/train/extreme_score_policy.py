import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

POLICY_CONTRACT = "risk-score-extreme-score-training-policy-v1"
POLICY_VERSION = "expected-max-focal-selfplay-v1"


def _positive_int(value: Any, role: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{role} must be a positive integer")
    return value


def _load_stable_json(path: Path) -> tuple[Mapping[str, Any], str]:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError(
            "extreme-score training policy must be an absolute regular file"
        )
    before = source.stat()
    data = source.read_bytes()
    after = source.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError("extreme-score training policy changed while loading")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"extreme-score training policy is invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TypeError("extreme-score training policy must have an object root")
    return value, hashlib.sha256(data).hexdigest()


def load_extreme_score_training_policy(path: Path, cohort_size: int) -> dict[str, Any]:
    """Validate and hash-bind the frozen score-only objective policy."""
    value, digest = _load_stable_json(Path(path))
    if (
        value.get("schema_version") != 1
        or value.get("contract") != POLICY_CONTRACT
        or value.get("policy_version") != POLICY_VERSION
        or value.get("status") != "frozen"
    ):
        raise ValueError("extreme-score training policy identity is invalid")

    objective = value.get("objective")
    training = value.get("training")
    league = value.get("league")
    curriculum = value.get("curriculum")
    if (
        not isinstance(objective, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(league, Mapping)
        or not isinstance(curriculum, list)
        or not curriculum
    ):
        raise ValueError("extreme-score training policy sections are missing")
    if (
        objective.get("name") != "expected_maximum_focal_terminal_score"
        or objective.get("win_loss_weight") != 0
        or training.get("extreme_score_only") is not True
        or training.get("allow_mixed_legacy_rows") is not False
        or training.get("optimistic_policy_from_win_loss") is not False
        or league.get("opponent_gradient") != "stopped"
    ):
        raise ValueError("extreme-score policy weakens score-only requirements")

    cohort_sizes_raw = objective.get("cohort_sizes")
    maximum = _positive_int(
        objective.get("maximum_supported_cohort_size"),
        "maximum supported cohort size",
    )
    if (
        not isinstance(cohort_sizes_raw, list)
        or not cohort_sizes_raw
        or any(type(item) is not int or item <= 0 for item in cohort_sizes_raw)
        or cohort_sizes_raw != sorted(set(cohort_sizes_raw))
        or maximum > 8
        or cohort_sizes_raw[-1] > maximum
    ):
        raise ValueError("extreme-score policy cohort sizes are invalid")
    if cohort_size not in cohort_sizes_raw:
        raise ValueError(
            f"cohort size {cohort_size} is absent from the frozen training policy"
        )

    stages: dict[int, int] = {}
    for index, stage in enumerate(curriculum):
        if not isinstance(stage, Mapping):
            raise TypeError(f"curriculum stage {index} must be an object")
        size = _positive_int(stage.get("cohort_size"), "curriculum cohort size")
        minimum = stage.get("minimum_selected_training_samples")
        if type(minimum) is not int or minimum < 0:
            raise ValueError(
                "minimum selected training samples must be a nonnegative integer"
            )
        if size in stages:
            raise ValueError("curriculum repeats a cohort size")
        stages[size] = minimum
    if list(stages) != cohort_sizes_raw:
        raise ValueError("curriculum does not match objective cohort sizes")

    production_size = _positive_int(
        objective.get("production_cohort_size"), "production cohort size"
    )
    if production_size not in stages:
        raise ValueError("production cohort size is absent from curriculum")
    opponent_weights = league.get("opponent_weights")
    if (
        not isinstance(opponent_weights, Mapping)
        or not opponent_weights
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
            for weight in opponent_weights.values()
        )
        or abs(sum(float(weight) for weight in opponent_weights.values()) - 1.0) > 1e-9
    ):
        raise ValueError("league opponent weights must sum to one")
    _positive_int(
        league.get("maximum_games_per_worker"),
        "maximum games per worker",
    )

    return {
        "path": str(Path(os.path.abspath(path))),
        "file_sha256": digest,
        "contract": POLICY_CONTRACT,
        "policy_version": POLICY_VERSION,
        "cohort_size": cohort_size,
        "cohort_sizes": cohort_sizes_raw,
        "minimum_selected_training_samples": stages[cohort_size],
        "production_cohort_size": production_size,
        "maximum_supported_cohort_size": maximum,
    }


def validate_extreme_score_curriculum_transition(
    *,
    previous_cohort_size: int,
    requested_policy: Mapping[str, Any],
    selected_training_samples: int,
    allow_transition: bool,
) -> bool:
    """Validate a one-stage, sample-powered curriculum transition."""
    requested_cohort_size = requested_policy["cohort_size"]
    if previous_cohort_size == requested_cohort_size:
        return False
    if not allow_transition:
        raise ValueError(
            "refusing extreme-score cohort-size change without "
            "-allow-extreme-score-curriculum-transition"
        )
    cohort_sizes = requested_policy["cohort_sizes"]
    if (
        type(previous_cohort_size) is not int
        or previous_cohort_size not in cohort_sizes
        or requested_cohort_size not in cohort_sizes
        or cohort_sizes.index(requested_cohort_size)
        != cohort_sizes.index(previous_cohort_size) + 1
    ):
        raise ValueError(
            "extreme-score curriculum transitions must advance exactly one frozen stage"
        )
    required_samples = requested_policy["minimum_selected_training_samples"]
    if selected_training_samples < required_samples:
        raise ValueError(
            "extreme-score curriculum transition is premature: "
            f"{selected_training_samples} selected samples, "
            f"{required_samples} required"
        )
    return True
