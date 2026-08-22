"""Publish checkpoint-bound score-only curriculum progress."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from risk_score.extreme_score_evaluator import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)

PROGRESS_CONTRACT = "risk-score-extreme-training-progress-v1"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ExtremeScoreProgressError(RuntimeError):
    """Score-only checkpoint progress is invalid or changed."""


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ExtremeScoreProgressError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    return source.resolve()


def _replace_read_only_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    data = (canonical_json(value) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ExtremeScoreProgressError("progress parent is unsafe")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_training_progress(
    *,
    output_path: Path,
    checkpoint_path: Path,
    train_state: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _regular_file(checkpoint_path, "trainer checkpoint")
    policy = train_state.get("extreme_score_training_policy")
    provenance = train_state.get("extreme_score_shuffle_provenance")
    samples = train_state.get("extreme_score_selected_samples")
    if (
        train_state.get("extreme_score_only") is not True
        or not isinstance(policy, Mapping)
        or not isinstance(provenance, Mapping)
        or type(samples) is not int
        or samples < 0
    ):
        raise ExtremeScoreProgressError(
            "train state lacks score-only policy, samples, or shuffle provenance"
        )
    value = {
        "schema_version": 1,
        "contract": PROGRESS_CONTRACT,
        "checkpoint": {
            "path": str(checkpoint),
            "file_sha256": file_sha256(checkpoint),
        },
        "training_policy": dict(policy),
        "cohort_size": policy["cohort_size"],
        "selected_training_samples": samples,
        "shuffle_provenance_sha256": canonical_sha256(provenance),
    }
    value["progress_sha256"] = canonical_sha256(value)
    _replace_read_only_json(output_path, value)
    return value


def load_training_progress(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "training progress")
    if source.stat().st_mode & _WRITE_BITS:
        raise ExtremeScoreProgressError("training progress must be read-only")
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtremeScoreProgressError(f"training progress is invalid: {exc}") from exc
    expected = {
        "schema_version",
        "contract",
        "checkpoint",
        "training_policy",
        "cohort_size",
        "selected_training_samples",
        "shuffle_provenance_sha256",
        "progress_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ExtremeScoreProgressError("training progress keys differ")
    supplied = value["progress_sha256"]
    payload = dict(value)
    payload.pop("progress_sha256")
    if (
        value["schema_version"] != 1
        or value["contract"] != PROGRESS_CONTRACT
        or supplied != canonical_sha256(payload)
    ):
        raise ExtremeScoreProgressError("training progress self-hash is invalid")
    checkpoint = value["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "file_sha256",
    }:
        raise ExtremeScoreProgressError(
            "training progress checkpoint binding is malformed"
        )
    checkpoint_path = _regular_file(
        Path(checkpoint["path"]), "training progress checkpoint"
    )
    observed = file_sha256(checkpoint_path)
    if (
        observed != checkpoint["file_sha256"]
        or file_sha256(checkpoint_path) != observed
    ):
        raise ExtremeScoreProgressError("training progress checkpoint changed")
    if (
        not isinstance(value["training_policy"], Mapping)
        or type(value["cohort_size"]) is not int
        or type(value["selected_training_samples"]) is not int
        or value["selected_training_samples"] < 0
    ):
        raise ExtremeScoreProgressError("training progress values are malformed")
    return value
