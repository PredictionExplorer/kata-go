from pathlib import Path

import pytest
from katago.train.extreme_score_policy import (
    load_extreme_score_training_policy,
)
from risk_score.extreme_score_league import DEFAULT_POLICY_PATH
from risk_score.extreme_score_progress import (
    ExtremeScoreProgressError,
    load_training_progress,
    publish_training_progress,
)


def test_progress_binds_policy_selected_samples_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "extreme-score-progress.json"
    policy = load_extreme_score_training_policy(DEFAULT_POLICY_PATH, 1)
    train_state = {
        "extreme_score_only": True,
        "extreme_score_training_policy": policy,
        "extreme_score_shuffle_provenance": {"receipt": "bound"},
        "extreme_score_selected_samples": 1234,
    }
    published = publish_training_progress(
        output_path=output,
        checkpoint_path=checkpoint,
        train_state=train_state,
    )
    assert load_training_progress(output) == published
    assert published["selected_training_samples"] == 1234
    assert output.stat().st_mode & 0o222 == 0

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ExtremeScoreProgressError, match="checkpoint changed"):
        load_training_progress(output)


def test_train_snapshot_includes_risk_score_package():
    script = (Path(__file__).resolve().parents[1] / "selfplay" / "train.sh").read_text(
        encoding="utf-8"
    )
    assert 'cp -r "$GITROOTDIR"/python/risk_score "$DATED_ARCHIVE"' in script
