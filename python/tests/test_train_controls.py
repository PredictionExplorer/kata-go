from collections import defaultdict

from katago.train.training_controls import (
    add_validation_telemetry,
    build_validation_manifest,
    export_interval_ready,
    validate_validation_manifest,
    validation_data_dir,
)
import pytest


def test_export_interval_uses_global_sample_high_water():
    assert export_interval_ready(100, 100, None)
    assert not export_interval_ready(599_999, 100_000, 500_000)
    assert export_interval_ready(600_000, 100_000, 500_000)


def test_fixed_validation_directory_overrides_rolling_shuffle_val(tmp_path):
    current = tmp_path / "shuffle"
    fixed = tmp_path / "fixed"
    assert validation_data_dir(str(current), None) == str(current / "val")
    assert validation_data_dir(str(current), str(fixed)) == str(fixed)


def test_validation_telemetry_is_one_epoch_record():
    sums = defaultdict(float, {"loss_sum": 24.0})
    weights = defaultdict(float, {"loss_sum": 12.0})
    running = {
        "sums": {"nsamp": 500_000.0, "wsum": 450_000.0},
        "weights": {"nsamp": 1.0, "wsum": 1.0},
    }

    add_validation_telemetry(
        sums,
        weights,
        global_step_samples=11_300_000_000,
        validation_samples=12_288,
        validation_batches=48,
        validation_wall_seconds=9.5,
        running_metrics=running,
    )

    assert sums["global_step_samples"] == 11_300_000_000
    assert sums["val_samples"] == 12_288
    assert sums["val_batches"] == 48
    assert sums["val_wall_seconds"] == 9.5
    assert sums["nsamp_train"] == 500_000
    assert sums["wsum_train"] == 450_000
    assert all(
        weights[key] == 1.0
        for key in (
            "global_step_samples",
            "val_samples",
            "val_batches",
            "val_wall_seconds",
            "nsamp_train",
            "wsum_train",
        )
    )


def test_fixed_validation_manifest_is_immutable_and_revalidated(tmp_path):
    directory = (tmp_path / "fixed-validation").resolve()
    directory.mkdir()
    first = directory / "first.npz"
    second = directory / "second.npz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest_path = (tmp_path / "fixed-validation-manifest.json").resolve()

    first_manifest = build_validation_manifest(directory, manifest_path)
    second_manifest = build_validation_manifest(directory, manifest_path)

    assert first_manifest == second_manifest
    assert validate_validation_manifest(directory, manifest_path) == first_manifest
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inventory or manifest changed"):
        validate_validation_manifest(directory, manifest_path)


def test_fixed_validation_manifest_requires_npz_inputs(tmp_path):
    directory = (tmp_path / "empty-validation").resolve()
    directory.mkdir()
    with pytest.raises(ValueError, match="contains no NPZ"):
        build_validation_manifest(
            directory, (tmp_path / "manifest.json").resolve()
        )


def test_fixed_validation_manifest_rejects_nested_npz_inputs(tmp_path):
    directory = (tmp_path / "nested-validation").resolve()
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (nested / "data.npz").write_bytes(b"data")
    with pytest.raises(ValueError, match="must be top-level"):
        build_validation_manifest(
            directory, (tmp_path / "manifest.json").resolve()
        )


def test_fixed_validation_manifest_rejects_symlinked_directory(tmp_path):
    directory = tmp_path / "validation"
    directory.mkdir()
    (directory / "data.npz").write_bytes(b"data")
    link = tmp_path / "validation-link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="absolute and non-symlink"):
        build_validation_manifest(
            link, (tmp_path / "manifest.json").resolve()
        )
