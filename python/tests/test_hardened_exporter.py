import dataclasses
import datetime
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from risk_score.hardened_exporter import (
    CommandResult,
    ExportError,
    ExportRequest,
    HardenedExporter,
    main,
)


class FakeExportRunner:
    def __init__(self, *, fail_export=False, fail_probe=False):
        self.fail_export = fail_export
        self.fail_probe = fail_probe
        self.commands = []

    def run(self, argv, *, timeout=None):
        argv = list(argv)
        self.commands.append(argv)
        if argv[0] == "fake-export":
            source_checkpoint = Path(argv[1])
            partial_dir = Path(argv[2])
            (partial_dir / "model.bin").write_bytes(
                b"model:" + source_checkpoint.read_bytes()
            )
            (partial_dir / "log.txt").write_text(
                f"temporary path: {partial_dir}", encoding="utf-8"
            )
            if self.fail_export:
                return CommandResult(17, stderr="injected export failure")
        elif argv[0] == "fake-clean":
            source_checkpoint = Path(argv[1])
            output_checkpoint = Path(argv[2])
            output_checkpoint.write_bytes(b"clean:" + source_checkpoint.read_bytes())
        elif argv[0] == "fake-probe" and self.fail_probe:
            return CommandResult(23, stderr="injected probe failure")
        return CommandResult(0)


def request_for(source_dir, destination_root, name="candidate-a"):
    return ExportRequest(
        source_dir=source_dir,
        destination_root=destination_root,
        candidate_name=name,
        model_name=f"run-{name}",
        export_command=(
            "fake-export",
            "{source_checkpoint}",
            "{partial_dir}",
        ),
        clean_checkpoint_command=(
            "fake-clean",
            "{source_checkpoint}",
            "{cleaned_checkpoint}",
        ),
        model_probe_command=("fake-probe", "{model_file}"),
    )


def source_candidate(tmp_path, content=b"source-checkpoint"):
    source_dir = tmp_path / "torchmodels_toexport" / "candidate-a"
    source_dir.mkdir(parents=True)
    checkpoint = source_dir / "model.ckpt"
    checkpoint.write_bytes(content)
    return source_dir, checkpoint


def test_publication_preserves_source_and_compresses_deterministically(tmp_path):
    source_dir, source_checkpoint = source_candidate(tmp_path)
    source_before = source_checkpoint.read_bytes()
    destination_root = tmp_path / "modelstobetested"
    runner = FakeExportRunner()

    result = HardenedExporter(command_runner=runner).publish(
        request_for(source_dir, destination_root)
    )

    assert result.idempotent is False
    assert source_checkpoint.read_bytes() == source_before
    assert source_dir.is_dir()
    assert gzip.decompress((result.final_dir / "model.bin.gz").read_bytes()) == (
        b"model:" + source_before
    )
    assert (result.final_dir / "model.ckpt").read_bytes() == (b"clean:" + source_before)
    assert not (result.final_dir / "model.bin").exists()
    assert not (result.final_dir / "log.txt").exists()


@pytest.mark.parametrize("failure", ["export", "probe"])
def test_partial_failure_never_exposes_final_directory(tmp_path, failure):
    source_dir, source_checkpoint = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"
    runner = FakeExportRunner(
        fail_export=failure == "export", fail_probe=failure == "probe"
    )

    with pytest.raises(ExportError):
        HardenedExporter(command_runner=runner).publish(
            request_for(source_dir, destination_root)
        )

    assert source_checkpoint.read_bytes() == b"source-checkpoint"
    assert not (destination_root / "candidate-a").exists()
    assert list(destination_root.iterdir()) == []


def test_manifest_hashes_every_artifact_and_is_canonical(tmp_path):
    source_dir, source_checkpoint = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"
    result = HardenedExporter(command_runner=FakeExportRunner()).publish(
        request_for(source_dir, destination_root)
    )
    manifest_path = result.final_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    canonical = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    assert manifest_bytes == canonical
    assert manifest["modelProbePassed"] is True
    assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256
    assert (
        manifest["sourceCheckpoint"]["sha256"]
        == hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    )

    manifested_paths = [entry["path"] for entry in manifest["files"]]
    assert manifested_paths == ["model.bin.gz", "model.ckpt"]
    for entry in manifest["files"]:
        artifact = result.final_dir / entry["path"]
        assert entry["size"] == artifact.stat().st_size
        assert entry["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_duplicate_name_with_same_manifest_is_idempotent(tmp_path):
    source_dir, _ = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"
    exporter = HardenedExporter(command_runner=FakeExportRunner())
    request = request_for(source_dir, destination_root)

    first = exporter.publish(request)
    command_count = len(exporter.runner.commands)
    second = exporter.publish(request)

    assert first.idempotent is False
    assert second.idempotent is True
    assert len(exporter.runner.commands) == command_count
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.final_dir == second.final_dir
    assert [path.name for path in destination_root.iterdir()] == ["candidate-a"]


def test_unprobed_request_cannot_publish_without_explicit_test_override(
    tmp_path,
):
    source_dir, _ = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"
    runner = FakeExportRunner()
    unprobed = dataclasses.replace(
        request_for(source_dir, destination_root),
        model_probe_command=(),
    )

    with pytest.raises(ExportError) as raised:
        HardenedExporter(command_runner=runner).publish(unprobed)

    assert raised.value.code == "model_probe_required"
    assert runner.commands == []
    assert not (destination_root / "candidate-a").exists()

    unsafe = dataclasses.replace(unprobed, unsafe_allow_unprobed_for_tests=True)
    result = HardenedExporter(command_runner=runner).publish(unsafe)
    manifest = json.loads((result.final_dir / "manifest.json").read_text())
    assert manifest["modelProbePassed"] is False


def test_cli_rejects_missing_probe_before_running_export(tmp_path, capsys):
    source_dir, _ = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"

    exit_code = main(
        [
            "--source-dir",
            str(source_dir),
            "--destination-root",
            str(destination_root),
            "--candidate-name",
            "candidate-a",
            "--model-name",
            "run-candidate-a",
            "--export-command-json",
            '["must-not-run"]',
            "--clean-command-json",
            '["must-not-run"]',
        ]
    )

    assert exit_code == 2
    assert "model_probe_required" in capsys.readouterr().err
    assert not (destination_root / "candidate-a").exists()


def test_duplicate_name_with_different_hash_is_fatal_and_keeps_original(
    tmp_path,
):
    source_dir, source_checkpoint = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"
    exporter = HardenedExporter(command_runner=FakeExportRunner())
    request = request_for(source_dir, destination_root)
    first = exporter.publish(request)
    original_manifest = (first.final_dir / "manifest.json").read_bytes()

    source_checkpoint.write_bytes(b"different-checkpoint")
    with pytest.raises(ExportError) as raised:
        exporter.publish(request)

    assert raised.value.code == "name_collision"
    assert (first.final_dir / "manifest.json").read_bytes() == original_manifest
    assert source_checkpoint.read_bytes() == b"different-checkpoint"
    assert [path.name for path in destination_root.iterdir()] == ["candidate-a"]


def test_existing_publication_detects_probe_contract_contradiction(tmp_path):
    source_dir, _ = source_candidate(tmp_path)
    destination_root = tmp_path / "modelstobetested"
    runner = FakeExportRunner()
    exporter = HardenedExporter(command_runner=runner)
    request = request_for(source_dir, destination_root)
    exporter.publish(request)
    command_count = len(runner.commands)
    changed_probe = dataclasses.replace(
        request,
        model_probe_command=("different-finite-probe", "{model_file}"),
    )

    with pytest.raises(ExportError) as raised:
        exporter.publish(changed_probe)

    assert raised.value.code == "name_collision"
    assert len(runner.commands) == command_count


def test_gated_shell_requires_probe_before_publication(tmp_path):
    python_root = Path(__file__).resolve().parents[1]
    script = python_root / "selfplay" / "export_model_for_selfplay.sh"
    env = os.environ.copy()
    env.pop("KATAGO_MODEL_PROBE_COMMAND_JSON", None)

    result = subprocess.run(
        ["bash", str(script), "test-run", str(tmp_path), "1"],
        cwd=python_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "KATAGO_MODEL_PROBE_COMMAND_JSON is required" in result.stderr
    assert list((tmp_path / "modelstobetested").iterdir()) == []


def test_gated_shell_honors_fresh_controller_export_backpressure(tmp_path):
    python_root = Path(__file__).resolve().parents[1]
    script = python_root / "selfplay" / "export_model_for_selfplay.sh"
    policy_hash = "a" * 64
    status_path = tmp_path / "backpressure.json"
    status = {
        "schema_version": 1,
        "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        "controller_hash": "b" * 64,
        "policy_hash": policy_hash,
        "allowExport": False,
    }
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["KATAGO_PROMOTION_BACKPRESSURE_FILE"] = str(status_path)
    env["KATAGO_PROMOTION_POLICY_HASH"] = policy_hash
    env.pop("KATAGO_MODEL_PROBE_COMMAND_JSON", None)

    result = subprocess.run(
        ["bash", str(script), "test-run", str(tmp_path), "1"],
        cwd=python_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "export paused by controller backpressure" in result.stdout
    assert list((tmp_path / "modelstobetested").iterdir()) == []


def test_gated_shell_rejects_stale_allow_export_status(tmp_path):
    python_root = Path(__file__).resolve().parents[1]
    script = python_root / "selfplay" / "export_model_for_selfplay.sh"
    policy_hash = "a" * 64
    status_path = tmp_path / "stale-backpressure.json"
    status = {
        "schema_version": 1,
        "updated_at_utc": "2020-01-01T00:00:00.000000Z",
        "policy_hash": policy_hash,
        "allowExport": True,
    }
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["KATAGO_PROMOTION_BACKPRESSURE_FILE"] = str(status_path)
    env["KATAGO_PROMOTION_POLICY_HASH"] = policy_hash
    env["KATAGO_MODEL_PROBE_COMMAND_JSON"] = '["finite-model-probe"]'

    result = subprocess.run(
        ["bash", str(script), "test-run", str(tmp_path), "1"],
        cwd=python_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "failing closed" in result.stderr
    assert list((tmp_path / "modelstobetested").iterdir()) == []


def test_gated_shell_archives_verified_source_after_publication(tmp_path):
    python_root = Path(__file__).resolve().parents[1]
    script = python_root / "selfplay" / "export_model_for_selfplay.sh"
    source = tmp_path / "torchmodels_toexport" / "candidate-a"
    source.mkdir(parents=True)
    (source / "model.ckpt").write_bytes(b"retained-checkpoint")
    fake_exporter = tmp_path / "fake_hardened_exporter.py"
    fake_exporter.write_text(
        """
import pathlib
import sys

args = sys.argv[1:]
destination = pathlib.Path(args[args.index("--destination-root") + 1])
name = args[args.index("--candidate-name") + 1]
final = destination / name
final.mkdir()
(final / "manifest.json").write_text("{}\\n", encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["KATAGO_HARDENED_EXPORTER"] = str(fake_exporter)
    env["KATAGO_MODEL_PROBE_COMMAND_JSON"] = '["finite-model-probe"]'

    result = subprocess.run(
        ["bash", str(script), "test-run", str(tmp_path), "1"],
        cwd=python_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not source.exists()
    archived = tmp_path / "torchmodels_exported" / "candidate-a"
    assert (archived / "model.ckpt").read_bytes() == b"retained-checkpoint"
    assert (tmp_path / "modelstobetested" / "candidate-a" / "manifest.json").is_file()
