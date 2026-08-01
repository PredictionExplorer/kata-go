import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from risk_score.promotion_host import (
    HostCommandError,
    atomic_write_json,
    build_trainer_exec,
    capture_process_identity,
    file_sha256,
    same_process,
    tree_manifest,
    worker_start,
    worker_watch_once,
    workers_drain,
)

HAS_PROCFS = Path("/proc/self/stat").is_file()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_trainer_spec_requires_gpu7_and_bucket_limited_exit(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    cwd = tmp_path / "repo"
    cwd.mkdir()
    log = tmp_path / "logs" / "trainer.log"
    spec = tmp_path / "trainer.json"
    value = {
        "contract": "risk-score-host-trainer-spec-v1",
        "cwd": str(cwd.resolve()),
        "argv": [
            "python3",
            "train.py",
            "-stop-when-train-bucket-limited",
            "--checkpoint",
            "{checkpoint_path}",
        ],
        "env": {"CUDA_VISIBLE_DEVICES": "7"},
        "logPath": str(log.resolve()),
    }
    write_json(spec, value)
    argv, selected_cwd, env, selected_log = build_trainer_exec(spec, checkpoint)
    assert argv[-1] == str(checkpoint.resolve())
    assert selected_cwd == cwd.resolve()
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert selected_log == log.resolve()

    value["argv"].remove("-stop-when-train-bucket-limited")
    write_json(tmp_path / "unsafe.json", value)
    with pytest.raises(HostCommandError, match="bucket-limited"):
        build_trainer_exec(tmp_path / "unsafe.json", checkpoint)


@pytest.mark.skipif(not HAS_PROCFS, reason="production identity uses Linux procfs")
def test_process_identity_rejects_pid_reuse_coordinates():
    process = subprocess.Popen(["sleep", "2"], start_new_session=True)
    try:
        identity = capture_process_identity(process.pid)
        assert same_process(identity)
        changed = dict(identity, command_sha256="0" * 64)
        assert not same_process(changed)
    finally:
        process.terminate()
        process.wait()


@pytest.mark.skipif(not HAS_PROCFS, reason="production identity uses Linux procfs")
def test_worker_launch_completion_ack_and_drain_are_replay_safe(tmp_path):
    fake = tmp_path / "katago"
    fake.write_text(
        """#!/usr/bin/env python3
import pathlib,sys,time
args=sys.argv
output=pathlib.Path(args[args.index('-output-dir')+1])
output.mkdir(parents=True,exist_ok=True)
(output/'games.sgfs').write_text('game\\n')
time.sleep(0.25)
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    config = tmp_path / "worker.cfg"
    config.write_text("switchNetsMidGame = false\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    model = models / "model.bin.gz"
    model.write_bytes(b"model")
    output = tmp_path / "output"
    state = tmp_path / "state"
    inbox = tmp_path / "acks"
    result = worker_start(
        state_root=state.resolve(),
        katago=fake.resolve(),
        config=config.resolve(),
        models_dir=models.resolve(),
        output_dir=output.resolve(),
        gpu=0,
        generation="generation-1",
        worker=0,
        phase="canary",
        threads=100,
        model_hash=file_sha256(model),
        selfplay_config_hash=file_sha256(config),
        policy_hash="1" * 64,
        ack_inbox=inbox.resolve(),
        max_games=1,
    )
    assert result["process_identity_verified"] is True
    deadline = time.monotonic() + 5
    while same_process(result["process_identity"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    watched = worker_watch_once(state.resolve(), stable_seconds=0.01)
    assert watched["completed"] == [0]
    report_path = inbox / "generation-1-worker-000.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_manifest, size = tree_manifest(output.resolve())
    assert size > 0
    assert report["output_manifest_hash"] == expected_manifest
    assert report["closed_files"] is True
    assert worker_watch_once(state.resolve(), stable_seconds=0.01)["completed"] == [0]

    drain = tmp_path / "drain.json"
    write_json(
        drain,
        {
            "schema_version": 1,
            "generation_id": "generation-1",
            "candidate_hash": file_sha256(model),
            "source_manifest_hash": expected_manifest,
            "closed_file_manifests": [expected_manifest],
            "process_identities": [result["process_identity"]],
        },
    )
    proof = workers_drain(
        state_root=state.resolve(),
        generation="generation-1",
        manifest_path=drain,
        timeout=1,
    )
    assert proof["quiescent"] is True
    assert proof["closed_file_manifests"] == [expected_manifest]


def test_atomic_json_and_tree_manifest_fail_closed(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 1})
    with pytest.raises(HostCommandError, match="conflicts"):
        atomic_write_json(path, {"value": 2})

    root = tmp_path / "tree"
    root.mkdir()
    (root / "file").write_bytes(b"value")
    first = tree_manifest(root.resolve())
    (root / "file").write_bytes(b"changed")
    assert tree_manifest(root.resolve()) != first
