import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import risk_score.promotion_host as promotion_host
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


@pytest.fixture(autouse=True)
def stable_test_boot(monkeypatch):
    monkeypatch.setattr(
        promotion_host, "_current_boot_id", lambda: "test-boot-id"
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def fake_identity(pid, start_time_ticks=None):
    ticks = pid if start_time_ticks is None else start_time_ticks
    return {
        "pid": pid,
        "start_time_ticks": ticks,
        "command_sha256": f"{pid:064x}"[-64:],
        "process_group_id": pid,
        "boot_id": "test-boot-id",
        "cgroup": "0::/test",
    }


def write_trainer_inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    cwd = tmp_path / "trainer-cwd"
    cwd.mkdir()
    spec = tmp_path / "trainer-spec.json"
    write_json(
        spec,
        {
            "contract": "risk-score-host-trainer-spec-v1",
            "cwd": str(cwd.resolve()),
            "argv": [
                "trainer",
                "-stop-when-train-bucket-limited",
                "{checkpoint_path}",
            ],
            "env": {"CUDA_VISIBLE_DEVICES": "7"},
            "logPath": str((tmp_path / "trainer.log").resolve()),
        },
    )
    return spec.resolve(), checkpoint.resolve()


def trainer_runtime(tmp_path):
    return SimpleNamespace(
        promotion_root=tmp_path / "runtime" / "promotion",
        gpu_lease_config_path=tmp_path / "gpu-lease.json",
    )


def write_trainer_launch_record(
    state,
    spec,
    checkpoint,
    *,
    identity,
    launch_id="launch-1",
    launch_boot_id=None,
):
    trainer_argv, _, _, _ = build_trainer_exec(spec, checkpoint)
    spec_hash = file_sha256(spec)
    argv_hash = promotion_host.canonical_sha256(list(trainer_argv))
    launch_key = promotion_host._trainer_launch_key(
        launch_id=launch_id,
        spec_path=spec,
        spec_sha256=spec_hash,
        checkpoint_path=checkpoint,
        argv_sha256=argv_hash,
    )
    completion = state / "trainer-completions" / f"{launch_id}.json"
    record = {
        "schema_version": 1,
        "role": "trainer",
        "process_identity": identity,
        "launch_boot_id": launch_boot_id,
        "spec_path": str(spec),
        "spec_sha256": spec_hash,
        "checkpoint_path": str(checkpoint),
        "argv_sha256": argv_hash,
        "launch_id": launch_id,
        "launch_key": launch_key,
        "completion_path": str(completion.resolve()),
    }
    write_json(state / "trainer.json", record)
    return record, completion


def write_gpu_lease(runtime, tmp_path, value=None):
    lease_path = tmp_path / "gpu-lease-state.json"
    write_json(
        runtime.gpu_lease_config_path,
        {"paths": {"leaseState": str(lease_path.resolve())}},
    )
    if value is not None:
        write_json(lease_path, value)
    return lease_path


def gpu_identity(identity):
    return {
        "pid": identity["pid"],
        "startTimeTicks": identity["start_time_ticks"],
        "processGroupId": identity["process_group_id"],
        "bootId": identity["boot_id"],
        "commandSha256": identity["command_sha256"],
        "cgroup": identity["cgroup"],
    }


def install_active_process_fakes(
    monkeypatch, *, first_pid=1000, boot_state=None
):
    if boot_state is None:
        boot_state = {"value": "test-boot-id"}
    live = {}
    spawned = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(*_args, **_kwargs):
        process = FakeProcess(first_pid + len(spawned))
        identity = {
            **fake_identity(process.pid),
            "boot_id": boot_state["value"],
        }
        live[process.pid] = identity
        spawned.append(process)
        return process

    def capture(pid):
        if pid not in live:
            raise HostCommandError("gone")
        return dict(live[pid])

    monkeypatch.setattr(promotion_host.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        promotion_host,
        "capture_spawned_process",
        lambda process: capture(process.pid),
    )
    monkeypatch.setattr(promotion_host, "capture_process_identity", capture)
    monkeypatch.setattr(
        promotion_host,
        "same_process",
        lambda identity: identity.get("pid") in live
        and live[identity["pid"]] == identity,
    )
    monkeypatch.setattr(
        promotion_host,
        "_find_active_processes",
        lambda _argv: tuple(live.values()),
    )
    return live, spawned


def active_worker_kwargs(tmp_path, *, retry_budget=2):
    binary = tmp_path / "katago"
    binary.write_bytes(b"binary")
    config = tmp_path / "selfplay.cfg"
    config.write_bytes(b"config")
    models = tmp_path / "models"
    models.mkdir()
    model = models / "model.bin.gz"
    model.write_bytes(b"model")
    return {
        "state_root": (tmp_path / "state").resolve(),
        "katago": binary.resolve(),
        "config": config.resolve(),
        "models_dir": models.resolve(),
        "output_dir": (tmp_path / "output").resolve(),
        "generation": "generation-1",
        "model_hash": file_sha256(model),
        "policy_hash": "1" * 64,
        "worker": 0,
        "threads": 100,
        "retry_budget": retry_budget,
        "current_boot_id": "test-boot-id",
    }


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


def test_trainer_launch_writes_hash_bound_completion_receipt(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    trainer_argv, _, _, _ = build_trainer_exec(spec, checkpoint)
    spec_hash = file_sha256(spec)
    argv_hash = promotion_host.canonical_sha256(list(trainer_argv))
    completion = tmp_path / "completion.json"
    monkeypatch.setattr(
        promotion_host.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    returncode = promotion_host.trainer_launch(
        spec,
        checkpoint,
        completion_path=completion,
        launch_id="launch-1",
        expected_spec_sha256=spec_hash,
        expected_argv_sha256=argv_hash,
    )
    receipt = json.loads(completion.read_text(encoding="utf-8"))
    assert returncode == 0
    assert receipt["contract"] == promotion_host.TRAINER_COMPLETION_CONTRACT
    assert receipt["bucket_limited"] is True
    assert receipt["spec_sha256"] == spec_hash
    assert receipt["argv_sha256"] == argv_hash


def test_trainer_start_persists_launch_before_identity_capture(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    identity_output = tmp_path / "state" / "trainer.json"
    process = SimpleNamespace(pid=999999)
    monkeypatch.setattr(
        promotion_host.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        promotion_host,
        "capture_spawned_process",
        lambda _process: (_ for _ in ()).throw(
            HostCommandError("identity capture failed")
        ),
    )
    monkeypatch.setattr(promotion_host.os, "killpg", lambda *_args: None)
    with pytest.raises(HostCommandError, match="identity capture failed"):
        promotion_host.trainer_start(
            spec_path=spec,
            checkpoint_path=checkpoint,
            identity_output=identity_output,
        )
    launch = json.loads(identity_output.read_text(encoding="utf-8"))
    assert launch["launch_status"] == "starting"
    assert launch["process_identity"] is None
    assert launch["completion_path"].endswith(f"{launch['launch_id']}.json")


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


def test_supervisor_reconcile_once_is_directly_testable(tmp_path, monkeypatch):
    state = tmp_path / "state"
    runtime_config = tmp_path / "runtime.json"
    runtime_config.write_text("{}\n", encoding="utf-8")
    runtime = SimpleNamespace(
        controller=SimpleNamespace(mutation_enabled=True),
        champion_path=tmp_path / "missing-champion.json",
    )
    identities = {
        role: []
        for role in ("selfplay", "shuffler", "trainer", "exporter", "evaluator")
    }
    observed = {}

    monkeypatch.setattr(
        promotion_host,
        "refresh_consumer_identities",
        lambda *_args: {"identities": identities},
    )

    def fake_reconcile_trainer(**kwargs):
        observed.update(kwargs)
        return {"status": "started"}

    monkeypatch.setattr(
        promotion_host, "reconcile_trainer", fake_reconcile_trainer
    )
    monkeypatch.setattr(
        promotion_host,
        "worker_watch_once",
        lambda *_args, **_kwargs: {"running": [0], "completed": []},
    )
    identity = fake_identity(900)
    boot_ready = state / "boot-ready.json"
    promotion_host.publish_boot_ready(
        runtime_config.resolve(),
        boot_ready.resolve(),
        boot_id=identity["boot_id"],
        now=1234.0,
    )
    result = promotion_host.supervisor_reconcile_once(
        runtime=runtime,
        runtime_config=runtime_config,
        boot_ready=boot_ready,
        state_root=state,
        katago=tmp_path / "katago",
        config=tmp_path / "selfplay.cfg",
        trainer_spec=tmp_path / "trainer.json",
        trainer_checkpoint=tmp_path / "checkpoint.ckpt",
        consumer_policy=tmp_path / "consumers.json",
        consumer_state=tmp_path / "consumer-state.json",
        service_identity=identity,
        now=1234.5,
    )

    assert result == {
        "consumers": identities,
        "trainer": {"status": "started"},
        "worker_watch": {"running": [0], "completed": []},
        "active": {"status": "waiting-for-champion-bootstrap"},
        "boot_ready": {"ready": True, "reason": "ready"},
    }
    assert observed["trainer_identities"] == []
    assert observed["now"] == 1234.5
    service = json.loads((state / "service.json").read_text(encoding="utf-8"))
    assert service["process_identity"] == identity
    assert service["updated_at_unix"] == 1234.5


def test_boot_ready_marker_gates_mutation_and_binds_runtime(tmp_path, monkeypatch):
    runtime_config = tmp_path / "runtime.json"
    write_json(runtime_config, {"runtime": "one"})
    marker_path = tmp_path / "state" / "boot-ready.json"
    marker = promotion_host.publish_boot_ready(
        runtime_config.resolve(),
        marker_path.resolve(),
        boot_id="test-boot-id",
        now=10.0,
    )
    assert marker["runtime_config_sha256"] == file_sha256(runtime_config)
    assert promotion_host.parse_args(
        [
            "boot-ready",
            "--runtime-config",
            str(runtime_config),
            "--output",
            str(marker_path),
        ]
    ).command == "boot-ready"

    runtime = SimpleNamespace(
        controller=SimpleNamespace(mutation_enabled=True),
        champion_path=tmp_path / "missing-champion.json",
    )
    identities = {
        role: []
        for role in ("selfplay", "shuffler", "trainer", "exporter", "evaluator")
    }
    monkeypatch.setattr(
        promotion_host,
        "refresh_consumer_identities",
        lambda *_args: {"identities": identities},
    )
    monkeypatch.setattr(
        promotion_host,
        "reconcile_trainer",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("trainer mutation must remain gated")
        ),
    )
    runtime_config.write_text('{"runtime":"two"}\n', encoding="utf-8")
    result = promotion_host.supervisor_reconcile_once(
        runtime=runtime,
        runtime_config=runtime_config.resolve(),
        boot_ready=marker_path.resolve(),
        state_root=(tmp_path / "state").resolve(),
        katago=tmp_path / "katago",
        config=tmp_path / "config",
        trainer_spec=tmp_path / "trainer.json",
        trainer_checkpoint=tmp_path / "checkpoint",
        consumer_policy=tmp_path / "consumers.json",
        consumer_state=tmp_path / "consumer-state.json",
        service_identity=fake_identity(905),
        now=11.0,
    )
    assert result["boot_ready"] == {
        "ready": False,
        "reason": "runtime-hash-mismatch",
    }
    assert result["trainer"]["status"] == "boot-not-ready"
    assert (tmp_path / "state" / "service.json").is_file()


def test_trainer_decisions_are_persisted_with_timestamps(tmp_path, monkeypatch):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    monkeypatch.setattr(
        promotion_host, "_lease_blocks_trainer", lambda _runtime: False
    )
    adopted_state = tmp_path / "adopted-state"
    adopted = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=adopted_state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[fake_identity(901)],
        now=10.0,
    )
    assert adopted["status"] == "adopted"
    observation = json.loads(
        (adopted_state / "trainer-observation.json").read_text(encoding="utf-8")
    )
    assert (observation["decision"], observation["updated_at_unix"]) == (
        "adopted",
        10.0,
    )

    lease_state = tmp_path / "lease-state"
    monkeypatch.setattr(
        promotion_host, "_lease_blocks_trainer", lambda _runtime: True
    )
    lease = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=lease_state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=20.0,
    )
    assert lease["status"] == "lease-handoff"
    observation = json.loads(
        (lease_state / "trainer-observation.json").read_text(encoding="utf-8")
    )
    assert (observation["decision"], observation["updated_at_unix"]) == (
        "lease-handoff",
        20.0,
    )

    monkeypatch.setattr(
        promotion_host, "_lease_blocks_trainer", lambda _runtime: False
    )
    export = runtime.promotion_root.parent / "torchmodels_toexport" / "pending"
    export.mkdir(parents=True)
    export_state = tmp_path / "export-state"
    waiting = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=export_state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=30.0,
    )
    assert waiting["status"] == "waiting-for-export"
    observation = json.loads(
        (export_state / "trainer-observation.json").read_text(encoding="utf-8")
    )
    assert (observation["decision"], observation["updated_at_unix"]) == (
        "waiting-for-export",
        30.0,
    )
    export.rmdir()

    started_identity = fake_identity(902)
    monkeypatch.setattr(
        promotion_host,
        "trainer_start",
        lambda **_kwargs: {
            "process_identity": started_identity,
            "launch_key": "2" * 64,
        },
    )
    started_state = tmp_path / "started-state"
    started = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=started_state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=40.0,
    )
    assert started["status"] == "started"
    observation = json.loads(
        (started_state / "trainer-observation.json").read_text(encoding="utf-8")
    )
    assert (observation["decision"], observation["updated_at_unix"]) == (
        "started",
        40.0,
    )


def test_short_clean_trainer_exit_uses_stable_bounded_backoff(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    state = tmp_path / "state"
    launch_id = "launch-1"
    spec_hash = file_sha256(spec)
    trainer_argv, _, _, _ = build_trainer_exec(spec, checkpoint)
    argv_hash = promotion_host.canonical_sha256(list(trainer_argv))
    launch_key = promotion_host._trainer_launch_key(
        launch_id=launch_id,
        spec_path=spec,
        spec_sha256=spec_hash,
        checkpoint_path=checkpoint,
        argv_sha256=argv_hash,
    )
    completion = state / "trainer-completions" / f"{launch_id}.json"
    write_json(
        state / "trainer.json",
        {
            "schema_version": 1,
            "role": "trainer",
            "process_identity": fake_identity(903),
            "spec_path": str(spec),
            "spec_sha256": spec_hash,
            "checkpoint_path": str(checkpoint),
            "argv_sha256": argv_hash,
            "launch_id": launch_id,
            "launch_key": launch_key,
            "completion_path": str(completion.resolve()),
        },
    )
    write_json(
        completion,
        {
            "schema_version": 1,
            "contract": promotion_host.TRAINER_COMPLETION_CONTRACT,
            "launch_id": launch_id,
            "launch_key": launch_key,
            "spec_path": str(spec),
            "spec_sha256": spec_hash,
            "checkpoint_path": str(checkpoint),
            "argv_sha256": argv_hash,
            "bucket_limited": True,
            "returncode": 0,
            "termination_signal": None,
            "started_at_unix": 99.0,
            "completed_at_unix": 100.0,
            "runtime_seconds": 1.0,
        },
    )
    starts = []
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    monkeypatch.setattr(
        promotion_host, "_lease_blocks_trainer", lambda _runtime: False
    )

    def fake_start(**kwargs):
        starts.append(kwargs)
        return {
            "process_identity": fake_identity(904),
            "launch_key": "4" * 64,
        }

    monkeypatch.setattr(promotion_host, "trainer_start", fake_start)
    first = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=101.0,
        short_lived_seconds=5.0,
        backoff_initial_seconds=10.0,
        backoff_max_seconds=40.0,
    )
    second = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=105.0,
        short_lived_seconds=5.0,
        backoff_initial_seconds=10.0,
        backoff_max_seconds=40.0,
    )
    assert first["restart_not_before_unix"] == 110.0
    assert second["restart_not_before_unix"] == 110.0
    assert starts == []
    observation = json.loads(
        (state / "trainer-observation.json").read_text(encoding="utf-8")
    )
    assert observation["decision_since_unix"] == 101.0

    restarted = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=110.0,
        short_lived_seconds=5.0,
        backoff_initial_seconds=10.0,
        backoff_max_seconds=40.0,
    )
    assert restarted["status"] == "started"
    assert len(starts) == 1


def test_planned_sigint_trainer_exit_does_not_backoff(tmp_path, monkeypatch):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    state = tmp_path / "state"
    record, completion = write_trainer_launch_record(
        state,
        spec,
        checkpoint,
        identity=fake_identity(930),
        launch_boot_id="test-boot-id",
    )
    write_json(
        completion,
        {
            "schema_version": 1,
            "contract": promotion_host.TRAINER_COMPLETION_CONTRACT,
            "launch_id": record["launch_id"],
            "launch_key": record["launch_key"],
            "spec_path": record["spec_path"],
            "spec_sha256": record["spec_sha256"],
            "checkpoint_path": record["checkpoint_path"],
            "argv_sha256": record["argv_sha256"],
            "bucket_limited": True,
            "returncode": -signal.SIGINT,
            "termination_signal": "SIGINT",
            "started_at_unix": 99.0,
            "completed_at_unix": 100.0,
            "runtime_seconds": 1.0,
        },
    )
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    monkeypatch.setattr(
        promotion_host, "_lease_blocks_trainer", lambda _runtime: False
    )
    starts = []
    monkeypatch.setattr(
        promotion_host,
        "trainer_start",
        lambda **kwargs: starts.append(kwargs)
        or {
            "process_identity": fake_identity(931),
            "launch_key": "9" * 64,
        },
    )
    result = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=101.0,
        boot_id="test-boot-id",
        short_lived_seconds=5.0,
        backoff_initial_seconds=10.0,
        backoff_max_seconds=40.0,
    )
    assert result["status"] == "started"
    assert len(starts) == 1
    observation = json.loads(
        (state / "trainer-observation.json").read_text(encoding="utf-8")
    )
    assert "restart_not_before_unix" not in observation
    assert "consecutive_short_clean_exits" not in observation


def test_gpu_restored_trainer_is_adopted_after_launcher_exit(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    state = tmp_path / "state"
    old_launcher = fake_identity(940)
    live = fake_identity(941)
    restored_launcher = {
        **fake_identity(942),
        "process_group_id": live["process_group_id"],
    }
    write_trainer_launch_record(
        state,
        spec,
        checkpoint,
        identity=old_launcher,
        launch_boot_id="test-boot-id",
    )
    lease_path = write_gpu_lease(
        runtime,
        tmp_path,
        {
            "phase": "trainer_running",
            "safetyHalt": False,
            "trainer": gpu_identity(restored_launcher),
            "restoredTrainer": gpu_identity(restored_launcher),
        },
    )
    monkeypatch.setattr(
        promotion_host,
        "same_process",
        lambda identity: identity["pid"] in {
            live["pid"],
            restored_launcher["pid"],
        },
    )
    result = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[live],
        now=50.0,
        boot_id="test-boot-id",
    )
    assert result["status"] == "adopted"
    adopted = json.loads((state / "trainer.json").read_text(encoding="utf-8"))
    assert adopted["process_identity"] == restored_launcher
    assert adopted["observed_trainer_identity"] == live
    assert adopted["adoption_source"] == "gpu-lease-restoredTrainer"
    assert adopted["lease_state_sha256"] == file_sha256(lease_path)
    assert len(list((state / "trainer-interrupted").glob("*.json"))) == 1


@pytest.mark.parametrize("starting", (False, True))
def test_previous_boot_missing_trainer_receipt_archives_and_restarts(
    tmp_path, monkeypatch, starting
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    write_gpu_lease(runtime, tmp_path)
    state = tmp_path / "state"
    old_identity = None if starting else {
        **fake_identity(945),
        "boot_id": "previous-boot-id",
    }
    write_trainer_launch_record(
        state,
        spec,
        checkpoint,
        identity=old_identity,
        launch_boot_id="previous-boot-id",
    )
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    starts = []
    monkeypatch.setattr(
        promotion_host,
        "trainer_start",
        lambda **kwargs: starts.append(kwargs)
        or {
            "process_identity": fake_identity(946),
            "launch_key": "8" * 64,
        },
    )
    result = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=60.0,
        boot_id="test-boot-id",
    )
    assert result["status"] == "started"
    assert len(starts) == 1
    archive = list((state / "trainer-interrupted").glob("*.json"))
    assert len(archive) == 1
    archived = json.loads(archive[0].read_text(encoding="utf-8"))
    assert archived["reason"] == "previous-boot-interrupted"
    assert archived["record_sha256"] == promotion_host.canonical_sha256(
        archived["record"]
    )


def test_same_boot_starting_trainer_without_identity_fails_closed(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    write_gpu_lease(runtime, tmp_path)
    state = tmp_path / "state"
    write_trainer_launch_record(
        state,
        spec,
        checkpoint,
        identity=None,
        launch_boot_id="test-boot-id",
    )
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    with pytest.raises(HostCommandError, match="no verifiable identity"):
        promotion_host.reconcile_trainer(
            runtime=runtime,
            state_root=state,
            trainer_spec=spec,
            checkpoint=checkpoint,
            consumer_policy=tmp_path / "unused-policy.json",
            consumer_state=tmp_path / "unused-consumers.json",
            trainer_identities=[],
            now=61.0,
            boot_id="test-boot-id",
        )
    assert not (state / "trainer-interrupted").exists()


def test_same_boot_dead_trainer_is_archived_and_bounded_before_restart(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    write_gpu_lease(runtime, tmp_path)
    state = tmp_path / "state"
    write_trainer_launch_record(
        state,
        spec,
        checkpoint,
        identity=fake_identity(948),
        launch_boot_id="test-boot-id",
    )
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    monkeypatch.setattr(
        promotion_host,
        "capture_process_identity",
        lambda _pid: (_ for _ in ()).throw(HostCommandError("gone")),
    )
    starts = []
    monkeypatch.setattr(
        promotion_host,
        "trainer_start",
        lambda **kwargs: starts.append(kwargs)
        or {
            "process_identity": fake_identity(949),
            "launch_key": "7" * 64,
        },
    )
    first = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=61.0,
        boot_id="test-boot-id",
        backoff_initial_seconds=10.0,
        backoff_max_seconds=40.0,
    )
    assert first["restart_not_before_unix"] == 71.0
    assert starts == []
    second = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=71.0,
        boot_id="test-boot-id",
        backoff_initial_seconds=10.0,
        backoff_max_seconds=40.0,
    )
    assert second["status"] == "started"
    assert len(starts) == 1
    archive = json.loads(
        next((state / "trainer-interrupted").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert archive["reason"] == "same-boot-dead-interrupted"


def test_running_trainer_keeps_launcher_identity_and_completion_binding(
    tmp_path, monkeypatch
):
    spec, checkpoint = write_trainer_inputs(tmp_path)
    runtime = trainer_runtime(tmp_path)
    state = tmp_path / "state"
    launcher = fake_identity(950)
    child = fake_identity(951)
    record = {
        "schema_version": 1,
        "role": "trainer",
        "process_identity": launcher,
        "spec_path": str(spec),
        "spec_sha256": file_sha256(spec),
        "checkpoint_path": str(checkpoint),
        "argv_sha256": "1" * 64,
        "launch_id": "launch-1",
        "launch_key": "2" * 64,
        "completion_path": str(
            (state / "trainer-completions" / "launch-1.json").resolve()
        ),
    }
    write_json(state / "trainer.json", record)
    monkeypatch.setattr(
        promotion_host,
        "same_process",
        lambda identity: identity["pid"] == launcher["pid"],
    )
    monkeypatch.setattr(
        promotion_host, "_lease_blocks_trainer", lambda _runtime: False
    )
    adopted = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[child],
        now=50.0,
    )
    assert adopted["status"] == "adopted"
    assert json.loads(
        (state / "trainer.json").read_text(encoding="utf-8")
    ) == record

    waiting = promotion_host.reconcile_trainer(
        runtime=runtime,
        state_root=state,
        trainer_spec=spec,
        checkpoint=checkpoint,
        consumer_policy=tmp_path / "unused-policy.json",
        consumer_state=tmp_path / "unused-consumers.json",
        trainer_identities=[],
        now=51.0,
    )
    assert waiting["status"] == "waiting-for-completion"


def test_active_worker_retries_are_bounded_and_quarantined(
    tmp_path, monkeypatch
):
    binary = tmp_path / "katago"
    binary.write_bytes(b"binary")
    config = tmp_path / "selfplay.cfg"
    config.write_bytes(b"config")
    models = tmp_path / "models"
    models.mkdir()
    model = models / "model.bin.gz"
    model.write_bytes(b"model")
    state = tmp_path / "state"
    live, spawned = install_active_process_fakes(monkeypatch)
    kwargs = {
        "state_root": state.resolve(),
        "katago": binary.resolve(),
        "config": config.resolve(),
        "models_dir": models.resolve(),
        "output_dir": (tmp_path / "output").resolve(),
        "generation": "generation-1",
        "model_hash": file_sha256(model),
        "policy_hash": "1" * 64,
        "worker": 0,
        "threads": 100,
        "retry_budget": 2,
        "current_boot_id": "test-boot-id",
    }
    first = promotion_host._start_active_worker(**kwargs)
    assert first["pid"] == 1000
    live.pop(first["pid"])
    second = promotion_host._start_active_worker(**kwargs)
    assert second["pid"] == 1001
    live.pop(second["pid"])
    third = promotion_host._start_active_worker(**kwargs)
    assert third["pid"] == 1002
    live.pop(third["pid"])
    with pytest.raises(HostCommandError, match="exhausted retry budget"):
        promotion_host._start_active_worker(**kwargs)
    assert len(spawned) == 3
    marker = json.loads(
        (
            state
            / "active-quarantine"
            / "generation-1"
            / "worker-000.json"
        ).read_text(encoding="utf-8")
    )
    assert marker["reason"] == "retry-budget-exhausted"
    assert marker["retry_count"] == 2
    with pytest.raises(HostCommandError, match="is quarantined"):
        promotion_host._start_active_worker(**kwargs)
    assert len(spawned) == 3


@pytest.mark.parametrize(
    "failure_point",
    (
        "after-retire-intent",
        "after-retire",
        "after-retired-state",
        "after-spawning-state",
        "after-spawn",
        "after-spawned-state",
        "after-record",
        "after-running-state",
    ),
)
def test_active_worker_retry_transition_replays_after_crash(
    tmp_path, monkeypatch, failure_point
):
    kwargs = active_worker_kwargs(tmp_path)
    live, _spawned = install_active_process_fakes(monkeypatch, first_pid=1300)
    first = promotion_host._start_active_worker(**kwargs, now=1.0)
    live.pop(first["pid"])
    failed = False

    def fail_once(step):
        nonlocal failed
        if step == failure_point and not failed:
            failed = True
            raise RuntimeError(f"crash at {step}")

    with pytest.raises(RuntimeError, match=failure_point):
        promotion_host._start_active_worker(
            **kwargs, now=2.0, failure_hook=fail_once
        )
    recovered = promotion_host._start_active_worker(**kwargs, now=3.0)
    assert recovered["pid"] in live
    record = json.loads(
        (
            kwargs["state_root"]
            / "active"
            / "generation-1"
            / "worker-000.json"
        ).read_text(encoding="utf-8")
    )
    retry = json.loads(
        (
            kwargs["state_root"]
            / "active-retries"
            / "generation-1"
            / "worker-000.json"
        ).read_text(encoding="utf-8")
    )
    assert record["process_identity"] == recovered
    assert retry["status"] == "running"
    assert retry["current_process_identity_sha256"] == (
        promotion_host.canonical_sha256(recovered)
    )


@pytest.mark.parametrize("transition", ("reboot", "service-stop"))
def test_planned_active_worker_restart_does_not_consume_budget(
    tmp_path, monkeypatch, transition
):
    kwargs = active_worker_kwargs(tmp_path)
    live, _spawned = install_active_process_fakes(monkeypatch, first_pid=1400)
    first = promotion_host._start_active_worker(**kwargs, now=1.0)
    record_path = (
        kwargs["state_root"]
        / "active"
        / "generation-1"
        / "worker-000.json"
    )
    if transition == "reboot":
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["process_identity"]["boot_id"] = "previous-boot-id"
        write_json(record_path, record)
    else:
        promotion_host.publish_planned_stop(
            kwargs["state_root"], fake_identity(1499), now=2.0
        )
    live.pop(first["pid"])
    restarted = promotion_host._start_active_worker(**kwargs, now=3.0)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    retry = json.loads(
        (
            kwargs["state_root"]
            / "active-retries"
            / "generation-1"
            / "worker-000.json"
        ).read_text(encoding="utf-8")
    )
    assert restarted["pid"] in live
    assert record["restart_attempt"] == 0
    assert retry["retry_count"] == 0
    assert len(list((kwargs["state_root"] / "active-retired").glob("*/*.json"))) == 1


@pytest.mark.parametrize(
    "failure_point", ("after-retire-intent", "after-retired-state")
)
def test_reboot_during_worker_retry_resets_failure_budget(
    tmp_path, monkeypatch, failure_point
):
    kwargs = active_worker_kwargs(tmp_path)
    boot_state = {"value": "test-boot-id"}
    live, _spawned = install_active_process_fakes(
        monkeypatch, first_pid=1450, boot_state=boot_state
    )
    first = promotion_host._start_active_worker(**kwargs, now=1.0)
    live.pop(first["pid"])

    def crash_during_retry(step):
        if step == failure_point:
            raise RuntimeError("reboot")

    with pytest.raises(RuntimeError, match="reboot"):
        promotion_host._start_active_worker(
            **kwargs, now=2.0, failure_hook=crash_during_retry
        )
    boot_state["value"] = "next-boot-id"
    rebooted_kwargs = {**kwargs, "current_boot_id": "next-boot-id"}
    restarted = promotion_host._start_active_worker(
        **rebooted_kwargs, now=3.0
    )
    record = json.loads(
        (
            kwargs["state_root"]
            / "active"
            / "generation-1"
            / "worker-000.json"
        ).read_text(encoding="utf-8")
    )
    assert restarted["boot_id"] == "next-boot-id"
    assert record["restart_attempt"] == 0


def test_healthy_active_worker_resets_consecutive_failures(
    tmp_path, monkeypatch
):
    kwargs = active_worker_kwargs(tmp_path)
    live, _spawned = install_active_process_fakes(monkeypatch, first_pid=1500)
    first = promotion_host._start_active_worker(**kwargs, now=1.0)
    live.pop(first["pid"])
    second = promotion_host._start_active_worker(**kwargs, now=2.0)
    record_path = (
        kwargs["state_root"]
        / "active"
        / "generation-1"
        / "worker-000.json"
    )
    assert json.loads(record_path.read_text(encoding="utf-8"))[
        "restart_attempt"
    ] == 1
    promotion_host._start_active_worker(**kwargs, now=63.0)
    assert json.loads(record_path.read_text(encoding="utf-8"))[
        "restart_attempt"
    ] == 0
    retry_path = (
        kwargs["state_root"]
        / "active-retries"
        / "generation-1"
        / "worker-000.json"
    )
    assert not retry_path.exists()
    live.pop(second["pid"])
    third = promotion_host._start_active_worker(**kwargs, now=64.0)
    assert third["pid"] in live
    assert json.loads(record_path.read_text(encoding="utf-8"))[
        "restart_attempt"
    ] == 1


@pytest.mark.parametrize(
    "failure_point",
    (
        "after-healthy-reset-intent",
        "after-healthy-record",
        "after-healthy-reset",
    ),
)
def test_active_worker_healthy_reset_replays_after_crash(
    tmp_path, monkeypatch, failure_point
):
    kwargs = active_worker_kwargs(tmp_path)
    live, _spawned = install_active_process_fakes(monkeypatch, first_pid=1550)
    first = promotion_host._start_active_worker(**kwargs, now=1.0)
    live.pop(first["pid"])
    second = promotion_host._start_active_worker(**kwargs, now=2.0)
    failed = False

    def fail_once(step):
        nonlocal failed
        if step == failure_point and not failed:
            failed = True
            raise RuntimeError(f"crash at {step}")

    with pytest.raises(RuntimeError, match=failure_point):
        promotion_host._start_active_worker(
            **kwargs, now=63.0, failure_hook=fail_once
        )
    recovered = promotion_host._start_active_worker(**kwargs, now=64.0)
    assert recovered == second
    record_path = (
        kwargs["state_root"]
        / "active"
        / "generation-1"
        / "worker-000.json"
    )
    assert json.loads(record_path.read_text(encoding="utf-8"))[
        "restart_attempt"
    ] == 0
    assert not (
        kwargs["state_root"]
        / "active-retries"
        / "generation-1"
        / "worker-000.json"
    ).exists()


def test_active_worker_hash_change_is_not_retried(tmp_path, monkeypatch):
    binary = tmp_path / "katago"
    binary.write_bytes(b"binary")
    config = tmp_path / "selfplay.cfg"
    config.write_bytes(b"config")
    models = tmp_path / "models"
    models.mkdir()
    model = models / "model.bin.gz"
    model.write_bytes(b"model")
    spawned = []

    class FakeProcess:
        pid = 1100

    monkeypatch.setattr(
        promotion_host.subprocess,
        "Popen",
        lambda *_args, **_kwargs: spawned.append(FakeProcess()) or spawned[-1],
    )
    monkeypatch.setattr(
        promotion_host,
        "capture_spawned_process",
        lambda process: fake_identity(process.pid),
    )
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    kwargs = {
        "state_root": (tmp_path / "state").resolve(),
        "katago": binary.resolve(),
        "config": config.resolve(),
        "models_dir": models.resolve(),
        "output_dir": (tmp_path / "output").resolve(),
        "generation": "generation-1",
        "model_hash": file_sha256(model),
        "policy_hash": "1" * 64,
        "worker": 0,
        "threads": 100,
    }
    promotion_host._start_active_worker(**kwargs)
    config.write_bytes(b"changed-config")
    with pytest.raises(HostCommandError, match="record contradicts launch"):
        promotion_host._start_active_worker(**kwargs)
    assert len(spawned) == 1


def test_canary_worker_disappearance_is_not_auto_retried(
    tmp_path, monkeypatch
):
    binary = tmp_path / "katago"
    binary.write_bytes(b"binary")
    config = tmp_path / "selfplay.cfg"
    config.write_bytes(b"config")
    models = tmp_path / "models"
    models.mkdir()
    model = models / "model.bin.gz"
    model.write_bytes(b"model")
    spawned = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(*_args, **_kwargs):
        process = FakeProcess(1200 + len(spawned))
        spawned.append(process)
        return process

    monkeypatch.setattr(promotion_host.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        promotion_host,
        "capture_spawned_process",
        lambda process: fake_identity(process.pid),
    )
    monkeypatch.setattr(promotion_host, "same_process", lambda _identity: False)
    kwargs = {
        "state_root": (tmp_path / "state").resolve(),
        "katago": binary.resolve(),
        "config": config.resolve(),
        "models_dir": models.resolve(),
        "output_dir": (tmp_path / "output").resolve(),
        "gpu": 0,
        "generation": "generation-1",
        "worker": 0,
        "phase": "canary",
        "threads": 100,
        "model_hash": file_sha256(model),
        "selfplay_config_hash": file_sha256(config),
        "policy_hash": "1" * 64,
        "ack_inbox": (tmp_path / "acks").resolve(),
        "max_games": 1,
    }
    worker_start(**kwargs)
    assert len(spawned) == 2
    with pytest.raises(HostCommandError, match="without completion receipt"):
        worker_start(**kwargs)
    assert len(spawned) == 2
