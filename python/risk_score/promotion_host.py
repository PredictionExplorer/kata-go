#!/usr/bin/env python3
"""Fail-closed host process supervision for checkpoint promotion."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


TRAINER_SPEC_CONTRACT = "risk-score-host-trainer-spec-v1"
WORKER_RECORD_CONTRACT = "risk-score-host-worker-record-v1"
CONSUMER_SPEC_CONTRACT = "risk-score-host-consumer-spec-v1"


class HostCommandError(RuntimeError):
    pass


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


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(value) + "\n").encode("utf-8")
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise HostCommandError(f"immutable JSON conflicts: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, target)
        _fsync_dir(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_dir(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_canonical_json(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HostCommandError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostCommandError(f"cannot load {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise HostCommandError(f"{role} must have an object root")
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise HostCommandError(f"{role} must be canonical newline-terminated JSON")
    return value


def _regular_absolute(path: Path, role: str, *, directory: bool = False) -> Path:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink():
        raise HostCommandError(f"{role} must be an absolute non-symlink path")
    valid = source.is_dir() if directory else source.is_file()
    if not valid:
        raise HostCommandError(f"{role} does not exist: {source}")
    return source


def _proc_start_ticks(pid: int) -> int:
    text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    close_paren = text.rfind(")")
    fields = text[close_paren + 1 :].split()
    return int(fields[19])


def capture_process_identity(pid: int) -> Dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise HostCommandError("PID must be a positive integer")
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes()
        if not command:
            raise OSError("empty command")
        identity = {
            "pid": pid,
            "start_time_ticks": _proc_start_ticks(pid),
            "command_sha256": hashlib.sha256(command).hexdigest(),
            "process_group_id": os.getpgid(pid),
            "boot_id": Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="utf-8")
            .strip(),
            "cgroup": (proc / "cgroup").read_text(encoding="utf-8").strip(),
        }
    except (OSError, ValueError) as exc:
        raise HostCommandError(f"cannot capture process identity for {pid}") from exc
    return identity


def same_process(identity: Mapping[str, Any]) -> bool:
    try:
        current = capture_process_identity(int(identity["pid"]))
    except (HostCommandError, KeyError, TypeError, ValueError):
        return False
    return all(
        current.get(key) == identity.get(key)
        for key in (
            "pid",
            "start_time_ticks",
            "command_sha256",
            "process_group_id",
            "boot_id",
            "cgroup",
        )
    )


def capture_spawned_process(
    process: subprocess.Popen[Any], *, timeout: float = 5.0
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Optional[BaseException] = None
    while True:
        if process.poll() is not None:
            raise HostCommandError(
                f"spawned process exited before identity capture: {process.returncode}"
            ) from last_error
        try:
            return capture_process_identity(process.pid)
        except HostCommandError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise HostCommandError("spawned process identity capture timed out") from last_error
        time.sleep(0.01)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_verified(identity: Mapping[str, Any], sig: int) -> None:
    if sig == signal.SIGSTOP:
        raise HostCommandError("SIGSTOP is prohibited")
    if not same_process(identity):
        raise HostCommandError("process identity changed before signal")
    pgid = int(identity["process_group_id"])
    os.killpg(pgid, sig)


def _wait_group_exit(identity: Mapping[str, Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    pgid = int(identity["process_group_id"])
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            raise HostCommandError("process group did not exit before timeout")
        time.sleep(0.25)


def tree_manifest(path: Path) -> Tuple[str, int]:
    root = _regular_absolute(path, "manifest tree", directory=True)
    rows = []
    total = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(directories):
            if (directory_path / name).is_symlink():
                raise HostCommandError("manifest tree contains a symlink directory")
        for name in sorted(files):
            child = directory_path / name
            if child.is_symlink() or not child.is_file():
                raise HostCommandError("manifest tree contains a nonregular file")
            stat_result = child.stat()
            rows.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "size": stat_result.st_size,
                    "sha256": file_sha256(child),
                }
            )
            total += stat_result.st_size
    return canonical_sha256({"schemaVersion": 1, "files": rows}), total


def _stable_tree(path: Path, seconds: float) -> Tuple[str, int]:
    first = tree_manifest(path)
    time.sleep(seconds)
    second = tree_manifest(path)
    if first != second:
        raise HostCommandError("worker output changed during close-stability window")
    return second


def build_trainer_exec(
    spec_path: Path, checkpoint_path: Path
) -> Tuple[Tuple[str, ...], Path, Dict[str, str], Path]:
    spec = _load_canonical_json(spec_path, "trainer launch spec")
    if spec.get("contract") != TRAINER_SPEC_CONTRACT:
        raise HostCommandError("trainer launch spec contract is unsupported")
    raw_argv = spec.get("argv")
    if not isinstance(raw_argv, list) or not raw_argv:
        raise HostCommandError("trainer launch spec argv is missing")
    checkpoint = _regular_absolute(checkpoint_path, "trainer checkpoint")
    argv = tuple(str(part).replace("{checkpoint_path}", str(checkpoint)) for part in raw_argv)
    if "-stop-when-train-bucket-limited" not in argv:
        raise HostCommandError("trainer launch must be bucket-limited")
    cwd = _regular_absolute(Path(spec.get("cwd", "")), "trainer cwd", directory=True)
    log_path = Path(spec.get("logPath", ""))
    if not log_path.is_absolute() or log_path.is_symlink():
        raise HostCommandError("trainer log path must be absolute and non-symlink")
    environment = dict(os.environ)
    raw_env = spec.get("env", {})
    if not isinstance(raw_env, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_env.items()
    ):
        raise HostCommandError("trainer environment must be a string map")
    environment.update(raw_env)
    if environment.get("CUDA_VISIBLE_DEVICES") != "7":
        raise HostCommandError("trainer must be pinned to physical GPU 7")
    return argv, cwd, environment, log_path


def trainer_launch(spec_path: Path, checkpoint_path: Path) -> int:
    argv, cwd, environment, log_path = build_trainer_exec(spec_path, checkpoint_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=environment,
            stdout=log,
            stderr=log,
            check=False,
            shell=False,
        )
    return completed.returncode


def trainer_start(
    *, spec_path: Path, checkpoint_path: Path, identity_output: Path
) -> Mapping[str, Any]:
    argv = (
        sys.executable,
        "-m",
        "risk_score.promotion_host",
        "trainer-launch",
        "--spec",
        str(spec_path),
        "--checkpoint",
        str(checkpoint_path),
    )
    process = subprocess.Popen(
        argv,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        shell=False,
    )
    try:
        identity = capture_spawned_process(process)
        value = {
            "schema_version": 1,
            "role": "trainer",
            "process_identity": identity,
            "spec_path": str(Path(spec_path).resolve()),
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
        }
        atomic_replace_json(identity_output, value)
        return value
    except BaseException:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGTERM)
        raise


def trainer_drain(
    *, pid: int, process_group: int, checkpoint: Path, timeout: float
) -> Mapping[str, Any]:
    checkpoint_path = _regular_absolute(checkpoint, "trainer checkpoint")
    identity = capture_process_identity(pid)
    if identity["process_group_id"] != process_group:
        raise HostCommandError("trainer process group does not match captured identity")
    before = file_sha256(checkpoint_path)
    _signal_verified(identity, signal.SIGINT)
    _wait_group_exit(identity, timeout)
    deadline = time.monotonic() + timeout
    previous = None
    stable_since = None
    while True:
        stat_result = checkpoint_path.stat()
        current = (stat_result.st_size, stat_result.st_mtime_ns)
        if current != previous:
            previous = current
            stable_since = time.monotonic()
        if stable_since is not None and time.monotonic() - stable_since >= 2.0:
            break
        if time.monotonic() >= deadline:
            raise HostCommandError("trainer checkpoint did not stabilize")
        time.sleep(0.25)
    return {
        "quiescent": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": file_sha256(checkpoint_path),
    }


def _worker_record_path(state_root: Path, generation: str, worker: int) -> Path:
    return Path(state_root) / "workers" / generation / f"worker-{worker:03d}.json"


def worker_run(intent_path: Path, completion_path: Path) -> int:
    intent = _load_canonical_json(intent_path, "worker run intent")
    argv = intent.get("argv")
    if not isinstance(argv, list) or not argv:
        raise HostCommandError("worker run intent has no argv")
    completed = subprocess.run(argv, check=False, shell=False)
    receipt = {
        "schema_version": 1,
        "generation_id": intent["generation_id"],
        "worker_id": intent["worker_id"],
        "returncode": completed.returncode,
    }
    atomic_write_json(completion_path, receipt)
    return completed.returncode


def _count_sgfs_games(root: Path) -> int:
    count = 0
    for path in root.rglob("*.sgfs"):
        if path.is_symlink() or not path.is_file():
            raise HostCommandError("worker output contains invalid SGFS")
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def _complete_worker_record(record_path: Path, stable_seconds: float) -> bool:
    record = _load_canonical_json(record_path, "worker record")
    identity = record.get("process_identity")
    if same_process(identity):
        return False
    completion_process = Path(record["run_completion"])
    if not completion_process.is_file():
        raise HostCommandError("worker disappeared without a run completion receipt")
    process_receipt = _load_canonical_json(
        completion_process, "worker run completion"
    )
    if (
        process_receipt.get("returncode") != 0
        or process_receipt.get("generation_id") != record["generation_id"]
        or process_receipt.get("worker_id") != record["worker_id"]
    ):
        raise HostCommandError("worker process did not complete successfully")
    game_count = _count_sgfs_games(Path(record["output_dir"]))
    if game_count < record["max_games"]:
        raise HostCommandError(
            f"worker completed only {game_count}/{record['max_games']} games"
        )
    manifest_hash, total = _stable_tree(
        Path(record["output_dir"]), stable_seconds
    )
    if total <= 0:
        raise HostCommandError("completed worker produced no output")
    report = {
        "schema_version": 1,
        "finalized": True,
        "generation_id": record["generation_id"],
        "worker_id": record["worker_id"],
        "model_hash": record["model_hash"],
        "selfplay_config_hash": record["selfplay_config_hash"],
        "policy_hash": record["policy_hash"],
        "threads": record["threads"],
        "output_manifest_hash": manifest_hash,
        "closed_files": True,
        "process_identity": identity,
    }
    report_path = (
        Path(record["ack_inbox"])
        / f"{record['generation_id']}-worker-{record['worker_id']:03d}.json"
    )
    atomic_write_json(report_path, report)
    completion = record_path.with_name(record_path.stem + ".complete.json")
    atomic_write_json(
        completion,
        {
            "schema_version": 1,
            "generation_id": record["generation_id"],
            "worker_id": record["worker_id"],
            "ack_report": str(report_path),
            "output_manifest_hash": manifest_hash,
            "game_count": game_count,
        },
    )
    return True


def worker_monitor(record_path: Path, *, stable_seconds: float) -> Mapping[str, Any]:
    record = _load_canonical_json(record_path, "worker record")
    identity = record["process_identity"]
    while same_process(identity):
        time.sleep(0.5)
    completed = _complete_worker_record(record_path, stable_seconds)
    return {"completed": completed, "worker_id": record["worker_id"]}


def worker_start(
    *,
    state_root: Path,
    katago: Path,
    config: Path,
    models_dir: Path,
    output_dir: Path,
    gpu: int,
    generation: str,
    worker: int,
    phase: str,
    threads: int,
    model_hash: str,
    selfplay_config_hash: str,
    policy_hash: str,
    ack_inbox: Path,
    max_games: int,
) -> Mapping[str, Any]:
    binary = _regular_absolute(katago, "KataGo binary")
    config_path = _regular_absolute(config, "self-play config")
    model_root = _regular_absolute(models_dir, "immutable model directory", directory=True)
    if file_sha256(config_path) != selfplay_config_hash:
        raise HostCommandError("self-play config hash mismatch")
    model_path = _regular_absolute(model_root / "model.bin.gz", "worker model")
    if file_sha256(model_path) != model_hash:
        raise HostCommandError("worker model hash mismatch")
    if not (0 <= gpu <= 6 and 0 <= worker <= 6):
        raise HostCommandError("worker/GPU must be in range 0..6")
    if threads != 100 or max_games <= 0:
        raise HostCommandError("production worker requires 100 threads and finite games")
    output = Path(output_dir)
    if not output.is_absolute() or output.is_symlink():
        raise HostCommandError("worker output path must be absolute and non-symlink")
    output.mkdir(parents=True, exist_ok=True)
    state = Path(state_root)
    state.mkdir(parents=True, exist_ok=True)
    ack = Path(ack_inbox)
    ack.mkdir(parents=True, exist_ok=True)
    record_path = _worker_record_path(state, generation, worker)
    expected_key = canonical_sha256(
        {
            "binary": file_sha256(binary),
            "config": selfplay_config_hash,
            "model": model_hash,
            "output": str(output),
            "gpu": gpu,
            "generation": generation,
            "worker": worker,
            "phase": phase,
            "threads": threads,
            "maxGames": max_games,
            "policy": policy_hash,
        }
    )
    if record_path.exists():
        record = _load_canonical_json(record_path, "worker record")
        if record.get("supervisor_key") != expected_key:
            raise HostCommandError("existing worker record contradicts requested launch")
        identity = record.get("process_identity")
        if same_process(identity):
            return {
                "process_identity": identity,
                "process_identity_verified": True,
                "reused": True,
            }
        if not record_path.with_name(record_path.stem + ".complete.json").is_file():
            raise HostCommandError("recorded worker exited without completion receipt")
        return {
            "process_identity": identity,
            "process_identity_verified": True,
            "reused": True,
            "completed": True,
        }
    log_path = state / "logs" / generation / f"worker-{worker:03d}-{phase}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab", buffering=0)
    argv = (
        str(binary),
        "selfplay",
        "-max-games-total",
        str(max_games),
        "-output-dir",
        str(output),
        "-models-dir",
        str(model_root),
        "-config",
        str(config_path),
        "-override-config",
        f"cudaDeviceToUseModel0Thread0={gpu},numGameThreads={threads},switchNetsMidGame=false",
    )
    run_intent = record_path.with_name(record_path.stem + ".run-intent.json")
    run_completion = record_path.with_name(
        record_path.stem + ".run-completion.json"
    )
    atomic_write_json(
        run_intent,
        {
            "schema_version": 1,
            "generation_id": generation,
            "worker_id": worker,
            "argv": list(argv),
        },
    )
    wrapper_argv = (
        sys.executable,
        "-m",
        "risk_score.promotion_host",
        "worker-run",
        "--intent",
        str(run_intent),
        "--completion",
        str(run_completion),
    )
    process = subprocess.Popen(
        wrapper_argv,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=log,
        stderr=log,
        start_new_session=True,
        shell=False,
    )
    log.close()
    try:
        identity = capture_spawned_process(process)
    except BaseException:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        raise
    record = {
        "schema_version": 1,
        "contract": WORKER_RECORD_CONTRACT,
        "supervisor_key": expected_key,
        "generation_id": generation,
        "worker_id": worker,
        "phase": phase,
        "model_hash": model_hash,
        "selfplay_config_hash": selfplay_config_hash,
        "policy_hash": policy_hash,
        "threads": threads,
        "max_games": max_games,
        "output_dir": str(output),
        "ack_inbox": str(ack),
        "argv": list(argv),
        "wrapper_argv": list(wrapper_argv),
        "run_completion": str(run_completion),
        "process_identity": identity,
        "process_identity_verified": True,
    }
    try:
        atomic_write_json(record_path, record)
    except BaseException:
        with contextlib.suppress(OSError):
            os.killpg(identity["process_group_id"], signal.SIGTERM)
        raise
    monitor_log = (
        state / "logs" / generation / f"worker-{worker:03d}-{phase}-monitor.log"
    )
    monitor = monitor_log.open("ab", buffering=0)
    subprocess.Popen(
        (
            sys.executable,
            "-m",
            "risk_score.promotion_host",
            "worker-monitor",
            "--record",
            str(record_path),
            "--stable-seconds",
            "2",
        ),
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=monitor,
        stderr=monitor,
        start_new_session=True,
        shell=False,
    )
    monitor.close()
    return {
        "process_identity": identity,
        "process_identity_verified": True,
        "reused": False,
    }


def worker_watch_once(state_root: Path, *, stable_seconds: float) -> Mapping[str, Any]:
    state = _regular_absolute(state_root, "worker state root", directory=True)
    completed = []
    running = []
    for record_path in sorted((state / "workers").glob("*/*.json")):
        if record_path.name.endswith(
            (".complete.json", ".run-intent.json", ".run-completion.json")
        ):
            continue
        record = _load_canonical_json(record_path, "worker record")
        if record.get("contract") != WORKER_RECORD_CONTRACT:
            raise HostCommandError("worker record contract is unsupported")
        identity = record.get("process_identity")
        if same_process(identity):
            running.append(record["worker_id"])
            continue
        completion = record_path.with_name(record_path.stem + ".complete.json")
        if completion.is_file():
            completed.append(record["worker_id"])
            continue
        if _complete_worker_record(record_path, stable_seconds):
            completed.append(record["worker_id"])
    return {"running": running, "completed": completed}


def workers_drain(
    *, state_root: Path, generation: str, manifest_path: Path, timeout: float
) -> Mapping[str, Any]:
    plan = _load_canonical_json(manifest_path, "worker drain plan")
    if plan.get("generation_id") != generation:
        raise HostCommandError("worker drain generation mismatch")
    for record_path in sorted((Path(state_root) / "workers" / generation).glob("*.json")):
        if record_path.name.endswith(
            (".complete.json", ".run-intent.json", ".run-completion.json")
        ):
            continue
        record = _load_canonical_json(record_path, "worker record")
        identity = record["process_identity"]
        if same_process(identity):
            _signal_verified(identity, signal.SIGINT)
            _wait_group_exit(identity, timeout)
    return {
        "quiescent": True,
        "closed_file_manifests": plan["closed_file_manifests"],
        "process_identities": plan["process_identities"],
    }


def _active_record_path(state_root: Path, generation: str, worker: int) -> Path:
    return Path(state_root) / "active" / generation / f"worker-{worker:03d}.json"


def _start_active_worker(
    *,
    state_root: Path,
    katago: Path,
    config: Path,
    models_dir: Path,
    output_dir: Path,
    generation: str,
    model_hash: str,
    policy_hash: str,
    worker: int,
    threads: int,
) -> Mapping[str, Any]:
    binary = _regular_absolute(katago, "KataGo binary")
    config_path = _regular_absolute(config, "self-play config")
    model_root = _regular_absolute(models_dir, "active model directory", directory=True)
    model = _regular_absolute(model_root / "model.bin.gz", "active model")
    if file_sha256(model) != model_hash:
        raise HostCommandError("active worker model identity is invalid")
    if not (0 <= worker <= 6) or threads != 100:
        raise HostCommandError("active worker topology must be 7x100")
    record_path = _active_record_path(state_root, generation, worker)
    key = canonical_sha256(
        {
            "generation": generation,
            "model": model_hash,
            "policy": policy_hash,
            "worker": worker,
            "threads": threads,
            "config": file_sha256(config_path),
            "output": str(output_dir),
        }
    )
    if record_path.is_file():
        existing = _load_canonical_json(record_path, "active worker record")
        if existing.get("supervisor_key") != key:
            raise HostCommandError("active worker record contradicts launch")
        if same_process(existing.get("process_identity", {})):
            return existing["process_identity"]
        expected_identity = existing.get("process_identity", {})
        try:
            current = capture_process_identity(int(expected_identity["pid"]))
        except (HostCommandError, KeyError, TypeError, ValueError):
            current = None
        if current is not None:
            raise HostCommandError("active worker PID was reused")
        retired = (
            Path(state_root)
            / "active-retired"
            / generation
            / (
                f"worker-{worker:03d}-"
                f"{expected_identity.get('start_time_ticks', 'unknown')}.json"
            )
        )
        retired.parent.mkdir(parents=True, exist_ok=True)
        if retired.exists():
            raise HostCommandError("active worker retirement receipt already exists")
        os.rename(record_path, retired)
        _fsync_dir(record_path.parent)
        _fsync_dir(retired.parent)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = Path(state_root) / "logs" / generation / f"active-{worker:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab", buffering=0)
    argv = (
        str(binary),
        "selfplay",
        "-output-dir",
        str(output),
        "-models-dir",
        str(model_root),
        "-config",
        str(config_path),
        "-override-config",
        f"cudaDeviceToUseModel0Thread0={worker},numGameThreads={threads},switchNetsMidGame=false",
    )
    process = subprocess.Popen(
        argv,
        cwd=str(binary.parent),
        stdout=log,
        stderr=log,
        start_new_session=True,
        shell=False,
    )
    log.close()
    try:
        identity = capture_spawned_process(process)
        atomic_write_json(
            record_path,
            {
                "schema_version": 1,
                "contract": WORKER_RECORD_CONTRACT,
                "persistent": True,
                "supervisor_key": key,
                "generation_id": generation,
                "worker_id": worker,
                "model_hash": model_hash,
                "policy_hash": policy_hash,
                "output_dir": str(output),
                "argv": list(argv),
                "process_identity": identity,
            },
        )
    except BaseException:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGTERM)
        raise
    return identity


def active_sync_once(
    *,
    runtime_config: Path,
    state_root: Path,
    katago: Path,
    config: Path,
) -> Mapping[str, Any]:
    from risk_score.promotion_controller import RuntimeConfig
    from risk_score.promotion_state import load_champion

    runtime = RuntimeConfig.load(runtime_config)
    if file_sha256(Path(config)) != runtime.controller.selfplay_config_hash:
        raise HostCommandError("active self-play config hash contradicts runtime")
    champion = load_champion(runtime.champion_path)
    generation = champion.generation_id
    model_hash = champion.champion_hash
    leaf = (
        runtime.accepted_models / "generations" / model_hash / generation
    )
    if not leaf.is_dir() and champion.bootstrap and model_hash == runtime.controller.original_hash:
        leaf = runtime.original_model_path.parent
    _regular_absolute(leaf, "active generation model leaf", directory=True)
    active_root = Path(state_root) / "active"
    for record_path in sorted(active_root.glob("*/*.json")):
        record = _load_canonical_json(record_path, "active worker record")
        if record.get("generation_id") == generation:
            continue
        identity = record.get("process_identity", {})
        if same_process(identity):
            _signal_verified(identity, signal.SIGINT)
            _wait_group_exit(identity, 300.0)
    identities = []
    for worker in range(runtime.controller.worker_count):
        identity = _start_active_worker(
            state_root=state_root,
            katago=katago,
            config=config,
            models_dir=leaf,
            output_dir=runtime.admitted_selfplay
            / "continuous"
            / generation
            / f"worker-{worker:03d}",
            generation=generation,
            model_hash=model_hash,
            policy_hash=runtime.controller.policy_hash,
            worker=worker,
            threads=runtime.controller.worker_threads,
        )
        identities.append(identity)
    return {
        "generation_id": generation,
        "model_hash": model_hash,
        "worker_count": len(identities),
        "process_identities": identities,
    }


def consumers_stop(
    spec_path: Path, *, generation: str, timeout: float
) -> Mapping[str, Any]:
    spec = _load_canonical_json(spec_path, "consumer stop spec")
    if spec.get("contract") != CONSUMER_SPEC_CONTRACT:
        raise HostCommandError("consumer stop spec contract is unsupported")
    roles = ["selfplay", "shuffler", "trainer", "exporter", "evaluator"]
    if not generation or "/" in generation or generation in {".", ".."}:
        raise HostCommandError("rollback generation is unsafe")
    supervisor_root = Path(spec.get("supervisorStateRoot", ""))
    if not supervisor_root.is_absolute():
        raise HostCommandError("consumer stop supervisor state root is invalid")
    pause_id = str(uuid.uuid4())
    atomic_replace_json(
        supervisor_root / "pause.json",
        {
            "schema_version": 1,
            "generation_id": generation,
            "pause_id": pause_id,
            "paused_at_unix": time.time(),
        },
    )
    ack_path = supervisor_root / "pause-ack.json"
    deadline = time.monotonic() + timeout
    while True:
        if ack_path.is_file():
            ack = _load_canonical_json(ack_path, "supervisor pause acknowledgement")
            if ack.get("pause_id") == pause_id:
                break
        if time.monotonic() >= deadline:
            raise HostCommandError("supervisor did not acknowledge rollback pause")
        time.sleep(0.1)
    spec = _load_canonical_json(spec_path, "consumer stop spec")
    if (
        not isinstance(spec.get("updated_at_unix"), (int, float))
        or time.time() - float(spec["updated_at_unix"]) > 30.0
    ):
        raise HostCommandError("consumer identity snapshot is stale")
    identities = spec.get("identities")
    if not isinstance(identities, dict) or set(identities) != set(roles):
        raise HostCommandError("consumer stop spec must bind every role identity")
    patterns = spec.get("rolePatterns")
    run_root = Path(spec.get("runRoot", ""))
    if (
        not isinstance(patterns, dict)
        or set(patterns) != set(roles)
        or not run_root.is_absolute()
    ):
        raise HostCommandError("consumer stop spec must bind run root and role patterns")

    def matching(role: str) -> Dict[int, bytes]:
        role_patterns = patterns[role]
        if not isinstance(role_patterns, list) or not role_patterns:
            raise HostCommandError(f"{role} patterns must be a nonempty array")
        matches = {}
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                command = (proc / "cmdline").read_bytes()
            except OSError:
                continue
            text = command.replace(b"\0", b" ").decode(errors="replace")
            if str(run_root) in text and any(pattern in text for pattern in role_patterns):
                matches[int(proc.name)] = command
        return matches

    for role in roles:
        bound = identities[role]
        if not isinstance(bound, list):
            raise HostCommandError(f"{role} identities must be an array")
        live_matches = matching(role)
        expected_live = {
            int(identity["pid"])
            for identity in bound
            if isinstance(identity, Mapping) and same_process(identity)
        }
        if set(live_matches) != expected_live:
            raise HostCommandError(
                f"{role} live processes do not match persisted identities"
            )
        for expected in bound:
            if not isinstance(expected, Mapping):
                raise HostCommandError(f"{role} identity is malformed")
            try:
                current = capture_process_identity(int(expected["pid"]))
            except HostCommandError:
                continue
            if not all(
                current.get(key) == expected.get(key)
                for key in (
                    "pid",
                    "start_time_ticks",
                    "command_sha256",
                    "process_group_id",
                    "boot_id",
                    "cgroup",
                )
                if key in expected
            ):
                raise HostCommandError(f"{role} PID was reused before rollback")
            _signal_verified(expected, signal.SIGINT)
            _wait_group_exit(expected, timeout)
        if matching(role):
            raise HostCommandError(f"{role} processes remain after rollback stop")
    active_root = Path(spec.get("activeRoot", ""))
    rollback_root = Path(spec.get("rollbackRoot", ""))
    if not active_root.is_absolute() or not rollback_root.is_absolute():
        raise HostCommandError("consumer stop spec active/rollback roots are invalid")
    active_generation = active_root / generation
    if active_generation.is_dir():
        rollback_root.mkdir(parents=True, exist_ok=True)
        destination = rollback_root / f"{generation}-continuous"
        if destination.exists():
            raise HostCommandError("rollback continuous output destination exists")
        os.rename(active_generation, destination)
        _fsync_dir(active_generation.parent)
        _fsync_dir(destination.parent)
    return {"quiescent": True, "quiescent_roles": roles}


def refresh_consumer_identities(
    policy_path: Path, output_path: Path
) -> Mapping[str, Any]:
    policy = _load_canonical_json(policy_path, "consumer stop policy")
    if policy.get("contract") != CONSUMER_SPEC_CONTRACT:
        raise HostCommandError("consumer stop policy contract is unsupported")
    roles = ["selfplay", "shuffler", "trainer", "exporter", "evaluator"]
    patterns = policy.get("rolePatterns")
    run_root = Path(policy.get("runRoot", ""))
    if (
        not isinstance(patterns, dict)
        or set(patterns) != set(roles)
        or not run_root.is_absolute()
    ):
        raise HostCommandError("consumer policy paths/patterns are invalid")
    identities: Dict[str, list[Mapping[str, Any]]] = {}
    for role in roles:
        role_patterns = patterns[role]
        values = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                command = (proc / "cmdline").read_bytes()
            except OSError:
                continue
            text = command.replace(b"\0", b" ").decode(errors="replace")
            if str(run_root) in text and any(
                pattern in text for pattern in role_patterns
            ):
                values.append(capture_process_identity(int(proc.name)))
        identities[role] = sorted(values, key=lambda value: value["pid"])
    value = {
        **policy,
        "identities": identities,
        "policy_sha256": file_sha256(policy_path),
        "updated_at_unix": time.time(),
    }
    atomic_replace_json(output_path, value)
    return value


def _lease_blocks_trainer(runtime: Any) -> bool:
    gpu_config = _load_canonical_json(
        runtime.gpu_lease_config_path, "GPU lease runtime"
    )
    lease_path = Path(gpu_config["paths"]["leaseState"])
    if not lease_path.is_file():
        return False
    lease = _load_canonical_json(lease_path, "GPU lease state")
    return lease.get("phase") != "trainer_running"


def reconcile_trainer(
    *,
    runtime: Any,
    state_root: Path,
    trainer_spec: Path,
    checkpoint: Path,
    consumer_policy: Path,
    consumer_state: Path,
) -> Mapping[str, Any]:
    identities = refresh_consumer_identities(
        consumer_policy, consumer_state
    )["identities"]["trainer"]
    if len(identities) > 1:
        raise HostCommandError("multiple live trainer processes detected")
    identity_path = Path(state_root) / "trainer.json"
    if len(identities) == 1:
        value = {
            "schema_version": 1,
            "role": "trainer",
            "process_identity": identities[0],
            "spec_path": str(Path(trainer_spec).resolve()),
            "checkpoint_path": str(Path(checkpoint).resolve()),
        }
        atomic_replace_json(identity_path, value)
        return {"status": "adopted", "process_identity": identities[0]}
    if _lease_blocks_trainer(runtime):
        return {"status": "lease-handoff"}
    export_root = runtime.promotion_root.parent / "torchmodels_toexport"
    if export_root.is_dir() and any(
        path.is_dir()
        and not path.name.endswith((".tmp", ".partial", ".exported"))
        for path in export_root.iterdir()
    ):
        return {"status": "waiting-for-hardened-export"}
    value = trainer_start(
        spec_path=trainer_spec,
        checkpoint_path=checkpoint,
        identity_output=identity_path,
    )
    return {"status": "started", "process_identity": value["process_identity"]}


def feedback_record(
    *, runtime_config: Path, generation: str, kind: str, evidence_path: Path
) -> Mapping[str, Any]:
    from risk_score.promotion_controller import PromotionController, RuntimeConfig

    evidence = _load_canonical_json(evidence_path, "promotion feedback evidence")
    runtime = RuntimeConfig.load(runtime_config)
    controller = PromotionController(runtime, automatic=True)
    path = controller.record_promotion_feedback(generation, kind, evidence=evidence)
    return {"path": str(path), "sha256": file_sha256(path)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trainer_start = subparsers.add_parser("trainer-launch")
    trainer_start.add_argument("--spec", required=True, type=Path)
    trainer_start.add_argument("--checkpoint", required=True, type=Path)
    trainer_spawn = subparsers.add_parser("trainer-start")
    trainer_spawn.add_argument("--spec", required=True, type=Path)
    trainer_spawn.add_argument("--checkpoint", required=True, type=Path)
    trainer_spawn.add_argument("--identity-output", required=True, type=Path)

    trainer_stop = subparsers.add_parser("trainer-drain")
    trainer_stop.add_argument("--pid", required=True, type=int)
    trainer_stop.add_argument("--process-group", required=True, type=int)
    trainer_stop.add_argument("--checkpoint", required=True, type=Path)
    trainer_stop.add_argument("--timeout", type=float, default=300.0)

    worker = subparsers.add_parser("worker-start")
    worker.add_argument("--state-root", required=True, type=Path)
    worker.add_argument("--katago", required=True, type=Path)
    worker.add_argument("--config", required=True, type=Path)
    worker.add_argument("--models-dir", required=True, type=Path)
    worker.add_argument("--output-dir", required=True, type=Path)
    worker.add_argument("--gpu", required=True, type=int)
    worker.add_argument("--generation", required=True)
    worker.add_argument("--worker", required=True, type=int)
    worker.add_argument("--phase", required=True)
    worker.add_argument("--threads", required=True, type=int)
    worker.add_argument("--model-hash", required=True)
    worker.add_argument("--selfplay-config-hash", required=True)
    worker.add_argument("--policy-hash", required=True)
    worker.add_argument("--ack-inbox", required=True, type=Path)
    worker.add_argument("--max-games", type=int, default=2000)

    worker_run_parser = subparsers.add_parser("worker-run")
    worker_run_parser.add_argument("--intent", required=True, type=Path)
    worker_run_parser.add_argument("--completion", required=True, type=Path)

    worker_monitor_parser = subparsers.add_parser("worker-monitor")
    worker_monitor_parser.add_argument("--record", required=True, type=Path)
    worker_monitor_parser.add_argument("--stable-seconds", type=float, default=2.0)

    watch = subparsers.add_parser("worker-watch-once")
    watch.add_argument("--state-root", required=True, type=Path)
    watch.add_argument("--stable-seconds", type=float, default=2.0)
    watch_loop = subparsers.add_parser("worker-watch")
    watch_loop.add_argument("--state-root", required=True, type=Path)
    watch_loop.add_argument("--stable-seconds", type=float, default=2.0)
    watch_loop.add_argument("--interval", type=float, default=5.0)

    drain = subparsers.add_parser("workers-drain")
    drain.add_argument("--state-root", required=True, type=Path)
    drain.add_argument("--generation", required=True)
    drain.add_argument("--manifest", required=True, type=Path)
    drain.add_argument("--timeout", type=float, default=300.0)

    stop = subparsers.add_parser("consumers-stop")
    stop.add_argument("--spec", required=True, type=Path)
    stop.add_argument("--generation", required=True)
    stop.add_argument("--timeout", type=float, default=300.0)

    feedback = subparsers.add_parser("feedback-record")
    feedback.add_argument("--runtime-config", required=True, type=Path)
    feedback.add_argument("--generation", required=True)
    feedback.add_argument("--kind", required=True)
    feedback.add_argument("--evidence", required=True, type=Path)

    active = subparsers.add_parser("active-sync-once")
    active.add_argument("--runtime-config", required=True, type=Path)
    active.add_argument("--state-root", required=True, type=Path)
    active.add_argument("--katago", required=True, type=Path)
    active.add_argument("--config", required=True, type=Path)
    supervise = subparsers.add_parser("supervise")
    supervise.add_argument("--runtime-config", required=True, type=Path)
    supervise.add_argument("--state-root", required=True, type=Path)
    supervise.add_argument("--katago", required=True, type=Path)
    supervise.add_argument("--config", required=True, type=Path)
    supervise.add_argument("--trainer-spec", required=True, type=Path)
    supervise.add_argument("--trainer-checkpoint", required=True, type=Path)
    supervise.add_argument("--consumer-policy", required=True, type=Path)
    supervise.add_argument("--consumer-state", required=True, type=Path)
    supervise.add_argument("--interval", type=float, default=5.0)
    subparsers.add_parser("evaluator-unsupported")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "trainer-launch":
            return trainer_launch(args.spec, args.checkpoint)
        if args.command == "trainer-start":
            result = trainer_start(
                spec_path=args.spec,
                checkpoint_path=args.checkpoint,
                identity_output=args.identity_output,
            )
        elif args.command == "trainer-drain":
            result = trainer_drain(
                pid=args.pid,
                process_group=args.process_group,
                checkpoint=args.checkpoint,
                timeout=args.timeout,
            )
        elif args.command == "worker-start":
            result = worker_start(
                state_root=args.state_root,
                katago=args.katago,
                config=args.config,
                models_dir=args.models_dir,
                output_dir=args.output_dir,
                gpu=args.gpu,
                generation=args.generation,
                worker=args.worker,
                phase=args.phase,
                threads=args.threads,
                model_hash=args.model_hash,
                selfplay_config_hash=args.selfplay_config_hash,
                policy_hash=args.policy_hash,
                ack_inbox=args.ack_inbox,
                max_games=args.max_games,
            )
        elif args.command == "worker-run":
            return worker_run(args.intent, args.completion)
        elif args.command == "worker-monitor":
            result = worker_monitor(
                args.record, stable_seconds=args.stable_seconds
            )
        elif args.command == "worker-watch-once":
            result = worker_watch_once(
                args.state_root, stable_seconds=args.stable_seconds
            )
        elif args.command == "worker-watch":
            if args.interval <= 0:
                raise HostCommandError("worker watch interval must be positive")
            while True:
                result = worker_watch_once(
                    args.state_root, stable_seconds=args.stable_seconds
                )
                print(canonical_json(result), flush=True)
                time.sleep(args.interval)
        elif args.command == "workers-drain":
            result = workers_drain(
                state_root=args.state_root,
                generation=args.generation,
                manifest_path=args.manifest,
                timeout=args.timeout,
            )
        elif args.command == "consumers-stop":
            result = consumers_stop(
                args.spec, generation=args.generation, timeout=args.timeout
            )
        elif args.command == "feedback-record":
            result = feedback_record(
                runtime_config=args.runtime_config,
                generation=args.generation,
                kind=args.kind,
                evidence_path=args.evidence,
            )
        elif args.command == "active-sync-once":
            result = active_sync_once(
                runtime_config=args.runtime_config,
                state_root=args.state_root,
                katago=args.katago,
                config=args.config,
            )
        elif args.command == "supervise":
            if args.interval <= 0:
                raise HostCommandError("supervisor interval must be positive")
            identity = capture_process_identity(os.getpid())
            while True:
                from risk_score.promotion_controller import RuntimeConfig

                runtime = RuntimeConfig.load(args.runtime_config)
                consumers = refresh_consumer_identities(
                    args.consumer_policy, args.consumer_state
                )
                atomic_replace_json(
                    Path(args.state_root) / "service.json",
                    {
                        "schema_version": 1,
                        "process_identity": identity,
                        "updated_at_unix": time.time(),
                        "runtime_config": str(Path(args.runtime_config).resolve()),
                        "mutation_enabled":
                            runtime.controller.mutation_enabled,
                    },
                )
                pause_path = Path(args.state_root) / "pause.json"
                paused = False
                if pause_path.is_file():
                    pause = _load_canonical_json(pause_path, "supervisor pause")
                    if runtime.champion_path.is_file():
                        from risk_score.promotion_state import load_champion

                        champion = load_champion(runtime.champion_path)
                        if champion.generation_id != pause.get("generation_id"):
                            pause_path.unlink()
                            _fsync_dir(pause_path.parent)
                            (Path(args.state_root) / "pause-ack.json").unlink(
                                missing_ok=True
                            )
                        else:
                            paused = True
                    else:
                        paused = True
                    if paused:
                        atomic_replace_json(
                            Path(args.state_root) / "pause-ack.json",
                            {
                                "schema_version": 1,
                                "generation_id": pause.get("generation_id"),
                                "pause_id": pause.get("pause_id"),
                                "acknowledged_at_unix": time.time(),
                                "service_identity": identity,
                            },
                        )
                if runtime.controller.mutation_enabled and not paused:
                    trainer = reconcile_trainer(
                        runtime=runtime,
                        state_root=args.state_root,
                        trainer_spec=args.trainer_spec,
                        checkpoint=args.trainer_checkpoint,
                        consumer_policy=args.consumer_policy,
                        consumer_state=args.consumer_state,
                    )
                    watched = worker_watch_once(
                        args.state_root, stable_seconds=2.0
                    )
                    if runtime.champion_path.is_file():
                        active = active_sync_once(
                            runtime_config=args.runtime_config,
                            state_root=args.state_root,
                            katago=args.katago,
                            config=args.config,
                        )
                    else:
                        active = {"status": "waiting-for-champion-bootstrap"}
                else:
                    trainer = {
                        "status": "shadow-disabled"
                        if not runtime.controller.mutation_enabled
                        else "rollback-paused"
                    }
                    watched = {"running": [], "completed": []}
                    active = dict(trainer)
                print(
                    canonical_json(
                        {
                            "consumers": consumers["identities"],
                            "trainer": trainer,
                            "worker_watch": watched,
                            "active": active,
                        }
                    ),
                    flush=True,
                )
                time.sleep(args.interval)
        else:
            raise HostCommandError(
                "direct evaluator lease launch is not configured; "
                "use the controller exclusive handoff adapter"
            )
    except (HostCommandError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
