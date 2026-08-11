#!/usr/bin/env python3
"""Fail-closed host process supervision for checkpoint promotion."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
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
TRAINER_COMPLETION_CONTRACT = "risk-score-host-trainer-completion-v1"
TRAINER_OBSERVATION_CONTRACT = "risk-score-host-trainer-observation-v1"
TRAINER_INTERRUPTED_CONTRACT = "risk-score-host-trainer-interrupted-v1"
BOOT_READY_CONTRACT = "risk-score-host-boot-ready-v1"
WORKER_RECORD_CONTRACT = "risk-score-host-worker-record-v1"
ACTIVE_WORKER_RETRY_CONTRACT = "risk-score-host-active-worker-retry-v1"
ACTIVE_WORKER_QUARANTINE_CONTRACT = (
    "risk-score-host-active-worker-quarantine-v1"
)
PLANNED_STOP_CONTRACT = "risk-score-host-planned-stop-v1"
CONSUMER_SPEC_CONTRACT = "risk-score-host-consumer-spec-v1"

TRAINER_SHORT_LIVED_SECONDS = 300.0
TRAINER_RESTART_BACKOFF_INITIAL_SECONDS = 30.0
TRAINER_RESTART_BACKOFF_MAX_SECONDS = 300.0
ACTIVE_WORKER_RETRY_BUDGET = 2
ACTIVE_WORKER_HEALTHY_RESET_SECONDS = 60.0

PROCESS_IDENTITY_FIELDS = (
    "pid",
    "start_time_ticks",
    "command_sha256",
    "process_group_id",
    "boot_id",
    "cgroup",
)


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


def _current_boot_id() -> str:
    try:
        value = (
            Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError as exc:
        raise HostCommandError("cannot read current boot identity") from exc
    if not value:
        raise HostCommandError("current boot identity is empty")
    return value


def publish_boot_ready(
    runtime_config: Path,
    output: Path,
    *,
    boot_id: Optional[str] = None,
    now: Optional[float] = None,
) -> Mapping[str, Any]:
    runtime_path = _regular_absolute(
        Path(runtime_config).resolve(), "promotion runtime config"
    )
    target = Path(output)
    if not target.is_absolute() or target.is_symlink():
        raise HostCommandError("boot-ready output must be absolute and non-symlink")
    current_boot_id = _current_boot_id() if boot_id is None else str(boot_id)
    published_at_unix = time.time() if now is None else float(now)
    if not current_boot_id or not math.isfinite(published_at_unix):
        raise HostCommandError("boot-ready identity or timestamp is invalid")
    value = {
        "schema_version": 1,
        "contract": BOOT_READY_CONTRACT,
        "boot_id": current_boot_id,
        "runtime_config": str(runtime_path),
        "runtime_config_sha256": file_sha256(runtime_path),
        "published_at_unix": published_at_unix,
    }
    atomic_replace_json(target, value)
    return value


def _boot_ready_status(
    marker_path: Path,
    runtime_config: Path,
    *,
    boot_id: str,
) -> Tuple[bool, str]:
    marker = _load_optional_canonical_json(marker_path, "boot-ready marker")
    if marker is None:
        return False, "missing"
    if (
        marker.get("schema_version") != 1
        or marker.get("contract") != BOOT_READY_CONTRACT
    ):
        raise HostCommandError("boot-ready marker contract is unsupported")
    runtime_path = _regular_absolute(
        Path(runtime_config).resolve(), "promotion runtime config"
    )
    if marker.get("boot_id") != boot_id:
        return False, "boot-mismatch"
    if marker.get("runtime_config") != str(runtime_path):
        return False, "runtime-path-mismatch"
    if marker.get("runtime_config_sha256") != file_sha256(runtime_path):
        return False, "runtime-hash-mismatch"
    published = marker.get("published_at_unix")
    if (
        isinstance(published, bool)
        or not isinstance(published, (int, float))
        or not math.isfinite(float(published))
    ):
        raise HostCommandError("boot-ready marker timestamp is malformed")
    return True, "ready"


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


def _load_optional_canonical_json(
    path: Path, role: str
) -> Optional[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return None
    return _load_canonical_json(source, role)


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
        for key in PROCESS_IDENTITY_FIELDS
    )


def _validated_process_identity(
    value: Any, role: str
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HostCommandError(f"{role} process identity is malformed")
    identity = {key: value.get(key) for key in PROCESS_IDENTITY_FIELDS}
    if (
        type(identity["pid"]) is not int
        or identity["pid"] <= 0
        or type(identity["start_time_ticks"]) is not int
        or identity["start_time_ticks"] < 0
        or type(identity["process_group_id"]) is not int
        or identity["process_group_id"] <= 0
        or not isinstance(identity["command_sha256"], str)
        or len(identity["command_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in identity["command_sha256"]
        )
        or not isinstance(identity["boot_id"], str)
        or not isinstance(identity["cgroup"], str)
    ):
        raise HostCommandError(f"{role} process identity is malformed")
    return identity


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


def _trainer_launch_key(
    *,
    launch_id: str,
    spec_path: Path,
    spec_sha256: str,
    checkpoint_path: Path,
    argv_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "contract": TRAINER_COMPLETION_CONTRACT,
            "launch_id": launch_id,
            "spec_path": str(Path(spec_path).resolve()),
            "spec_sha256": spec_sha256,
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "argv_sha256": argv_sha256,
        }
    )


def trainer_launch(
    spec_path: Path,
    checkpoint_path: Path,
    *,
    completion_path: Optional[Path] = None,
    launch_id: Optional[str] = None,
    expected_spec_sha256: Optional[str] = None,
    expected_argv_sha256: Optional[str] = None,
) -> int:
    receipt_fields = (
        completion_path,
        launch_id,
        expected_spec_sha256,
        expected_argv_sha256,
    )
    if any(value is not None for value in receipt_fields) and any(
        value is None for value in receipt_fields
    ):
        raise HostCommandError("trainer completion binding is incomplete")
    if expected_spec_sha256 is not None:
        if file_sha256(spec_path) != expected_spec_sha256:
            raise HostCommandError("trainer launch spec changed before launch")
    argv, cwd, environment, log_path = build_trainer_exec(spec_path, checkpoint_path)
    actual_spec_sha256 = file_sha256(spec_path)
    actual_argv_sha256 = canonical_sha256(list(argv))
    if (
        expected_spec_sha256 is not None
        and (
            actual_spec_sha256 != expected_spec_sha256
            or actual_argv_sha256 != expected_argv_sha256
        )
    ):
        raise HostCommandError("trainer launch binding changed before execution")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at_unix = time.time()
    started_at_monotonic = time.monotonic()
    received_signal: Optional[str] = None
    previous_sigint = signal.getsignal(signal.SIGINT)

    def record_sigint(_signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = "SIGINT"

    signal.signal(signal.SIGINT, record_sigint)
    try:
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
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    completed_at_unix = time.time()
    if completion_path is not None:
        launch_key = _trainer_launch_key(
            launch_id=str(launch_id),
            spec_path=spec_path,
            spec_sha256=str(expected_spec_sha256),
            checkpoint_path=checkpoint_path,
            argv_sha256=str(expected_argv_sha256),
        )
        atomic_write_json(
            completion_path,
            {
                "schema_version": 1,
                "contract": TRAINER_COMPLETION_CONTRACT,
                "launch_id": launch_id,
                "launch_key": launch_key,
                "spec_path": str(Path(spec_path).resolve()),
                "spec_sha256": expected_spec_sha256,
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "argv_sha256": expected_argv_sha256,
                "bucket_limited": True,
                "returncode": completed.returncode,
                "termination_signal": received_signal,
                "started_at_unix": started_at_unix,
                "completed_at_unix": completed_at_unix,
                "runtime_seconds": max(
                    0.0, time.monotonic() - started_at_monotonic
                ),
            },
        )
    return completed.returncode


def trainer_start(
    *, spec_path: Path, checkpoint_path: Path, identity_output: Path
) -> Mapping[str, Any]:
    spec = _regular_absolute(
        Path(spec_path).resolve(), "trainer launch spec"
    )
    checkpoint = _regular_absolute(checkpoint_path, "trainer checkpoint")
    spec_sha256 = file_sha256(spec)
    trainer_argv, _, _, _ = build_trainer_exec(spec, checkpoint)
    if file_sha256(spec) != spec_sha256:
        raise HostCommandError("trainer launch spec changed during validation")
    argv_sha256 = canonical_sha256(list(trainer_argv))
    launch_id = str(uuid.uuid4())
    launch_key = _trainer_launch_key(
        launch_id=launch_id,
        spec_path=spec,
        spec_sha256=spec_sha256,
        checkpoint_path=checkpoint,
        argv_sha256=argv_sha256,
    )
    completion_path = (
        Path(identity_output).parent
        / "trainer-completions"
        / f"{launch_id}.json"
    ).resolve()
    argv = (
        sys.executable,
        "-m",
        "risk_score.promotion_host",
        "trainer-launch",
        "--spec",
        str(spec),
        "--checkpoint",
        str(checkpoint),
        "--completion",
        str(completion_path),
        "--launch-id",
        launch_id,
        "--expected-spec-sha256",
        spec_sha256,
        "--expected-argv-sha256",
        argv_sha256,
    )
    launch_record = {
        "schema_version": 1,
        "role": "trainer",
        "process_identity": None,
        "launch_boot_id": _current_boot_id(),
        "spec_path": str(spec),
        "spec_sha256": spec_sha256,
        "checkpoint_path": str(checkpoint),
        "argv_sha256": argv_sha256,
        "launch_id": launch_id,
        "launch_key": launch_key,
        "completion_path": str(completion_path),
        "wrapper_argv": list(argv),
        "started_at_unix": time.time(),
        "launch_status": "starting",
    }
    atomic_replace_json(identity_output, launch_record)
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
            **launch_record,
            "process_identity": identity,
            "launch_status": "running",
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


def _active_retry_path(state_root: Path, generation: str, worker: int) -> Path:
    return (
        Path(state_root)
        / "active-retries"
        / generation
        / f"worker-{worker:03d}.json"
    )


def _active_quarantine_path(
    state_root: Path, generation: str, worker: int
) -> Path:
    return (
        Path(state_root)
        / "active-quarantine"
        / generation
        / f"worker-{worker:03d}.json"
    )


def _planned_stop_path(state_root: Path) -> Path:
    return Path(state_root) / "planned-stop.json"


def publish_planned_stop(
    state_root: Path,
    service_identity: Mapping[str, Any],
    *,
    now: Optional[float] = None,
) -> Mapping[str, Any]:
    identity = _validated_process_identity(
        service_identity, "planned-stop service"
    )
    published_at_unix = time.time() if now is None else float(now)
    if not math.isfinite(published_at_unix):
        raise HostCommandError("planned-stop timestamp is invalid")
    records: Dict[str, str] = {}
    for record_path in sorted((Path(state_root) / "active").glob("*/*.json")):
        record = _load_canonical_json(record_path, "active worker record")
        records[str(record_path.resolve())] = canonical_sha256(record)
    value = {
        "schema_version": 1,
        "contract": PLANNED_STOP_CONTRACT,
        "boot_id": identity["boot_id"],
        "service_identity": identity,
        "active_record_sha256": records,
        "published_at_unix": published_at_unix,
    }
    atomic_replace_json(_planned_stop_path(state_root), value)
    return value


def _planned_active_transition(
    state_root: Path,
    record_path: Path,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    current_boot_id: str,
) -> Optional[str]:
    if identity.get("boot_id") != current_boot_id:
        return "reboot"
    marker = _load_optional_canonical_json(
        _planned_stop_path(state_root), "planned-stop marker"
    )
    if marker is None or marker.get("boot_id") != current_boot_id:
        return None
    if marker.get("contract") != PLANNED_STOP_CONTRACT:
        raise HostCommandError("planned-stop marker contract is unsupported")
    records = marker.get("active_record_sha256")
    if not isinstance(records, Mapping):
        raise HostCommandError("planned-stop marker records are malformed")
    expected = records.get(str(record_path.resolve()))
    if expected is None:
        return None
    if expected != canonical_sha256(record):
        raise HostCommandError("planned-stop marker contradicts worker record")
    _validated_process_identity(
        marker.get("service_identity"), "planned-stop service"
    )
    return "service-stop"


def _active_command_bytes(argv: Sequence[str]) -> bytes:
    return b"\0".join(part.encode("utf-8") for part in argv) + b"\0"


def _find_active_processes(argv: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    expected = _active_command_bytes(argv)
    matches = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise HostCommandError("cannot reconcile active spawn without procfs")
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        if command == expected:
            matches.append(capture_process_identity(int(proc.name)))
    return tuple(sorted(matches, key=lambda value: value["pid"]))


def _unlink_fsync(path: Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()
        _fsync_dir(target.parent)


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
    retry_budget: int = ACTIVE_WORKER_RETRY_BUDGET,
    current_boot_id: Optional[str] = None,
    now: Optional[float] = None,
    failure_hook: Optional[Any] = None,
) -> Mapping[str, Any]:
    if type(retry_budget) is not int or not 0 <= retry_budget <= 5:
        raise HostCommandError("active worker retry budget must be in range 0..5")
    boot_id = _current_boot_id() if current_boot_id is None else str(current_boot_id)
    observed_at_unix = time.time() if now is None else float(now)
    if not boot_id or not math.isfinite(observed_at_unix):
        raise HostCommandError("active worker boot identity or timestamp is invalid")

    def fail(step: str) -> None:
        if failure_hook is not None:
            failure_hook(step)

    binary = _regular_absolute(katago, "KataGo binary")
    config_path = _regular_absolute(config, "self-play config")
    model_root = _regular_absolute(models_dir, "active model directory", directory=True)
    model = _regular_absolute(model_root / "model.bin.gz", "active model")
    if file_sha256(model) != model_hash:
        raise HostCommandError("active worker model identity is invalid")
    if not (0 <= worker <= 6) or threads != 100:
        raise HostCommandError("active worker topology must be 7x100")
    config_hash = file_sha256(config_path)
    binary_hash = file_sha256(binary)
    output = Path(output_dir)
    record_path = _active_record_path(state_root, generation, worker)
    retry_path = _active_retry_path(state_root, generation, worker)
    quarantine_path = _active_quarantine_path(state_root, generation, worker)
    key = canonical_sha256(
        {
            "generation": generation,
            "model": model_hash,
            "policy": policy_hash,
            "worker": worker,
            "threads": threads,
            "config": config_hash,
            "output": str(output),
        }
    )
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
    quarantine = _load_optional_canonical_json(
        quarantine_path, "active worker quarantine marker"
    )
    if quarantine is not None:
        if (
            quarantine.get("contract") != ACTIVE_WORKER_QUARANTINE_CONTRACT
            or quarantine.get("supervisor_key") != key
            or quarantine.get("generation_id") != generation
            or quarantine.get("worker_id") != worker
        ):
            raise HostCommandError("active worker quarantine marker contradicts launch")
        raise HostCommandError(
            f"active worker {worker} is quarantined after retry exhaustion"
        )
    retry_state = _load_optional_canonical_json(
        retry_path, "active worker retry state"
    )
    if retry_state is not None:
        if (
            retry_state.get("contract") != ACTIVE_WORKER_RETRY_CONTRACT
            or retry_state.get("supervisor_key") != key
            or retry_state.get("generation_id") != generation
            or retry_state.get("worker_id") != worker
            or retry_state.get("status")
            not in {
                "retire-intent",
                "retired",
                "spawning",
                "spawned",
                "running",
                "healthy-reset",
            }
            or type(retry_state.get("retry_count")) is not int
            or retry_state["retry_count"] < 0
        ):
            raise HostCommandError("active worker retry state contradicts launch")

    def validate_record(record: Mapping[str, Any]) -> Tuple[Dict[str, Any], int]:
        if (
            record.get("contract") != WORKER_RECORD_CONTRACT
            or record.get("persistent") is not True
            or record.get("supervisor_key") != key
            or record.get("generation_id") != generation
            or record.get("worker_id") != worker
            or record.get("model_hash") != model_hash
            or record.get("policy_hash") != policy_hash
            or record.get("output_dir") != str(output)
            or record.get("argv") != list(argv)
            or (
                "selfplay_config_hash" in record
                and record.get("selfplay_config_hash") != config_hash
            )
            or (
                "binary_hash" in record
                and record.get("binary_hash") != binary_hash
            )
        ):
            raise HostCommandError("active worker record contradicts launch")
        identity = _validated_process_identity(
            record.get("process_identity"), "active worker"
        )
        attempt = record.get("restart_attempt", 0)
        if type(attempt) is not int or attempt < 0:
            raise HostCommandError("active worker retry counter is malformed")
        return identity, attempt

    def record_value(
        identity: Mapping[str, Any], attempt: int, started_at_unix: float
    ) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "contract": WORKER_RECORD_CONTRACT,
            "persistent": True,
            "supervisor_key": key,
            "generation_id": generation,
            "worker_id": worker,
            "model_hash": model_hash,
            "selfplay_config_hash": config_hash,
            "policy_hash": policy_hash,
            "binary_hash": binary_hash,
            "threads": threads,
            "output_dir": str(output),
            "argv": list(argv),
            "process_identity": dict(identity),
            "restart_attempt": attempt,
            "started_at_unix": started_at_unix,
        }

    def write_retry(status: str, **fields: Any) -> Dict[str, Any]:
        value = {
            "schema_version": 1,
            "contract": ACTIVE_WORKER_RETRY_CONTRACT,
            "status": status,
            "generation_id": generation,
            "worker_id": worker,
            "supervisor_key": key,
            "updated_at_unix": observed_at_unix,
            **fields,
        }
        atomic_replace_json(retry_path, value)
        return value

    def quarantine(
        retry_count: int,
        identity: Mapping[str, Any],
        *,
        reason: str = "retry-budget-exhausted",
    ) -> None:
        atomic_write_json(
            quarantine_path,
            {
                "schema_version": 1,
                "contract": ACTIVE_WORKER_QUARANTINE_CONTRACT,
                "reason": reason,
                "generation_id": generation,
                "worker_id": worker,
                "model_hash": model_hash,
                "selfplay_config_hash": config_hash,
                "policy_hash": policy_hash,
                "supervisor_key": key,
                "retry_count": retry_count,
                "retry_budget": retry_budget,
                "process_identity": dict(identity),
                "process_identity_sha256": canonical_sha256(identity),
                "quarantined_at_unix": observed_at_unix,
            },
        )
        raise HostCommandError(f"active worker {worker} exhausted retry budget")

    existing = (
        _load_canonical_json(record_path, "active worker record")
        if record_path.is_file()
        else None
    )
    expected_identity: Optional[Dict[str, Any]] = None
    restart_attempt = 0
    if existing is not None:
        expected_identity, restart_attempt = validate_record(existing)
        identity_hash = canonical_sha256(expected_identity)
        record_hash = canonical_sha256(existing)
        if retry_state is not None:
            status = retry_state["status"]
            if status == "healthy-reset":
                target_record = retry_state.get("target_record")
                if not isinstance(target_record, Mapping):
                    raise HostCommandError(
                        "active worker healthy reset target is malformed"
                    )
                source_hash = retry_state.get("source_record_sha256")
                target_hash = retry_state.get("target_record_sha256")
                if (
                    target_hash != canonical_sha256(target_record)
                    or record_hash not in {source_hash, target_hash}
                ):
                    raise HostCommandError(
                        "active worker healthy reset contradicts record"
                    )
                if record_hash == source_hash:
                    atomic_replace_json(record_path, target_record)
                    fail("after-healthy-record")
                _unlink_fsync(retry_path)
                fail("after-healthy-reset")
                return _validated_process_identity(
                    target_record.get("process_identity"),
                    "healthy active worker",
                )
            if status == "running":
                if (
                    retry_state["retry_count"] != restart_attempt
                    or retry_state.get("current_process_identity_sha256")
                    != identity_hash
                ):
                    raise HostCommandError(
                        "active worker retry state contradicts record"
                    )
            elif status == "spawned":
                if (
                    retry_state["retry_count"] != restart_attempt
                    or retry_state.get("target_process_identity_sha256")
                    != identity_hash
                ):
                    raise HostCommandError(
                        "active worker retry transition contradicts record"
                    )
                retry_state = write_retry(
                    "running",
                    retry_count=restart_attempt,
                    current_process_identity_sha256=identity_hash,
                )
                fail("after-running-state")
            elif status in {"retired", "spawning"}:
                raise HostCommandError(
                    "active worker record appeared before spawn identity"
                )
            elif (
                status != "retire-intent"
                or retry_state.get("source_record_sha256") != record_hash
            ):
                raise HostCommandError(
                    "active worker retry transition contradicts record"
                )
        elif restart_attempt:
            raise HostCommandError("active worker retry state is missing")

        if same_process(expected_identity):
            started = existing.get("started_at_unix")
            if (
                retry_state is not None
                and retry_state.get("status") == "running"
                and not isinstance(started, bool)
                and isinstance(started, (int, float))
                and observed_at_unix - float(started)
                >= ACTIVE_WORKER_HEALTHY_RESET_SECONDS
            ):
                healthy = {**existing, "restart_attempt": 0}
                write_retry(
                    "healthy-reset",
                    retry_count=restart_attempt,
                    source_record_sha256=record_hash,
                    target_record=healthy,
                    target_record_sha256=canonical_sha256(healthy),
                )
                fail("after-healthy-reset-intent")
                atomic_replace_json(record_path, healthy)
                fail("after-healthy-record")
                _unlink_fsync(retry_path)
                fail("after-healthy-reset")
            return expected_identity

        if expected_identity["boot_id"] == boot_id:
            try:
                current = capture_process_identity(expected_identity["pid"])
            except HostCommandError:
                current = None
            if current is not None:
                raise HostCommandError("active worker PID was reused")
        planned_reason = _planned_active_transition(
            state_root,
            record_path,
            existing,
            expected_identity,
            current_boot_id=boot_id,
        )
        if retry_state is not None and retry_state["status"] == "retire-intent":
            retired = Path(retry_state["retired_record_path"])
            next_attempt = (
                0
                if planned_reason is not None
                else retry_state["retry_count"]
            )
            if next_attempt != retry_state["retry_count"]:
                retry_state = write_retry(
                    "retire-intent",
                    retry_count=next_attempt,
                    source_record_sha256=retry_state[
                        "source_record_sha256"
                    ],
                    source_process_identity=expected_identity,
                    source_process_identity_sha256=identity_hash,
                    retired_record_path=str(retired),
                    planned_reason=planned_reason,
                )
        else:
            next_attempt = 0 if planned_reason is not None else restart_attempt + 1
            if next_attempt > retry_budget:
                quarantine(restart_attempt, expected_identity)
            boot_fragment = hashlib.sha256(
                expected_identity["boot_id"].encode("utf-8")
            ).hexdigest()[:12]
            retired = (
                Path(state_root)
                / "active-retired"
                / generation
                / (
                    f"worker-{worker:03d}-{boot_fragment}-"
                    f"{expected_identity['start_time_ticks']}.json"
                )
            )
            retry_state = write_retry(
                "retire-intent",
                retry_count=next_attempt,
                source_record_sha256=record_hash,
                source_process_identity=expected_identity,
                source_process_identity_sha256=identity_hash,
                retired_record_path=str(retired),
                planned_reason=planned_reason,
            )
            fail("after-retire-intent")
        retired.parent.mkdir(parents=True, exist_ok=True)
        if record_path.exists():
            if retired.exists():
                raise HostCommandError(
                    "active worker retirement receipt already exists"
                )
            os.rename(record_path, retired)
            _fsync_dir(record_path.parent)
            _fsync_dir(retired.parent)
        fail("after-retire")
        retired_record = _load_canonical_json(retired, "retired active worker record")
        if canonical_sha256(retired_record) != retry_state["source_record_sha256"]:
            raise HostCommandError("retired active worker record hash changed")
        retry_state = write_retry(
            "retired",
            retry_count=retry_state["retry_count"],
            source_record_sha256=retry_state["source_record_sha256"],
            source_process_identity=retry_state["source_process_identity"],
            source_process_identity_sha256=retry_state[
                "source_process_identity_sha256"
            ],
            retired_record_path=str(retired),
            planned_reason=retry_state.get("planned_reason"),
            spawn_count=0,
            spawn_replay_count=0,
        )
        fail("after-retired-state")
        existing = None

    if existing is None and retry_state is not None:
        status = retry_state["status"]
        if status in {"running", "healthy-reset"}:
            raise HostCommandError("active worker record is missing with retry state")
        retired = Path(retry_state["retired_record_path"])
        if not retired.is_file():
            if status != "retire-intent":
                raise HostCommandError("retired active worker record is missing")
            raise HostCommandError(
                "active worker retirement intent has no source or receipt"
            )
        retired_record = _load_canonical_json(retired, "retired active worker record")
        if canonical_sha256(retired_record) != retry_state["source_record_sha256"]:
            raise HostCommandError("retired active worker record hash changed")
        source_identity = _validated_process_identity(
            retry_state["source_process_identity"], "retry source worker"
        )
        target_on_current_boot = False
        if retry_state["status"] == "spawned":
            target_identity = _validated_process_identity(
                retry_state.get("target_process_identity"),
                "retry target worker",
            )
            target_on_current_boot = target_identity["boot_id"] == boot_id
        if source_identity["boot_id"] != boot_id and not target_on_current_boot:
            retry_state = write_retry(
                "retired",
                retry_count=0,
                source_record_sha256=retry_state["source_record_sha256"],
                source_process_identity=source_identity,
                source_process_identity_sha256=retry_state[
                    "source_process_identity_sha256"
                ],
                retired_record_path=str(retired),
                planned_reason="reboot",
                spawn_count=0,
                spawn_replay_count=0,
            )
        if status == "retire-intent":
            retry_state = write_retry(
                "retired",
                retry_count=retry_state["retry_count"],
                source_record_sha256=retry_state["source_record_sha256"],
                source_process_identity=retry_state["source_process_identity"],
                source_process_identity_sha256=retry_state[
                    "source_process_identity_sha256"
                ],
                retired_record_path=str(retired),
                planned_reason=retry_state.get("planned_reason"),
                spawn_count=0,
                spawn_replay_count=0,
            )
            fail("after-retired-state")

        while retry_state["status"] in {"retired", "spawning"}:
            matches = list(_find_active_processes(argv))
            if len(matches) > 1:
                raise HostCommandError(
                    "multiple active worker processes match retry transition"
                )
            if matches:
                identity = _validated_process_identity(
                    matches[0], "recovered active worker"
                )
                if identity["boot_id"] != boot_id:
                    raise HostCommandError(
                        "recovered active worker belongs to another boot"
                    )
                if identity["process_group_id"] != identity["pid"]:
                    raise HostCommandError(
                        "recovered active worker is not a session leader"
                    )
                retry_state = write_retry(
                    "spawned",
                    retry_count=retry_state["retry_count"],
                    source_record_sha256=retry_state["source_record_sha256"],
                    source_process_identity=retry_state[
                        "source_process_identity"
                    ],
                    source_process_identity_sha256=retry_state[
                        "source_process_identity_sha256"
                    ],
                    retired_record_path=str(retired),
                    planned_reason=retry_state.get("planned_reason"),
                    spawn_count=retry_state.get("spawn_count", 0),
                    spawn_replay_count=retry_state.get(
                        "spawn_replay_count", 0
                    ),
                    spawn_started_at_unix=retry_state.get(
                        "spawn_started_at_unix", observed_at_unix
                    ),
                    target_process_identity=identity,
                    target_process_identity_sha256=canonical_sha256(identity),
                )
                fail("after-spawned-state")
                break
            if retry_state["status"] == "spawning":
                replay_count = retry_state.get("spawn_replay_count", 0)
                if type(replay_count) is not int or replay_count < 0:
                    raise HostCommandError(
                        "active worker spawn replay counter is malformed"
                    )
                if replay_count < 1:
                    retry_state = write_retry(
                        "retired",
                        retry_count=retry_state["retry_count"],
                        source_record_sha256=retry_state[
                            "source_record_sha256"
                        ],
                        source_process_identity=retry_state[
                            "source_process_identity"
                        ],
                        source_process_identity_sha256=retry_state[
                            "source_process_identity_sha256"
                        ],
                        retired_record_path=str(retired),
                        planned_reason=retry_state.get("planned_reason"),
                        spawn_count=retry_state.get("spawn_count", 0),
                        spawn_replay_count=replay_count + 1,
                    )
                    continue
                next_attempt = retry_state["retry_count"] + 1
                source_identity = _validated_process_identity(
                    retry_state["source_process_identity"], "retry source worker"
                )
                if next_attempt > retry_budget:
                    quarantine(retry_state["retry_count"], source_identity)
                retry_state = write_retry(
                    "retired",
                    retry_count=next_attempt,
                    source_record_sha256=retry_state["source_record_sha256"],
                    source_process_identity=source_identity,
                    source_process_identity_sha256=retry_state[
                        "source_process_identity_sha256"
                    ],
                    retired_record_path=str(retired),
                    planned_reason=None,
                    spawn_count=retry_state.get("spawn_count", 0),
                    spawn_replay_count=0,
                )
                continue
            retry_state = write_retry(
                "spawning",
                retry_count=retry_state["retry_count"],
                source_record_sha256=retry_state["source_record_sha256"],
                source_process_identity=retry_state["source_process_identity"],
                source_process_identity_sha256=retry_state[
                    "source_process_identity_sha256"
                ],
                retired_record_path=str(retired),
                planned_reason=retry_state.get("planned_reason"),
                spawn_count=retry_state.get("spawn_count", 0) + 1,
                spawn_replay_count=retry_state.get("spawn_replay_count", 0),
                spawn_started_at_unix=observed_at_unix,
            )
            fail("after-spawning-state")
            output.mkdir(parents=True, exist_ok=True)
            log_path = (
                Path(state_root)
                / "logs"
                / generation
                / f"active-{worker:03d}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    argv,
                    cwd=str(binary.parent),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    shell=False,
                )
            fail("after-spawn")
            identity = capture_spawned_process(process)
            if identity.get("boot_id") != boot_id:
                raise HostCommandError(
                    "spawned active worker belongs to another boot"
                )
            if identity.get("process_group_id") != identity.get("pid"):
                raise HostCommandError(
                    "spawned active worker is not a session leader"
                )
            retry_state = write_retry(
                "spawned",
                retry_count=retry_state["retry_count"],
                source_record_sha256=retry_state["source_record_sha256"],
                source_process_identity=retry_state["source_process_identity"],
                source_process_identity_sha256=retry_state[
                    "source_process_identity_sha256"
                ],
                retired_record_path=str(retired),
                planned_reason=retry_state.get("planned_reason"),
                spawn_count=retry_state["spawn_count"],
                spawn_replay_count=retry_state.get("spawn_replay_count", 0),
                spawn_started_at_unix=retry_state["spawn_started_at_unix"],
                target_process_identity=identity,
                target_process_identity_sha256=canonical_sha256(identity),
            )
            fail("after-spawned-state")

        if retry_state["status"] == "spawned":
            identity = _validated_process_identity(
                retry_state["target_process_identity"], "retry target worker"
            )
            if identity["boot_id"] != boot_id:
                raise HostCommandError(
                    "retry target worker belongs to another boot"
                )
            if identity["process_group_id"] != identity["pid"]:
                raise HostCommandError(
                    "retry target worker is not a session leader"
                )
            if not same_process(identity):
                next_attempt = retry_state["retry_count"] + 1
                if next_attempt > retry_budget:
                    quarantine(
                        retry_state["retry_count"],
                        _validated_process_identity(
                            retry_state["source_process_identity"],
                            "retry source worker",
                        ),
                    )
                retry_state = write_retry(
                    "retired",
                    retry_count=next_attempt,
                    source_record_sha256=retry_state["source_record_sha256"],
                    source_process_identity=retry_state[
                        "source_process_identity"
                    ],
                    source_process_identity_sha256=retry_state[
                        "source_process_identity_sha256"
                    ],
                    retired_record_path=str(retired),
                    planned_reason=None,
                    spawn_count=retry_state.get("spawn_count", 0),
                    spawn_replay_count=0,
                )
                return _start_active_worker(
                    state_root=state_root,
                    katago=katago,
                    config=config,
                    models_dir=models_dir,
                    output_dir=output_dir,
                    generation=generation,
                    model_hash=model_hash,
                    policy_hash=policy_hash,
                    worker=worker,
                    threads=threads,
                    retry_budget=retry_budget,
                    current_boot_id=boot_id,
                    now=observed_at_unix,
                    failure_hook=failure_hook,
                )
            restarted = record_value(
                identity,
                retry_state["retry_count"],
                float(retry_state["spawn_started_at_unix"]),
            )
            atomic_write_json(record_path, restarted)
            fail("after-record")
            write_retry(
                "running",
                retry_count=retry_state["retry_count"],
                current_process_identity_sha256=canonical_sha256(identity),
            )
            fail("after-running-state")
            return identity

    output.mkdir(parents=True, exist_ok=True)
    log_path = Path(state_root) / "logs" / generation / f"active-{worker:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            argv,
            cwd=str(binary.parent),
            stdout=log,
            stderr=log,
            start_new_session=True,
            shell=False,
        )
    identity = capture_spawned_process(process)
    if identity.get("boot_id") != boot_id:
        raise HostCommandError("spawned active worker belongs to another boot")
    if identity.get("process_group_id") != identity.get("pid"):
        raise HostCommandError("spawned active worker is not a session leader")
    atomic_write_json(
        record_path, record_value(identity, 0, observed_at_unix)
    )
    return identity


def active_sync_once(
    *,
    runtime_config: Path,
    state_root: Path,
    katago: Path,
    config: Path,
    retry_budget: int = ACTIVE_WORKER_RETRY_BUDGET,
    current_boot_id: Optional[str] = None,
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
            retry_budget=retry_budget,
            current_boot_id=current_boot_id,
        )
        identities.append(identity)
    _unlink_fsync(_planned_stop_path(state_root))
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


def _gpu_lease_snapshot(
    runtime: Any,
) -> Tuple[Path, Optional[Mapping[str, Any]]]:
    gpu_config = _load_canonical_json(
        runtime.gpu_lease_config_path, "GPU lease runtime"
    )
    lease_path = Path(gpu_config["paths"]["leaseState"])
    if not lease_path.is_file():
        return lease_path, None
    return lease_path, _load_canonical_json(lease_path, "GPU lease state")


def _gpu_identity_to_host(value: Any, role: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HostCommandError(f"{role} GPU lease identity is malformed")
    return _validated_process_identity(
        {
            "pid": value.get("pid"),
            "start_time_ticks": value.get("startTimeTicks"),
            "process_group_id": value.get("processGroupId"),
            "boot_id": value.get("bootId"),
            "command_sha256": value.get("commandSha256"),
            "cgroup": value.get("cgroup"),
        },
        role,
    )


def _lease_blocks_trainer(runtime: Any) -> bool:
    _, lease = _gpu_lease_snapshot(runtime)
    return lease is not None and (
        lease.get("phase") != "trainer_running"
        or lease.get("safetyHalt") is not False
        or lease.get("evaluators", []) != []
    )


def _gpu_trainer_adoption(
    runtime: Any, live_identity: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    lease_path, lease = _gpu_lease_snapshot(runtime)
    if (
        lease is None
        or lease.get("phase") != "trainer_running"
        or lease.get("safetyHalt") is not False
        or lease.get("evaluators", []) != []
    ):
        return None
    for field in ("restoredTrainer", "trainer"):
        raw_identity = lease.get(field)
        if raw_identity is None:
            continue
        identity = _gpu_identity_to_host(raw_identity, field)
        exact = all(
            identity.get(key) == live_identity.get(key)
            for key in PROCESS_IDENTITY_FIELDS
        )
        group_bound = (
            same_process(identity)
            and identity["process_group_id"] == live_identity.get("process_group_id")
            and identity["boot_id"] == live_identity.get("boot_id")
            and identity["cgroup"] == live_identity.get("cgroup")
        )
        if exact or group_bound:
            return {
                "source": f"gpu-lease-{field}",
                "lease_state_path": str(lease_path),
                "lease_state_sha256": file_sha256(lease_path),
                "process_identity": identity,
                "observed_trainer_identity": dict(live_identity),
            }
    return None


def _trainer_restart_gpu_safe(runtime: Any) -> bool:
    _, lease = _gpu_lease_snapshot(runtime)
    if lease is None:
        return True
    if (
        lease.get("phase") != "trainer_running"
        or lease.get("safetyHalt") is not False
        or lease.get("evaluators", []) != []
    ):
        return False
    for field in ("restoredTrainer", "trainer"):
        raw_identity = lease.get(field)
        if raw_identity is None:
            continue
        identity = _gpu_identity_to_host(raw_identity, field)
        if same_process(identity):
            return False
    return True


def _trainer_observation_path(state_root: Path) -> Path:
    return Path(state_root) / "trainer-observation.json"


def _archive_interrupted_trainer(
    state_root: Path,
    record: Mapping[str, Any],
    *,
    reason: str,
    boot_id: str,
    now: float,
) -> Mapping[str, Any]:
    launch_id = record.get("launch_id")
    if (
        not isinstance(launch_id, str)
        or not launch_id
        or "/" in launch_id
        or "\\" in launch_id
    ):
        raise HostCommandError("interrupted trainer launch ID is malformed")
    record_sha256 = canonical_sha256(record)
    path = (
        Path(state_root)
        / "trainer-interrupted"
        / f"{launch_id}-{record_sha256}.json"
    )
    existing = _load_optional_canonical_json(
        path, "interrupted trainer archive"
    )
    if existing is not None:
        if (
            existing.get("contract") != TRAINER_INTERRUPTED_CONTRACT
            or existing.get("launch_id") != launch_id
            or existing.get("record_sha256") != record_sha256
            or existing.get("reason") != reason
            or existing.get("record") != record
        ):
            raise HostCommandError("interrupted trainer archive contradicts record")
        return existing
    value = {
        "schema_version": 1,
        "contract": TRAINER_INTERRUPTED_CONTRACT,
        "launch_id": launch_id,
        "record_sha256": record_sha256,
        "reason": reason,
        "observed_boot_id": boot_id,
        "archived_at_unix": now,
        "record": dict(record),
    }
    atomic_write_json(path, value)
    return value


def _persist_trainer_observation(
    state_root: Path,
    *,
    observation: str,
    decision: str,
    updated_at_unix: float,
    details: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    existing = _load_optional_canonical_json(
        _trainer_observation_path(state_root), "trainer observation"
    )
    decision_since_unix = updated_at_unix
    if (
        existing is not None
        and existing.get("contract") == TRAINER_OBSERVATION_CONTRACT
        and existing.get("observation") == observation
        and existing.get("decision") == decision
    ):
        previous_since = existing.get("decision_since_unix")
        if (
            not isinstance(previous_since, bool)
            and isinstance(previous_since, (int, float))
            and math.isfinite(float(previous_since))
            and float(previous_since) <= updated_at_unix
        ):
            decision_since_unix = float(previous_since)
    value: Dict[str, Any] = {
        "schema_version": 1,
        "contract": TRAINER_OBSERVATION_CONTRACT,
        "role": "trainer",
        "observation": observation,
        "decision": decision,
        "decision_since_unix": decision_since_unix,
        "updated_at_unix": updated_at_unix,
    }
    if details:
        value.update(details)
    atomic_replace_json(_trainer_observation_path(state_root), value)
    return value


def _trainer_backoff_state(
    value: Optional[Mapping[str, Any]],
) -> Tuple[int, Optional[str], Optional[float]]:
    if value is None:
        return 0, None, None
    if value.get("contract") != TRAINER_OBSERVATION_CONTRACT:
        raise HostCommandError("trainer observation contract is unsupported")
    count = value.get("consecutive_short_clean_exits", 0)
    launch_key = value.get("last_exit_launch_key")
    not_before = value.get("restart_not_before_unix")
    if type(count) is not int or count < 0:
        raise HostCommandError("trainer restart counter is malformed")
    if launch_key is not None and not isinstance(launch_key, str):
        raise HostCommandError("trainer exit launch key is malformed")
    if not_before is not None and (
        isinstance(not_before, bool)
        or not isinstance(not_before, (int, float))
        or not math.isfinite(float(not_before))
    ):
        raise HostCommandError("trainer restart deadline is malformed")
    return count, launch_key, (
        float(not_before) if not_before is not None else None
    )


def _trainer_completion_for_record(
    *,
    record: Mapping[str, Any],
    state_root: Path,
    trainer_spec: Path,
    checkpoint: Path,
    allow_missing: bool = False,
) -> Optional[Mapping[str, Any]]:
    launch_fields = (
        "launch_id",
        "launch_key",
        "spec_sha256",
        "argv_sha256",
        "completion_path",
    )
    present = [field in record for field in launch_fields]
    if not any(present):
        return None
    if not all(present):
        raise HostCommandError("supervised trainer launch record is incomplete")
    launch_id = record["launch_id"]
    if (
        not isinstance(launch_id, str)
        or not launch_id
        or "/" in launch_id
        or "\\" in launch_id
    ):
        raise HostCommandError("supervised trainer launch ID is malformed")
    spec_path = Path(trainer_spec).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    if (
        record.get("spec_path") != str(spec_path)
        or record.get("checkpoint_path") != str(checkpoint_path)
    ):
        raise HostCommandError("supervised trainer paths contradict reconciliation")
    current_spec_sha256 = file_sha256(spec_path)
    trainer_argv, _, _, _ = build_trainer_exec(spec_path, checkpoint_path)
    current_argv_sha256 = canonical_sha256(list(trainer_argv))
    if (
        record.get("spec_sha256") != current_spec_sha256
        or record.get("argv_sha256") != current_argv_sha256
    ):
        raise HostCommandError("supervised trainer launch hashes changed")
    expected_launch_key = _trainer_launch_key(
        launch_id=launch_id,
        spec_path=spec_path,
        spec_sha256=current_spec_sha256,
        checkpoint_path=checkpoint_path,
        argv_sha256=current_argv_sha256,
    )
    if record.get("launch_key") != expected_launch_key:
        raise HostCommandError("supervised trainer launch key is invalid")
    expected_completion_path = (
        Path(state_root) / "trainer-completions" / f"{launch_id}.json"
    ).resolve()
    if Path(record["completion_path"]).resolve() != expected_completion_path:
        raise HostCommandError("supervised trainer completion path is invalid")
    if not expected_completion_path.is_file():
        if allow_missing:
            return None
        raise HostCommandError(
            "supervised trainer disappeared without a completion receipt"
        )
    receipt = _load_canonical_json(
        expected_completion_path, "trainer completion receipt"
    )
    expected = {
        "contract": TRAINER_COMPLETION_CONTRACT,
        "launch_id": launch_id,
        "launch_key": expected_launch_key,
        "spec_path": str(spec_path),
        "spec_sha256": current_spec_sha256,
        "checkpoint_path": str(checkpoint_path),
        "argv_sha256": current_argv_sha256,
        "bucket_limited": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise HostCommandError("trainer completion receipt contradicts launch")
    returncode = receipt.get("returncode")
    runtime_seconds = receipt.get("runtime_seconds")
    started_at_unix = receipt.get("started_at_unix")
    completed_at_unix = receipt.get("completed_at_unix")
    termination_signal = receipt.get("termination_signal")
    if type(returncode) is not int:
        raise HostCommandError("trainer completion return code is malformed")
    for field, value in (
        ("runtime", runtime_seconds),
        ("start timestamp", started_at_unix),
        ("completion timestamp", completed_at_unix),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise HostCommandError(f"trainer completion {field} is malformed")
    if float(runtime_seconds) < 0:
        raise HostCommandError("trainer completion runtime is negative")
    if termination_signal not in {None, "SIGINT"}:
        raise HostCommandError("trainer completion signal is malformed")
    return receipt


def reconcile_trainer(
    *,
    runtime: Any,
    state_root: Path,
    trainer_spec: Path,
    checkpoint: Path,
    consumer_policy: Path,
    consumer_state: Path,
    trainer_identities: Optional[Sequence[Mapping[str, Any]]] = None,
    now: Optional[float] = None,
    boot_id: Optional[str] = None,
    short_lived_seconds: float = TRAINER_SHORT_LIVED_SECONDS,
    backoff_initial_seconds: float = TRAINER_RESTART_BACKOFF_INITIAL_SECONDS,
    backoff_max_seconds: float = TRAINER_RESTART_BACKOFF_MAX_SECONDS,
) -> Mapping[str, Any]:
    observed_at_unix = time.time() if now is None else float(now)
    current_boot_id = _current_boot_id() if boot_id is None else str(boot_id)
    if (
        not math.isfinite(observed_at_unix)
        or not current_boot_id
        or not math.isfinite(float(short_lived_seconds))
        or not math.isfinite(float(backoff_initial_seconds))
        or not math.isfinite(float(backoff_max_seconds))
        or short_lived_seconds <= 0
        or backoff_initial_seconds <= 0
        or backoff_max_seconds < backoff_initial_seconds
    ):
        raise HostCommandError("trainer reconciliation timing is invalid")
    previous_observation = _load_optional_canonical_json(
        _trainer_observation_path(state_root), "trainer observation"
    )
    previous_count, previous_launch_key, previous_not_before = (
        _trainer_backoff_state(previous_observation)
    )
    if trainer_identities is None:
        identities = refresh_consumer_identities(
            consumer_policy, consumer_state
        )["identities"]["trainer"]
    else:
        identities = list(trainer_identities)
    if len(identities) > 1:
        raise HostCommandError("multiple live trainer processes detected")
    identity_path = Path(state_root) / "trainer.json"
    if len(identities) == 1:
        live_identity = _validated_process_identity(
            identities[0], "observed trainer"
        )
        existing = _load_optional_canonical_json(
            identity_path, "trainer identity record"
        )
        launch_fields = (
            "launch_id",
            "launch_key",
            "spec_sha256",
            "argv_sha256",
            "completion_path",
        )
        launched = (
            existing is not None
            and any(field in existing for field in launch_fields)
        )
        adoption: Optional[Mapping[str, Any]] = None
        if launched:
            if not all(field in existing for field in launch_fields):
                raise HostCommandError(
                    "supervised trainer launch record is incomplete"
                )
            launcher_identity = _validated_process_identity(
                existing.get("process_identity"), "trainer launcher"
            )
            if not same_process(launcher_identity):
                adoption = _gpu_trainer_adoption(runtime, live_identity)
                if adoption is None:
                    if _lease_blocks_trainer(runtime):
                        _persist_trainer_observation(
                            state_root,
                            observation="live-trainer-during-gpu-handoff",
                            decision="lease-handoff",
                            updated_at_unix=observed_at_unix,
                            details={"process_identity": live_identity},
                        )
                        return {
                            "status": "lease-handoff",
                            "process_identity": live_identity,
                        }
                    raise HostCommandError(
                        "trainer launcher disappeared while trainer remains live"
                    )
                _archive_interrupted_trainer(
                    state_root,
                    existing,
                    reason="gpu-restored-adoption",
                    boot_id=current_boot_id,
                    now=observed_at_unix,
                )
                value = {
                    "schema_version": 1,
                    "role": "trainer",
                    "process_identity": adoption["process_identity"],
                    "observed_trainer_identity": live_identity,
                    "spec_path": str(Path(trainer_spec).resolve()),
                    "checkpoint_path": str(Path(checkpoint).resolve()),
                    "adoption_source": adoption["source"],
                    "lease_state_path": adoption["lease_state_path"],
                    "lease_state_sha256": adoption["lease_state_sha256"],
                    "adopted_at_unix": observed_at_unix,
                }
                atomic_replace_json(identity_path, value)
                launched = False
        else:
            preserve_gpu_adoption = False
            if existing is not None and isinstance(
                existing.get("adoption_source"), str
            ):
                adopted_identity = _validated_process_identity(
                    existing.get("process_identity"), "adopted trainer launcher"
                )
                if not (
                    same_process(adopted_identity)
                    and adopted_identity["process_group_id"]
                    == live_identity["process_group_id"]
                    and adopted_identity["boot_id"] == live_identity["boot_id"]
                    and adopted_identity["cgroup"] == live_identity["cgroup"]
                ):
                    raise HostCommandError(
                        "adopted GPU trainer launcher identity changed"
                    )
                preserve_gpu_adoption = True
            value = {
                "schema_version": 1,
                "role": "trainer",
                "process_identity": live_identity,
                "spec_path": str(Path(trainer_spec).resolve()),
                "checkpoint_path": str(Path(checkpoint).resolve()),
            }
            if (
                not preserve_gpu_adoption
                and (
                existing is None
                or existing.get("process_identity") != live_identity
                or existing.get("spec_path") != value["spec_path"]
                or existing.get("checkpoint_path") != value["checkpoint_path"]
                )
            ):
                atomic_replace_json(identity_path, value)
        details: Dict[str, Any] = {"process_identity": live_identity}
        if adoption is not None:
            details["adoption_source"] = adoption["source"]
        if launched:
            details["launcher_process_identity"] = existing["process_identity"]
            details["launch_key"] = existing["launch_key"]
        if previous_count:
            details["consecutive_short_clean_exits"] = previous_count
        if previous_launch_key is not None:
            details["last_exit_launch_key"] = previous_launch_key
        if previous_not_before is not None:
            details["restart_not_before_unix"] = previous_not_before
        _persist_trainer_observation(
            state_root,
            observation="live-trainer",
            decision="adopted",
            updated_at_unix=observed_at_unix,
            details=details,
        )
        return {"status": "adopted", "process_identity": live_identity}
    if _lease_blocks_trainer(runtime):
        details = {}
        if previous_count:
            details["consecutive_short_clean_exits"] = previous_count
        if previous_launch_key is not None:
            details["last_exit_launch_key"] = previous_launch_key
        if previous_not_before is not None:
            details["restart_not_before_unix"] = previous_not_before
        _persist_trainer_observation(
            state_root,
            observation="no-live-trainer",
            decision="lease-handoff",
            updated_at_unix=observed_at_unix,
            details=details,
        )
        return {"status": "lease-handoff"}
    export_root = runtime.promotion_root.parent / "torchmodels_toexport"
    if export_root.is_dir() and any(
        path.is_dir()
        and not path.name.endswith((".tmp", ".partial", ".exported"))
        for path in export_root.iterdir()
    ):
        details = {}
        if previous_count:
            details["consecutive_short_clean_exits"] = previous_count
        if previous_launch_key is not None:
            details["last_exit_launch_key"] = previous_launch_key
        if previous_not_before is not None:
            details["restart_not_before_unix"] = previous_not_before
        _persist_trainer_observation(
            state_root,
            observation="pending-export",
            decision="waiting-for-export",
            updated_at_unix=observed_at_unix,
            details=details,
        )
        return {"status": "waiting-for-export"}
    restart_count = previous_count
    last_exit_launch_key = previous_launch_key
    identity_record = _load_optional_canonical_json(
        identity_path, "trainer identity record"
    )
    if identity_record is not None:
        bound_launch_fields = (
            "launch_id",
            "launch_key",
            "spec_sha256",
            "argv_sha256",
            "completion_path",
        )
        launched_record = any(
            field in identity_record for field in bound_launch_fields
        )
        raw_recorded_identity = identity_record.get("process_identity")
        if raw_recorded_identity is None:
            if not launched_record or not all(
                field in identity_record for field in bound_launch_fields
            ):
                raise HostCommandError("recorded trainer identity is missing")
        else:
            recorded_identity = _validated_process_identity(
                raw_recorded_identity, "recorded trainer"
            )
            if same_process(recorded_identity):
                if not all(
                    field in identity_record for field in bound_launch_fields
                ):
                    raise HostCommandError(
                        "supervised trainer is live but missing from consumer observation"
                    )
                details = {
                    "process_identity": recorded_identity,
                    "launch_key": identity_record["launch_key"],
                }
                if previous_count:
                    details["consecutive_short_clean_exits"] = previous_count
                if previous_launch_key is not None:
                    details["last_exit_launch_key"] = previous_launch_key
                if previous_not_before is not None:
                    details["restart_not_before_unix"] = previous_not_before
                _persist_trainer_observation(
                    state_root,
                    observation="trainer-launcher-live",
                    decision="waiting-for-completion",
                    updated_at_unix=observed_at_unix,
                    details=details,
                )
                return {"status": "waiting-for-completion"}
        receipt = _trainer_completion_for_record(
            record=identity_record,
            state_root=state_root,
            trainer_spec=trainer_spec,
            checkpoint=checkpoint,
            allow_missing=True,
        )
        if receipt is None and launched_record:
            if raw_recorded_identity is None:
                interrupted_boot_id = identity_record.get("launch_boot_id")
                if not isinstance(interrupted_boot_id, str):
                    raise HostCommandError(
                        "starting trainer record has no boot identity"
                    )
                if interrupted_boot_id == current_boot_id:
                    raise HostCommandError(
                        "same-boot starting trainer has no verifiable identity"
                    )
            else:
                interrupted_boot_id = recorded_identity["boot_id"]
                if interrupted_boot_id == current_boot_id:
                    try:
                        reused = capture_process_identity(recorded_identity["pid"])
                    except HostCommandError:
                        reused = None
                    if reused is not None:
                        raise HostCommandError(
                            "same-boot interrupted trainer PID was reused"
                        )
            if not _trainer_restart_gpu_safe(runtime):
                _persist_trainer_observation(
                    state_root,
                    observation="previous-boot-interrupted-trainer",
                    decision="lease-handoff",
                    updated_at_unix=observed_at_unix,
                    details={"launch_key": identity_record["launch_key"]},
                )
                return {"status": "lease-handoff"}
            archive = _archive_interrupted_trainer(
                state_root,
                identity_record,
                reason=(
                    "previous-boot-interrupted"
                    if interrupted_boot_id != current_boot_id
                    else "same-boot-dead-interrupted"
                ),
                boot_id=current_boot_id,
                now=observed_at_unix,
            )
            last_exit_launch_key = str(identity_record["launch_key"])
            if interrupted_boot_id != current_boot_id:
                restart_count = 0
            else:
                if previous_launch_key == last_exit_launch_key:
                    restart_count = previous_count
                    restart_not_before = previous_not_before
                    if restart_count <= 0 or restart_not_before is None:
                        raise HostCommandError(
                            "interrupted trainer backoff state is malformed"
                        )
                    expected_not_before = float(
                        archive["archived_at_unix"]
                    ) + min(
                        backoff_initial_seconds
                        * (2 ** min(restart_count - 1, 30)),
                        backoff_max_seconds,
                    )
                    if restart_not_before != expected_not_before:
                        raise HostCommandError(
                            "interrupted trainer backoff deadline changed"
                        )
                else:
                    restart_count = previous_count + 1
                    restart_not_before = float(archive["archived_at_unix"]) + min(
                        backoff_initial_seconds
                        * (2 ** min(restart_count - 1, 30)),
                        backoff_max_seconds,
                    )
                if observed_at_unix < restart_not_before:
                    _persist_trainer_observation(
                        state_root,
                        observation="same-boot-dead-interrupted-trainer",
                        decision="restart-backoff",
                        updated_at_unix=observed_at_unix,
                        details={
                            "consecutive_short_clean_exits": restart_count,
                            "last_exit_launch_key": last_exit_launch_key,
                            "restart_not_before_unix": restart_not_before,
                            "interrupted_record_sha256": archive[
                                "record_sha256"
                            ],
                        },
                    )
                    return {
                        "status": "restart-backoff",
                        "restart_not_before_unix": restart_not_before,
                    }
        if receipt is not None:
            graceful_sigint = (
                receipt.get("termination_signal") == "SIGINT"
                and receipt["returncode"] in {
                    0,
                    -signal.SIGINT,
                    128 + signal.SIGINT,
                }
            )
            if receipt["returncode"] != 0 and not graceful_sigint:
                _persist_trainer_observation(
                    state_root,
                    observation="completed-trainer",
                    decision="abnormal-exit",
                    updated_at_unix=observed_at_unix,
                    details={
                        "last_exit_launch_key": receipt["launch_key"],
                        "returncode": receipt["returncode"],
                    },
                )
                raise HostCommandError("supervised trainer exited abnormally")
            if (
                not graceful_sigint
                and float(receipt["runtime_seconds"]) < short_lived_seconds
            ):
                if previous_launch_key == receipt["launch_key"]:
                    restart_count = previous_count
                    restart_not_before = previous_not_before
                    if restart_count <= 0 or restart_not_before is None:
                        raise HostCommandError(
                            "trainer backoff state contradicts completion"
                        )
                    expected_delay = min(
                        backoff_initial_seconds
                        * (2 ** min(restart_count - 1, 30)),
                        backoff_max_seconds,
                    )
                    expected_not_before = (
                        float(receipt["completed_at_unix"]) + expected_delay
                    )
                    if restart_not_before != expected_not_before:
                        raise HostCommandError(
                            "trainer backoff deadline contradicts completion"
                        )
                else:
                    restart_count = previous_count + 1
                    exponent = min(restart_count - 1, 30)
                    delay = min(
                        backoff_initial_seconds * (2**exponent),
                        backoff_max_seconds,
                    )
                    restart_not_before = (
                        float(receipt["completed_at_unix"]) + delay
                    )
                last_exit_launch_key = str(receipt["launch_key"])
                if observed_at_unix < restart_not_before:
                    _persist_trainer_observation(
                        state_root,
                        observation="short-clean-bucket-limited-exit",
                        decision="restart-backoff",
                        updated_at_unix=observed_at_unix,
                        details={
                            "consecutive_short_clean_exits": restart_count,
                            "last_exit_launch_key": last_exit_launch_key,
                            "restart_not_before_unix": restart_not_before,
                            "runtime_seconds": receipt["runtime_seconds"],
                            "returncode": 0,
                        },
                    )
                    return {
                        "status": "restart-backoff",
                        "restart_not_before_unix": restart_not_before,
                    }
            else:
                restart_count = 0
                last_exit_launch_key = str(receipt["launch_key"])
    value = trainer_start(
        spec_path=trainer_spec,
        checkpoint_path=checkpoint,
        identity_output=identity_path,
    )
    details = {"process_identity": value["process_identity"]}
    if restart_count:
        details["consecutive_short_clean_exits"] = restart_count
    if last_exit_launch_key is not None:
        details["last_exit_launch_key"] = last_exit_launch_key
    if "launch_key" in value:
        details["launch_key"] = value["launch_key"]
    _persist_trainer_observation(
        state_root,
        observation="no-live-trainer",
        decision="started",
        updated_at_unix=observed_at_unix,
        details=details,
    )
    return {"status": "started", "process_identity": value["process_identity"]}


def supervisor_reconcile_once(
    *,
    runtime: Any,
    runtime_config: Path,
    boot_ready: Path,
    state_root: Path,
    katago: Path,
    config: Path,
    trainer_spec: Path,
    trainer_checkpoint: Path,
    consumer_policy: Path,
    consumer_state: Path,
    service_identity: Mapping[str, Any],
    now: Optional[float] = None,
) -> Mapping[str, Any]:
    observed_at_unix = time.time() if now is None else float(now)
    if not math.isfinite(observed_at_unix):
        raise HostCommandError("supervisor timestamp is invalid")
    identity = _validated_process_identity(
        service_identity, "supervisor service"
    )
    consumers = refresh_consumer_identities(consumer_policy, consumer_state)
    ready, ready_reason = _boot_ready_status(
        boot_ready,
        runtime_config,
        boot_id=identity["boot_id"],
    )
    atomic_replace_json(
        Path(state_root) / "service.json",
        {
            "schema_version": 1,
            "process_identity": identity,
            "updated_at_unix": observed_at_unix,
            "runtime_config": str(Path(runtime_config).resolve()),
            "mutation_enabled": runtime.controller.mutation_enabled,
            "boot_ready": ready,
            "boot_ready_reason": ready_reason,
        },
    )
    pause_path = Path(state_root) / "pause.json"
    paused = False
    if pause_path.is_file():
        pause = _load_canonical_json(pause_path, "supervisor pause")
        if runtime.champion_path.is_file():
            from risk_score.promotion_state import load_champion

            champion = load_champion(runtime.champion_path)
            if champion.generation_id != pause.get("generation_id"):
                pause_path.unlink()
                _fsync_dir(pause_path.parent)
                (Path(state_root) / "pause-ack.json").unlink(missing_ok=True)
            else:
                paused = True
        else:
            paused = True
        if paused:
            atomic_replace_json(
                Path(state_root) / "pause-ack.json",
                {
                    "schema_version": 1,
                    "generation_id": pause.get("generation_id"),
                    "pause_id": pause.get("pause_id"),
                    "acknowledged_at_unix": observed_at_unix,
                    "service_identity": identity,
                },
            )
    if runtime.controller.mutation_enabled and not paused and ready:
        trainer = reconcile_trainer(
            runtime=runtime,
            state_root=state_root,
            trainer_spec=trainer_spec,
            checkpoint=trainer_checkpoint,
            consumer_policy=consumer_policy,
            consumer_state=consumer_state,
            trainer_identities=consumers["identities"]["trainer"],
            now=observed_at_unix,
            boot_id=identity["boot_id"],
        )
        watched = worker_watch_once(state_root, stable_seconds=2.0)
        if runtime.champion_path.is_file():
            active = active_sync_once(
                runtime_config=runtime_config,
                state_root=state_root,
                katago=katago,
                config=config,
                current_boot_id=identity["boot_id"],
            )
        else:
            active = {"status": "waiting-for-champion-bootstrap"}
    else:
        decision = (
            "shadow-disabled"
            if not runtime.controller.mutation_enabled
            else "rollback-paused"
            if paused
            else "boot-not-ready"
        )
        _persist_trainer_observation(
            state_root,
            observation="supervision-gated",
            decision=decision,
            updated_at_unix=observed_at_unix,
        )
        trainer = {"status": decision}
        watched = {"running": [], "completed": []}
        active = dict(trainer)
    return {
        "consumers": consumers["identities"],
        "trainer": trainer,
        "worker_watch": watched,
        "active": active,
        "boot_ready": {"ready": ready, "reason": ready_reason},
    }


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

    boot_ready = subparsers.add_parser("boot-ready")
    boot_ready.add_argument("--runtime-config", required=True, type=Path)
    boot_ready.add_argument("--output", required=True, type=Path)

    trainer_start = subparsers.add_parser("trainer-launch")
    trainer_start.add_argument("--spec", required=True, type=Path)
    trainer_start.add_argument("--checkpoint", required=True, type=Path)
    trainer_start.add_argument("--completion", type=Path)
    trainer_start.add_argument("--launch-id")
    trainer_start.add_argument("--expected-spec-sha256")
    trainer_start.add_argument("--expected-argv-sha256")
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
    supervise.add_argument("--boot-ready", required=True, type=Path)
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
        if args.command == "boot-ready":
            result = publish_boot_ready(args.runtime_config, args.output)
        elif args.command == "trainer-launch":
            return trainer_launch(
                args.spec,
                args.checkpoint,
                completion_path=args.completion,
                launch_id=args.launch_id,
                expected_spec_sha256=args.expected_spec_sha256,
                expected_argv_sha256=args.expected_argv_sha256,
            )
        elif args.command == "trainer-start":
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
            previous_sigint = signal.getsignal(signal.SIGINT)
            previous_sigterm = signal.getsignal(signal.SIGTERM)

            def planned_stop_signal(_signum: int, _frame: Any) -> None:
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, planned_stop_signal)
            signal.signal(signal.SIGTERM, planned_stop_signal)
            try:
                while True:
                    from risk_score.promotion_controller import RuntimeConfig

                    runtime = RuntimeConfig.load(args.runtime_config)
                    result = supervisor_reconcile_once(
                        runtime=runtime,
                        runtime_config=args.runtime_config,
                        boot_ready=args.boot_ready,
                        state_root=args.state_root,
                        katago=args.katago,
                        config=args.config,
                        trainer_spec=args.trainer_spec,
                        trainer_checkpoint=args.trainer_checkpoint,
                        consumer_policy=args.consumer_policy,
                        consumer_state=args.consumer_state,
                        service_identity=identity,
                    )
                    print(canonical_json(result), flush=True)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                marker = publish_planned_stop(args.state_root, identity)
                result = {
                    "status": "planned-stop",
                    "planned_stop": marker,
                }
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
                signal.signal(signal.SIGTERM, previous_sigterm)
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
