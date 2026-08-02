#!/usr/bin/env python3
"""Run one shuffle only when its complete self-play input snapshot changed."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


STATE_SCHEMA_VERSION = 1
STATE_CONTRACT = "katago-shuffle-input-gate-v1"
MAX_INVENTORY_ATTEMPTS = 3


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"shuffle gate state must not be a symlink: {target}")
    data = (_canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_state(path: Path) -> Optional[Mapping[str, Any]]:
    source = Path(path)
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"shuffle gate state is not a regular file: {source}")
    data = source.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    payload = dict(value) if isinstance(value, dict) else {}
    state_hash = payload.pop("state_sha256", None)
    if (
        not isinstance(value, dict)
        or data != (_canonical_json(value) + "\n").encode("utf-8")
        or value.get("schema_version") != STATE_SCHEMA_VERSION
        or value.get("contract") != STATE_CONTRACT
        or state_hash != _sha256_json(payload)
    ):
        return None
    return value


def _is_temporary_npz_name(name: str) -> bool:
    # Keep this aligned with shuffle.py and summarize_old_selfplay_files.py.
    return "_" in name


def _inventory_once(input_root: Path) -> Sequence[Mapping[str, Any]]:
    records = []
    root = input_root.resolve()
    def raise_walk_error(error):
        raise error

    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
        directory_path = Path(directory)
        retained_dirnames = []
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"self-play input contains a symlink directory: {child}")
            if stat.S_ISDIR(metadata.st_mode):
                retained_dirnames.append(dirname)
        dirnames[:] = retained_dirnames
        for filename in sorted(filenames):
            if not filename.endswith(".npz") or _is_temporary_npz_name(filename):
                continue
            path = directory_path / filename
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"self-play NPZ is not a regular file: {path}")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": metadata.st_size,
                    "mtime_ns": metadata.st_mtime_ns,
                    "ctime_ns": metadata.st_ctime_ns,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                }
            )
    records.sort(key=lambda item: item["path"])
    return records


def input_inventory(input_root: Path) -> Sequence[Mapping[str, Any]]:
    root = Path(input_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("shuffle input root must be an absolute non-symlink directory")
    last_error: Optional[BaseException] = None
    for attempt in range(MAX_INVENTORY_ATTEMPTS):
        try:
            return _inventory_once(root)
        except (FileNotFoundError, OSError) as exc:
            last_error = exc
            if attempt + 1 < MAX_INVENTORY_ATTEMPTS:
                time.sleep(0.1)
    raise RuntimeError(f"self-play input inventory did not stabilize: {last_error}")


def _command_dependency_inventory(
    command: Sequence[str],
) -> Sequence[Mapping[str, Any]]:
    dependencies = []
    dependency_flags = {"-summary-file", "-exclude"}
    for index, part in enumerate(command[:-1]):
        if part not in dependency_flags:
            continue
        path = Path(command[index + 1])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            dependencies.append(
                {"flag": part, "path": str(path.resolve()), "missing": True}
            )
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"shuffle dependency is not a regular file: {path}")
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise RuntimeError(f"shuffle dependency changed while hashing: {path}")
        dependencies.append(
            {
                "flag": part,
                "path": str(path.resolve()),
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return dependencies


def build_fingerprint(
    input_root: Path, command: Sequence[str]
) -> Mapping[str, Any]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("shuffle command must be a nonempty argv array")
    inventory = input_inventory(input_root)
    inventory_digest = _sha256_json(inventory)
    dependencies = _command_dependency_inventory(command)
    dependency_digest = _sha256_json(dependencies)
    command_identity = {
        "cwd": str(Path.cwd().resolve()),
        "argv": list(command),
    }
    command_digest = _sha256_json(command_identity)
    identity = {
        "schema_version": STATE_SCHEMA_VERSION,
        "input_root": str(Path(input_root).resolve()),
        "inventory_sha256": inventory_digest,
        "dependency_sha256": dependency_digest,
        "command_sha256": command_digest,
    }
    return {
        **identity,
        "combined_sha256": _sha256_json(identity),
        "file_count": len(inventory),
        "total_bytes": sum(int(item["size"]) for item in inventory),
        "command": command_identity,
        "dependencies": dependencies,
    }


def _latest_output_name(output_root: Optional[Path]) -> Optional[str]:
    if output_root is None:
        return None
    root = Path(output_root)
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise ValueError("shuffle output root must be a non-symlink directory")
    candidates = []
    for entry in root.iterdir():
        if entry.name.endswith(".tmp") or entry.is_symlink() or not entry.is_dir():
            continue
        candidates.append((entry.stat().st_mtime_ns, entry.name))
    return max(candidates, default=(0, None))[1]


def _open_lock(path: Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(target), flags, 0o600)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError("shuffle gate lock must be a regular file")
    return os.fdopen(descriptor, "a+b", buffering=0)


def run_if_changed(
    *,
    input_root: Path,
    state_file: Path,
    lock_file: Path,
    output_root: Optional[Path],
    force_after_seconds: float,
    command: Sequence[str],
    environment: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    if not Path(state_file).is_absolute() or not Path(lock_file).is_absolute():
        raise ValueError("shuffle gate state and lock paths must be absolute")
    if force_after_seconds < 0:
        raise ValueError("force-after-seconds must be nonnegative")
    with _open_lock(lock_file) as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "SKIPPED_CONCURRENT"}

        fingerprint = build_fingerprint(input_root, command)
        state = _load_state(state_file)
        unchanged = (
            state is not None
            and state.get("combined_sha256") == fingerprint["combined_sha256"]
        )
        forced = False
        if unchanged and force_after_seconds > 0:
            recorded_at = state.get("recorded_at_unix")
            forced = (
                not isinstance(recorded_at, (int, float))
                or time.time() - float(recorded_at) >= force_after_seconds
            )
        if unchanged and not forced:
            return {
                "status": "SKIPPED_UNCHANGED",
                "combined_sha256": fingerprint["combined_sha256"],
                "file_count": fingerprint["file_count"],
                "total_bytes": fingerprint["total_bytes"],
            }

        child_environment = dict(os.environ if environment is None else environment)
        child_environment["KATAGO_SHUFFLE_GATE_BYPASS"] = "1"
        completed = subprocess.run(
            list(command),
            check=False,
            shell=False,
            env=child_environment,
            pass_fds=(lock.fileno(),),
        )
        if completed.returncode != 0:
            return {
                "status": "FAILED",
                "returncode": completed.returncode,
                "combined_sha256": fingerprint["combined_sha256"],
            }

        now = time.time()
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "contract": STATE_CONTRACT,
            **fingerprint,
            "last_successful_output": _latest_output_name(output_root),
            "recorded_at_unix": now,
            "recorded_at_utc": datetime.datetime.fromtimestamp(
                now, datetime.timezone.utc
            ).isoformat().replace("+00:00", "Z"),
        }
        state["state_sha256"] = _sha256_json(state)
        _atomic_write_json(state_file, state)
        return {
            "status": "SHUFFLED",
            "combined_sha256": fingerprint["combined_sha256"],
            "file_count": fingerprint["file_count"],
            "total_bytes": fingerprint["total_bytes"],
            "last_successful_output": state["last_successful_output"],
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--lock-file", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force-after-seconds", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_if_changed(
            input_root=args.input_root,
            state_file=args.state_file,
            lock_file=args.lock_file,
            output_root=args.output_root,
            force_after_seconds=args.force_after_seconds,
            command=args.command,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(_canonical_json(result), flush=True)
    if result["status"] == "FAILED":
        return int(result["returncode"]) or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
