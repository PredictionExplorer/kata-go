#!/usr/bin/env python3
"""Live-volume and immutable deployment preflight checks."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_host import (
    HostCommandError,
    atomic_replace_json,
    atomic_write_json,
)


def filesystem_test(root: Path) -> Mapping[str, Any]:
    target = Path(root)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise HostCommandError("filesystem test root must be an absolute directory")
    temporary = Path(tempfile.mkdtemp(prefix=".promotion-fs-test-", dir=str(target)))
    source = temporary / "source"
    destination = temporary / "destination"
    payload = os.urandom(4096)
    try:
        with source.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(str(temporary), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
            source_inode = source.stat().st_ino
            os.rename(source, destination)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if (
            source.exists()
            or not destination.is_file()
            or destination.read_bytes() != payload
            or destination.stat().st_ino != source_inode
        ):
            raise HostCommandError("atomic rename/fsync semantics failed")
        return {
            "schema_version": 1,
            "contract": "risk-score-live-filesystem-test-v1",
            "root": str(target.resolve()),
            "device": target.stat().st_dev,
            "atomic_rename_preserved_inode": True,
            "directory_fsync_succeeded": True,
            "payload_sha256": file_sha256(destination),
        }
    finally:
        for path in (source, destination):
            path.unlink(missing_ok=True)
        temporary.rmdir()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise HostCommandError(f"git {' '.join(args)} failed: {completed.stderr}")
    return completed.stdout.strip()


def deployment_snapshot(
    *,
    run_root: Path,
    repo: Path,
    output: Path,
    require_procfs: bool = True,
) -> Mapping[str, Any]:
    root = Path(run_root)
    source = Path(repo)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise HostCommandError("run root must be an absolute non-symlink directory")
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise HostCommandError("repository must be an absolute non-symlink directory")
    processes = []
    proc_root = Path("/proc")
    if require_procfs and not (proc_root / "self" / "stat").is_file():
        raise HostCommandError("verified process inventory requires Linux procfs")
    if proc_root.is_dir():
        for process in proc_root.iterdir():
            if not process.name.isdigit():
                continue
            try:
                command = (process / "cmdline").read_bytes()
            except OSError:
                continue
            if str(root).encode() in command:
                processes.append(
                    {
                        "pid": int(process.name),
                        "command_sha256": hashlib.sha256(command).hexdigest(),
                        "command": command.replace(b"\0", b" ").decode(
                            errors="replace"
                        ),
                    }
                )
    snapshot: Dict[str, Any] = {
        "schema_version": 1,
        "contract": "risk-score-live-deployment-snapshot-v1",
        "captured_at_unix": time.time(),
        "run_root": str(root.resolve()),
        "run_root_device": root.stat().st_dev,
        "repository": str(source.resolve()),
        "source_revision": _git(source, "rev-parse", "HEAD"),
        "source_status": _git(source, "status", "--porcelain=v1"),
        "process_inventory_source": (
            "linux-procfs" if proc_root.is_dir() else "unavailable"
        ),
        "processes": sorted(processes, key=lambda item: item["pid"]),
        "candidate_names": sorted(
            path.name
            for path in (root / "modelstobetested").iterdir()
            if path.is_dir() and not path.is_symlink()
        )
        if (root / "modelstobetested").is_dir()
        else [],
        "legacy_model_names": sorted(
            path.name
            for path in (root / "models").iterdir()
            if path.is_dir() and not path.is_symlink()
        )
        if (root / "models").is_dir()
        else [],
        "controller_accepted_names": sorted(
            path.name
            for path in (root / "promotion" / "accepted").iterdir()
            if path.is_dir() and not path.is_symlink()
        )
        if (root / "promotion" / "accepted").is_dir()
        else [],
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    atomic_write_json(output, snapshot)
    return snapshot


def candidate_inventory(inbox: Path, output: Path) -> Mapping[str, Any]:
    from risk_score.promotion_controller import inventory_candidates

    root = Path(inbox)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise HostCommandError("candidate inbox must be an absolute directory")
    candidates, ignored = inventory_candidates(root)
    rows = [
        {
            "name": candidate.name,
            "path": str(candidate.path),
            "model_sha256": candidate.model_hash,
            "checkpoint_sha256": candidate.checkpoint_hash,
            "directory_manifest_sha256": candidate.directory_manifest_hash,
            "sample_count": candidate.sample_count,
            "data_count": candidate.data_count,
            "size_bytes": candidate.size_bytes,
        }
        for candidate in candidates
    ]
    value = {
        "schema_version": 1,
        "contract": "risk-score-live-candidate-inventory-v1",
        "inbox": str(root.resolve()),
        "candidate_count": len(rows),
        "ignored": list(ignored),
        "candidates": rows,
    }
    value["inventory_sha256"] = canonical_sha256(value)
    atomic_write_json(output, value)
    return value


def bootstrap_backpressure(output: Path, policy_hash: str) -> Mapping[str, Any]:
    target = Path(output)
    if (
        not target.is_absolute()
        or target.is_symlink()
        or not re.fullmatch(r"[0-9a-f]{64}", policy_hash)
    ):
        raise HostCommandError(
            "backpressure output must be absolute and policy hash must be lowercase SHA-256"
        )
    if target.exists():
        if not target.is_file():
            raise HostCommandError("existing backpressure output is not a regular file")
        data = target.read_bytes()
        try:
            existing = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HostCommandError(f"existing backpressure output is invalid: {exc}") from exc
        if (
            not isinstance(existing, dict)
            or data != (canonical_json(existing) + "\n").encode("utf-8")
            or existing.get("schema_version") != 1
            or existing.get("policy_hash") != policy_hash
            or type(existing.get("allowExport")) is not bool
        ):
            raise HostCommandError("existing backpressure output conflicts with bootstrap")
        if (
            existing.get("allowExport") is False
            and existing.get("exportPaused") is True
        ):
            return existing
    value = {
        "schema_version": 1,
        "policy_hash": policy_hash,
        "controller_hash": "0" * 64,
        "updated_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "allowExport": False,
        "allowEvaluation": False,
        "exportPaused": True,
        "evaluationPaused": True,
        "exportBacklogDepth": 0,
        "evaluationBacklogDepth": 0,
        "maximumActiveEvaluatorEntries": 3,
        "importantQueueWarningDepth": 4,
        "reasons": ["bootstrap-pre-controller"],
    }
    if target.exists():
        atomic_replace_json(target, value)
    else:
        atomic_write_json(target, value)
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    filesystem = subparsers.add_parser("filesystem-test")
    filesystem.add_argument("--root", required=True, type=Path)
    filesystem.add_argument("--output", type=Path)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--run-root", required=True, type=Path)
    snapshot.add_argument("--repo", required=True, type=Path)
    snapshot.add_argument("--output", required=True, type=Path)
    inventory = subparsers.add_parser("candidate-inventory")
    inventory.add_argument("--inbox", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    backpressure = subparsers.add_parser("bootstrap-backpressure")
    backpressure.add_argument("--output", required=True, type=Path)
    backpressure.add_argument("--policy-hash", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "filesystem-test":
            result = filesystem_test(args.root)
            if args.output is not None:
                atomic_write_json(args.output, result)
        elif args.command == "snapshot":
            result = deployment_snapshot(
                run_root=args.run_root,
                repo=args.repo,
                output=args.output,
                require_procfs=True,
            )
        elif args.command == "candidate-inventory":
            result = candidate_inventory(args.inbox, args.output)
        else:
            result = bootstrap_backpressure(args.output, args.policy_hash)
    except (HostCommandError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
