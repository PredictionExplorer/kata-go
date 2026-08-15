#!/usr/bin/env python3
"""Self-play-only margin training with a permissive conventional-strength gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_controller import parse_candidate_counters
from risk_score.promotion_host import atomic_replace_json

SPEC_CONTRACT = "risk-score-selfplay-margin-loop-spec-v1"
POLICY_VERSION = "selfplay-margin-promotion-v1"
STATUS_CONTRACT = "risk-score-selfplay-margin-loop-status-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MarginLoopError(RuntimeError):
    """The self-play margin loop cannot safely continue."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    @classmethod
    def load(cls, value: Any, role: str) -> FileBinding:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise MarginLoopError(f"{role} must be a path/hash binding")
        path = _absolute_path(value["path"], f"{role}.path")
        digest = value["sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise MarginLoopError(f"{role}.sha256 must be lowercase SHA-256")
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise MarginLoopError(f"{role} is missing or changed")
        return cls(path, digest)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]

    @classmethod
    def load(cls, value: Any, role: str) -> CommandSpec:
        if not isinstance(value, Mapping) or set(value) != {"argv", "cwd", "env"}:
            raise MarginLoopError(f"{role} command keys differ from contract")
        argv = value["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(
                not isinstance(part, str)
                or not part
                or "\x00" in part
                or "\n" in part
                or "\r" in part
                for part in argv
            )
        ):
            raise MarginLoopError(f"{role}.argv must be a nonempty string array")
        executable = Path(argv[0])
        if not executable.is_absolute() or not executable.is_file():
            raise MarginLoopError(f"{role} executable is missing")
        cwd = _absolute_path(value["cwd"], f"{role}.cwd")
        if cwd.is_symlink() or not cwd.is_dir():
            raise MarginLoopError(f"{role}.cwd must be a regular directory")
        environment = value["env"]
        if not isinstance(environment, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or "\x00" in key
            or "\x00" in item
            for key, item in environment.items()
        ):
            raise MarginLoopError(f"{role}.env must be a string map")
        return cls(tuple(argv), cwd, dict(environment))


@dataclass(frozen=True)
class MarginPolicy:
    minimum_candidate_win_rate: float
    games_per_candidate: int
    visits_per_move: int
    maximum_active_candidates: int

    @classmethod
    def load(cls, binding: FileBinding) -> MarginPolicy:
        value = _load_json(binding.path, "margin policy", canonical=False)
        if (
            value.get("schema_version") != 1
            or value.get("policy_version") != POLICY_VERSION
            or value.get("status") != "active"
            or value.get("training", {}).get("data_source") != "self-play-only"
        ):
            raise MarginLoopError("margin policy identity is invalid")
        promotion = value.get("promotion")
        safety = value.get("safety")
        if not isinstance(promotion, Mapping) or not isinstance(safety, Mapping):
            raise MarginLoopError("margin policy sections are missing")
        if (
            promotion.get("external_position_data_allowed") is not False
            or promotion.get("comparison") != "fresh-candidate-vs-current-model"
            or promotion.get("comparison_search") != "standard"
            or promotion.get("automatic_promotion") is not True
            or safety.get("require_model_load_and_finite_output") is not True
            or safety.get("allow_handicap") is not False
            or safety.get("rollback_checkpoint_required") is not True
        ):
            raise MarginLoopError("margin policy weakens required safety controls")
        win_rate = _finite_number(
            promotion.get("minimum_candidate_win_rate"),
            "minimum candidate win rate",
        )
        if not 0.25 <= win_rate <= 0.5:
            raise MarginLoopError("minimum candidate win rate must be in [0.25, 0.5]")
        games = _positive_int(
            promotion.get("games_per_candidate"), "games per candidate"
        )
        visits = _positive_int(
            promotion.get("visits_per_move"), "visits per move"
        )
        maximum = _positive_int(
            promotion.get("maximum_active_candidates"),
            "maximum active candidates",
        )
        return cls(win_rate, games, visits, maximum)


@dataclass(frozen=True)
class MarginLoopSpec:
    path: Path
    file_sha256: str
    identity: str
    policy_binding: FileBinding
    policy: MarginPolicy
    run_root: Path
    candidate_inbox: Path
    accepted_models: Path
    rejected_models: Path
    superseded_models: Path
    selfplay_root: Path
    gate_sgf_root: Path
    status_path: Path
    lock_path: Path
    trainer: CommandSpec
    exporter: CommandSpec
    gatekeeper_binary: FileBinding
    gatekeeper_config: FileBinding
    gatekeeper_cwd: Path
    gatekeeper_env: Mapping[str, str]
    cycle_sleep_seconds: float

    @classmethod
    def load(cls, path: Path) -> MarginLoopSpec:
        source = Path(path).resolve()
        raw = _load_json(source, "margin loop specification")
        expected_keys = {
            "schema_version",
            "contract",
            "policy",
            "run_root",
            "paths",
            "trainer",
            "exporter",
            "gatekeeper",
            "cycle_sleep_seconds",
            "spec_sha256",
        }
        if set(raw) != expected_keys:
            raise MarginLoopError("margin loop specification keys differ")
        payload = dict(raw)
        identity = payload.pop("spec_sha256", None)
        if (
            raw.get("schema_version") != 1
            or raw.get("contract") != SPEC_CONTRACT
            or not isinstance(identity, str)
            or identity != canonical_sha256(payload)
        ):
            raise MarginLoopError("margin loop specification identity is invalid")
        policy_binding = FileBinding.load(raw["policy"], "policy")
        policy = MarginPolicy.load(policy_binding)
        run_root = _required_directory(raw["run_root"], "run root")
        paths = raw["paths"]
        path_keys = {
            "candidate_inbox",
            "accepted_models",
            "rejected_models",
            "superseded_models",
            "selfplay_root",
            "gate_sgf_root",
            "status",
            "lock",
        }
        if not isinstance(paths, Mapping) or set(paths) != path_keys:
            raise MarginLoopError("margin loop paths differ from contract")
        directories = {
            name: _required_directory(paths[name], name)
            for name in (
                "candidate_inbox",
                "accepted_models",
                "rejected_models",
                "superseded_models",
                "selfplay_root",
                "gate_sgf_root",
            )
        }
        for name, directory in directories.items():
            if not _strictly_within(directory, run_root):
                raise MarginLoopError(f"{name} must be inside run root")
        status = _absolute_path(paths["status"], "status path")
        lock = _absolute_path(paths["lock"], "lock path")
        if not _strictly_within(status, run_root) or not _strictly_within(
            lock, run_root
        ):
            raise MarginLoopError("status and lock must be inside run root")
        gatekeeper = raw["gatekeeper"]
        if not isinstance(gatekeeper, Mapping) or set(gatekeeper) != {
            "binary",
            "config",
            "cwd",
            "env",
        }:
            raise MarginLoopError("gatekeeper command keys differ from contract")
        gatekeeper_cwd = _required_directory(gatekeeper["cwd"], "gatekeeper cwd")
        gatekeeper_env = gatekeeper["env"]
        if (
            not isinstance(gatekeeper_env, Mapping)
            or gatekeeper_env.get("CUDA_VISIBLE_DEVICES") != "7"
        ):
            raise MarginLoopError("gatekeeper must be pinned to physical GPU 7")
        trainer = CommandSpec.load(raw["trainer"], "trainer")
        exporter = CommandSpec.load(raw["exporter"], "exporter")
        if trainer.env.get("CUDA_VISIBLE_DEVICES") != "7":
            raise MarginLoopError("trainer must be pinned to physical GPU 7")
        sleep_seconds = _finite_number(
            raw["cycle_sleep_seconds"], "cycle sleep seconds"
        )
        if not 1 <= sleep_seconds <= 3600:
            raise MarginLoopError("cycle sleep seconds must be in [1, 3600]")
        return cls(
            source,
            file_sha256(source),
            identity,
            policy_binding,
            policy,
            run_root,
            directories["candidate_inbox"],
            directories["accepted_models"],
            directories["rejected_models"],
            directories["superseded_models"],
            directories["selfplay_root"],
            directories["gate_sgf_root"],
            status,
            lock,
            trainer,
            exporter,
            FileBinding.load(gatekeeper["binary"], "gatekeeper binary"),
            FileBinding.load(gatekeeper["config"], "gatekeeper config"),
            gatekeeper_cwd,
            dict(gatekeeper_env),
            sleep_seconds,
        )


def _absolute_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MarginLoopError(f"{role} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise MarginLoopError(f"{role} must be absolute and normalized")
    return path


def _required_directory(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.is_symlink() or not path.is_dir():
        raise MarginLoopError(f"{role} must be a regular directory")
    return path


def _strictly_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return path != root
    except ValueError:
        return False


def _finite_number(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MarginLoopError(f"{role} must be finite")
    return float(value)


def _positive_int(value: Any, role: str) -> int:
    if type(value) is not int or value <= 0:
        raise MarginLoopError(f"{role} must be a positive integer")
    return value


def _load_json(
    path: Path, role: str, *, canonical: bool = True
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MarginLoopError(f"{role} is not a regular file")
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarginLoopError(f"cannot load {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise MarginLoopError(f"{role} must have an object root")
    if canonical and data != (canonical_json(value) + "\n").encode("utf-8"):
        raise MarginLoopError(f"{role} must be canonical newline-terminated JSON")
    return value


def build_gatekeeper_command(spec: MarginLoopSpec) -> tuple[str, ...]:
    return (
        str(spec.gatekeeper_binary.path),
        "gatekeeper",
        "-rejected-models-dir",
        str(spec.rejected_models),
        "-accepted-models-dir",
        str(spec.accepted_models),
        "-sgf-output-dir",
        str(spec.gate_sgf_root),
        "-test-models-dir",
        str(spec.candidate_inbox),
        "-selfplay-dir",
        str(spec.selfplay_root),
        "-config",
        str(spec.gatekeeper_config.path),
        "-required-candidate-win-prop",
        str(spec.policy.minimum_candidate_win_rate),
        "-quit-if-no-nets-to-test",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def coalesce_candidates(spec: MarginLoopSpec) -> tuple[str, ...]:
    candidates: list[tuple[int, int, str, Path]] = []
    for child in spec.candidate_inbox.iterdir():
        metadata = child.lstat()
        if child.name.startswith("."):
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MarginLoopError(f"candidate inbox contains unsafe entry: {child}")
        try:
            samples, rows = parse_candidate_counters(child.name)
        except ValueError as exc:
            raise MarginLoopError(f"candidate name is malformed: {child.name}") from exc
        candidates.append((samples, rows, child.name, child))
    candidates.sort()
    superseded = candidates[: -spec.policy.maximum_active_candidates]
    moved: list[str] = []
    for _, _, name, source in superseded:
        destination = spec.superseded_models / name
        if destination.exists():
            raise MarginLoopError(f"superseded destination already exists: {destination}")
        os.rename(source, destination)
        _fsync_directory(source.parent)
        _fsync_directory(destination.parent)
        moved.append(name)
    return tuple(moved)


class SelfplayMarginLoop:
    def __init__(
        self,
        spec: MarginLoopSpec | Path,
        *,
        runner: Callable[..., Any] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec if isinstance(spec, MarginLoopSpec) else MarginLoopSpec.load(spec)
        self.runner = runner
        self.sleeper = sleeper

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.spec.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.spec.lock_path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise MarginLoopError("another margin loop owns the lock") from exc
            yield

    def _persist(self, state: str, **details: Any) -> Mapping[str, Any]:
        value = {
            "schema_version": 1,
            "contract": STATUS_CONTRACT,
            "spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "state": state,
            **details,
        }
        value["status_sha256"] = canonical_sha256(value)
        atomic_replace_json(self.spec.status_path, value)
        return value

    def _run_command(self, role: str, command: CommandSpec) -> None:
        self._persist(f"running_{role}", argv=list(command.argv))
        environment = dict(os.environ)
        environment.update(command.env)
        completed = self.runner(
            command.argv,
            cwd=command.cwd,
            env=environment,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise MarginLoopError(f"{role} failed with status {completed.returncode}")

    def _run_gatekeeper(self) -> None:
        argv = build_gatekeeper_command(self.spec)
        self._persist("running_gatekeeper", argv=list(argv))
        environment = dict(os.environ)
        environment.update(self.spec.gatekeeper_env)
        completed = self.runner(
            argv,
            cwd=self.spec.gatekeeper_cwd,
            env=environment,
            shell=False,
            check=False,
        )
        if completed.returncode != 0:
            raise MarginLoopError(
                f"gatekeeper failed with status {completed.returncode}"
            )

    def once(self) -> Mapping[str, Any]:
        with self._lock():
            self._run_command("exporter", self.spec.exporter)
            moved = coalesce_candidates(self.spec)
            self._run_gatekeeper()
            self._run_command("trainer", self.spec.trainer)
            return self._persist("cycle_complete", superseded=list(moved))

    def watch(self) -> None:
        while True:
            self.once()
            self.sleeper(self.spec.cycle_sleep_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("once", "watch", "status", "command"))
    parser.add_argument("--spec", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = MarginLoopSpec.load(args.spec)
        if args.mode == "status":
            value = (
                _load_json(spec.status_path, "margin loop status")
                if spec.status_path.exists()
                else {"state": "not_started"}
            )
            print(canonical_json(value))
            return 0
        if args.mode == "command":
            print(canonical_json({"argv": list(build_gatekeeper_command(spec))}))
            return 0
        loop = SelfplayMarginLoop(spec)
        if args.mode == "once":
            print(canonical_json(loop.once()))
            return 0
        loop.watch()
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
