"""Freeze balanced focal-vs-snapshot worker plans for extreme-score self-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat as stat_module
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from risk_score.extreme_score_controller import load_accepted_state

REQUEST_CONTRACT = "risk-score-extreme-league-request-v1"
PLAN_CONTRACT = "risk-score-extreme-league-plan-v1"
STATUS_CONTRACT = "risk-score-extreme-league-status-v1"
WORKER_RECEIPT_CONTRACT = "risk-score-extreme-league-worker-receipt-v1"
WORKER_EXECUTION_CONTRACT = "risk-score-extreme-league-worker-execution-v1"
TRAINING_POLICY_CONTRACT = "risk-score-extreme-score-training-policy-v1"
TRAINING_POLICY_VERSION = "expected-max-focal-selfplay-v1"
DEFAULT_POLICY_PATH = Path(__file__).with_name("extreme_score_training_policy_v1.json")
_SHA256_LENGTH = 64
_COLORS = ("B", "W")
_MAX_COHORT_SIZE = 8
_RECEIPT_NAME = "worker-execution-receipt.json"
_WRITE_BITS = stat_module.S_IWUSR | stat_module.S_IWGRP | stat_module.S_IWOTH
_EXPECTED_OPPONENT_WEIGHTS = {
    "latest_frozen": 0.5,
    "recent_frozen": 0.3,
    "score_minimizing_exploiter": 0.2,
}
_EXPECTED_POLICY = {
    "schema_version": 1,
    "contract": TRAINING_POLICY_CONTRACT,
    "policy_version": TRAINING_POLICY_VERSION,
    "status": "frozen",
    "objective": {
        "name": "expected_maximum_focal_terminal_score",
        "win_loss_weight": 0,
        "cohort_sizes": [1, 2, 4, 8],
        "production_cohort_size": 8,
        "search_approximation": "legally_clamped_gaussian_order_statistic",
        "maximum_supported_cohort_size": 8,
    },
    "curriculum": [
        {"cohort_size": 1, "minimum_selected_training_samples": 0},
        {"cohort_size": 2, "minimum_selected_training_samples": 2_000_000},
        {"cohort_size": 4, "minimum_selected_training_samples": 4_000_000},
        {"cohort_size": 8, "minimum_selected_training_samples": 8_000_000},
    ],
    "league": {
        "focal_colors": ["B", "W"],
        "workers_per_gpu": 2,
        "threads_per_worker": 50,
        "opponent_weights": _EXPECTED_OPPONENT_WEIGHTS,
        "opponent_gradient": "stopped",
        "worker_allocation": "largest_remainder_per_focal_color",
        "minimum_workers_per_required_opponent_per_color": 1,
        "model_selection": "single_immutable_content_addressed_snapshot",
        "maximum_games_per_worker": 100_000,
    },
    "training": {
        "initial_checkpoint": "immutable_original_b40",
        "extreme_score_only": True,
        "allow_mixed_legacy_rows": False,
        "optimistic_policy_from_win_loss": False,
        "resignation": False,
    },
    "evaluation": {
        "policy": "risk-score-held-out-expected-max-v1",
        "raw_lifetime_record_is_gating": False,
        "required_slices": ["overall", "B", "W"],
    },
}


class ExtremeScoreLeagueError(ValueError):
    """A league request or frozen worker plan is invalid."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    try:
        if not stat_module.S_ISREG(os.fstat(descriptor).st_mode):
            raise ExtremeScoreLeagueError(f"artifact is not a regular file: {source}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ExtremeScoreLeagueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ExtremeScoreLeagueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _decode_json(data: bytes, role: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtremeScoreLeagueError(f"cannot load {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtremeScoreLeagueError(f"{role} must have an object root")
    return value


def _lstat(path: Path, role: str) -> os.stat_result:
    try:
        return Path(path).lstat()
    except FileNotFoundError as exc:
        raise ExtremeScoreLeagueError(f"{role} is missing: {path}") from exc


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    mode = _lstat(source, role).st_mode
    if source.is_symlink() or not stat_module.S_ISREG(mode):
        raise ExtremeScoreLeagueError(f"{role} must be a regular non-symlink file")
    return source


def _regular_directory(path: Path, role: str) -> Path:
    source = Path(path)
    mode = _lstat(source, role).st_mode
    if source.is_symlink() or not stat_module.S_ISDIR(mode):
        raise ExtremeScoreLeagueError(f"{role} must be a regular non-symlink directory")
    return source


def _is_read_only(mode: int) -> bool:
    return mode & _WRITE_BITS == 0


def _require_read_only_file(path: Path, role: str) -> Path:
    source = _regular_file(path, role)
    if not _is_read_only(source.lstat().st_mode):
        raise ExtremeScoreLeagueError(f"{role} must be read-only")
    return source


def _text(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ExtremeScoreLeagueError(
            f"{role} must be a nonempty trimmed single-line string"
        )
    return value


def _safe_identifier(value: Any, role: str) -> str:
    text = _text(value, role)
    if text in {".", ".."} or any(
        not (character.isascii() and (character.isalnum() or character in "._-"))
        for character in text
    ):
        raise ExtremeScoreLeagueError(
            f"{role} must contain only ASCII letters, digits, '.', '_' or '-'"
        )
    return text


def _sha256(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExtremeScoreLeagueError(f"{role} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, role: str, maximum: int | None = None) -> int:
    if type(value) is not int or value <= 0:
        raise ExtremeScoreLeagueError(f"{role} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ExtremeScoreLeagueError(
            f"{role} must be a positive integer no greater than {maximum}"
        )
    return value


def _absolute_path(value: Any, role: str) -> Path:
    text = _text(value, role)
    path = Path(text)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ExtremeScoreLeagueError(f"{role} must be absolute and normalized")
    return path


def _file_binding(value: Any, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ExtremeScoreLeagueError(f"{role} must be a path/hash binding")
    path = _absolute_path(value["path"], f"{role}.path")
    digest = _sha256(value["sha256"], f"{role}.sha256")
    try:
        source = _regular_file(path, role)
    except ExtremeScoreLeagueError as exc:
        raise ExtremeScoreLeagueError(f"{role} is missing or changed") from exc
    if file_sha256(source) != digest:
        raise ExtremeScoreLeagueError(f"{role} is missing or changed")
    return {"path": str(path), "sha256": digest}


def _model_binding(value: Any, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "snapshot_id",
        "directory",
        "model_sha256",
    }:
        raise ExtremeScoreLeagueError(f"{role} model binding is malformed")
    snapshot_id = _text(value["snapshot_id"], f"{role}.snapshot_id")
    directory = _absolute_path(value["directory"], f"{role}.directory")
    digest = _sha256(value["model_sha256"], f"{role}.model_sha256")
    model = directory / "model.bin.gz"
    try:
        _regular_directory(directory, f"{role} model directory")
        source = _regular_file(model, f"{role} model")
    except ExtremeScoreLeagueError as exc:
        raise ExtremeScoreLeagueError(
            f"{role} model directory is missing or changed"
        ) from exc
    if file_sha256(source) != digest:
        raise ExtremeScoreLeagueError(f"{role} model directory is missing or changed")
    return {
        "snapshot_id": snapshot_id,
        "directory": str(directory),
        "model_sha256": digest,
    }


def _read_policy(
    path: Path = DEFAULT_POLICY_PATH,
) -> tuple[Path, dict[str, Any], str]:
    supplied = _regular_file(Path(path), "extreme-score training policy")
    source = supplied.resolve()
    data = _read_regular_bytes(source, "extreme-score training policy")
    policy = _decode_json(data, "extreme-score training policy")
    if policy != _EXPECTED_POLICY:
        raise ExtremeScoreLeagueError(
            "extreme-score training policy differs from the frozen schema"
        )
    return source, policy, hashlib.sha256(data).hexdigest()


def load_training_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load and strictly validate the frozen production training policy."""

    _, policy, _ = _read_policy(path)
    return json.loads(canonical_json(policy))


def _policy_binding(
    source: Path, policy: Mapping[str, Any], file_digest: str
) -> dict[str, Any]:
    return {
        "path": str(source),
        "file_sha256": file_digest,
        "canonical_sha256": canonical_sha256(policy),
        "schema_version": policy["schema_version"],
        "contract": policy["contract"],
        "policy_version": policy["policy_version"],
    }


def _validate_policy_binding(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {
        "path",
        "file_sha256",
        "canonical_sha256",
        "schema_version",
        "contract",
        "policy_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ExtremeScoreLeagueError("plan policy binding is malformed")
    path = _absolute_path(value["path"], "plan policy path")
    checked = {
        "path": str(path),
        "file_sha256": _sha256(value["file_sha256"], "plan policy file SHA-256"),
        "canonical_sha256": _sha256(
            value["canonical_sha256"], "plan policy canonical SHA-256"
        ),
        "schema_version": value["schema_version"],
        "contract": value["contract"],
        "policy_version": value["policy_version"],
    }
    source, policy, file_digest = _read_policy(path)
    expected = _policy_binding(source, policy, file_digest)
    if checked != expected:
        raise ExtremeScoreLeagueError(
            "extreme-score training policy changed after plan publication"
        )
    return checked, policy


def _accepted_state_binding(
    value: Any,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "state_sha256",
    }:
        raise ExtremeScoreLeagueError(
            "accepted_state must bind path, file hash, and state hash"
        )
    path = _regular_file(
        _absolute_path(value["path"], "accepted_state.path"),
        "accepted model state",
    )
    file_digest = _sha256(value["file_sha256"], "accepted_state.file_sha256")
    if file_sha256(path) != file_digest:
        raise ExtremeScoreLeagueError("accepted model state file changed")
    try:
        state = load_accepted_state(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtremeScoreLeagueError(
            "accepted model state or artifact changed"
        ) from exc
    if state["state_sha256"] != _sha256(
        value["state_sha256"], "accepted_state.state_sha256"
    ):
        raise ExtremeScoreLeagueError("accepted model state identity changed")
    return {
        "path": str(path),
        "file_sha256": file_digest,
        "state_sha256": state["state_sha256"],
    }, state


def _curriculum_state(
    policy: Mapping[str, Any],
    *,
    group_size: int,
    selected_training_samples: int,
) -> dict[str, Any]:
    maximum = policy["objective"]["maximum_supported_cohort_size"]
    if group_size > maximum:
        raise ExtremeScoreLeagueError(
            f"group_size N must not exceed policy maximum {maximum}"
        )
    stages = policy["curriculum"]
    eligible = [
        (index, stage)
        for index, stage in enumerate(stages)
        if selected_training_samples >= stage["minimum_selected_training_samples"]
    ]
    if not eligible:
        raise ExtremeScoreLeagueError(
            "selected_training_samples has not reached the first curriculum stage"
        )
    stage_index, stage = eligible[-1]
    if group_size != stage["cohort_size"]:
        raise ExtremeScoreLeagueError(
            "group_size does not match the policy curriculum state"
        )
    return {
        "stage_index": stage_index,
        "cohort_size": stage["cohort_size"],
        "minimum_selected_training_samples": stage["minimum_selected_training_samples"],
        "selected_training_samples": selected_training_samples,
        "production": group_size == policy["objective"]["production_cohort_size"],
        "curriculum_sha256": canonical_sha256(stages),
    }


def _read_regular_bytes(path: Path, role: str) -> bytes:
    source = _regular_file(path, role)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        if not stat_module.S_ISREG(os.fstat(descriptor).st_mode):
            raise ExtremeScoreLeagueError(f"{role} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_immutable_bytes(path: Path, data: bytes, role: str) -> None:
    source = _require_read_only_file(path, role)
    if _read_regular_bytes(source, role) != data:
        raise ExtremeScoreLeagueError(f"existing immutable {role} conflicts")


def _publish_immutable_bytes(path: Path, data: bytes, role: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _regular_directory(target.parent, f"{role} parent")
    if target.exists() or target.is_symlink():
        _verify_immutable_bytes(target, data, role)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            _verify_immutable_bytes(target, data, role)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_immutable_hash(path: Path, digest: str, role: str) -> None:
    source = _require_read_only_file(path, role)
    if file_sha256(source) != digest:
        raise ExtremeScoreLeagueError(f"existing immutable {role} conflicts")


def _publish_immutable_copy(
    source: Path, target: Path, expected_digest: str, role: str
) -> None:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _regular_directory(destination.parent, f"{role} parent")
    if destination.exists() or destination.is_symlink():
        _verify_immutable_hash(destination, expected_digest, role)
        return

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(_regular_file(source, role), flags)
    try:
        if not stat_module.S_ISREG(os.fstat(source_fd).st_mode):
            raise ExtremeScoreLeagueError(f"{role} source is not regular")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    block = os.read(source_fd, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != expected_digest:
                raise ExtremeScoreLeagueError(f"{role} changed while snapshotting")
            os.chmod(temporary, 0o444)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                _verify_immutable_hash(destination, expected_digest, role)
            directory_fd = os.open(
                destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        os.close(source_fd)


def _verify_snapshot_directory(
    directory: Path,
    expected_digest: str,
    role: str,
    *,
    probe: Callable[[Path], str] = file_sha256,
) -> None:
    snapshot = _regular_directory(directory, f"{role} snapshot directory")
    if not _is_read_only(snapshot.lstat().st_mode):
        raise ExtremeScoreLeagueError(f"{role} snapshot directory must be read-only")
    entries = list(snapshot.iterdir())
    if len(entries) != 1 or entries[0].name != "model.bin.gz":
        raise ExtremeScoreLeagueError(
            f"{role} snapshot contains a model absent from the plan"
        )
    model = _require_read_only_file(entries[0], f"{role} snapshot model")
    if probe(model) != expected_digest:
        raise ExtremeScoreLeagueError(f"{role} snapshot model changed")


def _materialize_model_snapshot(
    binding: Mapping[str, str],
    *,
    output_root: Path,
    role: str,
) -> dict[str, str]:
    source_directory = Path(binding["directory"])
    source_model = source_directory / "model.bin.gz"
    digest = binding["model_sha256"]
    output_root.mkdir(parents=True, exist_ok=True)
    _regular_directory(output_root, "output_root")
    snapshot_root = output_root / ".extreme-score-model-snapshots"
    snapshot_root.mkdir(exist_ok=True)
    _regular_directory(snapshot_root, "model snapshot root")
    snapshot_directory = snapshot_root / digest
    if snapshot_directory.exists() or snapshot_directory.is_symlink():
        _verify_snapshot_directory(snapshot_directory, digest, role)
    else:
        snapshot_directory.mkdir(mode=0o755)
        _publish_immutable_copy(
            source_model,
            snapshot_directory / "model.bin.gz",
            digest,
            f"{role} model snapshot",
        )
        os.chmod(snapshot_directory, 0o555)
        _verify_snapshot_directory(snapshot_directory, digest, role)
    return {
        "snapshot_id": binding["snapshot_id"],
        "directory": str(snapshot_directory),
        "model_sha256": digest,
    }


def _allocation_counts(weights: Sequence[float], count: int) -> list[int]:
    exact = [weight * count for weight in weights]
    allocated = [math.floor(value) for value in exact]
    remaining = count - sum(allocated)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - allocated[index]), index),
    )
    for index in order[:remaining]:
        allocated[index] += 1
    return allocated


def _override_value(value: str | Path, role: str) -> str:
    text = str(value)
    if not text or any(character in text for character in ("\x00", "\n", "\r", ",")):
        raise ExtremeScoreLeagueError(f"{role} is unsafe for config overrides")
    return text


def _worker_command(
    *,
    binary: Path,
    config: Path,
    focal_directory: Path,
    opponent_directory: Path,
    output_directory: Path,
    gpu_index: int,
    focal_color: str,
    group_size: int,
    threads: int,
    games_per_worker: int,
) -> list[str]:
    overrides = [
        f"cudaDeviceToUseModel0Thread0={gpu_index}",
        f"numGameThreads={threads}",
        f"extremeCohortSize={group_size}",
        f"extremeScoreGroupSize={group_size}",
        f"extremeCohortFocalColor={focal_color}",
        f"expectedMaxFocalColor={focal_color}",
        "switchNetsMidGame=false",
        "useScoreMaximizingUtility=false",
        "useExpectedMaxScoreUtility=true",
        "winWeight=0",
    ]
    return [
        str(binary),
        "selfplay",
        "-models-dir",
        str(focal_directory),
        "-opponent-models-dir",
        str(opponent_directory),
        "-output-dir",
        str(output_directory),
        "-max-games-total",
        str(games_per_worker),
        "-config",
        str(config),
        "-override-config",
        ",".join(_override_value(value, "override") for value in overrides),
    ]


def build_plan(
    request: Mapping[str, Any],
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    """Validate a request and return one immutable, self-hashed worker plan."""

    expected_keys = {
        "schema_version",
        "contract",
        "generation_id",
        "binary",
        "config",
        "accepted_state",
        "opponents",
        "gpu_indices",
        "threads_per_worker",
        "group_size",
        "games_per_worker",
        "output_root",
    }
    if not isinstance(request, Mapping) or set(request) != expected_keys:
        raise ExtremeScoreLeagueError("league request keys differ from contract")
    if request["schema_version"] != 1 or request["contract"] != REQUEST_CONTRACT:
        raise ExtremeScoreLeagueError("league request identity is invalid")
    policy_source, policy, policy_file_digest = _read_policy(policy_path)
    policy_binding = _policy_binding(policy_source, policy, policy_file_digest)
    league_policy = policy["league"]

    generation_id = _safe_identifier(request["generation_id"], "generation_id")
    binary = _file_binding(request["binary"], "KataGo binary")
    if not os.access(binary["path"], os.X_OK):
        raise ExtremeScoreLeagueError("KataGo binary must be executable")
    config = _file_binding(request["config"], "self-play config")
    accepted_state_binding, accepted_state = _accepted_state_binding(
        request["accepted_state"]
    )
    accepted_policy = accepted_state["training_progress"]["training_policy"]
    if accepted_policy["file_sha256"] != policy_binding["file_sha256"]:
        raise ExtremeScoreLeagueError(
            "accepted state training policy differs from league policy"
        )
    accepted_model = accepted_state["model"]
    accepted_artifact = accepted_state["artifact"]
    focal_source = _model_binding(
        {
            "snapshot_id": accepted_model["model_id"],
            "directory": str(Path(accepted_artifact["path"]).parent),
            "model_sha256": accepted_model["sha256"],
        },
        "accepted focal",
    )
    output_root = _absolute_path(request["output_root"], "output_root")
    if output_root.exists() or output_root.is_symlink():
        _regular_directory(output_root, "output_root")

    raw_gpus = request["gpu_indices"]
    if (
        not isinstance(raw_gpus, list)
        or not raw_gpus
        or any(type(index) is not int or index < 0 for index in raw_gpus)
        or len(set(raw_gpus)) != len(raw_gpus)
    ):
        raise ExtremeScoreLeagueError("gpu_indices must be unique nonnegative integers")
    gpu_indices = tuple(raw_gpus)
    threads = _positive_int(request["threads_per_worker"], "threads_per_worker")
    if threads != league_policy["threads_per_worker"]:
        raise ExtremeScoreLeagueError(
            "threads_per_worker differs from the frozen training policy"
        )
    group_size = _positive_int(request["group_size"], "group_size", _MAX_COHORT_SIZE)
    games_per_worker = _positive_int(
        request["games_per_worker"],
        "games_per_worker",
        league_policy["maximum_games_per_worker"],
    )
    if games_per_worker % group_size != 0:
        raise ExtremeScoreLeagueError(
            "games_per_worker must be a multiple of group_size"
        )
    selected_training_samples = accepted_state["training_progress"][
        "selected_training_samples"
    ]
    curriculum_state = _curriculum_state(
        policy,
        group_size=group_size,
        selected_training_samples=selected_training_samples,
    )
    focal_colors = tuple(league_policy["focal_colors"])
    if focal_colors != _COLORS or league_policy["workers_per_gpu"] != len(focal_colors):
        raise ExtremeScoreLeagueError("policy worker topology is unsupported")

    raw_opponents = request["opponents"]
    if not isinstance(raw_opponents, list) or not raw_opponents:
        raise ExtremeScoreLeagueError("opponents must be a nonempty array")
    opponents_by_role: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw_opponents):
        if not isinstance(value, Mapping) or set(value) != {"role", "weight", "model"}:
            raise ExtremeScoreLeagueError(f"opponent {index} keys differ")
        role = _text(value["role"], f"opponent {index}.role")
        if role in opponents_by_role:
            raise ExtremeScoreLeagueError("opponent roles must be unique")
        weight = value["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise ExtremeScoreLeagueError(f"opponent {index}.weight must be positive")
        opponents_by_role[role] = {
            "role": role,
            "weight": float(weight),
            "model": _model_binding(value["model"], f"opponent {index}"),
        }

    policy_weights = league_policy["opponent_weights"]
    if set(opponents_by_role) != set(policy_weights):
        raise ExtremeScoreLeagueError(
            "opponent roles differ from the frozen training policy"
        )
    opponents = []
    for role in sorted(policy_weights):
        opponent = opponents_by_role[role]
        if opponent["weight"] != float(policy_weights[role]):
            raise ExtremeScoreLeagueError(
                f"opponent {role!r} weight differs from the frozen training policy"
            )
        opponents.append(opponent)
    if len({item["model"]["snapshot_id"] for item in opponents}) != len(opponents):
        raise ExtremeScoreLeagueError("opponent snapshot IDs must be unique")

    counts = _allocation_counts(
        [item["weight"] for item in opponents], len(gpu_indices)
    )
    required_minimum = league_policy["minimum_workers_per_required_opponent_per_color"]
    if any(count < required_minimum for count in counts):
        raise ExtremeScoreLeagueError(
            "insufficient worker slots for every required opponent"
        )

    focal = _materialize_model_snapshot(
        focal_source,
        output_root=output_root,
        role="focal",
    )
    frozen_opponents = []
    for opponent in opponents:
        frozen_opponents.append(
            {
                "role": opponent["role"],
                "weight": opponent["weight"],
                "model": _materialize_model_snapshot(
                    opponent["model"],
                    output_root=output_root,
                    role=f"opponent {opponent['role']}",
                ),
            }
        )
    opponents = frozen_opponents

    opponent_sequence = [
        opponent_index
        for opponent_index, allocation in enumerate(counts)
        for _ in range(allocation)
    ]
    workers = []
    for color in focal_colors:
        for slot, gpu_index in enumerate(gpu_indices):
            opponent = opponents[opponent_sequence[slot]]
            worker_id = f"{color.lower()}-gpu-{gpu_index:03d}"
            output = output_root / generation_id / worker_id
            command = _worker_command(
                binary=Path(binary["path"]),
                config=Path(config["path"]),
                focal_directory=Path(focal["directory"]),
                opponent_directory=Path(opponent["model"]["directory"]),
                output_directory=output,
                gpu_index=gpu_index,
                focal_color=color,
                group_size=group_size,
                threads=threads,
                games_per_worker=games_per_worker,
            )
            workers.append(
                {
                    "worker_id": worker_id,
                    "gpu_index": gpu_index,
                    "focal_color": color,
                    "opponent_role": opponent["role"],
                    "opponent": opponent["model"],
                    "output_directory": str(output),
                    "katago_argv": command,
                    "katago_argv_sha256": canonical_sha256(command),
                }
            )

    launcher_path = Path(__file__).resolve()
    launcher = {
        "path": str(_regular_file(launcher_path, "league launcher")),
        "sha256": file_sha256(launcher_path),
    }
    realized_allocations = [
        {
            "role": opponent["role"],
            "snapshot_id": opponent["model"]["snapshot_id"],
            "weight": opponent["weight"],
            "workers_per_focal_color": counts[index],
            "total_workers": counts[index] * len(focal_colors),
        }
        for index, opponent in enumerate(opponents)
    ]
    plan = {
        "schema_version": 1,
        "contract": PLAN_CONTRACT,
        "execution_contract": WORKER_EXECUTION_CONTRACT,
        "generation_id": generation_id,
        "policy": policy_binding,
        "accepted_state": accepted_state_binding,
        "curriculum_state": curriculum_state,
        "launcher": launcher,
        "binary": binary,
        "config": config,
        "focal_model": focal,
        "opponents": opponents,
        "gpu_indices": list(gpu_indices),
        "threads_per_worker": threads,
        "group_size": group_size,
        "games_per_worker": games_per_worker,
        "selected_training_samples": selected_training_samples,
        "output_root": str(output_root),
        "realized_allocation": {
            "algorithm": league_policy["worker_allocation"],
            "workers_per_gpu": league_policy["workers_per_gpu"],
            "required_minimum_per_opponent_per_color": required_minimum,
            "total_workers": len(workers),
            "opponents": realized_allocations,
        },
        "workers": workers,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract",
        "execution_contract",
        "generation_id",
        "policy",
        "accepted_state",
        "curriculum_state",
        "launcher",
        "binary",
        "config",
        "focal_model",
        "opponents",
        "gpu_indices",
        "threads_per_worker",
        "group_size",
        "games_per_worker",
        "selected_training_samples",
        "output_root",
        "realized_allocation",
        "workers",
        "plan_sha256",
    }
    if not isinstance(plan, Mapping) or set(plan) != expected_keys:
        raise ExtremeScoreLeagueError("league plan keys differ from contract")
    payload = dict(plan)
    supplied = payload.pop("plan_sha256", None)
    if (
        plan.get("schema_version") != 1
        or plan.get("contract") != PLAN_CONTRACT
        or plan.get("execution_contract") != WORKER_EXECUTION_CONTRACT
        or not isinstance(supplied, str)
        or supplied != canonical_sha256(payload)
    ):
        raise ExtremeScoreLeagueError("league plan self-hash is invalid")
    policy_binding, _ = _validate_policy_binding(plan["policy"])
    synthetic = {
        key: plan[key]
        for key in (
            "schema_version",
            "generation_id",
            "binary",
            "config",
            "accepted_state",
            "opponents",
            "gpu_indices",
            "threads_per_worker",
            "group_size",
            "games_per_worker",
            "output_root",
        )
    }
    synthetic["contract"] = REQUEST_CONTRACT
    rebuilt = build_plan(
        synthetic,
        policy_path=Path(policy_binding["path"]),
    )
    if rebuilt != dict(plan):
        raise ExtremeScoreLeagueError("league plan is not canonical")
    return json.loads(canonical_json(plan))


def _load_json(path: Path, role: str) -> dict[str, Any]:
    source = _regular_file(Path(path), role)
    return _decode_json(_read_regular_bytes(source, role), role)


def _load_immutable_json(path: Path, role: str) -> dict[str, Any]:
    source = _require_read_only_file(Path(path), role)
    data = _read_regular_bytes(source, role)
    value = _decode_json(data, role)
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ExtremeScoreLeagueError(f"{role} must use canonical JSON encoding")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    """Load a canonical read-only plan and validate every live binding."""

    return validate_plan(_load_immutable_json(Path(path), "league plan"))


def _publish_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    data = (canonical_json(value) + "\n").encode("utf-8")
    _publish_immutable_bytes(Path(path), data, "JSON artifact")


def _worker(plan: Mapping[str, Any], worker_id: str) -> dict[str, Any]:
    requested = _text(worker_id, "worker_id")
    matches = [
        worker for worker in plan["workers"] if worker.get("worker_id") == requested
    ]
    if len(matches) != 1:
        raise ExtremeScoreLeagueError(f"worker {requested!r} is absent from the plan")
    return dict(matches[0])


def _artifact_manifest(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": plan["policy"],
        "accepted_state": plan["accepted_state"],
        "launcher": plan["launcher"],
        "binary": plan["binary"],
        "config": plan["config"],
        "focal_model": plan["focal_model"],
        "opponents": [
            {
                "role": opponent["role"],
                "weight": opponent["weight"],
                "model": opponent["model"],
            }
            for opponent in plan["opponents"]
        ],
    }


def _verify_bound_file(
    binding: Mapping[str, Any],
    role: str,
    *,
    probe: Callable[[Path], str],
) -> None:
    path = _absolute_path(binding["path"], f"{role} path")
    expected = _sha256(binding["sha256"], f"{role} SHA-256")
    source = _regular_file(path, role)
    if probe(source) != expected:
        raise ExtremeScoreLeagueError(f"{role} changed after plan publication")


def _verify_execution_artifacts(
    plan: Mapping[str, Any],
    *,
    probe: Callable[[Path], str],
) -> dict[str, Any]:
    _verify_bound_file(plan["launcher"], "league launcher", probe=probe)
    _verify_bound_file(plan["binary"], "KataGo binary", probe=probe)
    _verify_bound_file(plan["config"], "self-play config", probe=probe)

    policy = plan["policy"]
    policy_path = _absolute_path(policy["path"], "training policy path")
    policy_source = _regular_file(policy_path, "extreme-score training policy")
    if probe(policy_source) != policy["file_sha256"]:
        raise ExtremeScoreLeagueError(
            "extreme-score training policy changed after plan publication"
        )
    accepted_binding, accepted_state = _accepted_state_binding(plan["accepted_state"])
    if accepted_binding != plan["accepted_state"]:
        raise ExtremeScoreLeagueError(
            "accepted model state changed after plan publication"
        )
    if accepted_state["model"]["sha256"] != plan["focal_model"]["model_sha256"]:
        raise ExtremeScoreLeagueError(
            "accepted model state no longer matches focal snapshot"
        )

    focal = plan["focal_model"]
    _verify_snapshot_directory(
        Path(focal["directory"]),
        focal["model_sha256"],
        "focal",
        probe=probe,
    )
    for opponent in plan["opponents"]:
        model = opponent["model"]
        _verify_snapshot_directory(
            Path(model["directory"]),
            model["model_sha256"],
            f"opponent {opponent['role']}",
            probe=probe,
        )
    return json.loads(canonical_json(_artifact_manifest(plan)))


def _walk_output_files(
    output: Path,
    *,
    freeze_files: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        _regular_directory(directory, "worker output directory")
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            child_stat = child.lstat()
            if child.is_symlink():
                raise ExtremeScoreLeagueError(
                    f"worker output contains a symlink: {child}"
                )
            if stat_module.S_ISDIR(child_stat.st_mode):
                visit(child)
                continue
            if not stat_module.S_ISREG(child_stat.st_mode):
                raise ExtremeScoreLeagueError(
                    f"worker output contains a non-regular artifact: {child}"
                )
            if child == output / _RECEIPT_NAME:
                continue
            if freeze_files:
                os.chmod(child, child_stat.st_mode & ~_WRITE_BITS)
                child_stat = child.lstat()
            if not _is_read_only(child_stat.st_mode):
                raise ExtremeScoreLeagueError(f"output shard is not read-only: {child}")
            records.append(
                {
                    "relative_path": child.relative_to(output).as_posix(),
                    "sha256": file_sha256(child),
                    "size_bytes": child_stat.st_size,
                }
            )

    visit(output)
    return records


def _freeze_output_directories(output: Path) -> None:
    directories: list[Path] = []

    def collect(directory: Path) -> None:
        _regular_directory(directory, "worker output directory")
        directories.append(directory)
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if child.is_symlink():
                raise ExtremeScoreLeagueError(
                    f"worker output contains a symlink: {child}"
                )
            if stat_module.S_ISDIR(child.lstat().st_mode):
                collect(child)

    collect(output)
    for directory in reversed(directories):
        os.chmod(directory, directory.lstat().st_mode & ~_WRITE_BITS)


def _verify_frozen_output_directories(output: Path) -> None:
    def visit(directory: Path) -> None:
        checked = _regular_directory(directory, "worker output directory")
        if not _is_read_only(checked.lstat().st_mode):
            raise ExtremeScoreLeagueError(
                f"worker output directory is not read-only: {directory}"
            )
        for child in directory.iterdir():
            if child.is_symlink():
                raise ExtremeScoreLeagueError(
                    f"worker output contains a symlink: {child}"
                )
            if stat_module.S_ISDIR(child.lstat().st_mode):
                visit(child)

    visit(output)


def _receipt_identity(
    plan: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract": WORKER_RECEIPT_CONTRACT,
        "execution_contract": WORKER_EXECUTION_CONTRACT,
        "plan_sha256": plan["plan_sha256"],
        "worker_id": worker["worker_id"],
        "worker_sha256": canonical_sha256(worker),
        "policy": plan["policy"],
        "curriculum_state": plan["curriculum_state"],
        "artifact_bindings": _artifact_manifest(plan),
        "katago_argv_sha256": worker["katago_argv_sha256"],
        "output_directory": worker["output_directory"],
    }


def _load_worker_receipt(
    path: Path,
    plan: Mapping[str, Any],
    worker: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_immutable_json(path, "worker execution receipt")
    expected_keys = {
        *_receipt_identity(plan, worker).keys(),
        "process_outcome",
        "artifact_verification",
        "output_shards",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ExtremeScoreLeagueError("worker execution receipt keys differ")
    if receipt["contract"] != WORKER_RECEIPT_CONTRACT:
        raise ExtremeScoreLeagueError("worker execution receipt contract is invalid")
    supplied = _sha256(receipt["receipt_sha256"], "receipt SHA-256")
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    if supplied != canonical_sha256(payload):
        raise ExtremeScoreLeagueError("worker execution receipt self-hash is invalid")
    identity = _receipt_identity(plan, worker)
    if any(receipt[key] != value for key, value in identity.items()):
        raise ExtremeScoreLeagueError(
            "worker execution receipt contradicts the league plan"
        )
    outcome = receipt["process_outcome"]
    if not isinstance(outcome, Mapping) or set(outcome) != {
        "status",
        "returncode",
        "error_type",
        "error_message",
    }:
        raise ExtremeScoreLeagueError("worker process outcome is malformed")
    returncode = outcome["returncode"]
    outcome_status = outcome["status"]
    valid_outcome = (
        (
            outcome_status == "succeeded"
            and type(returncode) is int
            and returncode == 0
            and outcome["error_type"] is None
            and outcome["error_message"] is None
        )
        or (
            outcome_status == "failed"
            and type(returncode) is int
            and returncode != 0
            and outcome["error_type"] is None
            and outcome["error_message"] is None
        )
        or (
            outcome_status == "launch_error"
            and returncode is None
            and isinstance(outcome["error_type"], str)
            and isinstance(outcome["error_message"], str)
        )
    )
    if not valid_outcome:
        raise ExtremeScoreLeagueError("worker process outcome is inconsistent")
    verification = receipt["artifact_verification"]
    if not isinstance(verification, Mapping) or set(verification) != {
        "before_exec_manifest_sha256",
        "after_exec_manifest_sha256",
        "artifacts_unchanged",
        "error_type",
        "error_message",
    }:
        raise ExtremeScoreLeagueError("worker artifact verification is malformed")
    expected_manifest_hash = canonical_sha256(identity["artifact_bindings"])
    unchanged = verification["artifacts_unchanged"]
    valid_verification_outcome = (
        unchanged
        and verification["after_exec_manifest_sha256"] == expected_manifest_hash
        and verification["error_type"] is None
        and verification["error_message"] is None
    ) or (
        not unchanged
        and verification["after_exec_manifest_sha256"] is None
        and isinstance(verification["error_type"], str)
        and isinstance(verification["error_message"], str)
    )
    if (
        verification["before_exec_manifest_sha256"] != expected_manifest_hash
        or not isinstance(unchanged, bool)
        or not valid_verification_outcome
    ):
        raise ExtremeScoreLeagueError(
            "worker artifact verification contradicts the plan"
        )
    output = Path(worker["output_directory"])
    actual_shards = _walk_output_files(output, freeze_files=False)
    if receipt["output_shards"] != actual_shards:
        raise ExtremeScoreLeagueError(
            "worker output shards contradict the immutable receipt"
        )
    _verify_frozen_output_directories(output)
    return receipt


def _ensure_output_parent(output_root: Path, output: Path) -> None:
    root = _regular_directory(output_root, "output_root")
    try:
        relative_parent = output.parent.relative_to(root)
    except ValueError as exc:
        raise ExtremeScoreLeagueError(
            "worker output directory escapes output_root"
        ) from exc
    current = root
    for part in relative_parent.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            _regular_directory(current, "worker output parent")
        else:
            # Multiple workers in one generation create the same parent
            # concurrently. Accept an exact directory-creation race, then
            # validate the resulting object before using it.
            current.mkdir(exist_ok=True)
            _regular_directory(current, "worker output parent")


def _prepare_output_directory(output_root: Path, output: Path) -> None:
    _ensure_output_parent(output_root, output)
    if output.exists() or output.is_symlink():
        directory = _regular_directory(output, "worker output directory")
        if any(directory.iterdir()):
            raise ExtremeScoreLeagueError(
                "worker output exists without an immutable receipt"
            )
        if _is_read_only(directory.lstat().st_mode):
            raise ExtremeScoreLeagueError("worker output directory is not writable")
        return
    output.mkdir(parents=True)
    _regular_directory(output, "worker output directory")


def _default_executor(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(list(argv), check=False)


def _executor_returncode(result: Any) -> int:
    value = result if type(result) is int else getattr(result, "returncode", None)
    if type(value) is not int:
        raise ExtremeScoreLeagueError(
            "worker executor must return an integer or result with integer returncode"
        )
    return value


def execute_worker(
    plan: Mapping[str, Any],
    worker_id: str,
    *,
    executor: Callable[[Sequence[str]], Any] = _default_executor,
    probe: Callable[[Path], str] = file_sha256,
) -> dict[str, Any]:
    """Verify all bindings, execute one worker, and freeze its receipt and shards."""

    checked = validate_plan(plan)
    worker = _worker(checked, worker_id)
    output = Path(worker["output_directory"])
    _ensure_output_parent(Path(checked["output_root"]), output)
    if output.exists() or output.is_symlink():
        _regular_directory(output, "worker output directory")
        receipt_path = output / _RECEIPT_NAME
        if receipt_path.exists() or receipt_path.is_symlink():
            return _load_worker_receipt(receipt_path, checked, worker)
    _prepare_output_directory(Path(checked["output_root"]), output)

    # This is deliberately the final operation before process creation.
    before_manifest = _verify_execution_artifacts(checked, probe=probe)
    execution_error: Exception | None = None
    returncode: int | None = None
    try:
        returncode = _executor_returncode(executor(tuple(worker["katago_argv"])))
    except Exception as exc:  # noqa: BLE001 - receipt must preserve arbitrary launcher failures
        execution_error = exc

    output_shards = _walk_output_files(output, freeze_files=True)
    after_manifest: dict[str, Any] | None = None
    verification_error: Exception | None = None
    try:
        after_manifest = _verify_execution_artifacts(checked, probe=probe)
    except Exception as exc:  # noqa: BLE001 - receipt must preserve arbitrary probe failures
        verification_error = exc

    process_outcome = {
        "status": (
            "launch_error"
            if execution_error is not None
            else "succeeded"
            if returncode == 0
            else "failed"
        ),
        "returncode": returncode,
        "error_type": (
            type(execution_error).__name__ if execution_error is not None else None
        ),
        "error_message": str(execution_error) if execution_error is not None else None,
    }
    artifact_verification = {
        "before_exec_manifest_sha256": canonical_sha256(before_manifest),
        "after_exec_manifest_sha256": (
            canonical_sha256(after_manifest) if after_manifest is not None else None
        ),
        "artifacts_unchanged": verification_error is None
        and after_manifest == before_manifest,
        "error_type": (
            type(verification_error).__name__
            if verification_error is not None
            else None
        ),
        "error_message": (
            str(verification_error) if verification_error is not None else None
        ),
    }
    receipt = {
        **_receipt_identity(checked, worker),
        "process_outcome": process_outcome,
        "artifact_verification": artifact_verification,
        "output_shards": output_shards,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _publish_immutable_json(output / _RECEIPT_NAME, receipt)
    _freeze_output_directories(output)

    if execution_error is not None:
        raise ExtremeScoreLeagueError(
            "worker process could not be launched; immutable failure receipt written"
        ) from execution_error
    if verification_error is not None or after_manifest != before_manifest:
        raise ExtremeScoreLeagueError(
            "worker artifacts changed during execution; "
            "immutable failure receipt written"
        ) from verification_error
    return receipt


def load_worker_receipt(plan: Mapping[str, Any], worker_id: str) -> dict[str, Any]:
    """Validate one completed worker receipt without launching a process."""
    checked = validate_plan(plan)
    worker = _worker(checked, worker_id)
    receipt_path = Path(worker["output_directory"]) / _RECEIPT_NAME
    return _load_worker_receipt(receipt_path, checked, worker)


def status(plan: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_plan(plan)
    observations = []
    for worker in checked["workers"]:
        output = Path(worker["output_directory"])
        output_exists = output.is_dir() and not output.is_symlink()
        receipt_path = output / _RECEIPT_NAME
        if receipt_path.exists() or receipt_path.is_symlink():
            receipt = _load_worker_receipt(receipt_path, checked, worker)
            process_status = receipt["process_outcome"]["status"]
            state = "SUCCEEDED" if process_status == "succeeded" else "FAILED"
            receipt_sha256 = receipt["receipt_sha256"]
        else:
            process_status = None
            state = "RUNNING" if output_exists else "PLANNED"
            receipt_sha256 = None
        observations.append(
            {
                "worker_id": worker["worker_id"],
                "state": state,
                "output_exists": output_exists,
                "receipt_sha256": receipt_sha256,
                "process_status": process_status,
            }
        )
    value = {
        "schema_version": 1,
        "contract": STATUS_CONTRACT,
        "plan_sha256": checked["plan_sha256"],
        "workers": observations,
    }
    value["status_sha256"] = canonical_sha256(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--request", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    inspect = subparsers.add_parser("status")
    inspect.add_argument("--plan", required=True, type=Path)
    run = subparsers.add_parser(
        "run-worker",
        help="hash-check and execute one worker from an immutable plan",
    )
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--worker-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            value = build_plan(
                _load_json(args.request, "league request"),
                policy_path=args.policy,
            )
            _publish_immutable_json(Path(args.output), value)
            returncode = 0
        elif args.command == "status":
            value = status(load_plan(args.plan))
            returncode = 0
        else:
            value = execute_worker(load_plan(args.plan), args.worker_id)
            worker_returncode = value["process_outcome"]["returncode"]
            returncode = (
                worker_returncode
                if type(worker_returncode) is int and 0 <= worker_returncode <= 255
                else 1
            )
        print(canonical_json(value))
        return returncode
    except (OSError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
