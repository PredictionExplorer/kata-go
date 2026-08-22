"""Production ``katago match`` adapter for held-out expected-max evaluation.

The adapter is callable with the runner protocol used by
``extreme_score_evaluator.evaluate_with_runner``.  It expands immutable
evaluator jobs into one deterministic empty-board match cell per frozen
opponent and focal color, validates all content bindings, and publishes
restart-safe receipts.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_score.extreme_score_evaluator import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)

RUNNER_SPEC_CONTRACT = "risk-score-extreme-score-match-runner-spec-v1"
RUNNER_RECEIPT_CONTRACT = "risk-score-extreme-score-match-runner-receipt-v1"
PAIRED_PLAN_RECEIPT_CONTRACT = "risk-score-extreme-score-match-paired-plan-receipt-v1"
CELL_RECEIPT_CONTRACT = "risk-score-extreme-score-match-cell-receipt-v1"
ARM_RECEIPT_CONTRACT = "risk-score-extreme-score-match-arm-receipt-v1"
EXECUTION_PROVENANCE_CONTRACT = "risk-score-extreme-score-match-execution-provenance-v1"
SCHEDULE_CONTRACT = "risk-score-extreme-score-empty-board-schedule-v1"
GPU_EXECUTION_CONTRACT = "risk-score-extreme-score-gpu-execution-v1"

_ARMS = ("candidate", "reference")
_COLORS = ("B", "W")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_IMMUTABLE_FILE_MODE = 0o444
_GPU_PROBE_TIMEOUT_SECONDS = 10.0
_JOB_KEYS = {
    "schema_version",
    "plan_sha256",
    "arm",
    "model_sha256",
    "cohort_id",
    "cohort_sha256",
    "trial_index",
    "seed",
    "config_sha256",
    "league_cell",
    "opponent_snapshot_id",
    "opponent_model_sha256",
    "focal_color",
}
_PAIRED_JOB_KEYS = tuple(sorted(_JOB_KEYS.difference({"arm", "model_sha256"})))
_EMPTY_BOARD = "/".join(["." * 19] * 19)
EMPTY_BOARD_POSITION = {
    "xSize": 19,
    "ySize": 19,
    "board": _EMPTY_BOARD,
    "nextPla": "B",
    "moveLocs": [],
    "movePlas": [],
    "initialTurnNumber": 0,
    "hintLoc": "null",
}
FOCAL_BOT_NAME = "focal"
OPPONENT_BOT_NAME = "frozen-opponent"


class ExtremeScoreMatchRunnerError(RuntimeError):
    """A production match invocation or artifact is invalid."""


class ExtremeScoreMatchRunnerValidationError(ExtremeScoreMatchRunnerError):
    """An input or KataGo result violates the bound runner contract."""


class ExtremeScoreMatchRunnerConflictError(ExtremeScoreMatchRunnerError):
    """An immutable execution artifact contradicts the requested cell."""


@dataclass(frozen=True)
class GpuIdentityObservation:
    """One physical GPU identity observed for a configured ordinal."""

    gpu_index: int
    gpu_uuid: str
    provenance: str


@dataclass(frozen=True)
class ExtremeScoreMatchRunnerSpec:
    """Files and execution coordinates that define one production runner."""

    katago_binary: Path
    focal_models: Mapping[str, Path]
    opponent_models: Mapping[str, Path]
    match_config: Path
    output_root: Path
    topology: str
    process_count: int
    expected_gpu_uuid: str
    gpu_lease_provenance: str
    gpu_index: int = 7

    def __post_init__(self) -> None:
        if not isinstance(self.focal_models, Mapping) or not self.focal_models:
            raise ValueError("focal_models must be a nonempty SHA-256-to-path map")
        if not isinstance(self.opponent_models, Mapping) or not self.opponent_models:
            raise ValueError("opponent_models must be a nonempty SHA-256-to-path map")
        _single_line_text(self.topology, "topology")
        if type(self.process_count) is not int or self.process_count <= 0:
            raise ValueError("process_count must be a positive integer")
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError("gpu_index must be a nonnegative integer")
        if (
            not isinstance(self.expected_gpu_uuid, str)
            or _GPU_UUID_RE.fullmatch(self.expected_gpu_uuid) is None
        ):
            raise ValueError("expected_gpu_uuid must be a physical NVIDIA GPU UUID")
        _single_line_text(self.gpu_lease_provenance, "gpu_lease_provenance")

    @property
    def config_path(self) -> Path:
        return Path(self.match_config)

    @property
    def focal_model_paths(self) -> Mapping[str, Path]:
        return self.focal_models

    @property
    def frozen_opponent_paths(self) -> Mapping[str, Path]:
        return self.opponent_models


# Short aliases for callers that prefer generic runner naming.
MatchRunnerSpec = ExtremeScoreMatchRunnerSpec


@dataclass(frozen=True)
class MatchCell:
    """One arm/opponent/color process with a single expected-max objective."""

    cell_id: str
    arm: str
    plan_sha256: str
    paired_plan_sha256: str
    focal_model_sha256: str
    opponent_model_sha256: str
    focal_color: str
    group_size: int
    jobs: tuple[dict[str, Any], ...]
    schedule_rows: tuple[dict[str, Any], ...]

    @property
    def schedule_id(self) -> str:
        return str(self.schedule_rows[0]["scheduleId"])


@dataclass(frozen=True)
class _ValidatedJobPlan:
    jobs: tuple[dict[str, Any], ...]
    paired_jobs: tuple[dict[str, Any], ...]
    arm: str
    plan_sha256: str
    model_sha256: str
    config_sha256: str
    group_size: int
    paired_plan_sha256: str


def _single_line_text(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{role} must be a nonempty trimmed single-line string")
    return value


def _sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role} must be a lowercase 64-character SHA-256"
        )
    return value


def _finite_number(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtremeScoreMatchRunnerValidationError(f"{role} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ExtremeScoreMatchRunnerValidationError(f"{role} must be finite numeric")
    return number


def _boolean(value: Any, role: str) -> bool:
    if not isinstance(value, bool):
        raise ExtremeScoreMatchRunnerValidationError(f"{role} must be boolean")
    return value


def _json_clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _coerce_gpu_identity_observation(value: Any, role: str) -> GpuIdentityObservation:
    if isinstance(value, GpuIdentityObservation):
        observation = value
    elif isinstance(value, Mapping) and set(value) == {
        "gpu_index",
        "gpu_uuid",
        "provenance",
    }:
        observation = GpuIdentityObservation(
            gpu_index=value["gpu_index"],
            gpu_uuid=value["gpu_uuid"],
            provenance=value["provenance"],
        )
    else:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role} must contain gpu_index, gpu_uuid, and provenance"
        )
    if type(observation.gpu_index) is not int or observation.gpu_index < 0:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role}.gpu_index must be a nonnegative integer"
        )
    if (
        not isinstance(observation.gpu_uuid, str)
        or _GPU_UUID_RE.fullmatch(observation.gpu_uuid) is None
    ):
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role}.gpu_uuid must be a physical NVIDIA GPU UUID"
        )
    try:
        _single_line_text(observation.provenance, f"{role}.provenance")
    except ValueError as exc:
        raise ExtremeScoreMatchRunnerValidationError(str(exc)) from exc
    return observation


def _nvidia_smi_gpu_identity_probe(gpu_index: int) -> GpuIdentityObservation:
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=_GPU_PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise ExtremeScoreMatchRunnerError(
            "cannot inventory physical GPUs with nvidia-smi: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if getattr(completed, "returncode", None) != 0:
        stderr = getattr(completed, "stderr", "")
        detail = f": {stderr[:1000]}" if isinstance(stderr, str) and stderr else ""
        raise ExtremeScoreMatchRunnerError(
            f"nvidia-smi GPU identity probe failed{detail}"
        )
    stdout = getattr(completed, "stdout", None)
    if not isinstance(stdout, str):
        raise ExtremeScoreMatchRunnerError(
            "nvidia-smi GPU identity probe returned non-text output"
        )
    by_index: dict[int, str] = {}
    seen_uuids: set[str] = set()
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",", 1)]
        try:
            index = int(fields[0]) if len(fields) == 2 else -1
        except ValueError:
            index = -1
        gpu_uuid = fields[1] if len(fields) == 2 else ""
        if (
            index < 0
            or index in by_index
            or _GPU_UUID_RE.fullmatch(gpu_uuid) is None
            or gpu_uuid in seen_uuids
        ):
            raise ExtremeScoreMatchRunnerValidationError(
                "nvidia-smi physical GPU identity inventory is malformed"
            )
        by_index[index] = gpu_uuid
        seen_uuids.add(gpu_uuid)
    observed_uuid = by_index.get(gpu_index)
    if observed_uuid is None:
        raise ExtremeScoreMatchRunnerValidationError(
            f"configured GPU ordinal {gpu_index} is absent from nvidia-smi inventory"
        )
    return GpuIdentityObservation(
        gpu_index=gpu_index,
        gpu_uuid=observed_uuid,
        provenance=" ".join(command),
    )


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _normalized_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    return source.resolve()


def _normalized_immutable_file(path: Path, role: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ExtremeScoreMatchRunnerConflictError(
            f"{role} is missing or inaccessible: {source}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ExtremeScoreMatchRunnerConflictError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != _IMMUTABLE_FILE_MODE:
        raise ExtremeScoreMatchRunnerConflictError(
            f"{role} is not immutable read-only mode "
            f"{_IMMUTABLE_FILE_MODE:04o}: {source} has {mode:04o}"
        )
    return source.resolve()


def _validate_existing_immutable_bytes(path: Path, data: bytes, role: str) -> None:
    source = _normalized_immutable_file(path, role)
    try:
        observed = source.read_bytes()
    except OSError as exc:
        raise ExtremeScoreMatchRunnerConflictError(
            f"{role} cannot be read: {source}"
        ) from exc
    expected_hash = hashlib.sha256(data).hexdigest()
    observed_hash = hashlib.sha256(observed).hexdigest()
    if observed_hash != expected_hash or observed != data:
        raise ExtremeScoreMatchRunnerConflictError(
            f"{role} content/hash contradicts {source}: "
            f"{observed_hash}, expected {expected_hash}"
        )


def _normalize_model_map(
    value: Mapping[str, Path], role: str
) -> dict[str, tuple[Path, str]]:
    normalized: dict[str, tuple[Path, str]] = {}
    for supplied_hash, supplied_path in value.items():
        digest = _sha256(supplied_hash, f"{role} map key")
        path = _normalized_file(Path(supplied_path), f"{role} {digest}")
        actual = file_sha256(path)
        if actual != digest:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} SHA-256 mismatch for {path}: {actual}, expected {digest}"
            )
        normalized[digest] = (path, actual)
    return dict(sorted(normalized.items()))


def _decode_json_line(raw: str, role: str) -> dict[str, Any]:
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExtremeScoreMatchRunnerValidationError(
                    f"{role} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role} contains non-finite JSON value {value}"
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ExtremeScoreMatchRunnerValidationError(
            f"cannot decode {role}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ExtremeScoreMatchRunnerValidationError(f"{role} must be an object")
    return value


def _load_jsonl(path: Path, role: str) -> tuple[dict[str, Any], ...]:
    source = _normalized_file(path, role)
    rows: list[dict[str, Any]] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role} is not UTF-8: {source}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            rows.append(_decode_json_line(line, f"{role} line {line_number}"))
    if not rows:
        raise ExtremeScoreMatchRunnerValidationError(f"{role} is empty: {source}")
    return tuple(rows)


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable_bytes(path: Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ExtremeScoreMatchRunnerConflictError(
            f"artifact parent is unsafe: {target.parent}"
        )
    if _path_entry_exists(target):
        _validate_existing_immutable_bytes(target, data, "existing immutable artifact")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), _IMMUTABLE_FILE_MODE)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            _validate_existing_immutable_bytes(
                target, data, "concurrent immutable artifact"
            )
        else:
            _validate_existing_immutable_bytes(
                target, data, "published immutable artifact"
            )
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _self_hashed_artifact(contract: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract": contract,
        **_json_clone(payload),
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _load_receipt(path: Path, contract: str) -> dict[str, Any]:
    source = _normalized_immutable_file(path, "execution receipt")
    data = source.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtremeScoreMatchRunnerConflictError(
            f"execution receipt is not UTF-8: {source}"
        ) from exc
    try:
        value = _decode_json_line(text.rstrip("\n"), "execution receipt")
    except ExtremeScoreMatchRunnerValidationError as exc:
        raise ExtremeScoreMatchRunnerConflictError(str(exc)) from exc
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ExtremeScoreMatchRunnerConflictError(
            f"execution receipt is not canonical: {source}"
        )
    if value.get("schema_version") != 1 or value.get("contract") != contract:
        raise ExtremeScoreMatchRunnerConflictError(
            f"execution receipt has the wrong contract: {source}"
        )
    supplied_hash = value.get("receipt_sha256")
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    if supplied_hash != canonical_sha256(payload):
        raise ExtremeScoreMatchRunnerConflictError(
            f"execution receipt self-hash is invalid: {source}"
        )
    return value


def _artifact_record(path: str, data: bytes, rows: int) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "rows": rows,
    }


def _verify_artifact(
    directory: Path,
    binding: Any,
    *,
    expected_path: str,
    expected_rows: int,
    role: str,
) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "file_sha256",
        "rows",
    }:
        raise ExtremeScoreMatchRunnerConflictError(f"{role} binding is malformed")
    if binding["path"] != expected_path or binding["rows"] != expected_rows:
        raise ExtremeScoreMatchRunnerConflictError(f"{role} binding contradicts cell")
    expected_hash = _sha256(binding["file_sha256"], f"{role} file SHA-256")
    path = directory / expected_path
    try:
        actual_hash = file_sha256(_normalized_immutable_file(path, role))
    except (
        ExtremeScoreMatchRunnerValidationError,
        ExtremeScoreMatchRunnerConflictError,
    ) as exc:
        raise ExtremeScoreMatchRunnerConflictError(str(exc)) from exc
    if actual_hash != expected_hash:
        raise ExtremeScoreMatchRunnerConflictError(f"{role} hash changed: {path}")
    return path


def _paired_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in _PAIRED_JOB_KEYS}


def _validate_jobs(
    arm: str,
    jobs: Iterable[Mapping[str, Any]],
    *,
    focal_model_hashes: set[str] | None = None,
    opponent_model_hashes: set[str] | None = None,
    config_sha256: str | None = None,
) -> _ValidatedJobPlan:
    if arm not in _ARMS:
        raise ExtremeScoreMatchRunnerValidationError(
            "runner arm must be candidate or reference"
        )
    checked: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[str, int]] = set()
    seen_seeds: set[str] = set()
    cohort_fields: dict[str, tuple[Any, ...]] = {}
    trials_by_cohort: dict[str, set[int]] = defaultdict(set)

    for index, input_job in enumerate(jobs):
        role = f"{arm} job {index}"
        if not isinstance(input_job, Mapping) or set(input_job) != _JOB_KEYS:
            actual = set(input_job) if isinstance(input_job, Mapping) else set()
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} keys differ; missing={sorted(_JOB_KEYS - actual)}, "
                f"unexpected={sorted(actual - _JOB_KEYS)}"
            )
        job = _json_clone(input_job)
        if job["schema_version"] != 1 or job["arm"] != arm:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} schema or arm identity is invalid"
            )
        for key in (
            "plan_sha256",
            "model_sha256",
            "cohort_sha256",
            "config_sha256",
            "opponent_model_sha256",
        ):
            _sha256(job[key], f"{role}.{key}")
        for key in (
            "cohort_id",
            "seed",
            "league_cell",
            "opponent_snapshot_id",
        ):
            try:
                _single_line_text(job[key], f"{role}.{key}")
            except ValueError as exc:
                raise ExtremeScoreMatchRunnerValidationError(str(exc)) from exc
        if job["focal_color"] not in _COLORS:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role}.focal_color must be B or W"
            )
        trial_index = job["trial_index"]
        if type(trial_index) is not int or trial_index < 0:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role}.trial_index must be a nonnegative integer"
            )
        coordinate = (job["cohort_id"], trial_index)
        if coordinate in seen_coordinates:
            raise ExtremeScoreMatchRunnerValidationError(
                f"duplicate runner coordinate {coordinate!r}"
            )
        if job["seed"] in seen_seeds:
            raise ExtremeScoreMatchRunnerValidationError(
                f"duplicate runner seed {job['seed']!r}"
            )
        seen_coordinates.add(coordinate)
        seen_seeds.add(job["seed"])
        trials_by_cohort[job["cohort_id"]].add(trial_index)
        identity = (
            job["cohort_sha256"],
            job["config_sha256"],
            job["league_cell"],
            job["opponent_snapshot_id"],
            job["opponent_model_sha256"],
            job["focal_color"],
        )
        previous = cohort_fields.setdefault(job["cohort_id"], identity)
        if previous != identity:
            raise ExtremeScoreMatchRunnerValidationError(
                f"cohort {job['cohort_id']!r} changes immutable cell identity"
            )
        checked.append(job)

    if not checked:
        raise ExtremeScoreMatchRunnerValidationError(f"{arm} jobs must not be empty")
    singleton_fields = {
        key: {job[key] for job in checked}
        for key in ("plan_sha256", "model_sha256", "config_sha256")
    }
    for key, values in singleton_fields.items():
        if len(values) != 1:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{arm} jobs use multiple {key} values"
            )
    model_hash = next(iter(singleton_fields["model_sha256"]))
    actual_config_hash = next(iter(singleton_fields["config_sha256"]))
    if focal_model_hashes is not None and model_hash not in focal_model_hashes:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{arm} focal model SHA-256 is not in the runner map"
        )
    if config_sha256 is not None and actual_config_hash != config_sha256:
        raise ExtremeScoreMatchRunnerValidationError(
            "runner jobs do not bind the configured match file SHA-256"
        )
    if opponent_model_hashes is not None:
        missing_opponents = sorted(
            {
                job["opponent_model_sha256"]
                for job in checked
                if job["opponent_model_sha256"] not in opponent_model_hashes
            }
        )
        if missing_opponents:
            raise ExtremeScoreMatchRunnerValidationError(
                f"frozen opponent hashes are absent from the runner map: "
                f"{missing_opponents}"
            )

    sizes = {len(indices) for indices in trials_by_cohort.values()}
    if len(sizes) != 1:
        raise ExtremeScoreMatchRunnerValidationError(
            "all precommitted cohorts must use one fixed N"
        )
    group_size = next(iter(sizes))
    if not 1 <= group_size <= 64:
        raise ExtremeScoreMatchRunnerValidationError(
            "expected-max group size must be between 1 and 64"
        )
    expected_trials = set(range(group_size))
    for cohort_id, indices in trials_by_cohort.items():
        if indices != expected_trials:
            raise ExtremeScoreMatchRunnerValidationError(
                f"cohort {cohort_id!r} must contain trial indices "
                f"0 through {group_size - 1}"
            )

    paired_jobs = tuple(_paired_job(job) for job in checked)
    return _ValidatedJobPlan(
        jobs=tuple(checked),
        paired_jobs=paired_jobs,
        arm=arm,
        plan_sha256=next(iter(singleton_fields["plan_sha256"])),
        model_sha256=model_hash,
        config_sha256=actual_config_hash,
        group_size=group_size,
        paired_plan_sha256=canonical_sha256(paired_jobs),
    )


def group_jobs_by_opponent_and_color(
    jobs: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[dict[str, Any], ...]]:
    """Group already-precommitted jobs without changing their relative order."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for input_job in jobs:
        if not isinstance(input_job, Mapping):
            raise ExtremeScoreMatchRunnerValidationError("runner job must be an object")
        opponent_hash = _sha256(
            input_job.get("opponent_model_sha256"), "opponent model SHA-256"
        )
        color = input_job.get("focal_color")
        if color not in _COLORS:
            raise ExtremeScoreMatchRunnerValidationError(
                "runner job focal_color must be B or W"
            )
        grouped.setdefault((opponent_hash, color), []).append(_json_clone(input_job))
    return {key: tuple(value) for key, value in grouped.items()}


def build_empty_board_schedule(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Map one opponent/color cell to deterministic KataGo schedule rows."""

    if not jobs:
        raise ExtremeScoreMatchRunnerValidationError(
            "cannot build a schedule without jobs"
        )
    opponent_hashes = {job.get("opponent_model_sha256") for job in jobs}
    colors = {job.get("focal_color") for job in jobs}
    plan_hashes = {job.get("plan_sha256") for job in jobs}
    if len(opponent_hashes) != 1 or len(colors) != 1 or len(plan_hashes) != 1:
        raise ExtremeScoreMatchRunnerValidationError(
            "one schedule must contain one plan, opponent, and focal color"
        )
    opponent_hash = _sha256(next(iter(opponent_hashes)), "opponent model SHA-256")
    color = next(iter(colors))
    if color not in _COLORS:
        raise ExtremeScoreMatchRunnerValidationError("focal color must be B or W")
    plan_hash = _sha256(next(iter(plan_hashes)), "plan SHA-256")
    paired_jobs = [_paired_job(job) for job in jobs]
    schedule_digest = canonical_sha256(
        {
            "contract": SCHEDULE_CONTRACT,
            "plan_sha256": plan_hash,
            "opponent_model_sha256": opponent_hash,
            "focal_color": color,
            "jobs": paired_jobs,
        }
    )
    schedule_id = f"risk-score-extreme-score-{schedule_digest}"
    position_id = f"empty-19x19-{canonical_sha256(EMPTY_BOARD_POSITION)}"
    rows: list[dict[str, Any]] = []
    game_ids: set[str] = set()
    for job, paired_job in zip(jobs, paired_jobs):
        game_id = f"expected-max-job-{canonical_sha256(paired_job)}"
        if game_id in game_ids:
            raise ExtremeScoreMatchRunnerValidationError(
                f"duplicate deterministic game identity {game_id}"
            )
        game_ids.add(game_id)
        rows.append(
            {
                "schemaVersion": 1,
                "generatorContract": SCHEDULE_CONTRACT,
                "scheduleId": schedule_id,
                "gameId": game_id,
                "pairId": f"expected-max-cohort-{job['cohort_sha256']}",
                "positionId": position_id,
                "seed": job["seed"],
                "blackBot": 0 if color == "B" else 1,
                "whiteBot": 1 if color == "B" else 0,
                "startPosition": _json_clone(EMPTY_BOARD_POSITION),
                "job": _json_clone(job),
                "pairedJobSha256": canonical_sha256(paired_job),
            }
        )
    return tuple(rows)


def _override_value(value: str | Path, role: str) -> str:
    text = str(value)
    if not text or any(character in text for character in ("\x00", "\n", "\r", ",")):
        raise ValueError(
            f"{role} cannot be empty or contain comma, NUL, or newline in overrides"
        )
    return text


def build_match_command(
    katago_binary: Path,
    config_path: Path,
    focal_model_path: Path,
    opponent_model_path: Path,
    schedule_path: Path,
    result_path: Path,
    *,
    focal_color: str,
    group_size: int,
    game_count: int,
) -> tuple[str, ...]:
    """Build a shell-free argv tuple for one expected-max match cell."""

    if focal_color not in _COLORS:
        raise ValueError("focal_color must be B or W")
    if type(group_size) is not int or not 1 <= group_size <= 64:
        raise ValueError("group_size must be an integer from 1 through 64")
    if type(game_count) is not int or game_count <= 0:
        raise ValueError("game_count must be a positive integer")
    overrides = [
        "numBots=2",
        f"botName0={FOCAL_BOT_NAME}",
        f"botName1={OPPONENT_BOT_NAME}",
        f"nnModelFile0={_override_value(focal_model_path, 'focal model path')}",
        f"nnModelFile1={_override_value(opponent_model_path, 'opponent model path')}",
        (
            "deterministicScheduleFile="
            + _override_value(schedule_path, "schedule path")
        ),
        f"matchResultJsonlFile={_override_value(result_path, 'result path')}",
        "matchMoveJsonlFile=",
        f"numGamesTotal={game_count}",
        "numGameThreads=1",
        "numSearchThreads0=1",
        "numSearchThreads1=1",
        "numNNServerThreadsPerModel=1",
        "useScoreMaximizingUtility=false",
        "useExpectedMaxScoreUtility=true",
        f"extremeScoreGroupSize={group_size}",
        f"expectedMaxFocalColor={focal_color}",
    ]
    return (
        _override_value(katago_binary, "KataGo binary"),
        "match",
        "-config",
        _override_value(config_path, "match config"),
        "-override-config",
        ",".join(overrides),
    )


def _scheduled_focal_color(schedule: Mapping[str, Any], role: str) -> str:
    job = schedule.get("job")
    if not isinstance(job, Mapping):
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role}.job must be an immutable scheduled job"
        )
    focal_color = job.get("focal_color")
    if focal_color not in _COLORS:
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role}.job.focal_color must be B or W"
        )
    expected_black = 0 if focal_color == "B" else 1
    expected_white = 1 if focal_color == "B" else 0
    if (
        schedule.get("blackBot") != expected_black
        or schedule.get("whiteBot") != expected_white
    ):
        raise ExtremeScoreMatchRunnerValidationError(
            f"{role} bot assignment contradicts its scheduled focal color"
        )
    return focal_color


def _validate_and_translate_match_results(
    schedule_rows: Sequence[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    *,
    focal_color: str | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if focal_color is not None and focal_color not in _COLORS:
        raise ExtremeScoreMatchRunnerValidationError(
            "asserted focal color must be B or W"
        )
    expected: dict[str, Mapping[str, Any]] = {}
    scheduled_colors: dict[str, str] = {}
    frozen_schedules: list[dict[str, Any]] = []
    for index, input_schedule in enumerate(schedule_rows):
        role = f"schedule row {index}"
        if not isinstance(input_schedule, Mapping):
            raise ExtremeScoreMatchRunnerValidationError(f"{role} must be an object")
        schedule = _json_clone(input_schedule)
        game_id = schedule.get("gameId")
        if not isinstance(game_id, str) or not game_id or game_id in expected:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} has an invalid or duplicate gameId"
            )
        scheduled_color = _scheduled_focal_color(schedule, role)
        if focal_color is not None and focal_color != scheduled_color:
            raise ExtremeScoreMatchRunnerValidationError(
                "caller focal_color contradicts the immutable scheduled job"
            )
        expected[game_id] = schedule
        scheduled_colors[game_id] = scheduled_color
        frozen_schedules.append(schedule)
    observed: dict[str, dict[str, Any]] = {}
    for index, input_row in enumerate(result_rows):
        role = f"match result {index}"
        if not isinstance(input_row, Mapping):
            raise ExtremeScoreMatchRunnerValidationError(f"{role} must be an object")
        row = _json_clone(input_row)
        game_id = row.get("gameId")
        if not isinstance(game_id, str) or game_id not in expected:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} has an unplanned gameId"
            )
        if game_id in observed:
            raise ExtremeScoreMatchRunnerValidationError(
                f"duplicate match result gameId {game_id!r}"
            )
        schedule = expected[game_id]
        if row.get("schemaVersion") != 1:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} has unsupported schemaVersion"
            )
        for key in ("scheduleId", "gameId", "pairId", "positionId", "seed"):
            if row.get(key) != schedule[key]:
                raise ExtremeScoreMatchRunnerValidationError(
                    f"{role} {key} does not match the immutable schedule"
                )
        if (
            row.get("blackBotIndex") != schedule["blackBot"]
            or row.get("whiteBotIndex") != schedule["whiteBot"]
        ):
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} bot indices do not match the immutable schedule"
            )
        expected_names = {0: FOCAL_BOT_NAME, 1: OPPONENT_BOT_NAME}
        if (
            row.get("blackBot") != expected_names[schedule["blackBot"]]
            or row.get("whiteBot") != expected_names[schedule["whiteBot"]]
        ):
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} bot names do not match bot0 focal/bot1 opponent"
            )
        board = row.get("board")
        if (
            not isinstance(board, Mapping)
            or board.get("xSize") != 19
            or board.get("ySize") != 19
        ):
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} must report a 19x19 board"
            )
        komi = _finite_number(row.get("komi"), f"{role}.komi")
        if komi != 7.5:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} must report fixed 7.5 komi"
            )
        expected_rules = {
            "ko": "POSITIONAL",
            "scoring": "AREA",
            "tax": "NONE",
            "suicide": True,
            "hasButton": False,
            "whiteHandicapBonus": "0",
            "friendlyPassOk": False,
            "komi": 7.5,
        }
        if row.get("rules") != expected_rules:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} rules do not match the frozen 19x19 match policy"
            )
        hit_turn_limit = _boolean(row.get("hitTurnLimit"), f"{role}.hitTurnLimit")
        no_result = _boolean(row.get("noResult"), f"{role}.noResult")
        resignation = _boolean(row.get("resignation"), f"{role}.resignation")
        scored = _boolean(row.get("scored"), f"{role}.scored")
        if resignation:
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} is a resignation despite resignation being disabled"
            )
        unresolved = no_result or hit_turn_limit
        if unresolved:
            if scored:
                raise ExtremeScoreMatchRunnerValidationError(
                    f"{role} unresolved result cannot be scored"
                )
            focal_score = 0.0
        else:
            if not scored:
                raise ExtremeScoreMatchRunnerValidationError(
                    f"{role} resolved game must have a terminal score"
                )
            white_minus_black = _finite_number(
                row.get("finalWhiteMinusBlackScore"),
                f"{role}.finalWhiteMinusBlackScore",
            )
            scheduled_color = scheduled_colors[game_id]
            focal_score = (
                -white_minus_black if scheduled_color == "B" else white_minus_black
            )
            winner = row.get("winner")
            expected_winner = (
                "W"
                if white_minus_black > 0.0
                else "B"
                if white_minus_black < 0.0
                else "draw"
            )
            if winner != expected_winner:
                raise ExtremeScoreMatchRunnerValidationError(
                    f"{role} winner contradicts the terminal score"
                )
        if not math.isfinite(focal_score):
            raise ExtremeScoreMatchRunnerValidationError(
                f"{role} focal terminal score is not finite"
            )
        observed[game_id] = row

    missing = [
        row["gameId"] for row in frozen_schedules if row["gameId"] not in observed
    ]
    if missing:
        raise ExtremeScoreMatchRunnerValidationError(
            f"match output is missing scheduled games: {missing}"
        )
    ordered_raw = tuple(observed[row["gameId"]] for row in frozen_schedules)
    evaluator_rows: list[dict[str, Any]] = []
    for schedule, raw in zip(frozen_schedules, ordered_raw):
        no_result = bool(raw["noResult"])
        hit_turn_limit = bool(raw["hitTurnLimit"])
        if no_result or hit_turn_limit:
            focal_score = 0.0
        else:
            terminal_score = float(raw["finalWhiteMinusBlackScore"])
            scheduled_color = scheduled_colors[schedule["gameId"]]
            focal_score = -terminal_score if scheduled_color == "B" else terminal_score
        evaluator_rows.append(
            {
                **_json_clone(schedule["job"]),
                "score": focal_score,
                "no_result": no_result,
                "hit_turn_limit": hit_turn_limit,
            }
        )
    return ordered_raw, tuple(evaluator_rows)


def evaluator_rows_from_match_results(
    schedule_rows: Sequence[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    *,
    focal_color: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate C++ results and derive score perspective from scheduled jobs.

    ``focal_color`` is only a compatibility assertion. It never controls score
    conversion and is rejected if it differs from an immutable scheduled job.
    """

    _, rows = _validate_and_translate_match_results(
        schedule_rows, result_rows, focal_color=focal_color
    )
    return rows


class ExtremeScoreMatchRunner:
    """Callable production adapter for ``evaluate_with_runner``."""

    def __init__(
        self,
        spec: ExtremeScoreMatchRunnerSpec,
        *,
        subprocess_runner: Callable[..., Any] = subprocess.run,
        fake_result_provider: Callable[[MatchCell], Iterable[Mapping[str, Any]]]
        | None = None,
        gpu_identity_probe: Callable[[int], GpuIdentityObservation | Mapping[str, Any]]
        | None = None,
    ) -> None:
        if not isinstance(spec, ExtremeScoreMatchRunnerSpec):
            raise TypeError("spec must be an ExtremeScoreMatchRunnerSpec")
        if not callable(subprocess_runner):
            raise TypeError("subprocess_runner must be callable")
        if fake_result_provider is not None and not callable(fake_result_provider):
            raise TypeError("fake_result_provider must be callable")
        if gpu_identity_probe is not None and not callable(gpu_identity_probe):
            raise TypeError("gpu_identity_probe must be callable")
        self.spec = spec
        self.subprocess_runner = subprocess_runner
        self.fake_result_provider = fake_result_provider
        self.gpu_identity_probe = (
            _nvidia_smi_gpu_identity_probe
            if gpu_identity_probe is None
            else gpu_identity_probe
        )
        self._binary_path = _normalized_file(spec.katago_binary, "KataGo binary")
        self._binary_sha256 = file_sha256(self._binary_path)
        self._config_path = _normalized_file(spec.match_config, "match config")
        self._config_sha256 = file_sha256(self._config_path)
        self._focal_models = _normalize_model_map(spec.focal_models, "focal model")
        self._opponent_models = _normalize_model_map(
            spec.opponent_models, "frozen opponent"
        )
        output = Path(os.path.abspath(spec.output_root))
        if output.exists() and (output.is_symlink() or not output.is_dir()):
            raise ExtremeScoreMatchRunnerValidationError(
                f"output_root must be a non-symlink directory: {output}"
            )
        self._output_root = output
        self._runner_binding = {
            "contract": RUNNER_SPEC_CONTRACT,
            "katago_binary": {
                "path": str(self._binary_path),
                "file_sha256": self._binary_sha256,
            },
            "focal_models": {
                digest: {"path": str(path), "file_sha256": bound_hash}
                for digest, (path, bound_hash) in self._focal_models.items()
            },
            "frozen_opponents": {
                digest: {"path": str(path), "file_sha256": bound_hash}
                for digest, (path, bound_hash) in self._opponent_models.items()
            },
            "match_config": {
                "path": str(self._config_path),
                "file_sha256": self._config_sha256,
            },
            "output_root": str(self._output_root),
            "topology": spec.topology,
            "process_count": spec.process_count,
            "gpu_index": spec.gpu_index,
            "expected_gpu_uuid": spec.expected_gpu_uuid,
            "gpu_lease_provenance": spec.gpu_lease_provenance,
            "result_source": (
                "fake-provider"
                if self.fake_result_provider is not None
                else "katago-subprocess"
            ),
        }
        self._runner_spec_sha256 = canonical_sha256(self._runner_binding)
        self._paired_plan_sha256: str | None = None
        self._arm_plan_sha256: dict[str, str] = {}
        self._source_bindings: dict[str, dict[str, str]] = {}
        self._state_lock = threading.Lock()

    @property
    def runner_binding(self) -> dict[str, Any]:
        return _json_clone(self._runner_binding)

    @property
    def runner_spec_sha256(self) -> str:
        return self._runner_spec_sha256

    @property
    def source_bindings(self) -> dict[str, dict[str, str]]:
        with self._state_lock:
            bindings = _json_clone(self._source_bindings)
        for arm, binding in bindings.items():
            path = _normalized_immutable_file(
                Path(binding["path"]), f"{arm} evaluator results"
            )
            if file_sha256(path) != binding["file_sha256"]:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{arm} evaluator results changed after publication"
                )
        return bindings

    @property
    def execution_provenance(self) -> dict[str, Any]:
        """Return a transitive binding to completed arm and cell receipts."""
        sources = self.source_bindings
        if set(sources) != set(_ARMS):
            raise ExtremeScoreMatchRunnerConflictError(
                "execution provenance requires both completed evaluator arms"
            )
        arm_receipts: dict[str, dict[str, Any]] = {}
        for arm in _ARMS:
            result_path = Path(sources[arm]["path"])
            receipt_path = result_path.parent / "receipt.json"
            receipt = _load_receipt(receipt_path, ARM_RECEIPT_CONTRACT)
            if (
                receipt.get("runner_spec_sha256") != self._runner_spec_sha256
                or receipt.get("results", {}).get("file_sha256")
                != sources[arm]["file_sha256"]
            ):
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{arm} receipt contradicts runner or result binding"
                )
            arm_receipts[arm] = {
                "path": str(receipt_path.resolve()),
                "file_sha256": file_sha256(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "cell_receipts": receipt["cells"],
            }
        value = {
            "schema_version": 1,
            "contract": EXECUTION_PROVENANCE_CONTRACT,
            "runner_binding": self.runner_binding,
            "runner_spec_sha256": self._runner_spec_sha256,
            "result_sources": sources,
            "arm_receipts": arm_receipts,
        }
        value["provenance_sha256"] = canonical_sha256(value)
        return value

    def _verify_bound_inputs(self) -> None:
        expected_files = [
            (self._binary_path, self._binary_sha256, "KataGo binary"),
            (self._config_path, self._config_sha256, "match config"),
            *[
                (path, digest, f"focal model {digest}")
                for digest, (path, _) in self._focal_models.items()
            ],
            *[
                (path, digest, f"frozen opponent {digest}")
                for digest, (path, _) in self._opponent_models.items()
            ],
        ]
        for path, expected_hash, role in expected_files:
            current = _normalized_file(path, role)
            if current != path or file_sha256(current) != expected_hash:
                raise ExtremeScoreMatchRunnerValidationError(
                    f"{role} changed after runner construction"
                )
        self._output_root.mkdir(parents=True, exist_ok=True)
        if self._output_root.is_symlink() or not self._output_root.is_dir():
            raise ExtremeScoreMatchRunnerValidationError(
                "output_root became an unsafe path"
            )

    def _observe_gpu_identity(self, stage: str) -> GpuIdentityObservation:
        try:
            supplied = self.gpu_identity_probe(self.spec.gpu_index)
        except ExtremeScoreMatchRunnerError:
            raise
        except Exception as exc:
            raise ExtremeScoreMatchRunnerError(
                f"GPU identity probe failed {stage}: {type(exc).__name__}: {exc}"
            ) from exc
        return _coerce_gpu_identity_observation(
            supplied, f"GPU identity observed {stage}"
        )

    def _gpu_execution_binding(
        self,
        before: GpuIdentityObservation,
        after: GpuIdentityObservation,
    ) -> dict[str, Any]:
        before_identity = (before.gpu_index, before.gpu_uuid)
        after_identity = (after.gpu_index, after.gpu_uuid)
        if before_identity != after_identity:
            raise ExtremeScoreMatchRunnerValidationError(
                "physical GPU identity changed between pre- and post-execution "
                f"probes: {before_identity!r} -> {after_identity!r}"
            )
        if before.gpu_index != self.spec.gpu_index:
            raise ExtremeScoreMatchRunnerValidationError(
                "GPU ordinal remapped before execution: "
                f"observed {before.gpu_index}, expected {self.spec.gpu_index}"
            )
        if before.gpu_uuid != self.spec.expected_gpu_uuid:
            raise ExtremeScoreMatchRunnerValidationError(
                "configured GPU ordinal resolves to the wrong physical UUID: "
                f"observed {before.gpu_uuid}, expected "
                f"{self.spec.expected_gpu_uuid}"
            )
        return {
            "contract": GPU_EXECUTION_CONTRACT,
            "source": "katago-subprocess",
            "gpu_index": self.spec.gpu_index,
            "expected_gpu_uuid": self.spec.expected_gpu_uuid,
            "gpu_lease_provenance": self.spec.gpu_lease_provenance,
            "observed_before": {
                "gpu_index": before.gpu_index,
                "gpu_uuid": before.gpu_uuid,
                "provenance": before.provenance,
            },
            "observed_after": {
                "gpu_index": after.gpu_index,
                "gpu_uuid": after.gpu_uuid,
                "provenance": after.provenance,
            },
        }

    def _validated_execution_binding(self, value: Any) -> dict[str, Any]:
        if self.fake_result_provider is not None:
            expected = {"source": "fake-provider"}
            if value != expected:
                raise ExtremeScoreMatchRunnerConflictError(
                    "cell execution provenance contradicts the fake result source"
                )
            return expected
        expected_keys = {
            "contract",
            "source",
            "gpu_index",
            "expected_gpu_uuid",
            "gpu_lease_provenance",
            "observed_before",
            "observed_after",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ExtremeScoreMatchRunnerConflictError(
                "cell GPU execution provenance is malformed"
            )
        if (
            value["contract"] != GPU_EXECUTION_CONTRACT
            or value["source"] != "katago-subprocess"
            or value["gpu_index"] != self.spec.gpu_index
            or value["expected_gpu_uuid"] != self.spec.expected_gpu_uuid
            or value["gpu_lease_provenance"] != self.spec.gpu_lease_provenance
        ):
            raise ExtremeScoreMatchRunnerConflictError(
                "cell GPU execution provenance contradicts the runner lease"
            )
        try:
            before = _coerce_gpu_identity_observation(
                value["observed_before"], "receipt GPU identity before execution"
            )
            after = _coerce_gpu_identity_observation(
                value["observed_after"], "receipt GPU identity after execution"
            )
            expected = self._gpu_execution_binding(before, after)
        except ExtremeScoreMatchRunnerValidationError as exc:
            raise ExtremeScoreMatchRunnerConflictError(str(exc)) from exc
        if value != expected:
            raise ExtremeScoreMatchRunnerConflictError(
                "cell GPU execution provenance is noncanonical"
            )
        return _json_clone(expected)

    def _bind_paired_plan(self, plan: _ValidatedJobPlan) -> None:
        with self._state_lock:
            if (
                self._paired_plan_sha256 is not None
                and self._paired_plan_sha256 != plan.paired_plan_sha256
            ):
                raise ExtremeScoreMatchRunnerConflictError(
                    "candidate and reference jobs do not use an identical "
                    "plan and seeds"
                )
            previous_arm_plan = self._arm_plan_sha256.get(plan.arm)
            jobs_hash = canonical_sha256(plan.jobs)
            if previous_arm_plan is not None and previous_arm_plan != jobs_hash:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} runner jobs changed within one execution"
                )
            self._paired_plan_sha256 = plan.paired_plan_sha256
            self._arm_plan_sha256[plan.arm] = jobs_hash

    def _plan_root(self, plan: _ValidatedJobPlan) -> Path:
        return self._output_root / "plans" / plan.plan_sha256

    def _ensure_runner_receipt(
        self, plan_root: Path, plan: _ValidatedJobPlan
    ) -> dict[str, Any]:
        receipt = _self_hashed_artifact(
            RUNNER_RECEIPT_CONTRACT,
            {
                "plan_sha256": plan.plan_sha256,
                "runner_spec": self._runner_binding,
                "runner_spec_sha256": self._runner_spec_sha256,
            },
        )
        path = plan_root / "runner.json"
        _publish_immutable_bytes(path, (canonical_json(receipt) + "\n").encode("utf-8"))
        return receipt

    def _ensure_paired_plan_receipt(
        self, plan_root: Path, plan: _ValidatedJobPlan
    ) -> dict[str, Any]:
        receipt = _self_hashed_artifact(
            PAIRED_PLAN_RECEIPT_CONTRACT,
            {
                "plan_sha256": plan.plan_sha256,
                "config_sha256": plan.config_sha256,
                "group_size": plan.group_size,
                "paired_plan_sha256": plan.paired_plan_sha256,
                "paired_jobs": plan.paired_jobs,
            },
        )
        _publish_immutable_bytes(
            plan_root / "paired-plan.json",
            (canonical_json(receipt) + "\n").encode("utf-8"),
        )
        return receipt

    def _check_counterpart_receipt(
        self, plan_root: Path, plan: _ValidatedJobPlan
    ) -> None:
        counterpart = "reference" if plan.arm == "candidate" else "candidate"
        path = plan_root / "arms" / counterpart / "receipt.json"
        if not _path_entry_exists(path):
            return
        receipt = _load_receipt(path, ARM_RECEIPT_CONTRACT)
        identity = receipt.get("arm_identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("plan_sha256") != plan.plan_sha256
            or identity.get("paired_plan_sha256") != plan.paired_plan_sha256
            or identity.get("group_size") != plan.group_size
            or receipt.get("runner_spec_sha256") != self._runner_spec_sha256
        ):
            raise ExtremeScoreMatchRunnerConflictError(
                "candidate and reference receipts do not share one plan and seeds"
            )

    def _build_cells(self, plan: _ValidatedJobPlan) -> tuple[MatchCell, ...]:
        grouped = group_jobs_by_opponent_and_color(plan.jobs)
        cells: list[MatchCell] = []
        for opponent_hash, color in sorted(grouped):
            cell_jobs = grouped[(opponent_hash, color)]
            schedule = build_empty_board_schedule(cell_jobs)
            cell_identity = {
                "arm": plan.arm,
                "plan_sha256": plan.plan_sha256,
                "paired_plan_sha256": plan.paired_plan_sha256,
                "focal_model_sha256": plan.model_sha256,
                "opponent_model_sha256": opponent_hash,
                "focal_color": color,
                "group_size": plan.group_size,
                "jobs_sha256": canonical_sha256(cell_jobs),
                "paired_jobs_sha256": canonical_sha256(
                    [_paired_job(job) for job in cell_jobs]
                ),
                "schedule_sha256": hashlib.sha256(
                    _canonical_jsonl(schedule)
                ).hexdigest(),
            }
            cell_id = f"extreme-score-cell-{canonical_sha256(cell_identity)}"
            cells.append(
                MatchCell(
                    cell_id=cell_id,
                    arm=plan.arm,
                    plan_sha256=plan.plan_sha256,
                    paired_plan_sha256=plan.paired_plan_sha256,
                    focal_model_sha256=plan.model_sha256,
                    opponent_model_sha256=opponent_hash,
                    focal_color=color,
                    group_size=plan.group_size,
                    jobs=cell_jobs,
                    schedule_rows=schedule,
                )
            )
        return tuple(cells)

    def _cell_directory(self, plan_root: Path, cell: MatchCell) -> Path:
        return (
            plan_root
            / "cells"
            / cell.arm
            / f"{cell.focal_color}-{cell.opponent_model_sha256}"
        )

    def _cell_identity(self, cell: MatchCell) -> dict[str, Any]:
        return {
            "cell_id": cell.cell_id,
            "arm": cell.arm,
            "plan_sha256": cell.plan_sha256,
            "paired_plan_sha256": cell.paired_plan_sha256,
            "focal_model_sha256": cell.focal_model_sha256,
            "opponent_model_sha256": cell.opponent_model_sha256,
            "focal_color": cell.focal_color,
            "group_size": cell.group_size,
            "jobs_sha256": canonical_sha256(cell.jobs),
            "paired_jobs_sha256": canonical_sha256(
                [_paired_job(job) for job in cell.jobs]
            ),
            "schedule_id": cell.schedule_id,
        }

    def _cell_receipt(
        self,
        cell: MatchCell,
        schedule_data: bytes,
        raw_data: bytes,
        evaluator_data: bytes,
        execution_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _self_hashed_artifact(
            CELL_RECEIPT_CONTRACT,
            {
                "runner_spec_sha256": self._runner_spec_sha256,
                "cell_identity": self._cell_identity(cell),
                "execution": _json_clone(execution_binding),
                "artifacts": {
                    "schedule": _artifact_record(
                        "schedule.jsonl", schedule_data, len(cell.schedule_rows)
                    ),
                    "match_results": _artifact_record(
                        "match-results.jsonl", raw_data, len(cell.schedule_rows)
                    ),
                    "evaluator_results": _artifact_record(
                        "results.jsonl", evaluator_data, len(cell.jobs)
                    ),
                },
            },
        )

    def _validate_evaluator_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        jobs: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if len(rows) != len(jobs):
            raise ExtremeScoreMatchRunnerConflictError(
                "evaluator result row count contradicts jobs"
            )
        checked: list[dict[str, Any]] = []
        for index, (row, job) in enumerate(zip(rows, jobs)):
            if not isinstance(row, Mapping):
                raise ExtremeScoreMatchRunnerConflictError(
                    f"evaluator result {index} is not an object"
                )
            if set(row) != _JOB_KEYS | {
                "score",
                "no_result",
                "hit_turn_limit",
            }:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"evaluator result {index} keys contradict the contract"
                )
            if any(row.get(key) != job[key] for key in _JOB_KEYS):
                raise ExtremeScoreMatchRunnerConflictError(
                    f"evaluator result {index} job identity changed"
                )
            _finite_number(row.get("score"), f"evaluator result {index}.score")
            _boolean(row.get("no_result"), f"evaluator result {index}.no_result")
            _boolean(
                row.get("hit_turn_limit"),
                f"evaluator result {index}.hit_turn_limit",
            )
            checked.append(_json_clone(row))
        return tuple(checked)

    def _resume_cell(
        self, directory: Path, cell: MatchCell
    ) -> tuple[dict[str, Any], ...] | None:
        receipt_path = directory / "receipt.json"
        if not _path_entry_exists(receipt_path):
            return None
        receipt = _load_receipt(receipt_path, CELL_RECEIPT_CONTRACT)
        if receipt.get("runner_spec_sha256") != self._runner_spec_sha256 or receipt.get(
            "cell_identity"
        ) != self._cell_identity(cell):
            raise ExtremeScoreMatchRunnerConflictError(
                f"completed cell receipt contradicts {cell.cell_id}"
            )
        self._validated_execution_binding(receipt.get("execution"))
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ExtremeScoreMatchRunnerConflictError("cell artifacts are malformed")
        schedule_path = _verify_artifact(
            directory,
            artifacts.get("schedule"),
            expected_path="schedule.jsonl",
            expected_rows=len(cell.schedule_rows),
            role="cell schedule",
        )
        raw_path = _verify_artifact(
            directory,
            artifacts.get("match_results"),
            expected_path="match-results.jsonl",
            expected_rows=len(cell.schedule_rows),
            role="cell match results",
        )
        evaluator_path = _verify_artifact(
            directory,
            artifacts.get("evaluator_results"),
            expected_path="results.jsonl",
            expected_rows=len(cell.jobs),
            role="cell evaluator results",
        )
        if schedule_path.read_bytes() != _canonical_jsonl(cell.schedule_rows):
            raise ExtremeScoreMatchRunnerConflictError(
                "completed cell schedule contradicts precommitted jobs"
            )
        raw_rows = _load_jsonl(raw_path, "cell match results")
        _, derived = _validate_and_translate_match_results(cell.schedule_rows, raw_rows)
        evaluator_rows = self._validate_evaluator_rows(
            _load_jsonl(evaluator_path, "cell evaluator results"), cell.jobs
        )
        if _canonical_jsonl(derived) != _canonical_jsonl(evaluator_rows):
            raise ExtremeScoreMatchRunnerConflictError(
                "cell evaluator results contradict raw match output"
            )
        return evaluator_rows

    def _run_match_process(
        self, cell: MatchCell, schedule_path: Path, result_path: Path
    ) -> dict[str, Any]:
        if self.fake_result_provider is not None:
            supplied = tuple(self.fake_result_provider(cell))
            result_path.write_bytes(_canonical_jsonl(supplied))
            return {"source": "fake-provider"}
        focal_path = self._focal_models[cell.focal_model_sha256][0]
        opponent_path = self._opponent_models[cell.opponent_model_sha256][0]
        command = build_match_command(
            self._binary_path,
            self._config_path,
            focal_path,
            opponent_path,
            schedule_path,
            result_path,
            focal_color=cell.focal_color,
            group_size=cell.group_size,
            game_count=len(cell.jobs),
        )
        environment = dict(os.environ)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(self.spec.gpu_index)
        before = self._observe_gpu_identity("before subprocess execution")
        self._gpu_execution_binding(before, before)
        completed: Any | None = None
        launch_error: Exception | None = None
        try:
            completed = self.subprocess_runner(
                list(command),
                cwd=str(self._output_root),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except Exception as exc:  # noqa: BLE001 - injected runner boundary
            launch_error = exc
        try:
            after = self._observe_gpu_identity("after subprocess execution")
        except Exception as probe_exc:
            if launch_error is not None:
                raise ExtremeScoreMatchRunnerError(
                    f"katago match failed to start for {cell.cell_id} and the "
                    "post-execution GPU identity probe also failed"
                ) from probe_exc
            raise
        execution_binding = self._gpu_execution_binding(before, after)
        if launch_error is not None:
            raise ExtremeScoreMatchRunnerError(
                f"katago match failed to start for {cell.cell_id}: "
                f"{type(launch_error).__name__}: {launch_error}"
            ) from launch_error
        returncode = getattr(completed, "returncode", None)
        if type(returncode) is not int or returncode != 0:
            stderr = getattr(completed, "stderr", "")
            detail = f": {stderr[:1000]}" if isinstance(stderr, str) and stderr else ""
            raise ExtremeScoreMatchRunnerError(
                f"katago match returned {returncode!r} for {cell.cell_id}{detail}"
            )
        return execution_binding

    def _execute_cell(
        self, plan_root: Path, cell: MatchCell
    ) -> tuple[MatchCell, tuple[dict[str, Any], ...], Path]:
        directory = self._cell_directory(plan_root, cell)
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ExtremeScoreMatchRunnerConflictError(
                f"cell directory is unsafe: {directory}"
            )
        schedule_path = directory / "schedule.jsonl"
        schedule_data = _canonical_jsonl(cell.schedule_rows)
        _publish_immutable_bytes(schedule_path, schedule_data)
        resumed = self._resume_cell(directory, cell)
        if resumed is not None:
            return cell, resumed, directory / "receipt.json"

        raw_path = directory / "match-results.jsonl"
        evaluator_path = directory / "results.jsonl"
        if _path_entry_exists(evaluator_path) and not _path_entry_exists(raw_path):
            raise ExtremeScoreMatchRunnerConflictError(
                f"orphan evaluator results cannot be resumed: {evaluator_path}"
            )
        if _path_entry_exists(raw_path):
            if self.fake_result_provider is None:
                raise ExtremeScoreMatchRunnerConflictError(
                    "subprocess match results without a completed GPU-bound "
                    f"cell receipt cannot be resumed: {raw_path}"
                )
            immutable_raw_path = _normalized_immutable_file(
                raw_path, "recoverable match results"
            )
            raw_rows = _load_jsonl(immutable_raw_path, "recoverable match results")
            execution_binding = {"source": "fake-provider"}
        else:
            with tempfile.TemporaryDirectory(
                prefix=".match-attempt-", dir=directory
            ) as temporary_name:
                attempt_path = Path(temporary_name) / "match-results.jsonl"
                execution_binding = self._run_match_process(
                    cell, schedule_path, attempt_path
                )
                self._verify_bound_inputs()
                if not attempt_path.is_file():
                    raise ExtremeScoreMatchRunnerError(
                        f"katago match produced no result file for {cell.cell_id}"
                    )
                raw_rows = _load_jsonl(attempt_path, "KataGo match results")
        ordered_raw, evaluator_rows = _validate_and_translate_match_results(
            cell.schedule_rows, raw_rows
        )
        raw_data = _canonical_jsonl(ordered_raw)
        evaluator_data = _canonical_jsonl(evaluator_rows)
        _publish_immutable_bytes(raw_path, raw_data)
        _publish_immutable_bytes(evaluator_path, evaluator_data)
        receipt = self._cell_receipt(
            cell,
            schedule_data,
            raw_data,
            evaluator_data,
            execution_binding,
        )
        receipt_path = directory / "receipt.json"
        _publish_immutable_bytes(
            receipt_path, (canonical_json(receipt) + "\n").encode("utf-8")
        )
        return cell, evaluator_rows, receipt_path

    def _arm_identity(self, plan: _ValidatedJobPlan) -> dict[str, Any]:
        return {
            "arm": plan.arm,
            "plan_sha256": plan.plan_sha256,
            "paired_plan_sha256": plan.paired_plan_sha256,
            "model_sha256": plan.model_sha256,
            "config_sha256": plan.config_sha256,
            "group_size": plan.group_size,
            "jobs_sha256": canonical_sha256(plan.jobs),
            "paired_jobs_sha256": canonical_sha256(plan.paired_jobs),
        }

    def _resume_arm(
        self, plan_root: Path, plan: _ValidatedJobPlan
    ) -> tuple[dict[str, Any], ...] | None:
        directory = plan_root / "arms" / plan.arm
        receipt_path = directory / "receipt.json"
        if not _path_entry_exists(receipt_path):
            return None
        receipt = _load_receipt(receipt_path, ARM_RECEIPT_CONTRACT)
        if receipt.get("runner_spec_sha256") != self._runner_spec_sha256 or receipt.get(
            "arm_identity"
        ) != self._arm_identity(plan):
            raise ExtremeScoreMatchRunnerConflictError(
                f"completed {plan.arm} receipt contradicts runner jobs"
            )
        result_path = _verify_artifact(
            directory,
            receipt.get("results"),
            expected_path="results.jsonl",
            expected_rows=len(plan.jobs),
            role=f"{plan.arm} merged results",
        )
        rows = self._validate_evaluator_rows(
            _load_jsonl(result_path, f"{plan.arm} merged results"), plan.jobs
        )
        receipt_cells = receipt.get("cells")
        expected_cells = self._build_cells(plan)
        if not isinstance(receipt_cells, list) or len(receipt_cells) != len(
            expected_cells
        ):
            raise ExtremeScoreMatchRunnerConflictError(
                f"{plan.arm} receipt has the wrong cell bindings"
            )
        bindings_by_id: dict[str, Mapping[str, Any]] = {}
        for index, binding in enumerate(receipt_cells):
            if not isinstance(binding, Mapping) or set(binding) != {
                "cell_id",
                "receipt_path",
                "receipt_file_sha256",
            }:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} cell receipt binding {index} is malformed"
                )
            cell_id = binding["cell_id"]
            if not isinstance(cell_id, str) or cell_id in bindings_by_id:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} cell receipt IDs are invalid"
                )
            bindings_by_id[cell_id] = binding

        resumed_by_job: dict[str, dict[str, Any]] = {}
        for cell in expected_cells:
            binding = bindings_by_id.get(cell.cell_id)
            if binding is None:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} receipt omits expected cell {cell.cell_id}"
                )
            relative = Path(str(binding["receipt_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} cell receipt path is unsafe"
                )
            expected_receipt_path = (
                self._cell_directory(plan_root, cell) / "receipt.json"
            )
            if plan_root / relative != expected_receipt_path:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} cell receipt path contradicts its identity"
                )
            expected_hash = _sha256(
                binding["receipt_file_sha256"], "cell receipt file SHA-256"
            )
            bound_path = plan_root / relative
            actual_hash = file_sha256(
                _normalized_immutable_file(bound_path, "cell receipt")
            )
            if actual_hash != expected_hash:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} cell receipt changed: {bound_path}"
                )
            cell_rows = self._resume_cell(self._cell_directory(plan_root, cell), cell)
            if cell_rows is None:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} completed cell cannot be resumed"
                )
            for row in cell_rows:
                job = {key: row[key] for key in _JOB_KEYS}
                job_hash = canonical_sha256(job)
                if job_hash in resumed_by_job:
                    raise ExtremeScoreMatchRunnerConflictError(
                        f"{plan.arm} resumed cells duplicate a job"
                    )
                resumed_by_job[job_hash] = row
        resumed_rows = tuple(
            resumed_by_job.get(canonical_sha256(job)) for job in plan.jobs
        )
        if any(row is None for row in resumed_rows) or _canonical_jsonl(
            row for row in resumed_rows if row is not None
        ) != _canonical_jsonl(rows):
            raise ExtremeScoreMatchRunnerConflictError(
                f"{plan.arm} merged results contradict completed cells"
            )
        self._set_source_binding(plan.arm, result_path)
        return rows

    def _set_source_binding(self, arm: str, path: Path) -> None:
        immutable_path = _normalized_immutable_file(path, f"{arm} evaluator results")
        binding = {
            "path": str(immutable_path),
            "file_sha256": file_sha256(immutable_path),
        }
        with self._state_lock:
            previous = self._source_bindings.get(arm)
            if previous is not None and previous != binding:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{arm} source binding changed"
                )
            self._source_bindings[arm] = binding

    def _publish_arm(
        self,
        plan_root: Path,
        plan: _ValidatedJobPlan,
        outcomes: Sequence[tuple[MatchCell, tuple[dict[str, Any], ...], Path]],
    ) -> tuple[dict[str, Any], ...]:
        by_job_hash: dict[str, dict[str, Any]] = {}
        cell_bindings: list[dict[str, Any]] = []
        for cell, rows, receipt_path in outcomes:
            for row in rows:
                job = {key: row[key] for key in _JOB_KEYS}
                key = canonical_sha256(job)
                if key in by_job_hash:
                    raise ExtremeScoreMatchRunnerConflictError(
                        f"duplicate completed job in {plan.arm} cells"
                    )
                by_job_hash[key] = row
            cell_bindings.append(
                {
                    "cell_id": cell.cell_id,
                    "receipt_path": str(receipt_path.relative_to(plan_root)),
                    "receipt_file_sha256": file_sha256(receipt_path),
                }
            )
        ordered: list[dict[str, Any]] = []
        for job in plan.jobs:
            key = canonical_sha256(job)
            row = by_job_hash.get(key)
            if row is None:
                raise ExtremeScoreMatchRunnerConflictError(
                    f"{plan.arm} cells omitted a precommitted job"
                )
            ordered.append(row)
        rows = self._validate_evaluator_rows(ordered, plan.jobs)
        directory = plan_root / "arms" / plan.arm
        result_path = directory / "results.jsonl"
        result_data = _canonical_jsonl(rows)
        _publish_immutable_bytes(result_path, result_data)
        receipt = _self_hashed_artifact(
            ARM_RECEIPT_CONTRACT,
            {
                "runner_spec_sha256": self._runner_spec_sha256,
                "arm_identity": self._arm_identity(plan),
                "results": _artifact_record("results.jsonl", result_data, len(rows)),
                "cells": sorted(cell_bindings, key=lambda item: item["cell_id"]),
            },
        )
        _publish_immutable_bytes(
            directory / "receipt.json",
            (canonical_json(receipt) + "\n").encode("utf-8"),
        )
        self._set_source_binding(plan.arm, result_path)
        return rows

    def __call__(
        self, arm: str, jobs: tuple[dict[str, Any], ...]
    ) -> tuple[dict[str, Any], ...]:
        """Execute or exactly resume one evaluator arm."""

        self._verify_bound_inputs()
        plan = _validate_jobs(
            arm,
            jobs,
            focal_model_hashes=set(self._focal_models),
            opponent_model_hashes=set(self._opponent_models),
            config_sha256=self._config_sha256,
        )
        self._bind_paired_plan(plan)
        plan_root = self._plan_root(plan)
        self._ensure_runner_receipt(plan_root, plan)
        self._ensure_paired_plan_receipt(plan_root, plan)
        self._check_counterpart_receipt(plan_root, plan)
        resumed = self._resume_arm(plan_root, plan)
        if resumed is not None:
            return resumed

        cells = self._build_cells(plan)
        outcomes: list[tuple[MatchCell, tuple[dict[str, Any], ...], Path]] = []
        workers = min(self.spec.process_count, len(cells))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._execute_cell, plan_root, cell) for cell in cells
            ]
            for future in futures:
                outcomes.append(future.result())
        rows = self._publish_arm(plan_root, plan, outcomes)
        self._verify_bound_inputs()
        return rows


ProductionMatchRunner = ExtremeScoreMatchRunner
convert_match_results = evaluator_rows_from_match_results
