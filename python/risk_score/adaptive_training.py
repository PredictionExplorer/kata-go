#!/usr/bin/env python3
"""Bounded, replay-safe adaptive training trial orchestration.

This module plans training-recipe experiments.  It deliberately does not
promote models: a winning trial can only emit a hash-bound handoff for the
normal promotion-v3 path.  Confirmation and audit evidence are rejected from
all tuning and ranking operations.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as datetime_module
import errno
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
import stat
import string
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from risk_score.cluster_executor import WORK_SPEC_CONTRACT
from risk_score.cluster_scheduler import (
    ClusterScheduler,
    SchedulerError,
    WorkKind,
    WorkRecord,
    WorkState,
)


PathLike = Union[str, os.PathLike]

SCHEMA_VERSION = 1
POLICY_VERSION = "risk-score-training-autonomy-v1"
POLICY_HASH = "5b2ddc255d05587a4067d398681443b15707438802c65a8a0a625fc92727522d"
EXPECTED_POLICY_VERSION = POLICY_VERSION
EXPECTED_POLICY_HASH = POLICY_HASH
DEFAULT_POLICY_PATH = Path(__file__).with_name("autonomy_policy_v1.json")
STATUS_CONTRACT = "risk-score-adaptive-training-status-v1"
EVENT_CONTRACT = "risk-score-adaptive-training-event-v1"
EPOCH_CONTRACT = "risk-score-adaptive-training-epoch-v1"
TRIAL_CONTRACT = "risk-score-adaptive-training-trial-v1"
RECIPE_CONTRACT = "risk-score-adaptive-training-recipe-v1"
EVIDENCE_CONTRACT = "risk-score-adaptive-training-evidence-v1"
HANDOFF_CONTRACT = "risk-score-adaptive-candidate-handoff-v1"
RECIPE_BINDING_CONTRACT = "risk-score-active-training-recipe-v1"
ROLLBACK_CONTRACT = "risk-score-training-recipe-rollback-v1"
SERVICE_SPEC_CONTRACT = "risk-score-adaptive-training-service-spec-v1"
OBSERVATION_CONTRACT = "risk-score-adaptive-training-observation-v1"
TRIAL_RESULT_CONTRACT = "risk-score-adaptive-training-trial-result-v1"
SERVICE_STATUS_CONTRACT = "risk-score-adaptive-training-service-status-v1"
ADAPTIVE_WORK_CONTRACT = "risk-score-adaptive-training-work-v1"
ADAPTIVE_SERVICE_SPEC_CONTRACT = SERVICE_SPEC_CONTRACT
ADAPTIVE_OBSERVATION_CONTRACT = OBSERVATION_CONTRACT
ADAPTIVE_TRIAL_RESULT_CONTRACT = TRIAL_RESULT_CONTRACT
GENESIS_HASH = "0" * 64
EVENT_SEQUENCE_WIDTH = 20
MAX_JSON_BYTES = 64 * 1024 * 1024
OBSERVATION_FRESHNESS_POLLS = 2.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_EVENT_FILE_RE = re.compile(r"^([0-9]{20})\.json$")
_FORBIDDEN_EVIDENCE_TERMS = (
    "audit",
    "confirmation",
    "holdout",
)
_EVENT_TYPES = {
    "epoch.planned",
    "trial.started",
    "trial.gpu_usage_recorded",
    "trial.evidence_recorded",
    "trial.completed",
    "trial.failed",
    "round.halved",
    "candidate.handoff_created",
}
_SERVICE_SPEC_FIELDS = {
    "actor",
    "autonomy_policy_path",
    "autonomy_policy_sha256",
    "contract",
    "gpu7_id",
    "gpu_lease_guardian_argv_prefix",
    "observation_path",
    "poll_interval_seconds",
    "root",
    "scheduler_directory",
    "schema_version",
    "spec_sha256",
    "trial_command_argv_template",
}
_OBSERVATION_FIELDS = {
    "admitted_data_manifest",
    "admitted_samples",
    "candidate_queue_depth",
    "champion_checkpoint",
    "contract",
    "current_champion_model_sha256",
    "last_promotion_admitted_samples",
    "observation_sha256",
    "schema_version",
    "updated_at_unix",
}
_TRIAL_RESULT_FIELDS = {
    "candidate_checkpoint",
    "candidate_model",
    "contract",
    "epoch_id",
    "evidence",
    "failure_reason",
    "gpu_usage",
    "result_sha256",
    "round_index",
    "schema_version",
    "status",
    "trial_id",
    "trial_manifest_path",
    "trial_manifest_sha256",
    "work_id",
}
_FILE_BINDING_FIELDS = {"path", "sha256"}
_RESUMABLE_FILE_BINDING_FIELDS = {"path", "resumable", "sha256"}
_GPU_USAGE_FIELDS = {
    "ended_at_unix",
    "gpu_count",
    "gpu_id",
    "started_at_unix",
}
_REQUIRED_TRIAL_TEMPLATE_FIELDS = {
    "trial_manifest_path",
    "trial_result_path",
    "work_id",
}
_ALLOWED_TRIAL_TEMPLATE_FIELDS = frozenset(
    _REQUIRED_TRIAL_TEMPLATE_FIELDS
    | {
        "epoch_id",
        "gpu_id",
        "recipe_path",
        "recipe_sha256",
        "round_index",
        "service_spec_sha256",
        "trial_id",
        "trial_manifest_sha256",
    }
)

__all__ = [
    "AdaptiveEvent",
    "ADAPTIVE_OBSERVATION_CONTRACT",
    "ADAPTIVE_SERVICE_SPEC_CONTRACT",
    "ADAPTIVE_TRIAL_RESULT_CONTRACT",
    "AdaptiveTrainingError",
    "AdaptiveService",
    "AdaptiveServiceSpec",
    "AdaptiveTrainingService",
    "AdaptiveTrainingServiceSpec",
    "AdaptiveTrainingStore",
    "ADAPTIVE_WORK_CONTRACT",
    "BudgetExceededError",
    "BudgetStatus",
    "DEFAULT_POLICY_PATH",
    "EXPECTED_POLICY_HASH",
    "EXPECTED_POLICY_VERSION",
    "EvidenceRejectedError",
    "GpuInterval",
    "HANDOFF_CONTRACT",
    "OBSERVATION_CONTRACT",
    "ObservationValidationError",
    "POLICY_HASH",
    "POLICY_VERSION",
    "PolicyValidationError",
    "RecipeConflictError",
    "SERVICE_SPEC_CONTRACT",
    "SERVICE_STATUS_CONTRACT",
    "ServiceSpec",
    "ServiceSpecError",
    "StateCorruptionError",
    "TRIAL_RESULT_CONTRACT",
    "TrialResultValidationError",
    "TriggerDecision",
    "TrialConflictError",
    "atomic_create_json",
    "atomic_write_json",
    "bootstrap_recipe_binding",
    "build_candidate_handoff",
    "build_rollback_metadata",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "compare_and_swap_recipe_binding",
    "content_addressed_epoch_id",
    "content_addressed_trial_id",
    "deterministic_rank",
    "deterministic_successive_halving",
    "evaluate_trigger",
    "file_sha256",
    "gpu_budget_status",
    "load_canonical_json",
    "load_candidate_handoff",
    "load_adaptive_observation",
    "load_adaptive_service_spec",
    "load_observation",
    "load_policy",
    "load_recipe_binding",
    "load_service_spec",
    "load_trial_result",
    "once",
    "parse_args",
    "publish_adaptive_observation",
    "publish_adaptive_service_spec",
    "publish_observation",
    "publish_service_spec",
    "publish_trial_result",
    "recipe_hash",
    "rollback_recipe_binding",
    "rolling_gpu_seconds",
    "select_epoch_recipes",
    "should_trigger_trial",
    "status",
    "utc_timestamp",
    "validate_evidence",
    "validate_policy",
    "validate_recipe",
    "watch",
]


class AdaptiveTrainingError(RuntimeError):
    """Operational failure with a stable, machine-readable error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": _json_copy(self.details),
            }
        }


class PolicyValidationError(AdaptiveTrainingError, ValueError):
    """The frozen adaptive-training policy is malformed or unpinned."""


class StateCorruptionError(AdaptiveTrainingError, ValueError):
    """Durable adaptive-training state is malformed or inconsistent."""


class TrialConflictError(AdaptiveTrainingError):
    """A trial operation conflicts with reconstructed lifecycle state."""


class BudgetExceededError(AdaptiveTrainingError):
    """A reservation or usage record would exceed the frozen GPU budget."""


class EvidenceRejectedError(AdaptiveTrainingError, ValueError):
    """Evidence is malformed or comes from a prohibited tuning source."""


class RecipeConflictError(AdaptiveTrainingError):
    """A recipe compare-and-swap is stale or conflicts with a retry."""


class ServiceSpecError(AdaptiveTrainingError, ValueError):
    """The immutable unattended-service specification is invalid."""


class ObservationValidationError(AdaptiveTrainingError, ValueError):
    """The mutable adaptive-training observation is stale or malformed."""


class TrialResultValidationError(AdaptiveTrainingError, ValueError):
    """A scheduler-produced adaptive trial result is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically, rejecting non-finite values."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdaptiveTrainingError(
            "noncanonical_json",
            f"value is not canonical-JSON compatible: {exc}",
        ) from exc


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def file_sha256(path: PathLike) -> str:
    source = Path(path)
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AdaptiveTrainingError(
            "invalid_file",
            f"expected a regular non-symlink file: {source}",
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _stable_file_sha256(path: PathLike) -> str:
    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise AdaptiveTrainingError(
            "invalid_binding_file",
            f"binding must name a regular non-symlink file: {source}",
        )
    digest = file_sha256(source)
    after = source.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise AdaptiveTrainingError(
            "binding_changed_while_hashing",
            f"binding changed while it was hashed: {source}",
        )
    return digest


def utc_timestamp(
    now: Optional[datetime_module.datetime] = None,
) -> str:
    current = now or datetime_module.datetime.now(datetime_module.timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise AdaptiveTrainingError(
            "invalid_timestamp",
            "timestamp must be timezone-aware",
        )
    return (
        current.astimezone(datetime_module.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value: Any, role: str) -> datetime_module.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AdaptiveTrainingError(
            "invalid_timestamp",
            f"{role} must be an ISO-8601 UTC timestamp ending in Z",
        )
    try:
        parsed = datetime_module.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AdaptiveTrainingError(
            "invalid_timestamp",
            f"{role} is not a valid ISO-8601 UTC timestamp",
        ) from exc
    if parsed.utcoffset() != datetime_module.timedelta(0):
        raise AdaptiveTrainingError(
            "invalid_timestamp",
            f"{role} must be UTC",
        )
    return parsed


def _epoch_seconds(value: Any, role: str) -> float:
    if isinstance(value, datetime_module.datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise AdaptiveTrainingError(
                "invalid_time",
                f"{role} datetime must be timezone-aware",
            )
        result = value.timestamp()
    elif isinstance(value, str):
        if value.endswith("Z"):
            result = _parse_utc_timestamp(value, role).timestamp()
        else:
            try:
                result = float(value)
            except ValueError as exc:
                raise AdaptiveTrainingError(
                    "invalid_time",
                    f"{role} must be numeric or an ISO-8601 UTC timestamp",
                ) from exc
    elif not isinstance(value, bool) and isinstance(value, (int, float)):
        result = float(value)
    else:
        raise AdaptiveTrainingError(
            "invalid_time",
            f"{role} must be epoch seconds or an ISO-8601 UTC timestamp",
        )
    if not math.isfinite(result):
        raise AdaptiveTrainingError("invalid_time", f"{role} must be finite")
    return result


def _require_hash(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AdaptiveTrainingError(
            "invalid_hash",
            f"{role} must be a lowercase 64-character SHA-256",
        )
    return value


def _safe_id(value: Any, role: str = "identifier") -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise AdaptiveTrainingError(
            "unsafe_identifier",
            f"{role} must be a safe single path component",
        )
    return value


def _nonnegative_integer(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdaptiveTrainingError(
            "invalid_integer",
            f"{role} must be a nonnegative integer",
        )
    return value


def _positive_integer(value: Any, role: str) -> int:
    result = _nonnegative_integer(value, role)
    if result < 1:
        raise AdaptiveTrainingError(
            "invalid_integer",
            f"{role} must be a positive integer",
        )
    return result


def _finite_number(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AdaptiveTrainingError(
            "invalid_number",
            f"{role} must be a finite number",
        )
    return float(value)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    missing: List[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise AdaptiveTrainingError(
            "invalid_directory",
            f"directory ancestor is not a real directory: {current}",
        )
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)
    if path.is_symlink() or not path.is_dir():
        raise AdaptiveTrainingError(
            "invalid_directory",
            f"expected a non-symlink directory: {path}",
        )


def _atomic_temp(path: Path, data: bytes, mode: int) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def atomic_write_json(
    path: PathLike,
    value: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> None:
    """Atomically replace a canonical JSON projection and fsync its directory."""

    destination = Path(path)
    _ensure_directory(destination.parent)
    if destination.exists() and (
        destination.is_symlink() or not destination.is_file()
    ):
        raise AdaptiveTrainingError(
            "invalid_json_destination",
            f"JSON destination is not a regular file: {destination}",
        )
    temporary = _atomic_temp(destination, _canonical_file_bytes(value), mode)
    try:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(
    path: PathLike,
    value: Mapping[str, Any],
    *,
    mode: int = 0o444,
) -> bool:
    """Create immutable canonical JSON, accepting an exact replay.

    Returns ``True`` when a byte-identical existing artifact was reused.
    """

    destination = Path(path)
    _ensure_directory(destination.parent)
    data = _canonical_file_bytes(value)
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != data
        ):
            raise StateCorruptionError(
                "immutable_artifact_conflict",
                f"immutable JSON artifact conflicts with replay: {destination}",
            )
        return True
    temporary = _atomic_temp(destination, data, mode)
    try:
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != data
            ):
                raise StateCorruptionError(
                    "immutable_artifact_conflict",
                    f"concurrent immutable artifact conflicts: {destination}",
                )
            return True
        _fsync_directory(destination.parent)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise AdaptiveTrainingError(
        "noncanonical_json",
        f"non-finite JSON number is forbidden: {value}",
    )


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AdaptiveTrainingError(
                "duplicate_json_key",
                f"duplicate JSON object key: {key}",
            )
        value[key] = item
    return value


def _read_regular_file(path: Path, maximum_bytes: int = MAX_JSON_BYTES) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AdaptiveTrainingError(
            "invalid_json_file",
            f"expected a regular non-symlink JSON file: {path}",
        )
    if metadata.st_size > maximum_bytes:
        raise AdaptiveTrainingError(
            "json_too_large",
            f"JSON file exceeds the size limit: {path}",
        )
    return path.read_bytes()


def load_canonical_json(
    path: PathLike,
    role: str = "JSON artifact",
) -> Dict[str, Any]:
    source = Path(path)
    try:
        data = _read_regular_file(source)
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except FileNotFoundError:
        raise
    except AdaptiveTrainingError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdaptiveTrainingError(
            "invalid_json",
            f"cannot load {role} {source}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise AdaptiveTrainingError(
            "invalid_json_root",
            f"{role} must have an object root",
        )
    if data != _canonical_file_bytes(value):
        raise AdaptiveTrainingError(
            "noncanonical_json_file",
            f"{role} must be canonical newline-terminated JSON: {source}",
        )
    return value


_POLICY_TOP_LEVEL_FIELDS = {
    "allowed_recipe_knobs",
    "bindings",
    "evidence",
    "frozen_plan",
    "frozen_surfaces",
    "gpu_budget",
    "handoff",
    "policy_version",
    "queue",
    "schema_version",
    "status",
    "successive_halving",
    "trials",
    "trigger",
}
_RECIPE_KEYS = {
    "bucket_cap_samples",
    "bucket_ratio",
    "data_recency_window_mixture",
    "export_cadence_epochs",
    "learning_rate_scale",
    "learning_rate_schedule",
    "swa_cadence_samples",
}


def _policy_error(message: str, *, details: Optional[Mapping[str, Any]] = None) -> None:
    raise PolicyValidationError(
        "invalid_frozen_policy",
        message,
        details=details,
    )


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Require the exact canonical frozen version-1 meta-policy."""

    if not isinstance(policy, Mapping):
        _policy_error("adaptive-training policy root must be an object")
    if set(policy) != _POLICY_TOP_LEVEL_FIELDS:
        _policy_error(
            "adaptive-training policy fields differ from the frozen schema",
            details={
                "missing": sorted(_POLICY_TOP_LEVEL_FIELDS - set(policy)),
                "extra": sorted(set(policy) - _POLICY_TOP_LEVEL_FIELDS),
            },
        )
    if policy.get("schema_version") != SCHEMA_VERSION:
        _policy_error("adaptive-training policy schema version is unsupported")
    if policy.get("policy_version") != POLICY_VERSION:
        _policy_error("adaptive-training policy version is unsupported")
    if policy.get("status") != "frozen":
        _policy_error("adaptive-training policy must remain frozen")
    actual_hash = canonical_sha256(policy)
    if actual_hash != POLICY_HASH:
        _policy_error(
            "adaptive-training policy content hash does not match the pinned policy",
            details={"expected": POLICY_HASH, "actual": actual_hash},
        )

    knobs = policy.get("allowed_recipe_knobs")
    if not isinstance(knobs, Mapping) or set(knobs) != _RECIPE_KEYS:
        _policy_error("allowed_recipe_knobs differs from the frozen allowlist")
    if any(not isinstance(knobs[key], list) or not knobs[key] for key in _RECIPE_KEYS):
        _policy_error("every allowlisted recipe knob must have a non-empty value list")
    frozen_surfaces = policy.get("frozen_surfaces")
    required_frozen = {
        "architecture",
        "audit_inputs",
        "confirmation_inputs",
        "game_rules",
        "objective",
        "promotion_thresholds",
    }
    if not isinstance(frozen_surfaces, list) or set(frozen_surfaces) != required_frozen:
        _policy_error("the frozen policy does not protect every required surface")
    evidence = policy.get("evidence")
    if not isinstance(evidence, Mapping):
        _policy_error("evidence policy must be an object")
    if evidence.get("allowed_sources") != ["discovery", "fixed_validation"]:
        _policy_error("only discovery and fixed_validation evidence may tune recipes")
    if evidence.get("forbidden_sources") != ["audit", "confirmation"]:
        _policy_error("confirmation and audit evidence must remain forbidden")
    trigger = policy.get("trigger")
    queue = policy.get("queue")
    trials = policy.get("trials")
    budget = policy.get("gpu_budget")
    halving = policy.get("successive_halving")
    handoff = policy.get("handoff")
    if not isinstance(trigger, Mapping) or trigger.get(
        "minimum_admitted_samples_without_promotion"
    ) != 3_000_000:
        _policy_error("trial trigger must remain exactly 3,000,000 admitted samples")
    if (
        not isinstance(queue, Mapping)
        or queue.get("maximum_candidate_queue_depth") != 3
    ):
        _policy_error("candidate queue bound must remain exactly three")
    if not isinstance(trials, Mapping) or trials.get("maximum_active") != 1:
        _policy_error("maximum active adaptive trials must remain exactly one")
    if (
        not isinstance(budget, Mapping)
        or budget.get("rolling_window_seconds") != 7 * 24 * 60 * 60
        or budget.get("maximum_fraction") != 0.1
        or budget.get("host_gpu_count") != 8
    ):
        _policy_error("rolling GPU budget must remain 10% of eight GPUs for seven days")
    if (
        not isinstance(halving, Mapping)
        or halving.get("initial_recipe_count") != 8
        or halving.get("reduction_factor") != 2
        or halving.get("minimum_survivors") != 1
        or halving.get("round_gpu_seconds") != [14_400, 28_800, 57_600]
    ):
        _policy_error("successive-halving configuration differs from the frozen policy")
    if (
        not isinstance(handoff, Mapping)
        or handoff.get("contract") != HANDOFF_CONTRACT
        or handoff.get("direct_promotion_permitted") is not False
        or handoff.get("promotion_policy_version")
        != "risk-seeking-checkpoint-promotion-v3"
    ):
        _policy_error("candidate handoff must use the unchanged promotion-v3 path")


def load_policy(path: PathLike = DEFAULT_POLICY_PATH) -> Dict[str, Any]:
    try:
        policy = load_canonical_json(path, "adaptive-training policy")
        validate_policy(policy)
        return policy
    except PolicyValidationError:
        raise
    except AdaptiveTrainingError as exc:
        raise PolicyValidationError(
            "invalid_frozen_policy",
            str(exc),
            details={"path": str(path), "cause": exc.code},
        ) from exc


def validate_recipe(
    recipe: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a canonical copy of a complete allowlisted recipe."""

    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    if not isinstance(recipe, Mapping):
        raise AdaptiveTrainingError(
            "invalid_recipe",
            "training recipe must be an object",
        )
    if set(recipe) != _RECIPE_KEYS:
        forbidden = sorted(set(recipe) - _RECIPE_KEYS)
        raise AdaptiveTrainingError(
            "recipe_surface_forbidden",
            "training recipe must contain exactly the frozen allowlisted knobs",
            details={
                "missing": sorted(_RECIPE_KEYS - set(recipe)),
                "forbidden_or_unknown": forbidden,
            },
        )
    knobs = active_policy["allowed_recipe_knobs"]
    normalized: Dict[str, Any] = {}
    for key in sorted(_RECIPE_KEYS):
        supplied = canonical_json_bytes(recipe[key])
        matches = [
            item
            for item in knobs[key]
            if canonical_json_bytes(item) == supplied
        ]
        if len(matches) != 1:
            raise AdaptiveTrainingError(
                "recipe_value_forbidden",
                f"recipe value for {key} is not in the frozen allowlist",
                details={"knob": key, "value": _json_copy(recipe[key])},
            )
        normalized[key] = _json_copy(matches[0])
    return normalized


def recipe_hash(
    recipe: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]] = None,
) -> str:
    return canonical_sha256(validate_recipe(recipe, policy))


def _all_recipes(policy: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    knobs = policy["allowed_recipe_knobs"]
    keys = sorted(_RECIPE_KEYS)
    for values in itertools.product(*(knobs[key] for key in keys)):
        yield {key: _json_copy(value) for key, value in zip(keys, values)}


def select_epoch_recipes(
    epoch_id: str,
    policy: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """Deterministically select the frozen number of recipes for an epoch."""

    _safe_id(epoch_id, "epoch_id")
    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    count = active_policy["successive_halving"]["initial_recipe_count"]
    ranked = sorted(
        _all_recipes(active_policy),
        key=lambda item: (
            canonical_sha256(
                {
                    "epoch_id": epoch_id,
                    "policy_hash": POLICY_HASH,
                    "recipe_sha256": canonical_sha256(item),
                }
            ),
            canonical_sha256(item),
        ),
    )
    return tuple(ranked[:count])


def content_addressed_epoch_id(
    *,
    parent_champion_model_sha256: str,
    champion_checkpoint_sha256: str,
    admitted_data_manifest_sha256: str,
    admitted_samples: int,
    last_promotion_admitted_samples: int,
    policy_hash: str = POLICY_HASH,
) -> str:
    identity = {
        "admitted_data_manifest_sha256": _require_hash(
            admitted_data_manifest_sha256,
            "admitted_data_manifest_sha256",
        ),
        "admitted_samples": _nonnegative_integer(
            admitted_samples,
            "admitted_samples",
        ),
        "champion_checkpoint_sha256": _require_hash(
            champion_checkpoint_sha256,
            "champion_checkpoint_sha256",
        ),
        "parent_champion_model_sha256": _require_hash(
            parent_champion_model_sha256,
            "parent_champion_model_sha256",
        ),
        "contract": EPOCH_CONTRACT,
        "last_promotion_admitted_samples": _nonnegative_integer(
            last_promotion_admitted_samples,
            "last_promotion_admitted_samples",
        ),
        "policy_hash": _require_hash(policy_hash, "policy_hash"),
    }
    if identity["last_promotion_admitted_samples"] > identity["admitted_samples"]:
        raise AdaptiveTrainingError(
            "invalid_sample_watermark",
            "last promotion sample watermark exceeds admitted samples",
        )
    return "epoch-" + canonical_sha256(identity)


def content_addressed_trial_id(
    *,
    epoch_id: str,
    recipe_sha256: str,
    parent_champion_model_sha256: str,
    champion_checkpoint_sha256: str,
    admitted_data_manifest_sha256: str,
    policy_hash: str = POLICY_HASH,
) -> str:
    identity = {
        "admitted_data_manifest_sha256": _require_hash(
            admitted_data_manifest_sha256,
            "admitted_data_manifest_sha256",
        ),
        "champion_checkpoint_sha256": _require_hash(
            champion_checkpoint_sha256,
            "champion_checkpoint_sha256",
        ),
        "parent_champion_model_sha256": _require_hash(
            parent_champion_model_sha256,
            "parent_champion_model_sha256",
        ),
        "contract": TRIAL_CONTRACT,
        "epoch_id": _safe_id(epoch_id, "epoch_id"),
        "policy_hash": _require_hash(policy_hash, "policy_hash"),
        "recipe_sha256": _require_hash(recipe_sha256, "recipe_sha256"),
    }
    return "trial-" + canonical_sha256(identity)


@dataclass(frozen=True)
class GpuInterval:
    started_at: Union[float, str, datetime_module.datetime]
    ended_at: Optional[Union[float, str, datetime_module.datetime]]
    gpu_count: int = 1
    trial_id: Optional[str] = None

    def normalized(self, now: Any) -> Tuple[float, float, int]:
        start = _epoch_seconds(self.started_at, "GPU interval start")
        end = (
            _epoch_seconds(now, "budget now")
            if self.ended_at is None
            else _epoch_seconds(self.ended_at, "GPU interval end")
        )
        count = _positive_integer(self.gpu_count, "GPU interval gpu_count")
        if end < start:
            raise AdaptiveTrainingError(
                "invalid_gpu_interval",
                "GPU interval ends before it starts",
            )
        return start, end, count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": (
                self.started_at
                if not isinstance(self.started_at, datetime_module.datetime)
                else utc_timestamp(self.started_at)
            ),
            "ended_at": (
                self.ended_at
                if not isinstance(self.ended_at, datetime_module.datetime)
                else utc_timestamp(self.ended_at)
            ),
            "gpu_count": self.gpu_count,
            "trial_id": self.trial_id,
        }


def _coerce_interval(value: Union[GpuInterval, Mapping[str, Any]]) -> GpuInterval:
    if isinstance(value, GpuInterval):
        return value
    if not isinstance(value, Mapping):
        raise AdaptiveTrainingError(
            "invalid_gpu_interval",
            "GPU usage entry must be a GpuInterval or object",
        )
    start = value.get("started_at", value.get("started_at_utc"))
    end = value.get("ended_at", value.get("ended_at_utc"))
    return GpuInterval(
        started_at=start,
        ended_at=end,
        gpu_count=value.get("gpu_count", 1),
        trial_id=value.get("trial_id"),
    )


def rolling_gpu_seconds(
    intervals: Iterable[Union[GpuInterval, Mapping[str, Any]]],
    *,
    now: Union[float, str, datetime_module.datetime],
    window_seconds: int = 7 * 24 * 60 * 60,
) -> float:
    """Return overlap-weighted GPU-seconds in ``[now-window, now]``."""

    end_of_window = _epoch_seconds(now, "budget now")
    width = _positive_integer(window_seconds, "window_seconds")
    start_of_window = end_of_window - width
    usage = 0.0
    for raw in intervals:
        started, ended, gpu_count = _coerce_interval(raw).normalized(end_of_window)
        clipped_start = max(started, start_of_window)
        clipped_end = min(ended, end_of_window)
        if clipped_end > clipped_start:
            usage += (clipped_end - clipped_start) * gpu_count
    return usage


@dataclass(frozen=True)
class BudgetStatus:
    rolling_gpu_seconds: float
    requested_gpu_seconds: float
    projected_gpu_seconds: float
    maximum_gpu_seconds: float
    remaining_gpu_seconds: float
    allowed: bool
    rolling_window_seconds: int
    host_gpu_count: int
    maximum_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "host_gpu_count": self.host_gpu_count,
            "maximum_fraction": self.maximum_fraction,
            "maximum_gpu_seconds": self.maximum_gpu_seconds,
            "projected_gpu_seconds": self.projected_gpu_seconds,
            "remaining_gpu_seconds": self.remaining_gpu_seconds,
            "requested_gpu_seconds": self.requested_gpu_seconds,
            "rolling_gpu_seconds": self.rolling_gpu_seconds,
            "rolling_window_seconds": self.rolling_window_seconds,
        }


def gpu_budget_status(
    intervals: Iterable[Union[GpuInterval, Mapping[str, Any]]],
    *,
    now: Union[float, str, datetime_module.datetime],
    requested_gpu_seconds: float = 0.0,
    policy: Optional[Mapping[str, Any]] = None,
) -> BudgetStatus:
    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    requested = _finite_number(requested_gpu_seconds, "requested_gpu_seconds")
    if requested < 0:
        raise AdaptiveTrainingError(
            "invalid_gpu_budget_request",
            "requested GPU seconds must not be negative",
        )
    budget = active_policy["gpu_budget"]
    window = budget["rolling_window_seconds"]
    host_gpus = budget["host_gpu_count"]
    fraction = float(budget["maximum_fraction"])
    maximum = float(window * host_gpus) * fraction
    used = rolling_gpu_seconds(intervals, now=now, window_seconds=window)
    projected = used + requested
    remaining = max(0.0, maximum - used)
    return BudgetStatus(
        rolling_gpu_seconds=used,
        requested_gpu_seconds=requested,
        projected_gpu_seconds=projected,
        maximum_gpu_seconds=maximum,
        remaining_gpu_seconds=remaining,
        allowed=projected <= maximum,
        rolling_window_seconds=window,
        host_gpu_count=host_gpus,
        maximum_fraction=fraction,
    )


@dataclass(frozen=True)
class TriggerDecision:
    eligible: bool
    reason_codes: Tuple[str, ...]
    admitted_samples_since_baseline: int
    required_admitted_samples: int
    candidate_queue_depth: int
    maximum_candidate_queue_depth: int
    active_trial_count: int
    budget: BudgetStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_trial_count": self.active_trial_count,
            "admitted_samples_since_baseline": self.admitted_samples_since_baseline,
            "budget": self.budget.to_dict(),
            "candidate_queue_depth": self.candidate_queue_depth,
            "eligible": self.eligible,
            "maximum_candidate_queue_depth": self.maximum_candidate_queue_depth,
            "reason_codes": list(self.reason_codes),
            "required_admitted_samples": self.required_admitted_samples,
        }


def evaluate_trigger(
    *,
    admitted_samples: int,
    last_promotion_admitted_samples: int,
    candidate_queue_depth: int,
    active_trial_count: int = 0,
    last_trial_epoch_admitted_samples: Optional[int] = None,
    gpu_intervals: Iterable[Union[GpuInterval, Mapping[str, Any]]] = (),
    now: Union[float, str, datetime_module.datetime] = 0.0,
    policy: Optional[Mapping[str, Any]] = None,
) -> TriggerDecision:
    """Evaluate every frozen trigger predicate without mutating state."""

    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    admitted = _nonnegative_integer(admitted_samples, "admitted_samples")
    promoted_at = _nonnegative_integer(
        last_promotion_admitted_samples,
        "last_promotion_admitted_samples",
    )
    queue_depth = _nonnegative_integer(
        candidate_queue_depth,
        "candidate_queue_depth",
    )
    active = _nonnegative_integer(active_trial_count, "active_trial_count")
    baseline = promoted_at
    if last_trial_epoch_admitted_samples is not None:
        baseline = max(
            baseline,
            _nonnegative_integer(
                last_trial_epoch_admitted_samples,
                "last_trial_epoch_admitted_samples",
            ),
        )
    if baseline > admitted:
        raise AdaptiveTrainingError(
            "invalid_sample_watermark",
            "promotion/trial watermark exceeds admitted samples",
        )
    since = admitted - baseline
    required = active_policy["trigger"][
        "minimum_admitted_samples_without_promotion"
    ]
    maximum_queue = active_policy["queue"]["maximum_candidate_queue_depth"]
    maximum_active = active_policy["trials"]["maximum_active"]
    first_round = active_policy["successive_halving"]["round_gpu_seconds"][0]
    budget = gpu_budget_status(
        gpu_intervals,
        now=now,
        requested_gpu_seconds=first_round,
        policy=active_policy,
    )
    reasons: List[str] = []
    if since < required:
        reasons.append("INSUFFICIENT_ADMITTED_SAMPLES")
    if queue_depth > maximum_queue:
        reasons.append("CANDIDATE_QUEUE_UNBOUNDED")
    if active >= maximum_active:
        reasons.append("ACTIVE_TRIAL_EXISTS")
    if not budget.allowed:
        reasons.append("GPU_BUDGET_EXHAUSTED")
    return TriggerDecision(
        eligible=not reasons,
        reason_codes=tuple(reasons),
        admitted_samples_since_baseline=since,
        required_admitted_samples=required,
        candidate_queue_depth=queue_depth,
        maximum_candidate_queue_depth=maximum_queue,
        active_trial_count=active,
        budget=budget,
    )


def should_trigger_trial(**kwargs: Any) -> bool:
    return evaluate_trigger(**kwargs).eligible


def _contains_forbidden_evidence_reference(
    value: Any,
    path: str = "$",
) -> Optional[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower().replace("-", "_")
            if any(term in lowered for term in _FORBIDDEN_EVIDENCE_TERMS):
                return f"{path}.{key}"
            found = _contains_forbidden_evidence_reference(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _contains_forbidden_evidence_reference(item, f"{path}[{index}]")
            if found is not None:
                return found
    elif isinstance(value, str):
        lowered = value.lower().replace("-", "_")
        tokens = set(re.split(r"[^a-z0-9_]+", lowered))
        if any(
            term in tokens
            or f"_{term}" in lowered
            or f"{term}_" in lowered
            for term in _FORBIDDEN_EVIDENCE_TERMS
        ):
            return path
    return None


def validate_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_trial_id: Optional[str] = None,
    expected_round_index: Optional[int] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate one discovery or fixed-validation tuning artifact."""

    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    if not isinstance(evidence, Mapping):
        raise EvidenceRejectedError(
            "invalid_tuning_evidence",
            "tuning evidence must be an object",
        )
    source = evidence.get("source")
    if source in active_policy["evidence"]["forbidden_sources"]:
        raise EvidenceRejectedError(
            "holdout_evidence_forbidden",
            f"{source} evidence must never feed adaptive training",
            details={"source": source},
        )
    if source not in active_policy["evidence"]["allowed_sources"]:
        raise EvidenceRejectedError(
            "evidence_source_forbidden",
            "only discovery and fixed_validation evidence may rank trials",
            details={"source": source},
        )
    forbidden_path = _contains_forbidden_evidence_reference(evidence)
    if forbidden_path is not None:
        raise EvidenceRejectedError(
            "holdout_reference_forbidden",
            "tuning evidence contains a confirmation/audit/holdout reference",
            details={"path": forbidden_path},
        )
    required_fields = {
        "artifact_sha256",
        "finalized",
        "metrics",
        "round_index",
        "sample_count",
        "schema_version",
        "source",
        "trial_id",
    }
    if set(evidence) != required_fields:
        raise EvidenceRejectedError(
            "invalid_tuning_evidence",
            "tuning evidence fields differ from the frozen schema",
            details={
                "missing": sorted(required_fields - set(evidence)),
                "extra": sorted(set(evidence) - required_fields),
            },
        )
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceRejectedError(
            "invalid_tuning_evidence",
            "tuning evidence schema version is unsupported",
        )
    if evidence.get("finalized") is not True:
        raise EvidenceRejectedError(
            "unfinalized_tuning_evidence",
            "tuning evidence must be finalized before ranking",
        )
    try:
        trial_id = _safe_id(evidence.get("trial_id"), "evidence.trial_id")
        artifact_hash = _require_hash(
            evidence.get("artifact_sha256"),
            "evidence.artifact_sha256",
        )
        round_index = _nonnegative_integer(
            evidence.get("round_index"),
            "evidence.round_index",
        )
        sample_count = _positive_integer(
            evidence.get("sample_count"),
            "evidence.sample_count",
        )
    except AdaptiveTrainingError as exc:
        raise EvidenceRejectedError(
            "invalid_tuning_evidence",
            str(exc),
            details={"cause": exc.code},
        ) from exc
    if expected_trial_id is not None and trial_id != expected_trial_id:
        raise EvidenceRejectedError(
            "evidence_trial_mismatch",
            "evidence trial identity does not match the target trial",
        )
    if expected_round_index is not None and round_index != expected_round_index:
        raise EvidenceRejectedError(
            "evidence_round_mismatch",
            "evidence round does not match the active trial round",
        )
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvidenceRejectedError(
            "invalid_tuning_evidence",
            "evidence metrics must be an object",
        )
    metric_name = (
        "discovery_powered_terminal_utility"
        if source == "discovery"
        else "fixed_validation_loss"
    )
    if set(metrics) != {metric_name}:
        raise EvidenceRejectedError(
            "ranking_metric_forbidden",
            f"{source} evidence must contain only {metric_name}",
        )
    try:
        metric_value = _finite_number(metrics[metric_name], metric_name)
    except AdaptiveTrainingError as exc:
        raise EvidenceRejectedError(
            "invalid_tuning_evidence",
            str(exc),
            details={"cause": exc.code},
        ) from exc
    return {
        "artifact_sha256": artifact_hash,
        "finalized": True,
        "metrics": {metric_name: metric_value},
        "round_index": round_index,
        "sample_count": sample_count,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "trial_id": trial_id,
    }


def _evidence_pairs(
    evidence: Iterable[Mapping[str, Any]],
    *,
    expected_round_index: Optional[int] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw in evidence:
        validated = validate_evidence(
            raw,
            expected_round_index=expected_round_index,
            policy=policy,
        )
        by_source = grouped.setdefault(validated["trial_id"], {})
        source = validated["source"]
        existing = by_source.get(source)
        if existing is not None and existing != validated:
            raise EvidenceRejectedError(
                "duplicate_tuning_evidence",
                "a trial has conflicting evidence for one source and round",
                details={
                    "trial_id": validated["trial_id"],
                    "source": source,
                },
            )
        by_source[source] = validated
    return grouped


def deterministic_rank(
    evidence: Iterable[Mapping[str, Any]],
    *,
    trial_ids: Optional[Iterable[str]] = None,
    round_index: Optional[int] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    """Rank complete evidence pairs with deterministic lexical tie breaking."""

    grouped = _evidence_pairs(
        evidence,
        expected_round_index=round_index,
        policy=policy,
    )
    expected = None if trial_ids is None else {_safe_id(item) for item in trial_ids}
    if expected is not None:
        unexpected = set(grouped) - expected
        if unexpected:
            raise EvidenceRejectedError(
                "unexpected_trial_evidence",
                "ranking evidence names a trial outside the requested set",
                details={"trial_ids": sorted(unexpected)},
            )
    complete: List[Tuple[float, float, str]] = []
    candidates = sorted(grouped if expected is None else expected)
    for trial_id in candidates:
        by_source = grouped.get(trial_id, {})
        if set(by_source) != {"discovery", "fixed_validation"}:
            raise EvidenceRejectedError(
                "incomplete_tuning_evidence",
                "every ranked trial requires discovery and fixed_validation evidence",
                details={
                    "trial_id": trial_id,
                    "sources": sorted(by_source),
                },
            )
        discovery = by_source["discovery"]["metrics"][
            "discovery_powered_terminal_utility"
        ]
        validation_loss = by_source["fixed_validation"]["metrics"][
            "fixed_validation_loss"
        ]
        complete.append((-discovery, validation_loss, trial_id))
    complete.sort()
    return tuple(item[2] for item in complete)


def deterministic_successive_halving(
    evidence: Iterable[Mapping[str, Any]],
    *,
    trial_ids: Iterable[str],
    round_index: int,
    policy: Optional[Mapping[str, Any]] = None,
    final_round: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return ``(ranking, survivors)`` under the frozen halving rule."""

    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    ids = tuple(_safe_id(item, "trial_id") for item in trial_ids)
    if not ids or len(set(ids)) != len(ids):
        raise TrialConflictError(
            "invalid_halving_set",
            "successive halving requires distinct trial identities",
        )
    ranking = deterministic_rank(
        evidence,
        trial_ids=ids,
        round_index=_nonnegative_integer(round_index, "round_index"),
        policy=active_policy,
    )
    if final_round:
        survivor_count = 1
    else:
        factor = active_policy["successive_halving"]["reduction_factor"]
        minimum = active_policy["successive_halving"]["minimum_survivors"]
        survivor_count = max(minimum, int(math.ceil(len(ranking) / factor)))
    return ranking, ranking[:survivor_count]


@dataclass(frozen=True)
class AdaptiveEvent:
    schema_version: int
    contract: str
    sequence: int
    previous_event_hash: str
    timestamp_utc: str
    event_type: str
    operation_id: str
    policy_hash: str
    epoch_id: Optional[str]
    trial_id: Optional[str]
    payload: Mapping[str, Any]
    event_hash: str

    def body_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "epoch_id": self.epoch_id,
            "event_type": self.event_type,
            "operation_id": self.operation_id,
            "payload": _json_copy(self.payload),
            "policy_hash": self.policy_hash,
            "previous_event_hash": self.previous_event_hash,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "trial_id": self.trial_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        result = self.body_dict()
        result["event_hash"] = self.event_hash
        return result

    def verify(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.contract != EVENT_CONTRACT:
            raise StateCorruptionError(
                "unsupported_event_schema",
                "adaptive event schema or contract is unsupported",
            )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise StateCorruptionError(
                "invalid_event_sequence",
                "adaptive event sequence must be positive",
            )
        try:
            _require_hash(self.previous_event_hash, "previous_event_hash")
            _parse_utc_timestamp(self.timestamp_utc, "event.timestamp_utc")
            _safe_id(self.operation_id, "event.operation_id")
            _require_hash(self.policy_hash, "event.policy_hash")
            _require_hash(self.event_hash, "event.event_hash")
            if self.epoch_id is not None:
                _safe_id(self.epoch_id, "event.epoch_id")
            if self.trial_id is not None:
                _safe_id(self.trial_id, "event.trial_id")
        except AdaptiveTrainingError as exc:
            raise StateCorruptionError(
                "invalid_event",
                str(exc),
                details={"cause": exc.code},
            ) from exc
        if self.event_type not in _EVENT_TYPES:
            raise StateCorruptionError(
                "unknown_event_type",
                f"unknown adaptive event type: {self.event_type!r}",
            )
        if self.policy_hash != POLICY_HASH:
            raise StateCorruptionError(
                "event_policy_mismatch",
                "adaptive event is bound to a different policy",
            )
        if not isinstance(self.payload, Mapping):
            raise StateCorruptionError(
                "invalid_event_payload",
                "adaptive event payload must be an object",
            )
        expected = canonical_sha256(self.body_dict())
        if expected != self.event_hash:
            raise StateCorruptionError(
                "event_hash_mismatch",
                "adaptive event self-hash does not match its canonical body",
                details={"stored": self.event_hash, "computed": expected},
            )

    @classmethod
    def build(
        cls,
        *,
        sequence: int,
        previous_event_hash: str,
        event_type: str,
        operation_id: str,
        epoch_id: Optional[str],
        trial_id: Optional[str],
        payload: Mapping[str, Any],
        timestamp_utc: Optional[str] = None,
    ) -> "AdaptiveEvent":
        timestamp = timestamp_utc or utc_timestamp()
        body = {
            "contract": EVENT_CONTRACT,
            "epoch_id": epoch_id,
            "event_type": event_type,
            "operation_id": operation_id,
            "payload": _json_copy(payload),
            "policy_hash": POLICY_HASH,
            "previous_event_hash": previous_event_hash,
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "timestamp_utc": timestamp,
            "trial_id": trial_id,
        }
        event = cls(
            schema_version=SCHEMA_VERSION,
            contract=EVENT_CONTRACT,
            sequence=sequence,
            previous_event_hash=previous_event_hash,
            timestamp_utc=timestamp,
            event_type=event_type,
            operation_id=operation_id,
            policy_hash=POLICY_HASH,
            epoch_id=epoch_id,
            trial_id=trial_id,
            payload=body["payload"],
            event_hash=canonical_sha256(body),
        )
        event.verify()
        return event

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdaptiveEvent":
        fields = {
            "contract",
            "epoch_id",
            "event_hash",
            "event_type",
            "operation_id",
            "payload",
            "policy_hash",
            "previous_event_hash",
            "schema_version",
            "sequence",
            "timestamp_utc",
            "trial_id",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise StateCorruptionError(
                "invalid_event_fields",
                "adaptive event fields differ from the schema",
            )
        event = cls(**dict(value))
        event.verify()
        return event


class _ControllerLock:
    def __init__(self, path: Path, owner: str) -> None:
        self.path = path
        self.owner = owner
        self._fd: Optional[int] = None

    def acquire(self) -> "_ControllerLock":
        if self._fd is not None:
            return self
        _ensure_directory(self.path.parent)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(self.path), flags, 0o600)
        locked = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise TrialConflictError(
                    "invalid_controller_lock",
                    "adaptive controller lock is not a regular file",
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise TrialConflictError(
                    "adaptive_controller_locked",
                    "another adaptive-training controller owns the lock",
                    details={"path": str(self.path)},
                ) from exc
            metadata = _canonical_file_bytes(
                {
                    "actor": self.owner,
                    "acquired_at_utc": utc_timestamp(),
                    "pid": os.getpid(),
                }
            )
            os.ftruncate(descriptor, 0)
            offset = 0
            while offset < len(metadata):
                written = os.write(descriptor, metadata[offset:])
                if written <= 0:
                    raise OSError("short write while recording lock owner")
                offset += written
            os.fsync(descriptor)
            _fsync_directory(self.path.parent)
        except BaseException:
            if locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        self._fd = descriptor
        return self

    def release(self) -> None:
        descriptor = self._fd
        if descriptor is None:
            return
        self._fd = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "_ControllerLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _empty_state() -> Dict[str, Any]:
    return {
        "active_epoch_id": None,
        "active_trial_id": None,
        "contract": STATUS_CONTRACT,
        "epochs": {},
        "gpu_usage": [],
        "last_epoch_admitted_samples": None,
        "last_event_hash": GENESIS_HASH,
        "last_sequence": 0,
        "policy_hash": POLICY_HASH,
        "schema_version": SCHEMA_VERSION,
        "trials": {},
    }


def _state_error(message: str, event: AdaptiveEvent) -> None:
    raise StateCorruptionError(
        "illegal_event_transition",
        message,
        details={"event_sequence": event.sequence, "event_type": event.event_type},
    )


def _apply_event(state: Dict[str, Any], event: AdaptiveEvent) -> None:
    payload = event.payload
    event_type = event.event_type
    if event_type == "epoch.planned":
        if state["active_epoch_id"] is not None:
            _state_error(
                "a new epoch was planned while another epoch was active",
                event,
            )
        if event.epoch_id is None or event.trial_id is not None:
            _state_error("epoch.planned has invalid event identities", event)
        if event.epoch_id in state["epochs"]:
            _state_error("adaptive epoch identity was planned twice", event)
        trials = payload.get("trials")
        if not isinstance(trials, list) or not trials:
            _state_error("epoch.planned requires a non-empty trial list", event)
        admitted_samples = payload.get("admitted_samples")
        if (
            isinstance(admitted_samples, bool)
            or not isinstance(admitted_samples, int)
            or admitted_samples < 0
        ):
            _state_error("epoch admitted sample watermark is invalid", event)
        trial_ids: List[str] = []
        for definition in trials:
            if not isinstance(definition, Mapping):
                _state_error("epoch trial definition is not an object", event)
            trial_id = definition.get("trial_id")
            recipe_sha = definition.get("recipe_sha256")
            manifest_path = definition.get("manifest_path")
            try:
                _safe_id(trial_id, "trial_id")
                _require_hash(recipe_sha, "recipe_sha256")
            except AdaptiveTrainingError:
                _state_error("epoch trial definition identity is invalid", event)
            if not isinstance(manifest_path, str) or not manifest_path:
                _state_error("epoch trial manifest path is invalid", event)
            if trial_id in state["trials"] or trial_id in trial_ids:
                _state_error("epoch contains a duplicate trial identity", event)
            trial_ids.append(trial_id)
            state["trials"][trial_id] = {
                "epoch_id": event.epoch_id,
                "evidence": [],
                "gpu_usage": [],
                "handoff_path": None,
                "manifest_path": manifest_path,
                "recipe_sha256": recipe_sha,
                "reservation_gpu_seconds": None,
                "round_index": 0,
                "state": "ready",
            }
        state["epochs"][event.epoch_id] = {
            "admitted_samples": admitted_samples,
            "last_promotion_admitted_samples": payload.get(
                "last_promotion_admitted_samples"
            ),
            "state": "running",
            "survivor_trial_ids": trial_ids,
            "trial_ids": trial_ids,
            "winner_trial_id": None,
        }
        state["active_epoch_id"] = event.epoch_id
        state["last_epoch_admitted_samples"] = admitted_samples

    elif event_type == "trial.started":
        if event.trial_id is None or event.epoch_id is None:
            _state_error("trial.started requires epoch and trial identities", event)
        trial = state["trials"].get(event.trial_id)
        if trial is None or trial["epoch_id"] != event.epoch_id:
            _state_error("trial.started names an unknown trial", event)
        if state["active_trial_id"] is not None or trial["state"] != "ready":
            _state_error("trial.started violates single-trial concurrency", event)
        if payload.get("round_index") != trial["round_index"]:
            _state_error("trial.started round does not match trial state", event)
        reservation = payload.get("reservation_gpu_seconds")
        if (
            isinstance(reservation, bool)
            or not isinstance(reservation, (int, float))
            or not math.isfinite(float(reservation))
            or reservation <= 0
        ):
            _state_error("trial.started reservation is invalid", event)
        trial["state"] = "active"
        trial["reservation_gpu_seconds"] = float(reservation)
        state["active_trial_id"] = event.trial_id

    elif event_type == "trial.gpu_usage_recorded":
        if event.trial_id is None:
            _state_error("GPU usage event requires a trial identity", event)
        trial = state["trials"].get(event.trial_id)
        if trial is None or trial["state"] != "active":
            _state_error("GPU usage was recorded for a non-active trial", event)
        if payload.get("round_index") != trial["round_index"]:
            _state_error("GPU usage round does not match active trial", event)
        interval = payload.get("interval")
        if not isinstance(interval, Mapping):
            _state_error("GPU usage interval is malformed", event)
        try:
            normalized = _normalized_completed_interval(interval)
        except AdaptiveTrainingError:
            _state_error("GPU usage interval is invalid", event)
            return
        trial["gpu_usage"].append(
            {**normalized, "round_index": payload["round_index"]}
        )
        state["gpu_usage"].append(normalized)

    elif event_type == "trial.evidence_recorded":
        if event.trial_id is None:
            _state_error("evidence event requires a trial identity", event)
        trial = state["trials"].get(event.trial_id)
        if trial is None or trial["state"] != "active":
            _state_error("evidence was recorded for a non-active trial", event)
        evidence = payload.get("evidence")
        try:
            normalized = validate_evidence(
                evidence,
                expected_trial_id=event.trial_id,
                expected_round_index=trial["round_index"],
            )
        except AdaptiveTrainingError:
            _state_error("event contains invalid tuning evidence", event)
            return
        if any(
            item["source"] == normalized["source"]
            and item["round_index"] == normalized["round_index"]
            for item in trial["evidence"]
        ):
            _state_error("event duplicates evidence source for a trial round", event)
        trial["evidence"].append(normalized)

    elif event_type in {"trial.completed", "trial.failed"}:
        if event.trial_id is None:
            _state_error("terminal trial event requires a trial identity", event)
        trial = state["trials"].get(event.trial_id)
        if trial is None or trial["state"] != "active":
            _state_error("terminal event names a non-active trial", event)
        if payload.get("round_index") != trial["round_index"]:
            _state_error("terminal event round does not match active trial", event)
        if event_type == "trial.completed":
            sources = {
                item["source"]
                for item in trial["evidence"]
                if item["round_index"] == trial["round_index"]
            }
            if sources != {"discovery", "fixed_validation"}:
                _state_error(
                    "completed trial lacks both allowed evidence sources",
                    event,
                )
            trial["state"] = "complete"
        else:
            reason = payload.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                _state_error("failed trial requires a non-empty reason", event)
            trial["state"] = "failed"
        trial["reservation_gpu_seconds"] = None
        state["active_trial_id"] = None

    elif event_type == "round.halved":
        if event.epoch_id is None or event.trial_id is not None:
            _state_error("round.halved has invalid event identities", event)
        epoch = state["epochs"].get(event.epoch_id)
        if epoch is None or state["active_epoch_id"] != event.epoch_id:
            _state_error("round.halved names a non-active epoch", event)
        if state["active_trial_id"] is not None:
            _state_error("round.halved occurred while a trial was active", event)
        round_index = payload.get("round_index")
        current_ids = list(epoch["survivor_trial_ids"])
        successful = [
            trial_id
            for trial_id in current_ids
            if state["trials"][trial_id]["state"] == "complete"
            and state["trials"][trial_id]["round_index"] == round_index
        ]
        unfinished = [
            trial_id
            for trial_id in current_ids
            if state["trials"][trial_id]["state"] not in {"complete", "failed"}
        ]
        if unfinished:
            _state_error("round.halved before all current trials terminated", event)
        evidence = [
            item
            for trial_id in successful
            for item in state["trials"][trial_id]["evidence"]
            if item["round_index"] == round_index
        ]
        expected_ranking: Tuple[str, ...]
        expected_survivors: Tuple[str, ...]
        if successful:
            expected_ranking, expected_survivors = deterministic_successive_halving(
                evidence,
                trial_ids=successful,
                round_index=round_index,
                final_round=payload.get("next_round_index") is None,
            )
        else:
            expected_ranking, expected_survivors = (), ()
        if payload.get("ranking") != list(expected_ranking) or payload.get(
            "survivors"
        ) != list(expected_survivors):
            _state_error("persisted halving result is not deterministic", event)
        for trial_id in successful:
            if trial_id not in expected_survivors:
                state["trials"][trial_id]["state"] = "eliminated"
        next_round = payload.get("next_round_index")
        epoch["survivor_trial_ids"] = list(expected_survivors)
        if not expected_survivors:
            epoch["state"] = "failed"
            state["active_epoch_id"] = None
        elif next_round is None:
            winner = expected_survivors[0]
            state["trials"][winner]["state"] = "winner"
            epoch["winner_trial_id"] = winner
            epoch["state"] = "winner_selected"
        else:
            if next_round != round_index + 1:
                _state_error("halving next round is not consecutive", event)
            for trial_id in expected_survivors:
                trial = state["trials"][trial_id]
                trial["round_index"] = next_round
                trial["state"] = "ready"
            epoch["state"] = "running"

    elif event_type == "candidate.handoff_created":
        if event.trial_id is None or event.epoch_id is None:
            _state_error("handoff event requires trial and epoch identities", event)
        trial = state["trials"].get(event.trial_id)
        epoch = state["epochs"].get(event.epoch_id)
        if (
            trial is None
            or epoch is None
            or trial["state"] != "winner"
            or epoch["winner_trial_id"] != event.trial_id
        ):
            _state_error("handoff event names a non-winning trial", event)
        path = payload.get("handoff_path")
        handoff_sha = payload.get("handoff_sha256")
        if not isinstance(path, str) or not path:
            _state_error("handoff path is invalid", event)
        try:
            _require_hash(handoff_sha, "handoff_sha256")
        except AdaptiveTrainingError:
            _state_error("handoff hash is invalid", event)
        trial["handoff_path"] = path
        trial["state"] = "handed_off"
        epoch["state"] = "handed_off"
        state["active_epoch_id"] = None

    state["last_sequence"] = event.sequence
    state["last_event_hash"] = event.event_hash


def _finalize_status(state: Mapping[str, Any]) -> Dict[str, Any]:
    body = _json_copy(state)
    body["epochs"] = {
        key: body["epochs"][key] for key in sorted(body["epochs"])
    }
    body["trials"] = {
        key: body["trials"][key] for key in sorted(body["trials"])
    }
    body["status_sha256"] = canonical_sha256(body)
    return body


def _normalized_completed_interval(
    interval: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = {"ended_at", "gpu_count", "started_at", "trial_id"}
    if set(interval) != expected:
        raise AdaptiveTrainingError(
            "invalid_gpu_interval",
            "persisted GPU interval fields differ from the schema",
        )
    if interval.get("ended_at") is None:
        raise AdaptiveTrainingError(
            "invalid_gpu_interval",
            "persisted GPU usage must have a completed end time",
        )
    item = _coerce_interval(interval)
    started, ended, gpu_count = item.normalized(interval["ended_at"])
    trial_id = _safe_id(interval.get("trial_id"), "interval.trial_id")
    return {
        "ended_at": ended,
        "gpu_count": gpu_count,
        "started_at": started,
        "trial_id": trial_id,
    }


class AdaptiveTrainingStore:
    """Durable single-writer adaptive-training planner and event registry."""

    def __init__(
        self,
        root: PathLike,
        *,
        policy_path: PathLike = DEFAULT_POLICY_PATH,
        actor: str = "adaptive-training-controller",
    ) -> None:
        self.root = Path(root).absolute()
        self.policy_path = Path(policy_path)
        self.policy = load_policy(self.policy_path)
        if not isinstance(actor, str) or not actor.strip():
            raise AdaptiveTrainingError(
                "invalid_actor",
                "adaptive-training actor must be a non-empty string",
            )
        self.actor = actor
        self.events_dir = self.root / "events"
        self.epochs_dir = self.root / "epochs"
        self.trials_dir = self.root / "trials"
        self.recipes_dir = self.root / "recipes"
        self.handoffs_dir = self.root / "handoffs"
        self.candidate_handoffs_dir = self.handoffs_dir / "by-candidate"
        self.status_path = self.root / "status.json"
        self.lock_path = self.root / "controller.lock"

    def _ensure_layout(self) -> None:
        for directory in (
            self.root,
            self.events_dir,
            self.epochs_dir,
            self.trials_dir,
            self.recipes_dir,
            self.handoffs_dir,
            self.candidate_handoffs_dir,
        ):
            _ensure_directory(directory)

    def lock(self) -> _ControllerLock:
        self._ensure_layout()
        return _ControllerLock(self.lock_path, self.actor)

    def _event_paths(self) -> Tuple[Tuple[int, Path], ...]:
        if not self.events_dir.exists():
            return ()
        if self.events_dir.is_symlink() or not self.events_dir.is_dir():
            raise StateCorruptionError(
                "invalid_events_directory",
                "adaptive events path is not a real directory",
            )
        paths: List[Tuple[int, Path]] = []
        with os.scandir(self.events_dir) as entries:
            for entry in entries:
                match = _EVENT_FILE_RE.fullmatch(entry.name)
                if match is not None:
                    paths.append((int(match.group(1)), Path(entry.path)))
                elif entry.name.startswith(".") or entry.name.endswith(".tmp"):
                    continue
                elif entry.name.endswith(".json"):
                    raise StateCorruptionError(
                        "malformed_event_filename",
                        f"malformed adaptive event filename: {entry.name}",
                    )
        return tuple(sorted(paths))

    def events(self) -> Tuple[AdaptiveEvent, ...]:
        result: List[AdaptiveEvent] = []
        expected_sequence = 1
        previous_hash = GENESIS_HASH
        seen_operations: Dict[str, AdaptiveEvent] = {}
        for filename_sequence, path in self._event_paths():
            if filename_sequence != expected_sequence:
                raise StateCorruptionError(
                    "event_sequence_gap",
                    f"expected event {expected_sequence}, found {filename_sequence}",
                )
            value = load_canonical_json(path, "adaptive event")
            event = AdaptiveEvent.from_dict(value)
            if event.sequence != filename_sequence:
                raise StateCorruptionError(
                    "event_filename_mismatch",
                    "adaptive event filename and payload sequences differ",
                )
            if event.previous_event_hash != previous_hash:
                raise StateCorruptionError(
                    "event_chain_mismatch",
                    "adaptive event previous hash does not match the chain",
                )
            if event.operation_id in seen_operations:
                raise StateCorruptionError(
                    "duplicate_event_operation",
                    "adaptive event operation identifier was reused",
                )
            seen_operations[event.operation_id] = event
            result.append(event)
            previous_hash = event.event_hash
            expected_sequence += 1
        return tuple(result)

    def reconstruct(self) -> Dict[str, Any]:
        state = _empty_state()
        for event in self.events():
            _apply_event(state, event)
        return _finalize_status(state)

    def status(self) -> Dict[str, Any]:
        return self.reconstruct()

    def reconcile(self) -> Dict[str, Any]:
        """Rebuild and atomically repair the mutable status projection."""

        with self.lock():
            status = self.reconstruct()
            atomic_write_json(self.status_path, status)
            return status

    def _existing_operation(self, operation_id: str) -> Optional[AdaptiveEvent]:
        return next(
            (event for event in self.events() if event.operation_id == operation_id),
            None,
        )

    def _append_event(
        self,
        event_type: str,
        *,
        epoch_id: Optional[str],
        trial_id: Optional[str],
        payload: Mapping[str, Any],
        operation_identity: Mapping[str, Any],
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        operation_id = "op-" + canonical_sha256(
            {
                "event_type": event_type,
                "identity": _json_copy(operation_identity),
                "policy_hash": POLICY_HASH,
            }
        )
        existing = self._existing_operation(operation_id)
        if existing is not None:
            if (
                existing.event_type != event_type
                or existing.epoch_id != epoch_id
                or existing.trial_id != trial_id
                or existing.payload != payload
            ):
                raise StateCorruptionError(
                    "idempotent_event_conflict",
                    "event replay changed immutable operation metadata",
                )
            atomic_write_json(self.status_path, self.reconstruct())
            return existing
        events = self.events()
        sequence = len(events) + 1
        previous_hash = events[-1].event_hash if events else GENESIS_HASH
        event = AdaptiveEvent.build(
            sequence=sequence,
            previous_event_hash=previous_hash,
            event_type=event_type,
            operation_id=operation_id,
            epoch_id=epoch_id,
            trial_id=trial_id,
            payload=payload,
            timestamp_utc=timestamp_utc,
        )
        state = _empty_state()
        for prior in events:
            _apply_event(state, prior)
        _apply_event(state, event)
        self._ensure_layout()
        destination = self.events_dir / f"{sequence:020d}.json"
        atomic_create_json(destination, event.to_dict())
        atomic_write_json(self.status_path, _finalize_status(state))
        return event

    def _binding(
        self,
        *,
        path: Optional[PathLike],
        supplied_hash: Optional[str],
        role: str,
    ) -> Dict[str, Any]:
        if path is None and supplied_hash is None:
            raise AdaptiveTrainingError(
                "missing_immutable_binding",
                f"{role} requires a path or SHA-256",
            )
        normalized_path: Optional[str] = None
        observed_hash: Optional[str] = None
        if path is not None:
            source = Path(path).absolute()
            observed_hash = _stable_file_sha256(source)
            normalized_path = str(source)
        if supplied_hash is not None:
            supplied_hash = _require_hash(supplied_hash, f"{role}_sha256")
        if (
            observed_hash is not None
            and supplied_hash is not None
            and observed_hash != supplied_hash
        ):
            raise AdaptiveTrainingError(
                "immutable_binding_hash_mismatch",
                f"{role} content does not match its supplied SHA-256",
                details={"expected": supplied_hash, "actual": observed_hash},
            )
        return {
            "path": normalized_path,
            "sha256": observed_hash or supplied_hash,
        }

    def _store_recipe(self, recipe: Mapping[str, Any]) -> Tuple[str, Path]:
        normalized = validate_recipe(recipe, self.policy)
        digest = canonical_sha256(normalized)
        value = {
            "contract": RECIPE_CONTRACT,
            "policy_hash": POLICY_HASH,
            "recipe": normalized,
            "recipe_sha256": digest,
            "schema_version": SCHEMA_VERSION,
        }
        destination = self.recipes_dir / f"{digest}.json"
        atomic_create_json(destination, value)
        return digest, destination

    def plan_epoch(
        self,
        *,
        admitted_samples: int,
        last_promotion_admitted_samples: int,
        candidate_queue_depth: int,
        parent_champion_model_sha256: str,
        champion_checkpoint_path: Optional[PathLike] = None,
        champion_checkpoint_sha256: Optional[str] = None,
        admitted_data_manifest_path: Optional[PathLike] = None,
        admitted_data_manifest_sha256: Optional[str] = None,
        now: Union[float, str, datetime_module.datetime] = 0.0,
        timestamp_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Plan one deterministic epoch if every frozen trigger is satisfied."""

        with self.lock():
            state = self.reconstruct()
            parent_champion_model_sha256 = _require_hash(
                parent_champion_model_sha256,
                "parent_champion_model_sha256",
            )
            decision = evaluate_trigger(
                admitted_samples=admitted_samples,
                last_promotion_admitted_samples=last_promotion_admitted_samples,
                last_trial_epoch_admitted_samples=state[
                    "last_epoch_admitted_samples"
                ],
                candidate_queue_depth=candidate_queue_depth,
                active_trial_count=1 if state["active_trial_id"] is not None else 0,
                gpu_intervals=state["gpu_usage"],
                now=now,
                policy=self.policy,
            )
            if state["active_epoch_id"] is not None:
                active_epoch_id = state["active_epoch_id"]
                planned_event = next(
                    (
                        event
                        for event in self.events()
                        if event.event_type == "epoch.planned"
                        and event.epoch_id == active_epoch_id
                    ),
                    None,
                )
                if (
                    planned_event is not None
                    and planned_event.payload.get("admitted_samples")
                    == admitted_samples
                    and planned_event.payload.get(
                        "last_promotion_admitted_samples"
                    )
                    == last_promotion_admitted_samples
                    and planned_event.payload.get(
                        "parent_champion_model_sha256"
                    )
                    == parent_champion_model_sha256
                ):
                    champion_retry = self._binding(
                        path=champion_checkpoint_path,
                        supplied_hash=champion_checkpoint_sha256,
                        role="champion_checkpoint",
                    )
                    data_retry = self._binding(
                        path=admitted_data_manifest_path,
                        supplied_hash=admitted_data_manifest_sha256,
                        role="admitted_data_manifest",
                    )
                    if (
                        planned_event.payload.get("champion_checkpoint")
                        == champion_retry
                        and planned_event.payload.get("admitted_data")
                        == data_retry
                    ):
                        atomic_write_json(self.status_path, self.reconstruct())
                        return {
                            "decision": decision.to_dict(),
                            "epoch_id": active_epoch_id,
                            "event_hash": planned_event.event_hash,
                            "planned": True,
                            "reused": True,
                            "trial_ids": [
                                item["trial_id"]
                                for item in planned_event.payload["trials"]
                            ],
                        }
                reasons = list(decision.reason_codes)
                if "ACTIVE_EPOCH_EXISTS" not in reasons:
                    reasons.append("ACTIVE_EPOCH_EXISTS")
                result = decision.to_dict()
                result["eligible"] = False
                result["reason_codes"] = reasons
                return {"decision": result, "planned": False}
            if not decision.eligible:
                return {"decision": decision.to_dict(), "planned": False}

            champion = self._binding(
                path=champion_checkpoint_path,
                supplied_hash=champion_checkpoint_sha256,
                role="champion_checkpoint",
            )
            admitted_data = self._binding(
                path=admitted_data_manifest_path,
                supplied_hash=admitted_data_manifest_sha256,
                role="admitted_data_manifest",
            )
            epoch_id = content_addressed_epoch_id(
                parent_champion_model_sha256=parent_champion_model_sha256,
                champion_checkpoint_sha256=champion["sha256"],
                admitted_data_manifest_sha256=admitted_data["sha256"],
                admitted_samples=admitted_samples,
                last_promotion_admitted_samples=last_promotion_admitted_samples,
            )
            recipes = select_epoch_recipes(epoch_id, self.policy)
            trial_definitions: List[Dict[str, Any]] = []
            epoch_directory = self.epochs_dir / epoch_id
            _ensure_directory(epoch_directory)
            for recipe in recipes:
                recipe_sha, recipe_path = self._store_recipe(recipe)
                trial_id = content_addressed_trial_id(
                    epoch_id=epoch_id,
                    recipe_sha256=recipe_sha,
                    parent_champion_model_sha256=parent_champion_model_sha256,
                    champion_checkpoint_sha256=champion["sha256"],
                    admitted_data_manifest_sha256=admitted_data["sha256"],
                )
                trial_directory = self.trials_dir / trial_id
                _ensure_directory(trial_directory)
                _ensure_directory(trial_directory / "evidence")
                manifest = {
                    "admitted_data": admitted_data,
                    "champion_checkpoint": champion,
                    "contract": TRIAL_CONTRACT,
                    "epoch_id": epoch_id,
                    "isolation_root": str(trial_directory),
                    "parent_champion_model_sha256":
                        parent_champion_model_sha256,
                    "policy_hash": POLICY_HASH,
                    "recipe_path": str(recipe_path),
                    "recipe_sha256": recipe_sha,
                    "schema_version": SCHEMA_VERSION,
                    "trial_id": trial_id,
                }
                manifest["manifest_sha256"] = canonical_sha256(manifest)
                manifest_path = trial_directory / "trial.json"
                atomic_create_json(manifest_path, manifest)
                trial_definitions.append(
                    {
                        "manifest_path": str(manifest_path),
                        "recipe_sha256": recipe_sha,
                        "trial_id": trial_id,
                    }
                )
            epoch_manifest = {
                "admitted_data": admitted_data,
                "admitted_samples": admitted_samples,
                "champion_checkpoint": champion,
                "contract": EPOCH_CONTRACT,
                "epoch_id": epoch_id,
                "last_promotion_admitted_samples": last_promotion_admitted_samples,
                "parent_champion_model_sha256":
                    parent_champion_model_sha256,
                "policy_hash": POLICY_HASH,
                "schema_version": SCHEMA_VERSION,
                "trials": trial_definitions,
            }
            epoch_manifest["manifest_sha256"] = canonical_sha256(epoch_manifest)
            epoch_path = epoch_directory / "epoch.json"
            atomic_create_json(epoch_path, epoch_manifest)
            payload = {
                "admitted_data": admitted_data,
                "admitted_samples": admitted_samples,
                "champion_checkpoint": champion,
                "epoch_manifest_path": str(epoch_path),
                "last_promotion_admitted_samples": last_promotion_admitted_samples,
                "parent_champion_model_sha256":
                    parent_champion_model_sha256,
                "trials": trial_definitions,
            }
            event = self._append_event(
                "epoch.planned",
                epoch_id=epoch_id,
                trial_id=None,
                payload=payload,
                operation_identity={"epoch_id": epoch_id},
                timestamp_utc=timestamp_utc,
            )
            return {
                "decision": decision.to_dict(),
                "epoch_id": epoch_id,
                "event_hash": event.event_hash,
                "planned": True,
                "trial_ids": [
                    definition["trial_id"] for definition in trial_definitions
                ],
            }

    plan_trial_epoch = plan_epoch

    def _trial_manifest(self, trial_id: str) -> Dict[str, Any]:
        trial_id = _safe_id(trial_id, "trial_id")
        path = self.trials_dir / trial_id / "trial.json"
        manifest = load_canonical_json(path, "adaptive trial manifest")
        stored_hash = manifest.get("manifest_sha256")
        body = dict(manifest)
        body.pop("manifest_sha256", None)
        if stored_hash != canonical_sha256(body):
            raise StateCorruptionError(
                "trial_manifest_hash_mismatch",
                "adaptive trial manifest self-hash is invalid",
            )
        if (
            manifest.get("contract") != TRIAL_CONTRACT
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("policy_hash") != POLICY_HASH
            or manifest.get("trial_id") != trial_id
            or manifest.get("isolation_root") != str(self.trials_dir / trial_id)
        ):
            raise StateCorruptionError(
                "trial_manifest_identity_mismatch",
                "adaptive trial manifest identity is invalid",
            )
        return manifest

    def verify_trial_bindings(self, trial_id: str) -> Dict[str, Any]:
        manifest = self._trial_manifest(trial_id)
        for key in ("champion_checkpoint", "admitted_data"):
            binding = manifest.get(key)
            if not isinstance(binding, Mapping):
                raise StateCorruptionError(
                    "invalid_trial_binding",
                    f"trial {key} binding is malformed",
                )
            expected = _require_hash(binding.get("sha256"), f"trial.{key}.sha256")
            path = binding.get("path")
            if path is not None:
                if not isinstance(path, str) or _stable_file_sha256(path) != expected:
                    raise TrialConflictError(
                        "immutable_trial_binding_changed",
                        f"trial {key} no longer matches its immutable binding",
                        details={"trial_id": trial_id, "path": path},
                    )
        recipe_path = manifest.get("recipe_path")
        recipe_sha = _require_hash(
            manifest.get("recipe_sha256"),
            "trial.recipe_sha256",
        )
        recipe_object = load_canonical_json(recipe_path, "adaptive recipe")
        if (
            recipe_object.get("contract") != RECIPE_CONTRACT
            or recipe_object.get("policy_hash") != POLICY_HASH
            or recipe_object.get("recipe_sha256") != recipe_sha
            or canonical_sha256(
                validate_recipe(recipe_object.get("recipe"), self.policy)
            )
            != recipe_sha
        ):
            raise TrialConflictError(
                "immutable_trial_recipe_changed",
                "trial recipe no longer matches its immutable binding",
            )
        return manifest

    def budget_status(
        self,
        *,
        now: Union[float, str, datetime_module.datetime],
        requested_gpu_seconds: float = 0.0,
    ) -> BudgetStatus:
        state = self.reconstruct()
        return gpu_budget_status(
            state["gpu_usage"],
            now=now,
            requested_gpu_seconds=requested_gpu_seconds,
            policy=self.policy,
        )

    def start_trial(
        self,
        trial_id: str,
        *,
        now: Union[float, str, datetime_module.datetime],
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        with self.lock():
            state = self.reconstruct()
            trial_id = _safe_id(trial_id, "trial_id")
            trial = state["trials"].get(trial_id)
            if trial is None:
                raise TrialConflictError(
                    "unknown_trial",
                    f"unknown adaptive trial: {trial_id}",
                )
            round_index = trial["round_index"]
            operation_identity = {
                "round_index": round_index,
                "trial_id": trial_id,
            }
            operation_id = "op-" + canonical_sha256(
                {
                    "event_type": "trial.started",
                    "identity": operation_identity,
                    "policy_hash": POLICY_HASH,
                }
            )
            existing = self._existing_operation(operation_id)
            if existing is not None:
                atomic_write_json(self.status_path, self.reconstruct())
                return existing
            if state["active_trial_id"] is not None:
                raise TrialConflictError(
                    "active_trial_exists",
                    "only one adaptive trial may run at a time",
                    details={"active_trial_id": state["active_trial_id"]},
                )
            if trial["state"] != "ready":
                raise TrialConflictError(
                    "trial_not_ready",
                    f"trial cannot start from state {trial['state']}",
                )
            self.verify_trial_bindings(trial_id)
            rounds = self.policy["successive_halving"]["round_gpu_seconds"]
            if round_index >= len(rounds):
                raise TrialConflictError(
                    "invalid_trial_round",
                    "trial round exceeds the frozen successive-halving schedule",
                )
            reservation = rounds[round_index]
            budget = gpu_budget_status(
                state["gpu_usage"],
                now=now,
                requested_gpu_seconds=reservation,
                policy=self.policy,
            )
            if not budget.allowed:
                raise BudgetExceededError(
                    "gpu_budget_exhausted",
                    "trial reservation would exceed the rolling seven-day GPU budget",
                    details=budget.to_dict(),
                )
            return self._append_event(
                "trial.started",
                epoch_id=trial["epoch_id"],
                trial_id=trial_id,
                payload={
                    "reservation_gpu_seconds": reservation,
                    "round_index": round_index,
                },
                operation_identity=operation_identity,
                timestamp_utc=timestamp_utc,
            )

    def record_gpu_usage(
        self,
        trial_id: str,
        *,
        started_at: Union[float, str, datetime_module.datetime],
        ended_at: Union[float, str, datetime_module.datetime],
        gpu_count: int = 1,
        now: Optional[Union[float, str, datetime_module.datetime]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        with self.lock():
            state = self.reconstruct()
            trial_id = _safe_id(trial_id, "trial_id")
            trial = state["trials"].get(trial_id)
            if trial is None:
                raise TrialConflictError("unknown_trial", "unknown adaptive trial")
            interval = _normalized_completed_interval(
                {
                    "ended_at": _epoch_seconds(ended_at, "GPU interval end"),
                    "gpu_count": gpu_count,
                    "started_at": _epoch_seconds(started_at, "GPU interval start"),
                    "trial_id": trial_id,
                }
            )
            operation_identity = {
                "interval": interval,
                "round_index": trial["round_index"],
                "trial_id": trial_id,
            }
            operation_id = "op-" + canonical_sha256(
                {
                    "event_type": "trial.gpu_usage_recorded",
                    "identity": operation_identity,
                    "policy_hash": POLICY_HASH,
                }
            )
            existing = self._existing_operation(operation_id)
            if existing is not None:
                atomic_write_json(self.status_path, self.reconstruct())
                return existing
            if trial["state"] != "active" or state["active_trial_id"] != trial_id:
                raise TrialConflictError(
                    "trial_not_active",
                    "GPU usage may be recorded only for the active trial",
                )
            usage_seconds = (
                interval["ended_at"] - interval["started_at"]
            ) * interval["gpu_count"]
            current_round_usage = sum(
                (item["ended_at"] - item["started_at"]) * item["gpu_count"]
                for item in trial["gpu_usage"]
                if item["round_index"] == trial["round_index"]
            )
            reservation = trial["reservation_gpu_seconds"]
            if current_round_usage + usage_seconds > reservation:
                raise BudgetExceededError(
                    "trial_reservation_exceeded",
                    "recorded GPU usage exceeds the trial round reservation",
                    details={
                        "recorded_gpu_seconds": current_round_usage,
                        "new_gpu_seconds": usage_seconds,
                        "reservation_gpu_seconds": reservation,
                    },
                )
            budget_now = interval["ended_at"] if now is None else now
            budget = gpu_budget_status(
                state["gpu_usage"],
                now=budget_now,
                requested_gpu_seconds=usage_seconds,
                policy=self.policy,
            )
            if not budget.allowed:
                raise BudgetExceededError(
                    "gpu_budget_exhausted",
                    "recorded usage would exceed the rolling seven-day GPU budget",
                    details=budget.to_dict(),
                )
            return self._append_event(
                "trial.gpu_usage_recorded",
                epoch_id=trial["epoch_id"],
                trial_id=trial_id,
                payload={
                    "interval": interval,
                    "round_index": trial["round_index"],
                },
                operation_identity=operation_identity,
                timestamp_utc=timestamp_utc,
            )

    def record_evidence(
        self,
        trial_id: str,
        evidence: Union[Mapping[str, Any], PathLike],
        *,
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        with self.lock():
            state = self.reconstruct()
            trial_id = _safe_id(trial_id, "trial_id")
            trial = state["trials"].get(trial_id)
            if trial is None:
                raise TrialConflictError("unknown_trial", "unknown adaptive trial")
            if isinstance(evidence, (str, os.PathLike)):
                raw = load_canonical_json(evidence, "adaptive tuning evidence")
            else:
                raw = evidence
            normalized = validate_evidence(
                raw,
                expected_trial_id=trial_id,
                expected_round_index=trial["round_index"],
                policy=self.policy,
            )
            receipt = {
                "contract": EVIDENCE_CONTRACT,
                "evidence": normalized,
                "policy_hash": POLICY_HASH,
                "schema_version": SCHEMA_VERSION,
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            receipt_path = (
                self.trials_dir
                / trial_id
                / "evidence"
                / (
                    f"round-{trial['round_index']:02d}-"
                    f"{normalized['source']}-{normalized['artifact_sha256']}.json"
                )
            )
            atomic_create_json(receipt_path, receipt)
            operation_identity = {
                "artifact_sha256": normalized["artifact_sha256"],
                "round_index": normalized["round_index"],
                "source": normalized["source"],
                "trial_id": trial_id,
            }
            operation_id = "op-" + canonical_sha256(
                {
                    "event_type": "trial.evidence_recorded",
                    "identity": operation_identity,
                    "policy_hash": POLICY_HASH,
                }
            )
            existing = self._existing_operation(operation_id)
            if existing is not None:
                atomic_write_json(self.status_path, self.reconstruct())
                return existing
            if trial["state"] != "active" or state["active_trial_id"] != trial_id:
                raise TrialConflictError(
                    "trial_not_active",
                    "evidence may be recorded only for the active trial",
                )
            return self._append_event(
                "trial.evidence_recorded",
                epoch_id=trial["epoch_id"],
                trial_id=trial_id,
                payload={
                    "evidence": normalized,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": receipt["receipt_sha256"],
                },
                operation_identity=operation_identity,
                timestamp_utc=timestamp_utc,
            )

    record_trial_evidence = record_evidence

    def complete_trial(
        self,
        trial_id: str,
        *,
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        with self.lock():
            state = self.reconstruct()
            trial_id = _safe_id(trial_id, "trial_id")
            trial = state["trials"].get(trial_id)
            if trial is None:
                raise TrialConflictError("unknown_trial", "unknown adaptive trial")
            operation_identity = {
                "round_index": trial["round_index"],
                "trial_id": trial_id,
            }
            operation_id = "op-" + canonical_sha256(
                {
                    "event_type": "trial.completed",
                    "identity": operation_identity,
                    "policy_hash": POLICY_HASH,
                }
            )
            existing = self._existing_operation(operation_id)
            if existing is not None:
                atomic_write_json(self.status_path, self.reconstruct())
                return existing
            sources = {
                item["source"]
                for item in trial["evidence"]
                if item["round_index"] == trial["round_index"]
            }
            if sources != {"discovery", "fixed_validation"}:
                raise TrialConflictError(
                    "incomplete_trial_evidence",
                    "trial completion requires discovery and fixed_validation evidence",
                    details={"sources": sorted(sources)},
                )
            return self._append_event(
                "trial.completed",
                epoch_id=trial["epoch_id"],
                trial_id=trial_id,
                payload={"round_index": trial["round_index"]},
                operation_identity=operation_identity,
                timestamp_utc=timestamp_utc,
            )

    def fail_trial(
        self,
        trial_id: str,
        *,
        reason: str,
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        if not isinstance(reason, str) or not reason.strip():
            raise TrialConflictError(
                "invalid_trial_failure",
                "trial failure reason must be non-empty",
            )
        with self.lock():
            state = self.reconstruct()
            trial_id = _safe_id(trial_id, "trial_id")
            trial = state["trials"].get(trial_id)
            if trial is None:
                raise TrialConflictError("unknown_trial", "unknown adaptive trial")
            return self._append_event(
                "trial.failed",
                epoch_id=trial["epoch_id"],
                trial_id=trial_id,
                payload={
                    "reason": reason.strip(),
                    "round_index": trial["round_index"],
                },
                operation_identity={
                    "round_index": trial["round_index"],
                    "trial_id": trial_id,
                },
                timestamp_utc=timestamp_utc,
            )

    def halve_round(
        self,
        epoch_id: str,
        *,
        round_index: int,
        timestamp_utc: Optional[str] = None,
    ) -> AdaptiveEvent:
        with self.lock():
            state = self.reconstruct()
            epoch_id = _safe_id(epoch_id, "epoch_id")
            round_index = _nonnegative_integer(round_index, "round_index")
            epoch = state["epochs"].get(epoch_id)
            if epoch is None or state["active_epoch_id"] != epoch_id:
                raise TrialConflictError(
                    "epoch_not_active",
                    "successive halving requires the active epoch",
                )
            current_ids = list(epoch["survivor_trial_ids"])
            unfinished = [
                trial_id
                for trial_id in current_ids
                if state["trials"][trial_id]["state"] not in {"complete", "failed"}
            ]
            if unfinished:
                raise TrialConflictError(
                    "round_incomplete",
                    "all current-round trials must terminate before halving",
                    details={"unfinished_trial_ids": unfinished},
                )
            successful = [
                trial_id
                for trial_id in current_ids
                if state["trials"][trial_id]["state"] == "complete"
            ]
            rounds = self.policy["successive_halving"]["round_gpu_seconds"]
            final_round = round_index + 1 >= len(rounds)
            if successful:
                evidence = [
                    item
                    for trial_id in successful
                    for item in state["trials"][trial_id]["evidence"]
                    if item["round_index"] == round_index
                ]
                ranking, survivors = deterministic_successive_halving(
                    evidence,
                    trial_ids=successful,
                    round_index=round_index,
                    policy=self.policy,
                    final_round=final_round,
                )
            else:
                ranking, survivors = (), ()
            next_round = None if final_round or len(survivors) <= 1 else round_index + 1
            if next_round is None and len(survivors) > 1:
                survivors = survivors[:1]
            payload = {
                "next_round_index": next_round,
                "ranking": list(ranking),
                "round_index": round_index,
                "survivors": list(survivors),
            }
            return self._append_event(
                "round.halved",
                epoch_id=epoch_id,
                trial_id=None,
                payload=payload,
                operation_identity={
                    "epoch_id": epoch_id,
                    "round_index": round_index,
                },
                timestamp_utc=timestamp_utc,
            )

    apply_successive_halving = halve_round

    def create_handoff(
        self,
        trial_id: str,
        *,
        candidate_path: PathLike,
        candidate_sha256: Optional[str] = None,
        candidate_checkpoint_path: Optional[PathLike] = None,
        candidate_checkpoint_sha256: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock():
            state = self.reconstruct()
            trial_id = _safe_id(trial_id, "trial_id")
            trial = state["trials"].get(trial_id)
            if trial is None or trial["state"] not in {"winner", "handed_off"}:
                raise TrialConflictError(
                    "trial_not_winner",
                    "only the deterministic epoch winner may be handed off",
                )
            manifest = self.verify_trial_bindings(trial_id)
            candidate = self._binding(
                path=candidate_path,
                supplied_hash=candidate_sha256,
                role="candidate",
            )
            candidate_checkpoint = (
                None
                if candidate_checkpoint_path is None
                and candidate_checkpoint_sha256 is None
                else self._binding(
                    path=candidate_checkpoint_path,
                    supplied_hash=candidate_checkpoint_sha256,
                    role="candidate_checkpoint",
                )
            )
            evidence = [
                item
                for item in trial["evidence"]
                if item["round_index"] == trial["round_index"]
            ]
            handoff = build_candidate_handoff(
                trial_manifest=manifest,
                candidate=candidate,
                candidate_checkpoint=candidate_checkpoint,
                evidence=evidence,
                round_index=trial["round_index"],
                policy=self.policy,
            )
            handoff_id = handoff["handoff_id"]
            destination = self.handoffs_dir / f"{handoff_id}.json"
            candidate_destination = (
                self.candidate_handoffs_dir
                / f"{handoff['candidate']['sha256']}.json"
            )
            trial_destination = self.trials_dir / trial_id / "candidate-handoff.json"
            atomic_create_json(destination, handoff)
            atomic_create_json(candidate_destination, handoff)
            atomic_create_json(trial_destination, handoff)
            event = self._append_event(
                "candidate.handoff_created",
                epoch_id=trial["epoch_id"],
                trial_id=trial_id,
                payload={
                    "handoff_path": str(destination),
                    "handoff_sha256": canonical_sha256(handoff),
                },
                operation_identity={"handoff_id": handoff_id},
                timestamp_utc=timestamp_utc,
            )
            return {
                "event_hash": event.event_hash,
                "handoff": handoff,
                "handoff_path": str(destination),
            }

    create_candidate_handoff = create_handoff


def build_candidate_handoff(
    *,
    trial_manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    evidence: Iterable[Mapping[str, Any]],
    round_index: int,
    candidate_checkpoint: Optional[Mapping[str, Any]] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a promotion-v3 intake manifest without performing promotion."""

    active_policy = load_policy() if policy is None else policy
    validate_policy(active_policy)
    if not isinstance(trial_manifest, Mapping):
        raise TrialConflictError(
            "invalid_trial_manifest",
            "trial manifest must be an object",
        )
    trial_id = _safe_id(trial_manifest.get("trial_id"), "trial_id")
    epoch_id = _safe_id(trial_manifest.get("epoch_id"), "epoch_id")
    recipe_sha = _require_hash(
        trial_manifest.get("recipe_sha256"),
        "recipe_sha256",
    )
    normalized_candidate = {
        "path": candidate.get("path"),
        "sha256": _require_hash(candidate.get("sha256"), "candidate.sha256"),
    }
    if not isinstance(normalized_candidate["path"], str) or not normalized_candidate[
        "path"
    ]:
        raise TrialConflictError(
            "invalid_candidate_binding",
            "candidate handoff path must be non-empty",
        )
    normalized_checkpoint = None
    if candidate_checkpoint is not None:
        normalized_checkpoint = {
            "path": candidate_checkpoint.get("path"),
            "sha256": _require_hash(
                candidate_checkpoint.get("sha256"),
                "candidate_checkpoint.sha256",
            ),
        }
        if normalized_checkpoint["path"] is not None and (
            not isinstance(normalized_checkpoint["path"], str)
            or not normalized_checkpoint["path"]
        ):
            raise TrialConflictError(
                "invalid_candidate_binding",
                "candidate checkpoint path must be non-empty when present",
            )
    normalized_evidence = [
        validate_evidence(
            item,
            expected_trial_id=trial_id,
            expected_round_index=round_index,
            policy=active_policy,
        )
        for item in evidence
    ]
    ranking = deterministic_rank(
        normalized_evidence,
        trial_ids=[trial_id],
        round_index=round_index,
        policy=active_policy,
    )
    if ranking != (trial_id,):
        raise TrialConflictError(
            "invalid_handoff_evidence",
            "candidate handoff does not have complete winner evidence",
        )
    identity = {
        "admitted_data_manifest_sha256": _require_hash(
            trial_manifest["admitted_data"]["sha256"],
            "admitted_data.sha256",
        ),
        "candidate_sha256": normalized_candidate["sha256"],
        "champion_checkpoint_sha256": _require_hash(
            trial_manifest["champion_checkpoint"]["sha256"],
            "champion_checkpoint.sha256",
        ),
        "contract": HANDOFF_CONTRACT,
        "epoch_id": epoch_id,
        "policy_hash": POLICY_HASH,
        "parent_champion_model_sha256": _require_hash(
            trial_manifest.get("parent_champion_model_sha256"),
            "parent_champion_model_sha256",
        ),
        "recipe_sha256": recipe_sha,
        "trial_id": trial_id,
    }
    if normalized_checkpoint is not None:
        identity["candidate_checkpoint_sha256"] = normalized_checkpoint["sha256"]
    handoff_id = "handoff-" + canonical_sha256(identity)
    value = {
        "candidate": normalized_candidate,
        "candidate_checkpoint": normalized_checkpoint,
        "contract": HANDOFF_CONTRACT,
        "direct_promotion_permitted": False,
        "epoch_id": epoch_id,
        "evidence": sorted(
            normalized_evidence,
            key=lambda item: (item["source"], item["artifact_sha256"]),
        ),
        "handoff_id": handoff_id,
        "parent_admitted_data": _json_copy(trial_manifest["admitted_data"]),
        "parent_champion_checkpoint": _json_copy(
            trial_manifest["champion_checkpoint"]
        ),
        "parent_champion_model_sha256":
            identity["parent_champion_model_sha256"],
        "policy_hash": POLICY_HASH,
        "promotion_path": {
            "policy_version": "risk-seeking-checkpoint-promotion-v3",
            "required_stages": ["confirmation", "canary", "audit"],
        },
        "recipe_path": trial_manifest.get("recipe_path"),
        "recipe_sha256": recipe_sha,
        "schema_version": SCHEMA_VERSION,
        "trial_id": trial_id,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def load_candidate_handoff(
    path: PathLike,
    *,
    expected_candidate_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    value = load_canonical_json(path, "adaptive candidate handoff")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != HANDOFF_CONTRACT
        or value.get("policy_hash") != POLICY_HASH
        or value.get("direct_promotion_permitted") is not False
        or value.get("promotion_path")
        != {
            "policy_version": "risk-seeking-checkpoint-promotion-v3",
            "required_stages": ["confirmation", "canary", "audit"],
        }
    ):
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive candidate handoff contract is invalid",
        )
    supplied_manifest_hash = value.get("manifest_sha256")
    body = dict(value)
    body.pop("manifest_sha256", None)
    if supplied_manifest_hash != canonical_sha256(body):
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive candidate handoff self-hash is invalid",
        )
    trial_id = _safe_id(value.get("trial_id"), "trial_id")
    epoch_id = _safe_id(value.get("epoch_id"), "epoch_id")
    candidate = value.get("candidate")
    checkpoint = value.get("candidate_checkpoint")
    parent_checkpoint = value.get("parent_champion_checkpoint")
    parent_data = value.get("parent_admitted_data")
    if any(
        not isinstance(binding, Mapping)
        for binding in (
            candidate,
            checkpoint,
            parent_checkpoint,
            parent_data,
        )
    ):
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive candidate handoff bindings are incomplete",
        )
    candidate_hash = _require_hash(
        candidate.get("sha256"), "candidate.sha256"
    )
    if (
        expected_candidate_sha256 is not None
        and candidate_hash
        != _require_hash(
            expected_candidate_sha256, "expected_candidate_sha256"
        )
    ):
        raise TrialConflictError(
            "candidate_handoff_mismatch",
            "adaptive handoff names a different candidate",
        )
    for role, binding in (
        ("candidate_checkpoint", checkpoint),
        ("parent_champion_checkpoint", parent_checkpoint),
        ("parent_admitted_data", parent_data),
    ):
        binding_path = binding.get("path")
        expected_hash = _require_hash(binding.get("sha256"), f"{role}.sha256")
        if (
            not isinstance(binding_path, str)
            or not binding_path
            or _stable_file_sha256(binding_path) != expected_hash
        ):
            raise TrialConflictError(
                "candidate_handoff_binding_changed",
                f"adaptive handoff {role} binding changed",
            )
    recipe_path = value.get("recipe_path")
    recipe_sha = _require_hash(value.get("recipe_sha256"), "recipe_sha256")
    if not isinstance(recipe_path, str) or not recipe_path:
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive handoff recipe path is invalid",
        )
    recipe_value = load_canonical_json(recipe_path, "adaptive recipe")
    if (
        recipe_value.get("contract") != RECIPE_CONTRACT
        or recipe_value.get("policy_hash") != POLICY_HASH
        or recipe_value.get("recipe_sha256") != recipe_sha
        or canonical_sha256(
            validate_recipe(recipe_value.get("recipe"))
        )
        != recipe_sha
    ):
        raise TrialConflictError(
            "candidate_handoff_binding_changed",
            "adaptive handoff recipe binding changed",
        )
    parent_model_hash = _require_hash(
        value.get("parent_champion_model_sha256"),
        "parent_champion_model_sha256",
    )
    raw_evidence = value.get("evidence")
    if (
        not isinstance(raw_evidence, list)
        or not raw_evidence
        or any(not isinstance(item, Mapping) for item in raw_evidence)
    ):
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive handoff evidence is missing",
        )
    rounds = {
        _nonnegative_integer(item.get("round_index"), "round_index")
        for item in raw_evidence
    }
    if len(rounds) != 1:
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive handoff evidence round is inconsistent",
        )
    round_index = next(iter(rounds))
    normalized_evidence = [
        validate_evidence(
            item,
            expected_trial_id=trial_id,
            expected_round_index=round_index,
        )
        for item in raw_evidence
    ]
    if {item["source"] for item in normalized_evidence} != {
        "discovery",
        "fixed_validation",
    }:
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive handoff has incomplete tuning evidence",
        )
    identity = {
        "admitted_data_manifest_sha256": parent_data["sha256"],
        "candidate_checkpoint_sha256": checkpoint["sha256"],
        "candidate_sha256": candidate_hash,
        "champion_checkpoint_sha256": parent_checkpoint["sha256"],
        "contract": HANDOFF_CONTRACT,
        "epoch_id": epoch_id,
        "parent_champion_model_sha256": parent_model_hash,
        "policy_hash": POLICY_HASH,
        "recipe_sha256": recipe_sha,
        "trial_id": trial_id,
    }
    if value.get("handoff_id") != "handoff-" + canonical_sha256(identity):
        raise TrialConflictError(
            "invalid_candidate_handoff",
            "adaptive handoff content identity is invalid",
        )
    return _json_copy(value)


_RECIPE_BINDING_BODY_FIELDS = {
    "activated_at_utc",
    "admitted_data_manifest_sha256",
    "champion_checkpoint_sha256",
    "champion_model_sha256",
    "contract",
    "data_watermark_sha256s",
    "generation_id",
    "previous_record_sha256",
    "recipe_path",
    "recipe_sha256",
    "rollback",
    "schema_version",
}


def _validate_watermarks(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise RecipeConflictError(
            "invalid_recipe_binding",
            "data watermark hashes must be a non-empty object",
        )
    result: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise RecipeConflictError(
                "invalid_recipe_binding",
                "data watermark names must be non-empty strings",
            )
        result[key] = _require_hash(item, f"data_watermark_sha256s.{key}")
    return {key: result[key] for key in sorted(result)}


def _build_recipe_binding(
    *,
    recipe_sha256: str,
    recipe_path: str,
    champion_model_sha256: str,
    champion_checkpoint_sha256: str,
    admitted_data_manifest_sha256: str,
    data_watermark_sha256s: Mapping[str, str],
    generation_id: str,
    previous_record_sha256: Optional[str],
    rollback: Optional[Mapping[str, Any]],
    activated_at_utc: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(recipe_path, str) or not recipe_path:
        raise RecipeConflictError(
            "invalid_recipe_binding",
            "recipe_path must be a non-empty string",
        )
    _safe_id(generation_id, "generation_id")
    timestamp = activated_at_utc or utc_timestamp()
    _parse_utc_timestamp(timestamp, "activated_at_utc")
    body = {
        "activated_at_utc": timestamp,
        "admitted_data_manifest_sha256": _require_hash(
            admitted_data_manifest_sha256,
            "admitted_data_manifest_sha256",
        ),
        "champion_checkpoint_sha256": _require_hash(
            champion_checkpoint_sha256,
            "champion_checkpoint_sha256",
        ),
        "champion_model_sha256": _require_hash(
            champion_model_sha256,
            "champion_model_sha256",
        ),
        "contract": RECIPE_BINDING_CONTRACT,
        "data_watermark_sha256s": _validate_watermarks(data_watermark_sha256s),
        "generation_id": generation_id,
        "previous_record_sha256": (
            None
            if previous_record_sha256 is None
            else _require_hash(previous_record_sha256, "previous_record_sha256")
        ),
        "recipe_path": recipe_path,
        "recipe_sha256": _require_hash(recipe_sha256, "recipe_sha256"),
        "rollback": None if rollback is None else _json_copy(rollback),
        "schema_version": SCHEMA_VERSION,
    }
    result = dict(body)
    result["record_sha256"] = canonical_sha256(body)
    return result


def _validate_recipe_binding(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != (
        _RECIPE_BINDING_BODY_FIELDS | {"record_sha256"}
    ):
        raise RecipeConflictError(
            "invalid_recipe_binding",
            "active recipe binding fields differ from the schema",
        )
    body = dict(value)
    record_hash = body.pop("record_sha256")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != RECIPE_BINDING_CONTRACT
        or record_hash != canonical_sha256(body)
    ):
        raise RecipeConflictError(
            "invalid_recipe_binding",
            "active recipe binding contract or self-hash is invalid",
        )
    _require_hash(value.get("recipe_sha256"), "recipe_sha256")
    _require_hash(value.get("champion_model_sha256"), "champion_model_sha256")
    _require_hash(
        value.get("champion_checkpoint_sha256"),
        "champion_checkpoint_sha256",
    )
    _require_hash(
        value.get("admitted_data_manifest_sha256"),
        "admitted_data_manifest_sha256",
    )
    _validate_watermarks(value.get("data_watermark_sha256s"))
    _parse_utc_timestamp(value.get("activated_at_utc"), "activated_at_utc")
    return _json_copy(value)


def load_recipe_binding(path: PathLike) -> Dict[str, Any]:
    try:
        return _validate_recipe_binding(
            load_canonical_json(path, "active recipe binding")
        )
    except RecipeConflictError:
        raise
    except AdaptiveTrainingError as exc:
        raise RecipeConflictError(
            "invalid_recipe_binding",
            str(exc),
            details={"cause": exc.code},
        ) from exc


def bootstrap_recipe_binding(
    path: PathLike,
    *,
    recipe_sha256: str,
    recipe_path: str,
    champion_model_sha256: str,
    champion_checkpoint_sha256: str,
    admitted_data_manifest_sha256: str,
    data_watermark_sha256s: Mapping[str, str],
    generation_id: str,
    activated_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    destination = Path(path)
    desired = _build_recipe_binding(
        recipe_sha256=recipe_sha256,
        recipe_path=recipe_path,
        champion_model_sha256=champion_model_sha256,
        champion_checkpoint_sha256=champion_checkpoint_sha256,
        admitted_data_manifest_sha256=admitted_data_manifest_sha256,
        data_watermark_sha256s=data_watermark_sha256s,
        generation_id=generation_id,
        previous_record_sha256=None,
        rollback=None,
        activated_at_utc=activated_at_utc,
    )
    _ensure_directory(destination.parent)
    with _ControllerLock(
        Path(str(destination) + ".lock"),
        "adaptive-recipe-bootstrap",
    ):
        if destination.exists():
            existing = load_recipe_binding(destination)
            comparable = dict(desired)
            comparable["activated_at_utc"] = existing["activated_at_utc"]
            comparable["record_sha256"] = existing["record_sha256"]
            existing_without_hash = dict(existing)
            existing_hash = existing_without_hash.pop("record_sha256")
            comparable_without_hash = dict(comparable)
            comparable_without_hash.pop("record_sha256")
            retry_conflicts = existing_without_hash != comparable_without_hash
            existing_hash_invalid = existing_hash != canonical_sha256(
                existing_without_hash
            )
            if retry_conflicts or existing_hash_invalid:
                raise RecipeConflictError(
                    "recipe_bootstrap_conflict",
                    "active recipe binding conflicts with bootstrap retry",
                )
            return existing
        atomic_create_json(destination, desired, mode=0o600)
        return desired


def build_rollback_metadata(binding: Mapping[str, Any]) -> Dict[str, Any]:
    current = _validate_recipe_binding(binding)
    body = {
        "contract": ROLLBACK_CONTRACT,
        "restore_admitted_data_manifest_sha256": current[
            "admitted_data_manifest_sha256"
        ],
        "restore_champion_checkpoint_sha256": current[
            "champion_checkpoint_sha256"
        ],
        "restore_champion_model_sha256": current["champion_model_sha256"],
        "restore_data_watermark_sha256s": current["data_watermark_sha256s"],
        "restore_generation_id": current["generation_id"],
        "restore_recipe_path": current["recipe_path"],
        "restore_recipe_sha256": current["recipe_sha256"],
        "schema_version": SCHEMA_VERSION,
        "source_record_sha256": current["record_sha256"],
    }
    value = dict(body)
    value["rollback_sha256"] = canonical_sha256(body)
    return value


def _validate_rollback_metadata(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "contract",
        "restore_admitted_data_manifest_sha256",
        "restore_champion_checkpoint_sha256",
        "restore_champion_model_sha256",
        "restore_data_watermark_sha256s",
        "restore_generation_id",
        "restore_recipe_path",
        "restore_recipe_sha256",
        "rollback_sha256",
        "schema_version",
        "source_record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RecipeConflictError(
            "invalid_rollback_metadata",
            "recipe rollback metadata fields differ from the schema",
        )
    body = dict(value)
    rollback_hash = body.pop("rollback_sha256")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != ROLLBACK_CONTRACT
        or rollback_hash != canonical_sha256(body)
    ):
        raise RecipeConflictError(
            "invalid_rollback_metadata",
            "recipe rollback metadata self-hash is invalid",
        )
    return _json_copy(value)


def compare_and_swap_recipe_binding(
    path: PathLike,
    *,
    expected_record_sha256: str,
    recipe_sha256: str,
    recipe_path: str,
    champion_model_sha256: str,
    champion_checkpoint_sha256: str,
    admitted_data_manifest_sha256: str,
    data_watermark_sha256s: Mapping[str, str],
    generation_id: str,
    activated_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically CAS recipe/model/checkpoint/data metadata as one projection."""

    destination = Path(path)
    expected = _require_hash(expected_record_sha256, "expected_record_sha256")
    _ensure_directory(destination.parent)
    with _ControllerLock(
        Path(str(destination) + ".lock"),
        "adaptive-recipe-cas",
    ):
        current = load_recipe_binding(destination)
        if current["record_sha256"] != expected:
            desired_identity = {
                "admitted_data_manifest_sha256": admitted_data_manifest_sha256,
                "champion_checkpoint_sha256": champion_checkpoint_sha256,
                "champion_model_sha256": champion_model_sha256,
                "data_watermark_sha256s": _json_copy(data_watermark_sha256s),
                "generation_id": generation_id,
                "recipe_path": recipe_path,
                "recipe_sha256": recipe_sha256,
            }
            if all(
                current.get(key) == value
                for key, value in desired_identity.items()
            ):
                return current
            raise RecipeConflictError(
                "stale_recipe_binding",
                "active recipe compare-and-swap record is stale",
                details={
                    "expected": expected,
                    "actual": current["record_sha256"],
                },
            )
        rollback = build_rollback_metadata(current)
        replacement = _build_recipe_binding(
            recipe_sha256=recipe_sha256,
            recipe_path=recipe_path,
            champion_model_sha256=champion_model_sha256,
            champion_checkpoint_sha256=champion_checkpoint_sha256,
            admitted_data_manifest_sha256=admitted_data_manifest_sha256,
            data_watermark_sha256s=data_watermark_sha256s,
            generation_id=generation_id,
            previous_record_sha256=current["record_sha256"],
            rollback=rollback,
            activated_at_utc=activated_at_utc,
        )
        atomic_write_json(destination, replacement)
        return replacement


def rollback_recipe_binding(
    path: PathLike,
    *,
    expected_record_sha256: str,
    rollback: Mapping[str, Any],
    activated_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    destination = Path(path)
    expected = _require_hash(expected_record_sha256, "expected_record_sha256")
    restore = _validate_rollback_metadata(rollback)
    with _ControllerLock(
        Path(str(destination) + ".lock"),
        "adaptive-recipe-rollback",
    ):
        current = load_recipe_binding(destination)
        if current["record_sha256"] != expected:
            raise RecipeConflictError(
                "stale_recipe_binding",
                "recipe rollback compare-and-swap record is stale",
            )
        if current.get("previous_record_sha256") != restore["source_record_sha256"]:
            raise RecipeConflictError(
                "rollback_lineage_mismatch",
                "recipe rollback metadata is not the current binding's predecessor",
            )
        replacement = _build_recipe_binding(
            recipe_sha256=restore["restore_recipe_sha256"],
            recipe_path=restore["restore_recipe_path"],
            champion_model_sha256=restore["restore_champion_model_sha256"],
            champion_checkpoint_sha256=restore[
                "restore_champion_checkpoint_sha256"
            ],
            admitted_data_manifest_sha256=restore[
                "restore_admitted_data_manifest_sha256"
            ],
            data_watermark_sha256s=restore[
                "restore_data_watermark_sha256s"
            ],
            generation_id=restore["restore_generation_id"],
            previous_record_sha256=current["record_sha256"],
            rollback=build_rollback_metadata(current),
            activated_at_utc=activated_at_utc,
        )
        atomic_write_json(destination, replacement)
        return replacement


def _raise_contract_error(
    error_type: type[AdaptiveTrainingError],
    code: str,
    message: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    raise error_type(code, message, details=details)


def _strict_absolute_path(
    value: Any,
    role: str,
    *,
    error_type: type[AdaptiveTrainingError],
    code: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        _raise_contract_error(
            error_type,
            code,
            f"{role} must be an absolute path string",
        )
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        _raise_contract_error(
            error_type,
            code,
            f"{role} must be lexically normalized and absolute",
            details={"path": str(path)},
        )
    for component in (path, *path.parents):
        if component.exists() and component.is_symlink():
            _raise_contract_error(
                error_type,
                code,
                f"{role} has a symlinked path component",
                details={"path": str(component)},
            )
    if require_file:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise error_type(
                code,
                f"{role} must be an existing regular file",
                details={"path": str(path)},
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _raise_contract_error(
                error_type,
                code,
                f"{role} must be an existing regular non-symlink file",
                details={"path": str(path)},
            )
    if require_directory and (
        not path.exists() or path.is_symlink() or not path.is_dir()
    ):
        _raise_contract_error(
            error_type,
            code,
            f"{role} must be an existing non-symlink directory",
            details={"path": str(path)},
        )
    if path.exists() and not require_file and not require_directory:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            _raise_contract_error(
                error_type,
                code,
                f"{role} must not be a symlink",
                details={"path": str(path)},
            )
    return path


def _publication_path(value: PathLike) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    return Path(os.path.abspath(os.fspath(requested)))


def _strict_argv(
    value: Any,
    role: str,
    *,
    template: bool,
) -> Tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            not isinstance(part, str)
            or not part
            or "\x00" in part
            or "\n" in part
            or "\r" in part
            for part in value
        )
    ):
        raise ServiceSpecError(
            "invalid_service_spec",
            f"{role} must be a nonempty JSON argv array",
        )
    result = tuple(value)
    if not template:
        return result
    formatter = string.Formatter()
    used: set[str] = set()
    for part in result:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise ServiceSpecError(
                "invalid_service_spec",
                f"{role} contains invalid formatting",
            ) from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                field_name not in _ALLOWED_TRIAL_TEMPLATE_FIELDS
                or format_spec
                or conversion is not None
            ):
                raise ServiceSpecError(
                    "invalid_service_spec",
                    f"{role} uses an unsupported placeholder",
                    details={"placeholder": field_name},
                )
            used.add(field_name)
    missing = _REQUIRED_TRIAL_TEMPLATE_FIELDS - used
    if missing:
        raise ServiceSpecError(
            "invalid_service_spec",
            f"{role} does not bind every required trial artifact",
            details={"missing_placeholders": sorted(missing)},
        )
    return result


@dataclass(frozen=True)
class AdaptiveTrainingServiceSpec:
    path: Path
    file_sha256: str
    spec_sha256: str
    root: Path
    autonomy_policy_path: Path
    autonomy_policy_sha256: str
    scheduler_directory: Path
    gpu7_id: str
    observation_path: Path
    trial_command_argv_template: Tuple[str, ...]
    gpu_lease_guardian_argv_prefix: Tuple[str, ...]
    poll_interval_seconds: float
    actor: str
    raw: Mapping[str, Any]

    @property
    def identity(self) -> str:
        return self.spec_sha256

    @property
    def guardian_argv_prefix(self) -> Tuple[str, ...]:
        return self.gpu_lease_guardian_argv_prefix

    @property
    def policy_path(self) -> Path:
        return self.autonomy_policy_path

    @property
    def policy_sha256(self) -> str:
        return self.autonomy_policy_sha256


AdaptiveServiceSpec = AdaptiveTrainingServiceSpec
ServiceSpec = AdaptiveTrainingServiceSpec


def load_adaptive_service_spec(
    path: PathLike,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> AdaptiveTrainingServiceSpec:
    """Load the canonical self-hashed unattended adaptive service contract."""

    source = _strict_absolute_path(
        _publication_path(path),
        "adaptive service specification",
        error_type=ServiceSpecError,
        code="invalid_service_spec",
        require_file=True,
    )
    try:
        raw = load_canonical_json(source, "adaptive service specification")
        if set(raw) != _SERVICE_SPEC_FIELDS:
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service specification fields differ from the schema",
                details={
                    "missing": sorted(_SERVICE_SPEC_FIELDS - set(raw)),
                    "extra": sorted(set(raw) - _SERVICE_SPEC_FIELDS),
                },
            )
        if (
            raw["schema_version"] != SCHEMA_VERSION
            or isinstance(raw["schema_version"], bool)
            or raw["contract"] != SERVICE_SPEC_CONTRACT
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service specification contract is unsupported",
            )
        body = dict(raw)
        supplied_hash = _require_hash(
            body.pop("spec_sha256"),
            "adaptive service specification identity",
        )
        if canonical_sha256(body) != supplied_hash:
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service specification self-hash is invalid",
            )
        if (
            expected_spec_sha256 is not None
            and _require_hash(
                expected_spec_sha256,
                "expected adaptive service specification identity",
            )
            != supplied_hash
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service specification identity is not expected",
            )
        root = _strict_absolute_path(
            raw["root"],
            "adaptive root",
            error_type=ServiceSpecError,
            code="invalid_service_spec",
        )
        if root.exists() and not root.is_dir():
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive root must be a directory or a future directory",
            )
        policy_path = _strict_absolute_path(
            raw["autonomy_policy_path"],
            "autonomy policy",
            error_type=ServiceSpecError,
            code="invalid_service_spec",
            require_file=True,
        )
        policy_identity = _require_hash(
            raw["autonomy_policy_sha256"],
            "autonomy_policy_sha256",
        )
        policy = load_policy(policy_path)
        if policy_identity != POLICY_HASH or canonical_sha256(policy) != policy_identity:
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service policy binding is not the frozen autonomy policy",
            )
        scheduler_directory = _strict_absolute_path(
            raw["scheduler_directory"],
            "cluster scheduler directory",
            error_type=ServiceSpecError,
            code="invalid_service_spec",
            require_directory=True,
        )
        if (
            scheduler_directory == root
            or scheduler_directory in root.parents
            or root in scheduler_directory.parents
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive root and scheduler directory must not overlap",
            )
        gpu7_id = raw["gpu7_id"]
        if gpu7_id != "7":
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive training must remain pinned to fixed GPU ID 7",
            )
        observation_path = _strict_absolute_path(
            raw["observation_path"],
            "adaptive observation path",
            error_type=ServiceSpecError,
            code="invalid_service_spec",
        )
        if observation_path.exists() and not observation_path.is_file():
            raise ServiceSpecError(
                "invalid_service_spec",
                "mutable observation path must be a regular file or absent",
            )
        if observation_path == source or observation_path == policy_path:
            raise ServiceSpecError(
                "invalid_service_spec",
                "mutable observation path overlaps an immutable service input",
            )
        if (
            observation_path == scheduler_directory
            or scheduler_directory in observation_path.parents
            or observation_path in scheduler_directory.parents
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "mutable observation path overlaps the scheduler directory",
            )
        if observation_path in {
            root / "controller.lock",
            root / "service-controller.lock",
            root / "service-status.json",
            root / "status.json",
        }:
            raise ServiceSpecError(
                "invalid_service_spec",
                "mutable observation path overlaps adaptive service state",
            )
        trial_template = _strict_argv(
            raw["trial_command_argv_template"],
            "trial command argv template",
            template=True,
        )
        guardian_prefix = _strict_argv(
            raw["gpu_lease_guardian_argv_prefix"],
            "GPU lease guardian argv prefix",
            template=False,
        )
        poll = _finite_number(raw["poll_interval_seconds"], "poll_interval_seconds")
        if poll <= 0:
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service poll interval must be positive",
            )
        actor = _safe_id(raw["actor"], "adaptive service actor")
    except ServiceSpecError:
        raise
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise ServiceSpecError(
            "invalid_service_spec",
            str(exc),
            details={"cause": getattr(exc, "code", type(exc).__name__)},
        ) from exc
    return AdaptiveTrainingServiceSpec(
        path=source,
        file_sha256=file_sha256(source),
        spec_sha256=supplied_hash,
        root=root,
        autonomy_policy_path=policy_path,
        autonomy_policy_sha256=policy_identity,
        scheduler_directory=scheduler_directory,
        gpu7_id=gpu7_id,
        observation_path=observation_path,
        trial_command_argv_template=trial_template,
        gpu_lease_guardian_argv_prefix=guardian_prefix,
        poll_interval_seconds=poll,
        actor=actor,
        raw=_json_copy(raw),
    )


load_service_spec = load_adaptive_service_spec


def publish_adaptive_service_spec(
    path: PathLike,
    *,
    root: PathLike,
    autonomy_policy_path: PathLike,
    scheduler_directory: PathLike,
    observation_path: PathLike,
    trial_command_argv_template: Sequence[str],
    gpu_lease_guardian_argv_prefix: Sequence[str],
    poll_interval_seconds: float,
    actor: str,
    gpu7_id: str = "7",
) -> AdaptiveTrainingServiceSpec:
    destination = _publication_path(path)
    policy_source = _publication_path(autonomy_policy_path)
    scheduler_source = _publication_path(scheduler_directory)
    value: Dict[str, Any] = {
        "actor": actor,
        "autonomy_policy_path": str(policy_source),
        "autonomy_policy_sha256": POLICY_HASH,
        "contract": SERVICE_SPEC_CONTRACT,
        "gpu7_id": gpu7_id,
        "gpu_lease_guardian_argv_prefix": list(
            gpu_lease_guardian_argv_prefix
        ),
        "observation_path": str(_publication_path(observation_path)),
        "poll_interval_seconds": poll_interval_seconds,
        "root": str(_publication_path(root)),
        "scheduler_directory": str(scheduler_source),
        "schema_version": SCHEMA_VERSION,
        "trial_command_argv_template": list(trial_command_argv_template),
    }
    value["spec_sha256"] = canonical_sha256(value)
    _ensure_directory(destination.parent)
    atomic_create_json(destination, value)
    return load_adaptive_service_spec(destination)


def publish_service_spec(
    path: PathLike,
    *,
    root: PathLike,
    policy_path: PathLike,
    scheduler_directory: PathLike,
    observation_path: PathLike,
    trial_command_argv_template: Sequence[str],
    guardian_argv_prefix: Sequence[str],
    poll_interval_seconds: float,
    actor: str,
    gpu7_id: str = "7",
) -> AdaptiveTrainingServiceSpec:
    return publish_adaptive_service_spec(
        path,
        root=root,
        autonomy_policy_path=policy_path,
        scheduler_directory=scheduler_directory,
        observation_path=observation_path,
        trial_command_argv_template=trial_command_argv_template,
        gpu_lease_guardian_argv_prefix=guardian_argv_prefix,
        poll_interval_seconds=poll_interval_seconds,
        actor=actor,
        gpu7_id=gpu7_id,
    )


def _validated_file_binding(
    value: Any,
    role: str,
    *,
    resumable: bool,
    error_type: type[AdaptiveTrainingError],
    code: str,
) -> Dict[str, Any]:
    fields = (
        _RESUMABLE_FILE_BINDING_FIELDS
        if resumable
        else _FILE_BINDING_FIELDS
    )
    if not isinstance(value, Mapping) or set(value) != fields:
        _raise_contract_error(
            error_type,
            code,
            f"{role} fields differ from the schema",
        )
    if resumable and value.get("resumable") is not True:
        _raise_contract_error(
            error_type,
            code,
            f"{role} must be explicitly resumable",
        )
    try:
        expected_hash = _require_hash(value.get("sha256"), f"{role}.sha256")
        path = _strict_absolute_path(
            value.get("path"),
            role,
            error_type=error_type,
            code=code,
            require_file=True,
        )
        actual_hash = _stable_file_sha256(path)
    except error_type:
        raise
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise error_type(code, str(exc)) from exc
    if actual_hash != expected_hash:
        _raise_contract_error(
            error_type,
            code,
            f"{role} no longer matches its immutable SHA-256 binding",
            details={
                "path": str(path),
                "expected": expected_hash,
                "actual": actual_hash,
            },
        )
    result: Dict[str, Any] = {
        "path": str(path),
        "sha256": expected_hash,
    }
    if resumable:
        result["resumable"] = True
    return result


def load_adaptive_observation(
    path: PathLike,
    *,
    now: Optional[Union[float, str, datetime_module.datetime]] = None,
    max_age_seconds: Optional[float] = None,
    expected_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Load and validate one canonical mutable producer observation."""

    source = _strict_absolute_path(
        _publication_path(path),
        "adaptive observation",
        error_type=ObservationValidationError,
        code="invalid_observation",
        require_file=True,
    )
    if expected_path is not None and source != _publication_path(expected_path):
        raise ObservationValidationError(
            "invalid_observation",
            "adaptive observation was loaded from an unexpected path",
        )
    try:
        raw = load_canonical_json(source, "adaptive training observation")
        if set(raw) != _OBSERVATION_FIELDS:
            raise ObservationValidationError(
                "invalid_observation",
                "adaptive observation fields differ from the schema",
                details={
                    "missing": sorted(_OBSERVATION_FIELDS - set(raw)),
                    "extra": sorted(set(raw) - _OBSERVATION_FIELDS),
                },
            )
        body = dict(raw)
        supplied_hash = _require_hash(
            body.pop("observation_sha256"),
            "adaptive observation identity",
        )
        if (
            raw["schema_version"] != SCHEMA_VERSION
            or isinstance(raw["schema_version"], bool)
            or raw["contract"] != OBSERVATION_CONTRACT
            or canonical_sha256(body) != supplied_hash
        ):
            raise ObservationValidationError(
                "invalid_observation",
                "adaptive observation contract or self-hash is invalid",
            )
        admitted_samples = _nonnegative_integer(
            raw["admitted_samples"], "admitted_samples"
        )
        promoted_samples = _nonnegative_integer(
            raw["last_promotion_admitted_samples"],
            "last_promotion_admitted_samples",
        )
        queue_depth = _nonnegative_integer(
            raw["candidate_queue_depth"], "candidate_queue_depth"
        )
        if promoted_samples > admitted_samples:
            raise ObservationValidationError(
                "invalid_observation",
                "last-promotion watermark exceeds admitted samples",
            )
        champion_model_hash = _require_hash(
            raw["current_champion_model_sha256"],
            "current_champion_model_sha256",
        )
        champion_checkpoint = _validated_file_binding(
            raw["champion_checkpoint"],
            "champion checkpoint",
            resumable=True,
            error_type=ObservationValidationError,
            code="invalid_observation_binding",
        )
        admitted_data = _validated_file_binding(
            raw["admitted_data_manifest"],
            "admitted-data manifest",
            resumable=False,
            error_type=ObservationValidationError,
            code="invalid_observation_binding",
        )
        updated_at = _finite_number(raw["updated_at_unix"], "updated_at_unix")
        if updated_at < 0:
            raise ObservationValidationError(
                "invalid_observation",
                "adaptive observation update time must be nonnegative",
            )
        if now is not None:
            observed_now = _epoch_seconds(now, "observation validation time")
            if updated_at > observed_now + 1e-6:
                raise ObservationValidationError(
                    "observation_from_future",
                    "adaptive observation update time is in the future",
                )
            if max_age_seconds is not None:
                maximum_age = _finite_number(
                    max_age_seconds, "observation maximum age"
                )
                if maximum_age < 0:
                    raise ObservationValidationError(
                        "invalid_observation",
                        "observation maximum age must be nonnegative",
                    )
                if observed_now - updated_at > maximum_age:
                    raise ObservationValidationError(
                        "stale_observation",
                        "adaptive observation is stale",
                        details={
                            "age_seconds": observed_now - updated_at,
                            "maximum_age_seconds": maximum_age,
                        },
                    )
    except ObservationValidationError:
        raise
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise ObservationValidationError(
            "invalid_observation",
            str(exc),
            details={"cause": getattr(exc, "code", type(exc).__name__)},
        ) from exc
    return {
        "admitted_data_manifest": admitted_data,
        "admitted_samples": admitted_samples,
        "candidate_queue_depth": queue_depth,
        "champion_checkpoint": champion_checkpoint,
        "contract": OBSERVATION_CONTRACT,
        "current_champion_model_sha256": champion_model_hash,
        "last_promotion_admitted_samples": promoted_samples,
        "observation_sha256": supplied_hash,
        "schema_version": SCHEMA_VERSION,
        "updated_at_unix": updated_at,
    }


load_observation = load_adaptive_observation


def publish_adaptive_observation(
    path: PathLike,
    *,
    admitted_samples: int,
    last_promotion_admitted_samples: int,
    candidate_queue_depth: int,
    current_champion_model_sha256: str,
    champion_checkpoint_path: PathLike,
    admitted_data_manifest_path: PathLike,
    updated_at_unix: Union[float, int],
) -> Dict[str, Any]:
    destination = _publication_path(path)
    checkpoint = _publication_path(champion_checkpoint_path)
    admitted_data = _publication_path(admitted_data_manifest_path)
    value: Dict[str, Any] = {
        "admitted_data_manifest": {
            "path": str(admitted_data),
            "sha256": _stable_file_sha256(admitted_data),
        },
        "admitted_samples": admitted_samples,
        "candidate_queue_depth": candidate_queue_depth,
        "champion_checkpoint": {
            "path": str(checkpoint),
            "resumable": True,
            "sha256": _stable_file_sha256(checkpoint),
        },
        "contract": OBSERVATION_CONTRACT,
        "current_champion_model_sha256": current_champion_model_sha256,
        "last_promotion_admitted_samples": (
            last_promotion_admitted_samples
        ),
        "schema_version": SCHEMA_VERSION,
        "updated_at_unix": updated_at_unix,
    }
    value["observation_sha256"] = canonical_sha256(value)
    atomic_write_json(destination, value)
    return load_adaptive_observation(destination)


publish_observation = publish_adaptive_observation


def _load_trial_manifest_binding(path: Path) -> Dict[str, Any]:
    try:
        manifest = load_canonical_json(path, "adaptive trial manifest")
        body = dict(manifest)
        supplied_hash = _require_hash(
            body.pop("manifest_sha256", None),
            "trial manifest identity",
        )
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("contract") != TRIAL_CONTRACT
            or manifest.get("policy_hash") != POLICY_HASH
            or canonical_sha256(body) != supplied_hash
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result names an invalid trial manifest",
            )
        return manifest
    except TrialResultValidationError:
        raise
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise TrialResultValidationError(
            "invalid_trial_result",
            f"trial result manifest binding is invalid: {exc}",
        ) from exc


def load_trial_result(
    path: PathLike,
    *,
    expected_trial_id: Optional[str] = None,
    expected_epoch_id: Optional[str] = None,
    expected_round_index: Optional[int] = None,
    expected_work_id: Optional[str] = None,
    expected_gpu_id: Optional[str] = None,
    expected_manifest_path: Optional[PathLike] = None,
    expected_manifest_sha256: Optional[str] = None,
    require_candidate_bindings: bool = False,
    now: Optional[Union[float, str, datetime_module.datetime]] = None,
) -> Dict[str, Any]:
    """Validate one immutable scheduler-produced trial result."""

    source = _strict_absolute_path(
        _publication_path(path),
        "adaptive trial result",
        error_type=TrialResultValidationError,
        code="invalid_trial_result",
        require_file=True,
    )
    try:
        raw = load_canonical_json(source, "adaptive trial result")
        if set(raw) != _TRIAL_RESULT_FIELDS:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "adaptive trial result fields differ from the schema",
                details={
                    "missing": sorted(_TRIAL_RESULT_FIELDS - set(raw)),
                    "extra": sorted(set(raw) - _TRIAL_RESULT_FIELDS),
                },
            )
        body = dict(raw)
        result_hash = _require_hash(
            body.pop("result_sha256"),
            "adaptive trial result identity",
        )
        if (
            raw["schema_version"] != SCHEMA_VERSION
            or isinstance(raw["schema_version"], bool)
            or raw["contract"] != TRIAL_RESULT_CONTRACT
            or canonical_sha256(body) != result_hash
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "adaptive trial result contract or self-hash is invalid",
            )
        trial_id = _safe_id(raw["trial_id"], "trial result trial_id")
        epoch_id = _safe_id(raw["epoch_id"], "trial result epoch_id")
        work_id = _safe_id(raw["work_id"], "trial result work_id")
        round_index = _nonnegative_integer(
            raw["round_index"], "trial result round_index"
        )
        if expected_trial_id is not None and trial_id != expected_trial_id:
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result names a different trial",
            )
        if expected_epoch_id is not None and epoch_id != expected_epoch_id:
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result names a different epoch",
            )
        if (
            expected_round_index is not None
            and round_index != expected_round_index
        ):
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result names a different successive-halving round",
            )
        if expected_work_id is not None and work_id != expected_work_id:
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result names a different scheduler work item",
            )
        manifest_path = _strict_absolute_path(
            raw["trial_manifest_path"],
            "trial result manifest path",
            error_type=TrialResultValidationError,
            code="invalid_trial_result",
            require_file=True,
        )
        if (
            expected_manifest_path is not None
            and manifest_path != _publication_path(expected_manifest_path)
        ):
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result names a different trial manifest path",
            )
        manifest = _load_trial_manifest_binding(manifest_path)
        manifest_hash = _require_hash(
            raw["trial_manifest_sha256"],
            "trial result manifest SHA-256",
        )
        if (
            manifest["manifest_sha256"] != manifest_hash
            or manifest["trial_id"] != trial_id
            or manifest["epoch_id"] != epoch_id
        ):
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result manifest identity contradicts the result",
            )
        canonical_result_path = (
            Path(manifest["isolation_root"])
            / "results"
            / f"round-{round_index:02d}.json"
        )
        if source != canonical_result_path:
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result is not under its isolated trial directory",
            )
        if (
            expected_manifest_sha256 is not None
            and manifest_hash
            != _require_hash(
                expected_manifest_sha256,
                "expected trial manifest SHA-256",
            )
        ):
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result manifest identity is not expected",
            )
        gpu_usage = raw["gpu_usage"]
        if (
            not isinstance(gpu_usage, Mapping)
            or set(gpu_usage) != _GPU_USAGE_FIELDS
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result GPU usage fields differ from the schema",
            )
        gpu_id = gpu_usage["gpu_id"]
        if gpu_id != "7":
            raise TrialResultValidationError(
                "invalid_trial_result",
                "adaptive trial result must account for fixed GPU ID 7",
            )
        if expected_gpu_id is not None and gpu_id != expected_gpu_id:
            raise TrialResultValidationError(
                "trial_result_binding_mismatch",
                "trial result was produced on a different GPU",
            )
        if gpu_usage["gpu_count"] != 1 or isinstance(
            gpu_usage["gpu_count"], bool
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "adaptive trials must account for exactly one GPU",
            )
        started_at = _finite_number(
            gpu_usage["started_at_unix"],
            "trial result GPU start",
        )
        ended_at = _finite_number(
            gpu_usage["ended_at_unix"],
            "trial result GPU end",
        )
        if started_at < 0 or ended_at < started_at:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result GPU interval is invalid",
            )
        round_reservations = load_policy()["successive_halving"][
            "round_gpu_seconds"
        ]
        if round_index >= len(round_reservations):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result round exceeds the frozen halving schedule",
            )
        if ended_at - started_at > round_reservations[round_index]:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result GPU interval exceeds its frozen round reservation",
            )
        if now is not None and ended_at > _epoch_seconds(now, "result time") + 1e-6:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result GPU interval ends in the future",
            )
        status_value = raw["status"]
        if status_value not in {"completed", "failed"}:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result status must be completed or failed",
            )
        failure_reason = raw["failure_reason"]
        if status_value == "completed":
            if failure_reason is not None:
                raise TrialResultValidationError(
                    "invalid_trial_result",
                    "completed trial result must not contain a failure reason",
                )
        elif (
            not isinstance(failure_reason, str)
            or not failure_reason.strip()
            or failure_reason != failure_reason.strip()
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "failed trial result requires a nonempty failure reason",
            )
        raw_evidence = raw["evidence"]
        if not isinstance(raw_evidence, list) or any(
            not isinstance(item, Mapping) for item in raw_evidence
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result evidence must be an array of objects",
            )
        normalized_evidence = [
            validate_evidence(
                item,
                expected_trial_id=trial_id,
                expected_round_index=round_index,
            )
            for item in raw_evidence
        ]
        sources = [item["source"] for item in normalized_evidence]
        if len(sources) != len(set(sources)):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "trial result duplicates a tuning evidence source",
            )
        if status_value == "completed" and set(sources) != {
            "discovery",
            "fixed_validation",
        }:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "completed trial result requires finalized discovery and "
                "fixed-validation evidence",
            )
        if status_value == "failed" and sources:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "failed trial result must not publish ranking evidence",
            )
        candidate_model_value = raw["candidate_model"]
        candidate_checkpoint_value = raw["candidate_checkpoint"]
        if (candidate_model_value is None) != (
            candidate_checkpoint_value is None
        ):
            raise TrialResultValidationError(
                "invalid_trial_result",
                "candidate model and resumable checkpoint bindings are atomic",
            )
        candidate_model = None
        candidate_checkpoint = None
        if candidate_model_value is not None:
            candidate_model = _validated_file_binding(
                candidate_model_value,
                "candidate model",
                resumable=False,
                error_type=TrialResultValidationError,
                code="invalid_trial_result_binding",
            )
            candidate_checkpoint = _validated_file_binding(
                candidate_checkpoint_value,
                "candidate checkpoint",
                resumable=True,
                error_type=TrialResultValidationError,
                code="invalid_trial_result_binding",
            )
            isolation_root = Path(manifest["isolation_root"])
            for role, binding in (
                ("candidate model", candidate_model),
                ("candidate checkpoint", candidate_checkpoint),
            ):
                binding_path = Path(binding["path"])
                if (
                    binding_path.parent != isolation_root
                    and isolation_root not in binding_path.parents
                ):
                    raise TrialResultValidationError(
                        "invalid_trial_result_binding",
                        f"{role} is outside the isolated trial directory",
                    )
        if status_value == "failed" and candidate_model is not None:
            raise TrialResultValidationError(
                "invalid_trial_result",
                "failed trial result must not bind a candidate",
            )
        if require_candidate_bindings and (
            status_value != "completed" or candidate_model is None
        ):
            raise TrialResultValidationError(
                "missing_winner_bindings",
                "final winner result requires candidate model and resumable "
                "checkpoint bindings",
            )
    except (
        EvidenceRejectedError,
        TrialResultValidationError,
    ):
        raise
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise TrialResultValidationError(
            "invalid_trial_result",
            str(exc),
            details={"cause": getattr(exc, "code", type(exc).__name__)},
        ) from exc
    return {
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_model": candidate_model,
        "contract": TRIAL_RESULT_CONTRACT,
        "epoch_id": epoch_id,
        "evidence": sorted(
            normalized_evidence,
            key=lambda item: (item["source"], item["artifact_sha256"]),
        ),
        "failure_reason": failure_reason,
        "gpu_usage": {
            "ended_at_unix": ended_at,
            "gpu_count": 1,
            "gpu_id": gpu_id,
            "started_at_unix": started_at,
        },
        "result_sha256": result_hash,
        "round_index": round_index,
        "schema_version": SCHEMA_VERSION,
        "status": status_value,
        "trial_id": trial_id,
        "trial_manifest_path": str(manifest_path),
        "trial_manifest_sha256": manifest_hash,
        "work_id": work_id,
    }


def publish_trial_result(
    path: PathLike,
    *,
    trial_manifest_path: PathLike,
    work_id: str,
    round_index: int,
    gpu_id: str,
    started_at_unix: Union[int, float],
    ended_at_unix: Union[int, float],
    status: str,
    evidence: Iterable[Mapping[str, Any]] = (),
    failure_reason: Optional[str] = None,
    candidate_model_path: Optional[PathLike] = None,
    candidate_checkpoint_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    destination = _publication_path(path)
    manifest_path = _publication_path(trial_manifest_path)
    manifest = _load_trial_manifest_binding(manifest_path)
    candidate_model: Optional[Dict[str, Any]] = None
    candidate_checkpoint: Optional[Dict[str, Any]] = None
    if (candidate_model_path is None) != (candidate_checkpoint_path is None):
        raise TrialResultValidationError(
            "invalid_trial_result",
            "candidate model and checkpoint paths are required together",
        )
    if candidate_model_path is not None:
        model_path = _publication_path(candidate_model_path)
        checkpoint_path = _publication_path(candidate_checkpoint_path)
        candidate_model = {
            "path": str(model_path),
            "sha256": _stable_file_sha256(model_path),
        }
        candidate_checkpoint = {
            "path": str(checkpoint_path),
            "resumable": True,
            "sha256": _stable_file_sha256(checkpoint_path),
        }
    value: Dict[str, Any] = {
        "candidate_checkpoint": candidate_checkpoint,
        "candidate_model": candidate_model,
        "contract": TRIAL_RESULT_CONTRACT,
        "epoch_id": manifest["epoch_id"],
        "evidence": [_json_copy(item) for item in evidence],
        "failure_reason": failure_reason,
        "gpu_usage": {
            "ended_at_unix": ended_at_unix,
            "gpu_count": 1,
            "gpu_id": gpu_id,
            "started_at_unix": started_at_unix,
        },
        "round_index": round_index,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "trial_id": manifest["trial_id"],
        "trial_manifest_path": str(manifest_path),
        "trial_manifest_sha256": manifest["manifest_sha256"],
        "work_id": work_id,
    }
    value["result_sha256"] = canonical_sha256(value)
    _ensure_directory(destination.parent)
    atomic_create_json(destination, value)
    return load_trial_result(
        destination,
        expected_trial_id=manifest["trial_id"],
        expected_epoch_id=manifest["epoch_id"],
        expected_round_index=round_index,
        expected_work_id=work_id,
        expected_gpu_id=gpu_id,
        expected_manifest_path=manifest_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )


class AdaptiveTrainingService:
    """Unattended producer that bridges adaptive trials to ClusterScheduler."""

    def __init__(
        self,
        spec: Union[AdaptiveTrainingServiceSpec, PathLike],
        *,
        expected_spec_sha256: Optional[str] = None,
        scheduler: Optional[ClusterScheduler] = None,
        store: Optional[AdaptiveTrainingStore] = None,
        clock: Callable[[], Any] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = (
            spec
            if isinstance(spec, AdaptiveTrainingServiceSpec)
            else load_adaptive_service_spec(
                spec,
                expected_spec_sha256=expected_spec_sha256,
            )
        )
        if (
            isinstance(spec, AdaptiveTrainingServiceSpec)
            and expected_spec_sha256 is not None
            and self.spec.spec_sha256
            != _require_hash(
                expected_spec_sha256,
                "expected adaptive service specification identity",
            )
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service specification identity is not expected",
            )
        self._clock = clock
        self._sleeper = sleeper
        self.store = store or AdaptiveTrainingStore(
            self.spec.root,
            policy_path=self.spec.autonomy_policy_path,
            actor=self.spec.actor,
        )
        if (
            Path(self.store.root) != self.spec.root
            or Path(self.store.policy_path) != self.spec.autonomy_policy_path
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive store contradicts the service specification",
            )
        self.scheduler = scheduler or ClusterScheduler(
            self.spec.scheduler_directory,
            clock=self._now,
        )
        if Path(self.scheduler.directory) != self.spec.scheduler_directory:
            raise ServiceSpecError(
                "invalid_service_spec",
                "cluster scheduler contradicts the service specification",
            )
        snapshot = self.scheduler.reconstruct()
        if (
            snapshot.dynamic_gpus
            or self.spec.gpu7_id not in snapshot.gpu_ids
        ):
            raise ServiceSpecError(
                "invalid_service_spec",
                "adaptive service requires a fixed scheduler inventory containing GPU7",
            )
        self.service_status_path = self.spec.root / "service-status.json"
        self.status_path = self.service_status_path
        self.service_lock_path = self.spec.root / "service-controller.lock"

    def _now(self) -> float:
        return _epoch_seconds(self._clock(), "adaptive service clock")

    def _observation(self, now: float) -> Dict[str, Any]:
        return load_adaptive_observation(
            self.spec.observation_path,
            now=now,
            max_age_seconds=(
                self.spec.poll_interval_seconds * OBSERVATION_FRESHNESS_POLLS
            ),
            expected_path=self.spec.observation_path,
        )

    def trial_result_path(self, trial_id: str, round_index: int) -> Path:
        trial_id = _safe_id(trial_id, "trial_id")
        round_index = _nonnegative_integer(round_index, "round_index")
        return (
            self.store.trials_dir
            / trial_id
            / "results"
            / f"round-{round_index:02d}.json"
        )

    def _work_id(
        self,
        *,
        trial_id: str,
        round_index: int,
        manifest_sha256: str,
        result_path: Path,
    ) -> str:
        identity = {
            "contract": ADAPTIVE_WORK_CONTRACT,
            "result_path": str(result_path),
            "round_index": round_index,
            "service_spec_sha256": self.spec.spec_sha256,
            "trial_id": trial_id,
            "trial_manifest_sha256": manifest_sha256,
        }
        return "adaptive-" + canonical_sha256(identity)

    def build_trial_work_payload(
        self,
        trial_id: str,
        *,
        round_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the exact immutable ClusterExecutor payload for one round."""

        state = self.store.status()
        trial_id = _safe_id(trial_id, "trial_id")
        trial = state["trials"].get(trial_id)
        if trial is None:
            raise TrialConflictError(
                "unknown_trial",
                f"unknown adaptive trial: {trial_id}",
            )
        active_round = trial["round_index"]
        if round_index is not None and round_index != active_round:
            raise TrialConflictError(
                "invalid_trial_round",
                "requested work payload round differs from trial state",
            )
        manifest = self.store.verify_trial_bindings(trial_id)
        manifest_path = Path(trial["manifest_path"])
        if (
            manifest_path
            != self.store.trials_dir / trial_id / "trial.json"
            or manifest_path != Path(manifest["isolation_root"]) / "trial.json"
        ):
            raise StateCorruptionError(
                "trial_manifest_identity_mismatch",
                "adaptive state names a noncanonical trial manifest path",
            )
        result_path = self.trial_result_path(trial_id, active_round)
        _ensure_directory(result_path.parent)
        work_id = self._work_id(
            trial_id=trial_id,
            round_index=active_round,
            manifest_sha256=manifest["manifest_sha256"],
            result_path=result_path,
        )
        substitutions = {
            "epoch_id": manifest["epoch_id"],
            "gpu_id": self.spec.gpu7_id,
            "recipe_path": manifest["recipe_path"],
            "recipe_sha256": manifest["recipe_sha256"],
            "round_index": str(active_round),
            "service_spec_sha256": self.spec.spec_sha256,
            "trial_id": trial_id,
            "trial_manifest_path": str(manifest_path),
            "trial_manifest_sha256": manifest["manifest_sha256"],
            "trial_result_path": str(result_path),
            "work_id": work_id,
        }
        try:
            trial_argv = tuple(
                part.format_map(substitutions)
                for part in self.spec.trial_command_argv_template
            )
        except (KeyError, ValueError) as exc:
            raise ServiceSpecError(
                "invalid_service_spec",
                f"trial command template could not be expanded: {exc}",
            ) from exc
        if self.spec.gpu_lease_guardian_argv_prefix[-1] == "--command-json":
            guarded_child_argv = (canonical_json(list(trial_argv)),)
        else:
            guarded_child_argv = trial_argv
        argv = (
            self.spec.gpu_lease_guardian_argv_prefix
            + guarded_child_argv
        )
        environment = {
            "RISK_SCORE_ADAPTIVE_SERVICE_SPEC_SHA256": (
                self.spec.spec_sha256
            ),
            "RISK_SCORE_ADAPTIVE_TRIAL_ID": trial_id,
            "RISK_SCORE_ADAPTIVE_TRIAL_MANIFEST_PATH": str(manifest_path),
            "RISK_SCORE_ADAPTIVE_TRIAL_RESULT_PATH": str(result_path),
            "RISK_SCORE_ADAPTIVE_WORK_ID": work_id,
        }
        work_spec: Dict[str, Any] = {
            "argv": list(argv),
            "contract": WORK_SPEC_CONTRACT,
            "cwd": manifest["isolation_root"],
            "eligible_gpus": [self.spec.gpu7_id],
            "environment": environment,
            "kind": WorkKind.BACKFILL.value,
            "lease_role": "none",
            "safe_drain": None,
            "schema_version": SCHEMA_VERSION,
            "work_id": work_id,
        }
        work_spec["spec_sha256"] = canonical_sha256(work_spec)
        return {
            "executor_spec": work_spec,
        }

    build_work_payload = build_trial_work_payload

    def _enqueue_trial(self, trial_id: str) -> WorkRecord:
        payload = self.build_trial_work_payload(trial_id)
        work_spec = payload["executor_spec"]
        return self.scheduler.enqueue(
            work_id=work_spec["work_id"],
            kind=WorkKind.BACKFILL,
            eligible_gpus=(self.spec.gpu7_id,),
            preferred_gpu=self.spec.gpu7_id,
            preemptible=True,
            payload=payload,
        )

    def _is_service_work(self, record: WorkRecord) -> bool:
        raw = record.payload.get("executor_spec")
        return (
            isinstance(raw, Mapping)
            and isinstance(raw.get("environment"), Mapping)
            and raw["environment"].get(
                "RISK_SCORE_ADAPTIVE_SERVICE_SPEC_SHA256"
            )
            == self.spec.spec_sha256
        )

    @staticmethod
    def _is_adaptive_work(record: WorkRecord) -> bool:
        raw = record.payload.get("executor_spec")
        return (
            isinstance(raw, Mapping)
            and isinstance(raw.get("environment"), Mapping)
            and isinstance(
                raw["environment"].get(
                    "RISK_SCORE_ADAPTIVE_SERVICE_SPEC_SHA256"
                ),
                str,
            )
        )

    def _service_work_records(self) -> Tuple[WorkRecord, ...]:
        snapshot = self.scheduler.reconstruct()
        return tuple(
            record
            for _, record in sorted(snapshot.work.items())
            if self._is_service_work(record)
        )

    def _assert_no_orphaned_active_work(self) -> None:
        unfinished = [
            record.work_id
            for record in self.scheduler.reconstruct().work.values()
            if self._is_adaptive_work(record)
            if record.state in {WorkState.QUEUED, WorkState.CLAIMED}
        ]
        if unfinished:
            raise StateCorruptionError(
                "orphaned_scheduler_work",
                "scheduler has adaptive work but the adaptive event log has no "
                "active trial",
                details={"work_ids": unfinished},
            )

    def _active_parent_is_current(
        self,
        state: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> None:
        epoch_id = state["active_epoch_id"]
        if epoch_id is None:
            return
        event = next(
            (
                item
                for item in self.store.events()
                if item.event_type == "epoch.planned"
                and item.epoch_id == epoch_id
            ),
            None,
        )
        if event is None:
            raise StateCorruptionError(
                "missing_epoch_plan",
                "active adaptive epoch has no planning event",
            )
        if (
            event.payload["parent_champion_model_sha256"]
            != observation["current_champion_model_sha256"]
            or event.payload["champion_checkpoint"]["sha256"]
            != observation["champion_checkpoint"]["sha256"]
        ):
            raise ObservationValidationError(
                "active_epoch_parent_changed",
                "current champion changed while an adaptive epoch was active",
            )

    def _result_for_active(
        self,
        state: Mapping[str, Any],
        trial_id: str,
        payload: Mapping[str, Any],
        *,
        now: float,
        require_candidate_bindings: bool = False,
    ) -> Dict[str, Any]:
        trial = state["trials"][trial_id]
        work_spec = payload["executor_spec"]
        manifest = self.store.verify_trial_bindings(trial_id)
        return load_trial_result(
            self.trial_result_path(trial_id, trial["round_index"]),
            expected_trial_id=trial_id,
            expected_epoch_id=trial["epoch_id"],
            expected_round_index=trial["round_index"],
            expected_work_id=work_spec["work_id"],
            expected_gpu_id=self.spec.gpu7_id,
            expected_manifest_path=trial["manifest_path"],
            expected_manifest_sha256=manifest["manifest_sha256"],
            require_candidate_bindings=require_candidate_bindings,
            now=now,
        )

    def _fail_active(
        self,
        trial_id: str,
        reason: str,
        actions: List[Dict[str, Any]],
    ) -> None:
        current = self.store.status()
        if current["active_trial_id"] == trial_id:
            self.store.fail_trial(trial_id, reason=reason)
            actions.append(
                {
                    "action": "trial-failed",
                    "reason": reason,
                    "trial_id": trial_id,
                }
            )

    def _reconcile_active_trial(
        self,
        state: Mapping[str, Any],
        *,
        now: float,
        actions: List[Dict[str, Any]],
    ) -> bool:
        """Return True while the active trial still has scheduler work in flight."""

        trial_id = state["active_trial_id"]
        if trial_id is None:
            return False
        payload = self.build_trial_work_payload(trial_id)
        work_id = payload["executor_spec"]["work_id"]
        existing_record = self.scheduler.get_work(work_id)
        record = self._enqueue_trial(trial_id)
        conflicting = [
            item.work_id
            for item in self.scheduler.reconstruct().work.values()
            if self._is_adaptive_work(item)
            and item.work_id != work_id
            and item.state in {WorkState.QUEUED, WorkState.CLAIMED}
        ]
        if conflicting:
            raise StateCorruptionError(
                "multiple_active_scheduler_trials",
                "more than one adaptive scheduler work item is active",
                details={"work_ids": sorted([work_id, *conflicting])},
            )
        if existing_record is None:
            actions.append(
                {
                    "action": "trial-enqueued",
                    "replay": True,
                    "trial_id": trial_id,
                    "work_id": record.work_id,
                }
            )
        if record.state in {WorkState.QUEUED, WorkState.CLAIMED}:
            return True
        if record.state not in {
            WorkState.COMPLETED,
            WorkState.FAILED,
            WorkState.CANCELLED,
        }:
            raise StateCorruptionError(
                "invalid_scheduler_state",
                "adaptive scheduler work is in an unsupported state",
                details={"state": record.state.value, "work_id": work_id},
            )
        result_path = self.trial_result_path(
            trial_id,
            state["trials"][trial_id]["round_index"],
        )
        result: Optional[Dict[str, Any]] = None
        if result_path.exists() or result_path.is_symlink():
            try:
                result = self._result_for_active(
                    state,
                    trial_id,
                    payload,
                    now=now,
                )
            except (AdaptiveTrainingError, OSError, ValueError) as exc:
                self._fail_active(
                    trial_id,
                    "invalid scheduler trial result: "
                    f"{getattr(exc, 'code', type(exc).__name__)}",
                    actions,
                )
                return False
        if record.state != WorkState.COMPLETED:
            if result is not None:
                try:
                    usage = result["gpu_usage"]
                    self.store.record_gpu_usage(
                        trial_id,
                        started_at=usage["started_at_unix"],
                        ended_at=usage["ended_at_unix"],
                        gpu_count=usage["gpu_count"],
                        now=now,
                    )
                except AdaptiveTrainingError:
                    pass
            self._fail_active(
                trial_id,
                f"scheduler work terminated as {record.state.value}",
                actions,
            )
            return False
        if result is None:
            self._fail_active(
                trial_id,
                "completed scheduler work published no trial result",
                actions,
            )
            return False
        try:
            usage = result["gpu_usage"]
            self.store.record_gpu_usage(
                trial_id,
                started_at=usage["started_at_unix"],
                ended_at=usage["ended_at_unix"],
                gpu_count=usage["gpu_count"],
                now=now,
            )
            if result["status"] == "failed":
                self._fail_active(
                    trial_id,
                    f"trial worker failed: {result['failure_reason']}",
                    actions,
                )
                return False
            for item in result["evidence"]:
                self.store.record_evidence(trial_id, item)
            self.store.complete_trial(trial_id)
        except AdaptiveTrainingError as exc:
            self._fail_active(
                trial_id,
                "trial result ingestion failed: "
                f"{getattr(exc, 'code', type(exc).__name__)}",
                actions,
            )
            return False
        actions.append(
            {
                "action": "trial-completed",
                "result_sha256": result["result_sha256"],
                "trial_id": trial_id,
                "work_id": work_id,
            }
        )
        return False

    def _emit_winner_handoff(
        self,
        state: Mapping[str, Any],
        epoch_id: str,
        *,
        now: float,
        actions: List[Dict[str, Any]],
    ) -> None:
        epoch = state["epochs"][epoch_id]
        trial_id = epoch["winner_trial_id"]
        if trial_id is None:
            raise StateCorruptionError(
                "missing_epoch_winner",
                "winner-selected epoch has no winner trial",
            )
        payload = self.build_trial_work_payload(trial_id)
        work_id = payload["executor_spec"]["work_id"]
        record = self.scheduler.get_work(work_id)
        if record is None or record.state != WorkState.COMPLETED:
            raise StateCorruptionError(
                "winner_scheduler_mismatch",
                "final winner does not have completed scheduler work",
            )
        result = self._result_for_active(
            state,
            trial_id,
            payload,
            now=now,
            require_candidate_bindings=True,
        )
        created = self.store.create_handoff(
            trial_id,
            candidate_path=result["candidate_model"]["path"],
            candidate_sha256=result["candidate_model"]["sha256"],
            candidate_checkpoint_path=result["candidate_checkpoint"]["path"],
            candidate_checkpoint_sha256=result["candidate_checkpoint"][
                "sha256"
            ],
        )
        candidate_hash = created["handoff"]["candidate"]["sha256"]
        indexed_path = (
            self.store.candidate_handoffs_dir / f"{candidate_hash}.json"
        )
        if not indexed_path.is_file():
            raise StateCorruptionError(
                "missing_candidate_handoff_index",
                "final winner handoff was not indexed by candidate hash",
            )
        actions.append(
            {
                "action": "candidate-handoff-created",
                "candidate_sha256": candidate_hash,
                "handoff_path": created["handoff_path"],
                "trial_id": trial_id,
            }
        )

    def _status_value(
        self,
        *,
        now: float,
        observation: Optional[Mapping[str, Any]],
        actions: Sequence[Mapping[str, Any]] = (),
        blocked_reason: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> Dict[str, Any]:
        adaptive = self.store.status()
        snapshot = self.scheduler.reconstruct()
        service_records = [
            {
                "active_claim_id": record.active_claim_id,
                "attempts": record.attempts,
                "state": record.state.value,
                "trial_id": record.payload["executor_spec"]["environment"][
                    "RISK_SCORE_ADAPTIVE_TRIAL_ID"
                ],
                "work_id": record.work_id,
            }
            for record in self._service_work_records()
        ]
        active_work_id = None
        active_trial_id = adaptive["active_trial_id"]
        if active_trial_id is not None:
            active_work_id = next(
                (
                    record["work_id"]
                    for record in service_records
                    if record["trial_id"] == active_trial_id
                    and record["state"]
                    in {WorkState.QUEUED.value, WorkState.CLAIMED.value}
                ),
                None,
            )
        value: Dict[str, Any] = {
            "actions": [_json_copy(item) for item in actions],
            "active_trial_id": active_trial_id,
            "active_work_id": active_work_id,
            "actor": self.spec.actor,
            "adaptive_status": adaptive,
            "blocked_reason": blocked_reason,
            "contract": SERVICE_STATUS_CONTRACT,
            "error": (
                None
                if error is None
                else {
                    "code": getattr(error, "code", type(error).__name__),
                    "message": str(error),
                    "type": type(error).__name__,
                }
            ),
            "observation": (
                None if observation is None else _json_copy(observation)
            ),
            "observed_at_unix": now,
            "scheduler": {
                "dynamic_gpus": snapshot.dynamic_gpus,
                "gpu_ids": list(snapshot.gpu_ids),
                "revision": snapshot.revision,
                "service_work": service_records,
                "state_sha256": snapshot.state_sha256,
            },
            "schema_version": SCHEMA_VERSION,
            "service_spec_sha256": self.spec.spec_sha256,
        }
        value["status_sha256"] = canonical_sha256(value)
        return value

    def status(self) -> Dict[str, Any]:
        now = self._now()
        observation: Optional[Mapping[str, Any]] = None
        error: Optional[BaseException] = None
        try:
            observation = self._observation(now)
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            error = exc
        return self._status_value(
            now=now,
            observation=observation,
            error=error,
        )

    def once(self) -> Dict[str, Any]:
        """Run one bounded replay-safe observation/scheduler reconciliation."""

        with _ControllerLock(self.service_lock_path, self.spec.actor):
            now = self._now()
            observation = self._observation(now)
            actions: List[Dict[str, Any]] = []
            blocked_reason: Optional[str] = None
            started_this_pass = False
            for _ in range(64):
                state = self.store.status()
                self._active_parent_is_current(state, observation)
                if state["active_trial_id"] is not None:
                    waiting = self._reconcile_active_trial(
                        state,
                        now=now,
                        actions=actions,
                    )
                    if waiting:
                        break
                    continue
                epoch_id = state["active_epoch_id"]
                if epoch_id is None:
                    self._assert_no_orphaned_active_work()
                    plan = self.store.plan_epoch(
                        admitted_samples=observation["admitted_samples"],
                        last_promotion_admitted_samples=observation[
                            "last_promotion_admitted_samples"
                        ],
                        candidate_queue_depth=observation[
                            "candidate_queue_depth"
                        ],
                        parent_champion_model_sha256=observation[
                            "current_champion_model_sha256"
                        ],
                        champion_checkpoint_path=observation[
                            "champion_checkpoint"
                        ]["path"],
                        champion_checkpoint_sha256=observation[
                            "champion_checkpoint"
                        ]["sha256"],
                        admitted_data_manifest_path=observation[
                            "admitted_data_manifest"
                        ]["path"],
                        admitted_data_manifest_sha256=observation[
                            "admitted_data_manifest"
                        ]["sha256"],
                        now=now,
                    )
                    if not plan["planned"]:
                        reasons = plan["decision"]["reason_codes"]
                        blocked_reason = (
                            None if not reasons else ",".join(reasons)
                        )
                        break
                    actions.append(
                        {
                            "action": "epoch-planned",
                            "epoch_id": plan["epoch_id"],
                            "reused": plan.get("reused", False),
                        }
                    )
                    continue
                epoch = state["epochs"][epoch_id]
                if epoch["state"] == "winner_selected":
                    self._emit_winner_handoff(
                        state,
                        epoch_id,
                        now=now,
                        actions=actions,
                    )
                    continue
                if epoch["state"] != "running":
                    break
                survivors = list(epoch["survivor_trial_ids"])
                if survivors and all(
                    state["trials"][trial_id]["state"]
                    in {"complete", "failed"}
                    for trial_id in survivors
                ):
                    round_indexes = {
                        state["trials"][trial_id]["round_index"]
                        for trial_id in survivors
                    }
                    if len(round_indexes) != 1:
                        raise StateCorruptionError(
                            "inconsistent_trial_round",
                            "epoch survivors do not share one round",
                        )
                    round_index = next(iter(round_indexes))
                    event = self.store.halve_round(
                        epoch_id,
                        round_index=round_index,
                    )
                    actions.append(
                        {
                            "action": "round-halved",
                            "epoch_id": epoch_id,
                            "event_hash": event.event_hash,
                            "round_index": round_index,
                        }
                    )
                    continue
                ready = [
                    trial_id
                    for trial_id in survivors
                    if state["trials"][trial_id]["state"] == "ready"
                ]
                if not ready or started_this_pass:
                    break
                self._assert_no_orphaned_active_work()
                trial_id = ready[0]
                try:
                    self.store.start_trial(trial_id, now=now)
                except BudgetExceededError:
                    blocked_reason = "GPU_BUDGET_EXHAUSTED"
                    break
                record = self._enqueue_trial(trial_id)
                started_this_pass = True
                actions.append(
                    {
                        "action": "trial-enqueued",
                        "replay": False,
                        "trial_id": trial_id,
                        "work_id": record.work_id,
                    }
                )
                break
            else:
                raise StateCorruptionError(
                    "service_reconciliation_loop",
                    "adaptive service exceeded its bounded reconciliation loop",
                )
            status_value = self._status_value(
                now=now,
                observation=observation,
                actions=actions,
                blocked_reason=blocked_reason,
            )
            atomic_write_json(self.service_status_path, status_value)
            return status_value

    reconcile_once = once

    def watch(
        self,
        *,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Reconcile forever, persisting status without routine stdout."""

        sleep = self._sleeper if sleeper is None else sleeper
        while True:
            try:
                self.once()
            except (AdaptiveTrainingError, SchedulerError, OSError, ValueError) as exc:
                now = self._now()
                status_value = self._status_value(
                    now=now,
                    observation=None,
                    error=exc,
                )
                atomic_write_json(self.service_status_path, status_value)
            sleep(self.spec.poll_interval_seconds)


AdaptiveService = AdaptiveTrainingService


def status(
    spec_path: PathLike,
    *,
    clock: Callable[[], Any] = time.time,
) -> Dict[str, Any]:
    return AdaptiveTrainingService(spec_path, clock=clock).status()


def once(
    spec_path: PathLike,
    *,
    clock: Callable[[], Any] = time.time,
) -> Dict[str, Any]:
    return AdaptiveTrainingService(spec_path, clock=clock).once()


def watch(
    spec_path: PathLike,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], Any] = time.time,
) -> None:
    AdaptiveTrainingService(
        spec_path,
        clock=clock,
        sleeper=sleeper,
    ).watch()


def _print_json(value: Mapping[str, Any], *, stream: Any = None) -> None:
    destination = sys.stdout.buffer if stream is None else stream
    destination.write(_canonical_file_bytes(value))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--actor", default="adaptive-training-service")
    parser.add_argument("--spec", dest="global_spec", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--spec", dest="command_spec", type=Path)
    once_parser = subparsers.add_parser("once")
    once_parser.add_argument("--spec", dest="command_spec", type=Path)
    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("--spec", dest="command_spec", type=Path)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--admitted-samples", required=True, type=int)
    plan.add_argument("--last-promotion-admitted-samples", required=True, type=int)
    plan.add_argument("--candidate-queue-depth", required=True, type=int)
    plan.add_argument("--parent-champion-model-sha256", required=True)
    plan.add_argument("--champion-checkpoint", type=Path)
    plan.add_argument("--champion-checkpoint-sha256")
    plan.add_argument("--admitted-data-manifest", type=Path)
    plan.add_argument("--admitted-data-manifest-sha256")
    plan.add_argument("--now", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--trial-id", required=True)
    start.add_argument("--now", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--trial-id", required=True)
    record.add_argument("--evidence", required=True, type=Path)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--trial-id", required=True)

    halve = subparsers.add_parser("halve")
    halve.add_argument("--epoch-id", required=True)
    halve.add_argument("--round-index", required=True, type=int)

    handoff = subparsers.add_parser("handoff")
    handoff.add_argument("--trial-id", required=True)
    handoff.add_argument("--candidate", required=True, type=Path)
    handoff.add_argument("--candidate-sha256")
    handoff.add_argument("--candidate-checkpoint", type=Path)
    handoff.add_argument("--candidate-checkpoint-sha256")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command_spec = getattr(args, "command_spec", None)
    if (
        args.global_spec is not None
        and command_spec is not None
        and args.global_spec != command_spec
    ):
        parser.error("global and command adaptive service specifications differ")
    args.spec = command_spec or args.global_spec
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        service_spec_path = args.spec
        if service_spec_path is not None and args.root is not None:
            raise ServiceSpecError(
                "invalid_service_spec",
                "--spec and legacy --root modes are mutually exclusive",
            )
        if args.action in {"once", "watch"} and service_spec_path is None:
            raise ServiceSpecError(
                "missing_service_spec",
                f"--spec is required for adaptive service {args.action}",
            )
        if args.action == "status" and service_spec_path is not None:
            service = AdaptiveTrainingService(service_spec_path)
            result = service.status()
            _print_json(result)
            return 0
        if args.action == "once":
            result = AdaptiveTrainingService(service_spec_path).once()
            _print_json(result)
            return 0
        if args.action == "watch":
            AdaptiveTrainingService(service_spec_path).watch()
            return 0
        if args.root is None:
            raise AdaptiveTrainingError(
                "missing_root",
                "--root is required for legacy adaptive-training commands",
            )
        store = AdaptiveTrainingStore(
            args.root,
            policy_path=args.policy,
            actor=args.actor,
        )
        if args.action == "status":
            result: Mapping[str, Any] = store.status()
        elif args.action == "plan":
            result = store.plan_epoch(
                admitted_samples=args.admitted_samples,
                last_promotion_admitted_samples=(
                    args.last_promotion_admitted_samples
                ),
                candidate_queue_depth=args.candidate_queue_depth,
                parent_champion_model_sha256=(
                    args.parent_champion_model_sha256
                ),
                champion_checkpoint_path=args.champion_checkpoint,
                champion_checkpoint_sha256=args.champion_checkpoint_sha256,
                admitted_data_manifest_path=args.admitted_data_manifest,
                admitted_data_manifest_sha256=(
                    args.admitted_data_manifest_sha256
                ),
                now=args.now,
            )
        elif args.action == "start":
            result = store.start_trial(args.trial_id, now=args.now).to_dict()
        elif args.action == "record":
            result = store.record_evidence(
                args.trial_id,
                args.evidence,
            ).to_dict()
        elif args.action == "complete":
            result = store.complete_trial(args.trial_id).to_dict()
        elif args.action == "halve":
            result = store.halve_round(
                args.epoch_id,
                round_index=args.round_index,
            ).to_dict()
        else:
            result = store.create_handoff(
                args.trial_id,
                candidate_path=args.candidate,
                candidate_sha256=args.candidate_sha256,
                candidate_checkpoint_path=args.candidate_checkpoint,
                candidate_checkpoint_sha256=args.candidate_checkpoint_sha256,
            )
        _print_json(result)
        return 0
    except KeyboardInterrupt:
        return 0
    except AdaptiveTrainingError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr.buffer)
        return 2
    except (OSError, SchedulerError, ValueError) as exc:
        error = AdaptiveTrainingError(
            "unexpected_io_or_value_error",
            str(exc),
            details={"type": type(exc).__name__},
        )
        _print_json(error.to_dict(), stream=sys.stderr.buffer)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
