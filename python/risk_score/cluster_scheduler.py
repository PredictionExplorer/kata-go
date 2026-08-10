"""Crash-safe, work-conserving scheduling for a small GPU cluster.

This module is deliberately an executor-independent scheduling core.  It owns
durable queue, claim, cooperative-preemption, and idle-telemetry state, but it
never starts, signals, pauses, or kills a process.  Executors are responsible
for honoring preemption requests at safe checkpoint boundaries.

The complete authoritative state is stored as canonical JSON in the configured
scheduler directory.  Every mutation is serialized by a retained advisory lock
and published with an fsynced temporary file, atomic rename, and directory
fsync.  Reopening :class:`ClusterScheduler` reconstructs all queue and ownership
state from that file.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


PathLike = Union[str, os.PathLike]
SCHEMA_VERSION = 1
MAX_STATE_BYTES = 64 * 1024 * 1024
STATE_FILENAME = "state.json"
LOCK_FILENAME = ".scheduler.lock"


class SchedulerError(Exception):
    """Base class for scheduler failures."""


class SchedulerStateError(SchedulerError, ValueError):
    """Durable scheduler state is malformed, noncanonical, or contradictory."""


class SchedulerConflictError(SchedulerError, RuntimeError):
    """A retry conflicts with already durable state."""


class UnknownGpuError(SchedulerError, KeyError):
    """A GPU is outside a fixed scheduler inventory."""


class GpuBusyError(SchedulerConflictError):
    """A different owner already has the GPU's active claim."""


class NoActiveClaimError(SchedulerConflictError):
    """A release did not match an active or idempotently completed claim."""


class WorkKind(str, Enum):
    """Fixed work classes; callers cannot supply arbitrary priority numbers."""

    RECOVERY = "recovery"
    PROMOTION_CONFIRMATION = "promotion-confirmation"
    PROMOTION_CANARY = "promotion-canary"
    SCREENING = "screening"
    CURATION = "curation"
    TRAINER = "trainer"
    SELF_PLAY = "self-play"
    BACKFILL = "backfill"


# Confirmation and canary share the promotion-critical tier.  The order within
# each tuple is the deterministic tie-break before enqueue order.
PRIORITY_TIERS: Tuple[Tuple[WorkKind, ...], ...] = (
    (WorkKind.RECOVERY,),
    (WorkKind.PROMOTION_CONFIRMATION, WorkKind.PROMOTION_CANARY),
    (WorkKind.SCREENING,),
    (WorkKind.CURATION,),
    (WorkKind.TRAINER,),
    (WorkKind.SELF_PLAY,),
    (WorkKind.BACKFILL,),
)
PRIORITY_ORDER: Tuple[WorkKind, ...] = tuple(
    kind for tier in PRIORITY_TIERS for kind in tier
)
PRIORITY_RANK: Mapping[WorkKind, int] = MappingProxyType(
    {
        kind: tier_index
        for tier_index, tier in enumerate(PRIORITY_TIERS)
        for kind in tier
    }
)
_PRIORITY_SUBRANK: Mapping[WorkKind, int] = MappingProxyType(
    {
        kind: subrank
        for tier in PRIORITY_TIERS
        for subrank, kind in enumerate(tier)
    }
)
# A descriptive compatibility alias for callers that prefer this name.
WORK_PRIORITY = PRIORITY_RANK


class WorkState(str, Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReleaseOutcome(str, Enum):
    COMPLETED = "completed"
    REQUEUE = "requeue"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PreemptionStatus(str, Enum):
    PENDING = "pending"
    BOUNDARY_REACHED = "boundary-reached"


class IdleReason(str, Enum):
    NO_RUNNABLE_WORK = "no-runnable-work"
    CHECKPOINT_BOUNDARY = "checkpoint-boundary"
    LEASE_HANDOFF = "lease-handoff"
    SAFETY_HALT = "safety-halt"


_WORK_KIND_ALIASES = {
    "confirmation": WorkKind.PROMOTION_CONFIRMATION,
    "promotion-confirmation": WorkKind.PROMOTION_CONFIRMATION,
    "promotion_confirmation": WorkKind.PROMOTION_CONFIRMATION,
    "canary": WorkKind.PROMOTION_CANARY,
    "promotion-canary": WorkKind.PROMOTION_CANARY,
    "promotion_canary": WorkKind.PROMOTION_CANARY,
    "selfplay": WorkKind.SELF_PLAY,
    "self-play": WorkKind.SELF_PLAY,
    "self_play": WorkKind.SELF_PLAY,
}
_IDLE_REASON_ALIASES = {
    reason.value: reason for reason in IdleReason
}
_IDLE_REASON_ALIASES.update(
    {reason.value.replace("-", "_"): reason for reason in IdleReason}
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the deterministic UTF-8 JSON representation of *value*."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON compatible: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON representation of *value*."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: PathLike, value: Any) -> None:
    """Atomically publish newline-terminated canonical JSON at *path*."""

    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise SchedulerStateError(
            f"state parent must be a regular non-symlink directory: {parent}"
        )
    if os.path.lexists(os.fspath(destination)):
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SchedulerStateError(
                f"state path must be a regular non-symlink file: {destination}"
            )

    data = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=os.fspath(parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(os.fspath(temporary), os.fspath(destination))
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _require_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} may not have surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{field_name} may not contain NUL")
    return value


def _json_object(value: Optional[Mapping[str, Any]], field_name: str) -> Dict[str, Any]:
    candidate: Any = {} if value is None else value
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    encoded = canonical_json_bytes(dict(candidate))
    copied = json.loads(encoded.decode("utf-8"))
    if not isinstance(copied, dict):  # Defensive; the source was a Mapping.
        raise ValueError(f"{field_name} must be a JSON object")
    return copied


def _normalize_work_kind(value: Union[WorkKind, str]) -> WorkKind:
    if isinstance(value, WorkKind):
        return value
    if not isinstance(value, str):
        raise ValueError("kind must be a WorkKind or string")
    normalized = value.strip().lower()
    alias = _WORK_KIND_ALIASES.get(normalized)
    if alias is not None:
        return alias
    normalized = normalized.replace("_", "-")
    try:
        return WorkKind(normalized)
    except ValueError as exc:
        raise ValueError(f"unknown work kind: {value!r}") from exc


def _normalize_idle_reason(value: Union[IdleReason, str]) -> IdleReason:
    if isinstance(value, IdleReason):
        return value
    if not isinstance(value, str):
        raise ValueError("idle reason must be an IdleReason or string")
    reason = _IDLE_REASON_ALIASES.get(value.strip().lower())
    if reason is None:
        raise ValueError(f"unknown idle reason: {value!r}")
    return reason


def _normalize_release_outcome(
    value: Union[ReleaseOutcome, str],
) -> ReleaseOutcome:
    if isinstance(value, ReleaseOutcome):
        return value
    if not isinstance(value, str):
        raise ValueError("release outcome must be a ReleaseOutcome or string")
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "complete": ReleaseOutcome.COMPLETED,
        "completed": ReleaseOutcome.COMPLETED,
        "requeue": ReleaseOutcome.REQUEUE,
        "requeued": ReleaseOutcome.REQUEUE,
        "failed": ReleaseOutcome.FAILED,
        "cancelled": ReleaseOutcome.CANCELLED,
        "canceled": ReleaseOutcome.CANCELLED,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown release outcome: {value!r}") from exc


def priority_rank(kind: Union[WorkKind, str]) -> int:
    """Return the fixed priority tier (smaller values run first)."""

    return PRIORITY_RANK[_normalize_work_kind(kind)]


def _priority_class_key(kind: WorkKind) -> Tuple[int, int]:
    return (PRIORITY_RANK[kind], _PRIORITY_SUBRANK[kind])


@dataclass(frozen=True)
class SchedulerConfig:
    """Filesystem root and optional fixed GPU inventory.

    If ``gpu_ids`` is ``None`` (or empty), GPUs are registered durably when
    first named.  A nonempty inventory is closed: unknown GPUs are rejected.
    """

    directory: PathLike
    gpu_ids: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        requested = Path(self.directory).expanduser()
        if not requested.is_absolute():
            raise ValueError("scheduler directory must be absolute")
        normalized = Path(os.path.abspath(os.fspath(requested)))
        if requested != normalized:
            raise ValueError(
                "scheduler directory must be lexically normalized: "
                f"{requested} != {normalized}"
            )
        object.__setattr__(self, "directory", normalized)

        if self.gpu_ids is None:
            return
        normalized_gpus = tuple(
            sorted(_require_identifier(gpu, "gpu_id") for gpu in self.gpu_ids)
        )
        if len(normalized_gpus) != len(set(normalized_gpus)):
            raise ValueError("gpu_ids must not contain duplicates")
        object.__setattr__(
            self,
            "gpu_ids",
            normalized_gpus if normalized_gpus else None,
        )

    @property
    def state_path(self) -> Path:
        return Path(self.directory) / STATE_FILENAME

    @property
    def lock_path(self) -> Path:
        return Path(self.directory) / LOCK_FILENAME


@dataclass(frozen=True)
class WorkItem:
    """Immutable scheduling identity supplied by a producer."""

    work_id: str
    kind: Union[WorkKind, str]
    eligible_gpus: Optional[Tuple[str, ...]] = None
    preemptible: bool = False
    preferred_gpu: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "work_id", _require_identifier(self.work_id, "work_id")
        )
        object.__setattr__(self, "kind", _normalize_work_kind(self.kind))
        if not isinstance(self.preemptible, bool):
            raise ValueError("preemptible must be boolean")

        eligible: Optional[Tuple[str, ...]]
        if self.eligible_gpus is None:
            eligible = None
        else:
            eligible = tuple(
                sorted(
                    _require_identifier(gpu, "eligible_gpus entry")
                    for gpu in self.eligible_gpus
                )
            )
            if len(eligible) != len(set(eligible)):
                raise ValueError("eligible_gpus must not contain duplicates")
            if not eligible:
                eligible = None
        object.__setattr__(self, "eligible_gpus", eligible)

        preferred = self.preferred_gpu
        if preferred is not None:
            preferred = _require_identifier(preferred, "preferred_gpu")
            if eligible is not None and preferred not in eligible:
                raise ValueError("preferred_gpu must be included in eligible_gpus")
        object.__setattr__(self, "preferred_gpu", preferred)
        object.__setattr__(self, "payload", _json_object(self.payload, "payload"))

    @property
    def priority_rank(self) -> int:
        return PRIORITY_RANK[self.kind]  # type: ignore[index]

    def is_eligible(self, gpu_id: str) -> bool:
        return self.eligible_gpus is None or gpu_id in self.eligible_gpus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_id": self.work_id,
            "kind": self.kind.value,  # type: ignore[union-attr]
            "eligible_gpus": (
                None if self.eligible_gpus is None else list(self.eligible_gpus)
            ),
            "preemptible": self.preemptible,
            "preferred_gpu": self.preferred_gpu,
            "payload": _json_object(self.payload, "payload"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkItem":
        expected = {
            "work_id",
            "kind",
            "eligible_gpus",
            "preemptible",
            "preferred_gpu",
            "payload",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchedulerStateError("work item fields differ from schema")
        raw_eligible = value["eligible_gpus"]
        if raw_eligible is not None and not isinstance(raw_eligible, list):
            raise SchedulerStateError("eligible_gpus must be null or an array")
        try:
            return cls(
                work_id=value["work_id"],
                kind=value["kind"],
                eligible_gpus=(
                    None if raw_eligible is None else tuple(raw_eligible)
                ),
                preemptible=value["preemptible"],
                preferred_gpu=value["preferred_gpu"],
                payload=value["payload"],
            )
        except (TypeError, ValueError) as exc:
            raise SchedulerStateError(f"invalid work item: {exc}") from exc


@dataclass(frozen=True)
class Claim:
    claim_id: str
    work_id: str
    gpu_id: str
    owner_id: str
    claimed_at: float
    sequence: int
    stolen: bool

    @property
    def owner(self) -> str:
        return self.owner_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "work_id": self.work_id,
            "gpu_id": self.gpu_id,
            "owner_id": self.owner_id,
            "claimed_at": self.claimed_at,
            "sequence": self.sequence,
            "stolen": self.stolen,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Claim":
        expected = {
            "claim_id",
            "work_id",
            "gpu_id",
            "owner_id",
            "claimed_at",
            "sequence",
            "stolen",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchedulerStateError("claim fields differ from schema")
        try:
            claim = cls(
                claim_id=_require_identifier(value["claim_id"], "claim_id"),
                work_id=_require_identifier(value["work_id"], "work_id"),
                gpu_id=_require_identifier(value["gpu_id"], "gpu_id"),
                owner_id=_require_identifier(value["owner_id"], "owner_id"),
                claimed_at=float(value["claimed_at"]),
                sequence=_require_positive_int(value["sequence"], "claim sequence"),
                stolen=value["stolen"],
            )
        except (TypeError, ValueError) as exc:
            raise SchedulerStateError(f"invalid claim: {exc}") from exc
        if not math.isfinite(claim.claimed_at):
            raise SchedulerStateError("claim time must be finite")
        if not isinstance(claim.stolen, bool):
            raise SchedulerStateError("claim stolen flag must be boolean")
        return claim


@dataclass(frozen=True)
class ReleaseRecord:
    release_id: str
    claim_id: str
    work_id: str
    gpu_id: str
    owner_id: str
    outcome: ReleaseOutcome
    released_at: float
    sequence: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "claim_id": self.claim_id,
            "work_id": self.work_id,
            "gpu_id": self.gpu_id,
            "owner_id": self.owner_id,
            "outcome": self.outcome.value,
            "released_at": self.released_at,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseRecord":
        expected = {
            "release_id",
            "claim_id",
            "work_id",
            "gpu_id",
            "owner_id",
            "outcome",
            "released_at",
            "sequence",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchedulerStateError("release fields differ from schema")
        try:
            release = cls(
                release_id=_require_identifier(value["release_id"], "release_id"),
                claim_id=_require_identifier(value["claim_id"], "claim_id"),
                work_id=_require_identifier(value["work_id"], "work_id"),
                gpu_id=_require_identifier(value["gpu_id"], "gpu_id"),
                owner_id=_require_identifier(value["owner_id"], "owner_id"),
                outcome=_normalize_release_outcome(value["outcome"]),
                released_at=float(value["released_at"]),
                sequence=_require_positive_int(
                    value["sequence"], "release sequence"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise SchedulerStateError(f"invalid release: {exc}") from exc
        if not math.isfinite(release.released_at):
            raise SchedulerStateError("release time must be finite")
        return release


@dataclass(frozen=True)
class WorkRecord:
    item: WorkItem
    state: WorkState
    enqueue_sequence: int
    attempts: int
    active_claim_id: Optional[str] = None
    last_release: Optional[ReleaseRecord] = None

    @property
    def work_id(self) -> str:
        return self.item.work_id

    @property
    def kind(self) -> WorkKind:
        return self.item.kind  # type: ignore[return-value]

    @property
    def eligible_gpus(self) -> Optional[Tuple[str, ...]]:
        return self.item.eligible_gpus

    @property
    def preemptible(self) -> bool:
        return self.item.preemptible

    @property
    def preferred_gpu(self) -> Optional[str]:
        return self.item.preferred_gpu

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.item.payload

    @property
    def priority_rank(self) -> int:
        return self.item.priority_rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "state": self.state.value,
            "enqueue_sequence": self.enqueue_sequence,
            "attempts": self.attempts,
            "active_claim_id": self.active_claim_id,
            "last_release": (
                None if self.last_release is None else self.last_release.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkRecord":
        expected = {
            "item",
            "state",
            "enqueue_sequence",
            "attempts",
            "active_claim_id",
            "last_release",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchedulerStateError("work record fields differ from schema")
        try:
            state = WorkState(value["state"])
            sequence = _require_positive_int(
                value["enqueue_sequence"], "enqueue_sequence"
            )
            attempts = _require_nonnegative_int(value["attempts"], "attempts")
            active_claim_id = value["active_claim_id"]
            if active_claim_id is not None:
                active_claim_id = _require_identifier(
                    active_claim_id, "active_claim_id"
                )
            raw_release = value["last_release"]
            if raw_release is not None and not isinstance(raw_release, Mapping):
                raise ValueError("last_release must be null or an object")
            return cls(
                item=WorkItem.from_dict(value["item"]),
                state=state,
                enqueue_sequence=sequence,
                attempts=attempts,
                active_claim_id=active_claim_id,
                last_release=(
                    None
                    if raw_release is None
                    else ReleaseRecord.from_dict(raw_release)
                ),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SchedulerStateError):
                raise
            raise SchedulerStateError(f"invalid work record: {exc}") from exc


@dataclass(frozen=True)
class PreemptionRequest:
    request_id: str
    gpu_id: str
    claim_id: str
    running_work_id: str
    requested_for_work_id: str
    requested_by: str
    reason: str
    requested_at: float
    sequence: int
    status: PreemptionStatus = PreemptionStatus.PENDING
    boundary_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "gpu_id": self.gpu_id,
            "claim_id": self.claim_id,
            "running_work_id": self.running_work_id,
            "requested_for_work_id": self.requested_for_work_id,
            "requested_by": self.requested_by,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "sequence": self.sequence,
            "status": self.status.value,
            "boundary_at": self.boundary_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreemptionRequest":
        expected = {
            "request_id",
            "gpu_id",
            "claim_id",
            "running_work_id",
            "requested_for_work_id",
            "requested_by",
            "reason",
            "requested_at",
            "sequence",
            "status",
            "boundary_at",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchedulerStateError("preemption request fields differ from schema")
        try:
            boundary = value["boundary_at"]
            request = cls(
                request_id=_require_identifier(value["request_id"], "request_id"),
                gpu_id=_require_identifier(value["gpu_id"], "gpu_id"),
                claim_id=_require_identifier(value["claim_id"], "claim_id"),
                running_work_id=_require_identifier(
                    value["running_work_id"], "running_work_id"
                ),
                requested_for_work_id=_require_identifier(
                    value["requested_for_work_id"], "requested_for_work_id"
                ),
                requested_by=_require_identifier(
                    value["requested_by"], "requested_by"
                ),
                reason=_require_identifier(value["reason"], "reason"),
                requested_at=float(value["requested_at"]),
                sequence=_require_positive_int(
                    value["sequence"], "preemption sequence"
                ),
                status=PreemptionStatus(value["status"]),
                boundary_at=None if boundary is None else float(boundary),
            )
        except (TypeError, ValueError) as exc:
            raise SchedulerStateError(f"invalid preemption request: {exc}") from exc
        if not math.isfinite(request.requested_at) or (
            request.boundary_at is not None and not math.isfinite(request.boundary_at)
        ):
            raise SchedulerStateError("preemption times must be finite")
        if (
            request.status == PreemptionStatus.PENDING
            and request.boundary_at is not None
        ) or (
            request.status == PreemptionStatus.BOUNDARY_REACHED
            and request.boundary_at is None
        ):
            raise SchedulerStateError(
                "preemption status contradicts boundary timestamp"
            )
        return request


@dataclass(frozen=True)
class IdleEvent:
    event_id: str
    gpu_id: str
    reason: IdleReason
    recorded_at: float
    sequence: int
    owner_id: Optional[str]
    work_id: Optional[str]
    details: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "gpu_id": self.gpu_id,
            "reason": self.reason.value,
            "recorded_at": self.recorded_at,
            "sequence": self.sequence,
            "owner_id": self.owner_id,
            "work_id": self.work_id,
            "details": _json_object(self.details, "idle details"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IdleEvent":
        expected = {
            "event_id",
            "gpu_id",
            "reason",
            "recorded_at",
            "sequence",
            "owner_id",
            "work_id",
            "details",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SchedulerStateError("idle event fields differ from schema")
        try:
            owner = value["owner_id"]
            work_id = value["work_id"]
            if owner is not None:
                owner = _require_identifier(owner, "idle owner_id")
            if work_id is not None:
                work_id = _require_identifier(work_id, "idle work_id")
            event = cls(
                event_id=_require_identifier(value["event_id"], "event_id"),
                gpu_id=_require_identifier(value["gpu_id"], "gpu_id"),
                reason=_normalize_idle_reason(value["reason"]),
                recorded_at=float(value["recorded_at"]),
                sequence=_require_positive_int(
                    value["sequence"], "idle event sequence"
                ),
                owner_id=owner,
                work_id=work_id,
                details=_json_object(value["details"], "idle details"),
            )
        except (TypeError, ValueError) as exc:
            raise SchedulerStateError(f"invalid idle event: {exc}") from exc
        if not math.isfinite(event.recorded_at):
            raise SchedulerStateError("idle event time must be finite")
        return event


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Validated in-memory reconstruction of durable scheduler state."""

    revision: int
    next_sequence: int
    gpu_ids: Tuple[str, ...]
    dynamic_gpus: bool
    work: Mapping[str, WorkRecord]
    claims: Mapping[str, Claim]
    preemption_requests: Tuple[PreemptionRequest, ...]
    idle: Mapping[str, IdleEvent]
    idle_history: Tuple[IdleEvent, ...]
    safety_halt: Optional[str]
    gpu_safety_halts: Mapping[str, str]
    state_sha256: str

    @property
    def queued(self) -> Tuple[WorkRecord, ...]:
        return tuple(
            sorted(
                (record for record in self.work.values() if record.state == WorkState.QUEUED),
                key=_work_priority_key,
            )
        )

    @property
    def active_owners(self) -> Mapping[str, str]:
        return MappingProxyType(
            {gpu: claim.owner_id for gpu, claim in self.claims.items()}
        )


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _work_priority_key(record: WorkRecord) -> Tuple[int, int, int, str]:
    tier, subrank = _priority_class_key(record.kind)
    return (tier, subrank, record.enqueue_sequence, record.work_id)


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchedulerStateError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SchedulerStateError(f"non-finite JSON number is forbidden: {value}")


_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.RLock] = {}


def _process_lock(path: Path) -> threading.RLock:
    key = os.fspath(path)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


class ClusterScheduler:
    """Durable scheduling core for mutually exclusive per-GPU work."""

    def __init__(
        self,
        scheduler_directory: Union[PathLike, SchedulerConfig],
        gpu_ids: Optional[Sequence[str]] = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(scheduler_directory, SchedulerConfig):
            if gpu_ids is not None:
                raise ValueError(
                    "gpu_ids must be supplied either in SchedulerConfig or separately"
                )
            config = scheduler_directory
        else:
            config = SchedulerConfig(
                scheduler_directory,
                None if gpu_ids is None else tuple(gpu_ids),
            )
        self.config = config
        self._clock = clock
        self._ensure_directory()

        with self._locked():
            if self.state_path.exists():
                state = self._load_state()
                expected_gpus = config.gpu_ids
                if expected_gpus is not None and tuple(state["gpu_ids"]) != expected_gpus:
                    raise SchedulerConflictError(
                        "configured GPU inventory differs from durable scheduler state"
                    )
            else:
                state = self._new_state()
                self._write_state(state)

    @property
    def directory(self) -> Path:
        return Path(self.config.directory)

    @property
    def state_path(self) -> Path:
        return self.config.state_path

    @property
    def lock_path(self) -> Path:
        return self.config.lock_path

    def enqueue(
        self,
        item: Optional[Union[WorkItem, str]] = None,
        kind: Optional[Union[WorkKind, str]] = None,
        *,
        work_id: Optional[str] = None,
        eligible_gpus: Optional[Sequence[str]] = None,
        preemptible: bool = False,
        preferred_gpu: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> WorkRecord:
        """Idempotently add one work item.

        ``work_id`` is the idempotency key.  An exact retry returns the existing
        record without writing; changed immutable metadata raises a conflict.
        """

        if isinstance(item, WorkItem):
            if any(
                value is not None
                for value in (kind, work_id, eligible_gpus, preferred_gpu, payload)
            ) or preemptible:
                raise ValueError("a WorkItem cannot be combined with item fields")
            desired = item
        else:
            if isinstance(item, str):
                if work_id is not None and work_id != item:
                    raise ValueError("positional and keyword work_id differ")
                work_id = item
            elif item is not None:
                raise TypeError("item must be a WorkItem, work ID string, or None")
            if work_id is None or kind is None:
                raise ValueError("work_id and kind are required")
            desired = WorkItem(
                work_id=work_id,
                kind=kind,
                eligible_gpus=(
                    None if eligible_gpus is None else tuple(eligible_gpus)
                ),
                preemptible=preemptible,
                preferred_gpu=preferred_gpu,
                payload={} if payload is None else payload,
            )

        with self._locked():
            state = self._load_state()
            raw_existing = state["work"].get(desired.work_id)
            if raw_existing is not None:
                existing = WorkRecord.from_dict(raw_existing)
                if existing.item.to_dict() != desired.to_dict():
                    raise SchedulerConflictError(
                        f"work_id {desired.work_id!r} was enqueued with different metadata"
                    )
                return existing

            self._validate_or_register_item_gpus(state, desired)
            sequence = self._allocate_sequence(state)
            record = WorkRecord(
                item=desired,
                state=WorkState.QUEUED,
                enqueue_sequence=sequence,
                attempts=0,
            )
            state["work"][desired.work_id] = record.to_dict()
            self._commit(state)
            return record

    def claim(self, gpu_id: str, owner_id: str) -> Optional[Claim]:
        """Claim the highest-priority runnable item for ``gpu_id``.

        The queue is global, so an idle eligible GPU steals work even when a
        different GPU was preferred.  Repeating a claim by the active owner is
        an idempotent read.  A different owner receives :class:`GpuBusyError`.
        """

        gpu_id = _require_identifier(gpu_id, "gpu_id")
        owner_id = _require_identifier(owner_id, "owner_id")
        with self._locked():
            state = self._load_state()
            registered = self._validate_or_register_gpu(state, gpu_id)
            raw_active = state["claims"].get(gpu_id)
            if raw_active is not None:
                active = Claim.from_dict(raw_active)
                if active.owner_id == owner_id:
                    return active
                raise GpuBusyError(
                    f"GPU {gpu_id!r} is owned by {active.owner_id!r} "
                    f"under claim {active.claim_id}"
                )

            halt_reason = self._halt_reason(state, gpu_id)
            if halt_reason is not None:
                _, idle_changed = self._record_idle_in_state(
                    state,
                    gpu_id,
                    IdleReason.SAFETY_HALT,
                    details={"reason": halt_reason},
                )
                if registered or idle_changed:
                    self._commit(state)
                return None

            runnable = [
                WorkRecord.from_dict(raw)
                for raw in state["work"].values()
                if raw["state"] == WorkState.QUEUED.value
                and WorkItem.from_dict(raw["item"]).is_eligible(gpu_id)
            ]
            if not runnable:
                queued_count = sum(
                    raw["state"] == WorkState.QUEUED.value
                    for raw in state["work"].values()
                )
                _, idle_changed = self._record_idle_in_state(
                    state,
                    gpu_id,
                    IdleReason.NO_RUNNABLE_WORK,
                    details={"queued_work": queued_count},
                )
                if registered or idle_changed:
                    self._commit(state)
                return None

            selected = min(runnable, key=_work_priority_key)
            sequence = self._allocate_sequence(state)
            claim = Claim(
                claim_id=f"claim-{sequence:020d}",
                work_id=selected.work_id,
                gpu_id=gpu_id,
                owner_id=owner_id,
                claimed_at=self._now(),
                sequence=sequence,
                stolen=(
                    selected.preferred_gpu is not None
                    and selected.preferred_gpu != gpu_id
                ),
            )
            state["claims"][gpu_id] = claim.to_dict()
            state["work"][selected.work_id] = WorkRecord(
                item=selected.item,
                state=WorkState.CLAIMED,
                enqueue_sequence=selected.enqueue_sequence,
                attempts=selected.attempts + 1,
                active_claim_id=claim.claim_id,
                last_release=selected.last_release,
            ).to_dict()
            state["idle"].pop(gpu_id, None)
            self._commit(state)
            return claim

    claim_next = claim

    def release(
        self,
        claim_or_gpu: Union[Claim, str],
        owner_id: Optional[str] = None,
        work_id: Optional[str] = None,
        *,
        outcome: Union[ReleaseOutcome, str] = ReleaseOutcome.COMPLETED,
        requeue: Optional[bool] = None,
        idle_reason: Optional[Union[IdleReason, str]] = None,
        idle_details: Optional[Mapping[str, Any]] = None,
    ) -> ReleaseRecord:
        """Release one claim at an executor-controlled boundary.

        ``requeue=True`` makes the item runnable again.  When a pending
        preemption request is released this way, checkpoint-boundary telemetry
        is recorded automatically.  The scheduler itself never revokes a claim.
        """

        expected_claim_id: Optional[str] = None
        if isinstance(claim_or_gpu, Claim):
            claim_arg = claim_or_gpu
            gpu_id = claim_arg.gpu_id
            expected_claim_id = claim_arg.claim_id
            if owner_id is not None and owner_id != claim_arg.owner_id:
                raise ValueError("owner_id differs from Claim")
            if work_id is not None and work_id != claim_arg.work_id:
                raise ValueError("work_id differs from Claim")
            owner_id = claim_arg.owner_id
            work_id = claim_arg.work_id
        else:
            gpu_id = _require_identifier(claim_or_gpu, "gpu_id")
        owner_id = _require_identifier(owner_id, "owner_id")
        if work_id is not None:
            work_id = _require_identifier(work_id, "work_id")

        normalized_outcome = _normalize_release_outcome(outcome)
        if requeue is True:
            if normalized_outcome not in {
                ReleaseOutcome.COMPLETED,
                ReleaseOutcome.REQUEUE,
            }:
                raise ValueError("requeue=True conflicts with terminal outcome")
            normalized_outcome = ReleaseOutcome.REQUEUE
        elif requeue is False and normalized_outcome == ReleaseOutcome.REQUEUE:
            raise ValueError("requeue=False conflicts with outcome='requeue'")
        normalized_idle = (
            None if idle_reason is None else _normalize_idle_reason(idle_reason)
        )
        normalized_idle_details = _json_object(idle_details, "idle_details")

        with self._locked():
            state = self._load_state()
            self._validate_known_gpu(state, gpu_id)
            raw_active = state["claims"].get(gpu_id)
            if raw_active is None:
                retry = self._matching_release_retry(
                    state,
                    gpu_id=gpu_id,
                    owner_id=owner_id,
                    work_id=work_id,
                    expected_claim_id=expected_claim_id,
                    outcome=normalized_outcome,
                )
                if retry is not None:
                    return retry
                raise NoActiveClaimError(f"GPU {gpu_id!r} has no active claim")

            active = Claim.from_dict(raw_active)
            if active.owner_id != owner_id:
                raise GpuBusyError(
                    f"GPU {gpu_id!r} is owned by {active.owner_id!r}, not {owner_id!r}"
                )
            if work_id is not None and active.work_id != work_id:
                raise SchedulerConflictError(
                    f"active work is {active.work_id!r}, not {work_id!r}"
                )
            if expected_claim_id is not None and active.claim_id != expected_claim_id:
                raise SchedulerConflictError(
                    f"claim {expected_claim_id!r} is stale; active claim is "
                    f"{active.claim_id!r}"
                )

            record = WorkRecord.from_dict(state["work"][active.work_id])
            sequence = self._allocate_sequence(state)
            released_at = self._now()
            release = ReleaseRecord(
                release_id=f"release-{sequence:020d}",
                claim_id=active.claim_id,
                work_id=active.work_id,
                gpu_id=gpu_id,
                owner_id=owner_id,
                outcome=normalized_outcome,
                released_at=released_at,
                sequence=sequence,
            )
            next_state = {
                ReleaseOutcome.COMPLETED: WorkState.COMPLETED,
                ReleaseOutcome.REQUEUE: WorkState.QUEUED,
                ReleaseOutcome.FAILED: WorkState.FAILED,
                ReleaseOutcome.CANCELLED: WorkState.CANCELLED,
            }[normalized_outcome]
            state["work"][active.work_id] = WorkRecord(
                item=record.item,
                state=next_state,
                enqueue_sequence=record.enqueue_sequence,
                attempts=record.attempts,
                active_claim_id=None,
                last_release=release,
            ).to_dict()
            del state["claims"][gpu_id]

            had_preemption_request = False
            for request_id, raw_request in list(
                state["preemption_requests"].items()
            ):
                request = PreemptionRequest.from_dict(raw_request)
                if (
                    request.claim_id == active.claim_id
                    and request.status == PreemptionStatus.PENDING
                ):
                    had_preemption_request = True
                    state["preemption_requests"][request_id] = PreemptionRequest(
                        request_id=request.request_id,
                        gpu_id=request.gpu_id,
                        claim_id=request.claim_id,
                        running_work_id=request.running_work_id,
                        requested_for_work_id=request.requested_for_work_id,
                        requested_by=request.requested_by,
                        reason=request.reason,
                        requested_at=request.requested_at,
                        sequence=request.sequence,
                        status=PreemptionStatus.BOUNDARY_REACHED,
                        boundary_at=released_at,
                    ).to_dict()

            if (
                normalized_idle is None
                and had_preemption_request
                and normalized_outcome == ReleaseOutcome.REQUEUE
            ):
                normalized_idle = IdleReason.CHECKPOINT_BOUNDARY
                normalized_idle_details = {
                    "claim_id": active.claim_id,
                    "work_id": active.work_id,
                }
            if normalized_idle is not None:
                self._record_idle_in_state(
                    state,
                    gpu_id,
                    normalized_idle,
                    owner_id=owner_id,
                    work_id=active.work_id,
                    details=normalized_idle_details,
                )
            self._commit(state)
            return release

    def request_preemption(
        self,
        gpu_id: str,
        requested_by: str = "scheduler",
        *,
        for_work_id: Optional[str] = None,
        reason: str = "higher-priority work is queued",
    ) -> Optional[PreemptionRequest]:
        """Record a cooperative request against lower-priority work.

        No owner or work state is changed.  ``None`` means the GPU is idle, the
        running item is nonpreemptible, or no strictly higher-priority eligible
        queued item exists.
        """

        gpu_id = _require_identifier(gpu_id, "gpu_id")
        requested_by = _require_identifier(requested_by, "requested_by")
        reason = _require_identifier(reason, "reason")
        if for_work_id is not None:
            for_work_id = _require_identifier(for_work_id, "for_work_id")

        with self._locked():
            state = self._load_state()
            self._validate_known_gpu(state, gpu_id)
            raw_claim = state["claims"].get(gpu_id)
            if raw_claim is None:
                return None
            claim = Claim.from_dict(raw_claim)
            running = WorkRecord.from_dict(state["work"][claim.work_id])
            if not running.preemptible:
                return None

            if for_work_id is None:
                candidates = [
                    WorkRecord.from_dict(raw)
                    for raw in state["work"].values()
                    if raw["state"] == WorkState.QUEUED.value
                    and WorkItem.from_dict(raw["item"]).is_eligible(gpu_id)
                    and _priority_class_key(
                        WorkItem.from_dict(raw["item"]).kind  # type: ignore[arg-type]
                    )
                    < _priority_class_key(running.kind)
                ]
                if not candidates:
                    return None
                requested_for = min(candidates, key=_work_priority_key)
            else:
                raw_requested = state["work"].get(for_work_id)
                if raw_requested is None:
                    raise KeyError(for_work_id)
                requested_for = WorkRecord.from_dict(raw_requested)
                if (
                    requested_for.state != WorkState.QUEUED
                    or not requested_for.item.is_eligible(gpu_id)
                    or _priority_class_key(requested_for.kind)
                    >= _priority_class_key(running.kind)
                ):
                    return None

            for raw_request in state["preemption_requests"].values():
                existing = PreemptionRequest.from_dict(raw_request)
                if (
                    existing.status == PreemptionStatus.PENDING
                    and existing.claim_id == claim.claim_id
                    and existing.requested_for_work_id == requested_for.work_id
                ):
                    return existing

            sequence = self._allocate_sequence(state)
            request = PreemptionRequest(
                request_id=f"preempt-{sequence:020d}",
                gpu_id=gpu_id,
                claim_id=claim.claim_id,
                running_work_id=running.work_id,
                requested_for_work_id=requested_for.work_id,
                requested_by=requested_by,
                reason=reason,
                requested_at=self._now(),
                sequence=sequence,
            )
            state["preemption_requests"][request.request_id] = request.to_dict()
            self._commit(state)
            return request

    def record_idle(
        self,
        gpu_id: str,
        reason: Union[IdleReason, str],
        *,
        owner_id: Optional[str] = None,
        work_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> IdleEvent:
        """Durably record one of the scheduler's four idle reasons."""

        gpu_id = _require_identifier(gpu_id, "gpu_id")
        normalized_reason = _normalize_idle_reason(reason)
        if owner_id is not None:
            owner_id = _require_identifier(owner_id, "owner_id")
        if work_id is not None:
            work_id = _require_identifier(work_id, "work_id")
        normalized_details = _json_object(details, "idle details")
        with self._locked():
            state = self._load_state()
            registered = self._validate_or_register_gpu(state, gpu_id)
            event, changed = self._record_idle_in_state(
                state,
                gpu_id,
                normalized_reason,
                owner_id=owner_id,
                work_id=work_id,
                details=normalized_details,
            )
            if registered or changed:
                self._commit(state)
            return event

    def set_safety_halt(
        self, reason: str, *, gpu_id: Optional[str] = None
    ) -> None:
        """Block future claims globally or on one GPU without revoking owners."""

        reason = _require_identifier(reason, "reason")
        if gpu_id is not None:
            gpu_id = _require_identifier(gpu_id, "gpu_id")
        with self._locked():
            state = self._load_state()
            changed = False
            if gpu_id is None:
                desired = {"reason": reason, "set_at": self._now()}
                if (
                    state["safety_halt"] is None
                    or state["safety_halt"]["reason"] != reason
                ):
                    state["safety_halt"] = desired
                    changed = True
                targets = tuple(state["gpu_ids"])
            else:
                changed = self._validate_or_register_gpu(state, gpu_id)
                existing = state["gpu_safety_halts"].get(gpu_id)
                if existing is None or existing["reason"] != reason:
                    state["gpu_safety_halts"][gpu_id] = {
                        "reason": reason,
                        "set_at": self._now(),
                    }
                    changed = True
                targets = (gpu_id,)
            for target in targets:
                if target not in state["claims"]:
                    _, idle_changed = self._record_idle_in_state(
                        state,
                        target,
                        IdleReason.SAFETY_HALT,
                        details={"reason": reason},
                    )
                    changed = changed or idle_changed
            if changed:
                self._commit(state)

    def clear_safety_halt(self, *, gpu_id: Optional[str] = None) -> None:
        """Idempotently clear a global or per-GPU safety halt."""

        if gpu_id is not None:
            gpu_id = _require_identifier(gpu_id, "gpu_id")
        with self._locked():
            state = self._load_state()
            if gpu_id is None:
                changed = state["safety_halt"] is not None
                state["safety_halt"] = None
            else:
                self._validate_known_gpu(state, gpu_id)
                changed = state["gpu_safety_halts"].pop(gpu_id, None) is not None
            if changed:
                self._commit(state)

    def reconstruct(self) -> SchedulerSnapshot:
        """Load, integrity-check, and reconstruct the durable scheduler state."""

        with self._locked():
            return self._snapshot(self._load_state())

    snapshot = reconstruct

    def state_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-compatible copy of authoritative state."""

        with self._locked():
            state = self._load_state()
            return json.loads(canonical_json_bytes(state).decode("utf-8"))

    read_state = state_dict

    def get_work(self, work_id: str) -> Optional[WorkRecord]:
        work_id = _require_identifier(work_id, "work_id")
        return self.reconstruct().work.get(work_id)

    def get_claim(self, gpu_id: str) -> Optional[Claim]:
        gpu_id = _require_identifier(gpu_id, "gpu_id")
        return self.reconstruct().claims.get(gpu_id)

    def queued_work(self, gpu_id: Optional[str] = None) -> Tuple[WorkRecord, ...]:
        snapshot = self.reconstruct()
        if gpu_id is None:
            return snapshot.queued
        gpu_id = _require_identifier(gpu_id, "gpu_id")
        if gpu_id not in snapshot.gpu_ids and not snapshot.dynamic_gpus:
            raise UnknownGpuError(gpu_id)
        return tuple(record for record in snapshot.queued if record.item.is_eligible(gpu_id))

    def pending_preemptions(
        self, gpu_id: Optional[str] = None
    ) -> Tuple[PreemptionRequest, ...]:
        if gpu_id is not None:
            gpu_id = _require_identifier(gpu_id, "gpu_id")
        return tuple(
            request
            for request in self.reconstruct().preemption_requests
            if request.status == PreemptionStatus.PENDING
            and (gpu_id is None or request.gpu_id == gpu_id)
        )

    def idle_status(self, gpu_id: str) -> Optional[IdleEvent]:
        gpu_id = _require_identifier(gpu_id, "gpu_id")
        return self.reconstruct().idle.get(gpu_id)

    def idle_events(self, gpu_id: Optional[str] = None) -> Tuple[IdleEvent, ...]:
        if gpu_id is not None:
            gpu_id = _require_identifier(gpu_id, "gpu_id")
        return tuple(
            event
            for event in self.reconstruct().idle_history
            if gpu_id is None or event.gpu_id == gpu_id
        )

    def _ensure_directory(self) -> None:
        directory = self.directory
        current = directory
        while True:
            if current.exists() and current.is_symlink():
                raise SchedulerStateError(
                    f"scheduler directory has a symlinked component: {current}"
                )
            if current.parent == current:
                break
            current = current.parent
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise SchedulerStateError(
                    f"scheduler directory is not a regular directory: {directory}"
                )
            return
        directory.mkdir(parents=True, exist_ok=False)
        _fsync_directory(directory.parent)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        process_lock = _process_lock(self.lock_path)
        with process_lock:
            existed = os.path.lexists(os.fspath(self.lock_path))
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(os.fspath(self.lock_path), flags, 0o600)
            locked = False
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise SchedulerStateError("scheduler lock is not a regular file")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                if not existed:
                    os.fsync(descriptor)
                    _fsync_directory(self.directory)
                yield
            finally:
                if locked:
                    with contextlib.suppress(OSError):
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _new_state(self) -> Dict[str, Any]:
        configured = self.config.gpu_ids
        state: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "revision": 0,
            "next_sequence": 1,
            "dynamic_gpus": configured is None,
            "gpu_ids": [] if configured is None else list(configured),
            "work": {},
            "claims": {},
            "preemption_requests": {},
            "idle_events": [],
            "idle": {},
            "safety_halt": None,
            "gpu_safety_halts": {},
        }
        return self._seal_state(state)

    def _load_state(self) -> Dict[str, Any]:
        path = self.state_path
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise SchedulerStateError(f"scheduler state is missing: {path}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SchedulerStateError(
                f"scheduler state must be a regular non-symlink file: {path}"
            )
        if metadata.st_size > MAX_STATE_BYTES:
            raise SchedulerStateError("scheduler state exceeds the size limit")
        try:
            data = path.read_bytes()
            value = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SchedulerStateError(f"cannot decode scheduler state: {exc}") from exc
        if not isinstance(value, dict):
            raise SchedulerStateError("scheduler state root must be an object")
        if data != canonical_json_bytes(value) + b"\n":
            raise SchedulerStateError(
                "scheduler state is not canonical newline-terminated JSON"
            )
        self._validate_state(value)
        return value

    def _write_state(self, state: Mapping[str, Any]) -> None:
        self._validate_state(state)
        atomic_write_json(self.state_path, state)

    def _commit(self, state: Dict[str, Any]) -> None:
        state.pop("state_sha256", None)
        state["revision"] = _require_nonnegative_int(
            state["revision"], "revision"
        ) + 1
        sealed = self._seal_state(state)
        self._write_state(sealed)

    @staticmethod
    def _seal_state(state: Mapping[str, Any]) -> Dict[str, Any]:
        body = json.loads(canonical_json_bytes(dict(state)).decode("utf-8"))
        body.pop("state_sha256", None)
        body["state_sha256"] = canonical_sha256(body)
        return body

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        expected_fields = {
            "schema_version",
            "revision",
            "next_sequence",
            "dynamic_gpus",
            "gpu_ids",
            "work",
            "claims",
            "preemption_requests",
            "idle_events",
            "idle",
            "safety_halt",
            "gpu_safety_halts",
            "state_sha256",
        }
        if set(state) != expected_fields:
            raise SchedulerStateError("scheduler state fields differ from schema")
        if state["schema_version"] != SCHEMA_VERSION or isinstance(
            state["schema_version"], bool
        ):
            raise SchedulerStateError("unsupported scheduler state schema")
        _require_nonnegative_int(state["revision"], "revision")
        next_sequence = _require_positive_int(
            state["next_sequence"], "next_sequence"
        )
        if not isinstance(state["dynamic_gpus"], bool):
            raise SchedulerStateError("dynamic_gpus must be boolean")
        raw_gpus = state["gpu_ids"]
        if not isinstance(raw_gpus, list):
            raise SchedulerStateError("gpu_ids must be an array")
        try:
            gpu_ids = tuple(_require_identifier(gpu, "gpu_id") for gpu in raw_gpus)
        except ValueError as exc:
            raise SchedulerStateError(str(exc)) from exc
        if gpu_ids != tuple(sorted(set(gpu_ids))):
            raise SchedulerStateError("gpu_ids must be sorted and unique")

        body = dict(state)
        actual_hash = body.pop("state_sha256")
        if (
            not isinstance(actual_hash, str)
            or len(actual_hash) != 64
            or actual_hash != canonical_sha256(body)
        ):
            raise SchedulerStateError("scheduler state hash mismatch")

        for field_name in (
            "work",
            "claims",
            "preemption_requests",
            "idle",
            "gpu_safety_halts",
        ):
            if not isinstance(state[field_name], Mapping):
                raise SchedulerStateError(f"{field_name} must be an object")
        if not isinstance(state["idle_events"], list):
            raise SchedulerStateError("idle_events must be an array")

        records: Dict[str, WorkRecord] = {}
        max_sequence = 0
        for work_id, raw_record in state["work"].items():
            record = WorkRecord.from_dict(raw_record)
            if work_id != record.work_id:
                raise SchedulerStateError("work map key differs from work_id")
            if (
                record.eligible_gpus is not None
                and not set(record.eligible_gpus).issubset(gpu_ids)
            ):
                raise SchedulerStateError("work names an unregistered eligible GPU")
            if record.preferred_gpu is not None and record.preferred_gpu not in gpu_ids:
                raise SchedulerStateError("work names an unregistered preferred GPU")
            records[work_id] = record
            max_sequence = max(max_sequence, record.enqueue_sequence)
            if record.last_release is not None:
                max_sequence = max(max_sequence, record.last_release.sequence)

        claims: Dict[str, Claim] = {}
        claimed_work: Dict[str, Claim] = {}
        claim_ids = set()
        for gpu_id, raw_claim in state["claims"].items():
            claim = Claim.from_dict(raw_claim)
            if gpu_id != claim.gpu_id or gpu_id not in gpu_ids:
                raise SchedulerStateError("claim GPU is unregistered or mismatched")
            if claim.claim_id in claim_ids or claim.work_id in claimed_work:
                raise SchedulerStateError("claim identity or claimed work is duplicated")
            claim_ids.add(claim.claim_id)
            claimed_work[claim.work_id] = claim
            claims[gpu_id] = claim
            max_sequence = max(max_sequence, claim.sequence)
            record = records.get(claim.work_id)
            if (
                record is None
                or record.state != WorkState.CLAIMED
                or record.active_claim_id != claim.claim_id
                or not record.item.is_eligible(gpu_id)
            ):
                raise SchedulerStateError("claim contradicts its work record")

        for record in records.values():
            claim = claimed_work.get(record.work_id)
            if record.state == WorkState.CLAIMED:
                if claim is None or record.active_claim_id != claim.claim_id:
                    raise SchedulerStateError("claimed work has no matching GPU claim")
            elif record.active_claim_id is not None or claim is not None:
                raise SchedulerStateError("unclaimed work retains an active claim")

        requests: Dict[str, PreemptionRequest] = {}
        for request_id, raw_request in state["preemption_requests"].items():
            request = PreemptionRequest.from_dict(raw_request)
            if request_id != request.request_id or request.gpu_id not in gpu_ids:
                raise SchedulerStateError("preemption request key or GPU mismatches")
            if (
                request.running_work_id not in records
                or request.requested_for_work_id not in records
            ):
                raise SchedulerStateError("preemption request names unknown work")
            if request.status == PreemptionStatus.PENDING:
                claim = claims.get(request.gpu_id)
                if (
                    claim is None
                    or claim.claim_id != request.claim_id
                    or claim.work_id != request.running_work_id
                    or not records[request.running_work_id].preemptible
                ):
                    raise SchedulerStateError(
                        "pending preemption request has no preemptible active claim"
                    )
            requests[request_id] = request
            max_sequence = max(max_sequence, request.sequence)

        events: Dict[str, IdleEvent] = {}
        previous_sequence = 0
        for raw_event in state["idle_events"]:
            event = IdleEvent.from_dict(raw_event)
            if event.gpu_id not in gpu_ids or event.event_id in events:
                raise SchedulerStateError("idle event GPU or identity is invalid")
            if event.sequence <= previous_sequence:
                raise SchedulerStateError("idle event history is not ordered")
            previous_sequence = event.sequence
            events[event.event_id] = event
            max_sequence = max(max_sequence, event.sequence)
        for gpu_id, event_id in state["idle"].items():
            if gpu_id not in gpu_ids or not isinstance(event_id, str):
                raise SchedulerStateError("current idle index is invalid")
            event = events.get(event_id)
            if event is None or event.gpu_id != gpu_id:
                raise SchedulerStateError("current idle index names the wrong event")

        self._validate_halt(state["safety_halt"], "safety_halt")
        for gpu_id, halt in state["gpu_safety_halts"].items():
            if gpu_id not in gpu_ids:
                raise SchedulerStateError("GPU safety halt names unknown GPU")
            self._validate_halt(halt, "gpu_safety_halt")

        if next_sequence <= max_sequence:
            raise SchedulerStateError(
                "next_sequence does not exceed all allocated sequences"
            )

    @staticmethod
    def _validate_halt(value: Any, field_name: str) -> None:
        if value is None:
            return
        if not isinstance(value, Mapping) or set(value) != {"reason", "set_at"}:
            raise SchedulerStateError(f"{field_name} must be null or a halt object")
        try:
            _require_identifier(value["reason"], f"{field_name}.reason")
            set_at = float(value["set_at"])
        except (TypeError, ValueError) as exc:
            raise SchedulerStateError(f"invalid {field_name}: {exc}") from exc
        if not math.isfinite(set_at):
            raise SchedulerStateError(f"{field_name}.set_at must be finite")

    def _snapshot(self, state: Mapping[str, Any]) -> SchedulerSnapshot:
        work = {
            work_id: WorkRecord.from_dict(raw)
            for work_id, raw in state["work"].items()
        }
        claims = {
            gpu_id: Claim.from_dict(raw)
            for gpu_id, raw in state["claims"].items()
        }
        requests = tuple(
            sorted(
                (
                    PreemptionRequest.from_dict(raw)
                    for raw in state["preemption_requests"].values()
                ),
                key=lambda request: request.sequence,
            )
        )
        history = tuple(IdleEvent.from_dict(raw) for raw in state["idle_events"])
        event_by_id = {event.event_id: event for event in history}
        global_halt = state["safety_halt"]
        return SchedulerSnapshot(
            revision=state["revision"],
            next_sequence=state["next_sequence"],
            gpu_ids=tuple(state["gpu_ids"]),
            dynamic_gpus=state["dynamic_gpus"],
            work=MappingProxyType(work),
            claims=MappingProxyType(claims),
            preemption_requests=requests,
            idle=MappingProxyType(
                {
                    gpu_id: event_by_id[event_id]
                    for gpu_id, event_id in state["idle"].items()
                }
            ),
            idle_history=history,
            safety_halt=(
                None if global_halt is None else global_halt["reason"]
            ),
            gpu_safety_halts=MappingProxyType(
                {
                    gpu_id: halt["reason"]
                    for gpu_id, halt in state["gpu_safety_halts"].items()
                }
            ),
            state_sha256=state["state_sha256"],
        )

    def _validate_or_register_item_gpus(
        self, state: Dict[str, Any], item: WorkItem
    ) -> None:
        named = set(item.eligible_gpus or ())
        if item.preferred_gpu is not None:
            named.add(item.preferred_gpu)
        for gpu_id in sorted(named):
            self._validate_or_register_gpu(state, gpu_id)

    @staticmethod
    def _validate_known_gpu(state: Mapping[str, Any], gpu_id: str) -> None:
        if gpu_id not in state["gpu_ids"]:
            raise UnknownGpuError(gpu_id)

    @staticmethod
    def _validate_or_register_gpu(state: Dict[str, Any], gpu_id: str) -> bool:
        if gpu_id in state["gpu_ids"]:
            return False
        if not state["dynamic_gpus"]:
            raise UnknownGpuError(gpu_id)
        state["gpu_ids"].append(gpu_id)
        state["gpu_ids"].sort()
        return True

    @staticmethod
    def _allocate_sequence(state: Dict[str, Any]) -> int:
        sequence = state["next_sequence"]
        state["next_sequence"] = sequence + 1
        return sequence

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise SchedulerError("clock returned a non-finite value")
        return now

    @staticmethod
    def _halt_reason(state: Mapping[str, Any], gpu_id: str) -> Optional[str]:
        global_halt = state["safety_halt"]
        if global_halt is not None:
            return global_halt["reason"]
        gpu_halt = state["gpu_safety_halts"].get(gpu_id)
        return None if gpu_halt is None else gpu_halt["reason"]

    def _record_idle_in_state(
        self,
        state: Dict[str, Any],
        gpu_id: str,
        reason: IdleReason,
        *,
        owner_id: Optional[str] = None,
        work_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[IdleEvent, bool]:
        normalized_details = _json_object(details, "idle details")
        active_raw = state["claims"].get(gpu_id)
        if active_raw is not None:
            active = Claim.from_dict(active_raw)
            if owner_id is not None and owner_id != active.owner_id:
                raise GpuBusyError(
                    f"GPU {gpu_id!r} is owned by {active.owner_id!r}, "
                    f"not telemetry actor {owner_id!r}"
                )
            if owner_id is None:
                owner_id = active.owner_id
            if work_id is None:
                work_id = active.work_id

        current_id = state["idle"].get(gpu_id)
        if current_id is not None:
            raw_current = next(
                (
                    raw
                    for raw in reversed(state["idle_events"])
                    if raw["event_id"] == current_id
                ),
                None,
            )
            if raw_current is None:
                raise SchedulerStateError("current idle event is missing")
            current = IdleEvent.from_dict(raw_current)
            if (
                current.reason == reason
                and current.owner_id == owner_id
                and current.work_id == work_id
                and current.details == normalized_details
            ):
                return current, False

        sequence = self._allocate_sequence(state)
        event = IdleEvent(
            event_id=f"idle-{sequence:020d}",
            gpu_id=gpu_id,
            reason=reason,
            recorded_at=self._now(),
            sequence=sequence,
            owner_id=owner_id,
            work_id=work_id,
            details=normalized_details,
        )
        state["idle_events"].append(event.to_dict())
        state["idle"][gpu_id] = event.event_id
        return event, True

    @staticmethod
    def _matching_release_retry(
        state: Mapping[str, Any],
        *,
        gpu_id: str,
        owner_id: str,
        work_id: Optional[str],
        expected_claim_id: Optional[str],
        outcome: ReleaseOutcome,
    ) -> Optional[ReleaseRecord]:
        matches: List[ReleaseRecord] = []
        for candidate_id, raw_record in state["work"].items():
            if work_id is not None and candidate_id != work_id:
                continue
            record = WorkRecord.from_dict(raw_record)
            release = record.last_release
            if (
                release is not None
                and release.gpu_id == gpu_id
                and release.owner_id == owner_id
                and release.outcome == outcome
                and (
                    expected_claim_id is None
                    or release.claim_id == expected_claim_id
                )
            ):
                matches.append(release)
        if not matches:
            return None
        return max(matches, key=lambda release: release.sequence)


# Short aliases keep the standalone core convenient without coupling it to
# another controller module.
Scheduler = ClusterScheduler
SchedulerCore = ClusterScheduler


__all__ = [
    "ClusterScheduler",
    "GpuBusyError",
    "IdleEvent",
    "IdleReason",
    "LOCK_FILENAME",
    "MAX_STATE_BYTES",
    "NoActiveClaimError",
    "PRIORITY_ORDER",
    "PRIORITY_RANK",
    "PRIORITY_TIERS",
    "PreemptionRequest",
    "PreemptionStatus",
    "ReleaseOutcome",
    "ReleaseRecord",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerConflictError",
    "SchedulerCore",
    "SchedulerError",
    "SchedulerSnapshot",
    "SchedulerStateError",
    "UnknownGpuError",
    "WORK_PRIORITY",
    "WorkItem",
    "WorkKind",
    "WorkRecord",
    "WorkState",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "priority_rank",
]
