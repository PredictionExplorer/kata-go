"""Crash-safe lifecycle state for checkpoint evaluation and promotion.

The immutable event log in ``promotion/events`` is authoritative.  The
``champion.json`` file is a small compare-and-swap projection used by self-play
launchers.  Callers must hold :class:`ControllerLock` while mutating either.

Only Python's standard library is used.  The persistence helpers rely on POSIX
``flock``, hard links, atomic rename, and directory ``fsync`` semantics.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union


PathLike = Union[str, os.PathLike]

__all__ = [
    "CandidateRecord",
    "CandidateState",
    "ChampionConflictError",
    "ChampionRecord",
    "ControllerLock",
    "ControllerLockError",
    "EventConflictError",
    "EventProvenance",
    "EventRegistry",
    "FinalizedPassReport",
    "GenerationRecord",
    "GenerationState",
    "IllegalTransitionError",
    "PassReportError",
    "PromotionEvent",
    "PromotionStateError",
    "ReferencePin",
    "RegistryCorruptionError",
    "RegistryState",
    "RetentionStatus",
    "RolloutState",
    "StaleChampionError",
    "Transition",
    "atomic_write_bytes",
    "atomic_write_json",
    "bootstrap_champion",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "compare_and_swap_champion",
    "fsync_directory",
    "load_champion",
    "load_finalized_pass_report",
    "sha256_bytes",
    "sha256_file",
    "utc_timestamp",
]

EVENT_SCHEMA_VERSION = 1
CHAMPION_SCHEMA_VERSION = 1
EVENT_SEQUENCE_WIDTH = 20
GENESIS_HASH = "0" * 64
MAX_JSON_BYTES = 64 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_FILE_RE = re.compile(r"^([0-9]{20})\.json$")


class PromotionStateError(Exception):
    """Base class for lifecycle persistence failures."""


class RegistryCorruptionError(PromotionStateError, ValueError):
    """The immutable event registry is malformed or internally inconsistent."""


class IllegalTransitionError(RegistryCorruptionError):
    """A requested or replayed lifecycle transition is not legal."""


class EventConflictError(PromotionStateError, FileExistsError):
    """An immutable event destination already exists."""


class ControllerLockError(PromotionStateError, RuntimeError):
    """The single-writer controller lock could not be acquired."""


class ChampionConflictError(PromotionStateError, RuntimeError):
    """Champion state conflicts with the requested bootstrap or retry."""


class StaleChampionError(ChampionConflictError):
    """The champion no longer matches the SHA tested by an evaluation."""


class PassReportError(PromotionStateError, ValueError):
    """A purported finalized PASS report is missing or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a JSON-compatible value.

    Object keys are sorted, insignificant whitespace is removed, non-ASCII
    text remains UTF-8, and non-finite numbers are rejected.
    """

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON compatible: {exc}") from exc
    return text.encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON text; see :func:`canonical_json_bytes`."""

    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON encoding of *value*."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its lowercase SHA-256 digest."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp(now: Optional[datetime] = None) -> str:
    """Return an ISO-8601 UTC timestamp with a ``Z`` suffix."""

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("timestamp datetime must be timezone-aware")
    current = current.astimezone(timezone.utc)
    return current.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_utc_timestamp(value: str, field: str = "timestamp_utc") -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be UTC")
    return value


def _require_sha256(value: Any, field: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field} may not contain NUL")
    return value


def _copy_json_object(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    candidate: Any = {} if value is None else value
    encoded = canonical_json_bytes(candidate)
    copied = json.loads(encoded.decode("utf-8"))
    if not isinstance(copied, dict):
        raise ValueError("payload must be a JSON object")
    return copied


def fsync_directory(path: PathLike) -> None:
    """Durably persist directory-entry changes for an existing directory."""

    directory = os.fspath(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_temp_file(parent: Path, name: str, data: bytes, mode: int) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=os.fspath(parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as output:
            fd = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary


def atomic_write_bytes(path: PathLike, data: bytes, *, mode: int = 0o644) -> None:
    """Atomically replace a file using a unique, fsynced sibling temporary.

    The destination's parent must already exist.  After ``os.replace``, the
    parent directory is fsynced so the rename is durable across a crash.
    """

    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {parent}")
    temporary = _write_temp_file(parent, destination.name, data, mode)
    try:
        os.replace(os.fspath(temporary), os.fspath(destination))
        fsync_directory(parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: PathLike, value: Any, *, mode: int = 0o644) -> None:
    """Atomically replace *path* with newline-terminated canonical JSON."""

    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", mode=mode)


def _atomic_create_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Atomically create *path* without ever replacing an existing file.

    A hard link publishes the fully fsynced temporary file.  POSIX guarantees
    that linking fails with ``EEXIST`` rather than overwriting the destination.
    """

    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {parent}")
    temporary = _write_temp_file(parent, path.name, data, mode)
    try:
        try:
            os.link(
                os.fspath(temporary),
                os.fspath(path),
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise EventConflictError(f"immutable file already exists: {path}") from exc
        temporary.unlink()
        fsync_directory(parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _decode_json(data: bytes, path: Path) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RegistryCorruptionError(f"invalid JSON in {path}: {exc}") from exc


def _read_regular_file(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RegistryCorruptionError(f"expected a regular non-symlink file: {path}")
    if metadata.st_size > maximum_bytes:
        raise RegistryCorruptionError(f"JSON file exceeds size limit: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(os.fspath(path), flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RegistryCorruptionError(f"expected a regular file: {path}")
        chunks = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum_bytes:
            raise RegistryCorruptionError(f"JSON file exceeds size limit: {path}")
        return data
    finally:
        os.close(fd)


class CandidateState(str, Enum):
    """Candidate intake and evaluation states."""

    DISCOVERED = "discovered"
    CLAIMED = "claimed"
    SUPERSEDED = "superseded"
    EVALUATING_INTEGRITY = "evaluating_integrity"
    EVALUATING_SCREEN = "evaluating_screen"
    EVALUATING_FINALIST = "evaluating_finalist"
    EVALUATING_CONFIRMATION = "evaluating_confirmation"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    CONFIRMED = "confirmed"


class GenerationState(str, Enum):
    """Promotion, rollout, activation, and rollback states."""

    PROMOTION_INTENT = "promotion_intent"
    CANARY = "canary"
    ROLLOUT = "rollout"
    ACTIVE = "active"
    ROLLBACK_PENDING = "rollback_pending"
    ROLLED_BACK = "rolled_back"
    QUARANTINED = "quarantined"


# A descriptive alias for controller code that models rollout state explicitly.
RolloutState = GenerationState


class Transition(str, Enum):
    """Names persisted in immutable lifecycle events."""

    CHAMPION_BOOTSTRAPPED = "champion.bootstrapped"
    CANDIDATE_DISCOVERED = "candidate.discovered"
    CANDIDATE_CLAIMED = "candidate.claimed"
    CANDIDATE_SUPERSEDED = "candidate.superseded"
    EVALUATION_INTEGRITY_STARTED = "evaluation.integrity_started"
    EVALUATION_SCREEN_STARTED = "evaluation.screen_started"
    EVALUATION_FINALIST_STARTED = "evaluation.finalist_started"
    EVALUATION_CONFIRMATION_STARTED = "evaluation.confirmation_started"
    CANDIDATE_REJECTED = "candidate.rejected"
    CANDIDATE_QUARANTINED = "candidate.quarantined"
    CANDIDATE_CONFIRMED = "candidate.confirmed"
    GENERATION_PROMOTION_INTENT = "generation.promotion_intent"
    GENERATION_CANARY_STARTED = "generation.canary_started"
    GENERATION_ROLLOUT_STARTED = "generation.rollout_started"
    GENERATION_ACTIVATED = "generation.activated"
    GENERATION_ROLLBACK_STARTED = "generation.rollback_started"
    GENERATION_ROLLED_BACK = "generation.rolled_back"
    GENERATION_QUARANTINED = "generation.quarantined"
    REFERENCE_PINNED = "reference.pinned"
    REFERENCE_UNPINNED = "reference.unpinned"


_CANDIDATE_TARGETS = {
    Transition.CANDIDATE_DISCOVERED: CandidateState.DISCOVERED,
    Transition.CANDIDATE_CLAIMED: CandidateState.CLAIMED,
    Transition.CANDIDATE_SUPERSEDED: CandidateState.SUPERSEDED,
    Transition.EVALUATION_INTEGRITY_STARTED: CandidateState.EVALUATING_INTEGRITY,
    Transition.EVALUATION_SCREEN_STARTED: CandidateState.EVALUATING_SCREEN,
    Transition.EVALUATION_FINALIST_STARTED: CandidateState.EVALUATING_FINALIST,
    Transition.EVALUATION_CONFIRMATION_STARTED: CandidateState.EVALUATING_CONFIRMATION,
    Transition.CANDIDATE_REJECTED: CandidateState.REJECTED,
    Transition.CANDIDATE_QUARANTINED: CandidateState.QUARANTINED,
    Transition.CANDIDATE_CONFIRMED: CandidateState.CONFIRMED,
}

_STATE_TO_CANDIDATE_TRANSITION = {
    state: transition for transition, state in _CANDIDATE_TARGETS.items()
}

_GENERATION_TARGETS = {
    Transition.GENERATION_PROMOTION_INTENT: GenerationState.PROMOTION_INTENT,
    Transition.GENERATION_CANARY_STARTED: GenerationState.CANARY,
    Transition.GENERATION_ROLLOUT_STARTED: GenerationState.ROLLOUT,
    Transition.GENERATION_ACTIVATED: GenerationState.ACTIVE,
    Transition.GENERATION_ROLLBACK_STARTED: GenerationState.ROLLBACK_PENDING,
    Transition.GENERATION_ROLLED_BACK: GenerationState.ROLLED_BACK,
    Transition.GENERATION_QUARANTINED: GenerationState.QUARANTINED,
}

_STATE_TO_GENERATION_TRANSITION = {
    state: transition for transition, state in _GENERATION_TARGETS.items()
}

_LEGAL_CANDIDATE_TRANSITIONS = {
    CandidateState.DISCOVERED: {
        CandidateState.CLAIMED,
        CandidateState.SUPERSEDED,
        CandidateState.QUARANTINED,
    },
    CandidateState.CLAIMED: {
        CandidateState.SUPERSEDED,
        CandidateState.EVALUATING_INTEGRITY,
        CandidateState.QUARANTINED,
    },
    CandidateState.EVALUATING_INTEGRITY: {
        CandidateState.EVALUATING_SCREEN,
        CandidateState.SUPERSEDED,
        CandidateState.REJECTED,
        CandidateState.QUARANTINED,
    },
    CandidateState.EVALUATING_SCREEN: {
        CandidateState.EVALUATING_FINALIST,
        CandidateState.SUPERSEDED,
        CandidateState.REJECTED,
        CandidateState.QUARANTINED,
    },
    CandidateState.EVALUATING_FINALIST: {
        CandidateState.EVALUATING_CONFIRMATION,
        CandidateState.SUPERSEDED,
        CandidateState.REJECTED,
        CandidateState.QUARANTINED,
    },
    CandidateState.EVALUATING_CONFIRMATION: {
        CandidateState.CONFIRMED,
        CandidateState.REJECTED,
        CandidateState.QUARANTINED,
    },
    CandidateState.SUPERSEDED: set(),
    CandidateState.REJECTED: set(),
    CandidateState.QUARANTINED: set(),
    CandidateState.CONFIRMED: set(),
}

_LEGAL_GENERATION_TRANSITIONS = {
    GenerationState.PROMOTION_INTENT: {
        GenerationState.CANARY,
        GenerationState.ROLLBACK_PENDING,
        GenerationState.QUARANTINED,
    },
    GenerationState.CANARY: {
        GenerationState.ROLLOUT,
        GenerationState.ROLLBACK_PENDING,
        GenerationState.QUARANTINED,
    },
    GenerationState.ROLLOUT: {
        GenerationState.ACTIVE,
        GenerationState.ROLLBACK_PENDING,
        GenerationState.QUARANTINED,
    },
    GenerationState.ACTIVE: {GenerationState.ROLLBACK_PENDING},
    GenerationState.ROLLBACK_PENDING: {
        GenerationState.ROLLED_BACK,
        GenerationState.QUARANTINED,
    },
    GenerationState.ROLLED_BACK: set(),
    GenerationState.QUARANTINED: set(),
}

_EVALUATING_STATES = {
    CandidateState.EVALUATING_INTEGRITY,
    CandidateState.EVALUATING_SCREEN,
    CandidateState.EVALUATING_FINALIST,
    CandidateState.EVALUATING_CONFIRMATION,
}


@dataclass(frozen=True)
class EventProvenance:
    """Content hashes required on every event and champion update."""

    controller_hash: str
    source_hash: str
    original_hash: str
    config_hash: str
    schedule_hash: str
    policy_hash: str

    def __post_init__(self) -> None:
        for field in (
            "controller_hash",
            "source_hash",
            "original_hash",
            "config_hash",
            "schedule_hash",
            "policy_hash",
        ):
            _require_sha256(getattr(self, field), field)


@dataclass(frozen=True)
class PromotionEvent:
    """One immutable, hash-chained lifecycle event."""

    schema_version: int
    sequence: int
    previous_hash: str
    timestamp_utc: str
    controller_hash: str
    source_hash: str
    candidate_hash: Optional[str]
    candidate_path: Optional[str]
    champion_hash: str
    original_hash: str
    transition: Transition
    evaluation_key: Optional[str]
    config_hash: str
    schedule_hash: str
    policy_hash: str
    reason: str
    actor: str
    payload: Mapping[str, Any]
    event_hash: str

    def body_dict(self) -> Dict[str, Any]:
        """Return the canonical event fields excluding the self-hash."""

        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "previous_hash": self.previous_hash,
            "timestamp_utc": self.timestamp_utc,
            "controller_hash": self.controller_hash,
            "source_hash": self.source_hash,
            "candidate_hash": self.candidate_hash,
            "candidate_path": self.candidate_path,
            "champion_hash": self.champion_hash,
            "original_hash": self.original_hash,
            "transition": self.transition.value,
            "evaluation_key": self.evaluation_key,
            "config_hash": self.config_hash,
            "schedule_hash": self.schedule_hash,
            "policy_hash": self.policy_hash,
            "reason": self.reason,
            "actor": self.actor,
            "payload": _copy_json_object(self.payload),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return all serializable fields, including ``event_hash``."""

        result = self.body_dict()
        result["event_hash"] = self.event_hash
        return result

    def verify(self) -> None:
        """Validate field types and the canonical self-hash."""

        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != EVENT_SCHEMA_VERSION
        ):
            raise RegistryCorruptionError(
                f"unsupported event schema version: {self.schema_version}"
            )
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise RegistryCorruptionError("event sequence must be an integer")
        if self.sequence < 1:
            raise RegistryCorruptionError("event sequence must be positive")
        if self.sequence >= 10**EVENT_SEQUENCE_WIDTH:
            raise RegistryCorruptionError(
                f"event sequence exceeds {EVENT_SEQUENCE_WIDTH}-digit filename width"
            )
        try:
            _require_sha256(self.previous_hash, "previous_hash")
            _validate_utc_timestamp(self.timestamp_utc)
            _require_sha256(self.controller_hash, "controller_hash")
            _require_sha256(self.source_hash, "source_hash")
            _require_sha256(self.champion_hash, "champion_hash")
            _require_sha256(self.original_hash, "original_hash")
            _require_sha256(self.config_hash, "config_hash")
            _require_sha256(self.schedule_hash, "schedule_hash")
            _require_sha256(self.policy_hash, "policy_hash")
            _require_sha256(self.event_hash, "event_hash")
            _require_nonempty(self.reason, "reason")
            _require_nonempty(self.actor, "actor")
            if self.evaluation_key is not None:
                _require_nonempty(self.evaluation_key, "evaluation_key")
            if (self.candidate_hash is None) != (self.candidate_path is None):
                raise ValueError(
                    "candidate_hash and candidate_path must both be set or both be null"
                )
            if self.candidate_hash is not None:
                _require_sha256(self.candidate_hash, "candidate_hash")
                _require_nonempty(self.candidate_path, "candidate_path")
            _copy_json_object(self.payload)
        except ValueError as exc:
            raise RegistryCorruptionError(str(exc)) from exc
        expected = canonical_sha256(self.body_dict())
        if self.event_hash != expected:
            raise RegistryCorruptionError(
                f"event {self.sequence} hash mismatch: "
                f"stored {self.event_hash}, computed {expected}"
            )

    @classmethod
    def build(
        cls,
        *,
        sequence: int,
        previous_hash: str,
        timestamp_utc: str,
        provenance: EventProvenance,
        candidate_hash: Optional[str],
        candidate_path: Optional[str],
        champion_hash: str,
        transition: Transition,
        evaluation_key: Optional[str],
        reason: str,
        actor: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> "PromotionEvent":
        """Construct and self-hash an event."""

        body = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": sequence,
            "previous_hash": previous_hash,
            "timestamp_utc": timestamp_utc,
            "controller_hash": provenance.controller_hash,
            "source_hash": provenance.source_hash,
            "candidate_hash": candidate_hash,
            "candidate_path": candidate_path,
            "champion_hash": champion_hash,
            "original_hash": provenance.original_hash,
            "transition": transition.value,
            "evaluation_key": evaluation_key,
            "config_hash": provenance.config_hash,
            "schedule_hash": provenance.schedule_hash,
            "policy_hash": provenance.policy_hash,
            "reason": reason,
            "actor": actor,
            "payload": _copy_json_object(payload),
        }
        event = cls(
            schema_version=EVENT_SCHEMA_VERSION,
            sequence=sequence,
            previous_hash=previous_hash,
            timestamp_utc=timestamp_utc,
            controller_hash=provenance.controller_hash,
            source_hash=provenance.source_hash,
            candidate_hash=candidate_hash,
            candidate_path=candidate_path,
            champion_hash=champion_hash,
            original_hash=provenance.original_hash,
            transition=transition,
            evaluation_key=evaluation_key,
            config_hash=provenance.config_hash,
            schedule_hash=provenance.schedule_hash,
            policy_hash=provenance.policy_hash,
            reason=reason,
            actor=actor,
            payload=body["payload"],
            event_hash=canonical_sha256(body),
        )
        event.verify()
        return event

    @classmethod
    def from_dict(cls, value: Any) -> "PromotionEvent":
        """Parse a version-1 event object and reject unknown fields."""

        if not isinstance(value, dict):
            raise RegistryCorruptionError("event JSON root must be an object")
        expected_fields = {
            "schema_version",
            "sequence",
            "previous_hash",
            "timestamp_utc",
            "controller_hash",
            "source_hash",
            "candidate_hash",
            "candidate_path",
            "champion_hash",
            "original_hash",
            "transition",
            "evaluation_key",
            "config_hash",
            "schedule_hash",
            "policy_hash",
            "reason",
            "actor",
            "payload",
            "event_hash",
        }
        if set(value) != expected_fields:
            missing = sorted(expected_fields - set(value))
            extra = sorted(set(value) - expected_fields)
            raise RegistryCorruptionError(
                f"event fields differ from schema; missing={missing}, extra={extra}"
            )
        try:
            transition = Transition(value["transition"])
        except (TypeError, ValueError) as exc:
            raise RegistryCorruptionError(
                f"unknown event transition: {value.get('transition')!r}"
            ) from exc
        event = cls(
            schema_version=value["schema_version"],
            sequence=value["sequence"],
            previous_hash=value["previous_hash"],
            timestamp_utc=value["timestamp_utc"],
            controller_hash=value["controller_hash"],
            source_hash=value["source_hash"],
            candidate_hash=value["candidate_hash"],
            candidate_path=value["candidate_path"],
            champion_hash=value["champion_hash"],
            original_hash=value["original_hash"],
            transition=transition,
            evaluation_key=value["evaluation_key"],
            config_hash=value["config_hash"],
            schedule_hash=value["schedule_hash"],
            policy_hash=value["policy_hash"],
            reason=value["reason"],
            actor=value["actor"],
            payload=value["payload"],
            event_hash=value["event_hash"],
        )
        event.verify()
        return event


@dataclass(frozen=True)
class CandidateRecord:
    """Current state reconstructed for one candidate SHA."""

    candidate_hash: str
    candidate_path: str
    state: CandidateState
    parent_champion_hash: str
    tested_champion_hash: Optional[str]
    evaluation_key: Optional[str]
    generation_id: Optional[str]
    last_sequence: int


@dataclass(frozen=True)
class GenerationRecord:
    """Current rollout state reconstructed for one generation."""

    generation_id: str
    candidate_hash: str
    candidate_path: Optional[str]
    previous_champion_hash: Optional[str]
    evaluation_key: Optional[str]
    state: GenerationState
    restore_champion_hash: Optional[str]
    last_sequence: int


@dataclass(frozen=True)
class ReferencePin:
    """An explicit retention pin reconstructed from pin/unpin events."""

    pin_id: str
    reference_hash: str
    kind: str
    owner: str
    reason: str
    created_sequence: int


@dataclass(frozen=True)
class RetentionStatus:
    """Reasons a content hash must be retained."""

    reference_hash: str
    reasons: Tuple[str, ...]

    @property
    def pinned(self) -> bool:
        """Whether any explicit or implicit reference protects the hash."""

        return bool(self.reasons)

    @property
    def safe_to_delete(self) -> bool:
        """Whether lifecycle state has no known live reference to the hash."""

        return not self.reasons


@dataclass(frozen=True)
class RegistryState:
    """A complete in-memory projection rebuilt solely from immutable events."""

    events: Tuple[PromotionEvent, ...]
    candidates: Mapping[str, CandidateRecord]
    generations: Mapping[str, GenerationRecord]
    pins: Mapping[str, ReferencePin]
    current_champion_hash: Optional[str]
    current_generation_id: Optional[str]
    original_hash: Optional[str]
    last_sequence: int
    last_event_hash: str

    def retention_status(self, reference_hash: str) -> RetentionStatus:
        """Return all registry reasons preventing deletion of a content hash."""

        _require_sha256(reference_hash, "reference_hash")
        reasons = set()
        if reference_hash == self.original_hash:
            reasons.add("immutable-original")
        if reference_hash == self.current_champion_hash:
            reasons.add("current-champion")

        protected_candidate_states = {
            CandidateState.DISCOVERED,
            CandidateState.CLAIMED,
            CandidateState.EVALUATING_INTEGRITY,
            CandidateState.EVALUATING_SCREEN,
            CandidateState.EVALUATING_FINALIST,
            CandidateState.EVALUATING_CONFIRMATION,
            CandidateState.QUARANTINED,
        }
        for candidate in self.candidates.values():
            if (
                candidate.candidate_hash == reference_hash
                and (
                    candidate.state in protected_candidate_states
                    or (
                        candidate.state == CandidateState.CONFIRMED
                        and candidate.generation_id is None
                    )
                )
            ):
                reasons.add(f"candidate:{candidate.state.value}")

        live_generation_states = {
            GenerationState.PROMOTION_INTENT,
            GenerationState.CANARY,
            GenerationState.ROLLOUT,
            GenerationState.ACTIVE,
            GenerationState.ROLLBACK_PENDING,
            GenerationState.QUARANTINED,
        }
        for generation in self.generations.values():
            if generation.state not in live_generation_states:
                continue
            if (
                generation.state == GenerationState.ACTIVE
                and generation.generation_id != self.current_generation_id
            ):
                # Historical generations retain ACTIVE as their terminal
                # activation fact. Only the current one is implicitly live;
                # newer generations separately pin their rollback champion.
                continue
            if generation.candidate_hash == reference_hash:
                reasons.add(f"generation:{generation.generation_id}:candidate")
            if generation.previous_champion_hash == reference_hash:
                reasons.add(
                    f"generation:{generation.generation_id}:rollback-champion"
                )

        for pin in self.pins.values():
            if pin.reference_hash == reference_hash:
                reasons.add(f"pin:{pin.pin_id}:{pin.kind}:{pin.owner}")
        return RetentionStatus(reference_hash, tuple(sorted(reasons)))

    def is_pinned(self, reference_hash: str) -> bool:
        """Return whether a hash has any live retention reference."""

        return self.retention_status(reference_hash).pinned

    def can_delete(self, reference_hash: str) -> bool:
        """Return whether deletion is reference-safe; this never deletes data."""

        return self.retention_status(reference_hash).safe_to_delete

    def retained_hashes(self) -> Mapping[str, RetentionStatus]:
        """Return retention information for every referenced content hash."""

        hashes = set()
        if self.original_hash is not None:
            hashes.add(self.original_hash)
        if self.current_champion_hash is not None:
            hashes.add(self.current_champion_hash)
        hashes.update(self.candidates)
        for generation in self.generations.values():
            hashes.add(generation.candidate_hash)
            if generation.previous_champion_hash is not None:
                hashes.add(generation.previous_champion_hash)
        hashes.update(pin.reference_hash for pin in self.pins.values())
        return MappingProxyType(
            {
                value: status
                for value in sorted(hashes)
                if (status := self.retention_status(value)).pinned
            }
        )


class _RegistryBuilder:
    def __init__(self) -> None:
        self.candidates: Dict[str, CandidateRecord] = {}
        self.generations: Dict[str, GenerationRecord] = {}
        self.pins: Dict[str, ReferencePin] = {}
        self.current_champion_hash: Optional[str] = None
        self.current_generation_id: Optional[str] = None
        self.original_hash: Optional[str] = None
        self.path_to_hash: Dict[str, str] = {}
        self.seen_pin_ids = set()

    @staticmethod
    def _generation_id(event: PromotionEvent) -> str:
        generation_id = event.payload.get("generation_id")
        try:
            return _require_nonempty(generation_id, "payload.generation_id")
        except ValueError as exc:
            raise IllegalTransitionError(str(exc)) from exc

    @staticmethod
    def _restore_hash(event: PromotionEvent) -> str:
        restore_hash = event.payload.get("restore_champion_hash")
        try:
            result = _require_sha256(
                restore_hash, "payload.restore_champion_hash"
            )
        except ValueError as exc:
            raise IllegalTransitionError(str(exc)) from exc
        assert result is not None
        return result

    def _validate_candidate_identity(self, event: PromotionEvent) -> None:
        if event.candidate_hash is None or event.candidate_path is None:
            raise IllegalTransitionError(
                f"{event.transition.value} requires candidate_hash and candidate_path"
            )
        known_hash = self.path_to_hash.get(event.candidate_path)
        if known_hash is not None and known_hash != event.candidate_hash:
            raise RegistryCorruptionError(
                f"candidate path {event.candidate_path!r} was recorded with both "
                f"{known_hash} and {event.candidate_hash}"
            )
        current = self.candidates.get(event.candidate_hash)
        previous_path = event.payload.get("previous_candidate_path")
        if current is None:
            if previous_path is not None:
                raise RegistryCorruptionError(
                    "first candidate event may not claim a previous path"
                )
        elif event.candidate_path != current.candidate_path:
            if previous_path != current.candidate_path:
                raise RegistryCorruptionError(
                    f"candidate hash {event.candidate_hash} moved from "
                    f"{current.candidate_path!r} to {event.candidate_path!r} "
                    "without matching previous_candidate_path"
                )
        elif previous_path is not None:
            raise RegistryCorruptionError(
                "previous_candidate_path is present without a path move"
            )
        self.path_to_hash[event.candidate_path] = event.candidate_hash

    def apply(self, event: PromotionEvent) -> None:
        if self.original_hash is None:
            self.original_hash = event.original_hash
        elif self.original_hash != event.original_hash:
            raise RegistryCorruptionError(
                f"event {event.sequence} changes immutable original hash"
            )

        transition = event.transition
        if transition == Transition.CHAMPION_BOOTSTRAPPED:
            self._apply_bootstrap(event)
            return
        if self.current_champion_hash is None:
            raise IllegalTransitionError(
                "champion.bootstrap must precede lifecycle mutations"
            )
        if transition in _CANDIDATE_TARGETS:
            self._apply_candidate(event)
        elif transition in _GENERATION_TARGETS:
            self._apply_generation(event)
        elif transition == Transition.REFERENCE_PINNED:
            self._apply_pin(event)
        elif transition == Transition.REFERENCE_UNPINNED:
            self._apply_unpin(event)
        else:
            raise IllegalTransitionError(f"unsupported transition: {transition.value}")

    def _apply_bootstrap(self, event: PromotionEvent) -> None:
        if (
            self.current_champion_hash is not None
            or self.generations
            or self.candidates
            or self.pins
            or event.sequence != 1
        ):
            raise IllegalTransitionError("champion may be bootstrapped only once first")
        if event.candidate_hash is not None or event.candidate_path is not None:
            raise IllegalTransitionError("champion bootstrap has no candidate identity")
        generation_id = self._generation_id(event)
        self.current_champion_hash = event.champion_hash
        self.current_generation_id = generation_id
        self.generations[generation_id] = GenerationRecord(
            generation_id=generation_id,
            candidate_hash=event.champion_hash,
            candidate_path=None,
            previous_champion_hash=None,
            evaluation_key=None,
            state=GenerationState.ACTIVE,
            restore_champion_hash=None,
            last_sequence=event.sequence,
        )

    def _apply_candidate(self, event: PromotionEvent) -> None:
        self._validate_candidate_identity(event)
        assert event.candidate_hash is not None
        assert event.candidate_path is not None
        if event.champion_hash != self.current_champion_hash:
            raise IllegalTransitionError(
                f"{event.transition.value} names stale champion "
                f"{event.champion_hash}; current is {self.current_champion_hash}"
            )

        target = _CANDIDATE_TARGETS[event.transition]
        current = self.candidates.get(event.candidate_hash)
        if target == CandidateState.DISCOVERED:
            if current is not None:
                raise IllegalTransitionError("candidate may be discovered only once")
            self.candidates[event.candidate_hash] = CandidateRecord(
                candidate_hash=event.candidate_hash,
                candidate_path=event.candidate_path,
                state=CandidateState.DISCOVERED,
                parent_champion_hash=event.champion_hash,
                tested_champion_hash=None,
                evaluation_key=None,
                generation_id=None,
                last_sequence=event.sequence,
            )
            return

        if current is None:
            raise IllegalTransitionError(
                f"candidate {event.candidate_hash} was not discovered"
            )
        if target not in _LEGAL_CANDIDATE_TRANSITIONS[current.state]:
            raise IllegalTransitionError(
                f"illegal candidate transition {current.state.value} -> {target.value}"
            )

        tested_champion = current.tested_champion_hash
        evaluation_key = current.evaluation_key
        if target in _EVALUATING_STATES:
            if event.evaluation_key is None:
                raise IllegalTransitionError(
                    f"{target.value} requires a non-empty evaluation_key"
                )
            tested_champion = event.champion_hash
            evaluation_key = event.evaluation_key
        elif target == CandidateState.CONFIRMED:
            if (
                event.evaluation_key is None
                or event.evaluation_key != current.evaluation_key
                or event.champion_hash != current.tested_champion_hash
            ):
                raise IllegalTransitionError(
                    "confirmation must match the active confirmation evaluation"
                )
        elif event.evaluation_key is not None and event.evaluation_key != evaluation_key:
            raise IllegalTransitionError(
                "terminal transition evaluation_key does not match active evaluation"
            )

        self.candidates[event.candidate_hash] = replace(
            current,
            candidate_path=event.candidate_path,
            state=target,
            tested_champion_hash=tested_champion,
            evaluation_key=evaluation_key,
            last_sequence=event.sequence,
        )

    def _apply_generation(self, event: PromotionEvent) -> None:
        self._validate_candidate_identity(event)
        assert event.candidate_hash is not None
        assert event.candidate_path is not None
        generation_id = self._generation_id(event)
        target = _GENERATION_TARGETS[event.transition]
        generation = self.generations.get(generation_id)

        if target == GenerationState.PROMOTION_INTENT:
            candidate = self.candidates.get(event.candidate_hash)
            if candidate is None or candidate.state != CandidateState.CONFIRMED:
                raise IllegalTransitionError(
                    "promotion intent requires a confirmed candidate"
                )
            if generation is not None:
                raise IllegalTransitionError("generation may have only one intent")
            if candidate.generation_id is not None:
                raise IllegalTransitionError(
                    "candidate is already assigned to a generation"
                )
            if (
                event.champion_hash != self.current_champion_hash
                or candidate.tested_champion_hash != event.champion_hash
                or candidate.evaluation_key != event.evaluation_key
            ):
                raise IllegalTransitionError(
                    "promotion intent must use the champion and evaluation "
                    "validated by confirmation"
                )
            self.generations[generation_id] = GenerationRecord(
                generation_id=generation_id,
                candidate_hash=event.candidate_hash,
                candidate_path=event.candidate_path,
                previous_champion_hash=event.champion_hash,
                evaluation_key=event.evaluation_key,
                state=target,
                restore_champion_hash=None,
                last_sequence=event.sequence,
            )
            self.candidates[event.candidate_hash] = replace(
                candidate,
                candidate_path=event.candidate_path,
                generation_id=generation_id,
                last_sequence=event.sequence,
            )
            return

        if generation is None:
            raise IllegalTransitionError(f"unknown generation: {generation_id}")
        if (
            generation.candidate_hash != event.candidate_hash
            or generation.candidate_path != event.candidate_path
        ):
            raise RegistryCorruptionError(
                f"generation {generation_id} candidate identity changed"
            )
        if event.champion_hash != generation.previous_champion_hash:
            raise IllegalTransitionError(
                f"generation {generation_id} previous champion changed"
            )
        if (
            event.evaluation_key is not None
            and event.evaluation_key != generation.evaluation_key
        ):
            raise IllegalTransitionError(
                f"generation {generation_id} evaluation_key changed"
            )
        if target not in _LEGAL_GENERATION_TRANSITIONS[generation.state]:
            raise IllegalTransitionError(
                f"illegal generation transition "
                f"{generation.state.value} -> {target.value}"
            )

        restore_hash = generation.restore_champion_hash
        if target in {
            GenerationState.ROLLBACK_PENDING,
            GenerationState.ROLLED_BACK,
        }:
            supplied_restore = self._restore_hash(event)
            if supplied_restore != generation.previous_champion_hash:
                raise IllegalTransitionError(
                    "rollback target must be the generation's previous champion"
                )
            if restore_hash is not None and restore_hash != supplied_restore:
                raise IllegalTransitionError("rollback target changed during recovery")
            restore_hash = supplied_restore

        if target == GenerationState.ACTIVE:
            if self.current_champion_hash != generation.previous_champion_hash:
                raise IllegalTransitionError(
                    "activation compare-and-swap champion is stale"
                )
            self.current_champion_hash = generation.candidate_hash
            self.current_generation_id = generation_id
        elif target == GenerationState.ROLLED_BACK:
            if self.current_champion_hash not in {
                generation.candidate_hash,
                generation.previous_champion_hash,
            }:
                raise IllegalTransitionError(
                    "cannot roll back over an unrelated active champion"
                )
            self.current_champion_hash = restore_hash
            self.current_generation_id = self._active_generation_for(restore_hash)

        self.generations[generation_id] = replace(
            generation,
            state=target,
            restore_champion_hash=restore_hash,
            last_sequence=event.sequence,
        )

    def _active_generation_for(self, champion_hash: Optional[str]) -> Optional[str]:
        if champion_hash is None:
            return None
        matches = [
            generation
            for generation in self.generations.values()
            if generation.candidate_hash == champion_hash
            and generation.state == GenerationState.ACTIVE
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: item.last_sequence).generation_id

    def _apply_pin(self, event: PromotionEvent) -> None:
        if event.candidate_hash is not None or event.candidate_path is not None:
            raise IllegalTransitionError("reference pin must not name a candidate")
        if event.champion_hash != self.current_champion_hash:
            raise IllegalTransitionError("reference pin names a stale champion")
        try:
            pin_id = _require_nonempty(event.payload.get("pin_id"), "payload.pin_id")
            reference_hash = _require_sha256(
                event.payload.get("reference_hash"), "payload.reference_hash"
            )
            kind = _require_nonempty(event.payload.get("kind"), "payload.kind")
            owner = _require_nonempty(event.payload.get("owner"), "payload.owner")
        except ValueError as exc:
            raise IllegalTransitionError(str(exc)) from exc
        assert reference_hash is not None
        if pin_id in self.seen_pin_ids:
            raise IllegalTransitionError(f"pin identifier was already used: {pin_id}")
        self.seen_pin_ids.add(pin_id)
        self.pins[pin_id] = ReferencePin(
            pin_id=pin_id,
            reference_hash=reference_hash,
            kind=kind,
            owner=owner,
            reason=event.reason,
            created_sequence=event.sequence,
        )

    def _apply_unpin(self, event: PromotionEvent) -> None:
        if event.candidate_hash is not None or event.candidate_path is not None:
            raise IllegalTransitionError("reference unpin must not name a candidate")
        if event.champion_hash != self.current_champion_hash:
            raise IllegalTransitionError("reference unpin names a stale champion")
        try:
            pin_id = _require_nonempty(event.payload.get("pin_id"), "payload.pin_id")
            reference_hash = _require_sha256(
                event.payload.get("reference_hash"), "payload.reference_hash"
            )
        except ValueError as exc:
            raise IllegalTransitionError(str(exc)) from exc
        existing = self.pins.get(pin_id)
        if existing is None:
            raise IllegalTransitionError(f"pin is not active: {pin_id}")
        if existing.reference_hash != reference_hash:
            raise IllegalTransitionError(f"unpin reference changed for {pin_id}")
        del self.pins[pin_id]


def _state_from_events(events: Tuple[PromotionEvent, ...]) -> RegistryState:
    builder = _RegistryBuilder()
    for event in events:
        builder.apply(event)
    last_sequence = events[-1].sequence if events else 0
    last_hash = events[-1].event_hash if events else GENESIS_HASH
    return RegistryState(
        events=events,
        candidates=MappingProxyType(dict(builder.candidates)),
        generations=MappingProxyType(dict(builder.generations)),
        pins=MappingProxyType(dict(builder.pins)),
        current_champion_hash=builder.current_champion_hash,
        current_generation_id=builder.current_generation_id,
        original_hash=builder.original_hash,
        last_sequence=last_sequence,
        last_event_hash=last_hash,
    )


class EventRegistry:
    """Append and replay the authoritative immutable event directory."""

    def __init__(self, promotion_root: PathLike):
        self.root = Path(promotion_root)
        self.events_dir = self.root / "events"

    def _ensure_directories(self) -> None:
        if not self.root.exists():
            self.root.mkdir()
            fsync_directory(self.root.parent)
        elif not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if not self.events_dir.exists():
            self.events_dir.mkdir()
            fsync_directory(self.root)
        elif not self.events_dir.is_dir():
            raise NotADirectoryError(self.events_dir)

    def _event_paths(self) -> Tuple[Tuple[int, Path], ...]:
        if not self.events_dir.exists():
            return ()
        if not self.events_dir.is_dir():
            raise RegistryCorruptionError(
                f"events path is not a directory: {self.events_dir}"
            )
        result = []
        with os.scandir(self.events_dir) as entries:
            for entry in entries:
                match = _EVENT_FILE_RE.fullmatch(entry.name)
                if match is not None:
                    result.append((int(match.group(1)), Path(entry.path)))
                    continue
                if entry.name.startswith(".") or entry.name.endswith(".tmp"):
                    continue
                if entry.name.endswith(".json") and entry.name[:1].isdigit():
                    raise RegistryCorruptionError(
                        f"malformed event filename: {entry.name}"
                    )
        result.sort(key=lambda item: item[0])
        return tuple(result)

    def reconstruct(self) -> RegistryState:
        """Validate and rebuild state without relying on any mutable index."""

        events = []
        previous_hash = GENESIS_HASH
        expected_sequence = 1
        for filename_sequence, path in self._event_paths():
            if filename_sequence != expected_sequence:
                raise RegistryCorruptionError(
                    f"event sequence gap: expected {expected_sequence}, "
                    f"found {filename_sequence}"
                )
            data = _read_regular_file(path)
            value = _decode_json(data, path)
            event = PromotionEvent.from_dict(value)
            canonical_file = canonical_json_bytes(event.to_dict()) + b"\n"
            if data != canonical_file:
                raise RegistryCorruptionError(
                    f"event file is not canonical JSON: {path.name}"
                )
            if event.sequence != filename_sequence:
                raise RegistryCorruptionError(
                    f"event filename sequence {filename_sequence} does not match "
                    f"payload sequence {event.sequence}"
                )
            if event.previous_hash != previous_hash:
                raise RegistryCorruptionError(
                    f"event {event.sequence} previous hash does not match chain"
                )
            events.append(event)
            previous_hash = event.event_hash
            expected_sequence += 1
        return _state_from_events(tuple(events))

    def append_event(
        self,
        transition: Union[Transition, str],
        *,
        provenance: EventProvenance,
        champion_hash: str,
        reason: str,
        actor: str,
        candidate_hash: Optional[str] = None,
        candidate_path: Optional[str] = None,
        evaluation_key: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> PromotionEvent:
        """Validate and append one event without overwriting any prior event.

        This method performs no implicit locking.  Production callers must hold
        the run's :class:`ControllerLock`.
        """

        try:
            normalized_transition = Transition(transition)
            _require_sha256(champion_hash, "champion_hash")
            _require_nonempty(reason, "reason")
            _require_nonempty(actor, "actor")
            if candidate_hash is not None:
                _require_sha256(candidate_hash, "candidate_hash")
            if candidate_path is not None:
                _require_nonempty(candidate_path, "candidate_path")
            timestamp = (
                utc_timestamp()
                if timestamp_utc is None
                else _validate_utc_timestamp(timestamp_utc)
            )
        except ValueError as exc:
            raise IllegalTransitionError(str(exc)) from exc

        state = self.reconstruct()
        event = PromotionEvent.build(
            sequence=state.last_sequence + 1,
            previous_hash=state.last_event_hash,
            timestamp_utc=timestamp,
            provenance=provenance,
            candidate_hash=candidate_hash,
            candidate_path=candidate_path,
            champion_hash=champion_hash,
            transition=normalized_transition,
            evaluation_key=evaluation_key,
            reason=reason,
            actor=actor,
            payload=payload,
        )
        # Reject illegal mutations before making them durable.
        _state_from_events(state.events + (event,))
        self._ensure_directories()
        destination = self.events_dir / (
            f"{event.sequence:0{EVENT_SEQUENCE_WIDTH}d}.json"
        )
        _atomic_create_bytes(
            destination,
            canonical_json_bytes(event.to_dict()) + b"\n",
            mode=0o444,
        )
        return event

    def bootstrap_champion(
        self,
        *,
        champion_hash: str,
        generation_id: str,
        provenance: EventProvenance,
        reason: str,
        actor: str,
        timestamp_utc: Optional[str] = None,
    ) -> PromotionEvent:
        """Idempotently establish the initial champion in the event registry."""

        state = self.reconstruct()
        if state.current_champion_hash is not None:
            first = state.events[0] if state.events else None
            if (
                first is not None
                and first.transition == Transition.CHAMPION_BOOTSTRAPPED
                and first.champion_hash == champion_hash
                and first.payload.get("generation_id") == generation_id
                and first.controller_hash == provenance.controller_hash
                and first.source_hash == provenance.source_hash
                and first.original_hash == provenance.original_hash
                and first.config_hash == provenance.config_hash
                and first.schedule_hash == provenance.schedule_hash
                and first.policy_hash == provenance.policy_hash
            ):
                return first
            raise ChampionConflictError("registry champion is already bootstrapped")
        return self.append_event(
            Transition.CHAMPION_BOOTSTRAPPED,
            provenance=provenance,
            champion_hash=champion_hash,
            reason=reason,
            actor=actor,
            payload={"generation_id": generation_id},
            timestamp_utc=timestamp_utc,
        )

    def transition_candidate(
        self,
        candidate_hash: str,
        candidate_path: str,
        target: CandidateState,
        *,
        provenance: EventProvenance,
        champion_hash: str,
        reason: str,
        actor: str,
        evaluation_key: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> PromotionEvent:
        """Idempotently move a candidate to one legal lifecycle state."""

        target = CandidateState(target)
        state = self.reconstruct()
        existing = state.candidates.get(candidate_hash)
        merged_payload = _copy_json_object(payload)
        if existing is not None and existing.state == target:
            if existing.candidate_path != candidate_path:
                raise IllegalTransitionError(
                    "idempotent candidate retry changed the destination path"
                )
            transition = _STATE_TO_CANDIDATE_TRANSITION[target]
            for event in reversed(state.events):
                if (
                    event.transition == transition
                    and event.candidate_hash == candidate_hash
                ):
                    if (
                        "previous_candidate_path" not in merged_payload
                        and "previous_candidate_path" in event.payload
                    ):
                        merged_payload["previous_candidate_path"] = event.payload[
                            "previous_candidate_path"
                        ]
                    conflicts = (
                        event.candidate_path != candidate_path
                        or event.champion_hash != champion_hash
                        or event.evaluation_key != evaluation_key
                        or event.controller_hash != provenance.controller_hash
                        or event.source_hash != provenance.source_hash
                        or event.original_hash != provenance.original_hash
                        or event.config_hash != provenance.config_hash
                        or event.schedule_hash != provenance.schedule_hash
                        or event.policy_hash != provenance.policy_hash
                        or event.payload != merged_payload
                    )
                    if conflicts:
                        raise IllegalTransitionError(
                            "idempotent candidate retry changed immutable metadata"
                        )
                    return event
            raise RegistryCorruptionError("candidate state has no originating event")

        if existing is not None and existing.candidate_path != candidate_path:
            supplied_previous = merged_payload.get("previous_candidate_path")
            if (
                supplied_previous is not None
                and supplied_previous != existing.candidate_path
            ):
                raise IllegalTransitionError(
                    "payload previous_candidate_path conflicts with registry state"
                )
            merged_payload["previous_candidate_path"] = existing.candidate_path

        return self.append_event(
            _STATE_TO_CANDIDATE_TRANSITION[target],
            provenance=provenance,
            candidate_hash=candidate_hash,
            candidate_path=candidate_path,
            champion_hash=champion_hash,
            evaluation_key=evaluation_key,
            reason=reason,
            actor=actor,
            payload=merged_payload,
            timestamp_utc=timestamp_utc,
        )

    def transition_generation(
        self,
        generation_id: str,
        candidate_hash: str,
        candidate_path: str,
        target: GenerationState,
        *,
        provenance: EventProvenance,
        tested_champion_hash: str,
        reason: str,
        actor: str,
        evaluation_key: Optional[str] = None,
        restore_champion_hash: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> PromotionEvent:
        """Idempotently advance one generation through rollout or rollback."""

        target = GenerationState(target)
        merged_payload = _copy_json_object(payload)
        supplied_generation = merged_payload.get("generation_id")
        if supplied_generation is not None and supplied_generation != generation_id:
            raise IllegalTransitionError("payload generation_id conflicts with argument")
        merged_payload["generation_id"] = generation_id
        if target in {
            GenerationState.ROLLBACK_PENDING,
            GenerationState.ROLLED_BACK,
        }:
            if restore_champion_hash is None:
                raise IllegalTransitionError(
                    f"{target.value} requires restore_champion_hash"
                )
            supplied_restore = merged_payload.get("restore_champion_hash")
            if (
                supplied_restore is not None
                and supplied_restore != restore_champion_hash
            ):
                raise IllegalTransitionError(
                    "payload restore_champion_hash conflicts with argument"
                )
            merged_payload["restore_champion_hash"] = restore_champion_hash

        state = self.reconstruct()
        existing = state.generations.get(generation_id)
        candidate = state.candidates.get(candidate_hash)
        if (
            existing is None
            and candidate is not None
            and candidate.candidate_path != candidate_path
        ):
            supplied_previous = merged_payload.get("previous_candidate_path")
            if (
                supplied_previous is not None
                and supplied_previous != candidate.candidate_path
            ):
                raise IllegalTransitionError(
                    "payload previous_candidate_path conflicts with registry state"
                )
            merged_payload["previous_candidate_path"] = candidate.candidate_path
        if existing is not None and existing.state == target:
            if (
                existing.candidate_hash != candidate_hash
                or existing.candidate_path != candidate_path
                or existing.previous_champion_hash != tested_champion_hash
            ):
                raise IllegalTransitionError(
                    "idempotent generation retry changed immutable identity"
                )
            transition = _STATE_TO_GENERATION_TRANSITION[target]
            for event in reversed(state.events):
                if (
                    event.transition == transition
                    and event.payload.get("generation_id") == generation_id
                ):
                    if (
                        "previous_candidate_path" not in merged_payload
                        and "previous_candidate_path" in event.payload
                    ):
                        merged_payload["previous_candidate_path"] = event.payload[
                            "previous_candidate_path"
                        ]
                    conflicts = (
                        event.evaluation_key != evaluation_key
                        or event.controller_hash != provenance.controller_hash
                        or event.source_hash != provenance.source_hash
                        or event.original_hash != provenance.original_hash
                        or event.config_hash != provenance.config_hash
                        or event.schedule_hash != provenance.schedule_hash
                        or event.policy_hash != provenance.policy_hash
                        or event.payload != merged_payload
                    )
                    if conflicts:
                        raise IllegalTransitionError(
                            "idempotent generation retry changed immutable metadata"
                        )
                    return event
            raise RegistryCorruptionError("generation state has no originating event")

        return self.append_event(
            _STATE_TO_GENERATION_TRANSITION[target],
            provenance=provenance,
            candidate_hash=candidate_hash,
            candidate_path=candidate_path,
            champion_hash=tested_champion_hash,
            evaluation_key=evaluation_key,
            reason=reason,
            actor=actor,
            payload=merged_payload,
            timestamp_utc=timestamp_utc,
        )

    def pin_reference(
        self,
        pin_id: str,
        reference_hash: str,
        *,
        kind: str,
        owner: str,
        provenance: EventProvenance,
        champion_hash: str,
        reason: str,
        actor: str,
        timestamp_utc: Optional[str] = None,
    ) -> PromotionEvent:
        """Idempotently add an explicit retention pin."""

        state = self.reconstruct()
        existing = state.pins.get(pin_id)
        if existing is not None:
            if (
                existing.reference_hash == reference_hash
                and existing.kind == kind
                and existing.owner == owner
                and existing.reason == reason
            ):
                for event in reversed(state.events):
                    if (
                        event.transition == Transition.REFERENCE_PINNED
                        and event.payload.get("pin_id") == pin_id
                    ):
                        if (
                            event.champion_hash != champion_hash
                            or event.controller_hash != provenance.controller_hash
                            or event.source_hash != provenance.source_hash
                            or event.original_hash != provenance.original_hash
                            or event.config_hash != provenance.config_hash
                            or event.schedule_hash != provenance.schedule_hash
                            or event.policy_hash != provenance.policy_hash
                        ):
                            raise IllegalTransitionError(
                                f"active pin {pin_id!r} changed provenance"
                            )
                        return event
            raise IllegalTransitionError(f"active pin {pin_id!r} conflicts with retry")
        for event in reversed(state.events):
            if (
                event.transition == Transition.REFERENCE_PINNED
                and event.payload.get("pin_id") == pin_id
            ):
                same_request = (
                    event.payload.get("reference_hash") == reference_hash
                    and event.payload.get("kind") == kind
                    and event.payload.get("owner") == owner
                    and event.reason == reason
                    and event.champion_hash == champion_hash
                    and event.controller_hash == provenance.controller_hash
                    and event.source_hash == provenance.source_hash
                    and event.original_hash == provenance.original_hash
                    and event.config_hash == provenance.config_hash
                    and event.schedule_hash == provenance.schedule_hash
                    and event.policy_hash == provenance.policy_hash
                )
                if same_request:
                    # Pin IDs are one-shot operation identifiers. A delayed
                    # retry after an unpin must not resurrect the reference.
                    return event
                raise IllegalTransitionError(
                    f"historical pin {pin_id!r} conflicts with retry"
                )
        return self.append_event(
            Transition.REFERENCE_PINNED,
            provenance=provenance,
            champion_hash=champion_hash,
            reason=reason,
            actor=actor,
            payload={
                "pin_id": pin_id,
                "reference_hash": reference_hash,
                "kind": kind,
                "owner": owner,
            },
            timestamp_utc=timestamp_utc,
        )

    def unpin_reference(
        self,
        pin_id: str,
        *,
        provenance: EventProvenance,
        champion_hash: str,
        reason: str,
        actor: str,
        timestamp_utc: Optional[str] = None,
    ) -> Optional[PromotionEvent]:
        """Idempotently remove an explicit pin; no filesystem data is deleted."""

        state = self.reconstruct()
        existing = state.pins.get(pin_id)
        if existing is None:
            return None
        return self.append_event(
            Transition.REFERENCE_UNPINNED,
            provenance=provenance,
            champion_hash=champion_hash,
            reason=reason,
            actor=actor,
            payload={
                "pin_id": pin_id,
                "reference_hash": existing.reference_hash,
            },
            timestamp_utc=timestamp_utc,
        )


class ControllerLock:
    """Nonblocking POSIX advisory lock for the single promotion controller.

    The lock file is intentionally retained after release; unlinking a lock
    file permits two processes to hold locks on different inodes.
    """

    def __init__(self, path: PathLike, *, owner: str = "promotion-controller"):
        self.path = Path(path)
        self.owner = _require_nonempty(owner, "owner")
        self._fd: Optional[int] = None

    @property
    def acquired(self) -> bool:
        """Whether this instance currently owns the advisory lock."""

        return self._fd is not None

    def acquire(self) -> "ControllerLock":
        """Acquire immediately or raise :class:`ControllerLockError`."""

        if self._fd is not None:
            return self
        if not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"lock parent does not exist: {self.path.parent}"
            )
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(os.fspath(self.path), flags, 0o600)
        locked = False
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ControllerLockError(
                    f"controller lock is not a regular file: {self.path}"
                )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise ControllerLockError(
                    f"another controller owns {self.path}"
                ) from exc
            os.fchmod(fd, 0o600)
            metadata = canonical_json_bytes(
                {
                    "actor": self.owner,
                    "acquired_at_utc": utc_timestamp(),
                    "pid": os.getpid(),
                }
            ) + b"\n"
            os.ftruncate(fd, 0)
            offset = 0
            while offset < len(metadata):
                written = os.write(fd, metadata[offset:])
                if written <= 0:
                    raise OSError("short write while recording lock owner")
                offset += written
            os.fsync(fd)
            fsync_directory(self.path.parent)
        except BaseException:
            if locked:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
            raise
        self._fd = fd
        return self

    def release(self) -> None:
        """Release and close the lock descriptor; safe to call repeatedly."""

        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "ControllerLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


@dataclass(frozen=True)
class ChampionRecord:
    """Integrity-checked contents of ``champion.json``."""

    schema_version: int
    champion_hash: str
    generation_id: str
    previous_champion_hash: Optional[str]
    original_hash: str
    activated_at_utc: str
    controller_hash: str
    source_hash: str
    config_hash: str
    schedule_hash: str
    policy_hash: str
    evaluation_key: Optional[str]
    pass_report_path: Optional[str]
    pass_report_hash: Optional[str]
    actor: str
    bootstrap: bool
    record_hash: str

    def body_dict(self) -> Dict[str, Any]:
        """Return champion fields excluding the self-hash."""

        return {
            "schema_version": self.schema_version,
            "champion_hash": self.champion_hash,
            "generation_id": self.generation_id,
            "previous_champion_hash": self.previous_champion_hash,
            "original_hash": self.original_hash,
            "activated_at_utc": self.activated_at_utc,
            "controller_hash": self.controller_hash,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "schedule_hash": self.schedule_hash,
            "policy_hash": self.policy_hash,
            "evaluation_key": self.evaluation_key,
            "pass_report_path": self.pass_report_path,
            "pass_report_hash": self.pass_report_hash,
            "actor": self.actor,
            "bootstrap": self.bootstrap,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return all serializable champion fields."""

        value = self.body_dict()
        value["record_hash"] = self.record_hash
        return value

    def verify(self) -> None:
        """Validate champion metadata and its canonical self-hash."""

        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != CHAMPION_SCHEMA_VERSION
        ):
            raise RegistryCorruptionError(
                f"unsupported champion schema version: {self.schema_version}"
            )
        try:
            _require_sha256(self.champion_hash, "champion_hash")
            _require_nonempty(self.generation_id, "generation_id")
            _require_sha256(
                self.previous_champion_hash,
                "previous_champion_hash",
                optional=True,
            )
            _require_sha256(self.original_hash, "original_hash")
            _validate_utc_timestamp(self.activated_at_utc, "activated_at_utc")
            _require_sha256(self.controller_hash, "controller_hash")
            _require_sha256(self.source_hash, "source_hash")
            _require_sha256(self.config_hash, "config_hash")
            _require_sha256(self.schedule_hash, "schedule_hash")
            _require_sha256(self.policy_hash, "policy_hash")
            _require_nonempty(self.actor, "actor")
            _require_sha256(self.record_hash, "record_hash")
            if not isinstance(self.bootstrap, bool):
                raise ValueError("bootstrap must be boolean")
            if self.bootstrap:
                if any(
                    value is not None
                    for value in (
                        self.previous_champion_hash,
                        self.evaluation_key,
                        self.pass_report_path,
                        self.pass_report_hash,
                    )
                ):
                    raise ValueError(
                        "bootstrap champion may not contain evaluation/report metadata"
                    )
            else:
                _require_sha256(
                    self.previous_champion_hash, "previous_champion_hash"
                )
                _require_nonempty(self.evaluation_key, "evaluation_key")
                _require_nonempty(self.pass_report_path, "pass_report_path")
                _require_sha256(self.pass_report_hash, "pass_report_hash")
        except ValueError as exc:
            raise RegistryCorruptionError(str(exc)) from exc
        expected = canonical_sha256(self.body_dict())
        if expected != self.record_hash:
            raise RegistryCorruptionError(
                f"champion record hash mismatch: stored {self.record_hash}, "
                f"computed {expected}"
            )

    @classmethod
    def build(
        cls,
        *,
        champion_hash: str,
        generation_id: str,
        previous_champion_hash: Optional[str],
        provenance: EventProvenance,
        activated_at_utc: str,
        evaluation_key: Optional[str],
        pass_report_path: Optional[str],
        pass_report_hash: Optional[str],
        actor: str,
        bootstrap: bool,
    ) -> "ChampionRecord":
        """Construct and self-hash champion state."""

        body = {
            "schema_version": CHAMPION_SCHEMA_VERSION,
            "champion_hash": champion_hash,
            "generation_id": generation_id,
            "previous_champion_hash": previous_champion_hash,
            "original_hash": provenance.original_hash,
            "activated_at_utc": activated_at_utc,
            "controller_hash": provenance.controller_hash,
            "source_hash": provenance.source_hash,
            "config_hash": provenance.config_hash,
            "schedule_hash": provenance.schedule_hash,
            "policy_hash": provenance.policy_hash,
            "evaluation_key": evaluation_key,
            "pass_report_path": pass_report_path,
            "pass_report_hash": pass_report_hash,
            "actor": actor,
            "bootstrap": bootstrap,
        }
        record = cls(
            schema_version=CHAMPION_SCHEMA_VERSION,
            champion_hash=champion_hash,
            generation_id=generation_id,
            previous_champion_hash=previous_champion_hash,
            original_hash=provenance.original_hash,
            activated_at_utc=activated_at_utc,
            controller_hash=provenance.controller_hash,
            source_hash=provenance.source_hash,
            config_hash=provenance.config_hash,
            schedule_hash=provenance.schedule_hash,
            policy_hash=provenance.policy_hash,
            evaluation_key=evaluation_key,
            pass_report_path=pass_report_path,
            pass_report_hash=pass_report_hash,
            actor=actor,
            bootstrap=bootstrap,
            record_hash=canonical_sha256(body),
        )
        record.verify()
        return record

    @classmethod
    def from_dict(cls, value: Any) -> "ChampionRecord":
        """Parse a champion record and reject unknown fields."""

        if not isinstance(value, dict):
            raise RegistryCorruptionError("champion JSON root must be an object")
        expected_fields = {
            "schema_version",
            "champion_hash",
            "generation_id",
            "previous_champion_hash",
            "original_hash",
            "activated_at_utc",
            "controller_hash",
            "source_hash",
            "config_hash",
            "schedule_hash",
            "policy_hash",
            "evaluation_key",
            "pass_report_path",
            "pass_report_hash",
            "actor",
            "bootstrap",
            "record_hash",
        }
        if set(value) != expected_fields:
            missing = sorted(expected_fields - set(value))
            extra = sorted(set(value) - expected_fields)
            raise RegistryCorruptionError(
                f"champion fields differ from schema; missing={missing}, extra={extra}"
            )
        record = cls(**value)
        record.verify()
        return record


def load_champion(path: PathLike) -> ChampionRecord:
    """Load and integrity-check canonical ``champion.json``."""

    champion_path = Path(path)
    data = _read_regular_file(champion_path)
    value = _decode_json(data, champion_path)
    record = ChampionRecord.from_dict(value)
    if data != canonical_json_bytes(record.to_dict()) + b"\n":
        raise RegistryCorruptionError("champion file is not canonical JSON")
    return record


def bootstrap_champion(
    path: PathLike,
    *,
    champion_hash: str,
    generation_id: str,
    provenance: EventProvenance,
    actor: str,
    timestamp_utc: Optional[str] = None,
) -> ChampionRecord:
    """Idempotently create the initial champion with explicit provenance.

    The parent directory must already exist.  An existing, byte-valid bootstrap
    is accepted only when all identity and provenance fields match.
    """

    champion_path = Path(path)
    timestamp = (
        utc_timestamp()
        if timestamp_utc is None
        else _validate_utc_timestamp(timestamp_utc)
    )
    desired = ChampionRecord.build(
        champion_hash=champion_hash,
        generation_id=generation_id,
        previous_champion_hash=None,
        provenance=provenance,
        activated_at_utc=timestamp,
        evaluation_key=None,
        pass_report_path=None,
        pass_report_hash=None,
        actor=actor,
        bootstrap=True,
    )
    if champion_path.exists():
        existing = load_champion(champion_path)
        _validate_bootstrap_retry(existing, desired)
        return existing
    try:
        _atomic_create_bytes(
            champion_path, canonical_json_bytes(desired.to_dict()) + b"\n"
        )
        return desired
    except EventConflictError:
        existing = load_champion(champion_path)
        _validate_bootstrap_retry(existing, desired)
        return existing


def _validate_bootstrap_retry(
    existing: ChampionRecord, desired: ChampionRecord
) -> None:
    comparable_fields = (
        "champion_hash",
        "generation_id",
        "original_hash",
        "controller_hash",
        "source_hash",
        "config_hash",
        "schedule_hash",
        "policy_hash",
        "actor",
        "bootstrap",
    )
    if any(
        getattr(existing, field) != getattr(desired, field)
        for field in comparable_fields
    ):
        raise ChampionConflictError(
            "existing champion conflicts with bootstrap request"
        )


@dataclass(frozen=True)
class FinalizedPassReport:
    """Verified promotion-critical metadata from a finalized gate report."""

    path: str
    report_hash: str
    schema_version: int
    decision: str
    candidate_hash: str
    tested_champion_hash: str
    original_hash: str
    evaluation_key: str
    config_hash: str
    schedule_hash: str
    policy_hash: str


def load_finalized_pass_report(
    path: PathLike, *, expected_report_hash: str
) -> FinalizedPassReport:
    """Load a report and require finalized ``PASS`` promotion metadata.

    Gate reports may contain additional metrics, but these top-level metadata
    keys are mandatory: ``schema_version``, ``decision``, ``finalized``,
    ``candidate_hash``, ``tested_champion_hash``, ``original_hash``,
    ``evaluation_key``, ``config_hash``, ``schedule_hash``, and ``policy_hash``.
    """

    report_path = Path(path)
    try:
        _require_sha256(expected_report_hash, "expected_report_hash")
        data = _read_regular_file(report_path)
        actual_hash = sha256_bytes(data)
        if actual_hash != expected_report_hash:
            raise PassReportError(
                f"report hash mismatch: expected {expected_report_hash}, "
                f"found {actual_hash}"
            )
        value = _decode_json(data, report_path)
        if not isinstance(value, dict):
            raise PassReportError("PASS report JSON root must be an object")
        schema_version = value.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 1
        ):
            raise PassReportError("report schema_version must be a positive integer")
        if value.get("finalized") is not True:
            raise PassReportError("report is not finalized")
        decision = value.get("decision")
        if decision != "PASS":
            raise PassReportError(f"report decision is not PASS: {decision!r}")
        candidate_hash = _require_sha256(
            value.get("candidate_hash"), "report.candidate_hash"
        )
        tested_champion_hash = _require_sha256(
            value.get("tested_champion_hash"),
            "report.tested_champion_hash",
        )
        original_hash = _require_sha256(
            value.get("original_hash"), "report.original_hash"
        )
        evaluation_key = _require_nonempty(
            value.get("evaluation_key"), "report.evaluation_key"
        )
        config_hash = _require_sha256(
            value.get("config_hash"), "report.config_hash"
        )
        schedule_hash = _require_sha256(
            value.get("schedule_hash"), "report.schedule_hash"
        )
        policy_hash = _require_sha256(
            value.get("policy_hash"), "report.policy_hash"
        )
        if "finalized_at_utc" in value:
            _validate_utc_timestamp(
                value["finalized_at_utc"], "report.finalized_at_utc"
            )
    except RegistryCorruptionError as exc:
        raise PassReportError(str(exc)) from exc
    except ValueError as exc:
        raise PassReportError(str(exc)) from exc

    assert candidate_hash is not None
    assert tested_champion_hash is not None
    assert original_hash is not None
    assert config_hash is not None
    assert schedule_hash is not None
    assert policy_hash is not None
    return FinalizedPassReport(
        path=os.fspath(report_path),
        report_hash=actual_hash,
        schema_version=schema_version,
        decision=decision,
        candidate_hash=candidate_hash,
        tested_champion_hash=tested_champion_hash,
        original_hash=original_hash,
        evaluation_key=evaluation_key,
        config_hash=config_hash,
        schedule_hash=schedule_hash,
        policy_hash=policy_hash,
    )


def compare_and_swap_champion(
    path: PathLike,
    *,
    expected_champion_hash: str,
    candidate_hash: str,
    generation_id: str,
    pass_report_path: PathLike,
    pass_report_hash: str,
    evaluation_key: str,
    provenance: EventProvenance,
    actor: str,
    timestamp_utc: Optional[str] = None,
) -> ChampionRecord:
    """Atomically promote a candidate only from the champion it actually tested.

    A retry after the exact same update is a read-only success.  A stale
    expected champion, a different candidate, or changed report/provenance
    metadata raises :class:`StaleChampionError` or
    :class:`ChampionConflictError`.  Production callers must hold the
    controller lock for the complete report/event/champion transaction.
    """

    try:
        _require_sha256(expected_champion_hash, "expected_champion_hash")
        _require_sha256(candidate_hash, "candidate_hash")
        _require_nonempty(generation_id, "generation_id")
        _require_nonempty(evaluation_key, "evaluation_key")
        _require_nonempty(actor, "actor")
    except ValueError as exc:
        raise ChampionConflictError(str(exc)) from exc
    if candidate_hash == expected_champion_hash:
        raise ChampionConflictError("candidate must differ from expected champion")

    report = load_finalized_pass_report(
        pass_report_path, expected_report_hash=pass_report_hash
    )
    expected_report_metadata = {
        "candidate_hash": candidate_hash,
        "tested_champion_hash": expected_champion_hash,
        "original_hash": provenance.original_hash,
        "evaluation_key": evaluation_key,
        "config_hash": provenance.config_hash,
        "schedule_hash": provenance.schedule_hash,
        "policy_hash": provenance.policy_hash,
    }
    for field, expected in expected_report_metadata.items():
        actual = getattr(report, field)
        if actual != expected:
            raise PassReportError(
                f"report {field} mismatch: expected {expected!r}, found {actual!r}"
            )

    champion_path = Path(path)
    current = load_champion(champion_path)
    if current.original_hash != provenance.original_hash:
        raise ChampionConflictError("champion and report name different originals")

    stable_report_path = os.fspath(Path(pass_report_path))
    if current.champion_hash == candidate_hash:
        retry_fields = {
            "previous_champion_hash": expected_champion_hash,
            "generation_id": generation_id,
            "original_hash": provenance.original_hash,
            "controller_hash": provenance.controller_hash,
            "source_hash": provenance.source_hash,
            "config_hash": provenance.config_hash,
            "schedule_hash": provenance.schedule_hash,
            "policy_hash": provenance.policy_hash,
            "evaluation_key": evaluation_key,
            "pass_report_path": stable_report_path,
            "pass_report_hash": pass_report_hash,
            "actor": actor,
            "bootstrap": False,
        }
        conflicts = [
            field
            for field, expected in retry_fields.items()
            if getattr(current, field) != expected
        ]
        if conflicts:
            raise ChampionConflictError(
                "candidate is already champion with different metadata: "
                + ", ".join(conflicts)
            )
        return current

    if current.champion_hash != expected_champion_hash:
        raise StaleChampionError(
            f"stale tested champion {expected_champion_hash}; "
            f"current champion is {current.champion_hash}"
        )

    timestamp = (
        utc_timestamp()
        if timestamp_utc is None
        else _validate_utc_timestamp(timestamp_utc)
    )
    replacement = ChampionRecord.build(
        champion_hash=candidate_hash,
        generation_id=generation_id,
        previous_champion_hash=expected_champion_hash,
        provenance=provenance,
        activated_at_utc=timestamp,
        evaluation_key=evaluation_key,
        pass_report_path=stable_report_path,
        pass_report_hash=pass_report_hash,
        actor=actor,
        bootstrap=False,
    )
    atomic_write_json(champion_path, replacement.to_dict())
    return replacement
