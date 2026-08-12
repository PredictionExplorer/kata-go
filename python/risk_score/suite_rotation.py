#!/usr/bin/env python3
"""Crash-safe, content-addressed evaluation-suite rotation registry.

The immutable event log is authoritative. ``active-suite.json`` and
``status.json`` are replaceable projections that can always be reconstructed.
Suite bundles, request manifests, continuity receipts, and registry events are
published once under content-derived names and are never removed by this
module.

Only accepted champion activations and UTC age participate in the rotation
cadence. Candidate evaluation outcomes are deliberately absent from the event
schema and public API.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from risk_score.build_evaluation_suites import (
    MACHINE_GENERATOR_CONTRACT,
    MACHINE_MANIFEST_CONTRACT,
    MACHINE_REVIEW_MANIFEST_CONTRACT,
    MACHINE_REVIEW_MODE,
    MACHINE_SCHEMA_VERSION,
    _load_policy_binding,
)
from risk_score.position_samples import semantic_position_sha256


REGISTRY_SPEC_CONTRACT = "risk-score-evaluation-suite-registry-spec-v1"
REGISTRY_EVENT_CONTRACT = "risk-score-evaluation-suite-registry-event-v1"
SUITE_VERSION_CONTRACT = "risk-score-evaluation-suite-version-v1"
ACTIVE_SUITE_CONTRACT = "risk-score-active-evaluation-suite-v1"
ROTATION_REQUEST_CONTRACT = "risk-score-evaluation-suite-rotation-request-v1"
SUPPLEMENT_REQUEST_CONTRACT = (
    "risk-score-suite-rotation-curation-supplement-request-v1"
)
PIPELINE_REQUEST_CONTRACT = "risk-score-suite-rotation-curation-pipeline-request-v1"
CONTINUITY_CONTRACT = "risk-score-suite-continuity-shadow-replay-v1"
STATUS_CONTRACT = "risk-score-evaluation-suite-rotation-status-v1"

SUPPLEMENT_SPEC_CONTRACT = "risk-score-curation-supplement-spec-v1"
PIPELINE_SPEC_CONTRACT = "risk-score-curation-pipeline-spec-v1"
POLICY_VERSION = "risk-seeking-checkpoint-promotion-v3"
ACCEPTED_CHAMPION_INTERVAL = 5
MAXIMUM_AGE_DAYS = 90
EVENT_SEQUENCE_WIDTH = 20
GENESIS_HASH = "0" * 64
MAX_JSON_BYTES = 64 * 1024 * 1024
HOLDOUTS = ("discovery", "confirmation", "audit")
LABELS = ("ordinary", "lead-40", "lead-80")
ALLOWED_LABELS = tuple(sorted(LABELS))

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_FILE_RE = re.compile(r"^([0-9]{20})\.json$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,254})$")

_SPEC_KEYS = {
    "schema_version",
    "contract",
    "registry_root",
    "created_at_utc",
    "policy",
    "models",
    "cadence",
    "suite_contract",
    "holdout_quotas",
    "source_quotas",
    "spec_sha256",
}
_EVENT_KEYS = {
    "schema_version",
    "contract",
    "sequence",
    "previous_event_sha256",
    "timestamp_utc",
    "spec_sha256",
    "event_type",
    "payload",
    "event_sha256",
}
_SUITE_MANIFEST_KEYS = {
    "schemaVersion",
    "manifestContract",
    "generatorContract",
    "scheduleGeneratorContract",
    "canonicalJsonContract",
    "ordinaryAssignmentContract",
    "semanticPositionContract",
    "seed",
    "policy_hash",
    "policy_version",
    "source_revision",
    "exactPolicyQuotas",
    "policyHoldoutQuotas",
    "ordinaryWeights",
    "pairsPerPosition",
    "botAIndex",
    "botBIndex",
    "acceptedLabels",
    "sources",
    "inputRowCount",
    "includedRowCount",
    "assignedRowCount",
    "unassigned",
    "exclusions",
    "banks",
    "cells",
    "discovery_schedule_hash",
    "machineReviewOnly",
    "curationSources",
    "manifestPayloadSha256",
}
_BANK_KEYS = {
    "name",
    "qualifiedName",
    "sourceLabel",
    "holdout",
    "kind",
    "contentSha256s",
    "semanticSha256s",
    "independentClusterIds",
    "independentClusterIdsSha256",
    "positionIds",
    "positionIdsSha256",
    "positions",
    "schedule",
}
_CELL_KEYS = {
    "cell_id",
    "cell_name",
    "stage",
    "look",
    "comparison",
    "suite",
    "search_mode",
    "visits",
    "color_pairs",
    "minimum_independent_position_clusters",
    "independent_cluster_ids",
    "independent_cluster_ids_hash",
    "position_ids",
    "position_ids_hash",
    "bank_name",
    "bank_path",
    "bank_hash",
    "schedule_path",
    "schedule_hash",
    "schedule_id",
    "schedule_row_count",
    "maximal_look_schedule",
    "policy_hash",
    "policy_version",
    "source_revision",
}
_CURATION_SOURCE_KEYS = {
    "source_name",
    "contract",
    "review_mode",
    "consensus_rules_version",
    "policy_hash",
    "allowed_labels",
    "output_sha256",
    "manifest_sha256",
    "manifest_identity",
    "rejected_count",
    "rejected_sha256",
    "models",
}
_EVENT_PAYLOAD_KEYS = {
    "registry.bootstrapped": {
        "suite_id",
        "version_sha256",
        "manifest_sha256",
        "manifest_identity",
        "champion",
        "previous_champion_sha256",
        "generation_id",
        "activated_at_utc",
    },
    "champion.accepted": {
        "champion",
        "generation_id",
        "previous_champion_sha256",
    },
    "evaluation.pinned": {
        "evaluation_id",
        "suite_id",
        "champion_sha256",
        "generation_id",
    },
    "evaluation.unpinned": {
        "evaluation_id",
        "suite_id",
        "champion_sha256",
        "generation_id",
    },
    "rotation.requested": {
        "request_id",
        "request_manifest",
        "base_suite_id",
        "champion_sha256",
        "generation_id",
        "trigger_at_utc",
    },
    "suite.registered": {
        "request_id",
        "suite_id",
        "version_sha256",
        "manifest_sha256",
        "manifest_identity",
        "curation_champion_sha256",
    },
    "continuity.recorded": {
        "request_id",
        "suite_id",
        "manifest",
        "current_champion_sha256",
        "previous_champion_sha256",
    },
    "generation.boundary": {
        "boundary_id",
        "champion_sha256",
        "generation_id",
        "clean",
    },
    "suite.activated": {
        "request_id",
        "suite_id",
        "previous_suite_id",
        "expected_champion_sha256",
        "generation_id",
        "boundary_id",
        "continuity_manifest_sha256",
    },
}


class SuiteRotationError(RuntimeError):
    """Base class for suite rotation failures."""


class SuiteSpecError(SuiteRotationError, ValueError):
    """The immutable registry specification is invalid or stale."""


class SuiteValidationError(SuiteRotationError, ValueError):
    """A proposed suite does not satisfy the frozen v3 contract."""


class SuiteRegistryCorruption(SuiteRotationError, ValueError):
    """Immutable registry evidence is malformed or contradictory."""


class SuiteRegistryBusy(SuiteRotationError):
    """Another process owns the suite registry lock."""


class SuiteConflictError(SuiteRotationError):
    """An immutable operation conflicts with prior registry state."""


class StaleActiveSuiteError(SuiteConflictError):
    """The caller's expected active suite is stale."""


class StaleChampionError(SuiteConflictError):
    """The caller or proposed suite names a stale champion."""


class EvaluationPinConflictError(SuiteConflictError):
    """An evaluation identifier is already pinned incompatibly."""


class ActivationBlockedError(SuiteConflictError):
    """A suite is valid but cannot be activated at this time."""


def canonical_json_bytes(value: Any) -> bytes:
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


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: Any, keys: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    if set(value) != keys:
        raise ValueError(
            f"{role} keys differ; missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase 64-character SHA-256")
    return value


def _require_id(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{role} is not a safe nonempty identifier")
    return value


def _parse_utc(value: Any, role: str = "timestamp") -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{role} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{role} is not a valid UTC timestamp") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{role} must be UTC")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _coerce_now(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock returned a naive datetime")
        return value.astimezone(timezone.utc)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("clock must return a timezone-aware datetime or Unix seconds")
    return datetime.fromtimestamp(float(value), timezone.utc)


def _read_regular(path: Path, role: str) -> bytes:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file")
    if metadata.st_size > MAX_JSON_BYTES:
        raise ValueError(f"{role} exceeds the JSON size limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(source), flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{role} is not a regular file")
        chunks = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b"".join(chunks)
        if len(data) > MAX_JSON_BYTES:
            raise ValueError(f"{role} exceeds the JSON size limit")
        return data
    finally:
        os.close(descriptor)


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    data = _read_regular(Path(path), role)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must have an object root")
    if data != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_canonical_jsonl(path: Path, role: str) -> Tuple[Dict[str, Any], ...]:
    data = _read_regular(Path(path), role)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{role} is not UTF-8") from exc
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{role} contains a blank row at line {line_number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{role} has invalid JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{role} row {line_number} must be an object")
        rows.append(value)
    expected = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if not rows or data != expected:
        raise ValueError(f"{role} must be nonempty canonical JSONL")
    return tuple(rows)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise SuiteRegistryCorruption(f"unsafe registry directory: {path}")
        return
    parent = path.parent
    if not parent.is_dir():
        _mkdir_durable(parent)
    path.mkdir()
    fsync_directory(parent)


def _write_temp(parent: Path, name: str, data: bytes, mode: int) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{name}.", suffix=".tmp", dir=os.fspath(parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_create(path: Path, data: bytes, *, mode: int = 0o444) -> None:
    temporary = _write_temp(path.parent, path.name, data, mode)
    try:
        try:
            os.link(
                os.fspath(temporary), os.fspath(path), follow_symlinks=False
            )
        except FileExistsError as exc:
            raise SuiteConflictError(f"immutable file already exists: {path}") from exc
        temporary.unlink()
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    temporary = _write_temp(path.parent, path.name, data, mode)
    try:
        os.replace(os.fspath(temporary), os.fspath(path))
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_canonical(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    data = canonical_json_bytes(value) + b"\n"
    if replace:
        _atomic_replace(path, data)
    else:
        _atomic_create(path, data)


def _absolute_file(path: Path, role: str, expected_hash: Optional[str] = None) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    if requested.is_symlink() or not requested.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    resolved = requested.resolve()
    if expected_hash is not None and file_sha256(resolved) != expected_hash:
        raise ValueError(f"{role} does not match its frozen SHA-256")
    return resolved


def _absolute_future_directory(path: Path, role: str) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    resolved_parent = requested.parent.resolve()
    resolved = resolved_parent / requested.name
    if os.path.lexists(os.fspath(resolved)) and (
        resolved.is_symlink() or not resolved.is_dir()
    ):
        raise ValueError(f"{role} is not a safe directory")
    return resolved


def _model_value(
    path: Path, sha256: str, role: str, *, generation_id: Optional[str] = None
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "role": role,
        "path": str(path),
        "sha256": sha256,
    }
    if generation_id is not None:
        value["generation_id"] = generation_id
    return value


@dataclass(frozen=True)
class ModelBinding:
    role: str
    path: Path
    sha256: str
    generation_id: Optional[str] = None

    def to_dict(self, *, include_generation: bool = True) -> Dict[str, Any]:
        return _model_value(
            self.path,
            self.sha256,
            self.role,
            generation_id=self.generation_id if include_generation else None,
        )


@dataclass(frozen=True)
class RegistrySpec:
    path: Path
    file_sha256: str
    identity: str
    root: Path
    created_at_utc: str
    policy_path: Path
    policy_file_sha256: str
    policy_identity: str
    original: ModelBinding
    initial_champion: ModelBinding
    holdout_quotas: Mapping[str, Mapping[str, int]]
    source_quotas: Mapping[str, int]
    raw: Mapping[str, Any]


def _normalize_quotas(value: Any, role: str) -> Dict[str, Dict[str, int]]:
    outer = _require_exact_keys(value, set(LABELS), role)
    result: Dict[str, Dict[str, int]] = {}
    for label in LABELS:
        inner = _require_exact_keys(
            outer[label], set(HOLDOUTS), f"{role}.{label}"
        )
        parsed = {}
        for holdout in HOLDOUTS:
            count = inner[holdout]
            if type(count) is not int or count < 0:
                raise ValueError(f"{role}.{label}.{holdout} must be nonnegative")
            parsed[holdout] = count
        result[label] = parsed
    return result


def _parse_model_binding(
    value: Any,
    role: str,
    expected_role: str,
    *,
    with_generation: bool,
) -> ModelBinding:
    expected_keys = {"role", "path", "sha256"}
    if with_generation:
        expected_keys.add("generation_id")
    checked = _require_exact_keys(value, expected_keys, role)
    if checked["role"] != expected_role:
        raise ValueError(f"{role} has the wrong role")
    sha256 = _require_sha256(checked["sha256"], f"{role}.sha256")
    path = _absolute_file(Path(checked["path"]), role, sha256)
    generation = None
    if with_generation:
        generation = _require_id(checked["generation_id"], f"{role}.generation_id")
    return ModelBinding(expected_role, path, sha256, generation)


def load_registry_spec(path: Path) -> RegistrySpec:
    requested = _absolute_file(Path(path), "registry specification")
    try:
        raw = _load_canonical_object(requested, "registry specification")
        _require_exact_keys(raw, _SPEC_KEYS, "registry specification")
        if raw["schema_version"] != 1 or raw["contract"] != REGISTRY_SPEC_CONTRACT:
            raise ValueError("unsupported registry specification contract")
        payload = dict(raw)
        identity = payload.pop("spec_sha256", None)
        if _require_sha256(identity, "registry specification identity") != canonical_sha256(
            payload
        ):
            raise ValueError("registry specification self-hash is invalid")
        created_at = _format_utc(_parse_utc(raw["created_at_utc"], "created_at_utc"))
        root = _absolute_future_directory(Path(raw["registry_root"]), "registry root")

        policy_value = _require_exact_keys(
            raw["policy"],
            {"path", "sha256", "identity", "policy_version"},
            "policy binding",
        )
        policy_hash = _require_sha256(policy_value["sha256"], "policy file hash")
        policy_path = _absolute_file(
            Path(policy_value["path"]), "promotion policy", policy_hash
        )
        policy_plan = _load_policy_binding(policy_path)
        if (
            not policy_plan.machine_review_contract
            or policy_plan.policy_version != POLICY_VERSION
            or policy_value["policy_version"] != POLICY_VERSION
            or policy_value["identity"] != policy_plan.policy_hash
        ):
            raise ValueError("registry policy is not the frozen policy-v3 contract")

        models = _require_exact_keys(
            raw["models"], {"original", "initial_champion"}, "model bindings"
        )
        original = _parse_model_binding(
            models["original"], "original model", "immutable_original", with_generation=False
        )
        champion = _parse_model_binding(
            models["initial_champion"],
            "initial champion model",
            "frozen_champion",
            with_generation=True,
        )
        if original.sha256 == champion.sha256 or original.path == champion.path:
            raise ValueError("original and initial champion must be distinct")

        cadence = _require_exact_keys(
            raw["cadence"],
            {
                "accepted_champion_interval",
                "maximum_age_days",
                "source",
                "candidate_results_allowed",
            },
            "rotation cadence",
        )
        if cadence != {
            "accepted_champion_interval": ACCEPTED_CHAMPION_INTERVAL,
            "maximum_age_days": MAXIMUM_AGE_DAYS,
            "source": "accepted-champion-events-and-utc-only-v1",
            "candidate_results_allowed": False,
        }:
            raise ValueError("rotation cadence differs from the frozen contract")

        suite_contract = _require_exact_keys(
            raw["suite_contract"],
            {
                "schema_version",
                "manifest_contract",
                "generator_contract",
                "policy_version",
                "review_mode",
            },
            "suite contract",
        )
        if suite_contract != {
            "schema_version": MACHINE_SCHEMA_VERSION,
            "manifest_contract": MACHINE_MANIFEST_CONTRACT,
            "generator_contract": MACHINE_GENERATOR_CONTRACT,
            "policy_version": POLICY_VERSION,
            "review_mode": MACHINE_REVIEW_MODE,
        }:
            raise ValueError("suite contract differs from policy-v3")

        quotas = _normalize_quotas(raw["holdout_quotas"], "holdout quotas")
        expected_quotas = {
            label: {holdout: int(policy_plan.holdout_quotas[label][holdout]) for holdout in HOLDOUTS}
            for label in LABELS
        }
        if quotas != expected_quotas:
            raise ValueError("holdout quotas differ from the frozen policy")
        source_values = _require_exact_keys(
            raw["source_quotas"], set(LABELS), "source quotas"
        )
        source_quotas = {}
        for label in LABELS:
            count = source_values[label]
            if type(count) is not int or count < 1:
                raise ValueError(f"source quota {label} must be positive")
            source_quotas[label] = count
        expected_sources = {
            label: sum(quotas[label][holdout] for holdout in HOLDOUTS)
            for label in LABELS
        }
        if source_quotas != expected_sources:
            raise ValueError("source quotas do not equal frozen holdout totals")
    except (OSError, ValueError) as exc:
        if isinstance(exc, SuiteSpecError):
            raise
        raise SuiteSpecError(str(exc)) from exc

    return RegistrySpec(
        path=requested,
        file_sha256=file_sha256(requested),
        identity=identity,
        root=root,
        created_at_utc=created_at,
        policy_path=policy_path,
        policy_file_sha256=policy_hash,
        policy_identity=policy_plan.policy_hash,
        original=original,
        initial_champion=champion,
        holdout_quotas=MappingProxyType(
            {label: MappingProxyType(dict(quotas[label])) for label in LABELS}
        ),
        source_quotas=MappingProxyType(dict(source_quotas)),
        raw=MappingProxyType(raw),
    )


def publish_registry_spec(
    path: Path,
    *,
    registry_root: Path,
    policy_path: Path,
    original_model_path: Path,
    initial_champion_path: Path,
    initial_generation_id: str,
    created_at_utc: Optional[str] = None,
) -> RegistrySpec:
    """Publish the one immutable registry specification."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = (Path.cwd() / destination).resolve()
    if destination.exists():
        existing = load_registry_spec(destination)
        requested_root = _absolute_future_directory(registry_root, "registry root")
        if (
            existing.root != requested_root
            or existing.policy_path != _absolute_file(policy_path, "promotion policy")
            or existing.original.path != _absolute_file(original_model_path, "original model")
            or existing.initial_champion.path
            != _absolute_file(initial_champion_path, "initial champion")
            or existing.initial_champion.generation_id != initial_generation_id
        ):
            raise SuiteConflictError("existing registry specification conflicts")
        return existing

    policy = _absolute_file(policy_path, "promotion policy")
    plan = _load_policy_binding(policy)
    if not plan.machine_review_contract or plan.policy_version != POLICY_VERSION:
        raise SuiteSpecError("promotion policy must satisfy policy-v3")
    original = _absolute_file(original_model_path, "original model")
    champion = _absolute_file(initial_champion_path, "initial champion")
    original_hash = file_sha256(original)
    champion_hash = file_sha256(champion)
    if original_hash == champion_hash or original == champion:
        raise SuiteSpecError("original and initial champion must be distinct")
    generation_id = _require_id(initial_generation_id, "initial generation ID")
    created = (
        _format_utc(datetime.now(timezone.utc))
        if created_at_utc is None
        else _format_utc(_parse_utc(created_at_utc, "created_at_utc"))
    )
    quotas = {
        label: {holdout: int(plan.holdout_quotas[label][holdout]) for holdout in HOLDOUTS}
        for label in LABELS
    }
    value: Dict[str, Any] = {
        "schema_version": 1,
        "contract": REGISTRY_SPEC_CONTRACT,
        "registry_root": str(_absolute_future_directory(registry_root, "registry root")),
        "created_at_utc": created,
        "policy": {
            "path": str(policy),
            "sha256": file_sha256(policy),
            "identity": plan.policy_hash,
            "policy_version": POLICY_VERSION,
        },
        "models": {
            "original": _model_value(original, original_hash, "immutable_original"),
            "initial_champion": _model_value(
                champion,
                champion_hash,
                "frozen_champion",
                generation_id=generation_id,
            ),
        },
        "cadence": {
            "accepted_champion_interval": ACCEPTED_CHAMPION_INTERVAL,
            "maximum_age_days": MAXIMUM_AGE_DAYS,
            "source": "accepted-champion-events-and-utc-only-v1",
            "candidate_results_allowed": False,
        },
        "suite_contract": {
            "schema_version": MACHINE_SCHEMA_VERSION,
            "manifest_contract": MACHINE_MANIFEST_CONTRACT,
            "generator_contract": MACHINE_GENERATOR_CONTRACT,
            "policy_version": POLICY_VERSION,
            "review_mode": MACHINE_REVIEW_MODE,
        },
        "holdout_quotas": quotas,
        "source_quotas": {
            label: sum(quotas[label].values()) for label in LABELS
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    _mkdir_durable(destination.parent)
    _publish_canonical(destination, value, replace=False)
    return load_registry_spec(destination)


@dataclass(frozen=True)
class ValidatedSuite:
    suite_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest_identity: str
    manifest: Mapping[str, Any]
    original_sha256: str
    champion_sha256: str
    artifacts: Mapping[str, str]
    semantic_holdouts: Mapping[str, Tuple[str, ...]]


def _safe_artifact(root: Path, relative: Any, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SuiteValidationError(f"{role} path must be nonempty")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SuiteValidationError(f"{role} path is unsafe")
    path = root.joinpath(*candidate.parts)
    if path.is_symlink() or not path.is_file():
        raise SuiteValidationError(f"{role} must be a regular non-symlink file")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SuiteValidationError(f"{role} escapes the suite bundle") from exc
    return path


def _validate_artifact_binding(
    root: Path,
    value: Any,
    role: str,
    *,
    position_rows: bool,
) -> Tuple[Path, Tuple[Dict[str, Any], ...]]:
    keys = {"path", "sha256", "rowCount"}
    if not position_rows:
        keys |= {"pairCount", "scheduleId", "baseSeed"}
    checked = _require_exact_keys(value, keys, role)
    expected_hash = _require_sha256(checked["sha256"], f"{role}.sha256")
    row_count = checked["rowCount"]
    if type(row_count) is not int or row_count < 1:
        raise SuiteValidationError(f"{role}.rowCount must be positive")
    path = _safe_artifact(root, checked["path"], role)
    if file_sha256(path) != expected_hash:
        raise SuiteValidationError(f"{role} changed from its manifest hash")
    rows = _load_canonical_jsonl(path, role)
    if len(rows) != row_count:
        raise SuiteValidationError(f"{role} row count changed")
    if not position_rows:
        if (
            type(checked["pairCount"]) is not int
            or checked["pairCount"] < 1
            or checked["pairCount"] * 2 != row_count
            or not isinstance(checked["scheduleId"], str)
            or not checked["scheduleId"]
            or not isinstance(checked["baseSeed"], str)
            or not checked["baseSeed"]
        ):
            raise SuiteValidationError(f"{role} schedule metadata is invalid")
    return path, rows


def validate_suite_manifest(
    manifest_path: Path,
    spec: RegistrySpec | Path,
    *,
    expected_champion_sha256: Optional[str] = None,
) -> ValidatedSuite:
    """Validate a complete policy-v3 suite and all referenced artifacts."""

    registry_spec = load_registry_spec(spec) if isinstance(spec, (str, os.PathLike)) else spec
    try:
        path = _absolute_file(Path(manifest_path), "suite manifest")
        manifest = _load_canonical_object(path, "suite manifest")
        _require_exact_keys(manifest, _SUITE_MANIFEST_KEYS, "suite manifest")
        payload = dict(manifest)
        identity = payload.pop("manifestPayloadSha256", None)
        if (
            manifest["schemaVersion"] != MACHINE_SCHEMA_VERSION
            or manifest["manifestContract"] != MACHINE_MANIFEST_CONTRACT
            or manifest["generatorContract"] != MACHINE_GENERATOR_CONTRACT
            or manifest["policy_version"] != POLICY_VERSION
            or manifest["policy_hash"] != registry_spec.policy_identity
            or manifest["exactPolicyQuotas"] is not True
            or manifest["machineReviewOnly"] is not True
            or manifest["pairsPerPosition"] != 1
            or manifest["acceptedLabels"] != list(ALLOWED_LABELS)
            or identity != canonical_sha256(payload)
        ):
            raise SuiteValidationError("suite does not satisfy the policy-v3 contract")
        manifest_quotas = _normalize_quotas(
            manifest["policyHoldoutQuotas"], "suite policy holdout quotas"
        )
        expected_quotas = {
            label: dict(registry_spec.holdout_quotas[label]) for label in LABELS
        }
        if manifest_quotas != expected_quotas:
            raise SuiteValidationError("suite quotas differ from the frozen policy")
        if not isinstance(manifest["unassigned"], list) or not isinstance(
            manifest["exclusions"], list
        ):
            raise SuiteValidationError("suite exclusion inventories must be arrays")

        sources = manifest["sources"]
        if not isinstance(sources, list) or not sources:
            raise SuiteValidationError("suite sources must be nonempty")
        source_hashes: Dict[str, str] = {}
        for index, source in enumerate(sources):
            checked = _require_exact_keys(
                source,
                {"name", "sha256", "rowCount", "blankLineCount"},
                f"suite source {index}",
            )
            name = checked["name"]
            if not isinstance(name, str) or not name or name in source_hashes:
                raise SuiteValidationError("suite source names must be unique")
            source_hashes[name] = _require_sha256(
                checked["sha256"], f"suite source {name} hash"
            )
            if (
                type(checked["rowCount"]) is not int
                or checked["rowCount"] < 1
                or type(checked["blankLineCount"]) is not int
                or checked["blankLineCount"] < 0
            ):
                raise SuiteValidationError("suite source counts are invalid")

        curation_sources = manifest["curationSources"]
        if not isinstance(curation_sources, list) or len(curation_sources) != len(sources):
            raise SuiteValidationError(
                "every suite source needs machine-consensus provenance"
            )
        curation_names = set()
        common_models: Optional[Dict[str, str]] = None
        for index, source in enumerate(curation_sources):
            checked = _require_exact_keys(
                source, _CURATION_SOURCE_KEYS, f"curation source {index}"
            )
            name = checked["source_name"]
            if (
                not isinstance(name, str)
                or name in curation_names
                or source_hashes.get(name) != checked["output_sha256"]
                or checked["contract"] != MACHINE_REVIEW_MANIFEST_CONTRACT
                or checked["review_mode"] != MACHINE_REVIEW_MODE
                or checked["consensus_rules_version"] != 1
                or checked["policy_hash"] != registry_spec.policy_identity
                or checked["allowed_labels"] != list(ALLOWED_LABELS)
            ):
                raise SuiteValidationError(
                    "curation source lacks valid machine-consensus provenance"
                )
            curation_names.add(name)
            for hash_field in (
                "output_sha256",
                "manifest_sha256",
                "manifest_identity",
                "rejected_sha256",
            ):
                _require_sha256(checked[hash_field], f"curation {hash_field}")
            if type(checked["rejected_count"]) is not int or checked["rejected_count"] < 0:
                raise SuiteValidationError("curation rejection count is invalid")
            models = _require_exact_keys(
                checked["models"], {"original", "champion"}, "curation models"
            )
            parsed_models: Dict[str, str] = {}
            for model_role, expected_role in (
                ("original", "immutable_original"),
                ("champion", "frozen_champion"),
            ):
                model = _require_exact_keys(
                    models[model_role], {"role", "sha256"}, f"curation {model_role}"
                )
                if model["role"] != expected_role:
                    raise SuiteValidationError("curation model role is invalid")
                parsed_models[model_role] = _require_sha256(
                    model["sha256"], f"curation {model_role} hash"
                )
            if parsed_models["original"] == parsed_models["champion"]:
                raise SuiteValidationError("curation models must be distinct")
            if common_models is None:
                common_models = parsed_models
            elif parsed_models != common_models:
                raise SuiteValidationError("curation sources use different models")
        if curation_names != set(source_hashes) or common_models is None:
            raise SuiteValidationError("curation source inventory is incomplete")
        if common_models["original"] != registry_spec.original.sha256:
            raise SuiteValidationError("suite uses the wrong immutable original")
        if (
            expected_champion_sha256 is not None
            and common_models["champion"]
            != _require_sha256(expected_champion_sha256, "expected champion hash")
        ):
            raise StaleChampionError("suite was curated against a different champion")

        expected_banks = {
            (
                holdout if label == "ordinary" else f"{label}-{holdout}"
            ): (label, holdout, expected_quotas[label][holdout])
            for label in LABELS
            for holdout in HOLDOUTS
        }
        banks = manifest["banks"]
        if not isinstance(banks, list) or len(banks) != len(expected_banks):
            raise SuiteValidationError("suite has an incomplete holdout bank inventory")
        root = path.parent
        artifacts: Dict[str, str] = {"manifest.json": file_sha256(path)}
        semantics_by_holdout: Dict[str, list[str]] = {
            holdout: [] for holdout in HOLDOUTS
        }
        bank_by_name: Dict[str, Mapping[str, Any]] = {}
        for bank in banks:
            checked = _require_exact_keys(bank, _BANK_KEYS, "suite bank")
            qualified = checked["qualifiedName"]
            if qualified not in expected_banks or qualified in bank_by_name:
                raise SuiteValidationError("suite bank name is missing or duplicated")
            label, holdout, quota = expected_banks[qualified]
            if (
                checked["sourceLabel"] != label
                or checked["holdout"] != holdout
                or checked["kind"]
                != ("ordinary" if label == "ordinary" else "specialized")
            ):
                raise SuiteValidationError(f"suite bank {qualified} metadata is invalid")
            semantic_hashes = checked["semanticSha256s"]
            content_hashes = checked["contentSha256s"]
            cluster_ids = checked["independentClusterIds"]
            position_ids = checked["positionIds"]
            if (
                not isinstance(semantic_hashes, list)
                or not isinstance(content_hashes, list)
                or not isinstance(cluster_ids, list)
                or not isinstance(position_ids, list)
                or len(semantic_hashes) != quota
                or len(content_hashes) != quota
                or cluster_ids != semantic_hashes
                or len(set(semantic_hashes)) != quota
                or any(_SHA256_RE.fullmatch(item or "") is None for item in semantic_hashes)
                or any(_SHA256_RE.fullmatch(item or "") is None for item in content_hashes)
                or checked["independentClusterIdsSha256"]
                != canonical_sha256(cluster_ids)
                or checked["positionIdsSha256"] != canonical_sha256(position_ids)
            ):
                raise SuiteValidationError(f"suite bank {qualified} inventory is invalid")
            positions_path, position_rows = _validate_artifact_binding(
                root, checked["positions"], f"{qualified} positions", position_rows=True
            )
            schedule_path, schedule_rows = _validate_artifact_binding(
                root, checked["schedule"], f"{qualified} schedule", position_rows=False
            )
            calculated_semantics = [
                semantic_position_sha256(row) for row in position_rows
            ]
            calculated_content = [canonical_sha256(row) for row in position_rows]
            if calculated_semantics != semantic_hashes or calculated_content != content_hashes:
                raise SuiteValidationError(
                    f"suite bank {qualified} position inventory changed"
                )
            if any(
                row.get("suiteQualifiedName") != qualified
                or row.get("positionSemanticSha256") not in set(semantic_hashes)
                for row in schedule_rows
            ):
                raise SuiteValidationError(
                    f"suite bank {qualified} schedule binding changed"
                )
            artifacts[checked["positions"]["path"]] = file_sha256(positions_path)
            artifacts[checked["schedule"]["path"]] = file_sha256(schedule_path)
            semantics_by_holdout[holdout].extend(semantic_hashes)
            bank_by_name[qualified] = checked
        if set(bank_by_name) != set(expected_banks):
            raise SuiteValidationError("suite bank inventory differs from frozen quotas")

        seen: set[str] = set()
        for holdout in HOLDOUTS:
            current = set(semantics_by_holdout[holdout])
            overlap = seen.intersection(current)
            if overlap:
                raise SuiteValidationError(
                    "gameplay-semantic overlap across discovery/confirmation/audit"
                )
            if len(current) != len(semantics_by_holdout[holdout]):
                raise SuiteValidationError(
                    f"duplicate semantic positions within {holdout}"
                )
            seen.update(current)

        cells = manifest["cells"]
        if not isinstance(cells, list) or not cells:
            raise SuiteValidationError("suite cells must be nonempty")
        for cell in cells:
            checked = _require_exact_keys(cell, _CELL_KEYS, "suite cell")
            body = dict(checked)
            cell_id = body.pop("cell_id")
            if cell_id != "suite-cell-" + canonical_sha256(body):
                raise SuiteValidationError("suite cell identity is invalid")
            if (
                checked["policy_hash"] != registry_spec.policy_identity
                or checked["policy_version"] != POLICY_VERSION
                or checked["bank_name"] not in bank_by_name
            ):
                raise SuiteValidationError("suite cell policy/bank binding is invalid")
            bank = bank_by_name[checked["bank_name"]]
            if (
                checked["bank_path"] != bank["positions"]["path"]
                or checked["bank_hash"] != bank["positions"]["sha256"]
                or checked["independent_cluster_ids_hash"]
                != canonical_sha256(checked["independent_cluster_ids"])
                or checked["position_ids_hash"] != canonical_sha256(checked["position_ids"])
                or not set(checked["independent_cluster_ids"]).issubset(
                    set(bank["semanticSha256s"])
                )
            ):
                raise SuiteValidationError("suite cell inventory binding is invalid")
            schedule_path = _safe_artifact(
                root, checked["schedule_path"], "suite cell schedule"
            )
            if file_sha256(schedule_path) != checked["schedule_hash"]:
                raise SuiteValidationError("suite cell schedule changed")
            artifacts[checked["schedule_path"]] = checked["schedule_hash"]

        discovery = bank_by_name["discovery"]["schedule"]["sha256"]
        if manifest["discovery_schedule_hash"] != discovery:
            raise SuiteValidationError("discovery schedule hash is invalid")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(
            exc,
            (
                SuiteValidationError,
                StaleChampionError,
            ),
        ):
            raise
        raise SuiteValidationError(str(exc)) from exc

    manifest_hash = file_sha256(path)
    return ValidatedSuite(
        suite_id=manifest_hash,
        manifest_path=path,
        manifest_sha256=manifest_hash,
        manifest_identity=identity,
        manifest=MappingProxyType(manifest),
        original_sha256=common_models["original"],
        champion_sha256=common_models["champion"],
        artifacts=MappingProxyType(dict(sorted(artifacts.items()))),
        semantic_holdouts=MappingProxyType(
            {
                holdout: tuple(semantics_by_holdout[holdout])
                for holdout in HOLDOUTS
            }
        ),
    )


@dataclass(frozen=True)
class SuiteVersion:
    suite_id: str
    version_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest_identity: str
    original_sha256: str
    champion_sha256: str
    semantic_holdouts: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class RegistryEvent:
    sequence: int
    previous_event_sha256: str
    timestamp_utc: str
    event_type: str
    payload: Mapping[str, Any]
    event_sha256: str

    def body_dict(self, spec_sha256: str) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "contract": REGISTRY_EVENT_CONTRACT,
            "sequence": self.sequence,
            "previous_event_sha256": self.previous_event_sha256,
            "timestamp_utc": self.timestamp_utc,
            "spec_sha256": spec_sha256,
            "event_type": self.event_type,
            "payload": json.loads(canonical_json(self.payload)),
        }

    def to_dict(self, spec_sha256: str) -> Dict[str, Any]:
        value = self.body_dict(spec_sha256)
        value["event_sha256"] = self.event_sha256
        return value


@dataclass(frozen=True)
class RotationEligibility:
    eligible: bool
    reasons: Tuple[str, ...]
    accepted_champions_since_activation: int
    accepted_champions_remaining: int
    age_deadline_utc: Optional[str]
    trigger_at_utc: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "accepted_champions_since_activation": self.accepted_champions_since_activation,
            "accepted_champions_remaining": self.accepted_champions_remaining,
            "accepted_champion_interval": ACCEPTED_CHAMPION_INTERVAL,
            "maximum_age_days": MAXIMUM_AGE_DAYS,
            "age_deadline_utc": self.age_deadline_utc,
            "trigger_at_utc": self.trigger_at_utc,
            "source": "accepted-champion-events-and-utc-only-v1",
            "candidate_results_used": False,
        }


@dataclass(frozen=True)
class RegistryState:
    events: Tuple[RegistryEvent, ...]
    versions: Mapping[str, SuiteVersion]
    requests: Mapping[str, Mapping[str, Any]]
    registrations: Mapping[str, Mapping[str, Any]]
    continuity: Mapping[str, Mapping[str, Any]]
    boundaries: Mapping[str, Mapping[str, Any]]
    pins: Mapping[str, Mapping[str, Any]]
    champion_history: Mapping[str, ModelBinding]
    active_suite_id: Optional[str]
    active_suite_activated_at_utc: Optional[str]
    active_activation_sequence: int
    current_champion: Optional[ModelBinding]
    previous_champion_sha256: Optional[str]
    accepted_champion_timestamps: Tuple[str, ...]
    last_sequence: int
    last_event_sha256: str


def rotation_eligibility(
    state: RegistryState, now: datetime | float
) -> RotationEligibility:
    """Return frozen cadence eligibility; candidate results are not an input."""

    current = _coerce_now(now)
    count = len(state.accepted_champion_timestamps)
    remaining = max(0, ACCEPTED_CHAMPION_INTERVAL - count)
    if state.active_suite_activated_at_utc is None:
        return RotationEligibility(False, (), count, remaining, None, None)
    activated = _parse_utc(
        state.active_suite_activated_at_utc, "active suite activation timestamp"
    )
    deadline = activated + timedelta(days=MAXIMUM_AGE_DAYS)
    reasons = []
    triggers = []
    if count >= ACCEPTED_CHAMPION_INTERVAL:
        fifth = _parse_utc(
            state.accepted_champion_timestamps[ACCEPTED_CHAMPION_INTERVAL - 1],
            "fifth accepted champion timestamp",
        )
        if current >= fifth:
            reasons.append("accepted-champion-interval")
            triggers.append(fifth)
    if current >= deadline:
        reasons.append("maximum-age")
        triggers.append(deadline)
    trigger = min(triggers) if triggers else None
    return RotationEligibility(
        bool(reasons),
        tuple(reasons),
        count,
        remaining,
        _format_utc(deadline),
        _format_utc(trigger) if trigger is not None else None,
    )


class _RegistryLock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor: Optional[int] = None

    def __enter__(self) -> "_RegistryLock":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(self.path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SuiteRegistryCorruption("registry lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise SuiteRegistryBusy(
                    f"another process owns {self.path}"
                ) from exc
            metadata = canonical_json_bytes(
                {"pid": os.getpid(), "acquired_at_utc": _format_utc(datetime.now(timezone.utc))}
            ) + b"\n"
            os.ftruncate(descriptor, 0)
            offset = 0
            while offset < len(metadata):
                written = os.write(descriptor, metadata[offset:])
                if written <= 0:
                    raise OSError("short write while recording registry lock owner")
                offset += written
            os.fsync(descriptor)
            fsync_directory(self.path.parent)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        descriptor = self.descriptor
        self.descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class SuiteRotationRegistry:
    """Authoritative suite registry and one-pass rotation reconciler."""

    def __init__(
        self,
        spec_path: Path,
        *,
        clock: Callable[[], Any] = lambda: datetime.now(timezone.utc),
        failure_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.spec = load_registry_spec(spec_path)
        self.clock = clock
        self.failure_hook = failure_hook or (lambda _: None)
        self.root = self.spec.root
        self.events_dir = self.root / "events"
        self.suites_dir = self.root / "suites"
        self.versions_dir = self.root / "versions"
        self.requests_dir = self.root / "requests"
        self.continuity_dir = self.root / "continuity"
        self.active_path = self.root / "active-suite.json"
        self.status_path = self.root / "status.json"
        self.lock_path = self.root / ".suite-rotation.lock"

    def _now(self) -> datetime:
        return _coerce_now(self.clock())

    def _ensure_layout(self) -> None:
        _mkdir_durable(self.root)
        for directory in (
            self.events_dir,
            self.suites_dir,
            self.versions_dir,
            self.requests_dir,
            self.continuity_dir,
        ):
            _mkdir_durable(directory)

    def _assert_frozen_inputs(self) -> None:
        frozen = (
            (self.spec.path, self.spec.file_sha256, "registry specification"),
            (self.spec.policy_path, self.spec.policy_file_sha256, "promotion policy"),
            (self.spec.original.path, self.spec.original.sha256, "immutable original"),
            (
                self.spec.initial_champion.path,
                self.spec.initial_champion.sha256,
                "initial champion",
            ),
        )
        for path, expected_hash, role in frozen:
            if (
                path.is_symlink()
                or not path.is_file()
                or file_sha256(path) != expected_hash
            ):
                raise SuiteRegistryCorruption(f"frozen {role} changed")

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure_layout()
        with _RegistryLock(self.lock_path):
            yield

    def _version_path(self, suite_id: str) -> Path:
        _require_sha256(suite_id, "suite ID")
        return self.versions_dir / f"{suite_id}.json"

    def _suite_path(self, suite_id: str) -> Path:
        _require_sha256(suite_id, "suite ID")
        return self.suites_dir / suite_id

    def _copy_fsynced(self, source: Path, destination: Path) -> None:
        descriptor = os.open(
            os.fspath(destination),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                descriptor = -1
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def _publish_suite(self, validated: ValidatedSuite) -> SuiteVersion:
        destination = self._suite_path(validated.suite_id)
        if not destination.exists():
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{validated.suite_id}.partial-", dir=os.fspath(self.suites_dir)
                )
            )
            try:
                for relative in sorted(validated.artifacts):
                    output_relative = Path(relative)
                    source = (
                        validated.manifest_path
                        if relative == "manifest.json"
                        else validated.manifest_path.parent.joinpath(*output_relative.parts)
                    )
                    output = temporary.joinpath(*output_relative.parts)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    self._copy_fsynced(source, output)
                for current_root, directories, _ in os.walk(temporary, topdown=False):
                    for name in directories:
                        fsync_directory(Path(current_root) / name)
                    fsync_directory(Path(current_root))
                try:
                    os.rename(os.fspath(temporary), os.fspath(destination))
                    fsync_directory(self.suites_dir)
                    temporary = None  # type: ignore[assignment]
                except OSError:
                    if not destination.exists():
                        raise
            finally:
                if temporary is not None and temporary.exists():
                    shutil.rmtree(temporary)
        copied = validate_suite_manifest(
            destination / "manifest.json",
            self.spec,
            expected_champion_sha256=validated.champion_sha256,
        )
        if copied.suite_id != validated.suite_id or dict(copied.artifacts) != dict(
            validated.artifacts
        ):
            raise SuiteRegistryCorruption("published suite snapshot is contradictory")

        semantic_summary = {
            holdout: {
                "count": len(validated.semantic_holdouts[holdout]),
                "sha256": canonical_sha256(
                    sorted(validated.semantic_holdouts[holdout])
                ),
            }
            for holdout in HOLDOUTS
        }
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": SUITE_VERSION_CONTRACT,
            "suite_id": validated.suite_id,
            "manifest": {
                "path": str((destination / "manifest.json").resolve()),
                "sha256": validated.manifest_sha256,
                "identity": validated.manifest_identity,
            },
            "policy": {
                "sha256": self.spec.policy_file_sha256,
                "identity": self.spec.policy_identity,
                "version": POLICY_VERSION,
            },
            "models": {
                "original_sha256": validated.original_sha256,
                "champion_sha256": validated.champion_sha256,
            },
            "holdout_quotas": {
                label: dict(self.spec.holdout_quotas[label]) for label in LABELS
            },
            "semantic_holdouts": semantic_summary,
            "artifacts": dict(validated.artifacts),
        }
        value["version_sha256"] = canonical_sha256(value)
        version_path = self._version_path(validated.suite_id)
        if version_path.exists():
            existing = _load_canonical_object(version_path, "suite version")
            if existing != value:
                raise SuiteRegistryCorruption("suite version record is contradictory")
        else:
            _publish_canonical(version_path, value, replace=False)
        return self._load_version(validated.suite_id)

    def _load_version(self, suite_id: str) -> SuiteVersion:
        path = self._version_path(suite_id)
        try:
            value = _load_canonical_object(path, f"suite version {suite_id}")
            _require_exact_keys(
                value,
                {
                    "schema_version",
                    "contract",
                    "suite_id",
                    "manifest",
                    "policy",
                    "models",
                    "holdout_quotas",
                    "semantic_holdouts",
                    "artifacts",
                    "version_sha256",
                },
                "suite version",
            )
            payload = dict(value)
            version_hash = payload.pop("version_sha256")
            if (
                value["schema_version"] != 1
                or value["contract"] != SUITE_VERSION_CONTRACT
                or value["suite_id"] != suite_id
                or _require_sha256(version_hash, "suite version hash")
                != canonical_sha256(payload)
                or value["policy"]
                != {
                    "sha256": self.spec.policy_file_sha256,
                    "identity": self.spec.policy_identity,
                    "version": POLICY_VERSION,
                }
            ):
                raise ValueError("suite version identity is invalid")
            models = _require_exact_keys(
                value["models"],
                {"original_sha256", "champion_sha256"},
                "suite version models",
            )
            original_hash = _require_sha256(
                models["original_sha256"], "suite original hash"
            )
            champion_hash = _require_sha256(
                models["champion_sha256"], "suite champion hash"
            )
            manifest_binding = _require_exact_keys(
                value["manifest"], {"path", "sha256", "identity"}, "suite manifest binding"
            )
            manifest_path = _absolute_file(
                Path(manifest_binding["path"]),
                "registered suite manifest",
                manifest_binding["sha256"],
            )
            if manifest_path != (self._suite_path(suite_id) / "manifest.json").resolve():
                raise ValueError("suite version manifest path is not registry-owned")
            validated = validate_suite_manifest(
                manifest_path, self.spec, expected_champion_sha256=champion_hash
            )
            bundle_root = self._suite_path(suite_id)
            actual_artifacts = set()
            for current_root, directories, filenames in os.walk(
                bundle_root, followlinks=False
            ):
                current = Path(current_root)
                for directory_name in directories:
                    directory = current / directory_name
                    if directory.is_symlink():
                        raise ValueError("registered suite contains a symlink directory")
                for filename in filenames:
                    artifact = current / filename
                    if artifact.is_symlink() or not artifact.is_file():
                        raise ValueError("registered suite contains an unsafe artifact")
                    actual_artifacts.add(
                        artifact.relative_to(bundle_root).as_posix()
                    )
            if (
                validated.suite_id != suite_id
                or validated.manifest_identity != manifest_binding["identity"]
                or validated.original_sha256 != original_hash
                or dict(validated.artifacts) != value["artifacts"]
                or actual_artifacts != set(value["artifacts"])
            ):
                raise ValueError("suite version does not match its immutable bundle")
            semantic = _require_exact_keys(
                value["semantic_holdouts"], set(HOLDOUTS), "semantic holdouts"
            )
            for holdout in HOLDOUTS:
                expected = {
                    "count": len(validated.semantic_holdouts[holdout]),
                    "sha256": canonical_sha256(
                        sorted(validated.semantic_holdouts[holdout])
                    ),
                }
                if semantic[holdout] != expected:
                    raise ValueError("suite semantic inventory changed")
        except (OSError, ValueError) as exc:
            if isinstance(exc, SuiteRegistryCorruption):
                raise
            raise SuiteRegistryCorruption(str(exc)) from exc
        return SuiteVersion(
            suite_id=suite_id,
            version_sha256=version_hash,
            manifest_path=manifest_path,
            manifest_sha256=validated.manifest_sha256,
            manifest_identity=validated.manifest_identity,
            original_sha256=original_hash,
            champion_sha256=champion_hash,
            semantic_holdouts=MappingProxyType(
                {
                    holdout: MappingProxyType(dict(semantic[holdout]))
                    for holdout in HOLDOUTS
                }
            ),
        )

    def _event_paths(self) -> Tuple[Tuple[int, Path], ...]:
        if not self.events_dir.exists():
            return ()
        if self.events_dir.is_symlink() or not self.events_dir.is_dir():
            raise SuiteRegistryCorruption("events directory is unsafe")
        result = []
        with os.scandir(self.events_dir) as entries:
            for entry in entries:
                match = _EVENT_FILE_RE.fullmatch(entry.name)
                if match is not None:
                    result.append((int(match.group(1)), Path(entry.path)))
                elif not entry.name.startswith("."):
                    raise SuiteRegistryCorruption(
                        f"unexpected registry event entry: {entry.name}"
                    )
        return tuple(sorted(result))

    def _parse_event(self, value: Any, filename_sequence: int) -> RegistryEvent:
        checked = _require_exact_keys(value, _EVENT_KEYS, "registry event")
        payload = checked["payload"]
        event_type = checked["event_type"]
        if event_type not in _EVENT_PAYLOAD_KEYS:
            raise SuiteRegistryCorruption(f"unsupported event type: {event_type!r}")
        _require_exact_keys(
            payload, _EVENT_PAYLOAD_KEYS[event_type], f"{event_type} payload"
        )
        body = dict(checked)
        event_hash = body.pop("event_sha256")
        if (
            checked["schema_version"] != 1
            or checked["contract"] != REGISTRY_EVENT_CONTRACT
            or checked["sequence"] != filename_sequence
            or checked["spec_sha256"] != self.spec.identity
            or _require_sha256(event_hash, "event hash") != canonical_sha256(body)
        ):
            raise SuiteRegistryCorruption("registry event identity is invalid")
        _parse_utc(checked["timestamp_utc"], "event timestamp")
        return RegistryEvent(
            sequence=filename_sequence,
            previous_event_sha256=_require_sha256(
                checked["previous_event_sha256"], "previous event hash"
            ),
            timestamp_utc=checked["timestamp_utc"],
            event_type=event_type,
            payload=MappingProxyType(dict(payload)),
            event_sha256=event_hash,
        )

    def _validate_model_event(
        self, value: Any, role: str = "event champion"
    ) -> ModelBinding:
        return _parse_model_binding(
            value, role, "frozen_champion", with_generation=False
        )

    def reconstruct(self) -> RegistryState:
        self._assert_frozen_inputs()
        events = []
        versions: Dict[str, SuiteVersion] = {}
        requests: Dict[str, Mapping[str, Any]] = {}
        registrations: Dict[str, Mapping[str, Any]] = {}
        continuity: Dict[str, Mapping[str, Any]] = {}
        boundaries: Dict[str, Mapping[str, Any]] = {}
        pins: Dict[str, Mapping[str, Any]] = {}
        history: Dict[str, ModelBinding] = {
            self.spec.initial_champion.sha256: self.spec.initial_champion
        }
        active_suite: Optional[str] = None
        active_at: Optional[str] = None
        activation_sequence = 0
        current: Optional[ModelBinding] = None
        previous_champion: Optional[str] = None
        accepted_timestamps: list[str] = []
        previous_hash = GENESIS_HASH
        prior_timestamp: Optional[datetime] = None

        for expected_sequence, (filename_sequence, path) in enumerate(
            self._event_paths(), start=1
        ):
            if filename_sequence != expected_sequence:
                raise SuiteRegistryCorruption(
                    f"event sequence gap: expected {expected_sequence}, found {filename_sequence}"
                )
            try:
                raw = _load_canonical_object(path, f"registry event {filename_sequence}")
                event = self._parse_event(raw, filename_sequence)
            except (OSError, ValueError) as exc:
                if isinstance(exc, SuiteRegistryCorruption):
                    raise
                raise SuiteRegistryCorruption(str(exc)) from exc
            if event.previous_event_sha256 != previous_hash:
                raise SuiteRegistryCorruption("registry event hash chain is broken")
            timestamp = _parse_utc(event.timestamp_utc, "event timestamp")
            if prior_timestamp is not None and timestamp < prior_timestamp:
                raise SuiteRegistryCorruption("registry event timestamps regress")
            prior_timestamp = timestamp
            payload = event.payload

            if event.event_type == "registry.bootstrapped":
                if events or active_suite is not None:
                    raise SuiteRegistryCorruption("registry may bootstrap only once")
                suite_id = _require_sha256(payload["suite_id"], "bootstrap suite ID")
                version = self._load_version(suite_id)
                champion = self._validate_model_event(payload["champion"])
                generation = _require_id(payload["generation_id"], "bootstrap generation")
                bootstrap_previous = payload["previous_champion_sha256"]
                if bootstrap_previous is not None:
                    bootstrap_previous = _require_sha256(
                        bootstrap_previous, "bootstrap previous champion hash"
                    )
                    if bootstrap_previous == champion.sha256:
                        raise SuiteRegistryCorruption(
                            "bootstrap previous champion equals current champion"
                        )
                if (
                    payload["version_sha256"] != version.version_sha256
                    or payload["manifest_sha256"] != version.manifest_sha256
                    or payload["manifest_identity"] != version.manifest_identity
                    or champion.sha256 != self.spec.initial_champion.sha256
                    or champion.path != self.spec.initial_champion.path
                    or generation != self.spec.initial_champion.generation_id
                    or version.champion_sha256 != champion.sha256
                    or _format_utc(
                        _parse_utc(payload["activated_at_utc"], "activation timestamp")
                    )
                    != event.timestamp_utc
                ):
                    raise SuiteRegistryCorruption("bootstrap binding is contradictory")
                versions[suite_id] = version
                current = ModelBinding(
                    champion.role, champion.path, champion.sha256, generation
                )
                history[champion.sha256] = current
                previous_champion = bootstrap_previous
                active_suite = suite_id
                active_at = event.timestamp_utc
                activation_sequence = event.sequence
                accepted_timestamps = []
            elif current is None or active_suite is None:
                raise SuiteRegistryCorruption("registry bootstrap must be first")
            elif event.event_type == "champion.accepted":
                champion = self._validate_model_event(payload["champion"])
                generation = _require_id(
                    payload["generation_id"], "accepted champion generation"
                )
                prior = _require_sha256(
                    payload["previous_champion_sha256"], "previous champion hash"
                )
                if prior != current.sha256 or champion.sha256 == prior:
                    raise SuiteRegistryCorruption(
                        "accepted champion compare-and-swap is stale"
                    )
                if any(
                    item.generation_id == generation for item in history.values()
                ):
                    raise SuiteRegistryCorruption("champion generation was reused")
                previous_champion = prior
                current = ModelBinding(
                    champion.role, champion.path, champion.sha256, generation
                )
                history[champion.sha256] = current
                accepted_timestamps.append(event.timestamp_utc)
            elif event.event_type == "evaluation.pinned":
                evaluation_id = _require_id(
                    payload["evaluation_id"], "evaluation ID"
                )
                if (
                    evaluation_id in pins
                    or payload["suite_id"] != active_suite
                    or payload["champion_sha256"] != current.sha256
                    or payload["generation_id"] != current.generation_id
                ):
                    raise SuiteRegistryCorruption("evaluation pin is contradictory")
                pins[evaluation_id] = MappingProxyType(
                    {**dict(payload), "_sequence": event.sequence}
                )
            elif event.event_type == "evaluation.unpinned":
                evaluation_id = _require_id(
                    payload["evaluation_id"], "evaluation ID"
                )
                pin = pins.get(evaluation_id)
                if pin is None or any(
                    payload[key] != pin[key]
                    for key in (
                        "suite_id",
                        "champion_sha256",
                        "generation_id",
                    )
                ):
                    raise SuiteRegistryCorruption("evaluation unpin is contradictory")
                del pins[evaluation_id]
            elif event.event_type == "rotation.requested":
                request_id = _require_id(payload["request_id"], "request ID")
                if request_id in requests:
                    raise SuiteRegistryCorruption("rotation request was duplicated")
                if (
                    payload["base_suite_id"] != active_suite
                    or payload["champion_sha256"] != current.sha256
                    or payload["generation_id"] != current.generation_id
                ):
                    raise SuiteRegistryCorruption("rotation request is stale")
                request = self._validate_request_binding(
                    payload["request_manifest"], request_id
                )
                eligibility = rotation_eligibility(
                    RegistryState(
                        tuple(events),
                        MappingProxyType(versions),
                        MappingProxyType(requests),
                        MappingProxyType(registrations),
                        MappingProxyType(continuity),
                        MappingProxyType(boundaries),
                        MappingProxyType(pins),
                        MappingProxyType(history),
                        active_suite,
                        active_at,
                        activation_sequence,
                        current,
                        previous_champion,
                        tuple(accepted_timestamps),
                        event.sequence - 1,
                        previous_hash,
                    ),
                    timestamp,
                )
                if (
                    not eligibility.eligible
                    or payload["trigger_at_utc"] != eligibility.trigger_at_utc
                    or request["base_active_suite"]["suite_id"] != active_suite
                    or request["models"]["champion"]["sha256"] != current.sha256
                ):
                    raise SuiteRegistryCorruption(
                        "rotation request was not cadence-eligible"
                    )
                requests[request_id] = MappingProxyType(
                    {
                        **dict(payload),
                        "_sequence": event.sequence,
                        "_manifest": request,
                    }
                )
            elif event.event_type == "suite.registered":
                suite_id = _require_sha256(payload["suite_id"], "registered suite ID")
                request_id = _require_id(payload["request_id"], "request ID")
                request = requests.get(request_id)
                if request is None or suite_id in registrations:
                    raise SuiteRegistryCorruption("suite registration is contradictory")
                version = self._load_version(suite_id)
                if (
                    payload["version_sha256"] != version.version_sha256
                    or payload["manifest_sha256"] != version.manifest_sha256
                    or payload["manifest_identity"] != version.manifest_identity
                    or payload["curation_champion_sha256"]
                    != request["champion_sha256"]
                    or version.champion_sha256 != request["champion_sha256"]
                ):
                    raise SuiteRegistryCorruption("registered suite binding changed")
                versions[suite_id] = version
                registrations[suite_id] = MappingProxyType(
                    {**dict(payload), "_sequence": event.sequence}
                )
            elif event.event_type == "continuity.recorded":
                suite_id = _require_sha256(payload["suite_id"], "continuity suite ID")
                request_id = _require_id(payload["request_id"], "request ID")
                if (
                    suite_id not in registrations
                    or registrations[suite_id]["request_id"] != request_id
                    or suite_id in continuity
                    or payload["current_champion_sha256"] != current.sha256
                    or payload["previous_champion_sha256"] != previous_champion
                ):
                    raise SuiteRegistryCorruption("continuity receipt is contradictory")
                receipt = self._validate_continuity_binding(
                    payload["manifest"],
                    request_id=request_id,
                    suite_id=suite_id,
                    base_suite_id=requests[request_id]["base_suite_id"],
                    current_champion_sha256=current.sha256,
                    previous_champion_sha256=previous_champion,
                )
                continuity[suite_id] = MappingProxyType(
                    {
                        **dict(payload),
                        "_sequence": event.sequence,
                        "_manifest": receipt,
                    }
                )
            elif event.event_type == "generation.boundary":
                boundary_id = _require_id(payload["boundary_id"], "boundary ID")
                if (
                    boundary_id in boundaries
                    or payload["champion_sha256"] != current.sha256
                    or payload["generation_id"] != current.generation_id
                    or payload["clean"] is not True
                ):
                    raise SuiteRegistryCorruption("generation boundary is contradictory")
                boundaries[boundary_id] = MappingProxyType(
                    {**dict(payload), "_sequence": event.sequence}
                )
            elif event.event_type == "suite.activated":
                suite_id = _require_sha256(payload["suite_id"], "activated suite ID")
                request_id = _require_id(payload["request_id"], "request ID")
                boundary = boundaries.get(payload["boundary_id"])
                receipt = continuity.get(suite_id)
                registration = registrations.get(suite_id)
                if (
                    registration is None
                    or registration["request_id"] != request_id
                    or receipt is None
                    or receipt["request_id"] != request_id
                    or payload["previous_suite_id"] != active_suite
                    or requests[request_id]["base_suite_id"] != active_suite
                    or payload["expected_champion_sha256"] != current.sha256
                    or payload["generation_id"] != current.generation_id
                    or boundary is None
                    or boundary["champion_sha256"] != current.sha256
                    or boundary["generation_id"] != current.generation_id
                    or boundary["_sequence"] <= receipt["_sequence"]
                    or pins
                    or payload["continuity_manifest_sha256"]
                    != receipt["manifest"]["sha256"]
                ):
                    raise SuiteRegistryCorruption("suite activation is contradictory")
                active_suite = suite_id
                active_at = event.timestamp_utc
                activation_sequence = event.sequence
                accepted_timestamps = []
            else:  # pragma: no cover - guarded by event parser.
                raise SuiteRegistryCorruption("unsupported event")

            events.append(event)
            previous_hash = event.event_sha256

        return RegistryState(
            events=tuple(events),
            versions=MappingProxyType(dict(versions)),
            requests=MappingProxyType(dict(requests)),
            registrations=MappingProxyType(dict(registrations)),
            continuity=MappingProxyType(dict(continuity)),
            boundaries=MappingProxyType(dict(boundaries)),
            pins=MappingProxyType(dict(pins)),
            champion_history=MappingProxyType(dict(history)),
            active_suite_id=active_suite,
            active_suite_activated_at_utc=active_at,
            active_activation_sequence=activation_sequence,
            current_champion=current,
            previous_champion_sha256=previous_champion,
            accepted_champion_timestamps=tuple(accepted_timestamps),
            last_sequence=len(events),
            last_event_sha256=previous_hash,
        )

    def _append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        timestamp: Optional[datetime] = None,
    ) -> RegistryEvent:
        _require_exact_keys(
            payload, _EVENT_PAYLOAD_KEYS[event_type], f"{event_type} payload"
        )
        state = self.reconstruct()
        now = self._now() if timestamp is None else timestamp.astimezone(timezone.utc)
        if state.events and now < _parse_utc(state.events[-1].timestamp_utc):
            raise SuiteConflictError("event timestamp would regress")
        body = {
            "schema_version": 1,
            "contract": REGISTRY_EVENT_CONTRACT,
            "sequence": state.last_sequence + 1,
            "previous_event_sha256": state.last_event_sha256,
            "timestamp_utc": _format_utc(now),
            "spec_sha256": self.spec.identity,
            "event_type": event_type,
            "payload": json.loads(canonical_json(payload)),
        }
        event_hash = canonical_sha256(body)
        value = dict(body)
        value["event_sha256"] = event_hash
        destination = self.events_dir / f"{body['sequence']:0{EVENT_SEQUENCE_WIDTH}d}.json"
        _publish_canonical(destination, value, replace=False)
        self.failure_hook("event-published")
        return RegistryEvent(
            sequence=body["sequence"],
            previous_event_sha256=body["previous_event_sha256"],
            timestamp_utc=body["timestamp_utc"],
            event_type=event_type,
            payload=MappingProxyType(dict(payload)),
            event_sha256=event_hash,
        )

    def bootstrap(
        self,
        suite_manifest_path: Path,
        *,
        previous_champion_sha256: Optional[str] = None,
    ) -> RegistryEvent:
        with self._locked():
            state = self.reconstruct()
            if state.active_suite_id is not None:
                first = state.events[0]
                if (
                    first.event_type == "registry.bootstrapped"
                    and state.versions[state.active_suite_id].champion_sha256
                    == self.spec.initial_champion.sha256
                    and first.payload["previous_champion_sha256"]
                    == previous_champion_sha256
                ):
                    self._repair_active_projection(state)
                    return first
                raise SuiteConflictError("suite registry is already bootstrapped")
            validated = validate_suite_manifest(
                suite_manifest_path,
                self.spec,
                expected_champion_sha256=self.spec.initial_champion.sha256,
            )
            version = self._publish_suite(validated)
            self.failure_hook("suite-version-published")
            if previous_champion_sha256 is not None:
                previous_champion_sha256 = _require_sha256(
                    previous_champion_sha256, "previous champion hash"
                )
                if previous_champion_sha256 == self.spec.initial_champion.sha256:
                    raise SuiteConflictError(
                        "previous champion must differ from initial champion"
                    )
            now = self._now()
            event = self._append_event(
                "registry.bootstrapped",
                {
                    "suite_id": version.suite_id,
                    "version_sha256": version.version_sha256,
                    "manifest_sha256": version.manifest_sha256,
                    "manifest_identity": version.manifest_identity,
                    "champion": self.spec.initial_champion.to_dict(
                        include_generation=False
                    ),
                    "previous_champion_sha256": previous_champion_sha256,
                    "generation_id": self.spec.initial_champion.generation_id,
                    "activated_at_utc": _format_utc(now),
                },
                timestamp=now,
            )
            state = self.reconstruct()
            self._repair_active_projection(state)
            return event

    def record_accepted_champion(
        self,
        champion_path: Path,
        *,
        generation_id: str,
        expected_previous_champion_sha256: str,
        timestamp_utc: Optional[str] = None,
    ) -> RegistryEvent:
        with self._locked():
            state = self.reconstruct()
            if state.current_champion is None:
                raise SuiteConflictError("registry is not bootstrapped")
            generation = _require_id(generation_id, "generation ID")
            previous = _require_sha256(
                expected_previous_champion_sha256, "expected previous champion"
            )
            path = _absolute_file(champion_path, "accepted champion")
            champion_hash = file_sha256(path)
            for event in state.events:
                if (
                    event.event_type == "champion.accepted"
                    and event.payload["generation_id"] == generation
                ):
                    if (
                        event.payload["champion"]["path"] == str(path)
                        and event.payload["champion"]["sha256"] == champion_hash
                        and event.payload["previous_champion_sha256"] == previous
                    ):
                        return event
                    raise SuiteConflictError("champion generation conflicts with retry")
            if state.current_champion.sha256 != previous:
                raise StaleChampionError(
                    f"expected champion {previous}, current is {state.current_champion.sha256}"
                )
            if champion_hash == previous:
                raise SuiteConflictError("accepted champion must change content")
            timestamp = (
                None
                if timestamp_utc is None
                else _parse_utc(timestamp_utc, "accepted champion timestamp")
            )
            return self._append_event(
                "champion.accepted",
                {
                    "champion": _model_value(
                        path, champion_hash, "frozen_champion"
                    ),
                    "generation_id": generation,
                    "previous_champion_sha256": previous,
                },
                timestamp=timestamp,
            )

    def pin_evaluation(
        self,
        evaluation_id: str,
        *,
        expected_active_suite_id: Optional[str] = None,
        expected_champion_sha256: Optional[str] = None,
    ) -> Mapping[str, Any]:
        with self._locked():
            state = self.reconstruct()
            if state.active_suite_id is None or state.current_champion is None:
                raise SuiteConflictError("registry is not bootstrapped")
            identifier = _require_id(evaluation_id, "evaluation ID")
            existing = state.pins.get(identifier)
            if existing is not None:
                if (
                    expected_active_suite_id in (None, existing["suite_id"])
                    and expected_champion_sha256
                    in (None, existing["champion_sha256"])
                ):
                    return existing
                raise EvaluationPinConflictError(
                    f"evaluation {identifier} is already pinned"
                )
            if (
                expected_active_suite_id is not None
                and expected_active_suite_id != state.active_suite_id
            ):
                raise StaleActiveSuiteError("active suite changed before evaluation pin")
            if (
                expected_champion_sha256 is not None
                and expected_champion_sha256 != state.current_champion.sha256
            ):
                raise StaleChampionError("champion changed before evaluation pin")
            payload = {
                "evaluation_id": identifier,
                "suite_id": state.active_suite_id,
                "champion_sha256": state.current_champion.sha256,
                "generation_id": state.current_champion.generation_id,
            }
            self._append_event("evaluation.pinned", payload)
            return self.reconstruct().pins[identifier]

    def unpin_evaluation(self, evaluation_id: str) -> Optional[RegistryEvent]:
        with self._locked():
            state = self.reconstruct()
            identifier = _require_id(evaluation_id, "evaluation ID")
            pin = state.pins.get(identifier)
            if pin is None:
                return None
            return self._append_event(
                "evaluation.unpinned",
                {
                    key: pin[key]
                    for key in (
                        "evaluation_id",
                        "suite_id",
                        "champion_sha256",
                        "generation_id",
                    )
                },
            )

    def _request_identity(
        self, state: RegistryState, eligibility: RotationEligibility
    ) -> str:
        assert state.active_suite_id is not None
        assert state.current_champion is not None
        return "rotation-" + canonical_sha256(
            {
                "contract": ROTATION_REQUEST_CONTRACT,
                "base_suite_id": state.active_suite_id,
                "champion_sha256": state.current_champion.sha256,
                "generation_id": state.current_champion.generation_id,
                "trigger_at_utc": eligibility.trigger_at_utc,
            }
        )

    def _publish_request_bundle(
        self,
        state: RegistryState,
        eligibility: RotationEligibility,
    ) -> Tuple[str, Mapping[str, Any]]:
        assert state.current_champion is not None
        assert state.active_suite_id is not None
        request_id = self._request_identity(state, eligibility)
        destination = self.requests_dir / request_id
        supplement_path = destination / "curation-supplement.json"
        pipeline_path = destination / "curation-pipeline.json"
        manifest_path = destination / "manifest.json"
        models = {
            "original": self.spec.original.to_dict(include_generation=False),
            "champion": state.current_champion.to_dict(include_generation=False),
        }
        policy = {
            "path": str(self.spec.policy_path),
            "sha256": self.spec.policy_file_sha256,
            "identity": self.spec.policy_identity,
            "version": POLICY_VERSION,
        }
        supplement: Dict[str, Any] = {
            "schema_version": 1,
            "contract": SUPPLEMENT_REQUEST_CONTRACT,
            "requested_spec_contract": SUPPLEMENT_SPEC_CONTRACT,
            "request_id": request_id,
            "models": models,
            "policy": policy,
            "target_counts": {
                label: self.spec.source_quotas[label]
                for label in ("lead-40", "lead-80")
            },
            "quarantined_source_generation": True,
            "output_root": str((destination / "supplement").resolve()),
        }
        supplement["request_sha256"] = canonical_sha256(supplement)
        supplement_bytes_hash = hashlib.sha256(
            canonical_json_bytes(supplement) + b"\n"
        ).hexdigest()
        pipeline: Dict[str, Any] = {
            "schema_version": 1,
            "contract": PIPELINE_REQUEST_CONTRACT,
            "requested_spec_contract": PIPELINE_SPEC_CONTRACT,
            "request_id": request_id,
            "models": models,
            "policy": policy,
            "source_quotas": dict(self.spec.source_quotas),
            "holdout_quotas": {
                label: dict(self.spec.holdout_quotas[label]) for label in LABELS
            },
            "supplement_request": {
                "path": str(supplement_path.resolve()),
                "sha256": supplement_bytes_hash,
                "identity": supplement["request_sha256"],
            },
            "suite_seed": "suite-rotation-" + request_id.removeprefix("rotation-"),
            "output_suite_contract": MACHINE_MANIFEST_CONTRACT,
            "output_root": str((destination / "pipeline").resolve()),
        }
        pipeline["request_sha256"] = canonical_sha256(pipeline)
        pipeline_bytes_hash = hashlib.sha256(
            canonical_json_bytes(pipeline) + b"\n"
        ).hexdigest()
        request: Dict[str, Any] = {
            "schema_version": 1,
            "contract": ROTATION_REQUEST_CONTRACT,
            "request_id": request_id,
            "registry_spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "base_active_suite": {
                "suite_id": state.active_suite_id,
                "version_sha256": state.versions[
                    state.active_suite_id
                ].version_sha256,
            },
            "models": models,
            "policy": policy,
            "trigger": eligibility.to_dict(),
            "requests": {
                "curation_supplement": {
                    "path": str(supplement_path.resolve()),
                    "sha256": supplement_bytes_hash,
                    "identity": supplement["request_sha256"],
                },
                "curation_pipeline": {
                    "path": str(pipeline_path.resolve()),
                    "sha256": pipeline_bytes_hash,
                    "identity": pipeline["request_sha256"],
                },
            },
        }
        request["request_sha256"] = canonical_sha256(request)
        if destination.exists():
            existing = self._validate_request_binding(
                {
                    "path": str(manifest_path.resolve()),
                    "sha256": file_sha256(manifest_path),
                    "identity": request["request_sha256"],
                },
                request_id,
            )
            if existing != request:
                raise SuiteRegistryCorruption("rotation request bundle is contradictory")
            return request_id, request

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{request_id}.partial-", dir=os.fspath(self.requests_dir))
        )
        try:
            _publish_canonical(temporary / "curation-supplement.json", supplement, replace=False)
            _publish_canonical(temporary / "curation-pipeline.json", pipeline, replace=False)
            _publish_canonical(temporary / "manifest.json", request, replace=False)
            fsync_directory(temporary)
            os.rename(os.fspath(temporary), os.fspath(destination))
            fsync_directory(self.requests_dir)
            temporary = None  # type: ignore[assignment]
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
        self.failure_hook("rotation-request-published")
        return request_id, request

    def _validate_request_binding(
        self, binding: Any, request_id: str
    ) -> Mapping[str, Any]:
        checked_binding = _require_exact_keys(
            binding, {"path", "sha256", "identity"}, "rotation request binding"
        )
        expected_hash = _require_sha256(
            checked_binding["sha256"], "rotation request file hash"
        )
        path = _absolute_file(
            Path(checked_binding["path"]), "rotation request manifest", expected_hash
        )
        expected_path = (self.requests_dir / request_id / "manifest.json").resolve()
        if path != expected_path:
            raise SuiteRegistryCorruption("rotation request is not registry-owned")
        request = _load_canonical_object(path, "rotation request manifest")
        _require_exact_keys(
            request,
            {
                "schema_version",
                "contract",
                "request_id",
                "registry_spec",
                "base_active_suite",
                "models",
                "policy",
                "trigger",
                "requests",
                "request_sha256",
            },
            "rotation request manifest",
        )
        payload = dict(request)
        identity = payload.pop("request_sha256")
        if (
            request["schema_version"] != 1
            or request["contract"] != ROTATION_REQUEST_CONTRACT
            or request["request_id"] != request_id
            or identity != canonical_sha256(payload)
            or identity != checked_binding["identity"]
            or request["registry_spec"]
            != {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            }
        ):
            raise SuiteRegistryCorruption("rotation request identity is invalid")
        children = _require_exact_keys(
            request["requests"],
            {"curation_supplement", "curation_pipeline"},
            "rotation child requests",
        )
        for name, contract in (
            ("curation_supplement", SUPPLEMENT_REQUEST_CONTRACT),
            ("curation_pipeline", PIPELINE_REQUEST_CONTRACT),
        ):
            child_binding = _require_exact_keys(
                children[name], {"path", "sha256", "identity"}, f"{name} request binding"
            )
            child_path = _absolute_file(
                Path(child_binding["path"]),
                f"{name} request",
                _require_sha256(child_binding["sha256"], f"{name} file hash"),
            )
            child = _load_canonical_object(child_path, f"{name} request")
            expected_child_keys = (
                {
                    "schema_version",
                    "contract",
                    "requested_spec_contract",
                    "request_id",
                    "models",
                    "policy",
                    "target_counts",
                    "quarantined_source_generation",
                    "output_root",
                    "request_sha256",
                }
                if name == "curation_supplement"
                else {
                    "schema_version",
                    "contract",
                    "requested_spec_contract",
                    "request_id",
                    "models",
                    "policy",
                    "source_quotas",
                    "holdout_quotas",
                    "supplement_request",
                    "suite_seed",
                    "output_suite_contract",
                    "output_root",
                    "request_sha256",
                }
            )
            _require_exact_keys(
                child, expected_child_keys, f"{name} request"
            )
            child_payload = dict(child)
            child_identity = child_payload.pop("request_sha256", None)
            if (
                child.get("schema_version") != 1
                or child.get("contract") != contract
                or child.get("request_id") != request_id
                or child_identity != canonical_sha256(child_payload)
                or child_identity != child_binding["identity"]
                or child.get("models") != request["models"]
                or child.get("policy") != request["policy"]
            ):
                raise SuiteRegistryCorruption(f"{name} request identity is invalid")
        return request

    def _request_rotation_locked(
        self, state: RegistryState, now: datetime
    ) -> Optional[RegistryEvent]:
        eligibility = rotation_eligibility(state, now)
        if not eligibility.eligible:
            return None
        request_id = self._request_identity(state, eligibility)
        existing = state.requests.get(request_id)
        if existing is not None:
            for event in state.events:
                if (
                    event.event_type == "rotation.requested"
                    and event.payload["request_id"] == request_id
                ):
                    return event
            raise SuiteRegistryCorruption("request state has no event")
        _, request = self._publish_request_bundle(state, eligibility)
        request_path = self.requests_dir / request_id / "manifest.json"
        return self._append_event(
            "rotation.requested",
            {
                "request_id": request_id,
                "request_manifest": {
                    "path": str(request_path.resolve()),
                    "sha256": file_sha256(request_path),
                    "identity": request["request_sha256"],
                },
                "base_suite_id": state.active_suite_id,
                "champion_sha256": state.current_champion.sha256,
                "generation_id": state.current_champion.generation_id,
                "trigger_at_utc": eligibility.trigger_at_utc,
            },
            timestamp=now,
        )

    def request_rotation(self) -> Optional[RegistryEvent]:
        with self._locked():
            state = self.reconstruct()
            if state.active_suite_id is None:
                raise SuiteConflictError("registry is not bootstrapped")
            return self._request_rotation_locked(state, self._now())

    def register_suite(
        self, request_id: str, suite_manifest_path: Path
    ) -> RegistryEvent:
        with self._locked():
            state = self.reconstruct()
            identifier = _require_id(request_id, "request ID")
            request = state.requests.get(identifier)
            if request is None:
                raise SuiteConflictError(f"unknown rotation request: {identifier}")
            validated = validate_suite_manifest(
                suite_manifest_path,
                self.spec,
                expected_champion_sha256=request["champion_sha256"],
            )
            existing = state.registrations.get(validated.suite_id)
            if existing is not None:
                if existing["request_id"] == identifier:
                    for event in state.events:
                        if (
                            event.event_type == "suite.registered"
                            and event.payload["suite_id"] == validated.suite_id
                        ):
                            return event
                raise SuiteConflictError("suite is already registered to another request")
            version = self._publish_suite(validated)
            self.failure_hook("suite-version-published")
            return self._append_event(
                "suite.registered",
                {
                    "request_id": identifier,
                    "suite_id": version.suite_id,
                    "version_sha256": version.version_sha256,
                    "manifest_sha256": version.manifest_sha256,
                    "manifest_identity": version.manifest_identity,
                    "curation_champion_sha256": version.champion_sha256,
                },
            )

    def _validate_continuity_binding(
        self,
        binding: Any,
        *,
        request_id: str,
        suite_id: str,
        base_suite_id: str,
        current_champion_sha256: str,
        previous_champion_sha256: Optional[str],
    ) -> Mapping[str, Any]:
        if previous_champion_sha256 is None:
            raise SuiteRegistryCorruption(
                "continuity replay requires a previous champion"
            )
        checked_binding = _require_exact_keys(
            binding, {"path", "sha256", "identity"}, "continuity manifest binding"
        )
        path = _absolute_file(
            Path(checked_binding["path"]),
            "continuity manifest",
            _require_sha256(checked_binding["sha256"], "continuity file hash"),
        )
        receipt = _load_canonical_object(path, "continuity manifest")
        _require_exact_keys(
            receipt,
            {
                "schema_version",
                "contract",
                "request_id",
                "candidate_suite_id",
                "base_suite_id",
                "policy_hash",
                "shadow_replays",
                "completed_at_utc",
                "manifest_sha256",
            },
            "continuity manifest",
        )
        payload = dict(receipt)
        identity = payload.pop("manifest_sha256")
        replays = _require_exact_keys(
            receipt["shadow_replays"],
            {"current_champion", "previous_champion"},
            "continuity shadow replays",
        )
        expected_models = {
            "current_champion": current_champion_sha256,
            "previous_champion": previous_champion_sha256,
        }
        if (
            receipt["schema_version"] != 1
            or receipt["contract"] != CONTINUITY_CONTRACT
            or receipt["request_id"] != request_id
            or receipt["candidate_suite_id"] != suite_id
            or receipt["base_suite_id"] != base_suite_id
            or receipt["policy_hash"] != self.spec.policy_identity
            or identity != canonical_sha256(payload)
            or identity != checked_binding["identity"]
        ):
            raise SuiteRegistryCorruption("continuity manifest identity is invalid")
        _parse_utc(receipt["completed_at_utc"], "continuity completion timestamp")
        for role, expected_hash in expected_models.items():
            replay = _require_exact_keys(
                replays[role],
                {
                    "role",
                    "model_sha256",
                    "suite_id",
                    "decision",
                    "evaluation_id",
                    "evidence",
                },
                f"{role} continuity replay",
            )
            evidence = _require_exact_keys(
                replay["evidence"], {"path", "sha256"}, f"{role} continuity evidence"
            )
            evidence_path = _absolute_file(
                Path(evidence["path"]),
                f"{role} continuity evidence",
                _require_sha256(evidence["sha256"], f"{role} evidence hash"),
            )
            if (
                replay["role"] != role
                or replay["model_sha256"] != expected_hash
                or replay["suite_id"] != suite_id
                or replay["decision"] != "PASS"
                or not _SAFE_ID_RE.fullmatch(replay["evaluation_id"] or "")
                or evidence_path != Path(evidence["path"])
            ):
                raise SuiteRegistryCorruption(
                    f"{role} continuity shadow replay did not PASS"
                )
        return receipt

    def record_continuity(
        self, request_id: str, suite_id: str, manifest_path: Path
    ) -> RegistryEvent:
        with self._locked():
            state = self.reconstruct()
            request_id = _require_id(request_id, "request ID")
            suite_id = _require_sha256(suite_id, "suite ID")
            registration = state.registrations.get(suite_id)
            if registration is None or registration["request_id"] != request_id:
                raise SuiteConflictError("suite is not registered to this request")
            existing = state.continuity.get(suite_id)
            source = _absolute_file(manifest_path, "continuity manifest")
            receipt = self._validate_continuity_binding(
                {
                    "path": str(source),
                    "sha256": file_sha256(source),
                    "identity": _load_canonical_object(
                        source, "continuity manifest"
                    ).get("manifest_sha256"),
                },
                request_id=request_id,
                suite_id=suite_id,
                base_suite_id=state.requests[request_id]["base_suite_id"],
                current_champion_sha256=state.current_champion.sha256,
                previous_champion_sha256=state.previous_champion_sha256,
            )
            if existing is not None:
                if existing["manifest"]["identity"] == receipt["manifest_sha256"]:
                    for event in state.events:
                        if (
                            event.event_type == "continuity.recorded"
                            and event.payload["suite_id"] == suite_id
                        ):
                            return event
                raise SuiteConflictError("suite continuity receipt conflicts")
            destination = self.continuity_dir / f"{file_sha256(source)}.json"
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise SuiteRegistryCorruption("continuity snapshot is contradictory")
            else:
                _atomic_create(destination, source.read_bytes(), mode=0o444)
            binding = {
                "path": str(destination.resolve()),
                "sha256": file_sha256(destination),
                "identity": receipt["manifest_sha256"],
            }
            return self._append_event(
                "continuity.recorded",
                {
                    "request_id": request_id,
                    "suite_id": suite_id,
                    "manifest": binding,
                    "current_champion_sha256": state.current_champion.sha256,
                    "previous_champion_sha256": state.previous_champion_sha256,
                },
            )

    def record_generation_boundary(
        self,
        boundary_id: str,
        *,
        generation_id: str,
        champion_sha256: str,
    ) -> RegistryEvent:
        with self._locked():
            return self._record_boundary_locked(
                self.reconstruct(),
                boundary_id=boundary_id,
                generation_id=generation_id,
                champion_sha256=champion_sha256,
            )

    def _record_boundary_locked(
        self,
        state: RegistryState,
        *,
        boundary_id: str,
        generation_id: str,
        champion_sha256: str,
    ) -> RegistryEvent:
        boundary_id = _require_id(boundary_id, "boundary ID")
        generation_id = _require_id(generation_id, "generation ID")
        champion_sha256 = _require_sha256(champion_sha256, "champion hash")
        existing = state.boundaries.get(boundary_id)
        if existing is not None:
            if (
                existing["generation_id"] == generation_id
                and existing["champion_sha256"] == champion_sha256
            ):
                for event in state.events:
                    if (
                        event.event_type == "generation.boundary"
                        and event.payload["boundary_id"] == boundary_id
                    ):
                        return event
            raise SuiteConflictError("generation boundary conflicts with retry")
        if (
            state.current_champion is None
            or state.current_champion.sha256 != champion_sha256
            or state.current_champion.generation_id != generation_id
        ):
            raise StaleChampionError("generation boundary names a stale champion")
        return self._append_event(
            "generation.boundary",
            {
                "boundary_id": boundary_id,
                "champion_sha256": champion_sha256,
                "generation_id": generation_id,
                "clean": True,
            },
        )

    def _activation_retry(
        self,
        state: RegistryState,
        *,
        request_id: str,
        suite_id: str,
        expected_active_suite_id: str,
        expected_champion_sha256: str,
        boundary_id: str,
    ) -> Optional[RegistryEvent]:
        if state.active_suite_id != suite_id:
            return None
        for event in reversed(state.events):
            if event.event_type != "suite.activated":
                continue
            payload = event.payload
            if payload["suite_id"] != suite_id:
                continue
            if (
                payload["request_id"] == request_id
                and payload["previous_suite_id"] == expected_active_suite_id
                and payload["expected_champion_sha256"] == expected_champion_sha256
                and payload["boundary_id"] == boundary_id
            ):
                self._repair_active_projection(state)
                return event
            raise SuiteConflictError("suite is active with different activation metadata")
        raise SuiteRegistryCorruption("active suite has no activation event")

    def _activate_locked(
        self,
        state: RegistryState,
        *,
        request_id: str,
        suite_id: str,
        expected_active_suite_id: str,
        expected_champion_sha256: str,
        boundary_id: str,
    ) -> RegistryEvent:
        retry = self._activation_retry(
            state,
            request_id=request_id,
            suite_id=suite_id,
            expected_active_suite_id=expected_active_suite_id,
            expected_champion_sha256=expected_champion_sha256,
            boundary_id=boundary_id,
        )
        if retry is not None:
            return retry
        if state.active_suite_id != expected_active_suite_id:
            raise StaleActiveSuiteError(
                f"expected active suite {expected_active_suite_id}, "
                f"current is {state.active_suite_id}"
            )
        if (
            state.current_champion is None
            or state.current_champion.sha256 != expected_champion_sha256
        ):
            raise StaleChampionError("champion changed before suite activation")
        registration = state.registrations.get(suite_id)
        receipt = state.continuity.get(suite_id)
        boundary = state.boundaries.get(boundary_id)
        request = state.requests.get(request_id)
        if (
            request is None
            or registration is None
            or registration["request_id"] != request_id
            or receipt is None
            or receipt["request_id"] != request_id
        ):
            raise ActivationBlockedError(
                "suite registration and continuity replay are incomplete"
            )
        if request["base_suite_id"] != expected_active_suite_id:
            raise StaleActiveSuiteError("rotation request names a stale base suite")
        if request["champion_sha256"] != expected_champion_sha256:
            raise StaleChampionError("rotation request names a stale champion")
        if boundary is None:
            raise ActivationBlockedError("clean generation boundary was not observed")
        if (
            boundary["champion_sha256"] != expected_champion_sha256
            or boundary["generation_id"] != state.current_champion.generation_id
            or boundary["_sequence"] <= receipt["_sequence"]
        ):
            raise ActivationBlockedError(
                "generation boundary is stale or predates continuity replay"
            )
        if state.pins:
            raise ActivationBlockedError(
                "conflicting in-flight evaluations remain pinned"
            )
        event = self._append_event(
            "suite.activated",
            {
                "request_id": request_id,
                "suite_id": suite_id,
                "previous_suite_id": expected_active_suite_id,
                "expected_champion_sha256": expected_champion_sha256,
                "generation_id": state.current_champion.generation_id,
                "boundary_id": boundary_id,
                "continuity_manifest_sha256": receipt["manifest"]["sha256"],
            },
        )
        self.failure_hook("activation-event-published")
        updated = self.reconstruct()
        self._repair_active_projection(updated)
        self.failure_hook("active-suite-published")
        return event

    def activate_suite(
        self,
        request_id: str,
        suite_id: str,
        *,
        expected_active_suite_id: str,
        expected_champion_sha256: str,
        boundary_id: str,
    ) -> RegistryEvent:
        with self._locked():
            return self._activate_locked(
                self.reconstruct(),
                request_id=_require_id(request_id, "request ID"),
                suite_id=_require_sha256(suite_id, "suite ID"),
                expected_active_suite_id=_require_sha256(
                    expected_active_suite_id, "expected active suite ID"
                ),
                expected_champion_sha256=_require_sha256(
                    expected_champion_sha256, "expected champion hash"
                ),
                boundary_id=_require_id(boundary_id, "boundary ID"),
            )

    def _active_projection(self, state: RegistryState) -> Optional[Dict[str, Any]]:
        if state.active_suite_id is None or state.current_champion is None:
            return None
        version = state.versions[state.active_suite_id]
        activation = next(
            event
            for event in reversed(state.events)
            if event.sequence == state.active_activation_sequence
        )
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": ACTIVE_SUITE_CONTRACT,
            "spec_sha256": self.spec.identity,
            "suite_id": version.suite_id,
            "version_sha256": version.version_sha256,
            "manifest_path": str(version.manifest_path),
            "manifest_sha256": version.manifest_sha256,
            "manifest_identity": version.manifest_identity,
            "activated_at_utc": state.active_suite_activated_at_utc,
            "activation_champion_sha256": activation.payload.get(
                "expected_champion_sha256",
                activation.payload.get("champion", {}).get("sha256"),
            ),
            "activation_generation_id": activation.payload.get(
                "generation_id"
            ),
            "event_sequence": activation.sequence,
            "event_sha256": activation.event_sha256,
        }
        value["record_sha256"] = canonical_sha256(value)
        return value

    def _projection_consistent(self, state: RegistryState) -> bool:
        expected = self._active_projection(state)
        if expected is None:
            return not os.path.lexists(os.fspath(self.active_path))
        if not os.path.lexists(os.fspath(self.active_path)):
            return False
        try:
            return _load_canonical_object(
                self.active_path, "active-suite projection"
            ) == expected
        except (OSError, ValueError):
            return False

    def _repair_active_projection(self, state: RegistryState) -> None:
        expected = self._active_projection(state)
        if expected is None:
            return
        if not self._projection_consistent(state):
            if os.path.lexists(os.fspath(self.active_path)) and (
                self.active_path.is_symlink() or not self.active_path.is_file()
            ):
                raise SuiteRegistryCorruption("active-suite projection path is unsafe")
            _publish_canonical(self.active_path, expected, replace=True)

    def _current_request(self, state: RegistryState) -> Optional[Tuple[str, Mapping[str, Any]]]:
        if state.current_champion is None or state.active_suite_id is None:
            return None
        matches = [
            (request_id, request)
            for request_id, request in state.requests.items()
            if request["base_suite_id"] == state.active_suite_id
            and request["champion_sha256"] == state.current_champion.sha256
        ]
        return max(matches, key=lambda item: item[1]["_sequence"]) if matches else None

    def _status_value(self, state: RegistryState, now: datetime) -> Dict[str, Any]:
        eligibility = rotation_eligibility(state, now)
        current_request = self._current_request(state)
        phase = "uninitialized"
        next_action = "bootstrap"
        candidate_suite = None
        if state.active_suite_id is not None:
            phase = "active"
            next_action = "wait-for-cadence"
            if eligibility.eligible:
                phase = "rotation-due"
                next_action = "publish-curation-requests"
            if current_request is not None:
                request_id, request = current_request
                phase = "curation-requested"
                next_action = "register-suite"
                registrations = [
                    (suite_id, registration)
                    for suite_id, registration in state.registrations.items()
                    if registration["request_id"] == request_id
                ]
                if registrations:
                    suite_id, registration = max(
                        registrations, key=lambda item: item[1]["_sequence"]
                    )
                    candidate_suite = suite_id
                    phase = "continuity-pending"
                    next_action = "record-continuity"
                    receipt = state.continuity.get(suite_id)
                    if receipt is not None:
                        phase = "awaiting-generation-boundary"
                        next_action = "record-clean-generation-boundary"
                        boundaries = [
                            (boundary_id, boundary)
                            for boundary_id, boundary in state.boundaries.items()
                            if boundary["champion_sha256"]
                            == state.current_champion.sha256
                            and boundary["_sequence"] > receipt["_sequence"]
                        ]
                        if boundaries:
                            phase = (
                                "blocked-in-flight"
                                if state.pins
                                else "ready-to-activate"
                            )
                            next_action = (
                                "finish-in-flight-evaluations"
                                if state.pins
                                else "activate-suite"
                            )
                if request["champion_sha256"] != state.current_champion.sha256:
                    phase = "stale-request"
                    next_action = "publish-curation-requests"
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": STATUS_CONTRACT,
            "generated_at_utc": _format_utc(now),
            "spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "state": phase,
            "next_action": next_action,
            "active_suite": (
                None
                if state.active_suite_id is None
                else {
                    "suite_id": state.active_suite_id,
                    "version_sha256": state.versions[
                        state.active_suite_id
                    ].version_sha256,
                    "activated_at_utc": state.active_suite_activated_at_utc,
                }
            ),
            "current_champion": (
                None
                if state.current_champion is None
                else {
                    "sha256": state.current_champion.sha256,
                    "generation_id": state.current_champion.generation_id,
                    "previous_sha256": state.previous_champion_sha256,
                }
            ),
            "cadence": eligibility.to_dict(),
            "current_request_id": current_request[0] if current_request else None,
            "candidate_suite_id": candidate_suite,
            "in_flight_evaluations": [
                {
                    key: pin[key]
                    for key in (
                        "evaluation_id",
                        "suite_id",
                        "champion_sha256",
                        "generation_id",
                    )
                }
                for _, pin in sorted(state.pins.items())
            ],
            "retained_suites": [
                {
                    "suite_id": suite_id,
                    "version_sha256": version.version_sha256,
                    "active": suite_id == state.active_suite_id,
                    "immutable": True,
                    "retained": True,
                }
                for suite_id, version in sorted(state.versions.items())
            ],
            "active_projection_consistent": self._projection_consistent(state),
            "last_event_sequence": state.last_sequence,
            "last_event_sha256": state.last_event_sha256,
        }
        value["status_sha256"] = canonical_sha256(value)
        return value

    def status(self, *, now: Optional[datetime] = None) -> Mapping[str, Any]:
        state = self.reconstruct()
        return self._status_value(state, self._now() if now is None else now)

    def once(
        self,
        *,
        boundary_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        champion_sha256: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Run one deterministic request/reconcile/activation pass."""

        supplied = (boundary_id, generation_id, champion_sha256)
        if any(item is not None for item in supplied) and not all(
            item is not None for item in supplied
        ):
            raise ValueError(
                "boundary_id, generation_id, and champion_sha256 are all required together"
            )
        with self._locked():
            now = self._now()
            state = self.reconstruct()
            self._repair_active_projection(state)
            if boundary_id is not None:
                self._record_boundary_locked(
                    state,
                    boundary_id=boundary_id,
                    generation_id=generation_id,
                    champion_sha256=champion_sha256,
                )
                state = self.reconstruct()
            if state.active_suite_id is not None:
                self._request_rotation_locked(state, now)
                state = self.reconstruct()

            # A registry reconciliation may request and validate a replacement
            # suite, but it must never switch the data-plane pointer by itself.
            # RuntimeConfig binds concrete suite and schedule hashes, so pointer
            # activation belongs to the privileged fenced deployment handshake
            # that rebuilds and atomically applies the matching runtime.
            self._repair_active_projection(state)
            status = self._status_value(state, now)
            _publish_canonical(self.status_path, status, replace=True)
            return status


EvaluationSuiteRegistry = SuiteRotationRegistry


def publish_continuity_manifest(
    path: Path,
    *,
    request_id: str,
    candidate_suite_id: str,
    base_suite_id: str,
    policy_hash: str,
    current_champion_sha256: str,
    previous_champion_sha256: str,
    current_evidence_path: Path,
    previous_evidence_path: Path,
    completed_at_utc: Optional[str] = None,
) -> Mapping[str, Any]:
    """Publish a strict two-model continuity PASS receipt for later registration."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = (Path.cwd() / destination).resolve()
    current_evidence = _absolute_file(
        current_evidence_path, "current champion continuity evidence"
    )
    previous_evidence = _absolute_file(
        previous_evidence_path, "previous champion continuity evidence"
    )
    completed = (
        _format_utc(datetime.now(timezone.utc))
        if completed_at_utc is None
        else _format_utc(_parse_utc(completed_at_utc, "completed_at_utc"))
    )
    suite_id = _require_sha256(candidate_suite_id, "candidate suite ID")
    request_id = _require_id(request_id, "request ID")
    value: Dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTINUITY_CONTRACT,
        "request_id": request_id,
        "candidate_suite_id": suite_id,
        "base_suite_id": _require_sha256(base_suite_id, "base suite ID"),
        "policy_hash": _require_sha256(policy_hash, "policy hash"),
        "shadow_replays": {
            "current_champion": {
                "role": "current_champion",
                "model_sha256": _require_sha256(
                    current_champion_sha256, "current champion hash"
                ),
                "suite_id": suite_id,
                "decision": "PASS",
                "evaluation_id": f"continuity-current-{suite_id[:16]}",
                "evidence": {
                    "path": str(current_evidence),
                    "sha256": file_sha256(current_evidence),
                },
            },
            "previous_champion": {
                "role": "previous_champion",
                "model_sha256": _require_sha256(
                    previous_champion_sha256, "previous champion hash"
                ),
                "suite_id": suite_id,
                "decision": "PASS",
                "evaluation_id": f"continuity-previous-{suite_id[:16]}",
                "evidence": {
                    "path": str(previous_evidence),
                    "sha256": file_sha256(previous_evidence),
                },
            },
        },
        "completed_at_utc": completed,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    _mkdir_durable(destination.parent)
    if destination.exists():
        if _load_canonical_object(destination, "continuity manifest") != value:
            raise SuiteConflictError("continuity manifest conflicts with retry")
    else:
        _publish_canonical(destination, value, replace=False)
    return value


def status(spec_path: Path, *, now: Optional[datetime] = None) -> Mapping[str, Any]:
    return SuiteRotationRegistry(spec_path).status(now=now)


def once(
    spec_path: Path,
    *,
    boundary_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    champion_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    return SuiteRotationRegistry(spec_path).once(
        boundary_id=boundary_id,
        generation_id=generation_id,
        champion_sha256=champion_sha256,
    )


def watch(
    spec_path: Path,
    *,
    interval_seconds: float = 30.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or interval_seconds <= 0
    ):
        raise ValueError("watch interval must be positive")
    registry = SuiteRotationRegistry(spec_path)
    while True:
        registry.once()
        sleeper(float(interval_seconds))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile the immutable evaluation-suite rotation registry."
    )
    parser.add_argument("mode", choices=("status", "once", "watch"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--boundary-id")
    parser.add_argument("--generation-id")
    parser.add_argument("--champion-sha256")
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        registry = SuiteRotationRegistry(args.spec)
        if args.mode in {"status", "watch"} and any(
            value is not None
            for value in (
                args.boundary_id,
                args.generation_id,
                args.champion_sha256,
            )
        ):
            raise ValueError(
                "generation-boundary options are valid only for once"
            )
        if args.mode == "status":
            result = registry.status()
        elif args.mode == "once":
            result = registry.once(
                boundary_id=args.boundary_id,
                generation_id=args.generation_id,
                champion_sha256=args.champion_sha256,
            )
        else:
            watch(args.spec, interval_seconds=args.interval)
            return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, SuiteRotationError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
