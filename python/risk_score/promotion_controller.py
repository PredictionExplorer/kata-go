#!/usr/bin/env python3
"""Crash-recoverable orchestration for risk-seeking checkpoint promotion.

The module is intentionally independent of GPU, gate, exporter, and process
supervision implementations.  Those services are supplied as callables.  The
only concrete integrations are :mod:`promotion_state` and the stable
``evaluation_runner.build_evaluation_matrix`` planning function.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from risk_score.promotion_state import (
    CandidateState,
    ChampionConflictError,
    ControllerLock,
    EventProvenance,
    EventRegistry,
    GenerationState,
    RegistryCorruptionError,
    StaleChampionError,
    Transition,
    atomic_write_bytes,
    bootstrap_champion,
    canonical_json_bytes,
    canonical_sha256,
    compare_and_swap_champion,
    fsync_directory,
    load_champion,
    load_finalized_pass_report,
    sha256_bytes,
    sha256_file,
    utc_timestamp,
)


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COUNTER_RE = re.compile(r"(?:^|-)s([0-9]+)-d([0-9]+)(?:-|$)")
_IGNORED_SUFFIXES = (".tmp", ".partial", ".exported")
_CANDIDATE_FILES = {"model.bin.gz", "model.ckpt"}
_OPTIONAL_CANDIDATE_FILES = {"manifest.json", "exporter_manifest.json"}
_HARDENED_EXPORT_CONTRACT = "katago-hardened-candidate-publication-v1"

PROMOTION_FAILURE_STEPS = (
    "promotion-intent-written",
    "promotion-intent-event",
    "promotion-pins-written",
    "promotion-candidate-accepted",
    "promotion-generation-leaf",
    "promotion-workers-staged",
    "promotion-canary-event",
    "promotion-canary-admitted",
    "promotion-rollout-event",
    "promotion-intermediate-passed",
    "promotion-all-workers-acknowledged",
    "promotion-champion-cas",
    "promotion-generation-data-admitted",
    "promotion-active-event",
)


class ControllerError(RuntimeError):
    """Base error for controller configuration and execution."""


class ConfigurationError(ControllerError, ValueError):
    """Runtime JSON is unsafe, incomplete, or ambiguous."""


class SafetyHalt(ControllerError):
    """A contradiction requires operator reconciliation before mutation."""


class IncompleteCandidate(ControllerError):
    """An export is not yet complete and should be ignored for this scan."""


class InsufficientDiskError(SafetyHalt):
    """Configured free-space reserve would be violated."""


@dataclass(frozen=True)
class ControllerConfig:
    """Security and policy controls loaded from strict runtime JSON."""

    mutation_enabled: bool
    actor: str
    controller_hash: str
    source_hash: str
    original_hash: str
    policy_hash: str
    powered_config_hash: str
    standard_config_hash: str
    discovery_schedule_hash: str
    confirmation_schedule_hash: str
    audit_schedule_hash: str
    lead40_schedule_hash: str
    lead80_schedule_hash: str
    standard_confirmation_schedule_hash: str
    selfplay_config_hash: str
    gpu_lease_config_hash: str
    suite_manifest_hash: str
    expected_gpu_uuid: str
    poll_interval_seconds: float
    max_active_queue: int
    anchor_interval_samples: int
    min_free_bytes: int
    worker_count: int
    canary_worker_count: int
    intermediate_worker_count: int
    worker_threads: int
    anomaly_names: Tuple[str, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    """Absolute live paths, argv templates, and controller policy."""

    controller: ControllerConfig
    promotion_root: Path
    lock_path: Path
    champion_path: Path
    candidate_inbox: Path
    candidate_quarantine: Path
    candidate_superseded: Path
    candidate_rejected: Path
    candidate_deduplicated: Path
    accepted_models: Path
    admitted_selfplay: Path
    rollout_quarantine: Path
    rollback_quarantine: Path
    trainer_checkpoint: Path
    evaluations: Path
    reports: Path
    suites: Path
    policy_path: Path
    powered_config_path: Path
    standard_config_path: Path
    discovery_schedule_path: Path
    confirmation_schedule_path: Path
    audit_schedule_path: Path
    lead40_schedule_path: Path
    lead80_schedule_path: Path
    standard_confirmation_schedule_path: Path
    selfplay_config_path: Path
    gpu_lease_config_path: Path
    data_watermark_path: Path
    shuffle_watermark_path: Path
    worker_ack_inbox: Path
    rollout_report_inbox: Path
    original_model_path: Path
    commands: Mapping[str, Tuple[str, ...]]
    frozen_policy: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "RuntimeConfig":
        """Load strict JSON without creating or modifying any live path."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot load runtime config {path}: {exc}") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeConfig":
        root = _strict_object(
            value,
            "runtime config",
            {
                "schemaVersion",
                "mutationEnabled",
                "actor",
                "hashes",
                "paths",
                "commands",
                "polling",
                "limits",
                "backlog",
                "rollout",
            },
        )
        if root["schemaVersion"] != 1 or type(root["schemaVersion"]) is not int:
            raise ConfigurationError("schemaVersion must be integer 1")
        if type(root["mutationEnabled"]) is not bool:
            raise ConfigurationError("mutationEnabled must be boolean")
        actor = _nonempty(root["actor"], "actor")
        hashes = _strict_object(
            root["hashes"],
            "hashes",
            {
                "controller",
                "source",
                "original",
                "policy",
                "poweredConfig",
                "standardConfig",
                "discoveryOrdinarySchedule",
                "confirmationOrdinarySchedule",
                "auditSchedule",
                "lead40Schedule",
                "lead80Schedule",
                "standardConfirmationSchedule",
                "selfplayConfig",
                "gpuLeaseConfig",
                "suiteManifest",
            },
        )
        for key, digest in hashes.items():
            _hash(digest, f"hashes.{key}")

        path_keys = {
            "promotionRoot",
            "controllerLock",
            "champion",
            "candidateInbox",
            "candidateQuarantine",
            "candidateSuperseded",
            "candidateRejected",
            "candidateDeduplicated",
            "acceptedModels",
            "admittedSelfplay",
            "rolloutQuarantine",
            "rollbackQuarantine",
            "trainerCheckpoint",
            "evaluations",
            "reports",
            "suites",
            "policy",
            "poweredConfig",
            "standardConfig",
            "discoveryOrdinarySchedule",
            "confirmationOrdinarySchedule",
            "auditSchedule",
            "lead40Schedule",
            "lead80Schedule",
            "standardConfirmationSchedule",
            "selfplayConfig",
            "gpuLeaseConfig",
            "dataWatermark",
            "shuffleWatermark",
            "workerAckInbox",
            "rolloutReportInbox",
            "originalModel",
        }
        paths = _strict_object(root["paths"], "paths", path_keys)
        normalized_paths = {
            key: _absolute_path(item, f"paths.{key}") for key, item in paths.items()
        }
        commands = _strict_object(
            root["commands"],
            "commands",
            {"trainer", "evaluator", "selfplay", "drain", "rollback"},
        )
        command_values = {
            key: _argv(value, f"commands.{key}") for key, value in commands.items()
        }
        polling = _strict_object(
            root["polling"], "polling", {"intervalSeconds"}
        )
        interval = _positive_number(
            polling["intervalSeconds"], "polling.intervalSeconds"
        )
        limits = _strict_object(
            root["limits"],
            "limits",
            {"maxActiveQueue", "minFreeBytes"},
        )
        max_queue = _positive_int(limits["maxActiveQueue"], "limits.maxActiveQueue")
        min_free = _nonnegative_int(limits["minFreeBytes"], "limits.minFreeBytes")
        backlog = _strict_object(
            root["backlog"],
            "backlog",
            {"anchorIntervalSamples", "anomalyNames"},
        )
        anchor_interval = _positive_int(
            backlog["anchorIntervalSamples"], "backlog.anchorIntervalSamples"
        )
        anomaly_value = backlog["anomalyNames"]
        if not isinstance(anomaly_value, list):
            raise ConfigurationError("backlog.anomalyNames must be an array")
        anomaly_names = tuple(
            _nonempty(name, "backlog.anomalyNames item") for name in anomaly_value
        )
        if len(set(anomaly_names)) != len(anomaly_names):
            raise ConfigurationError("backlog.anomalyNames contains duplicates")
        rollout = _strict_object(
            root["rollout"],
            "rollout",
            {
                "workerCount",
                "canaryWorkerCount",
                "intermediateWorkerCount",
                "threadsPerWorker",
            },
        )
        worker_count = _positive_int(
            rollout["workerCount"], "rollout.workerCount"
        )
        canary_count = _positive_int(
            rollout["canaryWorkerCount"], "rollout.canaryWorkerCount"
        )
        intermediate_count = _positive_int(
            rollout["intermediateWorkerCount"],
            "rollout.intermediateWorkerCount",
        )
        worker_threads = _positive_int(
            rollout["threadsPerWorker"], "rollout.threadsPerWorker"
        )
        if (
            worker_count != 7
            or canary_count != 1
            or intermediate_count != 3
            or worker_threads != 100
        ):
            raise ConfigurationError(
                "rollout topology must be exactly 7 workers, 1 canary, "
                "3 intermediate, and 100 threads per worker"
            )
        independent_schedule_keys = (
            "discoveryOrdinarySchedule",
            "confirmationOrdinarySchedule",
            "auditSchedule",
            "lead40Schedule",
            "lead80Schedule",
        )
        schedule_hashes = [hashes[key] for key in independent_schedule_keys]
        if len(set(schedule_hashes)) != len(schedule_hashes):
            raise ConfigurationError(
                "discovery and confirmation schedule banks must be pairwise distinct"
            )
        if (
            hashes["standardConfirmationSchedule"]
            != hashes["confirmationOrdinarySchedule"]
        ):
            raise ConfigurationError(
                "standard confirmation must reuse the frozen confirmation bank"
            )
        try:
            policy_value = json.loads(
                normalized_paths["policy"].read_text(encoding="utf-8")
            )
            gpu_value = json.loads(
                normalized_paths["gpuLeaseConfig"].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot load frozen policy/GPU config: {exc}") from exc
        if not isinstance(policy_value, dict):
            raise ConfigurationError("frozen policy root must be an object")
        if canonical_sha256(policy_value) != hashes["policy"]:
            raise ConfigurationError("frozen policy canonical hash mismatch")
        try:
            expected_gpu_uuid = _nonempty(
                gpu_value["gpu"]["expectedUuid"], "gpuLeaseConfig.gpu.expectedUuid"
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(
                "GPU lease config is missing gpu.expectedUuid"
            ) from exc
        _validate_runtime_paths(normalized_paths)

        controller = ControllerConfig(
            mutation_enabled=root["mutationEnabled"],
            actor=actor,
            controller_hash=hashes["controller"],
            source_hash=hashes["source"],
            original_hash=hashes["original"],
            policy_hash=hashes["policy"],
            powered_config_hash=hashes["poweredConfig"],
            standard_config_hash=hashes["standardConfig"],
            discovery_schedule_hash=hashes["discoveryOrdinarySchedule"],
            confirmation_schedule_hash=hashes["confirmationOrdinarySchedule"],
            audit_schedule_hash=hashes["auditSchedule"],
            lead40_schedule_hash=hashes["lead40Schedule"],
            lead80_schedule_hash=hashes["lead80Schedule"],
            standard_confirmation_schedule_hash=hashes[
                "standardConfirmationSchedule"
            ],
            selfplay_config_hash=hashes["selfplayConfig"],
            gpu_lease_config_hash=hashes["gpuLeaseConfig"],
            suite_manifest_hash=hashes["suiteManifest"],
            expected_gpu_uuid=expected_gpu_uuid,
            poll_interval_seconds=interval,
            max_active_queue=max_queue,
            anchor_interval_samples=anchor_interval,
            min_free_bytes=min_free,
            worker_count=worker_count,
            canary_worker_count=canary_count,
            intermediate_worker_count=intermediate_count,
            worker_threads=worker_threads,
            anomaly_names=anomaly_names,
        )
        return cls(
            controller=controller,
            promotion_root=normalized_paths["promotionRoot"],
            lock_path=normalized_paths["controllerLock"],
            champion_path=normalized_paths["champion"],
            candidate_inbox=normalized_paths["candidateInbox"],
            candidate_quarantine=normalized_paths["candidateQuarantine"],
            candidate_superseded=normalized_paths["candidateSuperseded"],
            candidate_rejected=normalized_paths["candidateRejected"],
            candidate_deduplicated=normalized_paths["candidateDeduplicated"],
            accepted_models=normalized_paths["acceptedModels"],
            admitted_selfplay=normalized_paths["admittedSelfplay"],
            rollout_quarantine=normalized_paths["rolloutQuarantine"],
            rollback_quarantine=normalized_paths["rollbackQuarantine"],
            trainer_checkpoint=normalized_paths["trainerCheckpoint"],
            evaluations=normalized_paths["evaluations"],
            reports=normalized_paths["reports"],
            suites=normalized_paths["suites"],
            policy_path=normalized_paths["policy"],
            powered_config_path=normalized_paths["poweredConfig"],
            standard_config_path=normalized_paths["standardConfig"],
            discovery_schedule_path=normalized_paths["discoveryOrdinarySchedule"],
            confirmation_schedule_path=normalized_paths[
                "confirmationOrdinarySchedule"
            ],
            audit_schedule_path=normalized_paths["auditSchedule"],
            lead40_schedule_path=normalized_paths["lead40Schedule"],
            lead80_schedule_path=normalized_paths["lead80Schedule"],
            standard_confirmation_schedule_path=normalized_paths[
                "standardConfirmationSchedule"
            ],
            selfplay_config_path=normalized_paths["selfplayConfig"],
            gpu_lease_config_path=normalized_paths["gpuLeaseConfig"],
            data_watermark_path=normalized_paths["dataWatermark"],
            shuffle_watermark_path=normalized_paths["shuffleWatermark"],
            worker_ack_inbox=normalized_paths["workerAckInbox"],
            rollout_report_inbox=normalized_paths["rolloutReportInbox"],
            original_model_path=normalized_paths["originalModel"],
            commands=command_values,
            frozen_policy=policy_value,
        )


@dataclass(frozen=True)
class CandidateArtifact:
    """Validated immutable identity of one complete candidate export."""

    name: str
    path: Path
    model_hash: str
    checkpoint_hash: str
    directory_manifest_hash: str
    sample_count: int
    data_count: int
    size_bytes: int
    files: Tuple[Tuple[str, int, str], ...]


@dataclass(frozen=True)
class BacklogSelection:
    """Deterministic candidate selection and coalescing result."""

    selected: Tuple[CandidateArtifact, ...]
    superseded: Tuple[CandidateArtifact, ...]
    reasons: Mapping[str, Tuple[str, ...]]


@dataclass(frozen=True)
class EvaluationMatrixPlan:
    """Immutable matrix plus aggregate hashes used by lifecycle events."""

    specs: Tuple[Any, ...]
    evaluation_key: str
    config_hash: str
    schedule_hash: str
    policy_hash: str
    selfplay_config_hash: str
    topology: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluationKey": self.evaluation_key,
            "configHash": self.config_hash,
            "scheduleHash": self.schedule_hash,
            "policyHash": self.policy_hash,
            "selfplayConfigHash": self.selfplay_config_hash,
            "topology": self.topology,
            "specs": [spec.to_dict() for spec in self.specs],
        }


def _strict_object(value: Any, name: str, keys: Iterable[str]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise ConfigurationError(
            f"{name} keys differ; missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}"
        )
    return dict(value)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConfigurationError(f"{name} must be a nonempty string without NUL")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ConfigurationError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigurationError(f"{name} must be a nonnegative integer")
    return value


def _absolute_path(value: Any, name: str) -> Path:
    text = _nonempty(value, name)
    path = Path(text)
    if not path.is_absolute():
        raise ConfigurationError(f"{name} must be absolute")
    if any(part == ".." for part in path.parts):
        raise ConfigurationError(f"{name} may not contain '..'")
    return path


def _validate_runtime_paths(paths: Mapping[str, Path]) -> None:
    """Reject aliases, symlink ancestors, and controller paths outside its root."""

    normalized: Dict[str, str] = {}
    for name, path in paths.items():
        key = os.path.normcase(os.path.normpath(str(path)))
        prior = normalized.get(key)
        if prior is not None:
            raise ConfigurationError(f"paths.{name} aliases paths.{prior}")
        normalized[key] = name
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise ConfigurationError(
                    f"paths.{name} has symlink ancestor {current}"
                )
    promotion_root = paths["promotionRoot"]
    controlled = {
        "controllerLock",
        "champion",
        "candidateQuarantine",
        "candidateSuperseded",
        "candidateRejected",
        "candidateDeduplicated",
        "acceptedModels",
        "rolloutQuarantine",
        "rollbackQuarantine",
        "evaluations",
        "reports",
        "workerAckInbox",
        "rolloutReportInbox",
    }
    for name in controlled:
        try:
            paths[name].relative_to(promotion_root)
        except ValueError as exc:
            raise ConfigurationError(
                f"paths.{name} must be contained by promotionRoot"
            ) from exc


def _argv(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{name} must be an argv array")
    return tuple(_nonempty(item, f"{name} item") for item in value)


def parse_candidate_counters(name: str) -> Tuple[int, int]:
    """Parse the required ``-sN-dM`` training counters from a candidate name."""

    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError("candidate name must be a safe path component")
    matches = list(_COUNTER_RE.finditer(name))
    if len(matches) != 1:
        raise ValueError(f"candidate name must contain exactly one -sN-dM: {name}")
    return int(matches[0].group(1)), int(matches[0].group(2))


def _inspect_hardened_candidate(path: Path) -> CandidateArtifact:
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SafetyHalt(f"hardened manifest is not a regular file: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise SafetyHalt(f"invalid hardened manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SafetyHalt("hardened manifest root must be an object")
    expected_keys = {
        "schemaVersion",
        "exportContract",
        "requestFingerprintSha256",
        "modelProbePassed",
        "candidateName",
        "modelName",
        "sourceCheckpoint",
        "files",
    }
    if set(manifest) != expected_keys:
        raise SafetyHalt(
            "hardened manifest keys differ; "
            f"missing={sorted(expected_keys-set(manifest))}, "
            f"unknown={sorted(set(manifest)-expected_keys)}"
        )
    if manifest_bytes != canonical_json_bytes(manifest) + b"\n":
        raise SafetyHalt("hardened manifest bytes are not canonical")
    if (
        manifest["schemaVersion"] != 1
        or type(manifest["schemaVersion"]) is not int
        or manifest["exportContract"] != _HARDENED_EXPORT_CONTRACT
        or manifest["modelProbePassed"] is not True
        or manifest["candidateName"] != path.name
        or not isinstance(manifest["modelName"], str)
        or not manifest["modelName"]
    ):
        raise SafetyHalt("hardened manifest identity is invalid")
    _hash(manifest["requestFingerprintSha256"], "requestFingerprintSha256")
    source = manifest["sourceCheckpoint"]
    if not isinstance(source, dict) or set(source) != {"name", "sha256", "size"}:
        raise SafetyHalt("hardened sourceCheckpoint metadata is invalid")
    _nonempty(source["name"], "sourceCheckpoint.name")
    _hash(source["sha256"], "sourceCheckpoint.sha256")
    _nonnegative_int(source["size"], "sourceCheckpoint.size")
    files_value = manifest["files"]
    if not isinstance(files_value, list):
        raise SafetyHalt("hardened manifest files must be an array")
    files: List[Tuple[str, int, str]] = []
    seen = set()
    for entry in files_value:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise SafetyHalt("hardened manifest file entry is invalid")
        relative = _nonempty(entry["path"], "manifest file path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative in {".", "..", "manifest.json"}
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative_path.as_posix() != relative
        ):
            raise SafetyHalt(f"unsafe hardened manifest path: {relative!r}")
        if relative in seen:
            raise SafetyHalt(f"duplicate hardened manifest path: {relative}")
        seen.add(relative)
        size = _nonnegative_int(entry["size"], f"manifest {relative} size")
        expected_hash = _hash(entry["sha256"], f"manifest {relative} sha256")
        artifact = path / relative_path
        if artifact.is_symlink() or not artifact.is_file():
            raise SafetyHalt(f"manifested artifact is not a regular file: {artifact}")
        if artifact.stat().st_size != size or sha256_file(artifact) != expected_hash:
            raise SafetyHalt(f"manifested artifact hash/size mismatch: {artifact}")
        files.append((relative, size, expected_hash))
    if [name for name, _, _ in files] != sorted(seen):
        raise SafetyHalt("hardened manifest file entries must be sorted")
    if not _CANDIDATE_FILES.issubset(seen):
        raise IncompleteCandidate("hardened export omits required model/checkpoint")
    actual = []
    for root, directories, filenames in os.walk(path, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            child = root_path / directory
            if child.is_symlink():
                raise SafetyHalt(f"symlinked candidate directory: {child}")
        for filename in filenames:
            child = root_path / filename
            if child == manifest_path:
                continue
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SafetyHalt(f"nonregular candidate artifact: {child}")
            if filename.endswith(_IGNORED_SUFFIXES):
                raise IncompleteCandidate(
                    f"candidate contains temporary content: {child}"
                )
            actual.append(child.relative_to(path).as_posix())
    if sorted(actual) != [name for name, _, _ in files]:
        raise SafetyHalt("hardened export has unmanifested or missing files")
    sample_count, data_count = parse_candidate_counters(path.name)
    hashes = {name: digest for name, _, digest in files}
    return CandidateArtifact(
        name=path.name,
        path=path,
        model_hash=hashes["model.bin.gz"],
        checkpoint_hash=hashes["model.ckpt"],
        directory_manifest_hash=sha256_bytes(manifest_bytes),
        sample_count=sample_count,
        data_count=data_count,
        size_bytes=sum(size for _, size, _ in files),
        files=tuple(files),
    )


def inspect_candidate(path: Path) -> CandidateArtifact:
    """Validate a hardened export or a safe legacy flat candidate directory."""

    path = Path(path)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SafetyHalt(f"candidate is not a regular directory: {path}")
    if (path / "manifest.json").exists() or (path / "manifest.json").is_symlink():
        try:
            return _inspect_hardened_candidate(path)
        except ConfigurationError as exc:
            raise SafetyHalt(f"invalid hardened manifest: {exc}") from exc
    entries: Dict[str, os.DirEntry] = {}
    with os.scandir(path) as scan:
        for entry in scan:
            if entry.name in entries:
                raise SafetyHalt(f"duplicate candidate entry: {entry.name}")
            if entry.name.endswith(_IGNORED_SUFFIXES):
                raise IncompleteCandidate(f"candidate contains temporary content: {path}")
            info = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or not stat.S_ISREG(info.st_mode):
                raise SafetyHalt(f"candidate entry is not a regular file: {entry.path}")
            entries[entry.name] = entry
    names = set(entries)
    if not _CANDIDATE_FILES.issubset(names):
        raise IncompleteCandidate(f"candidate export is incomplete: {path}")
    unexpected = names - _CANDIDATE_FILES - _OPTIONAL_CANDIDATE_FILES
    if unexpected:
        raise SafetyHalt(f"candidate has unexpected files: {sorted(unexpected)}")
    sample_count, data_count = parse_candidate_counters(path.name)
    files: List[Tuple[str, int, str]] = []
    for name in sorted(names):
        item = path / name
        item_stat = item.stat()
        files.append((name, item_stat.st_size, sha256_file(item)))
    file_hashes = {name: digest for name, _, digest in files}
    exporter_manifest = path / "exporter_manifest.json"
    if exporter_manifest.exists():
        try:
            exporter_value = json.loads(exporter_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"invalid exporter manifest {exporter_manifest}: {exc}") from exc
        if not isinstance(exporter_value, dict):
            raise SafetyHalt("exporter manifest must be a JSON object")
        if exporter_value.get("complete") is False:
            raise IncompleteCandidate(f"exporter manifest is not complete: {path}")
        declared_model = exporter_value.get("model_sha256")
        declared_checkpoint = exporter_value.get("checkpoint_sha256")
        if declared_model is not None and declared_model != file_hashes["model.bin.gz"]:
            raise SafetyHalt("exporter manifest model hash contradicts candidate")
        if (
            declared_checkpoint is not None
            and declared_checkpoint != file_hashes["model.ckpt"]
        ):
            raise SafetyHalt("exporter manifest checkpoint hash contradicts candidate")
    manifest = {
        "schemaVersion": 1,
        "files": [
            {"name": name, "size": size, "sha256": digest}
            for name, size, digest in files
        ],
    }
    return CandidateArtifact(
        name=path.name,
        path=path,
        model_hash=file_hashes["model.bin.gz"],
        checkpoint_hash=file_hashes["model.ckpt"],
        directory_manifest_hash=canonical_sha256(manifest),
        sample_count=sample_count,
        data_count=data_count,
        size_bytes=sum(size for _, size, _ in files),
        files=tuple(files),
    )


def inventory_candidates(inbox: Path) -> Tuple[Tuple[CandidateArtifact, ...], Tuple[str, ...]]:
    """Read complete candidates without modifying the inbox."""

    inbox = Path(inbox)
    if not inbox.exists():
        return (), ()
    candidates: List[CandidateArtifact] = []
    ignored: List[str] = []
    with os.scandir(inbox) as scan:
        entries = sorted(scan, key=lambda item: item.name)
    for entry in entries:
        if entry.name.startswith(".") or entry.name.endswith(_IGNORED_SUFFIXES):
            ignored.append(entry.name)
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise SafetyHalt(f"unexpected non-directory candidate entry: {entry.path}")
        try:
            candidates.append(inspect_candidate(Path(entry.path)))
        except IncompleteCandidate:
            ignored.append(entry.name)
    return tuple(candidates), tuple(ignored)


def select_backlog(
    candidates: Sequence[CandidateArtifact],
    *,
    original_hash: str,
    anchor_interval_samples: int = 500_000,
    anomaly_names: Iterable[str] = (),
    max_active_queue: int = 4,
    evaluation_started_hashes: Iterable[str] = (),
) -> BacklogSelection:
    """Select original/edge/anomaly/approximately-spaced anchors deterministically."""

    if anchor_interval_samples <= 0 or max_active_queue <= 0:
        raise ValueError("anchor interval and max queue must be positive")
    ordered = sorted(
        candidates, key=lambda item: (item.sample_count, item.data_count, item.name)
    )
    if not ordered:
        return BacklogSelection((), (), {})
    by_hash: Dict[str, CandidateArtifact] = {}
    by_name: Dict[str, CandidateArtifact] = {}
    for item in ordered:
        prior_name = by_name.get(item.name)
        if prior_name is not None and prior_name.model_hash != item.model_hash:
            raise SafetyHalt(f"duplicate candidate name has different hash: {item.name}")
        by_name[item.name] = item
        by_hash.setdefault(item.model_hash, item)
    ordered = sorted(
        by_hash.values(),
        key=lambda item: (item.sample_count, item.data_count, item.name),
    )

    reasons: Dict[str, set] = {}

    def keep(item: CandidateArtifact, reason: str) -> None:
        reasons.setdefault(item.model_hash, set()).add(reason)

    keep(ordered[0], "earliest")
    keep(ordered[-1], "newest")
    for item in ordered:
        if item.model_hash == original_hash:
            keep(item, "original")
        if item.name in set(anomaly_names):
            keep(item, "anomaly")
    started = set(evaluation_started_hashes)
    for item in ordered:
        if item.model_hash in started:
            keep(item, "evaluation-started")

    bins: Dict[int, CandidateArtifact] = {}
    for item in ordered:
        bucket = int(round(item.sample_count / anchor_interval_samples))
        target = bucket * anchor_interval_samples
        existing = bins.get(bucket)
        if existing is None or (
            abs(item.sample_count - target),
            item.sample_count,
            item.name,
        ) < (
            abs(existing.sample_count - target),
            existing.sample_count,
            existing.name,
        ):
            bins[bucket] = item
    mandatory = set(reasons)
    if len(mandatory) > max_active_queue:
        raise SafetyHalt("mandatory/evaluating candidates exceed maxActiveQueue")
    for bucket in sorted(bins):
        item = bins[bucket]
        if item.model_hash in reasons:
            reasons[item.model_hash].add("sample-anchor")
        elif len(reasons) < max_active_queue:
            keep(item, "sample-anchor")
    selected_hashes = set(reasons)
    selected = tuple(item for item in ordered if item.model_hash in selected_hashes)
    superseded = tuple(item for item in ordered if item.model_hash not in selected_hashes)
    return BacklogSelection(
        selected,
        superseded,
        {key: tuple(sorted(value)) for key, value in sorted(reasons.items())},
    )


def _tree_manifest(path: Path) -> Tuple[str, int]:
    """Hash a regular-file-only tree without following symlinks."""

    path = Path(path)
    rows = []
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in sorted(directories):
            child = root_path / name
            if child.is_symlink():
                raise SafetyHalt(f"symlink in immutable tree: {child}")
        for name in sorted(files):
            child = root_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SafetyHalt(f"nonregular file in immutable tree: {child}")
            relative = child.relative_to(path).as_posix()
            digest = sha256_file(child)
            rows.append({"path": relative, "size": info.st_size, "sha256": digest})
            total += info.st_size
    return canonical_sha256({"schemaVersion": 1, "files": rows}), total


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical_json_bytes(value) + b"\n"
    path = Path(path)
    if path.exists():
        if path.is_symlink():
            raise SafetyHalt(f"immutable artifact is a symlink: {path}")
        if path.is_file() and path.read_bytes() == data:
            return
        raise SafetyHalt(f"immutable artifact conflicts: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise SafetyHalt(f"immutable artifact conflicts: {path}")
        temporary.unlink()
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _same_device(paths: Iterable[Path]) -> None:
    devices = {}
    for configured in paths:
        path = Path(configured)
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.exists():
            raise SafetyHalt(f"no existing ancestor for path: {path}")
        devices[str(path)] = existing.stat().st_dev
    if len(set(devices.values())) != 1:
        raise SafetyHalt(f"atomic transition paths cross filesystems: {devices}")


def _recoverable_rename(
    source: Path,
    destination: Path,
    *,
    expected_manifest_hash: str,
    inspector: Callable[[Path], str],
) -> None:
    """Complete an exact-hash rename or reject any contradictory destination."""

    source = Path(source)
    destination = Path(destination)
    if source == destination:
        if not destination.exists() or inspector(destination) != expected_manifest_hash:
            raise SafetyHalt(f"in-place rename identity contradicts: {destination}")
        return
    if destination.exists():
        actual = inspector(destination)
        if actual != expected_manifest_hash:
            raise SafetyHalt(f"destination hash contradiction: {destination}")
        if source.exists():
            source_actual = inspector(source)
            if source_actual != expected_manifest_hash:
                raise SafetyHalt(f"source hash contradiction: {source}")
            raise SafetyHalt(f"both rename source and destination exist: {source}")
        return
    if not source.exists():
        raise SafetyHalt(f"rename source and destination are both absent: {source}")
    if inspector(source) != expected_manifest_hash:
        raise SafetyHalt(f"rename source hash contradiction: {source}")
    _same_device((source.parent, destination.parent))
    os.rename(source, destination)
    fsync_directory(source.parent)
    if source.parent != destination.parent:
        fsync_directory(destination.parent)


class PromotionController:
    """Single-writer controller with injectable evaluation and process seams."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        *,
        automatic: bool = False,
        evaluation_executor: Optional[Callable[[EvaluationMatrixPlan, CandidateArtifact], Any]] = None,
        gate_evaluator: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        command_executor: Optional[Callable[[Sequence[str]], Any]] = None,
        gpu_lease_factory: Optional[
            Callable[[Path, EvaluationMatrixPlan, CandidateArtifact], Any]
        ] = None,
        process_identity_verifier: Optional[Callable[[Mapping[str, Any]], bool]] = None,
        failure_hook: Optional[Callable[[str], None]] = None,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        held_controller_lock: Optional[ControllerLock] = None,
    ):
        self.runtime = runtime
        self.automatic = bool(automatic and runtime.controller.mutation_enabled)
        self.evaluation_executor = evaluation_executor
        self.gate_evaluator = gate_evaluator
        self.command_executor = command_executor
        self.gpu_lease_factory = gpu_lease_factory
        self.process_identity_verifier = process_identity_verifier
        self.failure_hook = failure_hook or (lambda _step: None)
        self.disk_usage = disk_usage
        self.registry = EventRegistry(runtime.promotion_root)
        if held_controller_lock is not None:
            if (
                held_controller_lock.path != runtime.lock_path
                or not held_controller_lock.acquired
            ):
                raise ConfigurationError(
                    "held_controller_lock must already own the configured lock path"
                )
        self.held_controller_lock = held_controller_lock

    def _validate_gpu_handoff(self, proof: Any) -> Mapping[str, Any]:
        if not isinstance(proof, Mapping):
            raise SafetyHalt("GPU handoff produced no immutable proof")
        required = {
            "lease_id",
            "expected_gpu_uuid",
            "handoff_checkpoint_hash",
            "clean_observations",
            "trainer_restored",
            "restored_trainer_identity",
        }
        if not required.issubset(proof):
            raise SafetyHalt("GPU handoff proof is incomplete")
        _nonempty(proof["lease_id"], "GPU lease ID")
        if proof["expected_gpu_uuid"] != self.runtime.controller.expected_gpu_uuid:
            raise SafetyHalt("GPU handoff UUID contradicts runtime configuration")
        checkpoint_hash = _hash(
            proof["handoff_checkpoint_hash"], "GPU handoff checkpoint hash"
        )
        if sha256_file(self.runtime.trainer_checkpoint) != checkpoint_hash:
            raise SafetyHalt("GPU handoff checkpoint no longer matches trainer state")
        observations = proof["clean_observations"]
        if (
            not isinstance(observations, list)
            or len(observations) < 2
            or any(
                not isinstance(item, Mapping)
                or item.get("gpu_uuid") != self.runtime.controller.expected_gpu_uuid
                or item.get("processes") != []
                for item in observations
            )
        ):
            raise SafetyHalt("GPU handoff lacks repeated clean observations")
        if proof["trainer_restored"] is not True:
            raise SafetyHalt("GPU handoff did not successfully restore trainer")
        identity = proof["restored_trainer_identity"]
        if not isinstance(identity, Mapping) or not identity:
            raise SafetyHalt("GPU handoff lacks restored trainer process identity")
        return dict(proof)

    @contextlib.contextmanager
    def _exclusive_gpu_handoff(
        self, plan: EvaluationMatrixPlan, candidate: CandidateArtifact
    ) -> Iterable[Mapping[str, Any]]:
        if self.gpu_lease_factory is None:
            raise SafetyHalt("automatic evaluation requires exclusive GPU handoff")
        context = self.gpu_lease_factory(
            self.runtime.gpu_lease_config_path, plan, candidate
        )
        proof: Any = None
        with context as value:
            proof = value
            yield proof
        self._validate_gpu_handoff(proof)

    @property
    def recommendation_only(self) -> bool:
        return not self.automatic

    def _checkpoint(self, step: str) -> None:
        self.failure_hook(step)

    @contextlib.contextmanager
    def _writer_lock(self) -> Iterable[None]:
        if self.held_controller_lock is not None:
            yield
            return
        with ControllerLock(
            self.runtime.lock_path, owner=self.runtime.controller.actor
        ):
            yield

    def _provenance(self, config_hash: str, schedule_hash: str) -> EventProvenance:
        config = self.runtime.controller
        return EventProvenance(
            controller_hash=config.controller_hash,
            source_hash=config.source_hash,
            original_hash=config.original_hash,
            config_hash=config_hash,
            schedule_hash=schedule_hash,
            policy_hash=config.policy_hash,
        )

    def validate_static_inputs(self) -> None:
        """Verify every configured immutable policy/model/config/schedule hash."""

        byte_checks = (
            (
                self.runtime.original_model_path,
                self.runtime.controller.original_hash,
                "original model",
            ),
            (
                self.runtime.powered_config_path,
                self.runtime.controller.powered_config_hash,
                "powered config",
            ),
            (
                self.runtime.standard_config_path,
                self.runtime.controller.standard_config_hash,
                "standard config",
            ),
            (
                self.runtime.discovery_schedule_path,
                self.runtime.controller.discovery_schedule_hash,
                "discovery ordinary schedule",
            ),
            (
                self.runtime.confirmation_schedule_path,
                self.runtime.controller.confirmation_schedule_hash,
                "confirmation ordinary schedule",
            ),
            (
                self.runtime.audit_schedule_path,
                self.runtime.controller.audit_schedule_hash,
                "audit schedule",
            ),
            (
                self.runtime.lead40_schedule_path,
                self.runtime.controller.lead40_schedule_hash,
                "Lead-40 schedule",
            ),
            (
                self.runtime.lead80_schedule_path,
                self.runtime.controller.lead80_schedule_hash,
                "Lead-80 schedule",
            ),
            (
                self.runtime.standard_confirmation_schedule_path,
                self.runtime.controller.standard_confirmation_schedule_hash,
                "standard confirmation schedule",
            ),
            (
                self.runtime.selfplay_config_path,
                self.runtime.controller.selfplay_config_hash,
                "self-play config",
            ),
            (
                self.runtime.gpu_lease_config_path,
                self.runtime.controller.gpu_lease_config_hash,
                "GPU lease config",
            ),
            (
                self.runtime.suites / "manifest.json",
                self.runtime.controller.suite_manifest_hash,
                "evaluation suite manifest",
            ),
        )
        for path, expected, role in byte_checks:
            if path.is_symlink() or not path.is_file():
                raise SafetyHalt(f"{role} is not a regular file: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise SafetyHalt(
                    f"{role} hash mismatch: expected {expected}, found {actual}"
                )
        policy_path = self.runtime.policy_path
        if policy_path.is_symlink() or not policy_path.is_file():
            raise SafetyHalt(f"policy is not a regular file: {policy_path}")
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"policy is not valid JSON: {exc}") from exc
        if not isinstance(policy, dict):
            raise SafetyHalt("policy JSON root must be an object")
        actual_policy_hash = canonical_sha256(policy)
        if actual_policy_hash != self.runtime.controller.policy_hash:
            raise SafetyHalt(
                "policy canonical hash mismatch: expected "
                f"{self.runtime.controller.policy_hash}, found {actual_policy_hash}"
            )
        rollout = policy.get("rollout")
        expected_rollout = {
            "worker_count": 7,
            "canary_workers": 1,
            "intermediate_workers": 3,
            "full_workers": 7,
            "games_per_worker_initial_threads": 100,
        }
        if not isinstance(rollout, dict) or any(
            rollout.get(key) != value for key, value in expected_rollout.items()
        ):
            raise SafetyHalt("frozen policy rollout topology is not production-safe")

    def ensure_layout(self) -> None:
        """Create controller-owned layout; caller must hold the controller lock."""

        directories = (
            self.runtime.promotion_root,
            self.runtime.promotion_root / "events",
            self.runtime.promotion_root / "candidates" / "claimed",
            self.runtime.promotion_root / "candidates" / "claim-intents",
            self.runtime.promotion_root / "candidates" / "lifecycle-intents",
            self.runtime.promotion_root / "candidates" / "dedup-intents",
            self.runtime.candidate_quarantine,
            self.runtime.candidate_superseded,
            self.runtime.candidate_rejected,
            self.runtime.candidate_deduplicated,
            self.runtime.accepted_models,
            self.runtime.admitted_selfplay,
            self.runtime.rollout_quarantine,
            self.runtime.rollback_quarantine,
            self.runtime.evaluations,
            self.runtime.reports,
            self.runtime.worker_ack_inbox,
            self.runtime.rollout_report_inbox,
            self.runtime.promotion_root / "transactions",
        )
        for directory in directories:
            if directory.exists() and not directory.is_dir():
                raise SafetyHalt(f"layout path is not a directory: {directory}")
            if not directory.exists():
                directory.mkdir(parents=True)
                fsync_directory(directory.parent)
        _same_device(
            (
                self.runtime.candidate_inbox,
                self.runtime.promotion_root / "candidates" / "claimed",
                self.runtime.accepted_models,
                self.runtime.rollout_quarantine,
                self.runtime.admitted_selfplay,
                self.runtime.rollback_quarantine,
            )
        )

    def _require_disk(self, additional_bytes: int = 0) -> None:
        usage = self.disk_usage(self.runtime.promotion_root)
        required = self.runtime.controller.min_free_bytes + max(0, additional_bytes)
        if usage.free < required:
            raise InsufficientDiskError(
                f"free bytes {usage.free} below required reserve {required}"
            )

    def build_evaluation_plan(
        self,
        candidate_hash: str,
        champion_hash: str,
        *,
        suite: str,
        stage: str,
        look: str,
        topology: str,
    ) -> EvaluationMatrixPlan:
        """Build a content-addressed matrix without touching the filesystem."""

        self.validate_static_inputs()
        required_topology = "7-workers-100-threads"
        if topology != required_topology:
            raise SafetyHalt(
                f"evaluation topology must be {required_topology}, found {topology}"
            )
        from risk_score.evaluation_runner import build_evaluation_matrix

        config = self.runtime.controller
        manifest_path = self.runtime.suites / "manifest.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"cannot load frozen suite manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise SafetyHalt("frozen suite manifest root must be an object")
        if sha256_bytes(manifest_bytes) != config.suite_manifest_hash:
            raise SafetyHalt("frozen suite manifest byte hash changed")
        if (
            manifest.get("policy_hash") != config.policy_hash
            or manifest.get("source_revision")
            != self.runtime.frozen_policy["frozen_plan"]["source_revision"]
        ):
            raise SafetyHalt("frozen suite manifest policy/source binding changed")
        banks = {
            bank.get("name"): bank
            for bank in manifest.get("banks", [])
            if isinstance(bank, dict) and isinstance(bank.get("name"), str)
        }

        def bank_binding(name: str, expected_schedule_hash: str) -> Tuple[str, str]:
            bank = banks.get(name)
            if not isinstance(bank, dict):
                raise SafetyHalt(f"frozen suite manifest has no {name!r} bank")
            positions = bank.get("positions")
            schedule_manifest = bank.get("schedule")
            if not isinstance(positions, dict) or not isinstance(
                schedule_manifest, dict
            ):
                raise SafetyHalt(f"frozen suite bank {name!r} is incomplete")
            if schedule_manifest.get("sha256") != expected_schedule_hash:
                raise SafetyHalt(
                    f"runtime schedule hash contradicts frozen {name!r} bank"
                )
            bank_hash = positions.get("sha256")
            schedule_id = schedule_manifest.get("scheduleId")
            _hash(bank_hash, f"suite bank {name} hash")
            _nonempty(schedule_id, f"suite bank {name} schedule ID")
            return bank_hash, schedule_id

        confirmation = stage in {"confirmation", "stage-3"}
        ordinary_bank_name = (
            "confirmation"
            if confirmation
            else "audit"
            if stage == "integrity"
            else "discovery"
        )
        ordinary_schedule_hash = (
            config.confirmation_schedule_hash
            if confirmation
            else config.audit_schedule_hash
            if stage == "integrity"
            else config.discovery_schedule_hash
        )
        ordinary_bank_hash, ordinary_schedule_id = bank_binding(
            ordinary_bank_name, ordinary_schedule_hash
        )
        matrix_kwargs: Dict[str, Any] = {}
        if confirmation:
            lead40_bank_hash, lead40_schedule_id = bank_binding(
                "lead-40", config.lead40_schedule_hash
            )
            lead80_bank_hash, lead80_schedule_id = bank_binding(
                "lead-80", config.lead80_schedule_hash
            )
            matrix_kwargs.update(
                {
                    "lead_40_schedule_sha": config.lead40_schedule_hash,
                    "lead_80_schedule_sha": config.lead80_schedule_hash,
                    "lead_40_suite_bank_sha": lead40_bank_hash,
                    "lead_80_suite_bank_sha": lead80_bank_hash,
                    "lead_40_schedule_id": lead40_schedule_id,
                    "lead_80_schedule_id": lead80_schedule_id,
                }
            )
        specs = tuple(
            build_evaluation_matrix(
                candidate_model_sha=candidate_hash,
                champion_model_sha=champion_hash,
                original_model_sha=config.original_hash,
                powered_config_sha=config.powered_config_hash,
                standard_config_sha=config.standard_config_hash,
                powered_schedule_sha=ordinary_schedule_hash,
                standard_schedule_sha=(
                    config.standard_confirmation_schedule_hash
                    if confirmation
                    else ordinary_schedule_hash
                ),
                policy_sha=config.policy_hash,
                suite=ordinary_bank_name,
                stage=stage,
                look=look,
                topology=topology,
                suite_manifest_sha=config.suite_manifest_hash,
                ordinary_suite_bank_sha=ordinary_bank_hash,
                powered_schedule_id=ordinary_schedule_id,
                standard_schedule_id=ordinary_schedule_id,
                **matrix_kwargs,
            )
        )
        config_hash = canonical_sha256(sorted({spec.config_sha for spec in specs}))
        schedule_hash = canonical_sha256(sorted({spec.schedule_sha for spec in specs}))
        evaluation_key = "matrix-" + canonical_sha256(
            [spec.to_dict() for spec in specs]
        )
        return EvaluationMatrixPlan(
            specs,
            evaluation_key,
            config_hash,
            schedule_hash,
            config.policy_hash,
            config.selfplay_config_hash,
            required_topology,
        )

    def evaluate_or_recommend(
        self, plan: EvaluationMatrixPlan, candidate: CandidateArtifact
    ) -> Mapping[str, Any]:
        """Return plans in recommendation mode or injected gate output in automatic mode."""

        if not plan.specs:
            return {"decision": "INCONCLUSIVE", "reason": "missing-evaluation-matrix"}
        if self.recommendation_only:
            return {"decision": "RECOMMEND", "plan": plan.to_dict()}
        if self.evaluation_executor is None:
            return {
                "decision": "INCONCLUSIVE",
                "reason": "missing-evaluation-executor",
            }
        proof: Any = None
        with self._exclusive_gpu_handoff(plan, candidate) as proof:
            raw_evidence = self.evaluation_executor(plan, candidate)
        verified = self._validate_gpu_handoff(proof)
        if not isinstance(raw_evidence, Mapping):
            raise SafetyHalt("evaluation executor returned no immutable envelope")
        evidence = {
            **dict(raw_evidence),
            "gpu_handoff": verified,
            "gpu_handoff_hash": canonical_sha256(verified),
            "selfplay_config_hash": plan.selfplay_config_hash,
            "topology": plan.topology,
        }
        if self.gate_evaluator is None:
            return {"decision": "INCONCLUSIVE", "reason": "missing-gate-evaluator"}
        gate = dict(self.gate_evaluator(evidence))
        if gate.get("gpu_handoff_hash") != evidence["gpu_handoff_hash"]:
            raise SafetyHalt("gate result omits GPU handoff provenance")
        return gate

    def configured_evaluation_executor(
        self, plan: EvaluationMatrixPlan, candidate: CandidateArtifact
    ) -> Mapping[str, Any]:
        """Run the configured shell-free evaluator adapter and verify its envelope."""

        if self.recommendation_only:
            raise SafetyHalt("configured evaluation execution requires automatic mode")
        if not plan.specs:
            raise SafetyHalt("configured evaluation has no matrix specifications")
        root = self.runtime.evaluations / "controller-adapter" / plan.evaluation_key
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "plan.json"
        evidence_path = root / "evidence.json"
        _write_immutable_json(plan_path, plan.to_dict())

        if not evidence_path.exists():
            champion_spec = next(
                (
                    spec
                    for spec in plan.specs
                    if spec.comparison == "candidate-vs-champion-powered"
                ),
                None,
            )
            if champion_spec is None:
                raise SafetyHalt("evaluation matrix has no champion comparison")
            schedule_paths = {
                self.runtime.controller.discovery_schedule_hash:
                    self.runtime.discovery_schedule_path,
                self.runtime.controller.confirmation_schedule_hash:
                    self.runtime.confirmation_schedule_path,
                self.runtime.controller.audit_schedule_hash:
                    self.runtime.audit_schedule_path,
                self.runtime.controller.lead40_schedule_hash:
                    self.runtime.lead40_schedule_path,
                self.runtime.controller.lead80_schedule_hash:
                    self.runtime.lead80_schedule_path,
                self.runtime.controller.standard_confirmation_schedule_hash:
                    self.runtime.standard_confirmation_schedule_path,
            }
            self.execute_argv(
                "evaluator",
                {
                    "plan": plan_path,
                    "evaluation_key": plan.evaluation_key,
                    "evidence_output": evidence_path,
                    "candidate_dir": candidate.path,
                    "candidate_model": candidate.path / "model.bin.gz",
                    "candidate_hash": candidate.model_hash,
                    "champion_hash": champion_spec.reference_model_sha,
                    "original_model": self.runtime.original_model_path,
                    "original_hash": self.runtime.controller.original_hash,
                    "policy": str(self.runtime.policy_path),
                    "policy_hash": plan.policy_hash,
                    "powered_config": self.runtime.powered_config_path,
                    "standard_config": self.runtime.standard_config_path,
                    "powered_schedule": schedule_paths[
                        champion_spec.schedule_sha
                    ],
                    "standard_schedule": schedule_paths[
                        next(
                            spec.schedule_sha
                            for spec in plan.specs
                            if "standard" in spec.comparison
                        )
                    ],
                    "evaluations_root": self.runtime.evaluations,
                },
            )
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise SafetyHalt(
                "configured evaluator did not atomically publish a regular evidence file"
            )
        try:
            evidence_bytes = evidence_path.read_bytes()
            evidence = json.loads(evidence_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"configured evaluator evidence is invalid: {exc}") from exc
        if not isinstance(evidence, dict):
            raise SafetyHalt("configured evaluator evidence root must be an object")
        if evidence_bytes != canonical_json_bytes(evidence) + b"\n":
            raise SafetyHalt("configured evaluator evidence must be canonical JSON")

        champion_spec = next(
            spec
            for spec in plan.specs
            if spec.comparison == "candidate-vs-champion-powered"
        )
        expected = {
            "schema_version": 1,
            "finalized": True,
            "controller_stage": plan.specs[0].stage,
            "candidate_hash": candidate.model_hash,
            "tested_champion_hash": champion_spec.reference_model_sha,
            "original_hash": self.runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
            "selfplay_config_hash": plan.selfplay_config_hash,
            "topology": plan.topology,
        }
        conflicts = [
            key for key, value in expected.items() if evidence.get(key) != value
        ]
        if conflicts:
            raise SafetyHalt(
                "configured evaluator evidence contradicts its immutable plan: "
                + ", ".join(sorted(conflicts))
            )
        return evidence

    def process_evaluation_stage(
        self,
        candidate_hash: str,
        *,
        stage: str,
        suite: str,
        look: str,
        topology: str,
    ) -> Mapping[str, Any]:
        """Plan or execute one lifecycle stage through injected adapters."""

        targets = {
            "integrity": CandidateState.EVALUATING_INTEGRITY,
            "screen": CandidateState.EVALUATING_SCREEN,
            "finalist": CandidateState.EVALUATING_FINALIST,
            "confirmation": CandidateState.EVALUATING_CONFIRMATION,
        }
        if stage not in targets:
            raise ValueError(f"unknown evaluation stage: {stage}")
        state = self.registry.reconstruct()
        candidate_record = state.candidates.get(candidate_hash)
        if candidate_record is None:
            raise SafetyHalt(f"unknown candidate: {candidate_hash}")
        artifact = inspect_candidate(Path(candidate_record.candidate_path))
        plan = self.build_evaluation_plan(
            candidate_hash,
            state.current_champion_hash,
            suite=suite,
            stage=stage,
            look=look,
            topology=topology,
        )
        if self.recommendation_only:
            return {"decision": "RECOMMEND", "plan": plan.to_dict()}
        if (
            stage == "confirmation"
            and candidate_record.state == CandidateState.CONFIRMED
            and candidate_record.evaluation_key == plan.evaluation_key
        ):
            report_path = self.runtime.reports / f"{plan.evaluation_key}.final.json"
            if not report_path.is_file():
                raise SafetyHalt("confirmed candidate is missing finalized report")
            return {
                "decision": "PASS",
                "evaluationKey": plan.evaluation_key,
                "stage": stage,
                "reportPath": str(report_path),
                "reportHash": sha256_file(report_path),
                "reused": True,
            }
        if self.evaluation_executor is None or self.gate_evaluator is None:
            return {
                "decision": "INCONCLUSIVE",
                "reason": "evaluation-or-gate-adapter-missing",
                "plan": plan.to_dict(),
            }
        with self._writer_lock():
            current = self.registry.reconstruct()
            if current.current_champion_hash != state.current_champion_hash:
                raise SafetyHalt("champion changed before evaluation stage started")
            provenance = self._provenance(plan.config_hash, plan.schedule_hash)
            transition_payload: Dict[str, Any] = {"matrix": plan.to_dict()}
            if stage == "confirmation":
                self._rank_finalists(current)
                ranking_path = (
                    self.runtime.evaluations
                    / "rankings"
                    / f"{state.current_generation_id}.{state.current_champion_hash}.json"
                )
                if not ranking_path.is_file():
                    raise SafetyHalt(
                        "confirmation allocation lacks immutable finalist ranking"
                    )
                ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
                if ranking.get("selected_candidate_hash") != candidate_hash:
                    raise SafetyHalt("candidate was not selected by finalist ranking")
                transition_payload.update(
                    {
                        "ranking_path": str(ranking_path),
                        "ranking_hash": sha256_file(ranking_path),
                    }
                )
            self.registry.transition_candidate(
                candidate_hash,
                str(artifact.path),
                targets[stage],
                provenance=provenance,
                champion_hash=state.current_champion_hash,
                evaluation_key=plan.evaluation_key,
                reason=f"{stage} evaluation matrix started",
                actor=self.runtime.controller.actor,
                payload=transition_payload,
            )
            handoff_proof: Any = None
            with self._exclusive_gpu_handoff(plan, artifact) as handoff_proof:
                raw_evidence = self.evaluation_executor(plan, artifact)
            verified_handoff = self._validate_gpu_handoff(handoff_proof)
            if not isinstance(raw_evidence, Mapping):
                raise SafetyHalt("evaluation executor returned no immutable envelope")
            evidence = {
                **dict(raw_evidence),
                "gpu_handoff": verified_handoff,
                "gpu_handoff_hash": canonical_sha256(verified_handoff),
                "selfplay_config_hash": plan.selfplay_config_hash,
                "topology": plan.topology,
                "controller_stage": stage,
            }
            gate = dict(self.gate_evaluator(evidence))
            if gate.get("gpu_handoff_hash") != evidence["gpu_handoff_hash"]:
                raise SafetyHalt("gate result omits or contradicts GPU handoff proof")
            gate["gpu_handoff"] = verified_handoff
            decision = gate.get("decision", "INCONCLUSIVE")
            if decision not in {"PASS", "FAIL", "INCONCLUSIVE"}:
                raise SafetyHalt(f"unknown gate decision: {decision!r}")
            result: Dict[str, Any] = {
                "decision": decision,
                "evaluationKey": plan.evaluation_key,
                "stage": stage,
            }
            result_root = (
                self.runtime.evaluations / "controller-results" / candidate_hash
            )
            result_root.mkdir(parents=True, exist_ok=True)
            result_name = (
                "confirmation-look-2.json"
                if stage == "confirmation" and look == "prespecified-second-look"
                else f"{stage}.json"
            )
            _write_immutable_json(
                result_root / result_name,
                {
                    "schema_version": 1,
                    "candidate_hash": candidate_hash,
                    "tested_champion_hash": state.current_champion_hash,
                    "evaluation_key": plan.evaluation_key,
                    "decision": decision,
                    "gate": gate,
                },
            )
            if stage == "confirmation":
                report_path, report_hash, _ = self.finalize_gate_report(
                    plan,
                    candidate_hash=candidate_hash,
                    tested_champion_hash=state.current_champion_hash,
                    gate_result=gate,
                )
                result.update(
                    {"reportPath": str(report_path), "reportHash": report_hash}
                )
                if decision == "PASS":
                    self.registry.transition_candidate(
                        candidate_hash,
                        str(artifact.path),
                        CandidateState.CONFIRMED,
                        provenance=provenance,
                        champion_hash=state.current_champion_hash,
                        evaluation_key=plan.evaluation_key,
                        reason="finalized confirmation PASS",
                        actor=self.runtime.controller.actor,
                        payload={"report_hash": report_hash},
                    )
                elif decision == "FAIL":
                    self._move_candidate_terminal(
                        artifact,
                        CandidateState.REJECTED,
                        provenance=provenance,
                        champion_hash=state.current_champion_hash,
                        evaluation_key=plan.evaluation_key,
                        reason="finalized confirmation FAIL",
                    )
            elif decision == "FAIL":
                self._move_candidate_terminal(
                    artifact,
                    CandidateState.REJECTED,
                    provenance=provenance,
                    champion_hash=state.current_champion_hash,
                    evaluation_key=plan.evaluation_key,
                    reason=f"{stage} evaluation failed",
                )
            return result

    def finalize_gate_report(
        self,
        plan: EvaluationMatrixPlan,
        *,
        candidate_hash: str,
        tested_champion_hash: str,
        gate_result: Mapping[str, Any],
    ) -> Tuple[Path, str, Mapping[str, Any]]:
        """Write an immutable finalized report compatible with champion CAS."""

        if self.recommendation_only:
            raise SafetyHalt("recommendation mode may not finalize report files")
        decision = gate_result.get("decision", "INCONCLUSIVE")
        if decision not in {"PASS", "FAIL", "INCONCLUSIVE"}:
            raise SafetyHalt(f"unknown gate decision: {decision!r}")
        expected_gate_metadata = {
            "finalized": True,
            "candidate_hash": candidate_hash,
            "tested_champion_hash": tested_champion_hash,
            "original_hash": self.runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
            "selfplay_config_hash": plan.selfplay_config_hash,
            "topology": plan.topology,
        }
        conflicts = [
            key
            for key, expected in expected_gate_metadata.items()
            if gate_result.get(key) != expected
        ]
        if conflicts:
            raise SafetyHalt(
                "gate result is not finalized for the exact evaluation: "
                + ", ".join(sorted(conflicts))
            )
        gpu_handoff_hash = _hash(
            gate_result.get("gpu_handoff_hash"), "gate GPU handoff hash"
        )
        gpu_handoff = self._validate_gpu_handoff(
            gate_result.get("gpu_handoff")
        )
        if canonical_sha256(gpu_handoff) != gpu_handoff_hash:
            raise SafetyHalt("gate GPU handoff proof hash mismatch")
        if not plan.specs:
            decision = "INCONCLUSIVE"
        path = self.runtime.reports / f"{plan.evaluation_key}.final.json"
        stable_fields = {
            "candidate_hash": candidate_hash,
            "tested_champion_hash": tested_champion_hash,
            "original_hash": self.runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
            "selfplay_config_hash": plan.selfplay_config_hash,
            "topology": plan.topology,
            "gpu_handoff_hash": gpu_handoff_hash,
            "decision": decision,
            "matrix": plan.to_dict(),
            "gate": dict(gate_result),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or any(
                existing.get(key) != value for key, value in stable_fields.items()
            ):
                raise SafetyHalt(f"finalized report conflicts: {path}")
            return path, sha256_file(path), existing
        report = {
            "schema_version": 1,
            "finalized": True,
            "finalized_at_utc": utc_timestamp(),
            **stable_fields,
        }
        _write_immutable_json(path, report)
        return path, sha256_file(path), report

    def _candidate_manifest_hash(self, path: Path) -> str:
        return inspect_candidate(path).directory_manifest_hash

    def _known_started_hashes(self, state: Any) -> set:
        return {
            candidate.candidate_hash
            for candidate in state.candidates.values()
            if candidate.state
            in {
                CandidateState.EVALUATING_INTEGRITY,
                CandidateState.EVALUATING_SCREEN,
                CandidateState.EVALUATING_FINALIST,
                CandidateState.EVALUATING_CONFIRMATION,
                CandidateState.CONFIRMED,
            }
        }

    def _validate_duplicate_names(
        self, candidates: Sequence[CandidateArtifact], state: Any
    ) -> None:
        known_names: Dict[str, str] = {
            Path(item.candidate_path).name: item.candidate_hash
            for item in state.candidates.values()
        }
        for item in candidates:
            existing = known_names.get(item.name)
            if existing is not None and existing != item.model_hash:
                raise SafetyHalt(
                    f"candidate name {item.name!r} changed hash "
                    f"from {existing} to {item.model_hash}"
                )

    def _quarantine_contradiction(self, path: Path, reason: str) -> Optional[Path]:
        """Move contradictory controller-owned evidence aside without deleting it."""

        path = Path(path)
        if not path.exists():
            return None
        suffix = canonical_sha256({"path": str(path), "reason": reason})[:16]
        destination = self.runtime.candidate_quarantine / (
            f"{path.name}.conflict-{suffix}"
        )
        if destination.exists():
            return destination
        _same_device((path.parent, destination.parent))
        os.rename(path, destination)
        fsync_directory(path.parent)
        if path.parent != destination.parent:
            fsync_directory(destination.parent)
        _write_immutable_json(
            self.runtime.candidate_quarantine / f"{destination.name}.json",
            {
                "schema_version": 1,
                "reason": reason,
                "source_path": str(path),
                "quarantine_path": str(destination),
            },
        )
        return destination

    def _archive_duplicate(
        self, candidate: CandidateArtifact, representative_path: str
    ) -> None:
        intent_root = self.runtime.promotion_root / "candidates" / "dedup-intents"
        destination = self.runtime.candidate_deduplicated / candidate.name
        intent = {
            "schema_version": 1,
            "candidate_name": candidate.name,
            "model_hash": candidate.model_hash,
            "manifest_hash": candidate.directory_manifest_hash,
            "source_path": str(candidate.path),
            "destination_path": str(destination),
            "representative_path": representative_path,
        }
        intent_path = intent_root / f"{candidate.name}.json"
        _write_immutable_json(intent_path, intent)
        _recoverable_rename(
            candidate.path,
            destination,
            expected_manifest_hash=candidate.directory_manifest_hash,
            inspector=self._candidate_manifest_hash,
        )
        _write_immutable_json(
            intent_root / f"{candidate.name}.complete.json",
            {
                "schema_version": 1,
                "model_hash": candidate.model_hash,
                "manifest_hash": candidate.directory_manifest_hash,
            },
        )

    def _deduplicate_inbox(
        self,
        candidates: Sequence[CandidateArtifact],
        state: Any,
        *,
        mutate: bool,
    ) -> Tuple[Tuple[CandidateArtifact, ...], Tuple[CandidateArtifact, ...]]:
        representatives: Dict[str, str] = {
            record.candidate_hash: record.candidate_path
            for record in state.candidates.values()
        }
        unique: List[CandidateArtifact] = []
        duplicates: List[CandidateArtifact] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (item.sample_count, item.data_count, item.name),
        ):
            representative = representatives.get(candidate.model_hash)
            if representative is None:
                representatives[candidate.model_hash] = str(candidate.path)
                unique.append(candidate)
                continue
            if Path(representative) == candidate.path:
                unique.append(candidate)
                continue
            duplicates.append(candidate)
            if mutate:
                self._archive_duplicate(candidate, representative)
        return tuple(unique), tuple(duplicates)

    def _move_candidate_terminal(
        self,
        candidate: CandidateArtifact,
        target: CandidateState,
        *,
        provenance: EventProvenance,
        champion_hash: str,
        evaluation_key: Optional[str],
        reason: str,
    ) -> CandidateArtifact:
        roots = {
            CandidateState.SUPERSEDED: self.runtime.candidate_superseded,
            CandidateState.REJECTED: self.runtime.candidate_rejected,
        }
        if target not in roots:
            raise ValueError(f"unsupported terminal candidate move: {target.value}")
        destination = roots[target] / candidate.name
        intent_root = self.runtime.promotion_root / "candidates" / "lifecycle-intents"
        intent_path = intent_root / f"{candidate.model_hash}.{target.value}.json"
        intent = {
            "schema_version": 1,
            "candidate_hash": candidate.model_hash,
            "source_path": str(candidate.path),
            "destination_path": str(destination),
            "manifest_hash": candidate.directory_manifest_hash,
            "target_state": target.value,
            "champion_hash": champion_hash,
            "evaluation_key": evaluation_key,
            "reason": reason,
            "config_hash": provenance.config_hash,
            "schedule_hash": provenance.schedule_hash,
        }
        _write_immutable_json(intent_path, intent)
        _recoverable_rename(
            candidate.path,
            destination,
            expected_manifest_hash=candidate.directory_manifest_hash,
            inspector=self._candidate_manifest_hash,
        )
        self._checkpoint(f"candidate-{target.value.lower()}-renamed")
        self.registry.transition_candidate(
            candidate.model_hash,
            str(destination),
            target,
            provenance=provenance,
            champion_hash=champion_hash,
            evaluation_key=evaluation_key,
            reason=reason,
            actor=self.runtime.controller.actor,
            payload={"manifest_hash": candidate.directory_manifest_hash},
        )
        _write_immutable_json(
            intent_path.with_name(intent_path.stem + ".complete.json"),
            {
                "schema_version": 1,
                "candidate_hash": candidate.model_hash,
                "target_state": target.value,
            },
        )
        return inspect_candidate(destination)

    def _reconcile_lifecycle_moves(self) -> None:
        root = self.runtime.promotion_root / "candidates" / "lifecycle-intents"
        for intent_path in sorted(root.glob("*.json")):
            if intent_path.name.endswith(".complete.json"):
                continue
            complete = intent_path.with_name(intent_path.stem + ".complete.json")
            if complete.exists():
                continue
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            target = CandidateState(intent["target_state"])
            state = self.registry.reconstruct()
            record = state.candidates.get(intent["candidate_hash"])
            if record is None:
                raise SafetyHalt("lifecycle move intent references unknown candidate")
            destination = Path(intent["destination_path"])
            _recoverable_rename(
                Path(intent["source_path"]),
                destination,
                expected_manifest_hash=intent["manifest_hash"],
                inspector=self._candidate_manifest_hash,
            )
            if record.state != target:
                artifact = inspect_candidate(destination)
                provenance = self._provenance(
                    intent["config_hash"], intent["schedule_hash"]
                )
                self.registry.transition_candidate(
                    artifact.model_hash,
                    str(destination),
                    target,
                    provenance=provenance,
                    champion_hash=intent["champion_hash"],
                    evaluation_key=intent["evaluation_key"],
                    reason=intent["reason"],
                    actor=self.runtime.controller.actor,
                    payload={"manifest_hash": artifact.directory_manifest_hash},
                )
            _write_immutable_json(
                complete,
                {
                    "schema_version": 1,
                    "candidate_hash": intent["candidate_hash"],
                    "target_state": target.value,
                },
            )

    def _claim_candidate(
        self, candidate: CandidateArtifact, state: Any
    ) -> CandidateArtifact:
        known = state.candidates.get(candidate.model_hash)
        if known is not None:
            return inspect_candidate(Path(known.candidate_path))
        self._require_disk(candidate.size_bytes)
        claimed_root = self.runtime.promotion_root / "candidates" / "claimed"
        destination = claimed_root / candidate.name
        intent_root = self.runtime.promotion_root / "candidates" / "claim-intents"
        claim_intent = {
            "schema_version": 1,
            "name": candidate.name,
            "source_path": str(candidate.path),
            "destination_path": str(destination),
            "model_hash": candidate.model_hash,
            "manifest_hash": candidate.directory_manifest_hash,
            "parent_champion_hash": state.current_champion_hash,
        }
        _write_immutable_json(intent_root / f"{candidate.name}.json", claim_intent)
        self._checkpoint("candidate-claim-intent-written")
        _recoverable_rename(
            candidate.path,
            destination,
            expected_manifest_hash=candidate.directory_manifest_hash,
            inspector=self._candidate_manifest_hash,
        )
        self._checkpoint("candidate-renamed")
        provenance = self._provenance(
            self.runtime.controller.powered_config_hash,
            self.runtime.controller.discovery_schedule_hash,
        )
        self.registry.transition_candidate(
            candidate.model_hash,
            str(candidate.path),
            CandidateState.DISCOVERED,
            provenance=provenance,
            champion_hash=state.current_champion_hash,
            reason="complete candidate export discovered",
            actor=self.runtime.controller.actor,
            payload={"manifest_hash": candidate.directory_manifest_hash},
        )
        self._checkpoint("candidate-discovered-event")
        self.registry.transition_candidate(
            candidate.model_hash,
            str(destination),
            CandidateState.CLAIMED,
            provenance=provenance,
            champion_hash=state.current_champion_hash,
            reason="candidate atomically claimed",
            actor=self.runtime.controller.actor,
            payload={"manifest_hash": candidate.directory_manifest_hash},
        )
        self._checkpoint("candidate-claimed-event")
        _write_immutable_json(
            intent_root / f"{candidate.name}.complete.json",
            {
                "schema_version": 1,
                "model_hash": candidate.model_hash,
                "manifest_hash": candidate.directory_manifest_hash,
            },
        )
        return inspect_candidate(destination)

    def _reconcile_unregistered_claims(self) -> None:
        claimed_root = self.runtime.promotion_root / "candidates" / "claimed"
        claimed, _ = inventory_candidates(claimed_root)
        state = self.registry.reconstruct()
        self._validate_duplicate_names(claimed, state)
        for candidate in claimed:
            intent_path = (
                self.runtime.promotion_root
                / "candidates"
                / "claim-intents"
                / f"{candidate.name}.json"
            )
            if not intent_path.exists():
                raise SafetyHalt(
                    f"unregistered claimed candidate has no durable intent: {candidate.name}"
                )
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if (
                intent.get("model_hash") != candidate.model_hash
                or intent.get("manifest_hash") != candidate.directory_manifest_hash
                or intent.get("destination_path") != str(candidate.path)
            ):
                self._quarantine_contradiction(
                    candidate.path, "claimed-destination-intent-mismatch"
                )
                raise SafetyHalt(
                    f"claimed destination contradicts durable intent: {candidate.path}"
                )
            known = state.candidates.get(candidate.model_hash)
            if known is not None and known.state != CandidateState.DISCOVERED:
                continue
            if intent.get("parent_champion_hash") != state.current_champion_hash:
                raise SafetyHalt(
                    "champion changed after candidate claim intent and before registration"
                )
            source = self.runtime.candidate_inbox / candidate.name
            provenance = self._provenance(
                self.runtime.controller.powered_config_hash,
                self.runtime.controller.discovery_schedule_hash,
            )
            if known is None:
                self.registry.transition_candidate(
                    candidate.model_hash,
                    str(source),
                    CandidateState.DISCOVERED,
                    provenance=provenance,
                    champion_hash=state.current_champion_hash,
                    reason="recovered candidate rename after restart",
                    actor=self.runtime.controller.actor,
                    payload={"manifest_hash": candidate.directory_manifest_hash},
                )
            self.registry.transition_candidate(
                candidate.model_hash,
                str(candidate.path),
                CandidateState.CLAIMED,
                provenance=provenance,
                champion_hash=state.current_champion_hash,
                reason="recovered claimed candidate after restart",
                actor=self.runtime.controller.actor,
                payload={"manifest_hash": candidate.directory_manifest_hash},
            )
            _write_immutable_json(
                intent_path.with_name(f"{candidate.name}.complete.json"),
                {
                    "schema_version": 1,
                    "model_hash": candidate.model_hash,
                    "manifest_hash": candidate.directory_manifest_hash,
                },
            )
            state = self.registry.reconstruct()

    def _inventory_and_select(
        self, mutate: bool
    ) -> Tuple[Any, BacklogSelection, Tuple[str, ...], Tuple[str, ...]]:
        candidates, ignored = inventory_candidates(self.runtime.candidate_inbox)
        state = self.registry.reconstruct()
        if state.current_champion_hash is None:
            raise SafetyHalt("event registry has no bootstrapped champion")
        self._validate_duplicate_names(candidates, state)
        candidates, duplicates = self._deduplicate_inbox(
            candidates, state, mutate=mutate
        )
        terminal = {
            CandidateState.SUPERSEDED,
            CandidateState.REJECTED,
            CandidateState.QUARANTINED,
        }
        active_existing = sum(
            candidate.state not in terminal
            and not (
                candidate.state == CandidateState.CONFIRMED
                and candidate.generation_id is not None
            )
            for candidate in state.candidates.values()
        )
        remaining_slots = self.runtime.controller.max_active_queue - active_existing
        if remaining_slots <= 0:
            selection = BacklogSelection((), tuple(candidates), {})
        else:
            selection = select_backlog(
                candidates,
                original_hash=self.runtime.controller.original_hash,
                anchor_interval_samples=self.runtime.controller.anchor_interval_samples,
                anomaly_names=self.runtime.controller.anomaly_names,
                max_active_queue=remaining_slots,
                evaluation_started_hashes=self._known_started_hashes(state),
            )
        if mutate:
            selected = []
            for item in selection.selected:
                selected.append(self._claim_candidate(item, self.registry.reconstruct()))
            provenance = self._provenance(
                self.runtime.controller.powered_config_hash,
                self.runtime.controller.discovery_schedule_hash,
            )
            for item in selection.superseded:
                claimed = self._claim_candidate(item, self.registry.reconstruct())
                current = self.registry.reconstruct().candidates[claimed.model_hash]
                if current.state == CandidateState.CLAIMED:
                    self._move_candidate_terminal(
                        claimed,
                        CandidateState.SUPERSEDED,
                        provenance=provenance,
                        champion_hash=self.registry.reconstruct().current_champion_hash,
                        evaluation_key=None,
                        reason="deterministic backlog coalescing",
                    )
        return state, selection, ignored, tuple(item.name for item in duplicates)

    def reconcile(self, *, mutate: bool = False) -> Mapping[str, Any]:
        """Validate projections and manifests; recommendation mode is read-only."""

        state = self.registry.reconstruct()
        warnings: List[str] = []
        champion = None
        if self.runtime.champion_path.exists():
            champion = load_champion(self.runtime.champion_path)
            if (
                state.current_champion_hash is not None
                and champion.champion_hash != state.current_champion_hash
            ):
                repairable = any(
                    (path.parent / "champion-cas.json").is_file()
                    and json.loads(
                        (path.parent / "champion-cas.json").read_text(
                            encoding="utf-8"
                        )
                    ).get("champion_hash")
                    == champion.champion_hash
                    for path in (
                        self.runtime.promotion_root / "transactions"
                    ).glob("*/intent.json")
                )
                if mutate and not repairable:
                    raise SafetyHalt(
                        "champion projection differs from authoritative registry"
                    )
                warnings.append(
                    "champion-projection-awaits-known-cas-repair"
                    if repairable
                    else "champion-projection-differs-from-registry"
                )
        elif state.current_champion_hash is not None:
            if mutate:
                raise SafetyHalt("champion projection is missing in mutating mode")
            warnings.append("champion-json-missing")
        candidate_status = []
        for candidate in state.candidates.values():
            path = Path(candidate.candidate_path)
            if path.exists():
                artifact = inspect_candidate(path)
                if artifact.model_hash != candidate.candidate_hash:
                    raise SafetyHalt(f"candidate manifest contradiction: {path}")
                candidate_status.append(
                    {"hash": candidate.candidate_hash, "state": candidate.state.value, "present": True}
                )
            else:
                candidate_status.append(
                    {"hash": candidate.candidate_hash, "state": candidate.state.value, "present": False}
                )
        transactions = []
        transaction_root = self.runtime.promotion_root / "transactions"
        if transaction_root.exists():
            for path in sorted(transaction_root.glob("*/intent.json")):
                try:
                    intent = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise SafetyHalt(f"invalid promotion intent {path}: {exc}") from exc
                transactions.append(
                    {
                        "generationId": intent.get("generation_id"),
                        "candidateHash": intent.get("candidate_hash"),
                        "complete": (path.parent / "complete.json").exists(),
                    }
                )
        return {
            "mode": "automatic" if mutate else "recommend-only",
            "lastSequence": state.last_sequence,
            "championHash": state.current_champion_hash,
            "championProjectionHash": champion.champion_hash if champion else None,
            "currentGenerationId": state.current_generation_id,
            "candidates": candidate_status,
            "pins": sorted(state.pins),
            "transactions": transactions,
            "warnings": sorted(warnings),
        }

    def _stage_result(
        self, candidate_hash: str, stage: str
    ) -> Optional[Mapping[str, Any]]:
        path = (
            self.runtime.evaluations
            / "controller-results"
            / candidate_hash
            / f"{stage}.json"
        )
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"invalid controller evaluation result {path}: {exc}") from exc
        if (
            not isinstance(value, dict)
            or value.get("candidate_hash") != candidate_hash
            or value.get("decision") not in {"PASS", "FAIL", "INCONCLUSIVE"}
        ):
            raise SafetyHalt(f"controller evaluation result contradicts candidate: {path}")
        return value

    def _rank_finalists(self, state: Any) -> Optional[Mapping[str, Any]]:
        ranking_path = (
            self.runtime.evaluations
            / "rankings"
            / f"{state.current_generation_id}.{state.current_champion_hash}.json"
        )
        if ranking_path.exists():
            value = json.loads(ranking_path.read_text(encoding="utf-8"))
            if (
                value.get("generation_id") != state.current_generation_id
                or value.get("champion_hash") != state.current_champion_hash
                or value.get("policy_hash") != self.runtime.controller.policy_hash
            ):
                raise SafetyHalt("persisted finalist ranking provenance changed")
            return value
        prefinal_states = {
            CandidateState.CLAIMED,
            CandidateState.EVALUATING_INTEGRITY,
            CandidateState.EVALUATING_SCREEN,
        }
        if any(
            record.state in prefinal_states
            for record in state.candidates.values()
        ):
            return None
        rows = []
        for record in state.candidates.values():
            if record.state != CandidateState.EVALUATING_FINALIST:
                continue
            result = self._stage_result(record.candidate_hash, "finalist")
            gate = result.get("gate") if result else None
            if not result:
                return None
            if result.get("decision") != "PASS":
                continue
            if not isinstance(gate, Mapping):
                return None
            utility = gate.get("realized_powered_utility_lower_bound")
            risk = gate.get("final50_risk_upper_bound")
            if (
                isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or isinstance(risk, bool)
                or not isinstance(risk, (int, float))
            ):
                continue
            sample_count, _ = parse_candidate_counters(
                Path(record.candidate_path).name
            )
            rows.append(
                {
                    "candidate_hash": record.candidate_hash,
                    "realized_powered_utility_lower_bound": float(utility),
                    "final50_risk_upper_bound": float(risk),
                    "sample_count": sample_count,
                }
            )
        if not rows:
            return None

        def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
            utility_delta = (
                left["realized_powered_utility_lower_bound"]
                - right["realized_powered_utility_lower_bound"]
            )
            if abs(utility_delta) > 0.10:
                return -1 if utility_delta > 0 else 1
            if left["final50_risk_upper_bound"] != right["final50_risk_upper_bound"]:
                return (
                    -1
                    if left["final50_risk_upper_bound"]
                    < right["final50_risk_upper_bound"]
                    else 1
                )
            if left["sample_count"] != right["sample_count"]:
                return -1 if left["sample_count"] > right["sample_count"] else 1
            return -1 if left["candidate_hash"] < right["candidate_hash"] else 1

        ranked = sorted(rows, key=functools.cmp_to_key(compare))
        artifact = {
            "schema_version": 1,
            "generation_id": state.current_generation_id,
            "champion_hash": state.current_champion_hash,
            "policy_hash": self.runtime.controller.policy_hash,
            "ranking_rule":
                "utility-lcb;within-0.10-final50-risk;later-sample",
            "ranked": ranked,
            "selected_candidate_hash": ranked[0]["candidate_hash"],
        }
        artifact_hash = canonical_sha256(artifact)
        artifact = {**artifact, "ranking_hash": artifact_hash}
        if not self.recommendation_only:
            ranking_root = ranking_path.parent
            ranking_root.mkdir(parents=True, exist_ok=True)
            _write_immutable_json(ranking_path, artifact)
        return artifact

    def _orchestrate_active_queue(self) -> Tuple[Mapping[str, Any], ...]:
        """Advance each active candidate by at most one deterministic stage."""

        state = self.registry.reconstruct()
        confirmation_allocations = {
            event.candidate_hash
            for event in state.events
            if event.transition == Transition.EVALUATION_CONFIRMATION_STARTED
            and event.champion_hash == state.current_champion_hash
            and event.candidate_hash is not None
        }
        confirmation_inflight = bool(confirmation_allocations)
        ranking = self._rank_finalists(state)
        ranked_confirmation = (
            None if ranking is None else ranking["selected_candidate_hash"]
        )
        outcomes: List[Mapping[str, Any]] = []
        stage_for_state = {
            CandidateState.CLAIMED: "integrity",
            CandidateState.EVALUATING_INTEGRITY: "integrity",
            CandidateState.EVALUATING_SCREEN: "screen",
            CandidateState.EVALUATING_FINALIST: "finalist",
            CandidateState.EVALUATING_CONFIRMATION: "confirmation",
        }
        next_after_pass = {
            CandidateState.EVALUATING_INTEGRITY: "screen",
            CandidateState.EVALUATING_SCREEN: "finalist",
            CandidateState.EVALUATING_FINALIST: "confirmation",
        }
        for record in sorted(
            state.candidates.values(),
            key=lambda item: (Path(item.candidate_path).name, item.candidate_hash),
        ):
            stage = stage_for_state.get(record.state)
            if stage is None:
                continue
            prior_stage = stage_for_state.get(record.state)
            prior_result = (
                self._stage_result(record.candidate_hash, prior_stage)
                if record.state != CandidateState.CLAIMED
                else None
            )
            if prior_result is not None:
                if prior_result["decision"] != "PASS":
                    if (
                        prior_stage == "confirmation"
                        and prior_result["decision"] == "INCONCLUSIVE"
                    ):
                        gate = prior_result.get("gate", {})
                        authorized = (
                            isinstance(gate, Mapping)
                            and gate.get("second_look_authorized") is True
                            and _SHA_RE.fullmatch(
                                str(gate.get("new_holdout_hash", ""))
                            )
                            is not None
                            and _SHA_RE.fullmatch(
                                str(gate.get("new_alpha_allocation_hash", ""))
                            )
                            is not None
                        )
                        outcomes.append(
                            {
                                "candidateHash": record.candidate_hash,
                                "stage": prior_stage,
                                "decision": "PENDING",
                                "reason": (
                                    "second-look-allocation-awaiting-execution"
                                    if authorized
                                    else
                                    "second-look-holdout-and-alpha-not-authorized"
                                ),
                            }
                        )
                        continue
                    else:
                        outcomes.append(
                            {
                                "candidateHash": record.candidate_hash,
                                "stage": prior_stage,
                                "decision": prior_result["decision"],
                                "reused": True,
                            }
                        )
                        continue
                stage = next_after_pass.get(record.state, stage)
            if stage == "confirmation":
                if ranked_confirmation is None:
                    outcomes.append(
                        {
                            "candidateHash": record.candidate_hash,
                            "stage": "confirmation",
                            "decision": "PENDING",
                            "reason": "finalist-ranking-evidence-incomplete",
                        }
                    )
                    continue
                if record.candidate_hash != ranked_confirmation:
                    outcomes.append(
                        {
                            "candidateHash": record.candidate_hash,
                            "stage": "confirmation",
                            "decision": "QUEUED",
                            "reason": "lower-ranked-safe-finalist",
                        }
                    )
                    continue
                if (
                    confirmation_inflight
                    and record.candidate_hash not in confirmation_allocations
                ):
                    outcomes.append(
                        {
                            "candidateHash": record.candidate_hash,
                            "stage": "confirmation",
                            "decision": "QUEUED",
                            "reason": "confirmation-attempt-already-allocated",
                        }
                    )
                    continue
                confirmation_inflight = True
                confirmation_allocations.add(record.candidate_hash)
            outcome = dict(
                self.process_evaluation_stage(
                    record.candidate_hash,
                    stage=stage,
                    suite="confirmation" if stage == "confirmation" else stage,
                    look=(
                        "prespecified-second-look"
                        if (
                            stage == "confirmation"
                            and prior_result is not None
                            and prior_result.get("decision") == "INCONCLUSIVE"
                        )
                        else "fresh"
                        if stage == "confirmation"
                        else "automatic"
                    ),
                    topology="7-workers-100-threads",
                )
            )
            outcome["candidateHash"] = record.candidate_hash
            outcomes.append(outcome)
        return tuple(outcomes)

    def _ingest_rollout_ipc(self) -> bool:
        ingested = False
        for path in sorted(self.runtime.worker_ack_inbox.glob("*.json")):
            if path.name.startswith("."):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            self.record_worker_ack(
                value["generation_id"],
                value["worker_id"],
                value["model_hash"],
                report_path=path,
                report_hash=sha256_file(path),
            )
            ingested = True
        for path in sorted(self.runtime.rollout_report_inbox.glob("*.json")):
            if path.name.startswith("."):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            kwargs = {"report_path": path, "report_hash": sha256_file(path)}
            if value.get("phase") == "canary":
                self.mark_canary_passed(
                    value["generation_id"], value["candidate_hash"], **kwargs
                )
            elif value.get("phase") == "intermediate":
                self.mark_intermediate_passed(
                    value["generation_id"], value["candidate_hash"], **kwargs
                )
            ingested = True
        return ingested

    def _advance_confirmed_promotions(self) -> Tuple[Mapping[str, Any], ...]:
        if self.recommendation_only:
            return ()
        outcomes = []
        state = self.registry.reconstruct()
        for candidate in sorted(
            state.candidates.values(), key=lambda item: item.candidate_hash
        ):
            if candidate.state != CandidateState.CONFIRMED:
                continue
            report_path = (
                self.runtime.reports / f"{candidate.evaluation_key}.final.json"
            )
            if not report_path.is_file():
                raise SafetyHalt("confirmed candidate lacks finalized PASS report")
            existing = next(
                (
                    generation
                    for generation in state.generations.values()
                    if generation.candidate_hash == candidate.candidate_hash
                    and generation.state
                    not in {
                        GenerationState.ROLLED_BACK,
                        GenerationState.QUARANTINED,
                    }
                ),
                None,
            )
            generation_id = (
                existing.generation_id
                if existing is not None
                else f"generation-{candidate.candidate_hash[:20]}"
            )
            kwargs = {
                "pass_report_path": report_path,
                "pass_report_hash": sha256_file(report_path),
                "trainer_checkpoint_hash": sha256_file(
                    self.runtime.trainer_checkpoint
                ),
                "data_watermark_hash": sha256_file(
                    self.runtime.data_watermark_path
                ),
                "shuffle_watermark_hash": sha256_file(
                    self.runtime.shuffle_watermark_path
                ),
            }
            first = self.promote(
                candidate.candidate_hash, generation_id, **kwargs
            )
            ingested = self._ingest_rollout_ipc()
            if first.get("status") != "ACTIVE" and ingested:
                first = self.promote(
                    candidate.candidate_hash, generation_id, **kwargs
                )
            outcomes.append(first)
        return tuple(outcomes)

    def run_once(self) -> Mapping[str, Any]:
        """Scan once. Recommendation mode performs no filesystem mutation."""

        self.validate_static_inputs()
        if self.recommendation_only:
            state, selection, ignored, duplicates = self._inventory_and_select(False)
            status = dict(self.reconcile(mutate=False))
            status["inventory"] = {
                "selected": [item.name for item in selection.selected],
                "superseded": [item.name for item in selection.superseded],
                "ignored": list(ignored),
                "deduplicated": list(duplicates),
                "reasons": dict(selection.reasons),
            }
            status["orchestration"] = list(self._orchestrate_active_queue())
            return status
        if not self.runtime.lock_path.parent.exists():
            raise SafetyHalt(
                f"controller lock parent does not exist: {self.runtime.lock_path.parent}"
            )
        with self._writer_lock():
            self.ensure_layout()
            self._require_disk()
            self._reconcile_unregistered_claims()
            self._reconcile_lifecycle_moves()
            _, selection, ignored, duplicates = self._inventory_and_select(True)
            status = dict(self.reconcile(mutate=True))
            status["inventory"] = {
                "selected": [item.name for item in selection.selected],
                "superseded": [item.name for item in selection.superseded],
                "ignored": list(ignored),
                "deduplicated": list(duplicates),
                "reasons": dict(selection.reasons),
            }
        status["orchestration"] = list(self._orchestrate_active_queue())
        status["promotions"] = list(self._advance_confirmed_promotions())
        return status

    def run_reconcile(self) -> Mapping[str, Any]:
        """Reconcile only; automatic mode repairs unregistered completed claims."""

        self.validate_static_inputs()
        if self.recommendation_only:
            return self.reconcile(mutate=False)
        with self._writer_lock():
            self.ensure_layout()
            self._require_disk()
            self._reconcile_unregistered_claims()
            self._reconcile_lifecycle_moves()
            pending = []
            rollback_pending = [
                generation.generation_id
                for generation in self.registry.reconstruct().generations.values()
                if generation.state == GenerationState.ROLLBACK_PENDING
            ]
            transaction_root = self.runtime.promotion_root / "transactions"
            for path in sorted(transaction_root.glob("*/intent.json")):
                if not (path.parent / "complete.json").exists():
                    pending.append(json.loads(path.read_text(encoding="utf-8")))
        for generation_id in rollback_pending:
            self.rollback(generation_id)
        for intent in pending:
            generation = self.registry.reconstruct().generations.get(
                intent["generation_id"]
            )
            if generation is not None and generation.state == GenerationState.ROLLBACK_PENDING:
                self.rollback(intent["generation_id"])
                continue
            if generation is not None and generation.state in {
                GenerationState.ROLLED_BACK,
                GenerationState.QUARANTINED,
            }:
                continue
            self.promote(
                intent["candidate_hash"],
                intent["generation_id"],
                pass_report_path=Path(intent["pass_report_path"]),
                pass_report_hash=intent["pass_report_hash"],
                trainer_checkpoint_hash=intent["trainer_checkpoint_hash"],
                data_watermark_hash=intent["data_watermark_hash"],
                shuffle_watermark_hash=intent["shuffle_watermark_hash"],
            )
        return self.reconcile(mutate=True)

    def bootstrap(
        self,
        champion_hash: str,
        generation_id: str,
        *,
        confirmation: str,
    ) -> None:
        """Explicitly initialize champion JSON and registry under the writer lock."""

        if not self.automatic:
            raise SafetyHalt("bootstrap requires --automatic and mutationEnabled=true")
        if confirmation != "BOOTSTRAP_INITIAL_CHAMPION":
            raise SafetyHalt("bootstrap confirmation phrase is incorrect")
        _hash(champion_hash, "bootstrap champion hash")
        self.validate_static_inputs()
        with self._writer_lock():
            self.ensure_layout()
            provenance = self._provenance(
                self.runtime.controller.powered_config_hash,
                self.runtime.controller.discovery_schedule_hash,
            )
            bootstrap_champion(
                self.runtime.champion_path,
                champion_hash=champion_hash,
                generation_id=generation_id,
                provenance=provenance,
                actor=self.runtime.controller.actor,
            )
            self.registry.bootstrap_champion(
                champion_hash=champion_hash,
                generation_id=generation_id,
                provenance=provenance,
                reason="explicit initial champion bootstrap",
                actor=self.runtime.controller.actor,
            )

    def _transaction_dir(self, generation_id: str) -> Path:
        if (
            not generation_id
            or generation_id in {".", ".."}
            or Path(generation_id).name != generation_id
            or "/" in generation_id
            or "\\" in generation_id
            or "\x00" in generation_id
        ):
            raise SafetyHalt("generation_id is not a safe path component")
        return self.runtime.promotion_root / "transactions" / generation_id

    def _mark(self, transaction: Path, name: str, payload: Mapping[str, Any]) -> None:
        path = transaction / f"{name}.json"
        stable = {"schema_version": 1, **dict(payload)}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            comparable = dict(existing)
            comparable.pop("timestamp_utc", None)
            if comparable != stable:
                raise SafetyHalt(f"durable marker conflicts: {path}")
            return
        _write_immutable_json(path, {"timestamp_utc": utc_timestamp(), **stable})

    def _snapshot_checkpoint(self, generation_id: str, expected_hash: str) -> Path:
        source = self.runtime.trainer_checkpoint
        if sha256_file(source) != expected_hash:
            raise SafetyHalt("trainer checkpoint changed before promotion")
        root = self.runtime.rollback_quarantine / generation_id
        root.mkdir(parents=True, exist_ok=True)
        destination = root / "trainer-checkpoint"
        if destination.exists():
            if sha256_file(destination) != expected_hash:
                raise SafetyHalt("rollback checkpoint snapshot contradicts intent")
            return destination
        descriptor, name = tempfile.mkstemp(prefix=".checkpoint.", dir=str(root))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(temporary) != expected_hash:
                raise SafetyHalt("checkpoint snapshot hash mismatch")
            os.replace(temporary, destination)
            fsync_directory(root)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return destination

    def _generation_leaf(
        self, generation_id: str, accepted: Path, candidate_hash: str
    ) -> Path:
        self._transaction_dir(generation_id)
        root = self.runtime.accepted_models / "generations" / candidate_hash
        root.mkdir(parents=True, exist_ok=True)
        leaf = root / generation_id
        if leaf.exists():
            return self._verify_generation_leaf(
                generation_id, candidate_hash
            )
        temporary = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=str(root)))
        try:
            source = accepted / "model.bin.gz"
            if source.is_symlink() or not source.is_file():
                raise SafetyHalt("accepted model is not a regular file")
            model = temporary / "model.bin.gz"
            with source.open("rb") as input_file, model.open("xb") as output:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(model) != candidate_hash:
                raise SafetyHalt("copied generation model hash mismatch")
            os.chmod(model, 0o444)
            _write_immutable_json(
                temporary / "generation.json",
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "model_hash": candidate_hash,
                },
            )
            fsync_directory(temporary)
            os.chmod(temporary, 0o555)
            os.rename(temporary, leaf)
            fsync_directory(root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return leaf

    def _verify_generation_leaf(
        self, generation_id: str, candidate_hash: str
    ) -> Path:
        self._transaction_dir(generation_id)
        leaf = (
            self.runtime.accepted_models
            / "generations"
            / candidate_hash
            / generation_id
        )
        model = leaf / "model.bin.gz"
        if (
            leaf.is_symlink()
            or not leaf.is_dir()
            or model.is_symlink()
            or not model.is_file()
            or model.stat().st_mode & 0o222
            or sha256_file(model) != candidate_hash
        ):
            raise SafetyHalt("immutable generation leaf contradicts candidate")
        return leaf

    def _stage_workers(self, generation_id: str, candidate_hash: str) -> Path:
        root = self.runtime.rollout_quarantine / generation_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "acknowledgements").mkdir(exist_ok=True)
        data_root = root / "data"
        data_root.mkdir(exist_ok=True)
        for index in range(self.runtime.controller.worker_count):
            worker = root / f"worker-{index:03d}"
            worker.mkdir(exist_ok=True)
            _write_immutable_json(
                worker / "intent.json",
                {
                    "schema_version": 1,
                    "worker_id": index,
                    "generation_id": generation_id,
                    "model_hash": candidate_hash,
                    "selfplay_config_hash":
                        self.runtime.controller.selfplay_config_hash,
                    "policy": str(self.runtime.policy_path),
                    "policy_hash": self.runtime.controller.policy_hash,
                    "threads": self.runtime.controller.worker_threads,
                },
            )
        fsync_directory(root)
        return root

    def _launch_worker_phase(
        self,
        generation_id: str,
        candidate_hash: str,
        *,
        phase: str,
        start: int,
        stop: int,
    ) -> Tuple[Mapping[str, Any], ...]:
        results = []
        self._transaction_dir(generation_id)
        root = self.runtime.rollout_quarantine / generation_id
        leaf = self._verify_generation_leaf(generation_id, candidate_hash)
        for worker_id in range(start, stop):
            marker = root / f"worker-{worker_id:03d}" / f"launch-{phase}.json"
            supervisor_key = f"{generation_id}:worker-{worker_id:03d}"
            if marker.exists():
                value = json.loads(marker.read_text(encoding="utf-8"))
                identity = value.get("process_identity")
                if (
                    value.get("model_hash") != candidate_hash
                    or value.get("selfplay_config_hash")
                    != self.runtime.controller.selfplay_config_hash
                    or value.get("policy_hash")
                    != self.runtime.controller.policy_hash
                    or value.get("supervisor_key") != supervisor_key
                    or value.get("process_identity_verified") is not True
                    or not self._valid_process_identity(identity)
                ):
                    raise SafetyHalt("worker launch marker identity contradiction")
                ack_path = (
                    root
                    / "acknowledgements"
                    / f"worker-{worker_id:03d}.json"
                )
                if (
                    not ack_path.exists()
                    and (
                        self.process_identity_verifier is None
                        or not self.process_identity_verifier(identity)
                    )
                ):
                    raise SafetyHalt(
                        "cannot prove launched worker is the same process"
                    )
                results.append({"workerId": worker_id, "reused": True})
                continue
            output = root / "data" / f"worker-{worker_id:03d}"
            output.mkdir(parents=True, exist_ok=True)
            launch_intent = root / f"worker-{worker_id:03d}" / f"launch-{phase}.intent.json"
            _write_immutable_json(
                launch_intent,
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "model_hash": candidate_hash,
                    "selfplay_config_hash":
                        self.runtime.controller.selfplay_config_hash,
                    "policy_hash": self.runtime.controller.policy_hash,
                    "worker_id": worker_id,
                    "phase": phase,
                    "supervisor_key": supervisor_key,
                    "output_path": str(output),
                },
            )
            outcome = self.execute_argv(
                "selfplay",
                {
                    "generation_id": generation_id,
                    "model_hash": candidate_hash,
                    "model": leaf / "model.bin.gz",
                    "immutable_model_directory": leaf,
                    "worker_id": worker_id,
                    "worker_index": worker_id,
                    "worker_output_directory": output,
                    "phase": phase,
                    "threads": self.runtime.controller.worker_threads,
                    "selfplay_config": self.runtime.selfplay_config_path,
                    "selfplay_config_hash":
                        self.runtime.controller.selfplay_config_hash,
                    "policy_hash": self.runtime.controller.policy_hash,
                    "supervisor_key": supervisor_key,
                },
            )
            identity = (
                outcome.get("process_identity")
                if isinstance(outcome, Mapping)
                else getattr(outcome, "process_identity", None)
            )
            identity_verified = (
                isinstance(outcome, Mapping)
                and outcome.get("process_identity_verified") is True
            ) or (
                self.process_identity_verifier is not None
                and self._valid_process_identity(identity)
                and self.process_identity_verifier(identity)
            )
            if not self._valid_process_identity(identity) or not identity_verified:
                raise SafetyHalt("selfplay supervisor returned no verified process identity")
            _write_immutable_json(
                marker,
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "model_hash": candidate_hash,
                    "worker_id": worker_id,
                    "phase": phase,
                    "selfplay_config_hash":
                        self.runtime.controller.selfplay_config_hash,
                    "policy_hash": self.runtime.controller.policy_hash,
                    "supervisor_key": supervisor_key,
                    "process_identity": dict(identity),
                    "process_identity_verified": True,
                },
            )
            results.append({"workerId": worker_id, "launched": True})
        return tuple(results)

    @staticmethod
    def _valid_process_identity(identity: Any) -> bool:
        return (
            isinstance(identity, Mapping)
            and type(identity.get("pid")) is int
            and identity["pid"] > 0
            and type(identity.get("start_time_ticks")) is int
            and identity["start_time_ticks"] >= 0
            and isinstance(identity.get("command_sha256"), str)
            and _SHA_RE.fullmatch(identity["command_sha256"]) is not None
        )

    def record_worker_ack(
        self,
        generation_id: str,
        worker_id: int,
        candidate_hash: str,
        *,
        report_path: Optional[Path] = None,
        report_hash: Optional[str] = None,
    ) -> Path:
        """Record one exact-SHA acknowledgement under the writer lock."""

        if not self.automatic:
            raise SafetyHalt("worker acknowledgement mutation is disabled")
        self._transaction_dir(generation_id)
        if type(worker_id) is not int or not (0 <= worker_id < self.runtime.controller.worker_count):
            raise ValueError("worker_id is out of range")
        if report_path is None or report_hash is None:
            raise SafetyHalt("worker acknowledgement requires a hashed IPC report")
        _hash(report_hash, "worker acknowledgement report hash")
        with self._writer_lock():
            self._verify_generation_leaf(generation_id, candidate_hash)
            report_path = Path(report_path)
            if (
                report_path.is_symlink()
                or not report_path.is_file()
                or sha256_file(report_path) != report_hash
            ):
                raise SafetyHalt("worker acknowledgement report hash mismatch")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            output = (
                self.runtime.rollout_quarantine
                / generation_id
                / "data"
                / f"worker-{worker_id:03d}"
            )
            if (
                not output.exists()
                and (
                    self._transaction_dir(generation_id)
                    / "generation-data-admitted.json"
                ).exists()
            ):
                output = (
                    self.runtime.admitted_selfplay
                    / generation_id
                    / f"worker-{worker_id:03d}"
                )
            if not output.is_dir():
                raise SafetyHalt("worker output directory is missing")
            output_manifest, output_size = _tree_manifest(output)
            launch_markers = sorted(
                (
                    self.runtime.rollout_quarantine
                    / generation_id
                    / f"worker-{worker_id:03d}"
                ).glob("launch-*.json")
            )
            launch_markers = [
                item for item in launch_markers if not item.name.endswith(".intent.json")
            ]
            if len(launch_markers) != 1:
                raise SafetyHalt("worker acknowledgement has no unique launch marker")
            launch = json.loads(launch_markers[0].read_text(encoding="utf-8"))
            expected = {
                "schema_version": 1,
                "finalized": True,
                "generation_id": generation_id,
                "worker_id": worker_id,
                "model_hash": candidate_hash,
                "selfplay_config_hash":
                    self.runtime.controller.selfplay_config_hash,
                "policy_hash": self.runtime.controller.policy_hash,
                "threads": self.runtime.controller.worker_threads,
                "output_manifest_hash": output_manifest,
                "closed_files": True,
                "process_identity": launch["process_identity"],
            }
            if (
                output_size <= 0
                or not isinstance(report, dict)
                or any(report.get(key) != value for key, value in expected.items())
            ):
                raise SafetyHalt("worker acknowledgement provenance is incomplete")
            path = (
                self.runtime.rollout_quarantine
                / generation_id
                / "acknowledgements"
                / f"worker-{worker_id:03d}.json"
            )
            _write_immutable_json(path, {**report, "report_hash": report_hash})
            return path

    def _acknowledged(self, generation_id: str, candidate_hash: str) -> set:
        root = self.runtime.rollout_quarantine / generation_id / "acknowledgements"
        acknowledged = set()
        if not root.exists():
            return acknowledged
        for path in sorted(root.glob("worker-*.json")):
            if path.is_symlink() or not path.is_file():
                raise SafetyHalt(f"invalid worker acknowledgement: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("generation_id") != generation_id
                or value.get("model_hash") != candidate_hash
                or value.get("finalized") is not True
                or value.get("closed_files") is not True
                or value.get("selfplay_config_hash")
                != self.runtime.controller.selfplay_config_hash
                or value.get("policy_hash")
                != self.runtime.controller.policy_hash
                or value.get("threads") != self.runtime.controller.worker_threads
                or type(value.get("worker_id")) is not int
                or not (
                    0
                    <= value["worker_id"]
                    < self.runtime.controller.worker_count
                )
            ):
                raise SafetyHalt(f"worker acknowledgement contradiction: {path}")
            output = (
                self.runtime.rollout_quarantine
                / generation_id
                / "data"
                / f"worker-{value['worker_id']:03d}"
            )
            if (
                not output.exists()
                and (
                    self._transaction_dir(generation_id)
                    / "generation-data-admitted.json"
                ).exists()
            ):
                output = (
                    self.runtime.admitted_selfplay
                    / generation_id
                    / f"worker-{value['worker_id']:03d}"
                )
            if (
                not output.is_dir()
                or _tree_manifest(output)[1] <= 0
                or _tree_manifest(output)[0] != value.get("output_manifest_hash")
            ):
                raise SafetyHalt(f"worker output changed after acknowledgement: {path}")
            acknowledged.add(value["worker_id"])
        return acknowledged

    def _validate_rollout_health_report(
        self,
        generation_id: str,
        candidate_hash: str,
        phase: str,
        report_path: Path,
        report_hash: str,
    ) -> Mapping[str, Any]:
        self._transaction_dir(generation_id)
        _hash(report_hash, f"{phase} report hash")
        report_path = Path(report_path)
        if (
            report_path.is_symlink()
            or not report_path.is_file()
            or sha256_file(report_path) != report_hash
        ):
            raise SafetyHalt(f"{phase} report hash mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        required_workers = (
            self.runtime.controller.canary_worker_count
            if phase == "canary"
            else self.runtime.controller.intermediate_worker_count
        )
        promotion_intent = json.loads(
            (
                self._transaction_dir(generation_id) / "intent.json"
            ).read_text(encoding="utf-8")
        )
        expected = {
            "schema_version": 1,
            "finalized": True,
            "decision": "PASS",
            "phase": phase,
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "policy_hash": self.runtime.controller.policy_hash,
            "promotion_config_hash": promotion_intent["config_hash"],
            "selfplay_config_hash": self.runtime.controller.selfplay_config_hash,
            "audit_schedule_hash": self.runtime.controller.audit_schedule_hash,
            "topology": "7-workers-100-threads",
            "worker_count": required_workers,
            "model_purity_pass": True,
            "output_schema_pass": True,
            "throughput_pass": True,
            "crash_error_pass": True,
            "behavior_pass": True,
            "tactical_pass": True,
            "exploitability_pass": True,
            "catastrophe_pass": True,
        }
        if not isinstance(report, dict) or any(
            report.get(key) != value for key, value in expected.items()
        ):
            raise SafetyHalt(f"{phase} report does not satisfy frozen policy")
        if phase == "canary" and (
            report.get("game_count", 0) < 2000
            or report.get("fresh_audit_pairs", 0) < 1024
        ):
            raise SafetyHalt("canary report lacks required games/fresh audit")
        return {**report, "report_hash": report_hash}

    def mark_canary_passed(
        self,
        generation_id: str,
        candidate_hash: str,
        *,
        report_path: Optional[Path] = None,
        report_hash: Optional[str] = None,
    ) -> Path:
        """Authorize canary admission only from a finalized hashed report."""

        if not self.automatic:
            raise SafetyHalt("canary admission mutation is disabled")
        if report_path is None or report_hash is None:
            raise SafetyHalt("bare canary PASS markers are forbidden")
        with self._writer_lock():
            expected = set(range(self.runtime.controller.canary_worker_count))
            if not expected.issubset(self._acknowledged(generation_id, candidate_hash)):
                raise SafetyHalt("required canary workers have not acknowledged")
            report = self._validate_rollout_health_report(
                generation_id,
                candidate_hash,
                "canary",
                report_path,
                report_hash,
            )
            path = self._transaction_dir(generation_id) / "canary-pass.json"
            _write_immutable_json(path, report)
            return path

    def mark_intermediate_passed(
        self,
        generation_id: str,
        candidate_hash: str,
        *,
        report_path: Optional[Path] = None,
        report_hash: Optional[str] = None,
    ) -> Path:
        """Authorize expansion from the configured intermediate phase to full."""

        if not self.automatic:
            raise SafetyHalt("intermediate health mutation is disabled")
        if report_path is None or report_hash is None:
            raise SafetyHalt("bare intermediate PASS markers are forbidden")
        with self._writer_lock():
            expected = set(
                range(self.runtime.controller.intermediate_worker_count)
            )
            if not expected.issubset(
                self._acknowledged(generation_id, candidate_hash)
            ):
                raise SafetyHalt(
                    "required intermediate workers have not acknowledged"
                )
            report = self._validate_rollout_health_report(
                generation_id,
                candidate_hash,
                "intermediate",
                report_path,
                report_hash,
            )
            path = self._transaction_dir(generation_id) / "intermediate-pass.json"
            _write_immutable_json(path, report)
            return path

    def _admit_canary(self, generation_id: str, candidate_hash: str) -> bool:
        transaction = self._transaction_dir(generation_id)
        pass_path = transaction / "canary-pass.json"
        if not pass_path.exists():
            return False
        passed = json.loads(pass_path.read_text(encoding="utf-8"))
        if (
            passed.get("decision") != "PASS"
            or passed.get("candidate_hash") != candidate_hash
            or passed.get("generation_id") != generation_id
        ):
            raise SafetyHalt("canary PASS marker contradicts promotion")
        marker = transaction / "canary-admitted.json"
        if marker.exists():
            return True
        data_root = self.runtime.rollout_quarantine / generation_id / "data"
        canary_paths = [
            data_root / f"worker-{index:03d}"
            for index in range(self.runtime.controller.canary_worker_count)
        ]
        if any(not path.is_dir() for path in canary_paths):
            return False
        manifest_hash = canonical_sha256(
            [_tree_manifest(path)[0] for path in canary_paths]
        )
        self._mark(
            transaction,
            "canary-admitted",
            {"model_hash": candidate_hash, "manifest_hash": manifest_hash},
        )
        return True

    def _commit_generation_data(
        self, generation_id: str, candidate_hash: str
    ) -> bool:
        transaction = self._transaction_dir(generation_id)
        source = self.runtime.rollout_quarantine / generation_id / "data"
        destination = self.runtime.admitted_selfplay / generation_id
        marker = transaction / "generation-data-admitted.json"
        if marker.exists():
            value = json.loads(marker.read_text(encoding="utf-8"))
            if (
                not destination.is_dir()
                or _tree_manifest(destination)[0] != value.get("manifest_hash")
            ):
                raise SafetyHalt(
                    "generation data marker contradicts admitted generation"
                )
            return True
        if not source.is_dir():
            return False
        closed_manifests = []
        process_identities = []
        for worker_id in range(self.runtime.controller.worker_count):
            output = source / f"worker-{worker_id:03d}"
            ack_path = (
                self.runtime.rollout_quarantine
                / generation_id
                / "acknowledgements"
                / f"worker-{worker_id:03d}.json"
            )
            if not output.is_dir() or not ack_path.is_file():
                return False
            ack = json.loads(ack_path.read_text(encoding="utf-8"))
            actual_hash, actual_size = _tree_manifest(output)
            if (
                actual_size <= 0
                or ack.get("output_manifest_hash") != actual_hash
                or not self._valid_process_identity(ack.get("process_identity"))
            ):
                raise SafetyHalt("worker output is not stably closed for admission")
            closed_manifests.append(actual_hash)
            process_identities.append(ack["process_identity"])
        manifest_hash, _ = _tree_manifest(source)
        drain_plan = {
            "schema_version": 1,
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "source_manifest_hash": manifest_hash,
            "closed_file_manifests": closed_manifests,
            "process_identities": process_identities,
        }
        drain_path = transaction / "data-admission-drain.json"
        _write_immutable_json(drain_path, drain_plan)
        outcome = self.execute_argv(
            "drain",
            {
                "generation_id": generation_id,
                "model_hash": candidate_hash,
                "drain_manifest": drain_path,
            },
        )
        result = (
            dict(outcome)
            if isinstance(outcome, Mapping)
            else {
                "quiescent": getattr(outcome, "quiescent", None),
                "closed_file_manifests": getattr(
                    outcome, "closed_file_manifests", None
                ),
                "process_identities": getattr(outcome, "process_identities", None),
            }
        )
        if (
            result.get("quiescent") is not True
            or result.get("closed_file_manifests") != closed_manifests
            or result.get("process_identities") != process_identities
            or _tree_manifest(source)[0] != manifest_hash
        ):
            raise SafetyHalt("worker drain did not prove quiescent closed data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _recoverable_rename(
            source,
            destination,
            expected_manifest_hash=manifest_hash,
            inspector=lambda path: _tree_manifest(path)[0],
        )
        self._mark(
            transaction,
            "generation-data-admitted",
            {"model_hash": candidate_hash, "manifest_hash": manifest_hash},
        )
        return True

    def promote(
        self,
        candidate_hash: str,
        generation_id: str,
        *,
        pass_report_path: Path,
        pass_report_hash: str,
        trainer_checkpoint_hash: str,
        data_watermark_hash: str,
        shuffle_watermark_hash: str,
    ) -> Mapping[str, Any]:
        """Converge an idempotent promotion transaction as far as evidence permits."""

        if not self.automatic:
            raise SafetyHalt("promotion requires --automatic and mutationEnabled=true")
        for name, value in (
            ("candidate_hash", candidate_hash),
            ("pass_report_hash", pass_report_hash),
            ("trainer_checkpoint_hash", trainer_checkpoint_hash),
            ("data_watermark_hash", data_watermark_hash),
            ("shuffle_watermark_hash", shuffle_watermark_hash),
        ):
            _hash(value, name)
        if (
            sha256_file(self.runtime.data_watermark_path) != data_watermark_hash
            or sha256_file(self.runtime.shuffle_watermark_path)
            != shuffle_watermark_hash
        ):
            raise SafetyHalt("promotion watermark files changed or are unbound")
        try:
            shuffle_watermark = json.loads(
                self.runtime.shuffle_watermark_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"invalid shuffle watermark: {exc}") from exc
        shuffle_paths = (
            shuffle_watermark.get("derived_paths", [])
            if isinstance(shuffle_watermark, dict)
            else []
        )
        if (
            not isinstance(shuffle_paths, list)
            or any(not isinstance(item, str) for item in shuffle_paths)
        ):
            raise SafetyHalt("shuffle watermark derived_paths must be an array")
        authoritative_shuffle_paths = []
        for item in shuffle_paths:
            path = Path(item)
            if not path.is_absolute():
                raise SafetyHalt("shuffle watermark paths must be absolute")
            authoritative_shuffle_paths.append(str(path))
        with self._writer_lock():
            self.ensure_layout()
            state = self.registry.reconstruct()
            candidate = state.candidates.get(candidate_hash)
            if candidate is None or candidate.state != CandidateState.CONFIRMED:
                raise SafetyHalt("promotion requires a confirmed candidate")
            generation = state.generations.get(generation_id)
            if generation is not None and generation.candidate_hash != candidate_hash:
                raise SafetyHalt("generation_id is assigned to a different candidate")
            if generation is not None and generation.state == GenerationState.ROLLED_BACK:
                return {"status": "ROLLED_BACK", "generation_id": generation_id}
            if generation is not None and generation.state == GenerationState.QUARANTINED:
                raise SafetyHalt("generation is quarantined")
            report = load_finalized_pass_report(
                pass_report_path, expected_report_hash=pass_report_hash
            )
            report_value = json.loads(
                Path(pass_report_path).read_text(encoding="utf-8")
            )
            if report.decision != "PASS":
                raise SafetyHalt("promotion report is not PASS")
            tested_champion = (
                generation.previous_champion_hash
                if generation is not None
                else state.current_champion_hash
            )
            if (
                report.candidate_hash != candidate_hash
                or report.tested_champion_hash != tested_champion
                or report.original_hash != self.runtime.controller.original_hash
            ):
                raise SafetyHalt("promotion report identity is stale")
            if (
                report_value.get("selfplay_config_hash")
                != self.runtime.controller.selfplay_config_hash
                or report_value.get("topology") != "7-workers-100-threads"
                or not _SHA_RE.fullmatch(
                    str(report_value.get("gpu_handoff_hash", ""))
                )
            ):
                raise SafetyHalt(
                    "promotion report lacks frozen topology/GPU provenance"
                )
            provenance = self._provenance(report.config_hash, report.schedule_hash)
            transaction = self._transaction_dir(generation_id)
            transaction.mkdir(parents=True, exist_ok=True)
            existing_intent = None
            if (transaction / "intent.json").exists():
                existing_intent = json.loads(
                    (transaction / "intent.json").read_text(encoding="utf-8")
                )
                if existing_intent.get("candidate_hash") != candidate_hash:
                    raise SafetyHalt("promotion intent candidate conflicts")
                source = Path(existing_intent["source_path"])
                accepted = Path(existing_intent["accepted_path"])
            else:
                source = Path(candidate.candidate_path)
                accepted = self.runtime.accepted_models / source.name
            artifact = (
                inspect_candidate(source)
                if source.exists()
                else inspect_candidate(accepted)
            )
            if artifact.model_hash != candidate_hash:
                raise SafetyHalt("candidate files contradict registry identity")
            self._require_disk(artifact.size_bytes)
            intent = {
                "schema_version": 1,
                "generation_id": generation_id,
                "candidate_hash": candidate_hash,
                "candidate_name": artifact.name,
                "source_path": str(source),
                "accepted_path": str(accepted),
                "manifest_hash": artifact.directory_manifest_hash,
                "tested_champion_hash": report.tested_champion_hash,
                "evaluation_key": report.evaluation_key,
                "config_hash": report.config_hash,
                "schedule_hash": report.schedule_hash,
                "policy_hash": report.policy_hash,
                "selfplay_config_hash":
                    self.runtime.controller.selfplay_config_hash,
                "topology": "7-workers-100-threads",
                "gpu_handoff_hash": report_value["gpu_handoff_hash"],
                "pass_report_path": str(Path(pass_report_path)),
                "pass_report_hash": pass_report_hash,
                "trainer_checkpoint_hash": trainer_checkpoint_hash,
                "data_watermark_hash": data_watermark_hash,
                "shuffle_watermark_hash": shuffle_watermark_hash,
                "data_watermark_path": str(self.runtime.data_watermark_path),
                "shuffle_watermark_path": str(self.runtime.shuffle_watermark_path),
                "derived_shuffle_paths": authoritative_shuffle_paths,
            }
            _write_immutable_json(transaction / "intent.json", intent)
            if not (transaction / "previous-champion.json").exists():
                _write_immutable_json(
                    transaction / "previous-champion.json",
                    load_champion(self.runtime.champion_path).to_dict(),
                )
            self._snapshot_checkpoint(generation_id, trainer_checkpoint_hash)
            self._checkpoint("promotion-intent-written")
            if generation is not None and generation.state == GenerationState.ACTIVE:
                if load_champion(self.runtime.champion_path).champion_hash != candidate_hash:
                    raise SafetyHalt("active generation contradicts champion projection")
                self._mark(transaction, "complete", {"champion_hash": candidate_hash})
                return {
                    "status": "ACTIVE",
                    "generation_id": generation_id,
                    "candidate_hash": candidate_hash,
                }
            generation = self.registry.reconstruct().generations.get(generation_id)
            if generation is None:
                self.registry.transition_generation(
                    generation_id,
                    candidate_hash,
                    str(accepted),
                    GenerationState.PROMOTION_INTENT,
                    provenance=provenance,
                    tested_champion_hash=report.tested_champion_hash,
                    evaluation_key=report.evaluation_key,
                    reason="finalized PASS promotion intent",
                    actor=self.runtime.controller.actor,
                    payload={"intent_hash": canonical_sha256(intent)},
                )
            self._checkpoint("promotion-intent-event")
            pins = (
                ("previous-champion", report.tested_champion_hash, "rollback-champion"),
                ("trainer-checkpoint", trainer_checkpoint_hash, "trainer-recovery"),
                ("data-watermark", data_watermark_hash, "data-watermark"),
                ("shuffle-watermark", shuffle_watermark_hash, "shuffle-watermark"),
            )
            for suffix, reference, kind in pins:
                self.registry.pin_reference(
                    f"{generation_id}:{suffix}",
                    reference,
                    kind=kind,
                    owner=generation_id,
                    provenance=provenance,
                    champion_hash=state.current_champion_hash,
                    reason=f"promotion rollback pin {suffix}",
                    actor=self.runtime.controller.actor,
                )
            self._checkpoint("promotion-pins-written")
            try:
                _recoverable_rename(
                    source,
                    accepted,
                    expected_manifest_hash=artifact.directory_manifest_hash,
                    inspector=self._candidate_manifest_hash,
                )
            except SafetyHalt:
                if accepted.exists():
                    self._quarantine_contradiction(
                        accepted, "accepted-destination-manifest-mismatch"
                    )
                raise
            self._mark(transaction, "accepted", {"manifest_hash": artifact.directory_manifest_hash})
            self._checkpoint("promotion-candidate-accepted")
            self._generation_leaf(generation_id, accepted, candidate_hash)
            self._mark(transaction, "generation-leaf", {"model_hash": candidate_hash})
            self._checkpoint("promotion-generation-leaf")
            self._stage_workers(generation_id, candidate_hash)
            self._mark(transaction, "workers-staged", {"worker_count": self.runtime.controller.worker_count})
            self._checkpoint("promotion-workers-staged")
            self._launch_worker_phase(
                generation_id,
                candidate_hash,
                phase="canary",
                start=0,
                stop=self.runtime.controller.canary_worker_count,
            )
            generation = self.registry.reconstruct().generations[generation_id]
            if generation.state == GenerationState.PROMOTION_INTENT:
                self.registry.transition_generation(
                    generation_id,
                    candidate_hash,
                    str(accepted),
                    GenerationState.CANARY,
                    provenance=provenance,
                    tested_champion_hash=report.tested_champion_hash,
                    reason="isolated canary workers staged",
                    actor=self.runtime.controller.actor,
                )
            elif generation.state not in {
                GenerationState.CANARY,
                GenerationState.ROLLOUT,
            }:
                raise SafetyHalt(
                    f"cannot resume promotion from {generation.state.value}"
                )
            self._checkpoint("promotion-canary-event")
            acknowledged = self._acknowledged(generation_id, candidate_hash)
            canary_workers = set(range(self.runtime.controller.canary_worker_count))
            if not canary_workers.issubset(acknowledged):
                return {"status": "WAITING_CANARY_ACK", "acknowledged": sorted(acknowledged)}
            if not self._admit_canary(generation_id, candidate_hash):
                return {"status": "WAITING_CANARY_ADMISSION", "acknowledged": sorted(acknowledged)}
            self._checkpoint("promotion-canary-admitted")
            generation = self.registry.reconstruct().generations[generation_id]
            if generation.state == GenerationState.CANARY:
                self.registry.transition_generation(
                    generation_id,
                    candidate_hash,
                    str(accepted),
                    GenerationState.ROLLOUT,
                    provenance=provenance,
                    tested_champion_hash=report.tested_champion_hash,
                    reason="canary data admitted; full rollout staged",
                    actor=self.runtime.controller.actor,
                )
            elif generation.state != GenerationState.ROLLOUT:
                raise SafetyHalt(
                    f"cannot resume rollout from {generation.state.value}"
                )
            self._checkpoint("promotion-rollout-event")
            self._launch_worker_phase(
                generation_id,
                candidate_hash,
                phase="intermediate",
                start=self.runtime.controller.canary_worker_count,
                stop=self.runtime.controller.intermediate_worker_count,
            )
            acknowledged = self._acknowledged(generation_id, candidate_hash)
            intermediate = set(
                range(self.runtime.controller.intermediate_worker_count)
            )
            if not intermediate.issubset(acknowledged):
                return {
                    "status": "WAITING_INTERMEDIATE_ACK",
                    "acknowledged": sorted(acknowledged),
                }
            intermediate_pass = transaction / "intermediate-pass.json"
            if not intermediate_pass.exists():
                return {
                    "status": "WAITING_INTERMEDIATE_HEALTH",
                    "acknowledged": sorted(acknowledged),
                }
            intermediate_value = json.loads(
                intermediate_pass.read_text(encoding="utf-8")
            )
            if (
                intermediate_value.get("decision") != "PASS"
                or intermediate_value.get("generation_id") != generation_id
                or intermediate_value.get("candidate_hash") != candidate_hash
            ):
                raise SafetyHalt("intermediate PASS marker contradicts promotion")
            self._checkpoint("promotion-intermediate-passed")
            self._launch_worker_phase(
                generation_id,
                candidate_hash,
                phase="full",
                start=self.runtime.controller.intermediate_worker_count,
                stop=self.runtime.controller.worker_count,
            )
            acknowledged = self._acknowledged(generation_id, candidate_hash)
            required = set(range(self.runtime.controller.worker_count))
            if not required.issubset(acknowledged):
                return {"status": "WAITING_ROLLOUT_ACK", "acknowledged": sorted(acknowledged)}
            self._checkpoint("promotion-all-workers-acknowledged")
            compare_and_swap_champion(
                self.runtime.champion_path,
                expected_champion_hash=report.tested_champion_hash,
                candidate_hash=candidate_hash,
                generation_id=generation_id,
                pass_report_path=pass_report_path,
                pass_report_hash=pass_report_hash,
                evaluation_key=report.evaluation_key,
                provenance=provenance,
                actor=self.runtime.controller.actor,
            )
            self._mark(transaction, "champion-cas", {"champion_hash": candidate_hash})
            self._checkpoint("promotion-champion-cas")
            if not self._commit_generation_data(generation_id, candidate_hash):
                return {
                    "status": "WAITING_GENERATION_DATA",
                    "acknowledged": sorted(acknowledged),
                }
            self._checkpoint("promotion-generation-data-admitted")
            generation = self.registry.reconstruct().generations[generation_id]
            if generation.state == GenerationState.ROLLOUT:
                self.registry.transition_generation(
                    generation_id,
                    candidate_hash,
                    str(accepted),
                    GenerationState.ACTIVE,
                    provenance=provenance,
                    tested_champion_hash=report.tested_champion_hash,
                    reason="all workers acknowledged and champion CAS committed",
                    actor=self.runtime.controller.actor,
                )
            elif generation.state != GenerationState.ACTIVE:
                raise SafetyHalt(
                    f"cannot commit activation from {generation.state.value}"
                )
            self._mark(transaction, "complete", {"champion_hash": candidate_hash})
            self._checkpoint("promotion-active-event")
            return {"status": "ACTIVE", "generation_id": generation_id, "candidate_hash": candidate_hash}

    def rollback(
        self,
        generation_id: str,
        *,
        derived_shuffle_paths: Sequence[Path] = (),
        trainer_consumed: bool = False,
    ) -> Mapping[str, Any]:
        """Quarantine generation data and restore durable pre-promotion state."""

        if not self.automatic:
            raise SafetyHalt("rollback requires automatic mutation mode")
        with self._writer_lock():
            transaction = self._transaction_dir(generation_id)
            intent_path = transaction / "intent.json"
            if not intent_path.exists():
                raise SafetyHalt(f"promotion intent is missing for {generation_id}")
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            state = self.registry.reconstruct()
            generation = state.generations.get(generation_id)
            if generation is None:
                raise SafetyHalt(f"unknown generation: {generation_id}")
            if derived_shuffle_paths:
                raise SafetyHalt(
                    "rollback paths must come from the frozen promotion lineage"
                )
            if generation.state == GenerationState.ROLLED_BACK:
                return {"status": "ROLLED_BACK", "generation_id": generation_id}
            if generation.state not in {
                GenerationState.ACTIVE,
                GenerationState.ROLLBACK_PENDING,
            }:
                raise SafetyHalt(
                    "historical or non-active rollback requires forensic flow"
                )
            if generation.state == GenerationState.ACTIVE and (
                state.current_generation_id != generation_id
                or state.current_champion_hash != generation.candidate_hash
                or load_champion(self.runtime.champion_path).champion_hash
                != generation.candidate_hash
            ):
                raise SafetyHalt(
                    "rollback target is not the current generation/champion"
                )
            provenance = self._provenance(intent["config_hash"], intent["schedule_hash"])
            quarantine = self.runtime.rollback_quarantine / generation_id / "data"
            staged = self.runtime.rollout_quarantine / generation_id
            admitted = self.runtime.admitted_selfplay / generation_id
            requested_moves = [
                ("staged-rollout", staged, quarantine / "staged-rollout"),
                ("admitted-generation", admitted, quarantine / "admitted-generation"),
            ]
            for index, item in enumerate(intent.get("derived_shuffle_paths", [])):
                source = Path(item)
                requested_moves.append(
                    (
                        f"shuffle-{index:03d}",
                        source,
                        quarantine / f"shuffle-{index:03d}",
                    )
                )
            rollback_intent_path = transaction / "rollback-intent.json"
            if rollback_intent_path.exists():
                rollback_intent = json.loads(
                    rollback_intent_path.read_text(encoding="utf-8")
                )
            else:
                move_intents = []
                for label, source, destination in requested_moves:
                    existing_path = source if source.exists() else destination
                    if existing_path.exists():
                        manifest, _ = _tree_manifest(existing_path)
                        move_intents.append(
                            {
                                "label": label,
                                "source": str(source),
                                "destination": str(destination),
                                "manifest_hash": manifest,
                            }
                        )
                rollback_intent = {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "candidate_hash": generation.candidate_hash,
                    "previous_champion_hash": generation.previous_champion_hash,
                    "trainer_checkpoint_hash": intent["trainer_checkpoint_hash"],
                    "data_watermark_hash": intent["data_watermark_hash"],
                    "shuffle_watermark_hash": intent["shuffle_watermark_hash"],
                    "trainer_consumed": trainer_consumed,
                    "moves": move_intents,
                }
                _write_immutable_json(rollback_intent_path, rollback_intent)
            if generation.state == GenerationState.ACTIVE:
                self.registry.transition_generation(
                    generation_id,
                    generation.candidate_hash,
                    generation.candidate_path,
                    GenerationState.ROLLBACK_PENDING,
                    provenance=provenance,
                    tested_champion_hash=generation.previous_champion_hash,
                    restore_champion_hash=generation.previous_champion_hash,
                    reason="rollback transaction started",
                    actor=self.runtime.controller.actor,
                )
            stop_result = self.execute_argv(
                "rollback",
                {
                    "generation_id": generation_id,
                    "model_hash": generation.candidate_hash,
                },
            )
            required_roles = [
                "selfplay",
                "shuffler",
                "trainer",
                "exporter",
                "evaluator",
            ]
            if (
                not isinstance(stop_result, Mapping)
                or stop_result.get("quiescent") is not True
                or stop_result.get("quiescent_roles") != required_roles
            ):
                raise SafetyHalt("rollback stop command lacks all-role quiescence proof")
            self._mark(
                transaction,
                "rollback-command",
                {"generation_id": generation_id, "status": "completed"},
            )
            quarantine.mkdir(parents=True, exist_ok=True)
            for move in rollback_intent["moves"]:
                _recoverable_rename(
                    Path(move["source"]),
                    Path(move["destination"]),
                    expected_manifest_hash=move["manifest_hash"],
                    inspector=lambda path: _tree_manifest(path)[0],
                )
            self._checkpoint("rollback-data-quarantined")
            effective_trainer_consumed = rollback_intent["trainer_consumed"]
            if effective_trainer_consumed:
                snapshot = self.runtime.rollback_quarantine / generation_id / "trainer-checkpoint"
                expected = intent["trainer_checkpoint_hash"]
                if sha256_file(snapshot) != expected:
                    raise SafetyHalt("rollback checkpoint snapshot is corrupt")
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{self.runtime.trainer_checkpoint.name}.",
                    suffix=".tmp",
                    dir=str(self.runtime.trainer_checkpoint.parent),
                )
                temporary = Path(name)
                try:
                    with os.fdopen(descriptor, "wb") as output, snapshot.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
                        output.flush()
                        os.fsync(output.fileno())
                    if sha256_file(temporary) != expected:
                        raise SafetyHalt("restored checkpoint hash mismatch")
                    os.replace(temporary, self.runtime.trainer_checkpoint)
                    fsync_directory(self.runtime.trainer_checkpoint.parent)
                finally:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
                self._checkpoint("rollback-checkpoint-restored")
            previous_snapshot = transaction / "previous-champion.json"
            if self.runtime.champion_path.exists():
                champion = load_champion(self.runtime.champion_path)
                if champion.champion_hash == generation.candidate_hash:
                    atomic_write_bytes(
                        self.runtime.champion_path, previous_snapshot.read_bytes()
                    )
                    self._checkpoint("rollback-champion-restored")
                elif champion.champion_hash != generation.previous_champion_hash:
                    raise SafetyHalt("champion projection contradicts rollback target")
            self.registry.transition_generation(
                generation_id,
                generation.candidate_hash,
                generation.candidate_path,
                GenerationState.ROLLED_BACK,
                provenance=provenance,
                tested_champion_hash=generation.previous_champion_hash,
                restore_champion_hash=generation.previous_champion_hash,
                reason="rollback filesystem transaction completed",
                actor=self.runtime.controller.actor,
            )
            self._mark(
                transaction,
                "rolled-back",
                {"trainer_consumed": effective_trainer_consumed},
            )
            return {"status": "ROLLED_BACK", "generation_id": generation_id}

    def execute_argv(self, template_name: str, substitutions: Mapping[str, Any]) -> Any:
        """Expand a configured argv template and invoke the injected executor."""

        if template_name not in self.runtime.commands:
            raise ConfigurationError(f"unknown command template: {template_name}")
        try:
            argv = tuple(
                argument.format_map(
                    {key: str(value) for key, value in substitutions.items()}
                )
                for argument in self.runtime.commands[template_name]
            )
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(
                f"cannot expand command template {template_name}: {exc}"
            ) from exc
        if self.recommendation_only:
            return {"argv": list(argv), "executed": False}
        if self.command_executor is None:
            raise SafetyHalt(
                f"automatic mode has no executor for command {template_name}"
            )
        result = self.command_executor(argv)
        returncode = (
            result.get("returncode")
            if isinstance(result, Mapping)
            else getattr(result, "returncode", None)
        )
        if type(returncode) is not int:
            raise SafetyHalt(
                f"command executor returned no integer status for {template_name}"
            )
        if returncode != 0:
            raise SafetyHalt(
                f"command {template_name} failed with status {returncode}"
            )
        return result


def configured_gate_evaluator(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a configured adapter envelope and run the in-repo final gate."""

    stage = evidence.get("controller_stage")
    if stage == "confirmation":
        promotion_evidence = evidence.get("promotion_evidence")
        if not isinstance(promotion_evidence, Mapping):
            raise SafetyHalt(
                "confirmation adapter evidence must contain promotion_evidence"
            )
        from risk_score.promotion_gate import evaluate_promotion_gate

        result = dict(evaluate_promotion_gate(
            promotion_evidence,
            expected_policy_hash=evidence["policy_hash"],
            expected_candidate_hash=evidence["candidate_hash"],
            expected_champion_hash=evidence["tested_champion_hash"],
            expected_original_hash=evidence["original_hash"],
        ))
        result.update(
            {
                "gpu_handoff_hash": evidence["gpu_handoff_hash"],
                "selfplay_config_hash": evidence["selfplay_config_hash"],
                "topology": evidence["topology"],
            }
        )
        return result

    stage_gate = evidence.get("stage_gate")
    if not isinstance(stage_gate, Mapping):
        raise SafetyHalt("non-confirmation adapter evidence must contain stage_gate")
    decision = stage_gate.get("decision")
    if decision not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        raise SafetyHalt(f"configured stage gate has invalid decision: {decision!r}")
    expected = {
        "finalized": True,
        "candidate_hash": evidence["candidate_hash"],
        "tested_champion_hash": evidence["tested_champion_hash"],
        "original_hash": evidence["original_hash"],
        "evaluation_key": evidence["evaluation_key"],
        "config_hash": evidence["config_hash"],
        "schedule_hash": evidence["schedule_hash"],
        "policy_hash": evidence["policy_hash"],
        "selfplay_config_hash": evidence["selfplay_config_hash"],
        "topology": evidence["topology"],
    }
    conflicts = [
        key for key, value in expected.items() if stage_gate.get(key) != value
    ]
    if conflicts:
        raise SafetyHalt(
            "configured stage gate contradicts evaluator evidence: "
            + ", ".join(sorted(conflicts))
        )
    return {
        **dict(stage_gate),
        "gpu_handoff_hash": evidence["gpu_handoff_hash"],
        "selfplay_config_hash": evidence["selfplay_config_hash"],
        "topology": evidence["topology"],
    }


@contextlib.contextmanager
def configured_gpu_lease_factory(
    config_path: Path,
    plan: EvaluationMatrixPlan,
    candidate: CandidateArtifact,
) -> Iterable[Mapping[str, Any]]:
    """Adapt gpu_lease's exclusive handoff into controller provenance."""

    from risk_score.gpu_lease import GpuLeaseManager, RuntimeConfig as GpuRuntimeConfig

    config = GpuRuntimeConfig.from_json_file(config_path)
    manager = GpuLeaseManager(config)
    proof: Dict[str, Any] = {}
    with manager.exclusive_handoff() as record:
        proof.update(
            {
                "lease_id": record.lease_id,
                "expected_gpu_uuid": record.expected_gpu_uuid,
                "handoff_checkpoint_hash": (
                    None
                    if record.handoff_checkpoint is None
                    else record.handoff_checkpoint.sha256
                ),
                "clean_observations": [
                    {
                        "gpu_uuid": record.expected_gpu_uuid,
                        "processes": [],
                        "observed_at": observed_at,
                    }
                    for observed_at in record.lease_clean_observation_times
                ],
                "trainer_restored": False,
                "restored_trainer_identity": {},
                "evaluation_key": plan.evaluation_key,
                "candidate_hash": candidate.model_hash,
            }
        )
        yield proof
    final = manager.read_record()
    if final is None:
        raise SafetyHalt("GPU handoff state disappeared during restoration")
    restored = final.restored_trainer or final.trainer
    proof.update(
        {
            "handoff_checkpoint_hash": (
                None
                if final.handoff_checkpoint is None
                else final.handoff_checkpoint.sha256
            ),
            "trainer_restored": final.restoration_status
            in {"restored", "not_needed"},
            "restored_trainer_identity": (
                {} if restored is None else restored.to_dict()
            ),
            "release_clean_observation_count":
                final.release_clean_observation_count,
        }
    )


def default_command_executor(argv: Sequence[str]) -> Any:
    """Run argv-only command and decode an optional supervisor JSON proof."""

    result = subprocess.run(
        list(argv), check=False, capture_output=True, text=True, shell=False
    )
    if result.stdout.strip():
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError:
            return result
        if isinstance(value, dict):
            return {
                **value,
                "returncode": result.returncode,
                "stderr": result.stderr,
            }
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--mode", choices=("once", "reconcile", "watch"), default="once")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recommend-only", action="store_true")
    mode.add_argument("--automatic", action="store_true")
    parser.add_argument("--interval", type=float)
    parser.add_argument("--bootstrap-champion", action="store_true")
    parser.add_argument("--bootstrap-champion-hash")
    parser.add_argument("--bootstrap-generation-id")
    parser.add_argument("--bootstrap-confirmation")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    instance_lock: Optional[ControllerLock] = None
    try:
        runtime = RuntimeConfig.load(args.runtime_config)
        if args.automatic and not runtime.controller.mutation_enabled:
            raise ConfigurationError(
                "--automatic requires mutationEnabled=true in runtime config"
            )
        if args.automatic:
            missing_commands = [
                name
                for name in ("evaluator", "selfplay", "drain", "rollback")
                if not runtime.commands[name]
            ]
            if missing_commands:
                raise ConfigurationError(
                    "automatic mode requires nonempty command templates: "
                    + ", ".join(missing_commands)
                )
        if args.interval is not None and args.interval <= 0:
            raise ConfigurationError("--interval must be positive")
        if args.automatic:
            instance_lock = ControllerLock(
                runtime.lock_path, owner=runtime.controller.actor
            ).acquire()
        controller = PromotionController(
            runtime,
            automatic=args.automatic,
            gate_evaluator=configured_gate_evaluator if args.automatic else None,
            command_executor=default_command_executor if args.automatic else None,
            gpu_lease_factory=(
                configured_gpu_lease_factory if args.automatic else None
            ),
            held_controller_lock=instance_lock,
        )
        if args.automatic:
            controller.evaluation_executor = (
                controller.configured_evaluation_executor
            )
        if args.bootstrap_champion:
            if not args.bootstrap_champion_hash or not args.bootstrap_generation_id:
                raise ConfigurationError(
                    "bootstrap requires --bootstrap-champion-hash and "
                    "--bootstrap-generation-id"
                )
            controller.bootstrap(
                args.bootstrap_champion_hash,
                args.bootstrap_generation_id,
                confirmation=args.bootstrap_confirmation or "",
            )
        interval = args.interval or runtime.controller.poll_interval_seconds
        while True:
            if args.mode == "reconcile":
                result = controller.run_reconcile()
            else:
                result = controller.run_once()
            print(json.dumps(result, sort_keys=True))
            if args.mode != "watch":
                break
            time.sleep(interval)
        return 0
    except (
        OSError,
        ValueError,
        ControllerError,
        RegistryCorruptionError,
        ChampionConflictError,
        StaleChampionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if instance_lock is not None:
            instance_lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
