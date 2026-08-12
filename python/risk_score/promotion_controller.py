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
import fcntl
import functools
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from risk_score.hardened_exporter import EXPORT_CONTRACT
from risk_score.adaptive_training import (
    POLICY_HASH as ADAPTIVE_POLICY_HASH,
    AdaptiveTrainingError,
    compare_and_swap_recipe_binding,
    load_candidate_handoff,
    load_recipe_binding,
    rollback_recipe_binding,
)
from risk_score.suite_rotation import (
    ACTIVE_SUITE_CONTRACT,
    SuiteRotationError,
    SuiteRotationRegistry,
    load_registry_spec,
)


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COUNTER_RE = re.compile(r"(?:^|-)s([0-9]+)-d([0-9]+)(?:-|$)")
_IGNORED_SUFFIXES = (".tmp", ".partial", ".exported")
_CANDIDATE_FILES = {"model.bin.gz", "model.ckpt"}
_OPTIONAL_CANDIDATE_FILES = {
    "manifest.json",
    "exporter_manifest.json",
    "metadata.json",
    "log.txt",
}
_HARDENED_EXPORT_CONTRACTS = frozenset(
    {
        "katago-hardened-candidate-publication-v1",
        EXPORT_CONTRACT,
    }
)
EVALUATION_PLAN_CONTRACT = "risk-score-evaluation-plan-v2"
PROMOTION_READY_POLICY_VERSION = "risk-seeking-checkpoint-promotion-v3"
PROMOTION_READY_SUITE_CONTRACT = "risk-score-authoritative-evaluation-manifest-v3"
PROMOTION_READY_BANK_CONTRACT = "risk-score-reviewed-position-bank-v2"
PROMOTION_READY_REVIEW_MODE = "machine-consensus"
PROMOTION_READY_LABELS = ("lead-40", "lead-80", "ordinary")

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
    "promotion-active-event",
    "promotion-generation-data-admitted",
)
ADAPTIVE_PROMOTION_FAILURE_STEPS = (
    "promotion-training-handoff-intent",
    "promotion-training-quiesced",
    "promotion-training-checkpoint-installed",
    "promotion-training-recipe-cas",
    "promotion-champion-cas",
    "promotion-training-commit",
)
ADAPTIVE_ROLLBACK_FAILURE_STEPS = (
    "rollback-training-checkpoint-restored",
    "rollback-training-recipe-restored",
    "rollback-champion-restored",
)
_ALL_ROLE_QUIESCENCE = [
    "selfplay",
    "shuffler",
    "trainer",
    "exporter",
    "evaluator",
]


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
class RuntimeAutonomyBinding:
    """Immutable control-plane bindings carried by runtime schema v2."""

    suite_registry_spec_path: Path
    suite_registry_spec_file_hash: str
    suite_registry_spec_identity: str
    active_suite_pointer_path: Path
    active_suite_pointer_file_hash: str
    active_suite_pointer_record_hash: str
    active_suite_id: str
    adaptive_policy_hash: str
    adaptive_root: Path


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
    schema_version: int = 1
    autonomy: Optional[RuntimeAutonomyBinding] = None

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
        if not isinstance(value, dict):
            raise ConfigurationError("runtime config must be an object")
        schema_version = value.get("schemaVersion")
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise ConfigurationError("schemaVersion must be integer 1 or 2")
        root_keys = {
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
        }
        if schema_version == 2 and "autonomy" in value:
            root_keys.add("autonomy")
        root = _strict_object(
            value,
            "runtime config",
            root_keys,
        )
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
            {
                "trainer",
                "stage0Probe",
                "evaluator",
                "selfplay",
                "drain",
                "rollback",
            },
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
        autonomy = (
            _parse_runtime_autonomy(
                root["autonomy"],
                paths=normalized_paths,
                hashes=hashes,
            )
            if schema_version == 2 and "autonomy" in root
            else None
        )

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
            schema_version=schema_version,
            autonomy=autonomy,
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
    candidate_hash: str
    champion_hash: str
    original_hash: str
    evaluation_key: str
    config_hash: str
    schedule_hash: str
    policy_hash: str
    selfplay_config_hash: str
    topology: str
    stage: str
    look: str
    policy_path: str
    policy_version: str
    suite_manifest_path: str
    suite_manifest_hash: str
    schedule_artifacts: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planContract": EVALUATION_PLAN_CONTRACT,
            "candidateModelSha256": self.candidate_hash,
            "championModelSha256": self.champion_hash,
            "originalModelSha256": self.original_hash,
            "evaluationKey": self.evaluation_key,
            "configHash": self.config_hash,
            "scheduleHash": self.schedule_hash,
            "policyHash": self.policy_hash,
            "policyPath": self.policy_path,
            "policyVersion": self.policy_version,
            "selfplayConfigHash": self.selfplay_config_hash,
            "topology": self.topology,
            "stage": self.stage,
            "look": self.look,
            "suiteManifestPath": self.suite_manifest_path,
            "suiteManifestHash": self.suite_manifest_hash,
            "scheduleArtifacts": {
                key: dict(value)
                for key, value in sorted(self.schedule_artifacts.items())
            },
            "cellOrder": list(self.schedule_artifacts),
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


def _canonical_confirmation_look(look: str) -> str:
    aliases = {
        "1": "look-1",
        "automatic": "look-1",
        "final": "look-1",
        "fresh": "look-1",
        "look-1": "look-1",
        "prespecified-first-look": "look-1",
        "2": "look-2",
        "look-2": "look-2",
        "prespecified-second-look": "look-2",
    }
    try:
        return aliases[look]
    except (KeyError, TypeError) as exc:
        raise SafetyHalt(
            "confirmation look must be canonical look-1 or look-2"
        ) from exc


def _policy_get(
    policy: Mapping[str, Any], *parts: str, default: Any = None
) -> Any:
    current: Any = policy
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _validate_queue_contract(
    controller: ControllerConfig, policy: Mapping[str, Any]
) -> None:
    if policy.get("schema_version") not in {2, 3}:
        return
    queue_policy = policy.get("queue")
    if not isinstance(queue_policy, Mapping):
        raise SafetyHalt("frozen policy has no queue contract")
    screen_interval = queue_policy.get("screen_interval_new_training_samples")
    confirmation_interval = queue_policy.get(
        "confirmation_interval_new_training_samples"
    )
    policy_queue_limit = queue_policy.get("maximum_active_evaluator_entries")
    if (
        type(screen_interval) is not int
        or screen_interval <= 0
        or type(confirmation_interval) is not int
        or confirmation_interval < screen_interval
        or confirmation_interval % screen_interval != 0
        or type(policy_queue_limit) is not int
        or policy_queue_limit <= 0
        or controller.anchor_interval_samples != screen_interval
        or controller.max_active_queue != policy_queue_limit
        or queue_policy.get("coalesce_newer_before_screening") is not True
        or queue_policy.get("never_replace_started_candidate") is not True
    ):
        raise SafetyHalt("runtime and frozen policy queue contracts disagree")


def _parse_utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be UTC and end in Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


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


def _runtime_autonomy_values(value: Any) -> Mapping[str, Any]:
    """Normalize the two strict v2 serialization forms used by runtime builders."""

    if not isinstance(value, dict):
        raise ConfigurationError("autonomy must be an object")
    if set(value) == {
        "suiteRegistrySpec",
        "activeSuitePointer",
        "adaptive",
    }:
        registry = value["suiteRegistrySpec"]
        active = value["activeSuitePointer"]
        adaptive = value["adaptive"]
        if not isinstance(registry, dict):
            raise ConfigurationError("autonomy.suiteRegistrySpec must be an object")
        if set(registry) == {"path", "sha256", "identity"}:
            registry_hash = registry["sha256"]
        elif set(registry) == {"path", "fileSha256", "identity"}:
            registry_hash = registry["fileSha256"]
        else:
            _strict_object(
                registry,
                "autonomy.suiteRegistrySpec",
                {"path", "sha256", "identity"},
            )
            raise AssertionError("unreachable")
        if not isinstance(active, dict):
            raise ConfigurationError("autonomy.activeSuitePointer must be an object")
        if set(active) == {"path", "sha256", "recordSha256", "suiteId"}:
            active_hash = active["sha256"]
        elif set(active) == {
            "path",
            "fileSha256",
            "recordSha256",
            "suiteId",
        }:
            active_hash = active["fileSha256"]
        else:
            _strict_object(
                active,
                "autonomy.activeSuitePointer",
                {"path", "sha256", "recordSha256", "suiteId"},
            )
            raise AssertionError("unreachable")
        if not isinstance(adaptive, dict):
            raise ConfigurationError("autonomy.adaptive must be an object")
        if set(adaptive) == {"policySha256", "root"}:
            adaptive_hash = adaptive["policySha256"]
        elif set(adaptive) == {"policyHash", "root"}:
            adaptive_hash = adaptive["policyHash"]
        else:
            _strict_object(
                adaptive,
                "autonomy.adaptive",
                {"policySha256", "root"},
            )
            raise AssertionError("unreachable")
        return {
            "suite_registry_spec_path": registry["path"],
            "suite_registry_spec_file_hash": registry_hash,
            "suite_registry_spec_identity": registry["identity"],
            "active_suite_pointer_path": active["path"],
            "active_suite_pointer_file_hash": active_hash,
            "active_suite_pointer_record_hash": active["recordSha256"],
            "active_suite_id": active["suiteId"],
            "adaptive_policy_hash": adaptive_hash,
            "adaptive_root": adaptive["root"],
        }

    flat_variants = (
        {
            "suiteRegistrySpecPath": "suite_registry_spec_path",
            "suiteRegistrySpecSha256": "suite_registry_spec_file_hash",
            "suiteRegistrySpecIdentity": "suite_registry_spec_identity",
            "activeSuitePointerPath": "active_suite_pointer_path",
            "activeSuitePointerSha256": "active_suite_pointer_file_hash",
            "activeSuitePointerRecordSha256": "active_suite_pointer_record_hash",
            "activeSuiteId": "active_suite_id",
            "adaptivePolicySha256": "adaptive_policy_hash",
            "adaptiveRoot": "adaptive_root",
        },
        {
            "suiteRegistrySpecPath": "suite_registry_spec_path",
            "suiteRegistrySpecFileSha256": "suite_registry_spec_file_hash",
            "suiteRegistrySpecIdentity": "suite_registry_spec_identity",
            "activeSuitePointerPath": "active_suite_pointer_path",
            "activeSuitePointerFileSha256": "active_suite_pointer_file_hash",
            "activeSuitePointerRecordSha256": "active_suite_pointer_record_hash",
            "activeSuiteId": "active_suite_id",
            "adaptivePolicyHash": "adaptive_policy_hash",
            "adaptiveRoot": "adaptive_root",
        },
    )
    for fields in flat_variants:
        if set(value) == set(fields):
            return {normalized: value[key] for key, normalized in fields.items()}
    _strict_object(
        value,
        "autonomy",
        {"suiteRegistrySpec", "activeSuitePointer", "adaptive"},
    )
    raise AssertionError("unreachable")


def _parse_runtime_autonomy(
    value: Any,
    *,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
) -> RuntimeAutonomyBinding:
    raw = _runtime_autonomy_values(value)
    spec_path = _absolute_path(
        raw["suite_registry_spec_path"],
        "autonomy suite registry specification path",
    )
    spec_file_hash = _hash(
        raw["suite_registry_spec_file_hash"],
        "autonomy suite registry specification file hash",
    )
    spec_identity = _hash(
        raw["suite_registry_spec_identity"],
        "autonomy suite registry specification identity",
    )
    pointer_path = _absolute_path(
        raw["active_suite_pointer_path"],
        "autonomy active-suite pointer path",
    )
    pointer_file_hash = _hash(
        raw["active_suite_pointer_file_hash"],
        "autonomy active-suite pointer file hash",
    )
    pointer_record_hash = _hash(
        raw["active_suite_pointer_record_hash"],
        "autonomy active-suite pointer record hash",
    )
    active_suite_id = _hash(
        raw["active_suite_id"],
        "autonomy active suite ID",
    )
    adaptive_policy_hash = _hash(
        raw["adaptive_policy_hash"],
        "autonomy adaptive policy hash",
    )
    adaptive_root = _absolute_path(
        raw["adaptive_root"],
        "autonomy adaptive root",
    )

    for path, expected, role in (
        (spec_path, spec_file_hash, "suite registry specification"),
        (pointer_path, pointer_file_hash, "active-suite pointer"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ConfigurationError(f"autonomy {role} is not a regular file: {path}")
        if sha256_file(path) != expected:
            raise ConfigurationError(f"autonomy {role} file hash mismatch")
    try:
        spec = load_registry_spec(spec_path)
    except (OSError, ValueError, SuiteRotationError) as exc:
        raise ConfigurationError(f"invalid autonomy suite registry spec: {exc}") from exc
    if (
        spec.file_sha256 != spec_file_hash
        or spec.identity != spec_identity
        or spec.policy_path != paths["policy"]
        or spec.policy_identity != hashes["policy"]
        or spec.original.path != paths["originalModel"]
        or spec.original.sha256 != hashes["original"]
    ):
        raise ConfigurationError(
            "autonomy suite registry specification contradicts frozen runtime"
        )
    if pointer_path != spec.root / "active-suite.json":
        raise ConfigurationError(
            "autonomy active-suite pointer is not owned by the suite registry"
        )

    try:
        pointer_data = pointer_path.read_bytes()
        pointer = json.loads(pointer_data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load autonomy active-suite pointer: {exc}") from exc
    pointer = _strict_object(
        pointer,
        "autonomy active-suite pointer",
        {
            "schema_version",
            "contract",
            "spec_sha256",
            "suite_id",
            "version_sha256",
            "manifest_path",
            "manifest_sha256",
            "manifest_identity",
            "activated_at_utc",
            "activation_champion_sha256",
            "activation_generation_id",
            "event_sequence",
            "event_sha256",
            "record_sha256",
        },
    )
    if pointer_data != canonical_json_bytes(pointer) + b"\n":
        raise ConfigurationError("autonomy active-suite pointer is not canonical")
    pointer_payload = dict(pointer)
    supplied_record_hash = pointer_payload.pop("record_sha256")
    if (
        pointer["schema_version"] != 1
        or pointer["contract"] != ACTIVE_SUITE_CONTRACT
        or pointer["spec_sha256"] != spec_identity
        or pointer["suite_id"] != active_suite_id
        or pointer["manifest_sha256"] != active_suite_id
        or supplied_record_hash != pointer_record_hash
        or supplied_record_hash != canonical_sha256(pointer_payload)
    ):
        raise ConfigurationError("autonomy active-suite pointer identity is invalid")
    manifest_path = _absolute_path(
        pointer["manifest_path"],
        "autonomy active suite manifest path",
    )
    if (
        manifest_path != paths["suites"] / "manifest.json"
        or paths["suites"].name != active_suite_id
        or pointer["manifest_sha256"] != hashes["suiteManifest"]
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != hashes["suiteManifest"]
    ):
        raise ConfigurationError(
            "autonomy active-suite pointer diverges from frozen runtime suite"
        )
    if adaptive_policy_hash != ADAPTIVE_POLICY_HASH:
        raise ConfigurationError("autonomy adaptive policy hash is not frozen")
    if adaptive_root != paths["promotionRoot"] / "adaptive":
        raise ConfigurationError(
            "autonomy adaptive root is not the canonical promotion/adaptive path"
        )
    if os.path.lexists(os.fspath(adaptive_root)) and (
        adaptive_root.is_symlink() or not adaptive_root.is_dir()
    ):
        raise ConfigurationError(
            "autonomy adaptive root must be a non-symlink directory"
        )
    return RuntimeAutonomyBinding(
        suite_registry_spec_path=spec_path,
        suite_registry_spec_file_hash=spec_file_hash,
        suite_registry_spec_identity=spec_identity,
        active_suite_pointer_path=pointer_path,
        active_suite_pointer_file_hash=pointer_file_hash,
        active_suite_pointer_record_hash=pointer_record_hash,
        active_suite_id=active_suite_id,
        adaptive_policy_hash=adaptive_policy_hash,
        adaptive_root=adaptive_root,
    )


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
        or manifest["exportContract"] not in _HARDENED_EXPORT_CONTRACTS
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
            -item.sample_count,
            item.name,
        ) < (
            abs(existing.sample_count - target),
            -existing.sample_count,
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
    # Some read-only validator fixtures construct a controller without running
    # __init__. Keep the optional v2 integration safely disabled in that case.
    suite_registry: Optional[SuiteRotationRegistry] = None

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
        now: Optional[Callable[[], datetime]] = None,
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
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.registry = EventRegistry(runtime.promotion_root)
        self.suite_registry: Optional[SuiteRotationRegistry] = None
        self._writer_lock_depth = 0
        if held_controller_lock is not None:
            if (
                held_controller_lock.path != runtime.lock_path
                or not held_controller_lock.acquired
            ):
                raise ConfigurationError(
                    "held_controller_lock must already own the configured lock path"
                )
        self.held_controller_lock = held_controller_lock
        if runtime.schema_version == 2 and runtime.autonomy is not None:
            try:
                self.suite_registry = SuiteRotationRegistry(
                    runtime.autonomy.suite_registry_spec_path,
                    clock=self.now,
                )
                self._validate_suite_runtime_binding()
                self._validate_suite_champion_startup()
            except SafetyHalt:
                raise
            except (OSError, ValueError, SuiteRotationError) as exc:
                raise SafetyHalt(
                    f"cannot initialize suite-registry data plane: {exc}"
                ) from exc

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
            self._writer_lock_depth += 1
            try:
                yield
            finally:
                self._writer_lock_depth -= 1
            return
        with ControllerLock(
            self.runtime.lock_path, owner=self.runtime.controller.actor
        ):
            self._writer_lock_depth += 1
            try:
                yield
            finally:
                self._writer_lock_depth -= 1

    @staticmethod
    def _suite_evaluation_id(candidate_hash: str) -> str:
        _hash(candidate_hash, "suite-registry evaluation candidate hash")
        return f"promotion-candidate-{candidate_hash}"

    def _validate_suite_runtime_binding(self) -> Optional[Any]:
        """Revalidate the v2 pointer against this frozen data-plane runtime."""

        if self.suite_registry is None:
            return None
        binding = self.runtime.autonomy
        if self.runtime.schema_version != 2 or binding is None:
            raise SafetyHalt("suite registry exists without a runtime-v2 binding")
        try:
            if (
                binding.adaptive_policy_hash != ADAPTIVE_POLICY_HASH
                or binding.adaptive_root
                != self.runtime.promotion_root / "adaptive"
                or self.suite_registry.spec.path
                != binding.suite_registry_spec_path
                or self.suite_registry.spec.file_sha256
                != binding.suite_registry_spec_file_hash
                or self.suite_registry.spec.identity
                != binding.suite_registry_spec_identity
                or self.suite_registry.active_path
                != binding.active_suite_pointer_path
                or sha256_file(binding.suite_registry_spec_path)
                != binding.suite_registry_spec_file_hash
            ):
                raise SafetyHalt(
                    "suite registry specification/adaptive binding diverges from runtime"
                )
            pointer_data = binding.active_suite_pointer_path.read_bytes()
            if sha256_bytes(pointer_data) != binding.active_suite_pointer_file_hash:
                raise SafetyHalt("active-suite pointer file hash diverges from runtime")
            pointer = json.loads(pointer_data)
            pointer = _strict_object(
                pointer,
                "active-suite pointer",
                {
                    "schema_version",
                    "contract",
                    "spec_sha256",
                    "suite_id",
                    "version_sha256",
                    "manifest_path",
                    "manifest_sha256",
                    "manifest_identity",
                    "activated_at_utc",
                    "activation_champion_sha256",
                    "activation_generation_id",
                    "event_sequence",
                    "event_sha256",
                    "record_sha256",
                },
            )
            if pointer_data != canonical_json_bytes(pointer) + b"\n":
                raise SafetyHalt("active-suite pointer is not canonical")
            payload = dict(pointer)
            record_hash = payload.pop("record_sha256")
            manifest_path = Path(pointer["manifest_path"])
            if (
                pointer["schema_version"] != 1
                or pointer["contract"] != ACTIVE_SUITE_CONTRACT
                or pointer["spec_sha256"]
                != binding.suite_registry_spec_identity
                or pointer["suite_id"] != binding.active_suite_id
                or pointer["manifest_sha256"] != binding.active_suite_id
                or record_hash != binding.active_suite_pointer_record_hash
                or record_hash != canonical_sha256(payload)
                or manifest_path
                != self.runtime.suites / "manifest.json"
                or manifest_path.is_symlink()
                or not manifest_path.is_file()
                or pointer["manifest_sha256"]
                != self.runtime.controller.suite_manifest_hash
                or sha256_file(manifest_path)
                != self.runtime.controller.suite_manifest_hash
            ):
                raise SafetyHalt(
                    "active-suite pointer diverges from frozen runtime suite"
                )
            state = self.suite_registry.reconstruct()
            version = state.versions.get(binding.active_suite_id)
            if (
                state.active_suite_id != binding.active_suite_id
                or version is None
                or version.manifest_path != manifest_path
                or version.manifest_sha256
                != self.runtime.controller.suite_manifest_hash
                or not self.suite_registry._projection_consistent(state)
            ):
                raise SafetyHalt(
                    "suite registry state diverges from its bound active pointer"
                )
            return state
        except SafetyHalt:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, SuiteRotationError) as exc:
            raise SafetyHalt(f"cannot validate suite-registry data plane: {exc}") from exc

    def _suite_champion_replayable(
        self,
        promotion_state: Any,
        suite_state: Any,
    ) -> bool:
        if (
            promotion_state.current_champion_hash is None
            or promotion_state.current_generation_id is None
            or suite_state.current_champion is None
        ):
            return False
        generation = promotion_state.generations.get(
            promotion_state.current_generation_id
        )
        transaction = (
            self.runtime.promotion_root
            / "transactions"
            / str(promotion_state.current_generation_id)
        )
        try:
            projected = load_champion(self.runtime.champion_path)
            champion_cas = self._marker_payload(
                transaction / "champion-cas.json"
            )
        except (OSError, RegistryCorruptionError, SafetyHalt):
            return False
        return bool(
            generation is not None
            and generation.state == GenerationState.ACTIVE
            and generation.candidate_hash
            == promotion_state.current_champion_hash
            and suite_state.current_champion.sha256
            == generation.previous_champion_hash
            and champion_cas.get("champion_hash")
            == generation.candidate_hash
            and projected.champion_hash == generation.candidate_hash
            and projected.generation_id == generation.generation_id
        )

    def _validate_suite_champion_startup(self) -> None:
        if self.suite_registry is None:
            return
        promotion_state = self.registry.reconstruct()
        if promotion_state.current_champion_hash is None:
            return
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        current = suite_state.current_champion
        if (
            current is not None
            and current.sha256 == promotion_state.current_champion_hash
            and current.generation_id == promotion_state.current_generation_id
        ):
            return
        if self._suite_champion_replayable(promotion_state, suite_state):
            return
        if self._suite_champion_stale_after_rollback(
            promotion_state,
            suite_state,
        ):
            # The suite registry has no rollback event type. Permit startup
            # only far enough for the controller-locked pin cleanup; the next
            # automatic iteration still fails closed on this divergence.
            return
        raise SafetyHalt(
            "suite registry champion is stale relative to promotion state"
        )

    def _suite_champion_stale_after_rollback(
        self,
        promotion_state: Any,
        suite_state: Any,
    ) -> bool:
        current = suite_state.current_champion
        if (
            current is None
            or promotion_state.current_champion_hash is None
        ):
            return False
        try:
            projection = load_champion(self.runtime.champion_path)
        except (OSError, RegistryCorruptionError):
            return False
        return any(
            generation.state == GenerationState.ROLLED_BACK
            and generation.candidate_hash == current.sha256
            and generation.generation_id == current.generation_id
            and generation.previous_champion_hash
            == promotion_state.current_champion_hash
            and projection.champion_hash
            == promotion_state.current_champion_hash
            and projection.generation_id
            == promotion_state.current_generation_id
            and (
                self._transaction_dir(generation.generation_id)
                / "rollback-intent.json"
            ).is_file()
            for generation in promotion_state.generations.values()
        )

    def _record_suite_accepted_champion_locked(
        self,
        generation_id: str,
        candidate_hash: str,
        previous_champion_hash: str,
    ) -> Optional[Any]:
        if self.suite_registry is None:
            return None
        if self._writer_lock_depth <= 0:
            raise SafetyHalt(
                "suite accepted-champion publication requires controller lock"
            )
        state = self.registry.reconstruct()
        generation = state.generations.get(generation_id)
        transaction = self._transaction_dir(generation_id)
        champion_cas = self._marker_payload(
            transaction / "champion-cas.json"
        )
        if (
            generation is None
            or generation.state != GenerationState.ACTIVE
            or generation.candidate_hash != candidate_hash
            or generation.previous_champion_hash != previous_champion_hash
            or state.current_champion_hash != candidate_hash
            or state.current_generation_id != generation_id
            or champion_cas.get("champion_hash") != candidate_hash
        ):
            raise SafetyHalt(
                "suite champion event requires committed champion CAS/ACTIVE state"
            )
        champion = load_champion(self.runtime.champion_path)
        if (
            champion.champion_hash != candidate_hash
            or champion.generation_id != generation_id
        ):
            raise SafetyHalt(
                "suite champion event contradicts champion projection"
            )
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        current = suite_state.current_champion
        if (
            current is not None
            and current.sha256 == candidate_hash
            and current.generation_id == generation_id
        ):
            matches = [
                event
                for event in suite_state.events
                if event.event_type == "champion.accepted"
                and event.payload["generation_id"] == generation_id
            ]
            if len(matches) != 1:
                raise SafetyHalt(
                    "suite registry champion lacks one accepted event"
                )
            return matches[0]
        if (
            current is None
            or current.sha256 != previous_champion_hash
        ):
            raise SafetyHalt(
                "suite registry champion is stale before accepted promotion"
            )
        model_path = (
            self._verify_generation_leaf(generation_id, candidate_hash)
            / "model.bin.gz"
        )
        try:
            event = self.suite_registry.record_accepted_champion(
                model_path,
                generation_id=generation_id,
                expected_previous_champion_sha256=previous_champion_hash,
            )
        except (OSError, ValueError, SuiteRotationError) as exc:
            raise SafetyHalt(
                f"cannot record accepted champion in suite registry: {exc}"
            ) from exc
        self._checkpoint("promotion-suite-champion-accepted")
        return event

    def _ensure_suite_evaluation_pin_locked(
        self,
        candidate_hash: str,
        expected_champion_hash: str,
    ) -> Optional[Mapping[str, Any]]:
        if self.suite_registry is None:
            return None
        if self._writer_lock_depth <= 0:
            raise SafetyHalt("suite evaluation pin requires controller lock")
        _hash(expected_champion_hash, "suite evaluation champion hash")
        self._validate_suite_runtime_binding()
        evaluation_id = self._suite_evaluation_id(candidate_hash)
        try:
            pin = self.suite_registry.pin_evaluation(
                evaluation_id,
                expected_active_suite_id=self.runtime.autonomy.active_suite_id,
                expected_champion_sha256=expected_champion_hash,
            )
        except (OSError, ValueError, SuiteRotationError) as exc:
            raise SafetyHalt(f"cannot pin suite evaluation: {exc}") from exc
        if (
            pin.get("evaluation_id") != evaluation_id
            or pin.get("suite_id") != self.runtime.autonomy.active_suite_id
            or pin.get("champion_sha256") != expected_champion_hash
        ):
            raise SafetyHalt("suite evaluation pin contradicts candidate identity")
        return pin

    def _unpin_suite_evaluation_locked(
        self,
        candidate_hash: str,
    ) -> Optional[Any]:
        if self.suite_registry is None:
            return None
        if self._writer_lock_depth <= 0:
            raise SafetyHalt("suite evaluation unpin requires controller lock")
        try:
            return self.suite_registry.unpin_evaluation(
                self._suite_evaluation_id(candidate_hash)
            )
        except (OSError, ValueError, SuiteRotationError) as exc:
            raise SafetyHalt(f"cannot unpin suite evaluation: {exc}") from exc

    def _deep_audit_pending(self, generation_id: str) -> bool:
        queue = (
            self.runtime.promotion_root
            / "audits"
            / "queue"
            / f"{generation_id}.json"
        )
        report = (
            self.runtime.promotion_root
            / "audits"
            / "reports"
            / f"{generation_id}.json"
        )
        return queue.is_file() and not report.is_file()

    def _reconcile_suite_pins_locked(self) -> None:
        if self.suite_registry is None:
            return
        if self._writer_lock_depth <= 0:
            raise SafetyHalt("suite pin reconciliation requires controller lock")
        state = self.registry.reconstruct()
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        evaluating = {
            CandidateState.EVALUATING_INTEGRITY,
            CandidateState.EVALUATING_SCREEN,
            CandidateState.EVALUATING_FINALIST,
            CandidateState.EVALUATING_CONFIRMATION,
        }
        terminal = {
            CandidateState.SUPERSEDED,
            CandidateState.REJECTED,
            CandidateState.QUARANTINED,
        }
        for candidate in state.candidates.values():
            evaluation_id = self._suite_evaluation_id(
                candidate.candidate_hash
            )
            pinned = evaluation_id in suite_state.pins
            if candidate.state in terminal:
                if pinned:
                    self._unpin_suite_evaluation_locked(
                        candidate.candidate_hash
                    )
                    suite_state = self._validate_suite_runtime_binding()
                    assert suite_state is not None
                continue
            if candidate.state in evaluating:
                expected = candidate.tested_champion_hash
                if expected is None:
                    raise SafetyHalt(
                        "evaluating candidate has no tested champion for suite pin"
                    )
                self._ensure_suite_evaluation_pin_locked(
                    candidate.candidate_hash,
                    expected,
                )
                suite_state = self._validate_suite_runtime_binding()
                assert suite_state is not None
                continue
            if candidate.state != CandidateState.CONFIRMED:
                # A pin can precede the first durable stage transition after a
                # crash. Keep that CLAIMED pin for the retry.
                continue
            expected = candidate.tested_champion_hash
            if expected is None:
                raise SafetyHalt(
                    "confirmed candidate has no tested champion for suite pin"
                )
            generation = (
                state.generations.get(candidate.generation_id)
                if candidate.generation_id is not None
                else None
            )
            terminal_generation = generation is not None and generation.state in {
                GenerationState.QUARANTINED,
                GenerationState.ROLLED_BACK,
            }
            completed_active = bool(
                generation is not None
                and generation.state == GenerationState.ACTIVE
                and (
                    self._transaction_dir(generation.generation_id)
                    / "complete.json"
                ).is_file()
                and not self._deep_audit_pending(generation.generation_id)
            )
            if terminal_generation or completed_active:
                if pinned:
                    self._unpin_suite_evaluation_locked(
                        candidate.candidate_hash
                    )
                    suite_state = self._validate_suite_runtime_binding()
                    assert suite_state is not None
            else:
                self._ensure_suite_evaluation_pin_locked(
                    candidate.candidate_hash,
                    expected,
                )
                suite_state = self._validate_suite_runtime_binding()
                assert suite_state is not None

    def _reconcile_suite_data_plane_locked(self) -> None:
        if self.suite_registry is None:
            return
        if self._writer_lock_depth <= 0:
            raise SafetyHalt(
                "suite data-plane reconciliation requires controller lock"
            )
        promotion_state = self.registry.reconstruct()
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        self._reconcile_suite_pins_locked()
        promotion_state = self.registry.reconstruct()
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        if promotion_state.current_champion_hash is not None:
            current = suite_state.current_champion
            if not (
                current is not None
                and current.sha256
                == promotion_state.current_champion_hash
                and current.generation_id
                == promotion_state.current_generation_id
            ):
                if not self._suite_champion_replayable(
                    promotion_state, suite_state
                ):
                    raise SafetyHalt(
                        "suite registry champion is stale relative to promotion state"
                    )
                generation = promotion_state.generations[
                    promotion_state.current_generation_id
                ]
                self._record_suite_accepted_champion_locked(
                    generation.generation_id,
                    generation.candidate_hash,
                    generation.previous_champion_hash,
                )
        promotion_state = self.registry.reconstruct()
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        if promotion_state.current_champion_hash is not None and (
            suite_state.current_champion is None
            or suite_state.current_champion.sha256
            != promotion_state.current_champion_hash
            or suite_state.current_champion.generation_id
            != promotion_state.current_generation_id
        ):
            raise SafetyHalt(
                "suite registry champion remains stale after reconciliation"
            )

    def _clean_boundary_blockers_locked(self) -> Tuple[str, ...]:
        if self.suite_registry is None:
            return ("suite-registry-disabled",)
        if self._writer_lock_depth <= 0:
            raise SafetyHalt("generation boundary check requires controller lock")
        state = self.registry.reconstruct()
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        blockers = set()
        evaluating = {
            CandidateState.EVALUATING_INTEGRITY,
            CandidateState.EVALUATING_SCREEN,
            CandidateState.EVALUATING_FINALIST,
            CandidateState.EVALUATING_CONFIRMATION,
        }
        for candidate in state.candidates.values():
            if candidate.state in evaluating:
                blockers.add("evaluating-candidates")
            elif candidate.state == CandidateState.CONFIRMED:
                generation = (
                    state.generations.get(candidate.generation_id)
                    if candidate.generation_id is not None
                    else None
                )
                if (
                    generation is None
                    or generation.state
                    not in {
                        GenerationState.ACTIVE,
                        GenerationState.ROLLED_BACK,
                        GenerationState.QUARANTINED,
                    }
                    or (
                        generation.state == GenerationState.ACTIVE
                        and not (
                            self._transaction_dir(generation.generation_id)
                            / "complete.json"
                        ).is_file()
                    )
                ):
                    blockers.add("confirmed-candidates")
        if any(
            generation.state
            in {
                GenerationState.PROMOTION_INTENT,
                GenerationState.CANARY,
                GenerationState.ROLLOUT,
                GenerationState.ROLLBACK_PENDING,
            }
            for generation in state.generations.values()
        ):
            blockers.add("incomplete-promotions-or-rollbacks")
        transaction_root = self.runtime.promotion_root / "transactions"
        if transaction_root.exists():
            for intent in transaction_root.glob("*/intent.json"):
                generation = state.generations.get(intent.parent.name)
                if (
                    generation is not None
                    and generation.state
                    in {
                        GenerationState.ROLLED_BACK,
                        GenerationState.QUARANTINED,
                    }
                ):
                    continue
                if not (intent.parent / "complete.json").is_file():
                    blockers.add("incomplete-promotions-or-rollbacks")
            for rollback_intent in transaction_root.glob(
                "*/rollback-intent.json"
            ):
                generation = state.generations.get(
                    rollback_intent.parent.name
                )
                if (
                    generation is None
                    or generation.state != GenerationState.ROLLED_BACK
                ):
                    blockers.add("incomplete-promotions-or-rollbacks")
        if self._audit_queue_status()["pendingDepth"]:
            blockers.add("pending-deep-audits")
        try:
            self._training_gpu_lease_proof()
        except SafetyHalt:
            blockers.add("non-trainer-gpu-lease")
        if suite_state.pins:
            blockers.add("suite-pins")
        return tuple(sorted(blockers))

    def _publish_clean_generation_boundary_locked(self) -> Mapping[str, Any]:
        """Publish one idempotent post-continuity boundary; never activate."""

        if self.suite_registry is None:
            return {"published": False, "blockers": ["suite-registry-disabled"]}
        if self._writer_lock_depth <= 0:
            raise SafetyHalt("generation boundary publication requires controller lock")
        self._reconcile_suite_data_plane_locked()
        blockers = self._clean_boundary_blockers_locked()
        if blockers:
            return {"published": False, "blockers": list(blockers)}
        promotion_state = self.registry.reconstruct()
        suite_state = self._validate_suite_runtime_binding()
        assert suite_state is not None
        current = suite_state.current_champion
        if (
            current is None
            or promotion_state.current_champion_hash != current.sha256
            or promotion_state.current_generation_id != current.generation_id
        ):
            raise SafetyHalt("generation boundary champion is stale")
        relevant_registrations = {
            suite_id: registration
            for suite_id, registration in suite_state.registrations.items()
            if (
                registration["request_id"] in suite_state.requests
                and suite_state.requests[registration["request_id"]][
                    "base_suite_id"
                ]
                == suite_state.active_suite_id
                and suite_state.requests[registration["request_id"]][
                    "champion_sha256"
                ]
                == current.sha256
            )
        }
        if not relevant_registrations:
            return {
                "published": False,
                "blockers": ["continuity-not-ready"],
            }
        missing_continuity = sorted(
            suite_id
            for suite_id in relevant_registrations
            if suite_id not in suite_state.continuity
        )
        if missing_continuity:
            return {
                "published": False,
                "blockers": ["continuity-not-ready"],
                "missingSuiteIds": missing_continuity,
            }
        latest_continuity_sequence = max(
            suite_state.continuity[suite_id]["_sequence"]
            for suite_id in relevant_registrations
        )
        existing = [
            (boundary_id, boundary)
            for boundary_id, boundary in suite_state.boundaries.items()
            if (
                boundary["champion_sha256"] == current.sha256
                and boundary["generation_id"] == current.generation_id
                and boundary["_sequence"] > latest_continuity_sequence
            )
        ]
        if existing:
            boundary_id, _ = max(
                existing, key=lambda item: item[1]["_sequence"]
            )
            return {
                "published": False,
                "reused": True,
                "boundaryId": boundary_id,
                "blockers": [],
            }
        boundary_id = "promotion-boundary-" + canonical_sha256(
            {
                "spec_identity": self.runtime.autonomy.suite_registry_spec_identity,
                "active_suite_id": suite_state.active_suite_id,
                "champion_sha256": current.sha256,
                "generation_id": current.generation_id,
                "latest_continuity_sequence": latest_continuity_sequence,
                "continuity_suite_ids": sorted(relevant_registrations),
            }
        )
        try:
            event = self.suite_registry.record_generation_boundary(
                boundary_id,
                generation_id=current.generation_id,
                champion_sha256=current.sha256,
            )
        except (OSError, ValueError, SuiteRotationError) as exc:
            raise SafetyHalt(
                f"cannot publish clean generation boundary: {exc}"
            ) from exc
        return {
            "published": True,
            "boundaryId": boundary_id,
            "eventHash": event.event_sha256,
            "blockers": [],
        }

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

    def _load_suite_manifest(self) -> Tuple[Mapping[str, Any], bytes]:
        path = self.runtime.suites / "manifest.json"
        try:
            data = path.read_bytes()
            value = json.loads(data)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"cannot load frozen suite manifest: {exc}") from exc
        if not isinstance(value, dict):
            raise SafetyHalt("frozen suite manifest root must be an object")
        if sha256_bytes(data) != self.runtime.controller.suite_manifest_hash:
            raise SafetyHalt("frozen suite manifest byte hash changed")
        payload = dict(value)
        supplied_payload_hash = payload.pop("manifestPayloadSha256", None)
        if (
            supplied_payload_hash is not None
            and supplied_payload_hash != canonical_sha256(payload)
        ):
            raise SafetyHalt("frozen suite manifest payload hash is invalid")
        if (
            value.get("policy_hash") != self.runtime.controller.policy_hash
            or value.get("source_revision")
            != _policy_get(
                self.runtime.frozen_policy,
                "frozen_plan",
                "source_revision",
            )
        ):
            raise SafetyHalt("frozen suite manifest policy/source binding changed")
        return value, data

    def _promotion_readiness_errors(self) -> Tuple[str, ...]:
        """Return immutable-provenance reasons that block rollout mutation."""

        errors: List[str] = []
        expected_labels = list(PROMOTION_READY_LABELS)

        def has_expected_labels(value: Any) -> bool:
            return (
                isinstance(value, list)
                and all(isinstance(label, str) for label in value)
                and sorted(value) == expected_labels
            )

        policy = self.runtime.frozen_policy
        if policy.get("policy_version") != PROMOTION_READY_POLICY_VERSION:
            errors.append("POLICY_NOT_MACHINE_REVIEW_V3")
        machine_curation = policy.get("machine_curation_contract")
        if not isinstance(machine_curation, Mapping):
            errors.append("POLICY_MACHINE_CURATION_MISSING")
        else:
            if machine_curation.get("final_contract") != PROMOTION_READY_BANK_CONTRACT:
                errors.append("POLICY_MACHINE_CONTRACT_MISMATCH")
            if machine_curation.get("review_mode") != PROMOTION_READY_REVIEW_MODE:
                errors.append("POLICY_REVIEW_MODE_MISMATCH")
            if machine_curation.get("consensus_rules_version") != 1:
                errors.append("POLICY_CONSENSUS_RULES_MISMATCH")
            if machine_curation.get("stability_margin") != 5.0:
                errors.append("POLICY_STABILITY_MARGIN_MISMATCH")
            if not has_expected_labels(machine_curation.get("allowed_labels")):
                errors.append("POLICY_ALLOWED_LABELS_MISMATCH")
            if machine_curation.get("model_roles") != [
                "immutable_original",
                "frozen_champion",
            ]:
                errors.append("POLICY_MODEL_ROLES_MISMATCH")
            if machine_curation.get("search_modes") != [
                "standard",
                "powered",
            ]:
                errors.append("POLICY_SEARCH_MODES_MISMATCH")
            if machine_curation.get("visits") != [2000, 8000]:
                errors.append("POLICY_CURATION_VISITS_MISMATCH")
            if (
                machine_curation.get("symmetry_semantics")
                != "katago-shape-preserving-d4-v1"
            ):
                errors.append("POLICY_SYMMETRY_SEMANTICS_MISMATCH")
            if (
                machine_curation.get(
                    "automatic_promotion_requires_transitive_suite_provenance"
                )
                is not True
            ):
                errors.append("POLICY_TRANSITIVE_PROVENANCE_NOT_REQUIRED")

        try:
            manifest, _ = self._load_suite_manifest()
        except SafetyHalt as exc:
            errors.append(f"SUITE_MANIFEST_INVALID:{exc}")
            return tuple(sorted(set(errors)))

        if manifest.get("manifestContract") != PROMOTION_READY_SUITE_CONTRACT:
            errors.append("SUITE_CONTRACT_NOT_MACHINE_REVIEW_V3")
        if not has_expected_labels(manifest.get("acceptedLabels")):
            errors.append("SUITE_ACCEPTED_LABELS_MISMATCH")
        if manifest.get("machineReviewOnly") is not True:
            errors.append("SUITE_MACHINE_REVIEW_ONLY_MISSING")

        raw_sources = manifest.get("sources")
        source_hashes = (
            {
                source.get("sha256")
                for source in raw_sources
                if isinstance(source, Mapping) and isinstance(source.get("sha256"), str)
            }
            if isinstance(raw_sources, list)
            else set()
        )
        if any(_SHA_RE.fullmatch(value) is None for value in source_hashes):
            errors.append("SUITE_SOURCE_HASH_INVALID")
        if (
            not isinstance(raw_sources, list)
            or not all(isinstance(source, Mapping) for source in raw_sources)
            or len(raw_sources) != len(source_hashes)
        ):
            errors.append("SUITE_SOURCE_INVENTORY_INVALID")
        curation_sources = manifest.get("curationSources")
        if not isinstance(curation_sources, list) or not curation_sources:
            errors.append("SUITE_CURATION_SOURCES_MISSING")
            return tuple(sorted(set(errors)))
        if len(curation_sources) != len(source_hashes):
            errors.append("SUITE_CURATION_SOURCE_COUNT_MISMATCH")

        bound_source_hashes = set()
        bound_source_names = set()
        common_source_models = None
        for source in curation_sources:
            if not isinstance(source, Mapping):
                errors.append("SUITE_CURATION_SOURCE_MALFORMED")
                continue
            if source.get("contract") != PROMOTION_READY_BANK_CONTRACT:
                errors.append("SOURCE_MACHINE_CONTRACT_MISMATCH")
            source_name = source.get("source_name")
            raw_source_names = {
                item.get("name")
                for item in raw_sources
                if isinstance(item, Mapping)
            }
            if (
                not isinstance(source_name, str)
                or source_name not in raw_source_names
                or source_name in bound_source_names
            ):
                errors.append("SOURCE_NAME_BINDING_INVALID")
            else:
                bound_source_names.add(source_name)
            if source.get("review_mode") != PROMOTION_READY_REVIEW_MODE:
                errors.append("SOURCE_REVIEW_MODE_MISMATCH")
            if source.get("consensus_rules_version") != 1:
                errors.append("SOURCE_CONSENSUS_RULES_MISMATCH")
            if source.get("policy_hash") != self.runtime.controller.policy_hash:
                errors.append("SOURCE_POLICY_HASH_MISMATCH")
            if not has_expected_labels(source.get("allowed_labels")):
                errors.append("SOURCE_ALLOWED_LABELS_MISMATCH")
            output_hash = source.get("output_sha256")
            if (
                not isinstance(output_hash, str)
                or _SHA_RE.fullmatch(output_hash) is None
            ):
                errors.append("SOURCE_OUTPUT_HASH_INVALID")
            else:
                bound_source_hashes.add(output_hash)
                if output_hash not in source_hashes:
                    errors.append("SOURCE_OUTPUT_NOT_BOUND_TO_SUITE")
            manifest_hash = source.get("manifest_sha256")
            if (
                not isinstance(manifest_hash, str)
                or _SHA_RE.fullmatch(manifest_hash) is None
            ):
                errors.append("SOURCE_MANIFEST_HASH_INVALID")
            rejected_count = source.get("rejected_count")
            rejected_hash = source.get("rejected_sha256")
            if (
                type(rejected_count) is not int
                or rejected_count < 0
                or not isinstance(rejected_hash, str)
                or _SHA_RE.fullmatch(rejected_hash) is None
            ):
                errors.append("SOURCE_REJECTION_PROVENANCE_INVALID")
            models = source.get("models")
            if not isinstance(models, Mapping):
                errors.append("SOURCE_MODELS_MISSING")
            else:
                original = models.get("original")
                champion = models.get("champion")
                if (
                    not isinstance(original, Mapping)
                    or original.get("role") != "immutable_original"
                    or original.get("sha256") != self.runtime.controller.original_hash
                ):
                    errors.append("SOURCE_ORIGINAL_MODEL_MISMATCH")
                if (
                    not isinstance(champion, Mapping)
                    or champion.get("role") != "frozen_champion"
                    or not isinstance(champion.get("sha256"), str)
                    or _SHA_RE.fullmatch(champion["sha256"]) is None
                ):
                    errors.append("SOURCE_CHAMPION_MODEL_INVALID")
                elif isinstance(original, Mapping) and champion[
                    "sha256"
                ] == original.get("sha256"):
                    errors.append("SOURCE_MODELS_NOT_INDEPENDENT")
                normalized_models = (
                    (
                        original.get("role"),
                        original.get("sha256"),
                        champion.get("role"),
                        champion.get("sha256"),
                    )
                    if isinstance(original, Mapping)
                    and isinstance(champion, Mapping)
                    else None
                )
                if common_source_models is None:
                    common_source_models = normalized_models
                elif normalized_models != common_source_models:
                    errors.append("SUITE_CURATION_MODELS_MISMATCH")
        if bound_source_hashes != source_hashes:
            errors.append("SUITE_SOURCE_HASH_SET_MISMATCH")
        return tuple(sorted(set(errors)))

    @staticmethod
    def _manifest_cells(manifest: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
        cells = manifest.get("cells")
        if isinstance(cells, list):
            if not all(isinstance(cell, Mapping) for cell in cells):
                raise SafetyHalt("frozen suite manifest contains malformed cells")
            return tuple(dict(cell) for cell in cells)
        if not isinstance(cells, Mapping):
            return ()

        flattened: List[Mapping[str, Any]] = []

        def walk(value: Any, coordinates: Tuple[str, ...]) -> None:
            if not isinstance(value, Mapping):
                return
            schedule = value.get("schedule")
            if (
                "schedule_path" in value
                or "schedule_hash" in value
                or isinstance(schedule, Mapping)
            ):
                row = dict(value)
                row.setdefault("_coordinates", list(coordinates))
                flattened.append(row)
                return
            for key, child in value.items():
                if isinstance(child, Mapping):
                    walk(child, coordinates + (str(key),))

        walk(cells, ())
        return tuple(flattened)

    @staticmethod
    def _cell_value(cell: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in cell:
                return cell[name]
        schedule = cell.get("schedule")
        if isinstance(schedule, Mapping):
            for name in names:
                if name in schedule:
                    return schedule[name]
        return None

    def _resolve_manifest_schedule(
        self,
        manifest: Mapping[str, Any],
        *,
        stage: str,
        look: str,
        cell_name: str,
        comparison: str,
    ) -> Optional[Mapping[str, Any]]:
        cells = self._manifest_cells(manifest)
        if not cells:
            return None
        expected_stages = {
            stage,
            "stage-3" if stage in {"confirmation", "stage-3"} else stage,
        }
        aliases = {
            cell_name,
            comparison,
            cell_name.replace("_", "-"),
        }
        matches: List[Mapping[str, Any]] = []
        for cell in cells:
            coordinates = tuple(str(item) for item in cell.get("_coordinates", ()))
            declared_stage = self._cell_value(
                cell, "stage", "evaluation_stage", "controller_stage"
            )
            declared_look = self._cell_value(cell, "look", "look_id")
            declared_name = self._cell_value(
                cell, "cell", "cell_name", "name", "comparison"
            )
            coordinate_set = set(coordinates)
            stage_matches = (
                declared_stage in expected_stages
                or bool(coordinate_set.intersection(expected_stages))
            )
            look_matches = (
                declared_look == look or look in coordinate_set
            )
            name_matches = (
                declared_name in aliases
                or bool(coordinate_set.intersection(aliases))
            )
            if stage_matches and look_matches and name_matches:
                matches.append(cell)
        if len(matches) != 1:
            raise SafetyHalt(
                "authoritative suite manifest must identify exactly one "
                f"{stage}/{look}/{cell_name} cell; found {len(matches)}"
            )
        cell = matches[0]
        relative = self._cell_value(cell, "schedule_path", "path")
        expected_hash = self._cell_value(
            cell, "schedule_hash", "sha256", "schedule_sha256"
        )
        schedule_id = self._cell_value(
            cell, "schedule_id", "scheduleId"
        )
        pair_count = self._cell_value(
            cell, "pair_count", "pairCount", "color_pairs"
        )
        bank_hash = self._cell_value(
            cell,
            "suite_bank_hash",
            "suiteBankSha256",
            "bank_hash",
        )
        cluster_hash = self._cell_value(
            cell,
            "independent_cluster_ids_hash",
            "independentClusterIdsSha256",
            "position_ids_hash",
        )
        minimum_clusters = self._cell_value(
            cell,
            "minimum_independent_position_clusters",
            "minimumIndependentPositionClusters",
        )
        visits = self._cell_value(cell, "visits", "max_visits", "maxVisits")
        if not isinstance(relative, str) or not relative:
            raise SafetyHalt("authoritative manifest cell has no schedule path")
        _hash(expected_hash, "authoritative manifest schedule hash")
        _nonempty(schedule_id, "authoritative manifest schedule ID")
        if type(pair_count) is not int or pair_count <= 0:
            raise SafetyHalt(
                "authoritative manifest cell has no positive pair count"
            )
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SafetyHalt("authoritative manifest schedule path is unsafe")
        path = self.runtime.suites / path
        if path.is_symlink() or not path.is_file():
            raise SafetyHalt(
                f"authoritative manifest schedule is not a regular file: {path}"
            )
        if sha256_file(path) != expected_hash:
            raise SafetyHalt(
                f"authoritative manifest schedule hash changed: {path}"
            )
        if bank_hash is not None:
            _hash(bank_hash, "authoritative manifest suite bank hash")
        if cluster_hash is not None:
            _hash(
                cluster_hash,
                "authoritative manifest independent cluster hash",
            )
        if minimum_clusters is not None and (
            type(minimum_clusters) is not int
            or minimum_clusters <= 0
            or minimum_clusters > pair_count
        ):
            raise SafetyHalt(
                "authoritative manifest independent cluster minimum is invalid"
            )
        if type(visits) is not int or visits <= 0:
            raise SafetyHalt(
                "authoritative manifest cell has no positive visit count"
            )
        for key, expected in (
            ("policy_hash", self.runtime.controller.policy_hash),
            (
                "policy_version",
                self.runtime.frozen_policy.get("policy_version"),
            ),
            (
                "source_revision",
                _policy_get(
                    self.runtime.frozen_policy,
                    "frozen_plan",
                    "source_revision",
                ),
            ),
        ):
            if key in cell and cell.get(key) != expected:
                raise SafetyHalt(
                    f"authoritative manifest cell {key} changed"
                )
        declared_comparison = self._cell_value(cell, "comparison")
        if (
            declared_comparison is not None
            and declared_comparison != comparison
        ):
            raise SafetyHalt(
                "authoritative manifest cell comparison contradicts controller"
            )
        return {
            "cell": cell_name,
            "comparison": comparison,
            "stage": stage,
            "look": look,
            "path": str(path),
            "sha256": expected_hash,
            "scheduleId": schedule_id,
            "pairCount": pair_count,
            "suiteBankSha256": bank_hash,
            "independentClusterIdsSha256": cluster_hash,
            "minimumIndependentPositionClusters": minimum_clusters,
            "visits": visits,
        }

    def validate_static_inputs(self) -> None:
        """Verify every configured immutable policy/model/config/schedule hash."""

        if self.suite_registry is not None:
            self._validate_suite_runtime_binding()
        _validate_queue_contract(
            self.runtime.controller, self.runtime.frozen_policy
        )
        manifest, _ = self._load_suite_manifest()
        authoritative_cells = self._manifest_cells(manifest)
        byte_checks: List[Tuple[Path, str, str]] = [
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
        ]
        if not authoritative_cells:
            byte_checks.extend(
                (
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
                )
            )
        for path, expected, role in byte_checks:
            if path.is_symlink() or not path.is_file():
                raise SafetyHalt(f"{role} is not a regular file: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise SafetyHalt(
                    f"{role} hash mismatch: expected {expected}, found {actual}"
                )
        if authoritative_cells:
            for index, cell in enumerate(authoritative_cells):
                relative = self._cell_value(cell, "schedule_path", "path")
                expected = self._cell_value(
                    cell, "schedule_hash", "sha256", "schedule_sha256"
                )
                if not isinstance(relative, str) or not relative:
                    raise SafetyHalt(
                        f"authoritative manifest cell {index} has no schedule path"
                    )
                _hash(expected, f"authoritative manifest cell {index} schedule hash")
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise SafetyHalt(
                        f"authoritative manifest cell {index} path is unsafe"
                    )
                path = self.runtime.suites / relative_path
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or sha256_file(path) != expected
                ):
                    raise SafetyHalt(
                        f"authoritative manifest cell {index} schedule changed"
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
            self.runtime.promotion_root / "operations",
            self.runtime.promotion_root / "trash" / "intents",
            self.runtime.promotion_root / "trash" / "manifests",
            self.runtime.promotion_root / "trash" / "objects",
            self.runtime.promotion_root / "trash" / "deleted",
            self.runtime.promotion_root / "trash" / "deletion-intents",
            self.runtime.promotion_root / "audits" / "queue",
            self.runtime.promotion_root / "audits" / "reports",
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
            self._publish_disk_backpressure_denial(
                free_bytes=int(usage.free),
                required_bytes=int(required),
            )
            raise InsufficientDiskError(
                f"free bytes {usage.free} below required reserve {required}"
            )

    def _publish_disk_backpressure_denial(
        self, *, free_bytes: int, required_bytes: int
    ) -> None:
        """Fail export closed before a disk-reserve exception stops the controller."""

        if not self.automatic:
            return
        target = (
            self.runtime.promotion_root
            / "operations"
            / "backpressure.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": 1,
            "updated_at_utc": utc_timestamp(self.now()),
            "controller_hash": self.runtime.controller.controller_hash,
            "policy_hash": self.runtime.controller.policy_hash,
            "allowExport": False,
            "allowEvaluation": False,
            "exportPaused": True,
            "evaluationPaused": True,
            "exportBacklogDepth": 0,
            "evaluationBacklogDepth": 0,
            "maximumActiveEvaluatorEntries":
                self.runtime.controller.max_active_queue,
            "importantQueueWarningDepth": _policy_get(
                self.runtime.frozen_policy,
                "queue",
                "important_queue_warning_depth",
                default=self.runtime.controller.max_active_queue + 1,
            ),
            "diskFreeBytes": free_bytes,
            "minimumFreeBytes": self.runtime.controller.min_free_bytes,
            "requiredFreeBytes": required_bytes,
            "reasons": ["disk-reserve-hard-limit"],
        }
        try:
            atomic_write_bytes(
                target, canonical_json_bytes(value) + b"\n"
            )
        except OSError:
            # A missing gate also fails the hardened exporter closed. Never
            # leave a fresh allowance visible after the reserve check failed.
            target.unlink(missing_ok=True)
            fsync_directory(target.parent)

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
        _hash(candidate_hash, "candidate hash")
        _hash(champion_hash, "champion hash")
        required_topology = "7-workers-100-threads"
        if topology != required_topology:
            raise SafetyHalt(
                f"evaluation topology must be {required_topology}, found {topology}"
            )
        from risk_score.evaluation_runner import build_evaluation_matrix

        config = self.runtime.controller
        manifest_path = self.runtime.suites / "manifest.json"
        manifest, _ = self._load_suite_manifest()
        authoritative_cells = self._manifest_cells(manifest)
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
        canonical_stage = {
            "integrity": "stage-0",
            "stage-0": "stage-0",
            "screen": "stage-1",
            "stage-1": "stage-1",
            "finalist": "stage-2",
            "stage-2": "stage-2",
            "confirmation": "stage-3",
            "stage-3": "stage-3",
        }.get(stage)
        if canonical_stage is None:
            raise SafetyHalt(f"unknown evaluation stage coordinate: {stage!r}")
        canonical_look = (
            _canonical_confirmation_look(look) if confirmation else look
        )
        policy_version = _nonempty(
            self.runtime.frozen_policy.get("policy_version"),
            "frozen policy version",
        )
        if canonical_stage == "stage-0":
            config_hash = canonical_sha256(
                sorted(
                    {
                        config.powered_config_hash,
                        config.standard_config_hash,
                    }
                )
            )
            schedule_hash = canonical_sha256([])
            evaluation_key = "probe-" + canonical_sha256(
                {
                    "planContract": EVALUATION_PLAN_CONTRACT,
                    "candidateModelSha256": candidate_hash,
                    "championModelSha256": champion_hash,
                    "originalModelSha256": config.original_hash,
                    "configHash": config_hash,
                    "scheduleHash": schedule_hash,
                    "policyHash": config.policy_hash,
                    "suiteManifestHash": config.suite_manifest_hash,
                    "stage": canonical_stage,
                    "look": canonical_look,
                    "topology": required_topology,
                }
            )
            return EvaluationMatrixPlan(
                specs=(),
                candidate_hash=candidate_hash,
                champion_hash=champion_hash,
                original_hash=config.original_hash,
                evaluation_key=evaluation_key,
                config_hash=config_hash,
                schedule_hash=schedule_hash,
                policy_hash=config.policy_hash,
                selfplay_config_hash=config.selfplay_config_hash,
                topology=required_topology,
                stage=canonical_stage,
                look=canonical_look,
                policy_path=str(self.runtime.policy_path),
                policy_version=policy_version,
                suite_manifest_path=str(manifest_path),
                suite_manifest_hash=config.suite_manifest_hash,
                schedule_artifacts={},
            )
        look_policy: Optional[Mapping[str, Any]] = None
        if confirmation:
            policy_looks = _policy_get(
                self.runtime.frozen_policy,
                "evaluation_stages",
                "stage_3_promotion_confirmation",
                "looks",
                default=[],
            )
            look_number = int(canonical_look.rsplit("-", 1)[1])
            if isinstance(policy_looks, list):
                look_policy = next(
                    (
                        item
                        for item in policy_looks
                        if isinstance(item, Mapping)
                        and item.get("look_number") == look_number
                    ),
                    None,
                )
            if (
                (authoritative_cells or policy_looks)
                and (
                    not isinstance(policy_looks, list)
                    or look_policy is None
                )
            ):
                raise SafetyHalt(
                    f"frozen policy does not define {canonical_look}"
                )
        stage_1_policy = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "stage_1_cheap_paired_screen",
        )
        stage_2_policy = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "stage_2_finalist_selection",
        )
        stage_3_policy = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "stage_3_promotion_confirmation",
        )
        v2_policy = self.runtime.frozen_policy.get("schema_version") in {2, 3}
        if canonical_stage == "stage-1":
            if not isinstance(stage_1_policy, Mapping):
                if v2_policy:
                    raise SafetyHalt("frozen policy has no Stage-1 execution contract")
                stage_1_policy = {}
            powered_visits = stage_1_policy.get("visits", 400)
            standard_visits = powered_visits
            include_powered_original = False
            include_standard_original = False
            lead_visits = powered_visits
            cell_coordinates = (
                (
                    "powered_candidate_vs_champion",
                    "candidate-vs-champion-powered",
                ),
            )
        elif canonical_stage == "stage-2":
            if not isinstance(stage_2_policy, Mapping):
                if v2_policy:
                    raise SafetyHalt("frozen policy has no Stage-2 execution contract")
                stage_2_policy = {}
            powered_visits = stage_2_policy.get("ordinary_visits", 800)
            standard_visits = powered_visits
            include_powered_original = champion_hash != config.original_hash
            include_standard_original = False
            lead_visits = stage_2_policy.get("lead_visits", 800)
            cell_coordinates = (
                (
                    "powered_candidate_vs_champion",
                    "candidate-vs-champion-powered",
                ),
                *(
                    (
                        (
                            "powered_candidate_vs_original",
                            "candidate-vs-original-powered",
                        ),
                    )
                    if include_powered_original
                    else ()
                ),
                (
                    "lead_40",
                    "candidate-vs-champion-powered-lead-40",
                ),
                (
                    "lead_80",
                    "candidate-vs-champion-powered-lead-80",
                ),
            )
        else:
            if not isinstance(stage_3_policy, Mapping):
                if v2_policy:
                    raise SafetyHalt("frozen policy has no Stage-3 execution contract")
                stage_3_policy = {}
            powered_visits = stage_3_policy.get("powered_visits", 2000)
            standard_visits = stage_3_policy.get("standard_visits", 800)
            include_powered_original = True
            include_standard_original = True
            lead_visits = powered_visits
            cell_coordinates = (
                (
                    "powered_candidate_vs_champion",
                    "candidate-vs-champion-powered",
                ),
                (
                    "powered_candidate_vs_original",
                    "candidate-vs-original-powered",
                ),
                (
                    "standard_candidate_vs_original",
                    "candidate-vs-original-standard",
                ),
                ("lead_40", "candidate-vs-champion-powered-lead-40"),
                ("lead_80", "candidate-vs-champion-powered-lead-80"),
            )
        for value, role in (
            (powered_visits, "powered visits"),
            (standard_visits, "standard visits"),
            (lead_visits, "lead visits"),
        ):
            if type(value) is not int or value <= 0:
                raise SafetyHalt(f"frozen policy {role} must be a positive integer")

        ordinary_bank_name = "confirmation" if confirmation else "discovery"
        schedule_artifacts: Dict[str, Mapping[str, Any]] = {}
        matrix_kwargs: Dict[str, Any] = {
            "powered_visits": powered_visits,
            "standard_visits": standard_visits,
            "lead_visits": lead_visits,
            "include_powered_original": include_powered_original,
            "include_standard_original": include_standard_original,
        }
        if authoritative_cells:
            for cell_name, comparison in cell_coordinates:
                artifact = self._resolve_manifest_schedule(
                    manifest,
                    stage=canonical_stage,
                    look=canonical_look,
                    cell_name=cell_name,
                    comparison=comparison,
                )
                if artifact is None:
                    raise SafetyHalt(
                        "authoritative suite manifest unexpectedly has no cells"
                    )
                schedule_artifacts[cell_name] = artifact

            powered = schedule_artifacts["powered_candidate_vs_champion"]
            powered_original = schedule_artifacts.get(
                "powered_candidate_vs_original"
            )
            if powered_original is not None:
                for field in (
                    "path",
                    "sha256",
                    "scheduleId",
                    "pairCount",
                    "suiteBankSha256",
                    "independentClusterIdsSha256",
                    "minimumIndependentPositionClusters",
                    "visits",
                ):
                    if powered.get(field) != powered_original.get(field):
                        raise SafetyHalt(
                            "powered ordinary cells must share the exact "
                            f"schedule and visit contract ({field})"
                        )
            ordinary_schedule_hash = str(powered["sha256"])
            ordinary_bank_hash = str(powered["suiteBankSha256"])
            ordinary_schedule_id = str(powered["scheduleId"])

            expected_pairs: Dict[str, Any]
            expected_minima: Mapping[str, Any] = {}
            expected_visits = {
                name: (
                    standard_visits
                    if name == "standard_candidate_vs_original"
                    else lead_visits
                    if name in {"lead_40", "lead_80"}
                    else powered_visits
                )
                for name in schedule_artifacts
            }
            if canonical_stage == "stage-1":
                expected_pairs = {
                    "powered_candidate_vs_champion":
                        stage_1_policy.get("ordinary_color_pairs")
                }
            elif canonical_stage == "stage-2":
                expected_pairs = {
                    "powered_candidate_vs_champion":
                        stage_2_policy.get("ordinary_color_pairs"),
                    "powered_candidate_vs_original":
                        stage_2_policy.get("ordinary_color_pairs"),
                    "lead_40": stage_2_policy.get("lead_40_color_pairs"),
                    "lead_80": stage_2_policy.get("lead_80_color_pairs"),
                }
            else:
                assert look_policy is not None
                expected_pairs = {
                    "powered_candidate_vs_champion":
                        look_policy.get("powered_ordinary_color_pairs_per_matchup"),
                    "powered_candidate_vs_original":
                        look_policy.get("powered_ordinary_color_pairs_per_matchup"),
                    "standard_candidate_vs_original":
                        look_policy.get("standard_ordinary_color_pairs"),
                    "lead_40": look_policy.get("lead_40_color_pairs"),
                    "lead_80": look_policy.get("lead_80_color_pairs"),
                }
                minima = look_policy.get("minimum_independent_position_clusters")
                expected_minima = minima if isinstance(minima, Mapping) else {}

            for name, artifact in schedule_artifacts.items():
                _hash(
                    artifact.get("suiteBankSha256"),
                    f"authoritative {name} suite bank hash",
                )
                _hash(
                    artifact.get("independentClusterIdsSha256"),
                    f"authoritative {name} independent cluster hash",
                )
                expected_pair_count = expected_pairs.get(name)
                if (
                    expected_pair_count is not None
                    and artifact.get("pairCount") != expected_pair_count
                ):
                    raise SafetyHalt(
                        f"authoritative {name} pair count contradicts policy"
                    )
                if artifact.get("visits") != expected_visits[name]:
                    raise SafetyHalt(
                        f"authoritative {name} visit count contradicts policy"
                    )
                expected_minimum = expected_minima.get(name)
                if (
                    expected_minimum is not None
                    and artifact.get("minimumIndependentPositionClusters")
                    != expected_minimum
                ):
                    raise SafetyHalt(
                        f"authoritative {name} cluster minimum contradicts policy"
                    )

            standard = schedule_artifacts.get("standard_candidate_vs_original")
            standard_schedule_hash = str(
                standard["sha256"] if standard is not None else powered["sha256"]
            )
            standard_schedule_id = str(
                standard["scheduleId"]
                if standard is not None
                else powered["scheduleId"]
            )
            lead40 = schedule_artifacts.get("lead_40")
            lead80 = schedule_artifacts.get("lead_80")
            if (lead40 is None) != (lead80 is None):
                raise SafetyHalt("evaluation plan has only one Lead discovery cell")
            if lead40 is not None and lead80 is not None:
                matrix_kwargs.update(
                    {
                        "lead_40_schedule_sha": lead40["sha256"],
                        "lead_80_schedule_sha": lead80["sha256"],
                        "lead_40_suite_bank_sha": lead40["suiteBankSha256"],
                        "lead_80_suite_bank_sha": lead80["suiteBankSha256"],
                        "lead_40_schedule_id": lead40["scheduleId"],
                        "lead_80_schedule_id": lead80["scheduleId"],
                    }
                )
        else:
            if v2_policy and canonical_stage in {"stage-1", "stage-2"}:
                raise SafetyHalt(
                    "Stage 1 and Stage 2 require exact authoritative manifest cells"
                )
            ordinary_schedule_hash = (
                config.confirmation_schedule_hash
                if confirmation
                else config.discovery_schedule_hash
            )
            ordinary_bank_hash, ordinary_schedule_id = bank_binding(
                ordinary_bank_name, ordinary_schedule_hash
            )
            standard_schedule_hash = (
                config.standard_confirmation_schedule_hash
                if confirmation
                else ordinary_schedule_hash
            )
            standard_schedule_id = ordinary_schedule_id
            ordinary_path = (
                self.runtime.confirmation_schedule_path
                if confirmation
                else self.runtime.discovery_schedule_path
            )
            base_artifact = {
                "stage": canonical_stage,
                "look": canonical_look,
                "path": str(ordinary_path),
                "sha256": ordinary_schedule_hash,
                "scheduleId": ordinary_schedule_id,
                "suiteBankSha256": ordinary_bank_hash,
                "visits": powered_visits,
            }
            for name, comparison in cell_coordinates:
                if name in {
                    "standard_candidate_vs_original",
                    "lead_40",
                    "lead_80",
                }:
                    continue
                schedule_artifacts[name] = {
                    **base_artifact,
                    "cell": name,
                    "comparison": comparison,
                }
            if include_standard_original:
                schedule_artifacts["standard_candidate_vs_original"] = {
                    **base_artifact,
                    "cell": "standard_candidate_vs_original",
                    "comparison": "candidate-vs-original-standard",
                    "path": str(self.runtime.standard_confirmation_schedule_path),
                    "sha256": standard_schedule_hash,
                    "visits": standard_visits,
                }
            if canonical_stage in {"stage-2", "stage-3"}:
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
                schedule_artifacts.update({
                    "lead_40": {
                        "cell": "lead_40",
                        "comparison":
                            "candidate-vs-champion-powered-lead-40",
                        "stage": canonical_stage,
                        "look": canonical_look,
                        "path": str(self.runtime.lead40_schedule_path),
                        "sha256": config.lead40_schedule_hash,
                        "scheduleId": lead40_schedule_id,
                        "suiteBankSha256": lead40_bank_hash,
                        "visits": lead_visits,
                    },
                    "lead_80": {
                        "cell": "lead_80",
                        "comparison":
                            "candidate-vs-champion-powered-lead-80",
                        "stage": canonical_stage,
                        "look": canonical_look,
                        "path": str(self.runtime.lead80_schedule_path),
                        "sha256": config.lead80_schedule_hash,
                        "scheduleId": lead80_schedule_id,
                        "suiteBankSha256": lead80_bank_hash,
                        "visits": lead_visits,
                    },
                })
        specs = tuple(
            build_evaluation_matrix(
                candidate_model_sha=candidate_hash,
                champion_model_sha=champion_hash,
                original_model_sha=config.original_hash,
                powered_config_sha=config.powered_config_hash,
                standard_config_sha=config.standard_config_hash,
                powered_schedule_sha=ordinary_schedule_hash,
                standard_schedule_sha=standard_schedule_hash,
                policy_sha=config.policy_hash,
                suite=ordinary_bank_name,
                stage=canonical_stage,
                look=canonical_look,
                topology=topology,
                suite_manifest_sha=config.suite_manifest_hash,
                ordinary_suite_bank_sha=ordinary_bank_hash,
                powered_schedule_id=ordinary_schedule_id,
                standard_schedule_id=standard_schedule_id,
                **matrix_kwargs,
            )
        )
        expected_artifacts = {
            artifact["comparison"]: artifact
            for artifact in schedule_artifacts.values()
        }
        for spec in specs:
            artifact = expected_artifacts.get(spec.comparison)
            if artifact is None:
                raise SafetyHalt(
                    f"evaluation matrix has unbound cell {spec.comparison}"
                )
            if (
                spec.schedule_sha != artifact["sha256"]
                or spec.schedule_id != artifact["scheduleId"]
                or spec.suite_bank_sha != artifact["suiteBankSha256"]
                or spec.max_visits != artifact["visits"]
                or spec.look != canonical_look
                or spec.stage != canonical_stage
            ):
                raise SafetyHalt(
                    f"evaluation matrix cell contradicts manifest: "
                    f"{spec.comparison}"
                )
        config_hash = canonical_sha256(sorted({spec.config_sha for spec in specs}))
        schedule_hash = canonical_sha256(sorted({spec.schedule_sha for spec in specs}))
        evaluation_key = "matrix-" + canonical_sha256(
            [spec.to_dict() for spec in specs]
        )
        return EvaluationMatrixPlan(
            specs=specs,
            candidate_hash=candidate_hash,
            champion_hash=champion_hash,
            original_hash=config.original_hash,
            evaluation_key=evaluation_key,
            config_hash=config_hash,
            schedule_hash=schedule_hash,
            policy_hash=config.policy_hash,
            selfplay_config_hash=config.selfplay_config_hash,
            topology=required_topology,
            stage=canonical_stage,
            look=canonical_look,
            policy_path=str(self.runtime.policy_path),
            policy_version=policy_version,
            suite_manifest_path=str(manifest_path),
            suite_manifest_hash=config.suite_manifest_hash,
            schedule_artifacts=schedule_artifacts,
        )

    def evaluate_or_recommend(
        self, plan: EvaluationMatrixPlan, candidate: CandidateArtifact
    ) -> Mapping[str, Any]:
        """Return plans in recommendation mode or injected gate output in automatic mode."""

        if not plan.specs and plan.stage != "stage-0":
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

    def _stage0_request(
        self,
        plan: EvaluationMatrixPlan,
        candidate: CandidateArtifact,
        champion_hash: str,
    ) -> Optional[Tuple[Path, str]]:
        if plan.stage != "stage-0":
            return None
        stage_policy = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "stage_0_integrity_and_fixed_probes",
        )
        if not isinstance(stage_policy, Mapping):
            manifest, _ = self._load_suite_manifest()
            if self._manifest_cells(manifest):
                raise SafetyHalt("frozen policy has no Stage-0 probe contract")
            stage_policy = {
                "legacy_adapter_compatibility": True,
                "required_checks": [],
            }
        executable_bindings: Dict[str, Any] = {}
        stage0_command = self.runtime.commands.get("stage0Probe", ())
        for flag, path_key, hash_key in (
            ("--katago", "katago_binary_path", "katago_binary_hash"),
            (
                "--analysis-config",
                "analysis_config_path",
                "analysis_config_hash",
            ),
        ):
            if flag not in stage0_command:
                continue
            index = stage0_command.index(flag) + 1
            if index >= len(stage0_command) or "{" in stage0_command[index]:
                raise SafetyHalt(f"Stage-0 {flag} must be a fixed immutable path")
            executable = Path(stage0_command[index])
            if executable.is_symlink() or not executable.is_file():
                raise SafetyHalt(f"Stage-0 {flag} path is not a regular file")
            executable_bindings[path_key] = str(executable)
            executable_bindings[hash_key] = sha256_file(executable)
        request = {
            "schema_version": 1,
            "contract": "risk-score-stage-0-request-v1",
            "candidate_hash": candidate.model_hash,
            "checkpoint_hash": candidate.checkpoint_hash,
            "candidate_manifest_hash": candidate.directory_manifest_hash,
            "tested_champion_hash": champion_hash,
            "original_hash": self.runtime.controller.original_hash,
            "policy_path": plan.policy_path,
            "policy_hash": plan.policy_hash,
            "policy_version": plan.policy_version,
            "suite_manifest_path": plan.suite_manifest_path,
            "suite_manifest_hash": plan.suite_manifest_hash,
            "config_hash": plan.config_hash,
            "powered_config_path": str(self.runtime.powered_config_path),
            "powered_config_hash": self.runtime.controller.powered_config_hash,
            "standard_config_path": str(self.runtime.standard_config_path),
            "standard_config_hash": self.runtime.controller.standard_config_hash,
            "evaluation_key": plan.evaluation_key,
            "stage": plan.stage,
            "look": plan.look,
            "probe_contract": dict(stage_policy),
            "schedule_artifacts": {
                key: dict(value)
                for key, value in sorted(plan.schedule_artifacts.items())
            },
            **executable_bindings,
        }
        request_hash = canonical_sha256(request)
        root = self.runtime.evaluations / "stage-0" / "requests"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{request_hash}.json"
        _write_immutable_json(path, request)
        return path, sha256_file(path)

    def _resolve_tested_champion_model_path(self, champion_hash: str) -> Path:
        """Resolve the exact registry-bound model used as the tested champion."""

        _hash(champion_hash, "tested champion hash")
        if champion_hash == self.runtime.controller.original_hash:
            model = self.runtime.original_model_path
            if (
                model.is_symlink()
                or not model.is_file()
                or sha256_file(model) != champion_hash
            ):
                raise SafetyHalt(
                    "configured original champion model path/hash changed"
                )
            return model

        state = self.registry.reconstruct()
        if state.current_champion_hash != champion_hash:
            raise SafetyHalt(
                "configured evaluator plan names a non-current champion"
            )
        record = state.candidates.get(champion_hash)
        generation = state.generations.get(state.current_generation_id)
        if (
            record is None
            or generation is None
            or generation.state != GenerationState.ACTIVE
            or generation.candidate_hash != champion_hash
            or generation.candidate_path != record.candidate_path
        ):
            raise SafetyHalt(
                "non-original champion has no exact active registry path"
            )
        accepted = Path(record.candidate_path)
        if (
            not accepted.is_absolute()
            or accepted.parent != self.runtime.accepted_models
            or accepted.is_symlink()
            or not accepted.is_dir()
        ):
            raise SafetyHalt(
                "registry champion path is not an exact accepted candidate path"
            )
        artifact = inspect_candidate(accepted)
        model = accepted / "model.bin.gz"
        if (
            artifact.model_hash != champion_hash
            or model.is_symlink()
            or not model.is_file()
            or sha256_file(model) != champion_hash
        ):
            raise SafetyHalt(
                "registry champion model path/hash contradicts champion identity"
            )
        return model

    def _validate_stage0_probe_artifact(
        self,
        *,
        probe_path: Path,
        probe_hash: str,
        request_path: Path,
        request_hash: str,
        candidate_hash: str,
        champion_hash: str,
    ) -> None:
        try:
            from risk_score.promotion_evidence import validate_stage0_probe

            validate_stage0_probe(
                probe_path,
                expected_sha256=probe_hash,
                policy=self.runtime.frozen_policy,
                candidate_hash=candidate_hash,
                champion_hash=champion_hash,
                original_hash=self.runtime.controller.original_hash,
                request_path=request_path,
                request_sha256=request_hash,
            )
            request_bytes = request_path.read_bytes()
            request = json.loads(request_bytes)
            expected_request_bindings = {
                "policy_path": str(self.runtime.policy_path),
                "policy_hash": self.runtime.controller.policy_hash,
                "policy_version":
                    self.runtime.frozen_policy.get("policy_version"),
                "suite_manifest_path":
                    str(self.runtime.suites / "manifest.json"),
                "suite_manifest_hash":
                    self.runtime.controller.suite_manifest_hash,
                "config_hash": canonical_sha256(
                    sorted(
                        {
                            self.runtime.controller.powered_config_hash,
                            self.runtime.controller.standard_config_hash,
                        }
                    )
                ),
                "powered_config_path": str(self.runtime.powered_config_path),
                "powered_config_hash":
                    self.runtime.controller.powered_config_hash,
                "standard_config_path": str(self.runtime.standard_config_path),
                "standard_config_hash":
                    self.runtime.controller.standard_config_hash,
            }
            if (
                not isinstance(request, dict)
                or request_bytes != canonical_json_bytes(request) + b"\n"
                or any(
                    request.get(key) != expected
                    for key, expected in expected_request_bindings.items()
                )
            ):
                raise ValueError(
                    "Stage-0 request contradicts frozen runtime inputs"
                )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SafetyHalt(
                f"configured Stage-0 probe output is invalid: {exc}"
            ) from exc

    def _configured_stage0_probe(
        self,
        plan: EvaluationMatrixPlan,
        candidate: CandidateArtifact,
        *,
        champion_hash: str,
        champion_model_path: Path,
        request_path: Path,
        request_hash: str,
    ) -> Tuple[Path, str]:
        """Execute or reconcile the one probe artifact bound to a Stage-0 request."""

        root = self.runtime.evaluations / "stage-0" / "probes"
        root.mkdir(parents=True, exist_ok=True)
        probe_path = root / f"{request_hash}.json"
        if not probe_path.exists():
            temporary_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{request_hash}.probe-", dir=str(root)
                )
            )
            command_output = temporary_root / "probe.json"
            try:
                self.execute_argv(
                    "stage0Probe",
                    {
                        "plan": (
                            self.runtime.evaluations
                            / "controller-adapter"
                            / plan.evaluation_key
                            / "plan.json"
                        ),
                        "evaluation_key": plan.evaluation_key,
                        "candidate_dir": candidate.path,
                        "candidate_model": candidate.path / "model.bin.gz",
                        "candidate_hash": candidate.model_hash,
                        "champion_model": champion_model_path,
                        "champion_hash": champion_hash,
                        "original_model": self.runtime.original_model_path,
                        "original_hash": self.runtime.controller.original_hash,
                        "powered_config": self.runtime.powered_config_path,
                        "powered_config_hash":
                            self.runtime.controller.powered_config_hash,
                        "standard_config": self.runtime.standard_config_path,
                        "standard_config_hash":
                            self.runtime.controller.standard_config_hash,
                        "policy": plan.policy_path,
                        "policy_hash": plan.policy_hash,
                        "policy_version": plan.policy_version,
                        "suite_manifest": plan.suite_manifest_path,
                        "suite_manifest_hash": plan.suite_manifest_hash,
                        "stage0_request": request_path,
                        "stage0_request_hash": request_hash,
                        "stage0_request_sha256": request_hash,
                        "stage0_probe": command_output,
                        "stage0_probe_output": command_output,
                    },
                )
                if command_output.is_symlink() or not command_output.is_file():
                    raise SafetyHalt(
                        "configured Stage-0 probe did not publish a regular "
                        "output file"
                    )
                command_hash = sha256_file(command_output)
                self._validate_stage0_probe_artifact(
                    probe_path=command_output,
                    probe_hash=command_hash,
                    request_path=request_path,
                    request_hash=request_hash,
                    candidate_hash=candidate.model_hash,
                    champion_hash=champion_hash,
                )
                try:
                    probe_value = json.loads(
                        command_output.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SafetyHalt(
                        f"configured Stage-0 probe output is invalid: {exc}"
                    ) from exc
                if not isinstance(probe_value, dict):
                    raise SafetyHalt(
                        "configured Stage-0 probe output root must be an object"
                    )
                _write_immutable_json(probe_path, probe_value)
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)
        if probe_path.is_symlink() or not probe_path.is_file():
            raise SafetyHalt(
                "configured Stage-0 probe did not publish a regular output file"
            )
        probe_hash = sha256_file(probe_path)
        self._validate_stage0_probe_artifact(
            probe_path=probe_path,
            probe_hash=probe_hash,
            request_path=request_path,
            request_hash=request_hash,
            candidate_hash=candidate.model_hash,
            champion_hash=champion_hash,
        )
        return probe_path, probe_hash

    def configured_evaluation_executor(
        self, plan: EvaluationMatrixPlan, candidate: CandidateArtifact
    ) -> Mapping[str, Any]:
        """Run the configured shell-free evaluator adapter and verify its envelope."""

        if self.recommendation_only:
            raise SafetyHalt("configured evaluation execution requires automatic mode")
        if not plan.specs and plan.stage != "stage-0":
            raise SafetyHalt("configured evaluation has no matrix specifications")
        if (
            plan.candidate_hash != candidate.model_hash
            or plan.original_hash != self.runtime.controller.original_hash
        ):
            raise SafetyHalt("configured evaluation model identities contradict plan")
        root = self.runtime.evaluations / "controller-adapter" / plan.evaluation_key
        root.mkdir(parents=True, exist_ok=True)
        plan_path = root / "plan.json"
        evidence_path = root / "evidence.json"
        _write_immutable_json(plan_path, plan.to_dict())
        champion_spec = next(
            (
                spec
                for spec in plan.specs
                if spec.comparison == "candidate-vs-champion-powered"
            ),
            None,
        )
        if champion_spec is None and plan.stage != "stage-0":
            raise SafetyHalt("evaluation matrix has no champion comparison")
        tested_champion_hash = (
            plan.champion_hash
            if champion_spec is None
            else champion_spec.reference_model_sha
        )
        if tested_champion_hash != plan.champion_hash:
            raise SafetyHalt("champion comparison contradicts plan identity")
        champion_model_path = self._resolve_tested_champion_model_path(
            tested_champion_hash
        )
        stage0_request = self._stage0_request(
            plan, candidate, tested_champion_hash
        )

        if not evidence_path.exists():
            powered_champion = plan.schedule_artifacts.get(
                "powered_candidate_vs_champion"
            )
            powered_original = plan.schedule_artifacts.get(
                "powered_candidate_vs_original"
            )
            standard_original = plan.schedule_artifacts.get(
                "standard_candidate_vs_original"
            )
            lead40 = plan.schedule_artifacts.get("lead_40")
            lead80 = plan.schedule_artifacts.get("lead_80")

            integrity_result = self._stage_result(
                candidate.model_hash, "integrity"
            )
            screen_result = self._stage_result(candidate.model_hash, "screen")
            finalist_result = self._stage_result(
                candidate.model_hash, "finalist"
            )
            integrity_artifacts = (
                integrity_result.get("adapter_artifacts", {})
                if isinstance(integrity_result, Mapping)
                else {}
            )
            finalist_artifacts = (
                finalist_result.get("adapter_artifacts", {})
                if isinstance(finalist_result, Mapping)
                else {}
            )

            if plan.stage == "stage-0":
                if stage0_request is None:
                    raise SafetyHalt("Stage-0 evaluation has no immutable request")
                stage0_request_path = stage0_request[0]
                stage0_request_hash = stage0_request[1]
                stage0_probe_path, stage0_probe_hash = (
                    self._configured_stage0_probe(
                        plan,
                        candidate,
                        champion_hash=tested_champion_hash,
                        champion_model_path=champion_model_path,
                        request_path=stage0_request_path,
                        request_hash=stage0_request_hash,
                    )
                )
            else:
                try:
                    stage0_request_path = Path(
                        integrity_artifacts["stage0_request_path"]
                    )
                    stage0_request_hash = _hash(
                        integrity_artifacts["stage0_request_hash"],
                        "reused Stage-0 request hash",
                    )
                    stage0_probe_path = Path(
                        integrity_artifacts["stage0_probe_path"]
                    )
                    stage0_probe_hash = _hash(
                        integrity_artifacts.get(
                            "stage0_probe_sha256",
                            integrity_artifacts.get("stage0_probe_hash"),
                        ),
                        "reused Stage-0 probe hash",
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise SafetyHalt(
                        "later evaluation stage lacks finalized Stage-0 artifacts"
                    ) from exc
                self._validate_stage0_probe_artifact(
                    probe_path=stage0_probe_path,
                    probe_hash=stage0_probe_hash,
                    request_path=stage0_request_path,
                    request_hash=stage0_request_hash,
                    candidate_hash=candidate.model_hash,
                    champion_hash=tested_champion_hash,
                )

            stage1_evidence_path: Any = ""
            stage1_evidence_hash = ""
            discovery_output: Any = ""
            if plan.stage == "stage-2":
                if not isinstance(screen_result, Mapping):
                    raise SafetyHalt(
                        "Stage-2 evaluation lacks finalized Stage-1 evidence"
                    )
                screen_artifacts = screen_result.get("adapter_artifacts")
                if not isinstance(screen_artifacts, Mapping):
                    raise SafetyHalt(
                        "Stage-1 result has no evaluator artifact binding"
                    )
                stage1_key = screen_result.get("evaluation_key")
                if not isinstance(stage1_key, str) or not stage1_key:
                    raise SafetyHalt(
                        "Stage-1 result has no immutable evaluation key"
                    )
                stage1_evidence_path = (
                    self.runtime.evaluations
                    / "controller-adapter"
                    / stage1_key
                    / "evidence.json"
                )
                recorded_stage1_path = screen_artifacts.get(
                    "evaluator_evidence_path"
                )
                recorded_stage1_hash = screen_artifacts.get(
                    "evaluator_evidence_hash"
                )
                if (
                    recorded_stage1_path != str(stage1_evidence_path)
                    or not isinstance(recorded_stage1_hash, str)
                    or _SHA_RE.fullmatch(recorded_stage1_hash) is None
                    or stage1_evidence_path.is_symlink()
                    or not stage1_evidence_path.is_file()
                    or sha256_file(stage1_evidence_path)
                    != recorded_stage1_hash
                ):
                    raise SafetyHalt(
                        "Stage-2 evaluation cannot resolve Stage-1 evidence"
                    )
                stage1_evidence_hash = recorded_stage1_hash
                discovery_output = root / "discovery-evidence.json"

            attempt_path: Any = ""
            attempt_hash = ""
            if plan.stage == "stage-3":
                attempt_path = root / "attempt.json"
                state = self.registry.reconstruct()
                attempt = {
                    "attempt_number": 1,
                    "generation_id": state.current_generation_id,
                    "promotions_for_generation": 0,
                    "new_holdout_block": False,
                    "new_alpha_allocation": False,
                    "prespecified_cumulative_look": plan.look == "look-2",
                }
                _write_immutable_json(attempt_path, attempt)
                attempt_hash = sha256_file(attempt_path)

            discovery_evidence_path = finalist_artifacts.get(
                "discovery_evidence_path", ""
            )
            discovery_evidence_hash = finalist_artifacts.get(
                "discovery_evidence_sha256",
                finalist_artifacts.get("discovery_evidence_hash", ""),
            )
            runner_manifests_path = root / "runner-manifests.json"

            def schedule_value(
                artifact: Optional[Mapping[str, Any]], key: str
            ) -> Any:
                return "" if artifact is None else artifact[key]

            self.execute_argv(
                "evaluator",
                {
                    "plan": plan_path,
                    "plan_hash": sha256_file(plan_path),
                    "plan_sha256": sha256_file(plan_path),
                    "stage": plan.stage,
                    "look": plan.look,
                    "evaluation_key": plan.evaluation_key,
                    "evidence_output": evidence_path,
                    "candidate_dir": candidate.path,
                    "candidate_model": candidate.path / "model.bin.gz",
                    "candidate_hash": candidate.model_hash,
                    "champion_model": champion_model_path,
                    "champion_hash": tested_champion_hash,
                    "original_model": self.runtime.original_model_path,
                    "original_hash": self.runtime.controller.original_hash,
                    "policy": plan.policy_path,
                    "policy_hash": plan.policy_hash,
                    "policy_version": plan.policy_version,
                    "suite_manifest": plan.suite_manifest_path,
                    "suite_manifest_hash": plan.suite_manifest_hash,
                    "powered_config": self.runtime.powered_config_path,
                    "powered_config_hash":
                        self.runtime.controller.powered_config_hash,
                    "standard_config": self.runtime.standard_config_path,
                    "standard_config_hash":
                        self.runtime.controller.standard_config_hash,
                    "powered_schedule":
                        schedule_value(powered_champion, "path"),
                    "powered_schedule_hash":
                        schedule_value(powered_champion, "sha256"),
                    "powered_schedule_id":
                        schedule_value(powered_champion, "scheduleId"),
                    "powered_champion_schedule":
                        schedule_value(powered_champion, "path"),
                    "powered_champion_schedule_hash":
                        schedule_value(powered_champion, "sha256"),
                    "powered_champion_schedule_id":
                        schedule_value(powered_champion, "scheduleId"),
                    "powered_original_schedule":
                        schedule_value(powered_original, "path"),
                    "powered_original_schedule_hash":
                        schedule_value(powered_original, "sha256"),
                    "powered_original_schedule_id":
                        schedule_value(powered_original, "scheduleId"),
                    "standard_schedule":
                        schedule_value(standard_original, "path"),
                    "standard_schedule_hash":
                        schedule_value(standard_original, "sha256"),
                    "standard_schedule_id":
                        schedule_value(standard_original, "scheduleId"),
                    "standard_original_schedule":
                        schedule_value(standard_original, "path"),
                    "standard_original_schedule_hash":
                        schedule_value(standard_original, "sha256"),
                    "standard_original_schedule_id":
                        schedule_value(standard_original, "scheduleId"),
                    "lead40_schedule": schedule_value(lead40, "path"),
                    "lead40_schedule_hash":
                        schedule_value(lead40, "sha256"),
                    "lead40_schedule_id":
                        schedule_value(lead40, "scheduleId"),
                    "lead80_schedule": schedule_value(lead80, "path"),
                    "lead80_schedule_hash":
                        schedule_value(lead80, "sha256"),
                    "lead80_schedule_id":
                        schedule_value(lead80, "scheduleId"),
                    "stage0_request": stage0_request_path,
                    "stage0_request_hash": stage0_request_hash,
                    "stage0_request_sha256": stage0_request_hash,
                    "stage0_probe": stage0_probe_path,
                    "stage0_probe_hash": stage0_probe_hash,
                    "stage0_probe_sha256": stage0_probe_hash,
                    "stage1_evidence": stage1_evidence_path,
                    "stage1_evidence_hash": stage1_evidence_hash,
                    "stage1_evidence_sha256": stage1_evidence_hash,
                    "discovery_output": discovery_output,
                    "discovery_evidence": discovery_evidence_path,
                    "discovery_evidence_hash": discovery_evidence_hash,
                    "discovery_evidence_sha256":
                        discovery_evidence_hash,
                    "runner_manifests": runner_manifests_path,
                    "runner_manifests_output": runner_manifests_path,
                    "attempt": attempt_path,
                    "attempt_hash": attempt_hash,
                    "attempt_sha256": attempt_hash,
                    "evaluations_root": self.runtime.evaluations,
                    "evaluation_root": self.runtime.evaluations,
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

        controller_stage = {
            "stage-0": "integrity",
            "stage-1": "screen",
            "stage-2": "finalist",
            "stage-3": "confirmation",
        }[plan.stage]
        expected = {
            "schema_version": 1,
            "finalized": True,
            "controller_stage": controller_stage,
            "candidate_hash": candidate.model_hash,
            "tested_champion_hash": tested_champion_hash,
            "original_hash": self.runtime.controller.original_hash,
            "evaluation_key": plan.evaluation_key,
            "config_hash": plan.config_hash,
            "schedule_hash": plan.schedule_hash,
            "policy_hash": plan.policy_hash,
            "policy_path": plan.policy_path,
            "policy_version": plan.policy_version,
            "suite_manifest_path": plan.suite_manifest_path,
            "suite_manifest_hash": plan.suite_manifest_hash,
            "look": plan.look,
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
        supplied_artifacts = evidence.get("schedule_artifacts")
        if (
            not isinstance(supplied_artifacts, Mapping)
            or {
                key: dict(value)
                for key, value in supplied_artifacts.items()
                if isinstance(value, Mapping)
            }
            != {
                key: dict(value)
                for key, value in plan.schedule_artifacts.items()
            }
        ):
            raise SafetyHalt(
                "configured evaluator evidence omits exact schedule artifacts"
            )
        runner_manifests_path = (
            self.runtime.evaluations
            / "controller-adapter"
            / plan.evaluation_key
            / "runner-manifests.json"
        )
        runner_manifests_hash = evidence.get("runner_manifests_hash")
        if (
            evidence.get("runner_manifests_path")
            != str(runner_manifests_path)
            or not isinstance(runner_manifests_hash, str)
            or _SHA_RE.fullmatch(runner_manifests_hash) is None
            or runner_manifests_path.is_symlink()
            or not runner_manifests_path.is_file()
            or sha256_file(runner_manifests_path) != runner_manifests_hash
        ):
            raise SafetyHalt(
                "configured evaluator runner manifest map is missing or changed"
            )
        try:
            runner_map_bytes = runner_manifests_path.read_bytes()
            runner_map = json.loads(runner_map_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(
                f"configured evaluator runner manifest map is invalid: {exc}"
            ) from exc
        if (
            not isinstance(runner_map, dict)
            or runner_map_bytes != canonical_json_bytes(runner_map) + b"\n"
            or set(runner_map) != set(plan.schedule_artifacts)
            or not all(
                isinstance(value, str)
                and Path(value).is_absolute()
                and not Path(value).is_symlink()
                and Path(value).is_file()
                for value in runner_map.values()
            )
        ):
            raise SafetyHalt(
                "configured evaluator runner manifest map is not canonical/complete"
            )

        try:
            evidence_request_path = Path(evidence["stage0_request_path"])
            evidence_request_hash = _hash(
                evidence["stage0_request_hash"],
                "configured evidence Stage-0 request hash",
            )
            evidence_probe_path = Path(evidence["stage0_probe_path"])
            evidence_probe_hash = _hash(
                evidence.get(
                    "stage0_probe_sha256",
                    evidence.get("stage0_probe_hash"),
                ),
                "configured evidence Stage-0 probe hash",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetyHalt(
                "configured evaluator omits finalized Stage-0 provenance"
            ) from exc
        self._validate_stage0_probe_artifact(
            probe_path=evidence_probe_path,
            probe_hash=evidence_probe_hash,
            request_path=evidence_request_path,
            request_hash=evidence_request_hash,
            candidate_hash=candidate.model_hash,
            champion_hash=tested_champion_hash,
        )
        if stage0_request is not None:
            request_path, request_hash = stage0_request
            if (
                evidence_request_path != request_path
                or evidence_request_hash != request_hash
            ):
                raise SafetyHalt(
                    "configured evaluator Stage-0 provenance contradicts request"
                )
        return {
            **evidence,
            "candidate_sample_count": candidate.sample_count,
            "candidate_manifest_hash": candidate.directory_manifest_hash,
            "evaluator_evidence_path": str(evidence_path),
            "evaluator_evidence_hash": sha256_file(evidence_path),
        }

    def _controller_result_path(
        self, candidate_hash: str, stage: str, look: Optional[str] = None
    ) -> Path:
        root = self.runtime.evaluations / "controller-results" / candidate_hash
        if stage == "confirmation":
            if look is None:
                raise ValueError("confirmation result path requires a look")
            name = f"confirmation-{_canonical_confirmation_look(look)}.json"
        else:
            name = f"{stage}.json"
        return root / name

    @staticmethod
    def _next_action(
        decision: str, look: str, gate: Mapping[str, Any]
    ) -> str:
        supplied = gate.get("next_action")
        allowed = {
            "PROMOTE",
            "CONTINUE_TO_LOOK_2",
            "STOP_HARM",
            "STOP_MAXIMUM_INCONCLUSIVE",
        }
        if supplied is None:
            supplied = (
                "PROMOTE"
                if decision == "PASS"
                else "STOP_HARM"
                if decision == "FAIL"
                else "CONTINUE_TO_LOOK_2"
                if look == "look-1"
                else "STOP_MAXIMUM_INCONCLUSIVE"
            )
        if supplied not in allowed:
            raise SafetyHalt(f"unknown gate next_action: {supplied!r}")
        expected_by_decision = {
            "PASS": {"PROMOTE"},
            "FAIL": {"STOP_HARM"},
            "INCONCLUSIVE": {
                "CONTINUE_TO_LOOK_2"
                if look == "look-1"
                else "STOP_MAXIMUM_INCONCLUSIVE"
            },
        }
        if supplied not in expected_by_decision[decision]:
            raise SafetyHalt(
                f"gate decision {decision} contradicts next_action {supplied}"
            )
        return supplied

    def _reconcile_confirmation_result(
        self,
        result: Mapping[str, Any],
        artifact: CandidateArtifact,
        plan: EvaluationMatrixPlan,
        champion_hash: str,
    ) -> None:
        if (
            result.get("candidate_hash") != artifact.model_hash
            or result.get("tested_champion_hash") != champion_hash
            or result.get("evaluation_key") != plan.evaluation_key
            or result.get("look") != plan.look
        ):
            raise SafetyHalt(
                "persisted confirmation result contradicts active evaluation"
            )
        decision = result.get("decision")
        gate = result.get("gate")
        if (
            decision not in {"PASS", "FAIL", "INCONCLUSIVE"}
            or not isinstance(gate, Mapping)
        ):
            raise SafetyHalt("persisted confirmation result is malformed")
        next_action = self._next_action(decision, plan.look, gate)
        current = self.registry.reconstruct()
        record = current.candidates.get(artifact.model_hash)
        if record is None:
            raise SafetyHalt("confirmation result names an unknown candidate")
        if record.state in {
            CandidateState.CONFIRMED,
            CandidateState.REJECTED,
            CandidateState.QUARANTINED,
        }:
            return
        if (
            current.current_champion_hash != champion_hash
            or record.state != CandidateState.EVALUATING_CONFIRMATION
            or record.evaluation_key != plan.evaluation_key
        ):
            raise SafetyHalt(
                "confirmation result no longer matches lifecycle state"
            )
        provenance = self._provenance(plan.config_hash, plan.schedule_hash)
        if next_action == "PROMOTE":
            report_hash = result.get("report_hash")
            _hash(report_hash, "confirmation report hash")
            self.registry.transition_candidate(
                artifact.model_hash,
                str(artifact.path),
                CandidateState.CONFIRMED,
                provenance=provenance,
                champion_hash=champion_hash,
                evaluation_key=plan.evaluation_key,
                reason=f"finalized {plan.look} confirmation PASS",
                actor=self.runtime.controller.actor,
                payload={
                    "look": plan.look,
                    "report_hash": report_hash,
                },
            )
        elif next_action in {"STOP_HARM", "STOP_MAXIMUM_INCONCLUSIVE"}:
            self._move_candidate_terminal(
                artifact,
                CandidateState.REJECTED,
                provenance=provenance,
                champion_hash=champion_hash,
                evaluation_key=plan.evaluation_key,
                reason=(
                    "finalized confirmation harm"
                    if next_action == "STOP_HARM"
                    else "maximum confirmation look remained inconclusive"
                ),
            )

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
        if self.suite_registry is not None:
            self._validate_suite_runtime_binding()
        state = self.registry.reconstruct()
        candidate_record = state.candidates.get(candidate_hash)
        if candidate_record is None:
            raise SafetyHalt(f"unknown candidate: {candidate_hash}")
        canonical_look = (
            _canonical_confirmation_look(look)
            if stage == "confirmation"
            else look
        )
        if (
            candidate_record.tested_champion_hash is not None
            and candidate_record.state
            in {
                CandidateState.EVALUATING_INTEGRITY,
                CandidateState.EVALUATING_SCREEN,
                CandidateState.EVALUATING_FINALIST,
                CandidateState.EVALUATING_CONFIRMATION,
            }
            and candidate_record.tested_champion_hash != state.current_champion_hash
        ):
            raise SafetyHalt(
                "candidate was evaluated against a stale champion"
            )
        artifact = inspect_candidate(Path(candidate_record.candidate_path))
        plan = self.build_evaluation_plan(
            candidate_hash,
            state.current_champion_hash,
            suite=suite,
            stage=stage,
            look=canonical_look,
            topology=topology,
        )
        if self.recommendation_only:
            return {"decision": "RECOMMEND", "plan": plan.to_dict()}
        if (
            self.suite_registry is not None
            and candidate_record.state != CandidateState.CLAIMED
        ):
            expected_pin_champion = candidate_record.tested_champion_hash
            if expected_pin_champion is None:
                raise SafetyHalt(
                    "started candidate has no champion for suite pin"
                )
            with self._writer_lock():
                self._ensure_suite_evaluation_pin_locked(
                    candidate_hash,
                    expected_pin_champion,
                )
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
        result_path = self._controller_result_path(
            candidate_hash,
            stage,
            plan.look if stage == "confirmation" else None,
        )
        if result_path.is_file():
            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                not isinstance(persisted, dict)
                or persisted.get("candidate_hash") != candidate_hash
                or persisted.get("evaluation_key") != plan.evaluation_key
                or persisted.get("tested_champion_hash")
                != state.current_champion_hash
            ):
                raise SafetyHalt(
                    f"persisted evaluation result conflicts: {result_path}"
                )
            if stage == "confirmation":
                with self._writer_lock():
                    self._reconcile_confirmation_result(
                        persisted,
                        artifact,
                        plan,
                        state.current_champion_hash,
                    )
            report_path = self.runtime.reports / f"{plan.evaluation_key}.final.json"
            return {
                "decision": persisted["decision"],
                "nextAction": persisted.get("next_action"),
                "evaluationKey": plan.evaluation_key,
                "stage": stage,
                "look": plan.look,
                "reportPath": (
                    str(report_path)
                    if stage == "confirmation" and report_path.is_file()
                    else None
                ),
                "reportHash": (
                    sha256_file(report_path)
                    if stage == "confirmation" and report_path.is_file()
                    else None
                ),
                "reused": True,
            }
        if stage == "confirmation":
            report_path = (
                self.runtime.reports
                / f"{plan.evaluation_key}.final.json"
            )
            if report_path.is_file():
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                report_hash = sha256_file(report_path)
                gate = report.get("gate")
                if (
                    not isinstance(report, dict)
                    or report.get("finalized") is not True
                    or report.get("candidate_hash") != candidate_hash
                    or report.get("tested_champion_hash")
                    != state.current_champion_hash
                    or report.get("evaluation_key")
                    != plan.evaluation_key
                    or report.get("policy_hash") != plan.policy_hash
                    or report.get("look") != plan.look
                    or report.get("matrix") != plan.to_dict()
                    or not isinstance(gate, Mapping)
                ):
                    raise SafetyHalt(
                        "orphaned confirmation report contradicts active plan"
                    )
                decision = report.get("decision")
                if decision not in {
                    "PASS",
                    "FAIL",
                    "INCONCLUSIVE",
                }:
                    raise SafetyHalt(
                        "orphaned confirmation report decision is invalid"
                    )
                next_action = self._next_action(
                    decision, plan.look, gate
                )
                persisted = {
                    "schema_version": 1,
                    "candidate_hash": candidate_hash,
                    "tested_champion_hash":
                        state.current_champion_hash,
                    "evaluation_key": plan.evaluation_key,
                    "stage": stage,
                    "look": plan.look,
                    "decision": decision,
                    "next_action": next_action,
                    "policy_path": plan.policy_path,
                    "policy_hash": plan.policy_hash,
                    "policy_version": plan.policy_version,
                    "suite_manifest_path":
                        plan.suite_manifest_path,
                    "suite_manifest_hash":
                        plan.suite_manifest_hash,
                    "schedule_artifacts": {
                        key: dict(value)
                        for key, value
                        in plan.schedule_artifacts.items()
                    },
                    "evidence_hash": report.get(
                        "gate_evidence_hash",
                        canonical_sha256(gate),
                    ),
                    "adapter_artifacts": {},
                    "gate": dict(gate),
                    "report_path": str(report_path),
                    "report_hash": report_hash,
                }
                result_path.parent.mkdir(parents=True, exist_ok=True)
                _write_immutable_json(result_path, persisted)
                with self._writer_lock():
                    self._reconcile_confirmation_result(
                        persisted,
                        artifact,
                        plan,
                        state.current_champion_hash,
                    )
                return {
                    "decision": decision,
                    "nextAction": next_action,
                    "evaluationKey": plan.evaluation_key,
                    "stage": stage,
                    "look": plan.look,
                    "reportPath": str(report_path),
                    "reportHash": report_hash,
                    "reused": True,
                    "recovered": True,
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
            if self.suite_registry is not None:
                expected_pin_champion = (
                    candidate_record.tested_champion_hash
                    or state.current_champion_hash
                )
                self._ensure_suite_evaluation_pin_locked(
                    candidate_hash,
                    expected_pin_champion,
                )
            provenance = self._provenance(plan.config_hash, plan.schedule_hash)
            transition_payload: Dict[str, Any] = {
                "matrix": plan.to_dict(),
                "look": plan.look,
            }
            if stage == "confirmation":
                if plan.look == "look-2":
                    look1 = self._stage_result(
                        candidate_hash, "confirmation", look="look-1"
                    )
                    look1_gate = look1.get("gate") if look1 else None
                    if (
                        look1 is None
                        or look1.get("decision") != "INCONCLUSIVE"
                        or not isinstance(look1_gate, Mapping)
                        or self._next_action(
                            "INCONCLUSIVE", "look-1", look1_gate
                        )
                        != "CONTINUE_TO_LOOK_2"
                    ):
                        raise SafetyHalt(
                            "look-2 lacks a finalized look-1 continuation"
                        )
                    if (
                        candidate_record.state
                        != CandidateState.EVALUATING_CONFIRMATION
                    ):
                        raise SafetyHalt(
                            "look-2 requires an active look-1 confirmation"
                        )
                    transition_payload.update(
                        {
                            "previous_evaluation_key":
                                look1["evaluation_key"],
                            "previous_result_hash": sha256_file(
                                self._controller_result_path(
                                    candidate_hash,
                                    "confirmation",
                                    "look-1",
                                )
                            ),
                            "prespecified_cumulative_look": True,
                        }
                    )
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
            stage0_request = self._stage0_request(
                plan, artifact, state.current_champion_hash
            )
            if stage0_request is not None:
                transition_payload.update(
                    {
                        "stage0_request_path": str(stage0_request[0]),
                        "stage0_request_hash": stage0_request[1],
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
            if stage == "confirmation" and plan.look == "look-2":
                self._checkpoint("confirmation-look-2-started")
            handoff_proof: Any = None
            with self._exclusive_gpu_handoff(plan, artifact) as handoff_proof:
                raw_evidence = self.evaluation_executor(plan, artifact)
            verified_handoff = self._validate_gpu_handoff(handoff_proof)
            if not isinstance(raw_evidence, Mapping):
                raise SafetyHalt("evaluation executor returned no immutable envelope")
            exact_bindings = {
                "policy_path": plan.policy_path,
                "policy_hash": plan.policy_hash,
                "policy_version": plan.policy_version,
                "suite_manifest_path": plan.suite_manifest_path,
                "suite_manifest_hash": plan.suite_manifest_hash,
                "schedule_artifacts": {
                    key: dict(value)
                    for key, value in plan.schedule_artifacts.items()
                },
                "look": plan.look,
                "candidate_sample_count": artifact.sample_count,
                "candidate_manifest_hash":
                    artifact.directory_manifest_hash,
            }
            contradicted = [
                key
                for key, expected in exact_bindings.items()
                if key in raw_evidence and raw_evidence[key] != expected
            ]
            if contradicted:
                raise SafetyHalt(
                    "evaluation evidence contradicts runtime-bound inputs: "
                    + ", ".join(sorted(contradicted))
                )
            evidence = {
                **dict(raw_evidence),
                "gpu_handoff": verified_handoff,
                "gpu_handoff_hash": canonical_sha256(verified_handoff),
                "selfplay_config_hash": plan.selfplay_config_hash,
                "topology": plan.topology,
                "controller_stage": stage,
                **exact_bindings,
            }
            gate = dict(self.gate_evaluator(evidence))
            if gate.get("gpu_handoff_hash") != evidence["gpu_handoff_hash"]:
                raise SafetyHalt("gate result omits or contradicts GPU handoff proof")
            gate_bindings = {
                "policy_path": plan.policy_path,
                "policy_hash": plan.policy_hash,
                "policy_version": plan.policy_version,
                "suite_manifest_path": plan.suite_manifest_path,
                "suite_manifest_hash": plan.suite_manifest_hash,
                "look": plan.look,
            }
            gate_conflicts = [
                key
                for key, expected in gate_bindings.items()
                if key in gate and gate[key] != expected
            ]
            if gate_conflicts:
                raise SafetyHalt(
                    "gate result contradicts runtime-bound policy/manifest: "
                    + ", ".join(sorted(gate_conflicts))
                )
            for key, value in gate_bindings.items():
                gate.setdefault(key, value)
            gate["gpu_handoff"] = verified_handoff
            decision = gate.get("decision", "INCONCLUSIVE")
            if decision not in {"PASS", "FAIL", "INCONCLUSIVE"}:
                raise SafetyHalt(f"unknown gate decision: {decision!r}")
            next_action = (
                self._next_action(decision, plan.look, gate)
                if stage == "confirmation"
                else None
            )
            if next_action is not None:
                gate["next_action"] = next_action
            result: Dict[str, Any] = {
                "decision": decision,
                "nextAction": next_action,
                "evaluationKey": plan.evaluation_key,
                "stage": stage,
                "look": plan.look,
            }
            report_path: Optional[Path] = None
            report_hash: Optional[str] = None
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
            result_path.parent.mkdir(parents=True, exist_ok=True)
            adapter_artifacts = {
                key: evidence[key]
                for key in (
                    "stage0_request_path",
                    "stage0_request_hash",
                    "stage0_probe_path",
                    "stage0_probe_hash",
                    "stage0_probe_sha256",
                    "discovery_evidence_path",
                    "discovery_evidence_hash",
                    "discovery_evidence_sha256",
                    "runner_manifests_path",
                    "runner_manifests_hash",
                    "attempt_path",
                    "attempt_hash",
                    "evaluator_evidence_path",
                    "evaluator_evidence_hash",
                )
                if key in evidence
            }
            persisted_result = {
                "schema_version": 1,
                "candidate_hash": candidate_hash,
                "tested_champion_hash": state.current_champion_hash,
                "evaluation_key": plan.evaluation_key,
                "stage": stage,
                "look": plan.look,
                "decision": decision,
                "next_action": next_action,
                "policy_path": plan.policy_path,
                "policy_hash": plan.policy_hash,
                "policy_version": plan.policy_version,
                "suite_manifest_path": plan.suite_manifest_path,
                "suite_manifest_hash": plan.suite_manifest_hash,
                "schedule_artifacts": {
                    key: dict(value)
                    for key, value in plan.schedule_artifacts.items()
                },
                "evidence_hash": canonical_sha256(evidence),
                "adapter_artifacts": adapter_artifacts,
                "gate": gate,
                "report_path": (
                    None if report_path is None else str(report_path)
                ),
                "report_hash": report_hash,
            }
            _write_immutable_json(result_path, persisted_result)
            if stage == "confirmation":
                self._checkpoint(f"confirmation-{plan.look}-finalized")
                self._reconcile_confirmation_result(
                    persisted_result,
                    artifact,
                    plan,
                    state.current_champion_hash,
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
        optional_gate_bindings = {
            "policy_path": plan.policy_path,
            "policy_version": plan.policy_version,
            "suite_manifest_path": plan.suite_manifest_path,
            "suite_manifest_hash": plan.suite_manifest_hash,
            "look": plan.look,
        }
        conflicts = [
            key
            for key, expected in expected_gate_metadata.items()
            if gate_result.get(key) != expected
        ]
        conflicts.extend(
            key
            for key, expected in optional_gate_bindings.items()
            if key in gate_result and gate_result.get(key) != expected
        )
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
        if not plan.specs and plan.stage != "stage-0":
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
            "policy_path": plan.policy_path,
            "policy_version": plan.policy_version,
            "suite_manifest_path": plan.suite_manifest_path,
            "suite_manifest_hash": plan.suite_manifest_hash,
            "look": plan.look,
            "schedule_artifacts": {
                key: dict(value)
                for key, value in plan.schedule_artifacts.items()
            },
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
        if self.suite_registry is not None:
            if self._writer_lock_depth > 0:
                self._unpin_suite_evaluation_locked(candidate.model_hash)
            else:
                with self._writer_lock():
                    self._unpin_suite_evaluation_locked(
                        candidate.model_hash
                    )
        _write_immutable_json(
            intent_path.with_name(intent_path.stem + ".complete.json"),
            {
                "schema_version": 1,
                "candidate_hash": candidate.model_hash,
                "target_state": target.value,
            },
        )
        if target in {CandidateState.SUPERSEDED, CandidateState.REJECTED}:
            self._create_trash_intent(
                candidate.model_hash,
                destination,
                reason=f"{target.value}:{reason}",
                move=False,
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
            if self.suite_registry is not None:
                self._unpin_suite_evaluation_locked(
                    intent["candidate_hash"]
                )
            _write_immutable_json(
                complete,
                {
                    "schema_version": 1,
                    "candidate_hash": intent["candidate_hash"],
                    "target_state": target.value,
                },
            )
            if target in {
                CandidateState.SUPERSEDED,
                CandidateState.REJECTED,
            }:
                self._create_trash_intent(
                    intent["candidate_hash"],
                    destination,
                    reason=f"{target.value}:{intent['reason']}",
                    move=False,
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
        remaining_slots = self._effective_evaluator_limit() - active_existing
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

    @staticmethod
    def _storage_manifest(path: Path) -> Tuple[str, int]:
        if path.is_symlink():
            raise SafetyHalt(f"retention source is a symlink: {path}")
        if path.is_dir():
            return _tree_manifest(path)
        if path.is_file():
            return (
                canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "files": [
                            {
                                "path": path.name,
                                "size": path.stat().st_size,
                                "sha256": sha256_file(path),
                            }
                        ],
                    }
                ),
                path.stat().st_size,
            )
        raise SafetyHalt(f"retention source is not a regular file/tree: {path}")

    def _reference_reasons(self, reference_hash: str) -> Tuple[str, ...]:
        _hash(reference_hash, "retention reference hash")
        state = self.registry.reconstruct()
        reasons = set(state.retention_status(reference_hash).reasons)
        roots = (
            ("report", self.runtime.reports),
            ("evaluation", self.runtime.evaluations),
            ("transaction", self.runtime.promotion_root / "transactions"),
            ("audit", self.runtime.promotion_root / "audits"),
        )
        for kind, root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.json"):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    if reference_hash in path.read_text(encoding="utf-8"):
                        reasons.add(
                            f"{kind}:{path.relative_to(root).as_posix()}"
                        )
                except (OSError, UnicodeDecodeError) as exc:
                    raise SafetyHalt(
                        f"cannot inspect retention reference {path}: {exc}"
                    ) from exc
        return tuple(sorted(reasons))

    def _create_trash_intent(
        self,
        reference_hash: str,
        source_path: Path,
        *,
        reason: str,
        move: bool,
    ) -> Mapping[str, Any]:
        _hash(reference_hash, "trash reference hash")
        _nonempty(reason, "trash reason")
        source = Path(source_path)
        intent_path = (
            self.runtime.promotion_root
            / "trash"
            / "intents"
            / f"{reference_hash}.json"
        )
        if intent_path.exists():
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            if (
                intent.get("reference_hash") != reference_hash
                or intent.get("source_path") != str(source)
            ):
                raise SafetyHalt("trash intent identity conflicts with retry")
        else:
            manifest_hash, retained_bytes = self._storage_manifest(source)
            now = self.now()
            if now.tzinfo is None or now.utcoffset() is None:
                raise SafetyHalt("controller clock must return timezone-aware time")
            now = now.astimezone(timezone.utc)
            grace_days = _policy_get(
                self.runtime.frozen_policy,
                "retention",
                "trash_grace_period_days",
                default=30,
            )
            if type(grace_days) is not int or grace_days < 30:
                raise SafetyHalt(
                    "trash grace period must be at least 30 days"
                )
            destination = (
                self.runtime.promotion_root
                / "trash"
                / "objects"
                / reference_hash
                / source.name
            )
            intent = {
                "schema_version": 1,
                "contract": "reference-aware-trash-intent-v1",
                "reference_hash": reference_hash,
                "source_path": str(source),
                "destination_path": str(destination),
                "source_manifest_hash": manifest_hash,
                "retained_bytes": retained_bytes,
                "reason": reason,
                "created_at_utc": utc_timestamp(now),
                "delete_not_before_utc": utc_timestamp(
                    now + timedelta(days=grace_days)
                ),
                "grace_period_days": grace_days,
                "policy_hash": self.runtime.controller.policy_hash,
            }
            _write_immutable_json(intent_path, intent)
        if not move:
            return {
                "status": "INTENT_CREATED",
                "intentPath": str(intent_path),
                **intent,
            }
        return self._reconcile_trash_intent(intent_path, mutate=True)

    def schedule_trash(
        self,
        reference_hash: str,
        source_path: Path,
        *,
        reason: str,
    ) -> Mapping[str, Any]:
        """Move an unreferenced object to grace storage without deleting it."""

        if self.recommendation_only:
            return {
                "status": "RECOMMEND",
                "referenceHash": reference_hash,
                "sourcePath": str(source_path),
                "reason": reason,
            }
        with self._writer_lock():
            self.ensure_layout()
            return self._create_trash_intent(
                reference_hash,
                source_path,
                reason=reason,
                move=True,
            )

    def _reconcile_trash_intent(
        self, intent_path: Path, *, mutate: bool
    ) -> Mapping[str, Any]:
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        reference_hash = intent.get("reference_hash")
        _hash(reference_hash, "trash intent reference hash")
        if intent.get("policy_hash") != self.runtime.controller.policy_hash:
            raise SafetyHalt("trash intent policy hash changed")
        source = Path(intent["source_path"])
        destination = Path(intent["destination_path"])
        deleted_path = (
            self.runtime.promotion_root
            / "trash"
            / "deleted"
            / f"{reference_hash}.json"
        )
        if deleted_path.is_file():
            deleted = json.loads(deleted_path.read_text(encoding="utf-8"))
            if deleted.get("reference_hash") != reference_hash:
                raise SafetyHalt("trash deletion marker identity changed")
            return {
                "status": "DELETED",
                "referenceHash": reference_hash,
                "deletedAtUtc": deleted.get("deleted_at_utc"),
                "reused": True,
            }
        deletion_intent_path = (
            self.runtime.promotion_root
            / "trash"
            / "deletion-intents"
            / f"{reference_hash}.json"
        )
        if (
            not source.exists()
            and not destination.exists()
            and deletion_intent_path.is_file()
        ):
            deletion_intent = json.loads(
                deletion_intent_path.read_text(encoding="utf-8")
            )
            if (
                deletion_intent.get("reference_hash") != reference_hash
                or deletion_intent.get("object_path") != str(destination)
                or self._reference_reasons(reference_hash)
            ):
                raise SafetyHalt(
                    "trash deletion recovery has contradictory provenance"
                )
            if not mutate:
                return {
                    "status": "DELETION_COMMIT_PENDING",
                    "referenceHash": reference_hash,
                }
            _write_immutable_json(
                deleted_path,
                {
                    "schema_version": 1,
                    "reference_hash": reference_hash,
                    "deleted_at_utc":
                        deletion_intent["authorized_at_utc"],
                    "manifest_hash":
                        deletion_intent["manifest_file_hash"],
                },
            )
            return {
                "status": "DELETED",
                "referenceHash": reference_hash,
                "deletedAtUtc": deletion_intent[
                    "authorized_at_utc"
                ],
                "recovered": True,
            }
        reasons = self._reference_reasons(reference_hash)
        moved = destination.exists()
        if not moved and reasons:
            return {
                "status": "BLOCKED_REFERENCES",
                "referenceHash": reference_hash,
                "reasons": list(reasons),
                "deleteNotBeforeUtc": intent["delete_not_before_utc"],
            }
        if not moved:
            if not source.exists():
                raise SafetyHalt(
                    "trash source and destination are both missing"
                )
            if not mutate:
                return {
                    "status": "READY_TO_TRASH",
                    "referenceHash": reference_hash,
                    "deleteNotBeforeUtc": intent["delete_not_before_utc"],
                }
            destination.parent.mkdir(parents=True, exist_ok=True)
            _recoverable_rename(
                source,
                destination,
                expected_manifest_hash=intent["source_manifest_hash"],
                inspector=lambda path: self._storage_manifest(path)[0],
            )
            moved = True
        actual_hash, actual_bytes = self._storage_manifest(destination)
        if (
            actual_hash != intent["source_manifest_hash"]
            or actual_bytes != intent["retained_bytes"]
        ):
            raise SafetyHalt("trash object contradicts immutable intent")
        manifest_path = (
            self.runtime.promotion_root
            / "trash"
            / "manifests"
            / f"{reference_hash}.json"
        )
        manifest = {
            "schema_version": 1,
            "contract": "reference-aware-trash-manifest-v1",
            "reference_hash": reference_hash,
            "object_path": str(destination),
            "source_path": str(source),
            "manifest_hash": actual_hash,
            "retained_bytes": actual_bytes,
            "created_at_utc": intent["created_at_utc"],
            "delete_not_before_utc": intent["delete_not_before_utc"],
            "grace_period_days": intent["grace_period_days"],
            "policy_hash": intent["policy_hash"],
        }
        if mutate:
            _write_immutable_json(manifest_path, manifest)
        deadline = _parse_utc_timestamp(intent["delete_not_before_utc"])
        now = self.now().astimezone(timezone.utc)
        reasons = self._reference_reasons(reference_hash)
        if now < deadline:
            return {
                "status": "GRACE_PERIOD",
                "referenceHash": reference_hash,
                "objectPath": str(destination),
                "retainedBytes": actual_bytes,
                "deleteNotBeforeUtc": intent["delete_not_before_utc"],
                "reasons": list(reasons),
            }
        if reasons:
            return {
                "status": "BLOCKED_REFERENCES",
                "referenceHash": reference_hash,
                "objectPath": str(destination),
                "retainedBytes": actual_bytes,
                "reasons": list(reasons),
            }
        if not mutate:
            return {
                "status": "READY_TO_DELETE",
                "referenceHash": reference_hash,
                "objectPath": str(destination),
            }
        if deletion_intent_path.is_file():
            deletion_intent = json.loads(
                deletion_intent_path.read_text(encoding="utf-8")
            )
            expected_deletion = {
                "reference_hash": reference_hash,
                "object_path": str(destination),
                "object_manifest_hash": actual_hash,
                "manifest_file_hash": sha256_file(manifest_path),
                "delete_not_before_utc": intent["delete_not_before_utc"],
                "reference_reasons": [],
            }
            if any(
                deletion_intent.get(key) != value
                for key, value in expected_deletion.items()
            ):
                raise SafetyHalt("trash deletion intent changed on retry")
        else:
            deletion_intent = {
                "schema_version": 1,
                "reference_hash": reference_hash,
                "object_path": str(destination),
                "object_manifest_hash": actual_hash,
                "manifest_file_hash": sha256_file(manifest_path),
                "authorized_at_utc": utc_timestamp(now),
                "delete_not_before_utc": intent["delete_not_before_utc"],
                "reference_reasons": [],
            }
            _write_immutable_json(
                deletion_intent_path, deletion_intent
            )
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
        fsync_directory(destination.parent)
        _write_immutable_json(
            deleted_path,
            {
                "schema_version": 1,
                "reference_hash": reference_hash,
                "deleted_at_utc":
                    deletion_intent["authorized_at_utc"],
                "manifest_hash": sha256_file(manifest_path),
            },
        )
        return {
            "status": "DELETED",
            "referenceHash": reference_hash,
            "deletedAtUtc": deletion_intent["authorized_at_utc"],
        }

    def reconcile_trash(self, *, mutate: bool = False) -> Tuple[Mapping[str, Any], ...]:
        root = self.runtime.promotion_root / "trash" / "intents"
        if not root.exists():
            return ()
        return tuple(
            self._reconcile_trash_intent(path, mutate=mutate)
            for path in sorted(root.glob("*.json"))
        )

    def _effective_evaluator_limit(self) -> int:
        policy_limit = _policy_get(
            self.runtime.frozen_policy,
            "queue",
            "maximum_active_evaluator_entries",
            default=self.runtime.controller.max_active_queue,
        )
        if type(policy_limit) is not int or policy_limit <= 0:
            raise SafetyHalt("policy evaluator queue limit is invalid")
        return min(self.runtime.controller.max_active_queue, policy_limit)

    def _backpressure_status(self, state: Any) -> Mapping[str, Any]:
        candidates, ignored = inventory_candidates(self.runtime.candidate_inbox)
        active_states = {
            CandidateState.CLAIMED,
            CandidateState.EVALUATING_INTEGRITY,
            CandidateState.EVALUATING_SCREEN,
            CandidateState.EVALUATING_FINALIST,
            CandidateState.EVALUATING_CONFIRMATION,
        }
        active = [
            candidate
            for candidate in state.candidates.values()
            if candidate.state in active_states
        ]
        evaluator_limit = self._effective_evaluator_limit()
        warning_depth = _policy_get(
            self.runtime.frozen_policy,
            "queue",
            "important_queue_warning_depth",
            default=evaluator_limit + 1,
        )
        if type(warning_depth) is not int or warning_depth <= 0:
            raise SafetyHalt("policy queue warning depth is invalid")
        export_depth = len(candidates) + len(ignored)
        evaluation_depth = len(active)
        disk_free_bytes = int(self.disk_usage(self.runtime.promotion_root).free)
        minimum_free_bytes = self.runtime.controller.min_free_bytes
        disk_warning_bytes = max(
            minimum_free_bytes,
            math.ceil(minimum_free_bytes * 1.1),
        )
        disk_pressure = disk_free_bytes < disk_warning_bytes
        export_paused = (
            evaluation_depth >= evaluator_limit
            or export_depth >= warning_depth
            or disk_pressure
        )
        evaluation_paused = evaluation_depth >= evaluator_limit
        reasons = []
        if evaluation_depth >= evaluator_limit:
            reasons.append("active-evaluator-limit")
        if export_depth >= warning_depth:
            reasons.append("export-backlog-warning-depth")
        if disk_pressure:
            reasons.append("disk-reserve-approaching")
        return {
            "exportBacklogDepth": export_depth,
            "evaluationBacklogDepth": evaluation_depth,
            "maximumActiveEvaluatorEntries": evaluator_limit,
            "importantQueueWarningDepth": warning_depth,
            "diskFreeBytes": disk_free_bytes,
            "minimumFreeBytes": minimum_free_bytes,
            "diskWarningBytes": disk_warning_bytes,
            "exportPaused": export_paused,
            "evaluationPaused": evaluation_paused,
            "allowExport": not export_paused,
            "allowEvaluation": not evaluation_paused,
            "reasons": reasons,
        }

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_symlink() or not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            if child.is_symlink():
                continue
            if child.is_file():
                total += child.stat().st_size
        return total

    def _retained_bytes(self, state: Any) -> int:
        paths = {
            self.runtime.original_model_path,
            self.runtime.policy_path,
            self.runtime.powered_config_path,
            self.runtime.standard_config_path,
            self.runtime.selfplay_config_path,
            self.runtime.gpu_lease_config_path,
            self.runtime.suites / "manifest.json",
            self.runtime.reports,
            self.runtime.rollback_quarantine,
        }
        manifest, _ = self._load_suite_manifest()
        for bank in manifest.get("banks", []):
            if not isinstance(bank, Mapping):
                continue
            for role in ("positions", "schedule"):
                artifact = bank.get(role)
                relative = (
                    artifact.get("path")
                    if isinstance(artifact, Mapping)
                    else None
                )
                if isinstance(relative, str):
                    paths.add(self.runtime.suites / relative)
        for cell in self._manifest_cells(manifest):
            relative = self._cell_value(cell, "schedule_path", "path")
            if isinstance(relative, str):
                paths.add(self.runtime.suites / relative)
        if not self._manifest_cells(manifest):
            paths.update(
                {
                    self.runtime.discovery_schedule_path,
                    self.runtime.confirmation_schedule_path,
                    self.runtime.audit_schedule_path,
                    self.runtime.lead40_schedule_path,
                    self.runtime.lead80_schedule_path,
                    self.runtime.standard_confirmation_schedule_path,
                }
            )
        for candidate in state.candidates.values():
            if state.is_pinned(candidate.candidate_hash):
                paths.add(Path(candidate.candidate_path))
        for generation in state.generations.values():
            if generation.candidate_path is not None and state.is_pinned(
                generation.candidate_hash
            ):
                paths.add(Path(generation.candidate_path))
        for manifest_path in (
            self.runtime.promotion_root / "trash" / "manifests"
        ).glob("*.json"):
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths.add(Path(value["object_path"]))
        return sum(self._path_size(path) for path in paths)

    def _lease_status(self) -> Mapping[str, Any]:
        if not self.runtime.lock_path.exists():
            return {"owner": None, "pid": None, "acquiredAtUtc": None}
        try:
            value = json.loads(
                self.runtime.lock_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {
                "owner": None,
                "pid": None,
                "acquiredAtUtc": None,
                "status": "unreadable",
            }
        active = self._writer_lock_depth > 0
        if not active:
            descriptor = os.open(str(self.runtime.lock_path), os.O_RDONLY)
            try:
                try:
                    fcntl.flock(
                        descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError:
                    active = True
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        return {
            "owner": value.get("actor") if active else None,
            "pid": value.get("pid") if active else None,
            "acquiredAtUtc": (
                value.get("acquired_at_utc") if active else None
            ),
            "lastOwner": value.get("actor"),
        }

    def _active_evaluation_status(self, state: Any) -> List[Mapping[str, Any]]:
        evaluating = {
            CandidateState.EVALUATING_INTEGRITY,
            CandidateState.EVALUATING_SCREEN,
            CandidateState.EVALUATING_FINALIST,
            CandidateState.EVALUATING_CONFIRMATION,
        }
        rows = []
        for candidate in sorted(
            state.candidates.values(),
            key=lambda item: item.candidate_hash,
        ):
            if candidate.state not in evaluating:
                continue
            event = next(
                (
                    item
                    for item in reversed(state.events)
                    if item.candidate_hash == candidate.candidate_hash
                    and item.sequence == candidate.last_sequence
                ),
                None,
            )
            matrix = (
                event.payload.get("matrix")
                if event is not None
                and isinstance(event.payload.get("matrix"), Mapping)
                else {}
            )
            rows.append(
                {
                    "candidateHash": candidate.candidate_hash,
                    "state": candidate.state.value,
                    "stage": matrix.get(
                        "stage", candidate.state.value
                    ),
                    "look": (
                        event.payload.get("look")
                        if event is not None
                        else matrix.get("look")
                    ),
                    "evaluationKey": candidate.evaluation_key,
                    "startedAtUtc": (
                        event.timestamp_utc if event is not None else None
                    ),
                }
            )
        return rows

    def _worker_ack_status(self, state: Any) -> List[Mapping[str, Any]]:
        rows = []
        for generation in sorted(
            state.generations.values(),
            key=lambda item: item.generation_id,
        ):
            if generation.state not in {
                GenerationState.CANARY,
                GenerationState.ROLLOUT,
                GenerationState.ACTIVE,
                GenerationState.ROLLBACK_PENDING,
            } or generation.candidate_path is None:
                continue
            acknowledged = sorted(
                self._acknowledged(
                    generation.generation_id,
                    generation.candidate_hash,
                )
            )
            rows.append(
                {
                    "generationId": generation.generation_id,
                    "candidateHash": generation.candidate_hash,
                    "state": generation.state.value,
                    "acknowledgedWorkerIds": acknowledged,
                    "acknowledgedCount": len(acknowledged),
                    "expectedCount": self.runtime.controller.worker_count,
                }
            )
        return rows

    def record_promotion_feedback(
        self,
        generation_id: str,
        kind: str,
        *,
        evidence: Mapping[str, Any],
    ) -> Path:
        """Record a hash-bound first-data/shuffle/training feedback milestone."""

        if not self.automatic:
            raise SafetyHalt("promotion feedback mutation is disabled")
        allowed = {
            "first-game",
            "first-tdata",
            "first-shuffle",
            "first-training-consumption",
        }
        if kind not in allowed:
            raise ValueError(f"unknown promotion feedback kind: {kind}")
        if not isinstance(evidence, Mapping):
            raise ValueError("promotion feedback evidence must be an object")
        with self._writer_lock():
            transaction = self._transaction_dir(generation_id)
            state = self.registry.reconstruct()
            generation = state.generations.get(generation_id)
            if generation is None:
                raise SafetyHalt("promotion feedback names unknown generation")
            evidence_copy = dict(evidence)
            supplied_generation = evidence_copy.get("generation_id")
            supplied_candidate = evidence_copy.get("candidate_hash")
            if supplied_generation not in {None, generation_id}:
                raise SafetyHalt("promotion feedback generation changed")
            if supplied_candidate not in {None, generation.candidate_hash}:
                raise SafetyHalt("promotion feedback candidate changed")
            path = transaction / f"feedback-{kind}.json"
            evidence_hash = canonical_sha256(evidence_copy)
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if (
                    existing.get("generation_id") != generation_id
                    or existing.get("candidate_hash")
                    != generation.candidate_hash
                    or existing.get("kind") != kind
                    or existing.get("evidence_hash") != evidence_hash
                    or existing.get("evidence") != evidence_copy
                ):
                    raise SafetyHalt(
                        "promotion feedback retry changed immutable evidence"
                    )
                return path
            payload = {
                "schema_version": 1,
                "generation_id": generation_id,
                "candidate_hash": generation.candidate_hash,
                "kind": kind,
                "observed_at_utc": utc_timestamp(self.now()),
                "evidence": evidence_copy,
                "evidence_hash": evidence_hash,
            }
            _write_immutable_json(path, payload)
            return path

    def _promotion_feedback_status(self, state: Any) -> List[Mapping[str, Any]]:
        rows = []
        for generation in sorted(
            state.generations.values(),
            key=lambda item: item.generation_id,
        ):
            transaction = (
                self.runtime.promotion_root
                / "transactions"
                / generation.generation_id
            )
            if not transaction.exists():
                continue
            active_event = next(
                (
                    event
                    for event in reversed(state.events)
                    if event.transition == Transition.GENERATION_ACTIVATED
                    and event.payload.get("generation_id")
                    == generation.generation_id
                ),
                None,
            )
            feedback: Dict[str, Any] = {
                "generationId": generation.generation_id,
                "candidateHash": generation.candidate_hash,
                "promotedAtUtc": (
                    active_event.timestamp_utc
                    if active_event is not None
                    else None
                ),
            }
            for kind in (
                "first-game",
                "first-tdata",
                "first-shuffle",
                "first-training-consumption",
            ):
                path = transaction / f"feedback-{kind}.json"
                observed = (
                    json.loads(path.read_text(encoding="utf-8")).get(
                        "observed_at_utc"
                    )
                    if path.is_file()
                    else None
                )
                feedback[
                    kind.replace("-", "_") + "_at_utc"
                ] = observed
                camel = {
                    "first-game": "firstGameAtUtc",
                    "first-tdata": "firstTdataAtUtc",
                    "first-shuffle": "firstShuffleAtUtc",
                    "first-training-consumption":
                        "firstTrainingConsumptionAtUtc",
                }[kind]
                feedback[camel] = observed
                if (
                    observed is not None
                    and feedback["promotedAtUtc"] is not None
                ):
                    feedback[
                        camel.removesuffix("AtUtc") + "LatencySeconds"
                    ] = max(
                        0.0,
                        (
                            _parse_utc_timestamp(observed)
                            - _parse_utc_timestamp(
                                feedback["promotedAtUtc"]
                            )
                        ).total_seconds(),
                    )
            rows.append(feedback)
        return rows

    def _queue_status(self, state: Any) -> Mapping[str, Any]:
        candidates, ignored = inventory_candidates(self.runtime.candidate_inbox)
        paths = [item.path for item in candidates]
        for candidate in state.candidates.values():
            path = Path(candidate.candidate_path)
            if path.exists():
                paths.append(path)
        now_epoch = self.now().timestamp()
        ages = [
            max(0.0, now_epoch - path.stat().st_mtime)
            for path in paths
            if path.exists()
        ]
        active_depth = len(self._active_evaluation_status(state))
        pending_depth = len(candidates) + len(ignored)
        return {
            "depth": pending_depth + active_depth,
            "pendingDepth": pending_depth,
            "readyDepth": len(candidates),
            "ignoredDepth": len(ignored),
            "oldestAgeSeconds": max(ages) if ages else 0.0,
            "activeDepth": active_depth,
        }

    def _audit_queue_status(self) -> Mapping[str, Any]:
        root = self.runtime.promotion_root / "audits" / "queue"
        requests = sorted(root.glob("*.json")) if root.exists() else []
        reports_root = self.runtime.promotion_root / "audits" / "reports"
        pending = [
            path
            for path in requests
            if not (reports_root / path.name).is_file()
        ]
        now = self.now().timestamp()
        ages = [
            max(0.0, now - path.stat().st_mtime)
            for path in pending
        ]
        return {
            "depth": len(requests),
            "pendingDepth": len(pending),
            "oldestPendingAgeSeconds": max(ages) if ages else 0.0,
        }

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
        backpressure = self._backpressure_status(state)
        if backpressure["exportPaused"]:
            warnings.append("export-backpressure-active")
        if backpressure["evaluationPaused"]:
            warnings.append("evaluation-backpressure-active")
        if mutate:
            operations_path = (
                self.runtime.promotion_root
                / "operations"
                / "backpressure.json"
            )
            atomic_write_bytes(
                operations_path,
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "updated_at_utc": utc_timestamp(self.now()),
                        "controller_hash":
                            self.runtime.controller.controller_hash,
                        "policy_hash": self.runtime.controller.policy_hash,
                        **backpressure,
                    }
                )
                + b"\n",
            )
        queue_status = self._queue_status(state)
        active_evaluations = self._active_evaluation_status(state)
        worker_acknowledgements = self._worker_ack_status(state)
        promotion_feedback = self._promotion_feedback_status(state)
        audit_queue = self._audit_queue_status()
        trash_status = list(self.reconcile_trash(mutate=False))
        lease_status = self._lease_status()
        retained_bytes = self._retained_bytes(state)
        return {
            "mode": "automatic" if mutate else "recommend-only",
            "lastSequence": state.last_sequence,
            "championHash": state.current_champion_hash,
            "championProjectionHash": champion.champion_hash if champion else None,
            "currentGenerationId": state.current_generation_id,
            "candidates": candidate_status,
            "pins": sorted(state.pins),
            "transactions": transactions,
            "queue": queue_status,
            "queueDepth": queue_status["depth"],
            "queueAgeSeconds": queue_status["oldestAgeSeconds"],
            "backpressure": backpressure,
            "lease": lease_status,
            "leaseOwner": lease_status["owner"],
            "activeEvaluations": active_evaluations,
            "activeStage": (
                active_evaluations[0]["stage"]
                if active_evaluations
                else None
            ),
            "activeLook": (
                active_evaluations[0]["look"]
                if active_evaluations
                else None
            ),
            "retention": {
                "retainedBytes": retained_bytes,
                "retainedHashes": sorted(state.retained_hashes()),
            },
            "retainedBytes": retained_bytes,
            "workerAcknowledgements": worker_acknowledgements,
            "workerAcks": worker_acknowledgements,
            "promotionFeedback": promotion_feedback,
            "promotionFeedbackTimestamps": promotion_feedback,
            "deepAuditQueue": audit_queue,
            "trash": trash_status,
            "warnings": sorted(warnings),
        }

    def _stage_result(
        self,
        candidate_hash: str,
        stage: str,
        *,
        look: Optional[str] = None,
    ) -> Optional[Mapping[str, Any]]:
        if stage == "confirmation" and look is None:
            paths = (
                self._controller_result_path(
                    candidate_hash, stage, "look-2"
                ),
                self._controller_result_path(
                    candidate_hash, stage, "look-1"
                ),
                self.runtime.evaluations
                / "controller-results"
                / candidate_hash
                / "confirmation.json",
            )
            path = next((item for item in paths if item.exists()), paths[0])
        else:
            path = self._controller_result_path(
                candidate_hash, stage, look
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
            or (
                stage == "confirmation"
                and look is not None
                and value.get("look") != _canonical_confirmation_look(look)
            )
            or (
                value.get("policy_hash") is not None
                and value.get("policy_hash")
                != self.runtime.controller.policy_hash
            )
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
            stored_hash = value.get("ranking_hash")
            unhashed = dict(value)
            unhashed.pop("ranking_hash", None)
            if (
                value.get("generation_id") != state.current_generation_id
                or value.get("champion_hash") != state.current_champion_hash
                or value.get("policy_hash") != self.runtime.controller.policy_hash
                or not isinstance(stored_hash, str)
                or stored_hash != canonical_sha256(unhashed)
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
            summary = gate.get("ranking_summary")
            if not isinstance(summary, Mapping):
                continue
            utility = summary.get(
                "realized_powered_utility_lower_bound"
            )
            risk = summary.get(
                "final50_risk_upper_bound",
                summary.get("final_50_risk_upper_bound"),
            )
            sample_count = summary.get(
                "sample_count",
                summary.get("training_sample_count"),
            )
            statistics_artifact_hash = summary.get(
                "statistics_artifact_hash"
            )
            statistics_manifest_hash = summary.get(
                "statistics_manifest_hash"
            )
            candidate_manifest_hash = summary.get(
                "candidate_manifest_hash"
            )
            if (
                summary.get("schema_version") != 1
                or summary.get("source_bound") is not True
                or summary.get("candidate_hash") != record.candidate_hash
                or not isinstance(statistics_artifact_hash, str)
                or _SHA_RE.fullmatch(statistics_artifact_hash) is None
                or not isinstance(statistics_manifest_hash, str)
                or _SHA_RE.fullmatch(statistics_manifest_hash) is None
                or not isinstance(candidate_manifest_hash, str)
                or _SHA_RE.fullmatch(candidate_manifest_hash) is None
                or candidate_manifest_hash
                != inspect_candidate(
                    Path(record.candidate_path)
                ).directory_manifest_hash
                or isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or isinstance(risk, bool)
                or not isinstance(risk, (int, float))
                or type(sample_count) is not int
                or sample_count < 0
            ):
                continue
            rows.append(
                {
                    "candidate_hash": record.candidate_hash,
                    "realized_powered_utility_lower_bound": float(utility),
                    "final50_risk_upper_bound": float(risk),
                    "sample_count": sample_count,
                    "statistics_artifact_hash":
                        statistics_artifact_hash,
                    "statistics_manifest_hash":
                        statistics_manifest_hash,
                    "candidate_manifest_hash": candidate_manifest_hash,
                }
            )
        if not rows:
            return None

        tie_width = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "stage_2_finalist_selection",
            "utility_tie_width",
            default=0.10,
        )
        tie_width_number = _finite_number(tie_width)
        if tie_width_number is None or tie_width_number < 0.0:
            raise SafetyHalt("frozen finalist utility tie width is invalid")

        def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
            utility_delta = (
                left["realized_powered_utility_lower_bound"]
                - right["realized_powered_utility_lower_bound"]
            )
            if abs(utility_delta) > tie_width_number:
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
            if left["candidate_hash"] == right["candidate_hash"]:
                return 0
            return -1 if left["candidate_hash"] < right["candidate_hash"] else 1

        ranked = sorted(rows, key=functools.cmp_to_key(compare))
        artifact = {
            "schema_version": 1,
            "generation_id": state.current_generation_id,
            "champion_hash": state.current_champion_hash,
            "policy_hash": self.runtime.controller.policy_hash,
            "ranking_rule":
                "utility-lcb;within-0.10-final50-risk;later-sample",
            "utility_tie_width": tie_width_number,
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

        def active_confirmation_look(candidate_hash: str) -> str:
            for event in reversed(state.events):
                if (
                    event.transition
                    == Transition.EVALUATION_CONFIRMATION_STARTED
                    and event.candidate_hash == candidate_hash
                ):
                    event_look = event.payload.get("look")
                    if event_look is None:
                        matrix = event.payload.get("matrix")
                        event_look = (
                            matrix.get("look")
                            if isinstance(matrix, Mapping)
                            else None
                        )
                    return _canonical_confirmation_look(
                        event_look or "look-1"
                    )
            return "look-1"

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
            selected_look = "automatic"
            if prior_result is not None:
                if prior_stage == "confirmation":
                    result_look = _canonical_confirmation_look(
                        prior_result.get("look")
                        or active_confirmation_look(record.candidate_hash)
                    )
                    gate = prior_result.get("gate")
                    if not isinstance(gate, Mapping):
                        raise SafetyHalt(
                            "confirmation result lacks a gate report"
                        )
                    action = self._next_action(
                        prior_result["decision"], result_look, gate
                    )
                    if action == "CONTINUE_TO_LOOK_2":
                        stage = "confirmation"
                        selected_look = "look-2"
                    else:
                        selected_look = result_look
                elif prior_result["decision"] != "PASS":
                    outcomes.append(
                        {
                            "candidateHash": record.candidate_hash,
                            "stage": prior_stage,
                            "decision": prior_result["decision"],
                            "reused": True,
                        }
                    )
                    continue
                else:
                    stage = next_after_pass.get(record.state, stage)
            elif stage == "confirmation":
                selected_look = active_confirmation_look(
                    record.candidate_hash
                )
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
                if selected_look == "automatic":
                    selected_look = "look-1"
            outcome = dict(
                self.process_evaluation_stage(
                    record.candidate_hash,
                    stage=stage,
                    suite="confirmation" if stage == "confirmation" else stage,
                    look=selected_look,
                    topology="7-workers-100-threads",
                )
            )
            outcome["candidateHash"] = record.candidate_hash
            outcomes.append(outcome)
        return tuple(outcomes)

    def _near_safety_boundary(self, report: Mapping[str, Any]) -> bool:
        gate = report.get("gate")
        if not isinstance(gate, Mapping):
            return False
        summary = gate.get("ranking_summary")
        for container in (gate, summary):
            if isinstance(container, Mapping) and container.get(
                "near_safety_boundary"
            ) is True:
                return True
        fraction = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "deep_audit",
            "near_boundary_fraction",
            default=0.10,
        )
        fraction_value = _finite_number(fraction)
        if fraction_value is None or fraction_value < 0.0:
            raise SafetyHalt("deep-audit near-boundary policy is invalid")
        zero_margin = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "deep_audit",
            "near_zero_absolute_margin",
            default=0.01,
        )
        zero_margin_value = _finite_number(zero_margin)
        if zero_margin_value is None or zero_margin_value < 0.0:
            raise SafetyHalt("deep-audit near-zero margin is invalid")
        checks = gate.get("checks")
        if not isinstance(checks, list):
            return False
        for check in checks:
            if not isinstance(check, Mapping) or check.get("status") != "PASS":
                continue
            code = str(check.get("code", ""))
            if not any(
                token in code
                for token in ("RISK_", "UTILITY_", "WIN_RATE_")
            ):
                continue
            actual = _finite_number(check.get("actual"))
            expected = _finite_number(check.get("expected"))
            if actual is None or expected is None:
                continue
            tolerance = (
                zero_margin_value
                if expected == 0.0
                else max(1.0e-12, abs(expected) * fraction_value)
            )
            if abs(actual - expected) <= tolerance:
                return True
        return False

    def schedule_deep_audit(
        self,
        generation_id: str,
        candidate_hash: str,
        *,
        reasons: Sequence[str],
    ) -> Mapping[str, Any]:
        """Durably enqueue one deterministic audit without launching a thread."""

        if not self.automatic:
            return {
                "scheduled": False,
                "recommendation": True,
                "generation_id": generation_id,
                "candidate_hash": candidate_hash,
                "reasons": sorted(set(reasons)),
            }
        if self.suite_registry is not None and self._writer_lock_depth <= 0:
            with self._writer_lock():
                return self.schedule_deep_audit(
                    generation_id,
                    candidate_hash,
                    reasons=reasons,
                )
        self._transaction_dir(generation_id)
        _hash(candidate_hash, "deep-audit candidate hash")
        normalized_reasons = sorted(
            {
                _nonempty(reason, "deep-audit reason")
                for reason in reasons
            }
        )
        if not normalized_reasons:
            raise SafetyHalt("deep audit requires at least one reason")
        state = self.registry.reconstruct()
        generation = state.generations.get(generation_id)
        if (
            generation is None
            or generation.candidate_hash != candidate_hash
            or generation.state != GenerationState.ACTIVE
        ):
            raise SafetyHalt("deep audit requires the active generation")
        if self.suite_registry is not None:
            self._ensure_suite_evaluation_pin_locked(
                candidate_hash,
                generation.previous_champion_hash,
            )
        activation = next(
            (
                event
                for event in reversed(state.events)
                if event.transition == Transition.GENERATION_ACTIVATED
                and event.payload.get("generation_id") == generation_id
            ),
            None,
        )
        if activation is None:
            raise SafetyHalt("active generation has no activation event")
        policy = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "deep_audit",
        )
        if not isinstance(policy, Mapping):
            raise SafetyHalt("frozen policy has no deep-audit contract")
        manifest, _ = self._load_suite_manifest()
        banks = manifest.get("banks")
        if not isinstance(banks, list):
            raise SafetyHalt("frozen suite manifest has no audit banks")

        def bind_audit_bank(
            names: Sequence[str], expected_pairs: Optional[int]
        ) -> Mapping[str, Any]:
            bank = next(
                (
                    item
                    for item in banks
                    if isinstance(item, Mapping)
                    and (
                        item.get("qualifiedName") in names or item.get("name") in names
                    )
                ),
                None,
            )
            if not isinstance(bank, Mapping):
                raise SafetyHalt(f"deep-audit bank is missing: {names[0]}")
            schedule = bank.get("schedule")
            positions = bank.get("positions")
            if not isinstance(schedule, Mapping) or not isinstance(positions, Mapping):
                raise SafetyHalt("audit suite bank is incomplete")
            relative = schedule.get("path")
            if not isinstance(relative, str):
                raise SafetyHalt("audit suite bank has no schedule path")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise SafetyHalt("deep-audit schedule path is unsafe")
            schedule_path = self.runtime.suites / relative_path
            schedule_hash = _hash(schedule.get("sha256"), "deep-audit schedule hash")
            bank_hash = _hash(positions.get("sha256"), "deep-audit bank hash")
            schedule_id = _nonempty(
                schedule.get("scheduleId"), "deep-audit schedule ID"
            )
            pair_count = schedule.get("pairCount")
            if expected_pairs is not None and (
                type(pair_count) is not int or pair_count != expected_pairs
            ):
                raise SafetyHalt("deep-audit schedule pair count contradicts policy")
            if (
                schedule_path.is_symlink()
                or not schedule_path.is_file()
                or sha256_file(schedule_path) != schedule_hash
            ):
                raise SafetyHalt("deep-audit schedule changed")
            return {
                "qualified_name": bank.get("qualifiedName", bank.get("name")),
                "schedule_path": str(schedule_path),
                "schedule_hash": schedule_hash,
                "schedule_id": schedule_id,
                "bank_hash": bank_hash,
                "color_pairs": pair_count,
            }

        v3_audit = (
            self.runtime.frozen_policy.get("policy_version")
            == PROMOTION_READY_POLICY_VERSION
        )
        if v3_audit:
            counts = (
                ("ordinary", ("audit",), policy.get("ordinary_color_pairs")),
                (
                    "lead-40",
                    ("lead-40-audit", "lead-40"),
                    policy.get("lead_40_color_pairs"),
                ),
                (
                    "lead-80",
                    ("lead-80-audit", "lead-80"),
                    policy.get("lead_80_color_pairs"),
                ),
            )
            audit_banks = []
            for label, names, expected_pairs in counts:
                if type(expected_pairs) is not int or expected_pairs <= 0:
                    raise SafetyHalt(f"deep-audit {label} color-pair count is invalid")
                audit_banks.append(
                    {
                        "label": label,
                        **bind_audit_bank(names, expected_pairs),
                    }
                )
            visits = policy.get("visits")
            controls = policy.get("controls")
            if (
                visits != [2000, 8000]
                or not isinstance(controls, list)
                or not controls
                or any(not isinstance(control, str) for control in controls)
            ):
                raise SafetyHalt("v3 deep-audit visit/control matrix is invalid")
            b28_path = (
                self.runtime.promotion_root
                / "controls"
                / "b28"
                / "model.bin.gz"
            )
            if b28_path.is_symlink() or not b28_path.is_file():
                raise SafetyHalt(
                    "v3 deep-audit b28 control model is not frozen"
                )
            control_model_hashes = {
                "candidate": candidate_hash,
                "champion": generation.previous_champion_hash,
                "original": self.runtime.controller.original_hash,
                "b28": sha256_file(b28_path),
            }
            if set(controls) != set(control_model_hashes):
                raise SafetyHalt("v3 deep-audit controls are unsupported")
            audit_cells = []
            for bank in audit_banks:
                for visit_count in visits:
                    for control in controls:
                        payload = {
                            "label": bank["label"],
                            "visit_count": visit_count,
                            "control": control,
                            "control_model_hash":
                                control_model_hashes[control],
                            "schedule_hash": bank["schedule_hash"],
                            "bank_hash": bank["bank_hash"],
                            "color_pairs": bank["color_pairs"],
                        }
                        audit_cells.append(
                            {
                                "cell_id": "deep-audit-cell-"
                                + canonical_sha256(payload),
                                **payload,
                            }
                        )
            primary_audit = audit_banks[0]
        else:
            primary_audit = bind_audit_bank(("audit",), None)
            audit_banks = []
        request = {
            "schema_version": 2 if v3_audit else 1,
            "contract": (
                "risk-score-deep-audit-request-v2"
                if v3_audit
                else "risk-score-deep-audit-request-v1"
            ),
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "previous_champion_hash": generation.previous_champion_hash,
            "policy_path": str(self.runtime.policy_path),
            "policy_hash": self.runtime.controller.policy_hash,
            "policy_version": self.runtime.frozen_policy.get("policy_version"),
            "suite_manifest_path": str(self.runtime.suites / "manifest.json"),
            "suite_manifest_hash": self.runtime.controller.suite_manifest_hash,
            "audit_schedule_path": primary_audit["schedule_path"],
            "audit_schedule_hash": primary_audit["schedule_hash"],
            "audit_schedule_id": primary_audit["schedule_id"],
            "audit_bank_hash": primary_audit["bank_hash"],
            "activation_event_hash": activation.event_hash,
            "scheduled_at_utc": activation.timestamp_utc,
            "reasons": normalized_reasons,
            "audit_contract": dict(policy),
        }
        if v3_audit:
            request.update(
                {
                    "audit_banks": audit_banks,
                    "visit_tiers": list(visits),
                    "controls": list(controls),
                    "control_model_hashes": control_model_hashes,
                    "b28_model_path": str(b28_path.resolve()),
                    "audit_cells": audit_cells,
                }
            )
        path = (
            self.runtime.promotion_root
            / "audits"
            / "queue"
            / f"{generation_id}.json"
        )
        _write_immutable_json(path, request)
        return {
            "scheduled": True,
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "request_path": str(path),
            "request_hash": sha256_file(path),
            "reasons": normalized_reasons,
        }

    def _schedule_deep_audit_if_needed(
        self,
        generation_id: str,
        candidate_hash: str,
        report: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        state = self.registry.reconstruct()
        activation_events = [
            event
            for event in state.events
            if event.transition == Transition.GENERATION_ACTIVATED
        ]
        interval = _policy_get(
            self.runtime.frozen_policy,
            "evaluation_stages",
            "deep_audit",
            "promotion_interval",
            default=5,
        )
        if type(interval) is not int or interval <= 0:
            raise SafetyHalt("deep-audit promotion interval is invalid")
        reasons = []
        if activation_events and len(activation_events) % interval == 0:
            reasons.append(f"every-{interval}-promotions")
        if self._near_safety_boundary(report):
            reasons.append("near-safety-boundary")
        if not reasons:
            return None
        return self.schedule_deep_audit(
            generation_id,
            candidate_hash,
            reasons=reasons,
        )

    def record_deep_audit_report(
        self,
        generation_id: str,
        *,
        report_path: Path,
        report_hash: str,
    ) -> Path:
        """Validate and retain one audit report for reconcile-time action."""

        if not self.automatic:
            raise SafetyHalt("deep-audit report mutation is disabled")
        self._transaction_dir(generation_id)
        _hash(report_hash, "deep-audit report hash")
        queue_path = (
            self.runtime.promotion_root
            / "audits"
            / "queue"
            / f"{generation_id}.json"
        )
        if not queue_path.is_file():
            raise SafetyHalt("deep-audit report has no queued request")
        source = Path(report_path)
        if (
            source.is_symlink()
            or not source.is_file()
            or sha256_file(source) != report_hash
        ):
            raise SafetyHalt("deep-audit report hash mismatch")
        value = json.loads(source.read_text(encoding="utf-8"))
        request = json.loads(queue_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": (
                2
                if request.get("contract")
                == "risk-score-deep-audit-request-v2"
                else 1
            ),
            "finalized": True,
            "generation_id": generation_id,
            "candidate_hash": request["candidate_hash"],
            "policy_hash": self.runtime.controller.policy_hash,
            "audit_request_hash": sha256_file(queue_path),
        }
        if not isinstance(value, dict) or any(
            value.get(key) != expected_value
            for key, expected_value in expected.items()
        ):
            raise SafetyHalt("deep-audit report contradicts queued request")
        if request.get("contract") == "risk-score-deep-audit-request-v2":
            if value.get("contract") != "risk-score-deep-audit-report-v2":
                raise SafetyHalt("deep-audit report contract is invalid")
            supplied_identity = value.get("manifest_sha256")
            payload = dict(value)
            payload.pop("manifest_sha256", None)
            if supplied_identity != canonical_sha256(payload):
                raise SafetyHalt("deep-audit report self-hash is invalid")
            control_hashes = value.get("control_model_hashes")
            request_controls = request.get("control_model_hashes")
            b28_path_value = request.get("b28_model_path")
            b28_path = (
                Path(b28_path_value)
                if isinstance(b28_path_value, str)
                else None
            )
            if (
                not isinstance(control_hashes, Mapping)
                or not isinstance(request_controls, Mapping)
                or set(control_hashes) != set(request_controls)
                or len(set(control_hashes.values())) != len(control_hashes)
                or any(
                    (
                        control_hashes[control] != expected_hash
                        if expected_hash is not None
                        else not isinstance(control_hashes[control], str)
                        or _SHA_RE.fullmatch(control_hashes[control]) is None
                    )
                    for control, expected_hash in request_controls.items()
                )
                or b28_path is None
                or not b28_path.is_absolute()
                or str(b28_path.resolve()) != b28_path_value
                or b28_path.is_symlink()
                or not b28_path.is_file()
                or sha256_file(b28_path) != request_controls.get("b28")
            ):
                raise SafetyHalt("deep-audit control model bindings are invalid")
            cells = value.get("cells")
            request_cells = request.get("audit_cells")
            if (
                not isinstance(cells, list)
                or not isinstance(request_cells, list)
                or len(cells) != len(request_cells)
                or not all(isinstance(cell, Mapping) for cell in cells)
                or len({cell.get("cell_id") for cell in cells}) != len(cells)
            ):
                raise SafetyHalt("deep-audit report matrix is incomplete")
            cells_by_id = {
                cell.get("cell_id"): cell
                for cell in cells
                if isinstance(cell, Mapping)
            }
            cell_decisions = []
            for request_cell in request_cells:
                cell = cells_by_id.get(request_cell.get("cell_id"))
                control = request_cell.get("control")
                expected_cell = {
                    **request_cell,
                    "control_model_hash": control_hashes.get(control),
                }
                if (
                    not isinstance(cell, Mapping)
                    or any(
                        cell.get(key) != expected_value
                        for key, expected_value in expected_cell.items()
                    )
                    or cell.get("decision") not in {"PASS", "FAIL"}
                    or not isinstance(
                        cell.get("runner_manifest_path"), str
                    )
                    or not isinstance(
                        cell.get("runner_manifest_sha256"), str
                    )
                    or _SHA_RE.fullmatch(
                        cell["runner_manifest_sha256"]
                    )
                    is None
                    or not isinstance(
                        cell.get("statistics_artifact_path"), str
                    )
                    or not isinstance(
                        cell.get("statistics_artifact_sha256"), str
                    )
                    or _SHA_RE.fullmatch(
                        cell["statistics_artifact_sha256"]
                    )
                    is None
                ):
                    raise SafetyHalt(
                        "deep-audit report matrix cell is invalid"
                    )
                runner_outputs = None
                derived_cell_decision = None
                derived_candidate_win_rate = None
                minimum_candidate_win_rate = _policy_get(
                    self.runtime.frozen_policy,
                    "promotion_thresholds",
                    "powered_win_rate_vs_champion_lower_bound_strictly_above",
                    default=0.47,
                )
                if (
                    not isinstance(minimum_candidate_win_rate, (int, float))
                    or isinstance(minimum_candidate_win_rate, bool)
                    or not math.isfinite(
                        float(minimum_candidate_win_rate)
                    )
                ):
                    raise SafetyHalt(
                        "deep-audit win-rate threshold is invalid"
                    )
                minimum_candidate_win_rate = float(
                    minimum_candidate_win_rate
                )
                for artifact_name, contract in (
                    (
                        "runner_manifest",
                        "risk-score-deep-audit-runner-manifest-v1",
                    ),
                    (
                        "statistics_artifact",
                        "risk-score-deep-audit-statistics-v1",
                    ),
                ):
                    artifact_path_value = cell[
                        f"{artifact_name}_path"
                    ]
                    artifact_hash = cell[
                        f"{artifact_name}_sha256"
                    ]
                    artifact_path = Path(artifact_path_value)
                    try:
                        artifact_path.resolve().relative_to(
                            self.runtime.promotion_root.resolve()
                        )
                    except ValueError as exc:
                        raise SafetyHalt(
                            "deep-audit artifact path is outside promotion root"
                        ) from exc
                    if (
                        not artifact_path.is_absolute()
                        or str(artifact_path.resolve())
                        != artifact_path_value
                        or artifact_path.is_symlink()
                        or not artifact_path.is_file()
                        or sha256_file(artifact_path) != artifact_hash
                    ):
                        raise SafetyHalt(
                            "deep-audit matrix artifact changed"
                        )
                    artifact_data = artifact_path.read_bytes()
                    try:
                        artifact = json.loads(artifact_data)
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise SafetyHalt(
                            "deep-audit matrix artifact is invalid"
                        ) from exc
                    if (
                        not isinstance(artifact, dict)
                        or artifact_data
                        != canonical_json_bytes(artifact) + b"\n"
                    ):
                        raise SafetyHalt(
                            "deep-audit matrix artifact is not canonical"
                        )
                    artifact_payload = dict(artifact)
                    artifact_identity = artifact_payload.pop(
                        "manifest_sha256", None
                    )
                    if (
                        artifact_identity
                        != canonical_sha256(artifact_payload)
                        or artifact.get("schema_version") != 1
                        or artifact.get("contract") != contract
                        or artifact.get("finalized") is not True
                        or artifact.get("cell") != expected_cell
                        or artifact.get("decision") != cell["decision"]
                        or artifact.get("policy_hash")
                        != self.runtime.controller.policy_hash
                        or artifact.get("audit_request_hash")
                        != sha256_file(queue_path)
                    ):
                        raise SafetyHalt(
                            "deep-audit matrix artifact contradicts request"
                        )
                    if artifact_name == "runner_manifest":
                        output_hashes = {}
                        output_paths = {}
                        for output_name in ("results", "moves"):
                            output_path_value = artifact.get(
                                f"{output_name}_path"
                            )
                            output_hash = artifact.get(
                                f"{output_name}_sha256"
                            )
                            if (
                                not isinstance(output_path_value, str)
                                or not output_path_value
                                or not isinstance(output_hash, str)
                                or _SHA_RE.fullmatch(output_hash) is None
                            ):
                                raise SafetyHalt(
                                    "deep-audit runner output binding is missing"
                                )
                            output_path = Path(output_path_value)
                            try:
                                output_path.resolve().relative_to(
                                    self.runtime.promotion_root.resolve()
                                )
                            except ValueError as exc:
                                raise SafetyHalt(
                                    "deep-audit runner output is outside promotion root"
                                ) from exc
                            if (
                                not output_path.is_absolute()
                                or str(output_path.resolve())
                                != output_path_value
                                or output_path.is_symlink()
                                or not output_path.is_file()
                                or output_path.stat().st_size <= 0
                                or sha256_file(output_path) != output_hash
                            ):
                                raise SafetyHalt(
                                    "deep-audit runner output changed"
                                )
                            try:
                                output_rows = [
                                    json.loads(line)
                                    for line in output_path.read_text(
                                        encoding="utf-8"
                                    ).splitlines()
                                    if line
                                ]
                            except (
                                UnicodeDecodeError,
                                json.JSONDecodeError,
                            ) as exc:
                                raise SafetyHalt(
                                    "deep-audit runner output is invalid"
                                ) from exc
                            if not output_rows or not all(
                                isinstance(row, dict)
                                for row in output_rows
                            ):
                                raise SafetyHalt(
                                    "deep-audit runner output is empty"
                                )
                            output_hashes[output_name] = output_hash
                            output_paths[output_name] = output_path
                        bank_binding = next(
                            (
                                bank
                                for bank in request.get("audit_banks", [])
                                if isinstance(bank, Mapping)
                                and bank.get("schedule_hash")
                                == expected_cell["schedule_hash"]
                            ),
                            None,
                        )
                        if not isinstance(bank_binding, Mapping):
                            raise SafetyHalt(
                                "deep-audit cell has no schedule binding"
                            )
                        bound_schedule_path = Path(
                            bank_binding["schedule_path"]
                        )
                        if (
                            not bound_schedule_path.is_absolute()
                            or str(bound_schedule_path.resolve())
                            != bank_binding["schedule_path"]
                            or bound_schedule_path.is_symlink()
                            or not bound_schedule_path.is_file()
                            or sha256_file(bound_schedule_path)
                            != bank_binding["schedule_hash"]
                        ):
                            raise SafetyHalt(
                                "deep-audit frozen schedule changed"
                            )
                        try:
                            from risk_score.evaluation_runner import (
                                load_schedule,
                                validate_move_jsonl,
                                validate_result_jsonl,
                            )

                            schedule_rows = load_schedule(
                                bound_schedule_path
                            )
                            schedule_prefix = schedule_rows[
                                : 2 * expected_cell["color_pairs"]
                            ]
                            if len(schedule_prefix) != 2 * expected_cell[
                                "color_pairs"
                            ]:
                                raise ValueError(
                                    "deep-audit schedule prefix is incomplete"
                                )
                            result_rows = validate_result_jsonl(
                                output_paths["results"], schedule_prefix
                            )
                            validate_move_jsonl(
                                output_paths["moves"],
                                schedule_prefix,
                                result_rows,
                            )
                        except (OSError, RuntimeError, ValueError) as exc:
                            raise SafetyHalt(
                                "deep-audit game evidence is invalid"
                            ) from exc
                        candidate_scores = []
                        for result_row in result_rows:
                            winner = result_row.get("winner")
                            if result_row.get("noResult") is True or winner in {
                                None,
                                "D",
                                "Draw",
                            }:
                                candidate_scores.append(0.5)
                                continue
                            candidate_color = (
                                "B"
                                if result_row.get("blackBot") == "candidate"
                                else "W"
                            )
                            candidate_scores.append(
                                1.0 if winner == candidate_color else 0.0
                            )
                        derived_candidate_win_rate = sum(
                            candidate_scores
                        ) / len(candidate_scores)
                        derived_cell_decision = (
                            "PASS"
                            if derived_candidate_win_rate
                            > minimum_candidate_win_rate
                            else "FAIL"
                        )
                        runner_outputs = output_hashes
                    if artifact_name == "statistics_artifact":
                        safety_failures = artifact.get(
                            "safety_failures"
                        )
                        if (
                            type(safety_failures) is not int
                            or safety_failures < 0
                            or (
                                cell["decision"] == "PASS"
                                and safety_failures != 0
                            )
                            or (
                                cell["decision"] == "FAIL"
                                and safety_failures == 0
                            )
                            or artifact.get("candidate_win_rate")
                            != derived_candidate_win_rate
                            or artifact.get(
                                "minimum_candidate_win_rate"
                            )
                            != minimum_candidate_win_rate
                        ):
                            raise SafetyHalt(
                                "deep-audit statistics decision is invalid"
                            )
                        if (
                            runner_outputs is None
                            or artifact.get("results_sha256")
                            != runner_outputs["results"]
                            or artifact.get("moves_sha256")
                            != runner_outputs["moves"]
                        ):
                            raise SafetyHalt(
                                "deep-audit statistics are not runner-bound"
                            )
                if cell.get("decision") != derived_cell_decision:
                    raise SafetyHalt(
                        "deep-audit cell decision is not output-derived"
                    )
                cell_decisions.append(cell["decision"])
            expected_decision = (
                "PASS"
                if all(decision == "PASS" for decision in cell_decisions)
                else "FAIL"
            )
            if value.get("decision") != expected_decision:
                raise SafetyHalt(
                    "deep-audit report decision contradicts matrix"
                )
            if (
                expected_decision == "FAIL"
                and value.get("rollback_required") is not True
            ):
                raise SafetyHalt(
                    "deep-audit failure must require rollback"
                )
        elif value.get("decision") not in {"PASS", "FAIL"}:
            raise SafetyHalt("deep-audit report decision is invalid")
        stored = {
            **value,
            "source_report_path": str(source),
            "source_report_hash": report_hash,
        }
        destination = (
            self.runtime.promotion_root
            / "audits"
            / "reports"
            / f"{generation_id}.json"
        )
        _write_immutable_json(destination, stored)
        if self.suite_registry is not None:
            if self._writer_lock_depth > 0:
                self._reconcile_suite_pins_locked()
            else:
                with self._writer_lock():
                    self._reconcile_suite_pins_locked()
        return destination

    def _deep_audit_rollbacks(self) -> Tuple[str, ...]:
        rollbacks = []
        root = self.runtime.promotion_root / "audits" / "reports"
        if not root.exists():
            return ()
        state = self.registry.reconstruct()
        for path in sorted(root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise SafetyHalt(f"invalid deep-audit report: {path}")
            data = path.read_bytes()
            value = json.loads(data)
            if data != canonical_json_bytes(value) + b"\n":
                raise SafetyHalt(
                    f"deep-audit report is not canonical: {path}"
                )
            generation_id = value.get("generation_id")
            generation = state.generations.get(generation_id)
            queue_path = (
                self.runtime.promotion_root
                / "audits"
                / "queue"
                / f"{generation_id}.json"
            )
            valid_binding = (
                queue_path.is_file()
                and value.get("audit_request_hash")
                == sha256_file(queue_path)
                and value.get("policy_hash")
                == self.runtime.controller.policy_hash
                and isinstance(value.get("source_report_hash"), str)
                and _SHA_RE.fullmatch(value["source_report_hash"])
                is not None
            )
            if not valid_binding:
                raise SafetyHalt(
                    f"deep-audit report provenance is invalid: {path}"
                )
            if (
                value.get("finalized") is True
                and value.get("decision") == "FAIL"
                and value.get("rollback_required", True) is True
                and generation is not None
                and generation.state == GenerationState.ACTIVE
                and state.current_generation_id == generation_id
            ):
                rollbacks.append(generation_id)
        return tuple(rollbacks)

    def _ingest_rollout_ipc(self) -> bool:
        ingested = False
        for path in sorted(self.runtime.worker_ack_inbox.glob("*.json")):
            if path.name.startswith("."):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            generation = self.registry.reconstruct().generations.get(
                value.get("generation_id")
            )
            if generation is not None and generation.state in {
                GenerationState.ROLLBACK_PENDING,
                GenerationState.ROLLED_BACK,
                GenerationState.QUARANTINED,
            }:
                continue
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
            generation = self.registry.reconstruct().generations.get(
                value.get("generation_id")
            )
            if generation is not None and generation.state in {
                GenerationState.ROLLBACK_PENDING,
                GenerationState.ROLLED_BACK,
                GenerationState.QUARANTINED,
            }:
                continue
            kwargs = {"report_path": path, "report_hash": sha256_file(path)}
            phase = value.get("phase")
            decision = value.get("decision")
            if phase not in {"canary", "intermediate"}:
                raise SafetyHalt(f"unknown rollout report phase: {phase!r}")
            if decision == "FAIL":
                self.mark_rollout_failed(
                    value["generation_id"],
                    value["candidate_hash"],
                    phase,
                    **kwargs,
                )
            elif decision == "PASS" and phase == "canary":
                self.mark_canary_passed(
                    value["generation_id"], value["candidate_hash"], **kwargs
                )
            elif decision == "PASS":
                self.mark_intermediate_passed(
                    value["generation_id"], value["candidate_hash"], **kwargs
                )
            else:
                raise SafetyHalt(
                    f"unknown rollout report decision: {decision!r}"
                )
            ingested = True
        audit_outbox = self.runtime.promotion_root / "audits" / "outbox"
        if audit_outbox.exists():
            for path in sorted(audit_outbox.glob("*.json")):
                if path.name.startswith("."):
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                self.record_deep_audit_report(
                    value["generation_id"],
                    report_path=path,
                    report_hash=sha256_file(path),
                )
                ingested = True
        return ingested

    def _advance_confirmed_promotions(self) -> Tuple[Mapping[str, Any], ...]:
        if self.recommendation_only:
            return ()
        outcomes = []
        self._ingest_rollout_ipc()
        state = self.registry.reconstruct()
        readiness_errors: Optional[Tuple[str, ...]] = None
        for candidate in sorted(
            state.candidates.values(), key=lambda item: item.candidate_hash
        ):
            if candidate.state != CandidateState.CONFIRMED:
                continue
            terminal_generations = [
                generation
                for generation in state.generations.values()
                if generation.candidate_hash == candidate.candidate_hash
                and generation.state
                in {
                    GenerationState.QUARANTINED,
                    GenerationState.ROLLED_BACK,
                }
            ]
            if terminal_generations:
                outcomes.append(
                    {
                        "status": terminal_generations[0].state.value,
                        "candidate_hash": candidate.candidate_hash,
                        "generation_id": terminal_generations[0].generation_id,
                    }
                )
                continue
            if readiness_errors is None:
                readiness_errors = self._promotion_readiness_errors()
            if readiness_errors:
                outcomes.append(
                    {
                        "status": "PROMOTION_BLOCKED",
                        "candidate_hash": candidate.candidate_hash,
                        "reason": "machine-review-readiness",
                        "readiness_errors": list(readiness_errors),
                    }
                )
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
            intent_path = self._transaction_dir(generation_id) / "intent.json"
            if intent_path.exists():
                intent = json.loads(intent_path.read_text(encoding="utf-8"))
                if intent.get("candidate_hash") != candidate.candidate_hash:
                    raise SafetyHalt(
                        "existing promotion intent names a different candidate"
                    )
                kwargs = {
                    "pass_report_path": Path(intent["pass_report_path"]),
                    "pass_report_hash": intent["pass_report_hash"],
                    "trainer_checkpoint_hash":
                        intent["trainer_checkpoint_hash"],
                    "data_watermark_hash": intent["data_watermark_hash"],
                    "shuffle_watermark_hash":
                        intent["shuffle_watermark_hash"],
                }
            else:
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
            ingested = self._ingest_rollout_ipc()
            first = self.promote(
                candidate.candidate_hash, generation_id, **kwargs
            )
            ingested = self._ingest_rollout_ipc() or ingested
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
            self._reconcile_suite_data_plane_locked()
            self._require_disk()
            self._reconcile_unregistered_claims()
            self._reconcile_lifecycle_moves()
            self.reconcile_trash(mutate=True)
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
        if self.suite_registry is not None:
            with self._writer_lock():
                self._reconcile_suite_data_plane_locked()
                status["suiteGenerationBoundary"] = (
                    self._publish_clean_generation_boundary_locked()
                )
        return status

    def run_reconcile(self) -> Mapping[str, Any]:
        """Reconcile only; automatic mode repairs unregistered completed claims."""

        self.validate_static_inputs()
        if self.recommendation_only:
            return self.reconcile(mutate=False)
        if self.suite_registry is not None:
            with self._writer_lock():
                self._reconcile_suite_data_plane_locked()
        self._ingest_rollout_ipc()
        with self._writer_lock():
            self.ensure_layout()
            self._reconcile_suite_data_plane_locked()
            self._require_disk()
            self._reconcile_unregistered_claims()
            self._reconcile_lifecycle_moves()
            self.reconcile_trash(mutate=True)
            pending = []
            orphaned_adaptive = []
            rollback_pending = [
                generation.generation_id
                for generation in self.registry.reconstruct().generations.values()
                if generation.state == GenerationState.ROLLBACK_PENDING
            ]
            audit_rollbacks = list(self._deep_audit_rollbacks())
            transaction_root = self.runtime.promotion_root / "transactions"
            for path in sorted(transaction_root.glob("*/intent.json")):
                if not (path.parent / "complete.json").exists():
                    pending.append(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(
                transaction_root.glob("*/training-handoff-intent.json")
            ):
                if not (path.parent / "intent.json").exists():
                    orphaned_adaptive.append(self._marker_payload(path))
        for generation_id in sorted(set(rollback_pending + audit_rollbacks)):
            self.rollback(generation_id)
        for marker in orphaned_adaptive:
            candidate_hash = marker.get("candidate_model_sha256")
            generation_id = marker.get("generation_id")
            _hash(candidate_hash, "orphaned adaptive candidate hash")
            self._transaction_dir(generation_id)
            candidate = self.registry.reconstruct().candidates.get(
                candidate_hash
            )
            if (
                candidate is None
                or candidate.state != CandidateState.CONFIRMED
                or not candidate.evaluation_key
            ):
                raise SafetyHalt(
                    "orphaned adaptive handoff has no confirmed candidate"
                )
            report_path = self._canonical_binding_path(
                marker.get("pass_report_path"),
                "orphaned adaptive PASS report",
            )
            report_hash = _hash(
                marker.get("pass_report_sha256"),
                "orphaned adaptive PASS report hash",
            )
            if (
                not report_path.is_file()
                or sha256_file(report_path) != report_hash
            ):
                raise SafetyHalt(
                    "orphaned adaptive handoff PASS report changed"
                )
            watermark_hashes = marker.get("data_watermark_sha256s")
            if not isinstance(watermark_hashes, Mapping):
                raise SafetyHalt(
                    "orphaned adaptive handoff has no pinned watermarks"
                )
            self.promote(
                candidate_hash,
                generation_id,
                pass_report_path=report_path,
                pass_report_hash=report_hash,
                trainer_checkpoint_hash=marker[
                    "parent_champion_checkpoint_sha256"
                ],
                data_watermark_hash=watermark_hashes["data"],
                shuffle_watermark_hash=watermark_hashes["shuffle"],
            )
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
        if self.suite_registry is not None:
            with self._writer_lock():
                self._reconcile_suite_data_plane_locked()
                result = dict(self.reconcile(mutate=True))
                result["suiteGenerationBoundary"] = (
                    self._publish_clean_generation_boundary_locked()
                )
                return result
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
            if self.suite_registry is not None:
                suite_state = self._validate_suite_runtime_binding()
                assert suite_state is not None
                if (
                    suite_state.current_champion is None
                    or suite_state.current_champion.sha256
                    != champion_hash
                    or suite_state.current_champion.generation_id
                    != generation_id
                ):
                    raise SafetyHalt(
                        "promotion bootstrap champion is stale in suite registry"
                    )
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

    def _snapshot_checkpoint(
        self,
        generation_id: str,
        expected_hash: str,
        *,
        allow_installed_replay: bool = False,
    ) -> Path:
        source = self.runtime.trainer_checkpoint
        root = self.runtime.rollback_quarantine / generation_id
        destination = root / "trainer-checkpoint"
        if allow_installed_replay and destination.exists():
            if sha256_file(destination) != expected_hash:
                raise SafetyHalt("rollback checkpoint snapshot contradicts intent")
            return destination
        if sha256_file(source) != expected_hash:
            raise SafetyHalt("trainer checkpoint changed before promotion")
        root.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _marker_payload(path: Path) -> Dict[str, Any]:
        try:
            data = path.read_bytes()
            value = json.loads(data)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(f"invalid durable marker {path}: {exc}") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or data != canonical_json_bytes(value) + b"\n"
        ):
            raise SafetyHalt(f"durable marker is not canonical: {path}")
        payload = dict(value)
        payload.pop("schema_version")
        timestamp = payload.pop("timestamp_utc", None)
        if not isinstance(timestamp, str) or not timestamp:
            raise SafetyHalt(f"durable marker has no timestamp: {path}")
        return payload

    @staticmethod
    def _canonical_binding_path(value: Any, role: str) -> Path:
        if not isinstance(value, str) or not value:
            raise SafetyHalt(f"{role} path is missing")
        path = Path(value)
        if not path.is_absolute() or str(path.absolute()) != value:
            raise SafetyHalt(f"{role} path is not canonical and absolute")
        return path

    @staticmethod
    def _stable_regular_file_hash(
        path: Path,
        role: str,
        *,
        expected_hash: Optional[str] = None,
    ) -> str:
        try:
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise SafetyHalt(f"{role} is not a regular non-symlink file")
            actual = sha256_file(path)
            after = path.lstat()
        except FileNotFoundError as exc:
            raise SafetyHalt(f"{role} is missing: {path}") from exc
        except OSError as exc:
            raise SafetyHalt(f"cannot inspect {role} {path}: {exc}") from exc
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise SafetyHalt(f"{role} changed while it was hashed")
        if expected_hash is not None and actual != expected_hash:
            raise SafetyHalt(f"{role} hash contradicts its immutable binding")
        return actual

    def _adaptive_handoff_index(self, candidate_hash: str) -> Path:
        return (
            self.runtime.promotion_root
            / "adaptive"
            / "handoffs"
            / "by-candidate"
            / f"{candidate_hash}.json"
        )

    @staticmethod
    def _recipe_binding_matches(
        binding: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> bool:
        return all(binding.get(key) == value for key, value in expected.items())

    def _load_adaptive_promotion_context(
        self,
        *,
        transaction: Path,
        existing_intent: Optional[Mapping[str, Any]],
        artifact: CandidateArtifact,
        candidate_hash: str,
        generation_id: str,
        tested_champion_hash: str,
        pass_report_path: Path,
        pass_report_hash: str,
        trainer_checkpoint_hash: str,
        data_watermark_hash: str,
        shuffle_watermark_hash: str,
    ) -> Optional[Mapping[str, Any]]:
        """Validate and pin an optional adaptive handoff before promotion intent.

        The adaptive index is consulted only for a transaction that has not yet
        written its normal promotion intent. Once ``intent.json`` exists, that
        immutable intent decides whether the transaction is adaptive, so a
        handoff cannot be attached to an in-flight normal promotion.
        """

        marker_path = transaction / "training-handoff-intent.json"
        pinned: Optional[Dict[str, Any]] = None
        if existing_intent is not None:
            raw = existing_intent.get("adaptive_training")
            if raw is None:
                if marker_path.exists():
                    raise SafetyHalt(
                        "normal promotion intent conflicts with adaptive marker"
                    )
                return None
            if not isinstance(raw, Mapping):
                raise SafetyHalt("adaptive promotion intent is malformed")
            pinned = dict(raw)
        elif marker_path.exists():
            pinned = self._marker_payload(marker_path)

        expected_handoff_path = self._adaptive_handoff_index(candidate_hash)
        if pinned is None:
            if expected_handoff_path.is_symlink():
                raise SafetyHalt("adaptive handoff index must not be a symlink")
            if not expected_handoff_path.exists():
                return None
            handoff_path = expected_handoff_path
        else:
            handoff_path = self._canonical_binding_path(
                pinned.get("handoff_path"),
                "adaptive handoff",
            )
            if handoff_path != expected_handoff_path:
                raise SafetyHalt(
                    "adaptive handoff is not at the candidate-indexed path"
                )

        try:
            handoff = load_candidate_handoff(
                handoff_path,
                expected_candidate_sha256=candidate_hash,
            )
        except (AdaptiveTrainingError, FileNotFoundError, OSError) as exc:
            raise SafetyHalt(f"adaptive candidate handoff is invalid: {exc}") from exc
        handoff_file_hash = self._stable_regular_file_hash(
            handoff_path,
            "adaptive handoff",
        )

        candidate_binding = handoff.get("candidate")
        checkpoint_binding = handoff.get("candidate_checkpoint")
        parent_checkpoint = handoff.get("parent_champion_checkpoint")
        parent_data = handoff.get("parent_admitted_data")
        if any(
            not isinstance(item, Mapping)
            for item in (
                candidate_binding,
                checkpoint_binding,
                parent_checkpoint,
                parent_data,
            )
        ):
            raise SafetyHalt("adaptive handoff bindings are incomplete")
        assert isinstance(candidate_binding, Mapping)
        assert isinstance(checkpoint_binding, Mapping)
        assert isinstance(parent_checkpoint, Mapping)
        assert isinstance(parent_data, Mapping)

        candidate_path = self._canonical_binding_path(
            candidate_binding.get("path"),
            "adaptive candidate model",
        )
        if candidate_path.exists():
            self._stable_regular_file_hash(
                candidate_path,
                "adaptive candidate model",
                expected_hash=candidate_hash,
            )
        elif (
            existing_intent is None
            or candidate_path
            != Path(existing_intent.get("source_path", ""))
            / "model.bin.gz"
        ):
            raise SafetyHalt("adaptive candidate model path binding is missing")
        else:
            # The normal promotion transaction durably renames the candidate
            # directory. After that rename, the pinned pre-intent path can be
            # absent, but the accepted artifact still proves the same bytes.
            self._stable_regular_file_hash(
                artifact.path / "model.bin.gz",
                "accepted adaptive candidate model",
                expected_hash=candidate_hash,
            )

        resume_checkpoint_path = self._canonical_binding_path(
            checkpoint_binding.get("path"),
            "adaptive resumable checkpoint",
        )
        parent_checkpoint_path = self._canonical_binding_path(
            parent_checkpoint.get("path"),
            "adaptive parent champion checkpoint",
        )
        parent_data_path = self._canonical_binding_path(
            parent_data.get("path"),
            "adaptive parent admitted manifest",
        )
        recipe_path = self._canonical_binding_path(
            handoff.get("recipe_path"),
            "adaptive recipe",
        )
        live_checkpoint = self.runtime.trainer_checkpoint.absolute()
        if resume_checkpoint_path == live_checkpoint:
            raise SafetyHalt(
                "adaptive resumable checkpoint must be immutable and distinct "
                "from the live trainer checkpoint"
            )
        if parent_checkpoint_path == live_checkpoint:
            raise SafetyHalt(
                "adaptive parent checkpoint must be an immutable snapshot; "
                "the live trainer checkpoint cannot survive replay validation"
            )

        resume_checkpoint_hash = _hash(
            checkpoint_binding.get("sha256"),
            "adaptive resumable checkpoint hash",
        )
        parent_checkpoint_hash = _hash(
            parent_checkpoint.get("sha256"),
            "adaptive parent champion checkpoint hash",
        )
        parent_data_hash = _hash(
            parent_data.get("sha256"),
            "adaptive parent admitted manifest hash",
        )
        recipe_hash = _hash(
            handoff.get("recipe_sha256"),
            "adaptive recipe hash",
        )
        parent_model_hash = _hash(
            handoff.get("parent_champion_model_sha256"),
            "adaptive parent champion model hash",
        )
        if parent_model_hash != tested_champion_hash:
            raise SafetyHalt(
                "adaptive handoff parent champion does not match the tested "
                "current champion"
            )
        if handoff.get("policy_hash") != ADAPTIVE_POLICY_HASH:
            raise SafetyHalt("adaptive handoff autonomy policy is not frozen")

        previous_snapshot = transaction / "previous-champion.json"
        try:
            previous_champion = load_champion(
                previous_snapshot
                if previous_snapshot.exists()
                else self.runtime.champion_path
            )
        except (OSError, RegistryCorruptionError) as exc:
            raise SafetyHalt(
                f"cannot validate adaptive parent champion record: {exc}"
            ) from exc
        if previous_champion.champion_hash != tested_champion_hash:
            raise SafetyHalt(
                "adaptive handoff parent champion is not the champion record "
                "tested by confirmation"
            )

        active_recipe_path = (
            self.runtime.promotion_root / "adaptive" / "active-recipe.json"
        )
        try:
            active_recipe = load_recipe_binding(active_recipe_path)
        except (AdaptiveTrainingError, FileNotFoundError, OSError) as exc:
            raise SafetyHalt(
                f"adaptive active recipe binding is missing or invalid: {exc}"
            ) from exc

        watermark_hashes = {
            "data": data_watermark_hash,
            "shuffle": shuffle_watermark_hash,
        }
        expected_record_hash = (
            _hash(
                pinned.get("active_recipe_record_sha256"),
                "pinned active recipe record hash",
            )
            if pinned is not None
            else active_recipe["record_sha256"]
        )
        parent_recipe_identity = {
            "champion_model_sha256": parent_model_hash,
            "champion_checkpoint_sha256": parent_checkpoint_hash,
            "admitted_data_manifest_sha256": parent_data_hash,
            "data_watermark_sha256s": watermark_hashes,
            "generation_id": previous_champion.generation_id,
        }
        target_recipe_identity = {
            "recipe_sha256": recipe_hash,
            "recipe_path": str(recipe_path),
            "champion_model_sha256": candidate_hash,
            "champion_checkpoint_sha256": resume_checkpoint_hash,
            "admitted_data_manifest_sha256": parent_data_hash,
            "data_watermark_sha256s": watermark_hashes,
            "generation_id": generation_id,
        }
        current_is_parent = (
            active_recipe.get("record_sha256") == expected_record_hash
            and self._recipe_binding_matches(
                active_recipe,
                parent_recipe_identity,
            )
        )
        rollback_value = active_recipe.get("rollback")
        current_is_target = (
            self._recipe_binding_matches(
                active_recipe,
                target_recipe_identity,
            )
            and active_recipe.get("previous_record_sha256")
            == expected_record_hash
            and isinstance(rollback_value, Mapping)
            and rollback_value.get("source_record_sha256")
            == expected_record_hash
        )
        if not current_is_parent and not current_is_target:
            raise SafetyHalt(
                "adaptive active recipe does not match the pinned parent or "
                "replayed candidate binding"
            )
        if pinned is None and not current_is_parent:
            raise SafetyHalt(
                "adaptive recipe was already activated without promotion intent"
            )

        metadata = {
            "generation_id": generation_id,
            "pass_report_path": str(Path(pass_report_path).absolute()),
            "pass_report_sha256": pass_report_hash,
            "handoff_id": handoff["handoff_id"],
            "handoff_path": str(handoff_path),
            "handoff_file_sha256": handoff_file_hash,
            "handoff_manifest_sha256": handoff["manifest_sha256"],
            "candidate_model_path": str(candidate_path),
            "candidate_model_sha256": candidate_hash,
            "resume_checkpoint_path": str(resume_checkpoint_path),
            "resume_checkpoint_sha256": resume_checkpoint_hash,
            "recipe_path": str(recipe_path),
            "recipe_sha256": recipe_hash,
            "parent_champion_model_sha256": parent_model_hash,
            "parent_champion_checkpoint_path": str(parent_checkpoint_path),
            "parent_champion_checkpoint_sha256": parent_checkpoint_hash,
            "parent_admitted_manifest_path": str(parent_data_path),
            "parent_admitted_manifest_sha256": parent_data_hash,
            "autonomy_policy_sha256": ADAPTIVE_POLICY_HASH,
            "active_recipe_path": str(active_recipe_path),
            "active_recipe_record_sha256": expected_record_hash,
            "previous_champion_generation_id": previous_champion.generation_id,
            "previous_champion_record_sha256": previous_champion.record_hash,
            "data_watermark_sha256s": watermark_hashes,
        }
        if pinned is not None and pinned != metadata:
            raise SafetyHalt(
                "adaptive handoff no longer matches its durable promotion intent"
            )
        return {
            "handoff": handoff,
            "metadata": metadata,
            "active_recipe": active_recipe,
        }

    def _training_gpu_lease_proof(self) -> Mapping[str, Any]:
        """Require a canonical trainer-only GPU lease before checkpoint mutation.

        ``rollback/consumers-stop`` proves process quiescence, but it does not
        own the GPU lease state machine. The controller therefore fails closed
        unless the independently persisted lease is healthy and in its
        trainer-only resting phase. In particular, an evaluator/leased phase
        is never treated as safe merely because the stop command returned.
        """

        config_path = self.runtime.gpu_lease_config_path
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(
                f"cannot inspect GPU lease runtime for training handoff: {exc}"
            ) from exc
        paths = config.get("paths") if isinstance(config, Mapping) else None
        if not isinstance(paths, Mapping):
            raise SafetyHalt(
                "GPU lease runtime cannot prove a lease-state path for "
                "checkpoint replacement"
            )
        lease_path = self._canonical_binding_path(
            paths.get("leaseState"),
            "GPU lease state",
        )
        try:
            before = lease_path.lstat()
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
            ):
                raise SafetyHalt(
                    "GPU lease state is not a regular non-symlink file"
                )
            data = lease_path.read_bytes()
            lease = json.loads(data)
            after = lease_path.lstat()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetyHalt(
                f"cannot inspect GPU lease state for training handoff: {exc}"
            ) from exc
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SafetyHalt("GPU lease state changed while it was inspected")
        if (
            not isinstance(lease, dict)
            or data != canonical_json_bytes(lease) + b"\n"
        ):
            raise SafetyHalt("GPU lease state is not canonical")
        trainer_identity = lease.get("trainer")
        if (
            lease.get("schemaVersion") != 2
            or not isinstance(lease.get("leaseId"), str)
            or not lease["leaseId"]
            or not isinstance(lease.get("ownerId"), str)
            or not lease["ownerId"]
            or lease.get("expectedGpuUuid")
            != self.runtime.controller.expected_gpu_uuid
            or lease.get("phase") != "trainer_running"
            or lease.get("safetyHalt") is not False
            or lease.get("safetyReason") not in {None, ""}
            or lease.get("restorationStatus") == "safety_halt"
            or lease.get("evaluators") != []
            or not isinstance(trainer_identity, Mapping)
            or type(trainer_identity.get("pid")) is not int
            or trainer_identity["pid"] <= 0
            or type(trainer_identity.get("startTimeTicks")) is not int
            or trainer_identity["startTimeTicks"] < 0
        ):
            raise SafetyHalt(
                "GPU lease does not prove healthy trainer-only ownership"
            )
        return {
            "lease_state_path": str(lease_path),
            "lease_state_sha256": sha256_bytes(data),
            "lease_id": lease["leaseId"],
            "owner_id": lease["ownerId"],
            "phase": lease["phase"],
            "safety_halt": False,
            "non_trainer_owners": [],
        }

    def _atomic_install_checkpoint(
        self,
        source: Path,
        *,
        expected_hash: str,
        expected_current_hash: Optional[str] = None,
    ) -> None:
        destination = self.runtime.trainer_checkpoint
        self._stable_regular_file_hash(
            source,
            "adaptive resumable checkpoint",
            expected_hash=expected_hash,
        )
        if destination.exists():
            current_hash = sha256_file(destination)
            if current_hash == expected_hash:
                return
            if (
                expected_current_hash is not None
                and current_hash != expected_current_hash
            ):
                raise SafetyHalt(
                    "live trainer checkpoint contradicts adaptive handoff replay"
                )
        elif expected_current_hash is not None:
            raise SafetyHalt("live trainer checkpoint disappeared before handoff")

        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.adaptive.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(temporary) != expected_hash:
                raise SafetyHalt("installed adaptive checkpoint hash mismatch")
            self._stable_regular_file_hash(
                source,
                "adaptive resumable checkpoint",
                expected_hash=expected_hash,
            )
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
            if sha256_file(destination) != expected_hash:
                raise SafetyHalt("adaptive checkpoint changed during installation")
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _restore_checkpoint_snapshot(
        self,
        generation_id: str,
        expected_hash: str,
    ) -> None:
        snapshot = (
            self.runtime.rollback_quarantine
            / generation_id
            / "trainer-checkpoint"
        )
        self._stable_regular_file_hash(
            snapshot,
            "rollback checkpoint snapshot",
            expected_hash=expected_hash,
        )
        self._atomic_install_checkpoint(
            snapshot,
            expected_hash=expected_hash,
        )

    def _converge_adaptive_training_handoff(
        self,
        *,
        transaction: Path,
        generation_id: str,
        candidate_hash: str,
        intent: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        metadata = context["metadata"]
        if not isinstance(metadata, Mapping):
            raise SafetyHalt("adaptive training context is malformed")
        commit_path = transaction / "training-commit.json"
        recipe_marker_path = transaction / "training-recipe-cas.json"
        checkpoint_marker_path = transaction / "training-checkpoint-installed.json"
        if commit_path.exists():
            commit = self._marker_payload(commit_path)
            if (
                commit.get("candidate_hash") != candidate_hash
                or commit.get("handoff_file_sha256")
                != metadata["handoff_file_sha256"]
                or commit.get("resume_checkpoint_sha256")
                != metadata["resume_checkpoint_sha256"]
                or not recipe_marker_path.exists()
                or not checkpoint_marker_path.exists()
            ):
                raise SafetyHalt("adaptive training commit marker is inconsistent")
            checkpoint_marker = self._marker_payload(checkpoint_marker_path)
            if (
                checkpoint_marker.get("checkpoint_sha256")
                != metadata["resume_checkpoint_sha256"]
                or checkpoint_marker.get("previous_checkpoint_sha256")
                != intent["trainer_checkpoint_hash"]
            ):
                raise SafetyHalt(
                    "adaptive training commit contradicts checkpoint journal"
                )
            recipe_marker = self._marker_payload(recipe_marker_path)
            active_recipe = load_recipe_binding(
                Path(metadata["active_recipe_path"])
            )
            if (
                active_recipe["record_sha256"]
                != recipe_marker.get("record_sha256")
                or commit.get("recipe_record_sha256")
                != active_recipe["record_sha256"]
            ):
                raise SafetyHalt(
                    "adaptive training commit contradicts active recipe"
                )
            return active_recipe

        stop_result = self.execute_argv(
            "rollback",
            {
                "generation_id": metadata["previous_champion_generation_id"],
                "model_hash": metadata["parent_champion_model_sha256"],
            },
        )
        if (
            not isinstance(stop_result, Mapping)
            or stop_result.get("quiescent") is not True
            or stop_result.get("quiescent_roles") != _ALL_ROLE_QUIESCENCE
        ):
            raise SafetyHalt(
                "adaptive handoff stop command lacks exact all-role "
                "quiescence proof"
            )
        lease_proof = self._training_gpu_lease_proof()
        quiescence_path = transaction / "training-quiesced.json"
        quiescence_payload = {
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "stopped_generation_id":
                metadata["previous_champion_generation_id"],
            "stopped_model_sha256":
                metadata["parent_champion_model_sha256"],
            "quiescent": True,
            "quiescent_roles": list(_ALL_ROLE_QUIESCENCE),
            "gpu_lease": dict(lease_proof),
        }
        if quiescence_path.exists():
            prior = self._marker_payload(quiescence_path)
            if (
                prior.get("generation_id") != generation_id
                or prior.get("candidate_hash") != candidate_hash
                or prior.get("stopped_generation_id")
                != metadata["previous_champion_generation_id"]
                or prior.get("stopped_model_sha256")
                != metadata["parent_champion_model_sha256"]
                or prior.get("quiescent") is not True
                or prior.get("quiescent_roles") != _ALL_ROLE_QUIESCENCE
                or not isinstance(prior.get("gpu_lease"), Mapping)
            ):
                raise SafetyHalt(
                    "adaptive training quiescence marker is inconsistent"
                )
        else:
            self._mark(
                transaction,
                "training-quiesced",
                quiescence_payload,
            )
        self._checkpoint("promotion-training-quiesced")

        self._snapshot_checkpoint(
            generation_id,
            intent["trainer_checkpoint_hash"],
            allow_installed_replay=True,
        )
        resume_path = Path(metadata["resume_checkpoint_path"])
        self._atomic_install_checkpoint(
            resume_path,
            expected_hash=metadata["resume_checkpoint_sha256"],
            expected_current_hash=intent["trainer_checkpoint_hash"],
        )
        checkpoint_payload = {
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "source_path": str(resume_path),
            "destination_path": str(self.runtime.trainer_checkpoint),
            "previous_checkpoint_sha256": intent["trainer_checkpoint_hash"],
            "checkpoint_sha256": metadata["resume_checkpoint_sha256"],
        }
        if checkpoint_marker_path.exists():
            if self._marker_payload(checkpoint_marker_path) != checkpoint_payload:
                raise SafetyHalt(
                    "adaptive checkpoint installation marker is inconsistent"
                )
        else:
            self._mark(
                transaction,
                "training-checkpoint-installed",
                checkpoint_payload,
            )
        self._checkpoint("promotion-training-checkpoint-installed")

        active_recipe_path = Path(metadata["active_recipe_path"])
        if recipe_marker_path.exists():
            recipe_marker = self._marker_payload(recipe_marker_path)
            active_recipe = load_recipe_binding(active_recipe_path)
            if (
                recipe_marker.get("generation_id") != generation_id
                or recipe_marker.get("candidate_hash") != candidate_hash
                or recipe_marker.get("previous_record_sha256")
                != metadata["active_recipe_record_sha256"]
                or active_recipe["record_sha256"]
                != recipe_marker.get("record_sha256")
                or not isinstance(recipe_marker.get("rollback"), Mapping)
            ):
                raise SafetyHalt("adaptive recipe CAS marker is inconsistent")
        else:
            try:
                active_recipe = compare_and_swap_recipe_binding(
                    active_recipe_path,
                    expected_record_sha256=
                        metadata["active_recipe_record_sha256"],
                    recipe_sha256=metadata["recipe_sha256"],
                    recipe_path=metadata["recipe_path"],
                    champion_model_sha256=candidate_hash,
                    champion_checkpoint_sha256=
                        metadata["resume_checkpoint_sha256"],
                    admitted_data_manifest_sha256=
                        metadata["parent_admitted_manifest_sha256"],
                    data_watermark_sha256s=
                        metadata["data_watermark_sha256s"],
                    generation_id=generation_id,
                )
            except (AdaptiveTrainingError, FileNotFoundError, OSError) as exc:
                raise SafetyHalt(
                    f"adaptive active recipe CAS failed: {exc}"
                ) from exc
            rollback_value = active_recipe.get("rollback")
            if (
                active_recipe.get("previous_record_sha256")
                != metadata["active_recipe_record_sha256"]
                or not isinstance(rollback_value, Mapping)
                or rollback_value.get("source_record_sha256")
                != metadata["active_recipe_record_sha256"]
            ):
                raise SafetyHalt(
                    "adaptive active recipe CAS returned invalid lineage"
                )
            recipe_marker = {
                "generation_id": generation_id,
                "candidate_hash": candidate_hash,
                "recipe_sha256": metadata["recipe_sha256"],
                "previous_record_sha256":
                    metadata["active_recipe_record_sha256"],
                "record_sha256": active_recipe["record_sha256"],
                "rollback": dict(rollback_value),
            }
            self._mark(
                transaction,
                "training-recipe-cas",
                recipe_marker,
            )
        self._checkpoint("promotion-training-recipe-cas")
        return active_recipe

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
                and any(
                    (
                        self._transaction_dir(generation_id) / name
                    ).exists()
                    for name in (
                        "generation-data-admission-intent.json",
                        "generation-data-admitted.json",
                    )
                )
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
                and any(
                    (
                        self._transaction_dir(generation_id) / name
                    ).exists()
                    for name in (
                        "generation-data-admission-intent.json",
                        "generation-data-admitted.json",
                    )
                )
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
        expected_decision: str = "PASS",
    ) -> Mapping[str, Any]:
        if expected_decision not in {"PASS", "FAIL"}:
            raise ValueError("rollout report decision must be PASS or FAIL")
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
            "decision": expected_decision,
            "phase": phase,
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "policy_hash": self.runtime.controller.policy_hash,
            "promotion_config_hash": promotion_intent["config_hash"],
            "selfplay_config_hash": self.runtime.controller.selfplay_config_hash,
            "audit_schedule_hash": self.runtime.controller.audit_schedule_hash,
            "topology": "7-workers-100-threads",
            "worker_count": required_workers,
        }
        check_names = [
            "model_purity_pass",
            "output_schema_pass",
            "throughput_pass",
            "crash_error_pass",
            "behavior_pass",
            "catastrophe_pass",
        ]
        if (
            self.runtime.frozen_policy.get("policy_version")
            != PROMOTION_READY_POLICY_VERSION
        ):
            check_names.extend(["tactical_pass", "exploitability_pass"])
        if not isinstance(report, dict) or any(
            report.get(key) != value for key, value in expected.items()
        ):
            raise SafetyHalt(f"{phase} report does not satisfy frozen policy")
        checks = {name: report.get(name) for name in check_names}
        if any(type(value) is not bool for value in checks.values()):
            raise SafetyHalt(f"{phase} report health checks must be booleans")
        derived_decision = "PASS" if all(checks.values()) else "FAIL"
        if derived_decision != expected_decision:
            raise SafetyHalt(f"{phase} report decision contradicts health checks")
        minimum_games = report.get("minimum_game_count")
        game_count = report.get("game_count")
        if (
            type(minimum_games) is not int
            or minimum_games <= 0
            or type(game_count) is not int
            or game_count < 0
            or report["throughput_pass"] != (game_count >= minimum_games)
        ):
            raise SafetyHalt(f"{phase} report throughput decision is invalid")
        if phase == "canary":
            rollout = self.runtime.frozen_policy.get("rollout")
            if not isinstance(rollout, Mapping):
                raise SafetyHalt("frozen policy has no rollout contract")
            required_games = rollout.get("canary_games")
            required_pairs = rollout.get("canary_fresh_audit_color_pairs")
            if (
                type(required_games) is not int
                or required_games <= 0
                or type(required_pairs) is not int
                or required_pairs <= 0
                or minimum_games != required_games
                or type(report.get("fresh_audit_pairs")) is not int
                or report["fresh_audit_pairs"] < required_pairs
            ):
                raise SafetyHalt("canary report lacks required games/fresh audit")
            if (
                self.runtime.frozen_policy.get("policy_version")
                == PROMOTION_READY_POLICY_VERSION
            ):
                pair_ids = report.get("fresh_audit_pair_ids")
                pair_ids_hash = report.get("fresh_audit_pair_ids_sha256")
                if (
                    not isinstance(pair_ids, list)
                    or len(pair_ids) != required_pairs
                    or len(set(pair_ids)) != len(pair_ids)
                    or any(
                        not isinstance(pair_id, str)
                        or not pair_id
                        or "\n" in pair_id
                        or "\r" in pair_id
                        for pair_id in pair_ids
                    )
                    or pair_ids != sorted(pair_ids)
                    or pair_ids_hash != canonical_sha256(pair_ids)
                ):
                    raise SafetyHalt("canary fresh-audit pair identities are invalid")
                schedule_pair_counts: Dict[str, int] = {}
                if (
                    self.runtime.audit_schedule_path.is_symlink()
                    or not self.runtime.audit_schedule_path.is_file()
                    or sha256_file(self.runtime.audit_schedule_path)
                    != self.runtime.controller.audit_schedule_hash
                ):
                    raise SafetyHalt(
                        "canary frozen audit schedule changed"
                    )
                try:
                    for raw_line in self.runtime.audit_schedule_path.read_text(
                        encoding="utf-8"
                    ).splitlines():
                        if not raw_line:
                            continue
                        schedule_row = json.loads(raw_line)
                        if not isinstance(schedule_row, dict):
                            raise ValueError("invalid audit schedule row")
                        schedule_pair_id = schedule_row.get("pairId")
                        if not isinstance(schedule_pair_id, str) or not schedule_pair_id:
                            raise ValueError("invalid audit schedule pair ID")
                        schedule_pair_counts[schedule_pair_id] = (
                            schedule_pair_counts.get(schedule_pair_id, 0) + 1
                        )
                except (
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    raise SafetyHalt(
                        "canary audit schedule is invalid"
                    ) from exc
                scheduled_pair_ids = sorted(schedule_pair_counts)
                if (
                    any(count != 2 for count in schedule_pair_counts.values())
                    or scheduled_pair_ids != pair_ids
                ):
                    raise SafetyHalt(
                        "canary pair identities do not match frozen schedule"
                    )
                audit_manifest_path_value = report.get(
                    "fresh_audit_manifest_path"
                )
                audit_manifest_hash = report.get(
                    "fresh_audit_manifest_sha256"
                )
                if (
                    not isinstance(audit_manifest_path_value, str)
                    or not audit_manifest_path_value
                    or not isinstance(audit_manifest_hash, str)
                    or _SHA_RE.fullmatch(audit_manifest_hash) is None
                ):
                    raise SafetyHalt(
                        "canary fresh-audit manifest binding is missing"
                    )
                audit_manifest_path = Path(audit_manifest_path_value)
                try:
                    audit_manifest_path.resolve().relative_to(
                        self.runtime.promotion_root.resolve()
                    )
                except ValueError as exc:
                    raise SafetyHalt(
                        "canary fresh-audit manifest path is outside promotion root"
                    ) from exc
                if (
                    not audit_manifest_path.is_absolute()
                    or str(audit_manifest_path.resolve())
                    != audit_manifest_path_value
                    or audit_manifest_path.is_symlink()
                    or not audit_manifest_path.is_file()
                    or sha256_file(audit_manifest_path)
                    != audit_manifest_hash
                ):
                    raise SafetyHalt("canary fresh-audit manifest changed")
                audit_data = audit_manifest_path.read_bytes()
                try:
                    audit_manifest = json.loads(audit_data)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SafetyHalt(
                        "canary fresh-audit manifest is invalid"
                    ) from exc
                if (
                    not isinstance(audit_manifest, dict)
                    or audit_data
                    != canonical_json_bytes(audit_manifest) + b"\n"
                ):
                    raise SafetyHalt(
                        "canary fresh-audit manifest is not canonical"
                    )
                audit_payload = dict(audit_manifest)
                audit_identity = audit_payload.pop("manifest_sha256", None)
                audit_decision = (
                    "PASS"
                    if report["behavior_pass"] and report["catastrophe_pass"]
                    else "FAIL"
                )
                audit_safety_failures = audit_manifest.get("safety_failures")
                expected_audit = {
                    "schema_version": 1,
                    "contract": "risk-score-canary-fresh-audit-v1",
                    "finalized": True,
                    "decision": audit_decision,
                    "generation_id": generation_id,
                    "candidate_hash": candidate_hash,
                    "reference_hash": promotion_intent[
                        "tested_champion_hash"
                    ],
                    "policy_hash": self.runtime.controller.policy_hash,
                    "suite_manifest_hash":
                        self.runtime.controller.suite_manifest_hash,
                    "audit_schedule_hash":
                        self.runtime.controller.audit_schedule_hash,
                    "color_pairs": required_pairs,
                    "pair_ids_sha256": pair_ids_hash,
                }
                statistics_hash = audit_manifest.get(
                    "statistics_artifact_sha256"
                )
                statistics_path_value = audit_manifest.get(
                    "statistics_artifact_path"
                )
                if (
                    audit_identity != canonical_sha256(audit_payload)
                    or any(
                        audit_manifest.get(key) != value
                        for key, value in expected_audit.items()
                    )
                    or type(audit_safety_failures) is not int
                    or audit_safety_failures < 0
                    or (audit_decision == "PASS" and audit_safety_failures != 0)
                    or (audit_decision == "FAIL" and audit_safety_failures == 0)
                    or not isinstance(statistics_hash, str)
                    or _SHA_RE.fullmatch(statistics_hash) is None
                    or not isinstance(statistics_path_value, str)
                    or not statistics_path_value
                ):
                    raise SafetyHalt(
                        "canary fresh-audit evidence does not satisfy policy"
                    )
                statistics_path = Path(statistics_path_value)
                try:
                    statistics_path.resolve().relative_to(
                        self.runtime.promotion_root.resolve()
                    )
                except ValueError as exc:
                    raise SafetyHalt(
                        "canary statistics path is outside promotion root"
                    ) from exc
                if (
                    not statistics_path.is_absolute()
                    or str(statistics_path.resolve())
                    != statistics_path_value
                    or statistics_path.is_symlink()
                    or not statistics_path.is_file()
                    or sha256_file(statistics_path) != statistics_hash
                ):
                    raise SafetyHalt("canary statistics artifact changed")
                statistics_data = statistics_path.read_bytes()
                try:
                    statistics = json.loads(statistics_data)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SafetyHalt(
                        "canary statistics artifact is invalid"
                    ) from exc
                if (
                    not isinstance(statistics, dict)
                    or statistics_data
                    != canonical_json_bytes(statistics) + b"\n"
                ):
                    raise SafetyHalt(
                        "canary statistics artifact is not canonical"
                    )
                statistics_payload = dict(statistics)
                statistics_identity = statistics_payload.pop(
                    "manifest_sha256", None
                )
                statistics_safety_failures = statistics.get(
                    "safety_failures"
                )
                expected_statistics = {
                    "schema_version": 1,
                    "contract": "risk-score-canary-fresh-audit-statistics-v1",
                    "finalized": True,
                    "decision": audit_decision,
                    "candidate_hash": candidate_hash,
                    "reference_hash": promotion_intent[
                        "tested_champion_hash"
                    ],
                    "policy_hash": self.runtime.controller.policy_hash,
                    "suite_manifest_hash":
                        self.runtime.controller.suite_manifest_hash,
                    "audit_schedule_hash":
                        self.runtime.controller.audit_schedule_hash,
                    "color_pairs": required_pairs,
                    "pair_ids": pair_ids,
                    "pair_ids_sha256": pair_ids_hash,
                }
                if (
                    statistics_identity
                    != canonical_sha256(statistics_payload)
                    or any(
                        statistics.get(key) != value
                        for key, value in expected_statistics.items()
                    )
                    or statistics_safety_failures != audit_safety_failures
                ):
                    raise SafetyHalt(
                        "canary statistics artifact does not satisfy policy"
                    )
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

    def mark_rollout_failed(
        self,
        generation_id: str,
        candidate_hash: str,
        phase: str,
        *,
        report_path: Path,
        report_hash: str,
    ) -> Path:
        """Quarantine a pre-activation generation from derived rollout failure."""

        if not self.automatic:
            raise SafetyHalt("rollout failure mutation is disabled")
        if phase not in {"canary", "intermediate"}:
            raise ValueError("rollout failure phase must be canary or intermediate")
        with self._writer_lock():
            report = self._validate_rollout_health_report(
                generation_id,
                candidate_hash,
                phase,
                report_path,
                report_hash,
                expected_decision="FAIL",
            )
            transaction = self._transaction_dir(generation_id)
            marker = transaction / f"{phase}-failure.json"
            _write_immutable_json(marker, report)
            state = self.registry.reconstruct()
            generation = state.generations.get(generation_id)
            if generation is None or generation.candidate_hash != candidate_hash:
                raise SafetyHalt("rollout failure names an unknown generation")
            if generation.state == GenerationState.QUARANTINED:
                return marker
            expected_state = (
                GenerationState.CANARY
                if phase == "canary"
                else GenerationState.ROLLOUT
            )
            if generation.state != expected_state:
                raise SafetyHalt(
                    f"{phase} failure contradicts generation state "
                    f"{generation.state.value}"
                )
            intent = json.loads((transaction / "intent.json").read_text(encoding="utf-8"))
            provenance = self._provenance(
                intent["config_hash"],
                intent["schedule_hash"],
            )
            self.registry.transition_generation(
                generation_id,
                candidate_hash,
                generation.candidate_path,
                GenerationState.QUARANTINED,
                provenance=provenance,
                tested_champion_hash=generation.previous_champion_hash,
                reason=f"{phase} rollout evidence derived FAIL",
                actor=self.runtime.controller.actor,
                payload={
                    "report_path": str(Path(report_path).resolve()),
                    "report_hash": report_hash,
                },
            )
            self._unpin_suite_evaluation_locked(candidate_hash)
            return marker

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
        admission_intent_path = transaction / "generation-data-admission-intent.json"
        if not source.is_dir() and not destination.is_dir():
            return False
        closed_manifests = []
        process_identities = []
        for worker_id in range(self.runtime.controller.worker_count):
            output = source / f"worker-{worker_id:03d}"
            if not output.is_dir() and destination.is_dir():
                output = destination / f"worker-{worker_id:03d}"
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
        manifest_root = source if source.is_dir() else destination
        manifest_hash, _ = _tree_manifest(manifest_root)
        admission_intent = {
            "schema_version": 1,
            "generation_id": generation_id,
            "candidate_hash": candidate_hash,
            "source_path": str(source),
            "destination_path": str(destination),
            "manifest_hash": manifest_hash,
            "closed_file_manifests": closed_manifests,
            "process_identities": process_identities,
            "requires_active_registry_generation": True,
        }
        _write_immutable_json(admission_intent_path, admission_intent)
        generation = self.registry.reconstruct().generations.get(generation_id)
        if (
            generation is None
            or generation.state != GenerationState.ACTIVE
            or generation.candidate_hash != candidate_hash
        ):
            raise SafetyHalt(
                "generation data cannot become shuffler-visible before activation"
            )
        if destination.is_dir() and not source.exists():
            self._mark(
                transaction,
                "generation-data-admitted",
                {"model_hash": candidate_hash, "manifest_hash": manifest_hash},
            )
            return True
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
        if self.suite_registry is not None:
            self._validate_suite_runtime_binding()
        readiness_errors = self._promotion_readiness_errors()
        if readiness_errors:
            raise SafetyHalt(
                "promotion readiness invariant is not satisfied: "
                + ",".join(readiness_errors)
            )
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
        generation_shuffle_contract = (
            shuffle_watermark.get("contract")
            == "risk-score-generation-shuffle-watermark-v1"
        )
        if generation_shuffle_contract:
            derived_by_generation = shuffle_watermark.get(
                "derived_paths_by_generation"
            )
            consumed_by_generation = shuffle_watermark.get(
                "trainer_consumed_by_generation"
            )
            if (
                not isinstance(derived_by_generation, dict)
                or not isinstance(consumed_by_generation, dict)
            ):
                raise SafetyHalt(
                    "generation shuffle watermark lacks rollback lineage"
                )
            shuffle_paths = derived_by_generation.get(generation_id, [])
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
            self._reconcile_suite_data_plane_locked()
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
                or report.policy_hash != self.runtime.controller.policy_hash
                or report.evaluation_key != candidate.evaluation_key
            ):
                raise SafetyHalt("promotion report identity is stale")
            if (
                report_value.get("selfplay_config_hash")
                != self.runtime.controller.selfplay_config_hash
                or report_value.get("policy_version")
                != self.runtime.frozen_policy.get("policy_version")
                or report_value.get("suite_manifest_path")
                != str(self.runtime.suites / "manifest.json")
                or report_value.get("suite_manifest_hash")
                != self.runtime.controller.suite_manifest_hash
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
            adaptive_context = self._load_adaptive_promotion_context(
                transaction=transaction,
                existing_intent=existing_intent,
                artifact=artifact,
                candidate_hash=candidate_hash,
                generation_id=generation_id,
                tested_champion_hash=report.tested_champion_hash,
                pass_report_path=Path(pass_report_path),
                pass_report_hash=pass_report_hash,
                trainer_checkpoint_hash=trainer_checkpoint_hash,
                data_watermark_hash=data_watermark_hash,
                shuffle_watermark_hash=shuffle_watermark_hash,
            )
            transaction.mkdir(parents=True, exist_ok=True)
            if adaptive_context is not None:
                adaptive_metadata = adaptive_context["metadata"]
                if not isinstance(adaptive_metadata, Mapping):
                    raise SafetyHalt("adaptive handoff context is malformed")
                self._mark(
                    transaction,
                    "training-handoff-intent",
                    adaptive_metadata,
                )
                self._checkpoint("promotion-training-handoff-intent")
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
                "shuffle_watermark_contract": shuffle_watermark.get(
                    "contract"
                ),
                "derived_shuffle_paths": authoritative_shuffle_paths,
            }
            if adaptive_context is not None:
                intent["adaptive_training"] = dict(
                    adaptive_context["metadata"]
                )
            _write_immutable_json(transaction / "intent.json", intent)
            if not (transaction / "previous-champion.json").exists():
                _write_immutable_json(
                    transaction / "previous-champion.json",
                    load_champion(self.runtime.champion_path).to_dict(),
                )
            if adaptive_context is None:
                self._snapshot_checkpoint(generation_id, trainer_checkpoint_hash)
            self._checkpoint("promotion-intent-written")
            if generation is not None and generation.state == GenerationState.ACTIVE:
                if load_champion(self.runtime.champion_path).champion_hash != candidate_hash:
                    raise SafetyHalt("active generation contradicts champion projection")
                if adaptive_context is not None:
                    if not (transaction / "training-commit.json").exists():
                        raise SafetyHalt(
                            "active adaptive generation has no training commit"
                        )
                    self._converge_adaptive_training_handoff(
                        transaction=transaction,
                        generation_id=generation_id,
                        candidate_hash=candidate_hash,
                        intent=intent,
                        context=adaptive_context,
                    )
                self._record_suite_accepted_champion_locked(
                    generation_id,
                    candidate_hash,
                    report.tested_champion_hash,
                )
                acknowledged = self._acknowledged(
                    generation_id, candidate_hash
                )
                if not self._commit_generation_data(
                    generation_id, candidate_hash
                ):
                    return {
                        "status": "WAITING_GENERATION_DATA",
                        "acknowledged": sorted(acknowledged),
                    }
                self._checkpoint("promotion-generation-data-admitted")
                self._schedule_deep_audit_if_needed(
                    generation_id,
                    candidate_hash,
                    report_value,
                )
                self._mark(transaction, "complete", {"champion_hash": candidate_hash})
                self._reconcile_suite_pins_locked()
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
            if adaptive_context is not None:
                adaptive_metadata = adaptive_context["metadata"]
                pins += (
                    (
                        "training-handoff",
                        adaptive_metadata["handoff_file_sha256"],
                        "adaptive-training-handoff",
                    ),
                    (
                        "resume-checkpoint",
                        adaptive_metadata["resume_checkpoint_sha256"],
                        "adaptive-resume-checkpoint",
                    ),
                    (
                        "training-recipe",
                        adaptive_metadata["recipe_sha256"],
                        "adaptive-training-recipe",
                    ),
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
            activated_recipe = None
            if adaptive_context is not None:
                activated_recipe = self._converge_adaptive_training_handoff(
                    transaction=transaction,
                    generation_id=generation_id,
                    candidate_hash=candidate_hash,
                    intent=intent,
                    context=adaptive_context,
                )
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
            if adaptive_context is not None:
                assert isinstance(activated_recipe, Mapping)
                adaptive_metadata = adaptive_context["metadata"]
                self._mark(
                    transaction,
                    "training-commit",
                    {
                        "generation_id": generation_id,
                        "candidate_hash": candidate_hash,
                        "handoff_file_sha256":
                            adaptive_metadata["handoff_file_sha256"],
                        "resume_checkpoint_sha256":
                            adaptive_metadata["resume_checkpoint_sha256"],
                        "recipe_record_sha256":
                            activated_recipe["record_sha256"],
                        "champion_hash": candidate_hash,
                    },
                )
                self._checkpoint("promotion-training-commit")
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
            self._checkpoint("promotion-active-event")
            self._record_suite_accepted_champion_locked(
                generation_id,
                candidate_hash,
                report.tested_champion_hash,
            )
            if not self._commit_generation_data(generation_id, candidate_hash):
                return {
                    "status": "WAITING_GENERATION_DATA",
                    "acknowledged": sorted(acknowledged),
                }
            self._checkpoint("promotion-generation-data-admitted")
            self._schedule_deep_audit_if_needed(
                generation_id,
                candidate_hash,
                report_value,
            )
            self._mark(transaction, "complete", {"champion_hash": candidate_hash})
            self._reconcile_suite_pins_locked()
            return {"status": "ACTIVE", "generation_id": generation_id, "candidate_hash": candidate_hash}

    def _adaptive_handoff_was_applied(
        self,
        transaction: Path,
        intent: Mapping[str, Any],
    ) -> bool:
        metadata = intent.get("adaptive_training")
        if not isinstance(metadata, Mapping):
            return False
        if any(
            (transaction / name).exists()
            for name in (
                "training-checkpoint-installed.json",
                "training-recipe-cas.json",
                "training-commit.json",
            )
        ):
            return True
        snapshot = (
            self.runtime.rollback_quarantine
            / str(intent.get("generation_id"))
            / "trainer-checkpoint"
        )
        resume_hash = metadata.get("resume_checkpoint_sha256")
        return (
            isinstance(resume_hash, str)
            and _SHA_RE.fullmatch(resume_hash) is not None
            and snapshot.is_file()
            and self.runtime.trainer_checkpoint.is_file()
            and sha256_file(self.runtime.trainer_checkpoint) == resume_hash
        )

    def _rollback_adaptive_recipe(
        self,
        transaction: Path,
        intent: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        metadata = intent.get("adaptive_training")
        if not isinstance(metadata, Mapping):
            raise SafetyHalt("adaptive rollback has no pinned handoff metadata")
        active_path = self._canonical_binding_path(
            metadata.get("active_recipe_path"),
            "adaptive active recipe",
        )
        expected_parent_record = _hash(
            metadata.get("active_recipe_record_sha256"),
            "adaptive parent recipe record hash",
        )
        try:
            current = load_recipe_binding(active_path)
        except (AdaptiveTrainingError, FileNotFoundError, OSError) as exc:
            raise SafetyHalt(
                f"cannot load adaptive recipe during rollback: {exc}"
            ) from exc

        rollback_marker_path = transaction / "training-recipe-rolled-back.json"
        if rollback_marker_path.exists():
            marker = self._marker_payload(rollback_marker_path)
            if (
                marker.get("restored_record_sha256")
                != current["record_sha256"]
                or marker.get("parent_record_sha256")
                != expected_parent_record
            ):
                raise SafetyHalt(
                    "adaptive recipe rollback marker contradicts active recipe"
                )
            return current

        cas_marker_path = transaction / "training-recipe-cas.json"
        if cas_marker_path.exists():
            cas_marker = self._marker_payload(cas_marker_path)
        elif current["record_sha256"] == expected_parent_record:
            self._mark(
                transaction,
                "training-recipe-rolled-back",
                {
                    "generation_id": intent["generation_id"],
                    "candidate_hash": intent["candidate_hash"],
                    "parent_record_sha256": expected_parent_record,
                    "activated_record_sha256": None,
                    "restored_record_sha256": current["record_sha256"],
                    "rollback_applied": False,
                },
            )
            self._checkpoint("rollback-training-recipe-restored")
            return current
        else:
            rollback_value = current.get("rollback")
            target_identity = {
                "recipe_sha256": metadata.get("recipe_sha256"),
                "recipe_path": metadata.get("recipe_path"),
                "champion_model_sha256": intent["candidate_hash"],
                "champion_checkpoint_sha256":
                    metadata.get("resume_checkpoint_sha256"),
                "admitted_data_manifest_sha256":
                    metadata.get("parent_admitted_manifest_sha256"),
                "data_watermark_sha256s":
                    metadata.get("data_watermark_sha256s"),
                "generation_id": intent["generation_id"],
            }
            if (
                not self._recipe_binding_matches(current, target_identity)
                or current.get("previous_record_sha256")
                != expected_parent_record
                or not isinstance(rollback_value, Mapping)
                or rollback_value.get("source_record_sha256")
                != expected_parent_record
            ):
                raise SafetyHalt(
                    "adaptive recipe state cannot be reconciled for rollback"
                )
            cas_marker = {
                "generation_id": intent["generation_id"],
                "candidate_hash": intent["candidate_hash"],
                "recipe_sha256": metadata["recipe_sha256"],
                "previous_record_sha256": expected_parent_record,
                "record_sha256": current["record_sha256"],
                "rollback": dict(rollback_value),
            }
            self._mark(
                transaction,
                "training-recipe-cas",
                cas_marker,
            )

        activated_record = _hash(
            cas_marker.get("record_sha256"),
            "activated adaptive recipe record hash",
        )
        rollback_value = cas_marker.get("rollback")
        if (
            cas_marker.get("previous_record_sha256") != expected_parent_record
            or not isinstance(rollback_value, Mapping)
            or rollback_value.get("source_record_sha256")
            != expected_parent_record
        ):
            raise SafetyHalt("adaptive recipe CAS rollback lineage is invalid")

        if current["record_sha256"] == activated_record:
            try:
                restored = rollback_recipe_binding(
                    active_path,
                    expected_record_sha256=activated_record,
                    rollback=rollback_value,
                )
            except (AdaptiveTrainingError, FileNotFoundError, OSError) as exc:
                raise SafetyHalt(
                    f"adaptive recipe rollback CAS failed: {exc}"
                ) from exc
        else:
            restore_identity = {
                "recipe_sha256": rollback_value.get("restore_recipe_sha256"),
                "recipe_path": rollback_value.get("restore_recipe_path"),
                "champion_model_sha256":
                    rollback_value.get("restore_champion_model_sha256"),
                "champion_checkpoint_sha256":
                    rollback_value.get("restore_champion_checkpoint_sha256"),
                "admitted_data_manifest_sha256":
                    rollback_value.get(
                        "restore_admitted_data_manifest_sha256"
                    ),
                "data_watermark_sha256s":
                    rollback_value.get("restore_data_watermark_sha256s"),
                "generation_id": rollback_value.get("restore_generation_id"),
            }
            if (
                current.get("previous_record_sha256") != activated_record
                or not self._recipe_binding_matches(current, restore_identity)
            ):
                raise SafetyHalt(
                    "adaptive recipe rollback replay found a foreign binding"
                )
            restored = current

        self._mark(
            transaction,
            "training-recipe-rolled-back",
            {
                "generation_id": intent["generation_id"],
                "candidate_hash": intent["candidate_hash"],
                "parent_record_sha256": expected_parent_record,
                "activated_record_sha256": activated_record,
                "restored_record_sha256": restored["record_sha256"],
                "rollback_applied": True,
            },
        )
        self._checkpoint("rollback-training-recipe-restored")
        return restored

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
        if self.suite_registry is not None:
            self._validate_suite_runtime_binding()
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
            lineage_paths = list(intent.get("derived_shuffle_paths", []))
            effective_trainer_consumed = trainer_consumed
            live_shuffle_watermark_hash = None
            if (
                intent.get("shuffle_watermark_contract")
                == "risk-score-generation-shuffle-watermark-v1"
            ):
                if derived_shuffle_paths or trainer_consumed:
                    raise SafetyHalt(
                        "generation rollback lineage cannot be supplied by a caller"
                    )
                from risk_score.promotion_feedback import (
                    load_shuffle_watermark,
                )

                live_watermark = load_shuffle_watermark(
                    self.runtime.shuffle_watermark_path
                )
                live_shuffle_watermark_hash = sha256_file(
                    self.runtime.shuffle_watermark_path
                )
                by_generation = live_watermark.get(
                    "derived_paths_by_generation"
                )
                consumed_by_generation = live_watermark.get(
                    "trainer_consumed_by_generation"
                )
                if (
                    not isinstance(by_generation, dict)
                    or not isinstance(consumed_by_generation, dict)
                    or generation_id not in by_generation
                    or generation_id not in consumed_by_generation
                    or not isinstance(
                        by_generation[generation_id], list
                    )
                    or any(
                        not isinstance(path, str)
                        for path in by_generation[generation_id]
                    )
                    or type(consumed_by_generation[generation_id]) is not bool
                ):
                    raise SafetyHalt(
                        "live generation rollback watermark is malformed"
                    )
                lineage_paths = list(
                    by_generation[generation_id]
                )
                effective_trainer_consumed = consumed_by_generation[
                    generation_id
                ]
            elif derived_shuffle_paths:
                raise SafetyHalt(
                    "rollback paths must come from the frozen promotion lineage"
                )
            if generation.state == GenerationState.ROLLED_BACK:
                self._unpin_suite_evaluation_locked(
                    generation.candidate_hash
                )
                return {"status": "ROLLED_BACK", "generation_id": generation_id}
            adaptive_transaction = isinstance(
                intent.get("adaptive_training"),
                Mapping,
            )
            adaptive_applied = self._adaptive_handoff_was_applied(
                transaction,
                intent,
            )
            if (
                adaptive_transaction
                and generation.state == GenerationState.ACTIVE
                and not adaptive_applied
            ):
                raise SafetyHalt(
                    "active adaptive generation has no applied handoff journal"
                )
            allowed_states = {
                GenerationState.ACTIVE,
                GenerationState.ROLLBACK_PENDING,
            }
            if adaptive_applied:
                allowed_states.add(GenerationState.ROLLOUT)
            if generation.state not in allowed_states:
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
            if generation.state == GenerationState.ROLLOUT:
                champion_projection = load_champion(
                    self.runtime.champion_path
                ).champion_hash
                if (
                    state.current_champion_hash
                    != generation.previous_champion_hash
                    or champion_projection
                    not in {
                        generation.previous_champion_hash,
                        generation.candidate_hash,
                    }
                ):
                    raise SafetyHalt(
                        "partially applied adaptive rollback has a foreign "
                        "champion projection"
                    )
            provenance = self._provenance(intent["config_hash"], intent["schedule_hash"])
            quarantine = self.runtime.rollback_quarantine / generation_id / "data"
            staged = self.runtime.rollout_quarantine / generation_id
            admitted = self.runtime.admitted_selfplay / generation_id
            requested_moves = [
                ("staged-rollout", staged, quarantine / "staged-rollout"),
                ("admitted-generation", admitted, quarantine / "admitted-generation"),
            ]
            for index, item in enumerate(lineage_paths):
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
                    "live_shuffle_watermark_hash":
                        live_shuffle_watermark_hash,
                    "trainer_consumed": effective_trainer_consumed,
                    "moves": move_intents,
                }
                if adaptive_transaction:
                    rollback_intent["adaptive_handoff_applied"] = adaptive_applied
                _write_immutable_json(rollback_intent_path, rollback_intent)
            if adaptive_transaction and (
                rollback_intent.get("adaptive_handoff_applied")
                is not adaptive_applied
            ):
                raise SafetyHalt(
                    "adaptive rollback intent contradicts applied handoff state"
                )
            if generation.state != GenerationState.ROLLBACK_PENDING:
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
            if (
                not isinstance(stop_result, Mapping)
                or stop_result.get("quiescent") is not True
                or stop_result.get("quiescent_roles") != _ALL_ROLE_QUIESCENCE
            ):
                raise SafetyHalt("rollback stop command lacks all-role quiescence proof")
            if adaptive_applied:
                self._training_gpu_lease_proof()
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
            if effective_trainer_consumed or adaptive_applied:
                expected = intent["trainer_checkpoint_hash"]
                self._restore_checkpoint_snapshot(
                    generation_id,
                    expected,
                )
                if adaptive_applied:
                    adaptive_metadata = intent["adaptive_training"]
                    self._mark(
                        transaction,
                        "training-checkpoint-rolled-back",
                        {
                            "generation_id": generation_id,
                            "candidate_hash": generation.candidate_hash,
                            "installed_checkpoint_sha256":
                                adaptive_metadata[
                                    "resume_checkpoint_sha256"
                                ],
                            "restored_checkpoint_sha256": expected,
                            "destination_path":
                                str(self.runtime.trainer_checkpoint),
                        },
                    )
                self._checkpoint("rollback-checkpoint-restored")
                if adaptive_applied:
                    self._checkpoint(
                        "rollback-training-checkpoint-restored"
                    )
            if adaptive_applied:
                # Recipe rollback restores the prior metadata projection only.
                # Watermark files may contain newer immutable lineage and are
                # intentionally never byte-rewound by this transaction.
                self._rollback_adaptive_recipe(transaction, intent)
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
            self._unpin_suite_evaluation_locked(
                generation.candidate_hash
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

    required_envelope = {
        "controller_stage",
        "candidate_hash",
        "tested_champion_hash",
        "original_hash",
        "evaluation_key",
        "config_hash",
        "schedule_hash",
        "policy_path",
        "policy_hash",
        "policy_version",
        "suite_manifest_path",
        "suite_manifest_hash",
        "look",
        "selfplay_config_hash",
        "topology",
        "gpu_handoff_hash",
    }
    missing = sorted(required_envelope.difference(evidence))
    if missing:
        raise SafetyHalt(
            "configured evidence envelope is incomplete: "
            + ", ".join(missing)
        )
    stage = evidence.get("controller_stage")
    if stage not in {"integrity", "screen", "finalist", "confirmation"}:
        raise SafetyHalt(f"configured evidence has unknown stage: {stage!r}")
    policy_path_value = evidence.get("policy_path")
    if not isinstance(policy_path_value, str):
        raise SafetyHalt("configured evidence has no runtime-bound policy path")
    policy_path = Path(policy_path_value)
    if (
        not policy_path.is_absolute()
        or policy_path.is_symlink()
        or not policy_path.is_file()
    ):
        raise SafetyHalt("configured evidence policy path is not a regular file")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetyHalt(f"configured evidence policy is invalid: {exc}") from exc
    if not isinstance(policy, dict):
        raise SafetyHalt("configured evidence policy root must be an object")
    policy_hash = canonical_sha256(policy)
    policy_version = policy.get("policy_version")
    if (
        policy_hash != evidence.get("policy_hash")
        or policy_version != evidence.get("policy_version")
    ):
        raise SafetyHalt(
            "configured evidence policy version/hash differs from runtime binding"
        )
    if stage == "confirmation":
        promotion_evidence = evidence.get("promotion_evidence")
        if not isinstance(promotion_evidence, Mapping):
            raise SafetyHalt(
                "confirmation adapter evidence must contain promotion_evidence"
            )
        if (
            promotion_evidence.get("policy_hash") != policy_hash
            or promotion_evidence.get("policy_version") != policy_version
        ):
            raise SafetyHalt(
                "promotion evidence policy version/hash differs from runtime binding"
            )
        from risk_score.promotion_gate import evaluate_promotion_gate

        result = dict(evaluate_promotion_gate(
            promotion_evidence,
            policy=policy,
            expected_policy_hash=evidence["policy_hash"],
            expected_candidate_hash=evidence["candidate_hash"],
            expected_champion_hash=evidence["tested_champion_hash"],
            expected_original_hash=evidence["original_hash"],
        ))
        if (
            result.get("policy_hash") != policy_hash
            or result.get("policy_version") != policy_version
        ):
            raise SafetyHalt(
                "configured gate evaluated a different policy version/hash"
            )
        result.update(
            {
                "gpu_handoff_hash": evidence["gpu_handoff_hash"],
                "selfplay_config_hash": evidence["selfplay_config_hash"],
                "topology": evidence["topology"],
                "policy_path": str(policy_path),
                "suite_manifest_path": evidence["suite_manifest_path"],
                "suite_manifest_hash": evidence["suite_manifest_hash"],
                "look": evidence["look"],
            }
        )
        return result

    stage_gate = evidence.get("stage_gate")
    stage_evidence = evidence.get("stage_evidence")
    if isinstance(stage_gate, Mapping) and stage_gate.get(
        "contract"
    ) == "risk-score-derived-stage-evidence-v1":
        derivation_hash = stage_gate.get("derivation_hash")
        _hash(derivation_hash, "derived stage evidence hash")
        stage_payload = dict(stage_gate)
        stage_payload.pop("derivation_hash", None)
        if canonical_sha256(stage_payload) != derivation_hash:
            raise SafetyHalt(
                "derived non-confirmation stage evidence hash is invalid"
            )
        validated_stage = dict(stage_gate)
        stage_evidence_hash = derivation_hash
    elif isinstance(stage_evidence, Mapping):
        stage_evidence_hash = evidence.get("stage_evidence_hash")
        _hash(stage_evidence_hash, "stage evidence hash")
        if canonical_sha256(stage_evidence) != stage_evidence_hash:
            raise SafetyHalt("non-confirmation stage evidence hash is invalid")
        validated_stage = dict(stage_evidence)
        if isinstance(stage_gate, Mapping):
            if stage_gate.get("decision") != validated_stage.get("decision"):
                raise SafetyHalt(
                    "stage gate decision contradicts validated stage evidence"
                )
            supplied_stage_hash = stage_gate.get("stage_evidence_hash")
            if supplied_stage_hash not in {None, stage_evidence_hash}:
                raise SafetyHalt(
                    "stage gate is bound to different stage evidence"
                )
    else:
        raise SafetyHalt(
            "non-confirmation adapter evidence must contain derived stage evidence"
        )
    decision = validated_stage.get("decision")
    if decision not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        raise SafetyHalt(f"configured stage gate has invalid decision: {decision!r}")
    if validated_stage.get("contract") == "risk-score-derived-stage-evidence-v1":
        derived_artifacts = validated_stage.get("derived_artifacts")
        if not isinstance(derived_artifacts, Mapping):
            raise SafetyHalt("derived stage evidence has no artifacts")
        if stage == "integrity":
            stage0 = derived_artifacts.get("stage_0")
            if not isinstance(stage0, Mapping):
                raise SafetyHalt("derived Stage-0 evidence is missing")
            derived_decision = (
                "PASS" if stage0.get("stage_0_passed") is True else "FAIL"
            )
        else:
            statistics = derived_artifacts.get("statistics")
            if not isinstance(statistics, Mapping) or not statistics:
                raise SafetyHalt("derived match stage statistics are missing")
            champion_finalized = None
            for name, finalized in statistics.items():
                if not isinstance(finalized, Mapping):
                    raise SafetyHalt(
                        f"derived statistics cell {name!r} is malformed"
                    )
                artifact = finalized.get("statistics_artifact")
                manifest = finalized.get("statistics_manifest")
                artifact_hash = finalized.get(
                    "statistics_artifact_hash"
                )
                manifest_hash = finalized.get(
                    "statistics_manifest_hash"
                )
                if (
                    not isinstance(artifact, Mapping)
                    or not isinstance(manifest, Mapping)
                    or artifact_hash != canonical_sha256(artifact)
                    or manifest_hash != canonical_sha256(manifest)
                    or manifest.get("statistics_artifact_hash")
                    != artifact_hash
                ):
                    raise SafetyHalt(
                        f"derived statistics cell {name!r} hash is invalid"
                    )
                if _policy_get(
                    artifact,
                    "data_binding",
                    "comparison",
                ) == "candidate-vs-champion-powered":
                    champion_finalized = finalized
            if not isinstance(champion_finalized, Mapping):
                raise SafetyHalt(
                    "derived statistics omit champion powered cell"
                )
            champion_artifact = champion_finalized[
                "statistics_artifact"
            ]
            metric = _policy_get(
                champion_artifact,
                "metrics",
                "realized_utility",
                default={},
            )
            valid = (
                _policy_get(
                    champion_artifact,
                    "validation",
                    "promotion_valid",
                )
                is True
            )
            available = (
                isinstance(metric, Mapping)
                and metric.get("available") is True
                and metric.get("complete") is True
            )
            if not valid:
                derived_decision = "FAIL"
            elif not available:
                derived_decision = "INCONCLUSIVE"
            elif stage == "screen":
                upper = _finite_number(metric.get("upper_bound"))
                threshold = _finite_number(
                    _policy_get(
                        policy,
                        "evaluation_stages",
                        "stage_1_cheap_paired_screen",
                        "utility_futility_upper_bound",
                    )
                )
                if upper is None or threshold is None:
                    derived_decision = "INCONCLUSIVE"
                elif upper <= threshold:
                    derived_decision = "FAIL"
                else:
                    derived_decision = "PASS"
            else:
                derived_decision = "PASS"
        if decision != derived_decision:
            raise SafetyHalt(
                "stage decision is not derivable from validated artifacts"
            )
    expected = {
        "finalized": True,
        "controller_stage": stage,
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
        key
        for key, value in expected.items()
        if validated_stage.get(key) != value
    ]
    if conflicts:
        raise SafetyHalt(
            "validated stage evidence contradicts evaluator envelope: "
            + ", ".join(sorted(conflicts))
        )
    source_hashes = validated_stage.get(
        "source_artifact_hashes",
        validated_stage.get("artifact_hashes"),
    )
    if isinstance(source_hashes, Mapping):
        normalized_hashes = dict(source_hashes)
    else:
        normalized_hashes: Dict[str, str] = {}

        def collect_hashes(value: Any, prefix: str = "") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    path = f"{prefix}.{key}" if prefix else str(key)
                    if (
                        isinstance(item, str)
                        and _SHA_RE.fullmatch(item) is not None
                        and (
                            str(key).endswith("hash")
                            or str(key).endswith("_sha256")
                        )
                    ):
                        normalized_hashes[path] = item
                    else:
                        collect_hashes(item, path)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    collect_hashes(item, f"{prefix}[{index}]")

        collect_hashes(validated_stage.get("derived_artifacts"))
    if not normalized_hashes or any(
        not isinstance(value, str) or _SHA_RE.fullmatch(value) is None
        for value in normalized_hashes.values()
    ):
        raise SafetyHalt(
            "non-confirmation stage evidence lacks hash-bound source artifacts"
        )
    ranking_summary = validated_stage.get("ranking_summary")
    if stage == "finalist" and not isinstance(ranking_summary, Mapping):
        statistics = _policy_get(
            validated_stage,
            "derived_artifacts",
            "statistics",
            default={},
        )
        champion_statistics = None
        if isinstance(statistics, Mapping):
            for finalized in statistics.values():
                artifact = (
                    finalized.get("statistics_artifact")
                    if isinstance(finalized, Mapping)
                    else None
                )
                binding = (
                    artifact.get("data_binding")
                    if isinstance(artifact, Mapping)
                    else None
                )
                if (
                    isinstance(binding, Mapping)
                    and binding.get("comparison")
                    == "candidate-vs-champion-powered"
                ):
                    champion_statistics = finalized
                    break
        artifact = (
            champion_statistics.get("statistics_artifact")
            if isinstance(champion_statistics, Mapping)
            else None
        )
        utility = _policy_get(
            artifact or {},
            "metrics",
            "realized_utility",
            "lower_bound",
        )
        risk = _policy_get(
            artifact or {},
            "risk_differences",
            "final_50",
            "upper_bound",
        )
        utility_value = _finite_number(utility)
        risk_value = _finite_number(risk)
        sample_count = evidence.get("candidate_sample_count")
        candidate_manifest_hash = evidence.get("candidate_manifest_hash")
        if (
            utility_value is None
            or risk_value is None
            or type(sample_count) is not int
            or sample_count < 0
            or not isinstance(candidate_manifest_hash, str)
            or _SHA_RE.fullmatch(candidate_manifest_hash) is None
            or not isinstance(champion_statistics, Mapping)
        ):
            raise SafetyHalt(
                "finalist stage evidence lacks source-bound ranking statistics"
            )
        statistics_artifact_hash = champion_statistics.get(
            "statistics_artifact_hash"
        )
        statistics_manifest_hash = champion_statistics.get(
            "statistics_manifest_hash"
        )
        _hash(
            statistics_artifact_hash,
            "finalist statistics artifact hash",
        )
        _hash(
            statistics_manifest_hash,
            "finalist statistics manifest hash",
        )
        ranking_summary = {
            "schema_version": 1,
            "source_bound": True,
            "source_cell": "powered_candidate_vs_champion",
            "candidate_hash": evidence["candidate_hash"],
            "statistics_artifact_hash": statistics_artifact_hash,
            "statistics_manifest_hash": statistics_manifest_hash,
            "candidate_manifest_hash": candidate_manifest_hash,
            "realized_powered_utility_lower_bound": utility_value,
            "final50_risk_upper_bound": risk_value,
            "final_50_risk_upper_bound": risk_value,
            "sample_count": sample_count,
        }
    return {
        **validated_stage,
        **expected,
        "policy_path": evidence["policy_path"],
        "policy_version": evidence["policy_version"],
        "suite_manifest_path": evidence["suite_manifest_path"],
        "suite_manifest_hash": evidence["suite_manifest_hash"],
        "look": evidence["look"],
        "decision": decision,
        "stage_evidence_hash": stage_evidence_hash,
        "source_artifact_hashes": normalized_hashes,
        **(
            {"ranking_summary": dict(ranking_summary)}
            if isinstance(ranking_summary, Mapping)
            else {}
        ),
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

    from risk_score.gpu_lease import (
        GpuLeaseManager,
        ProcessIdentity,
        RuntimeConfig as GpuRuntimeConfig,
    )
    from risk_score.promotion_host import same_process

    config = GpuRuntimeConfig.from_json_file(config_path)
    manager = GpuLeaseManager(config)
    proof: Dict[str, Any] = {}
    trainer_identity = None
    if manager.read_record() is None:
        identity_path = (
            config.promotion_root / "supervisor" / "trainer.json"
        )
        if identity_path.is_symlink() or not identity_path.is_file():
            raise SafetyHalt(
                "first GPU handoff requires promotion/supervisor/trainer.json"
            )
        identity_value = json.loads(identity_path.read_text(encoding="utf-8"))
        identity = identity_value.get("process_identity")
        if not isinstance(identity, Mapping) or not same_process(identity):
            raise SafetyHalt("recorded trainer identity is absent or stale")
        trainer_identity = ProcessIdentity.capture(identity["pid"])
    with manager.exclusive_handoff(trainer_identity) as record:
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
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--bootstrap-champion", action="store_true")
    parser.add_argument("--bootstrap-champion-hash")
    parser.add_argument("--bootstrap-generation-id")
    parser.add_argument("--bootstrap-confirmation")
    return parser.parse_args(argv)


def _write_controller_status(
    path: Path,
    runtime: RuntimeConfig,
    result: Mapping[str, Any],
) -> None:
    target = Path(path)
    promotion_root = runtime.promotion_root.resolve()
    if (
        not target.is_absolute()
        or (target.exists() and target.is_symlink())
        or target.parent.resolve() != promotion_root
        or target.name != "status.json"
    ):
        raise ConfigurationError(
            "--status-output must be the non-symlink promotion/status.json path"
        )
    payload = {
        "schema_version": 1,
        "contract": "risk-score-controller-status-v1",
        "observed_at_utc": utc_timestamp(datetime.now(timezone.utc)),
        "controller_actor": runtime.controller.actor,
        "source_revision_hash": runtime.controller.source_hash,
        "policy_hash": runtime.controller.policy_hash,
        "result": result,
    }
    atomic_write_bytes(target, canonical_json_bytes(payload) + b"\n")


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
        if args.status_output is not None and not args.automatic:
            raise ConfigurationError("--status-output requires --automatic")
        verify_live_dependencies: Optional[Callable[[], None]] = None
        if args.automatic:
            from risk_score.build_live_runtime import verify_deployment_manifest
            from risk_score.promotion_host import same_process

            def verify_live_dependencies() -> None:
                verify_deployment_manifest(
                    Path(args.runtime_config).parent / "deployment-manifest.json"
                )
                service_path = (
                    runtime.promotion_root / "supervisor" / "service.json"
                )
                if service_path.is_symlink() or not service_path.is_file():
                    raise SafetyHalt(
                        "automatic mode requires the host supervisor service"
                    )
                service = json.loads(service_path.read_text(encoding="utf-8"))
                if (
                    not same_process(service.get("process_identity", {}))
                    or service.get("runtime_config")
                    != str(Path(args.runtime_config).resolve())
                    or service.get("mutation_enabled") is not True
                    or not isinstance(
                        service.get("updated_at_unix"), (int, float)
                    )
                    or time.time() - float(service["updated_at_unix"]) > 30.0
                ):
                    raise SafetyHalt("host supervisor service heartbeat is stale")

            verify_live_dependencies()
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
            process_identity_verifier=(
                same_process if args.automatic else None
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
            if verify_live_dependencies is not None:
                verify_live_dependencies()
            if args.mode == "reconcile":
                result = controller.run_reconcile()
            else:
                result = controller.run_once()
            print(json.dumps(result, sort_keys=True))
            if args.status_output is not None:
                _write_controller_status(args.status_output, runtime, result)
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
