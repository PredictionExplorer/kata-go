#!/usr/bin/env python3
"""Prepare and install the revision-bound production autonomy bootstrap.

The provisioner is intentionally a layer above ``autonomy_bootstrap_spec``.
Its path unit may be installed before the authoritative suite exists.  Once
both readiness artifacts are complete, it freezes the live control-plane
inputs, builds the gate executor specifications, and delegates the final
bootstrap contract to ``autonomy_bootstrap_spec.materialize_bootstrap``.

No command in this module starts training, admits a candidate, exports a
model, or enables the bootstrap service directly.  The only systemd units
that may be enabled are path units.  Systemd mutation is opt-in and accepts an
injected command runner for root-free tests.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import hashlib
import json
import math
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
)

from risk_score import autonomy_bootstrap, autonomy_bootstrap_spec
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256

SCHEMA_VERSION = 1
SPEC_CONTRACT = "risk-score-autonomy-provisioner-spec-v1"
PLAN_CONTRACT = "risk-score-autonomy-provisioner-plan-v1"
STATUS_CONTRACT = "risk-score-autonomy-provisioner-status-v1"
PREPARATION_RECEIPT_CONTRACT = "risk-score-autonomy-provisioner-preparation-receipt-v1"
READINESS_RECEIPT_CONTRACT = "risk-score-autonomy-provisioner-readiness-receipt-v1"
MATERIALIZATION_RECEIPT_CONTRACT = (
    "risk-score-autonomy-provisioner-materialization-receipt-v1"
)
INSTALLATION_RECEIPT_CONTRACT = (
    "risk-score-autonomy-provisioner-installation-receipt-v1"
)
FAILURE_RECEIPT_CONTRACT = "risk-score-autonomy-provisioner-failure-v1"

PREPARE_SERVICE_UNIT_NAME = "katago-risk-autonomy-prepare.service"
PREPARE_PATH_UNIT_NAME = "katago-risk-autonomy-prepare.path"
LEGACY_PATH_UNIT_NAME = "katago-risk-shadow-bootstrap-e7901739-v2.path"
BOOTSTRAP_SERVICE_UNIT_NAME = autonomy_bootstrap.BOOTSTRAP_UNIT_NAME
BOOTSTRAP_PATH_UNIT_NAME = autonomy_bootstrap_spec.BOOTSTRAP_PATH_UNIT_NAME
PRODUCTION_TARGET_UNIT_NAME = "katago-risk-training.target"

PROVISIONER_SPEC_CONTRACT = SPEC_CONTRACT
PROVISIONER_STATUS_CONTRACT = STATUS_CONTRACT
PREPARE_UNIT_NAME = PREPARE_SERVICE_UNIT_NAME
PATH_UNIT_NAME = PREPARE_PATH_UNIT_NAME

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.(?:path|service)$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,254})$")
_SERVICE_USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_TOPOLOGY_STARTUP_RESERVE_SECONDS = 5.0


def _lease_timeout_budget(outer_timeout: float) -> tuple[float, float, float]:
    from risk_score.autonomy_lease_probe import (
        DEFAULT_CLEANUP_MARGIN_SECONDS,
        DEFAULT_POLL_INTERVAL_SECONDS,
        OUTER_DRILL_TIMEOUT_RESERVE_SECONDS,
    )

    reserve = (
        OUTER_DRILL_TIMEOUT_RESERVE_SECONDS
        + DEFAULT_CLEANUP_MARGIN_SECONDS
        + DEFAULT_POLL_INTERVAL_SECONDS
    )
    if not math.isfinite(float(outer_timeout)) or float(outer_timeout) <= reserve:
        raise ProvisionerSpecError(
            "lease drill outer timeout must exceed startup reserve, cleanup margin, "
            "and poll interval"
        )
    return (
        float(outer_timeout) - reserve,
        DEFAULT_CLEANUP_MARGIN_SECONDS,
        DEFAULT_POLL_INTERVAL_SECONDS,
    )


_REQUIRED_INPUTS = frozenset(
    {
        "python_executable",
        "katago_binary",
        "trainer_spec",
        "consumer_spec",
        "original_model",
        "suite_champion_model",
        "trainer_checkpoint",
        "deployment_manifest",
        "promotion_policy",
        "autonomy_policy",
        "model_probe_config_source",
    }
)
_RUNTIME_COMMANDS = ("shuffler", "exporter", "evaluator")
_OUTPUT_KEYS = {
    "prepare_service_unit",
    "prepare_path_unit",
    "status",
    "receipts_root",
    "artifacts_root",
    "bootstrap_spec",
    "bootstrap_service_unit",
    "bootstrap_path_unit",
}
_SPEC_KEYS = {
    "schema_version",
    "contract",
    "repository",
    "source_revision",
    "run_root",
    "state_root",
    "bootstrap_state_root",
    "readiness",
    "candidate_inbox",
    "activation_destination",
    "legacy_path_unit",
    "immutable_inputs",
    "models",
    "extra_inputs",
    "runtime_commands",
    "publisher_config",
    "gpu",
    "actor",
    "service_user",
    "initial_generation_id",
    "created_at_utc",
    "poll_interval_seconds",
    "minimum_clean_observations",
    "outputs",
    "spec_sha256",
}
_PUBLISHER_CONFIG_KEYS = {
    "scheduler_directory",
    "cluster_executor",
    "adaptive_training",
    "suite_rotation",
    "topology_benchmark",
    "lease_drill",
    "promotion_drill",
    "shadow_runtime",
}
_CLUSTER_KEYS = {
    "state_directory",
    "owner_id",
    "gpu_ids",
    "poll_interval_seconds",
    "heartbeat_interval_seconds",
    "stale_after_seconds",
    "retry_budget",
    "backoff_initial_seconds",
    "backoff_max_seconds",
    "lease_proof_command",
    "lease_proof_timeout_seconds",
    "guardian_argv_prefix",
}
_ADAPTIVE_KEYS = {
    "root",
    "observation_path",
    "trial_command_argv_template",
    "poll_interval_seconds",
}
_SUITE_KEYS = {
    "registry_root",
    "root",
    "materializer_argv_template",
    "curation_argv_template",
    "continuity_argv_template",
    "results",
    "poll_interval_seconds",
}
_TOPOLOGY_KEYS = {"publisher_options", "timeout_seconds"}
_LEASE_KEYS = {
    "publisher_options",
    "probe_timeout_seconds",
    "probe_minimum_completed_work",
}
_PROMOTION_KEYS = {
    "disposable_root",
    "command_timeout_seconds",
    "max_evaluator_rows",
    "max_worker_games",
    "max_replay_attempts",
}
_SHADOW_KEYS = {"evaluator_process_count"}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ProvisionerError(RuntimeError):
    """Base class for provisioner failures."""


class ProvisionerSpecError(ProvisionerError, ValueError):
    """The canonical provisioner specification is malformed or stale."""


class ProvisionerDriftError(ProvisionerError):
    """A source, readiness, inventory, or generated input changed."""


class ProvisionerConflictError(ProvisionerError):
    """An immutable destination conflicts with the requested publication."""


class ProvisionerDependencyError(ProvisionerError):
    """A required publisher adapter is unavailable or incomplete."""


class ProvisionerApplyError(ProvisionerError):
    """A systemd cutover failed and rollback was attempted."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _ensure_finite(value: Any, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProvisionerSpecError(f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProvisionerSpecError(f"{role} contains a non-string key")
            _ensure_finite(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite(child, f"{role}[{index}]")


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvisionerSpecError(f"{role} must be an object")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ProvisionerSpecError(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _load_canonical_object(
    path: Path,
    role: str,
    *,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> dict[str, Any]:
    source = Path(path)
    _reject_symlink_ancestors(source, role)
    if source.is_symlink() or not source.is_file():
        raise ProvisionerSpecError(f"{role} must be a regular non-symlink file")
    if source.stat().st_size > maximum_bytes:
        raise ProvisionerSpecError(f"{role} exceeds the size limit")
    data = source.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProvisionerSpecError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvisionerSpecError(f"{role} root must be an object")
    _ensure_finite(value, role)
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ProvisionerSpecError(f"{role} must be canonical newline-terminated JSON")
    return value


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise ProvisionerSpecError(
                f"{role} has a symlinked path component: {current}"
            )
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(value: Any, role: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ProvisionerSpecError(f"{role} must be a nonempty absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise ProvisionerSpecError(f"{role} must be absolute and lexically normalized")
    _reject_symlink_ancestors(path, role)
    return path


def _required_file(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.is_symlink() or not path.is_file():
        raise ProvisionerSpecError(f"{role} must be a regular non-symlink file")
    return path


def _required_directory(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.is_symlink() or not path.is_dir():
        raise ProvisionerSpecError(f"{role} must be an existing non-symlink directory")
    return path


def _future_path(value: Any, role: str, *, directory: bool = False) -> Path:
    path = _absolute_path(value, role)
    if os.path.lexists(os.fspath(path)) and (
        path.is_symlink()
        or (directory and not path.is_dir())
        or (not directory and not path.is_file())
    ):
        kind = "directory" if directory else "file"
        raise ProvisionerSpecError(f"{role} must be a non-symlink {kind} when present")
    parent = path if directory and path.exists() else path.parent
    while not parent.exists():
        if parent.parent == parent:
            break
        parent = parent.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProvisionerSpecError(f"{role} has no safe existing ancestor")
    return path


def _strictly_within(path: Path, root: Path) -> bool:
    if path == root:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or _strictly_within(first, second)
        or _strictly_within(second, first)
    )


def _required_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProvisionerSpecError(f"{role} must be a lowercase SHA-256")
    return value


def _required_id(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ProvisionerSpecError(f"{role} is not a safe identifier")
    return value


def _required_integer(value: Any, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProvisionerSpecError(f"{role} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ProvisionerSpecError(f"{role} must be a positive finite number")
    return float(value)


def _validate_argv(
    value: Any, role: str, *, allow_none: bool = False
) -> tuple[str, ...]:
    if value is None and allow_none:
        return ()
    if (
        not isinstance(value, list)
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
        raise ProvisionerSpecError(f"{role} must be a nonempty argv string array")
    return tuple(value)


def _validate_bound_argv_paths(
    argv: Sequence[str],
    fixed_inputs: Mapping[Path, str],
    role: str,
) -> None:
    executable = _required_file(argv[0], f"{role}[0]")
    if executable not in fixed_inputs:
        raise ProvisionerSpecError(f"{role} executable must be hash-bound")
    for index, argument in enumerate(argv[1:], start=1):
        candidate_text = (
            argument.split("=", 1)[1]
            if argument.startswith("--") and "=" in argument
            else argument
        )
        candidate = Path(candidate_text)
        if not candidate.is_absolute() and (
            candidate_text.startswith((".", "~"))
            or "/" in candidate_text
            or "\\" in candidate_text
        ):
            raise ProvisionerSpecError(
                f"{role}[{index}] contains an unsupported relative path"
            )
        if not candidate.is_absolute() or not candidate.exists():
            continue
        path = _absolute_path(candidate_text, f"{role}[{index}]")
        if path.is_file() and path not in fixed_inputs:
            raise ProvisionerSpecError(f"{role}[{index}] file must be hash-bound")


def _parse_created_at(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProvisionerSpecError("created_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = _datetime.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProvisionerSpecError("created_at_utc is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != _datetime.timedelta(0):
        raise ProvisionerSpecError("created_at_utc must be UTC")
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    @classmethod
    def load(cls, value: Any, role: str) -> FileBinding:
        checked = _exact_keys(value, {"path", "sha256"}, role)
        path = _required_file(checked["path"], f"{role}.path")
        expected = _required_sha256(checked["sha256"], f"{role}.sha256")
        if file_sha256(path) != expected:
            raise ProvisionerSpecError(f"{role}.path hash changed")
        return cls(path, expected)

    def to_dict(self) -> Mapping[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}

    def verify(self, role: str) -> None:
        _reject_symlink_ancestors(self.path, role)
        if (
            self.path.is_symlink()
            or not self.path.is_file()
            or file_sha256(self.path) != self.sha256
        ):
            raise ProvisionerDriftError(f"{role} changed: {self.path}")


@dataclass(frozen=True)
class ReadinessBinding:
    path: Path
    sha256: str | None

    @classmethod
    def load(cls, value: Any, role: str) -> ReadinessBinding:
        checked = _exact_keys(value, {"path", "sha256"}, role)
        path = _absolute_path(checked["path"], f"{role}.path")
        expected = checked["sha256"]
        if expected is not None:
            expected = _required_sha256(expected, f"{role}.sha256")
        if os.path.lexists(os.fspath(path)):
            if path.is_symlink() or not path.is_file():
                raise ProvisionerSpecError(
                    f"{role}.path must be a regular non-symlink file"
                )
            if expected is not None and file_sha256(path) != expected:
                raise ProvisionerSpecError(f"{role}.path hash changed")
        elif expected is not None:
            raise ProvisionerSpecError(f"{role}.path is missing despite a bound hash")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise ProvisionerSpecError(f"{role}.path parent must already exist")
        return cls(path, expected)

    def to_dict(self) -> Mapping[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class ProvisionerOutputs:
    prepare_service_unit: Path
    prepare_path_unit: Path
    status: Path
    receipts_root: Path
    artifacts_root: Path
    bootstrap_spec: Path
    bootstrap_service_unit: Path
    bootstrap_path_unit: Path


@dataclass(frozen=True)
class ProvisionerSpec:
    path: Path
    file_sha256: str
    identity: str
    repository: Path
    source_revision: str
    run_root: Path
    state_root: Path
    bootstrap_state_root: Path
    curation_status: ReadinessBinding
    suite_manifest: ReadinessBinding
    candidate_inbox: Path
    activation_destination: Path
    legacy_path_unit: str
    immutable_inputs: Mapping[str, FileBinding]
    models: tuple[FileBinding, ...]
    extra_inputs: tuple[FileBinding, ...]
    runtime_commands: Mapping[str, tuple[str, ...]]
    publisher_config: Mapping[str, Any]
    gpu_index: int
    gpu_uuid: str
    actor: str
    service_user: str
    initial_generation_id: str
    created_at_utc: str
    poll_interval_seconds: float
    minimum_clean_observations: int
    outputs: ProvisionerOutputs
    raw: Mapping[str, Any]

    @property
    def original_generation_id(self) -> str:
        return (
            "immutable-original-" + self.immutable_inputs["original_model"].sha256[:16]
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_spec_sha256: str | None = None,
    ) -> ProvisionerSpec:
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = source.resolve()
        raw = _load_canonical_object(source, "autonomy provisioner specification")
        _exact_keys(raw, _SPEC_KEYS, "autonomy provisioner specification")
        if raw["schema_version"] != SCHEMA_VERSION or raw["contract"] != SPEC_CONTRACT:
            raise ProvisionerSpecError(
                "autonomy provisioner specification contract is unsupported"
            )
        body = dict(raw)
        identity = _required_sha256(
            body.pop("spec_sha256", None), "provisioner specification identity"
        )
        if identity != canonical_sha256(body):
            raise ProvisionerSpecError("provisioner specification self-hash is invalid")
        observed_file_hash = file_sha256(source)
        if expected_spec_sha256 is not None:
            expected = _required_sha256(
                expected_spec_sha256, "expected provisioner specification hash"
            )
            if expected not in {identity, observed_file_hash}:
                raise ProvisionerSpecError(
                    "provisioner specification hash is not expected"
                )

        repository = _required_directory(raw["repository"], "repository")
        python_root = _required_directory(
            str(repository / "python"), "repository Python root"
        )
        if python_root.parent != repository:
            raise ProvisionerSpecError("repository Python root is not canonical")
        revision = raw["source_revision"]
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ProvisionerSpecError(
                "source_revision must be a lowercase Git object ID"
            )
        run_root = _required_directory(raw["run_root"], "run_root")
        if _paths_overlap(repository, run_root):
            raise ProvisionerSpecError(
                "run_root must be outside the deployed source checkout"
            )
        state_root = _future_path(raw["state_root"], "state_root", directory=True)
        bootstrap_state = _future_path(
            raw["bootstrap_state_root"],
            "bootstrap_state_root",
            directory=True,
        )
        if not _strictly_within(state_root, run_root) or not _strictly_within(
            bootstrap_state, run_root
        ):
            raise ProvisionerSpecError(
                "provisioner and bootstrap state roots must be inside run_root"
            )
        if _paths_overlap(state_root, bootstrap_state):
            raise ProvisionerSpecError(
                "provisioner and bootstrap state roots must not overlap"
            )

        readiness = _exact_keys(
            raw["readiness"],
            {"curation_status", "suite_manifest"},
            "readiness",
        )
        curation = ReadinessBinding.load(
            readiness["curation_status"], "readiness.curation_status"
        )
        suite = ReadinessBinding.load(
            readiness["suite_manifest"], "readiness.suite_manifest"
        )
        if (
            curation.path == suite.path
            or _paths_overlap(curation.path, state_root)
            or _paths_overlap(curation.path, bootstrap_state)
            or _paths_overlap(suite.path, state_root)
            or _paths_overlap(suite.path, bootstrap_state)
        ):
            raise ProvisionerSpecError(
                "readiness paths are duplicated or overlap mutable state"
            )
        candidate_inbox = _required_directory(raw["candidate_inbox"], "candidate_inbox")
        if _paths_overlap(candidate_inbox, state_root) or _paths_overlap(
            candidate_inbox, bootstrap_state
        ):
            raise ProvisionerSpecError(
                "candidate inbox must not overlap provisioner state"
            )
        activation = _required_directory(
            raw["activation_destination"], "activation_destination"
        )
        if (
            _paths_overlap(repository, activation)
            or _paths_overlap(state_root, activation)
            or _paths_overlap(bootstrap_state, activation)
        ):
            raise ProvisionerSpecError(
                "activation_destination must be outside source and state roots"
            )
        legacy = raw["legacy_path_unit"]
        if (
            not isinstance(legacy, str)
            or _UNIT_RE.fullmatch(legacy) is None
            or not legacy.endswith(".path")
            or legacy
            in {
                PREPARE_PATH_UNIT_NAME,
                PREPARE_SERVICE_UNIT_NAME,
                BOOTSTRAP_PATH_UNIT_NAME,
                BOOTSTRAP_SERVICE_UNIT_NAME,
            }
        ):
            raise ProvisionerSpecError("legacy_path_unit is invalid or duplicated")

        raw_inputs = _exact_keys(
            raw["immutable_inputs"], set(_REQUIRED_INPUTS), "immutable_inputs"
        )
        inputs = {
            name: FileBinding.load(value, f"immutable_inputs.{name}")
            for name, value in raw_inputs.items()
        }
        if not os.access(inputs["python_executable"].path, os.X_OK):
            raise ProvisionerSpecError("python_executable must be executable")
        if not os.access(inputs["katago_binary"].path, os.X_OK):
            raise ProvisionerSpecError("katago_binary must be executable")
        if (
            inputs["original_model"].path == inputs["suite_champion_model"].path
            or inputs["original_model"].sha256 == inputs["suite_champion_model"].sha256
        ):
            raise ProvisionerSpecError(
                "suite champion model must differ from the immutable original"
            )

        raw_models = raw["models"]
        if not isinstance(raw_models, list) or not raw_models:
            raise ProvisionerSpecError("models must be a nonempty array")
        models = tuple(
            FileBinding.load(value, f"models[{index}]")
            for index, value in enumerate(raw_models)
        )
        model_paths = [str(binding.path) for binding in models]
        if model_paths != sorted(model_paths) or len(set(model_paths)) != len(
            model_paths
        ):
            raise ProvisionerSpecError("models must have unique paths sorted by path")
        if len({binding.sha256 for binding in models}) != len(models):
            raise ProvisionerSpecError("models must have unique content hashes")
        required_model_paths = {
            inputs["original_model"].path,
            inputs["suite_champion_model"].path,
        }
        if not required_model_paths.issubset({binding.path for binding in models}):
            raise ProvisionerSpecError(
                "models must include the original and suite champion"
            )

        raw_extra = raw["extra_inputs"]
        if not isinstance(raw_extra, list):
            raise ProvisionerSpecError("extra_inputs must be an array")
        extra = tuple(
            FileBinding.load(value, f"extra_inputs[{index}]")
            for index, value in enumerate(raw_extra)
        )
        extra_paths = [str(binding.path) for binding in extra]
        if extra_paths != sorted(extra_paths) or len(set(extra_paths)) != len(
            extra_paths
        ):
            raise ProvisionerSpecError(
                "extra_inputs must have unique paths sorted by path"
            )

        all_fixed = {
            binding.path: binding.sha256
            for binding in [*inputs.values(), *models, *extra]
        }
        for binding in [*inputs.values(), *models, *extra]:
            if all_fixed[binding.path] != binding.sha256:
                raise ProvisionerSpecError(
                    f"one immutable path has conflicting hashes: {binding.path}"
                )
        if curation.path in all_fixed or suite.path in all_fixed:
            raise ProvisionerSpecError(
                "future readiness paths may not be fixed immutable inputs"
            )
        forbidden_fixed = [
            str(path)
            for path in all_fixed
            if _paths_overlap(path, state_root)
            or _paths_overlap(path, bootstrap_state)
            or _paths_overlap(path, candidate_inbox)
        ]
        if forbidden_fixed:
            raise ProvisionerSpecError(
                f"immutable inputs overlap mutable provisioner state/inbox: "
                f"{sorted(forbidden_fixed)}"
            )

        command_values = _exact_keys(
            raw["runtime_commands"], set(_RUNTIME_COMMANDS), "runtime_commands"
        )
        commands = {
            name: _validate_argv(command_values[name], f"runtime_commands.{name}")
            for name in _RUNTIME_COMMANDS
        }
        for name, argv in commands.items():
            _validate_bound_argv_paths(argv, all_fixed, f"runtime_commands.{name}")

        publisher_config = _validate_publisher_config(
            raw["publisher_config"],
            run_root=run_root,
            bootstrap_state_root=bootstrap_state,
            gpu_index=raw.get("gpu", {}).get("index"),
            fixed_inputs=all_fixed,
        )
        provisioned_roots = [
            Path(publisher_config["cluster_executor"]["state_directory"]),
            Path(publisher_config["adaptive_training"]["root"]),
            Path(publisher_config["suite_rotation"]["registry_root"]),
            Path(publisher_config["suite_rotation"]["root"]),
            Path(publisher_config["promotion_drill"]["disposable_root"]),
        ]
        if any(_paths_overlap(state_root, path) for path in provisioned_roots):
            raise ProvisionerSpecError(
                "provisioner state overlaps generated service state"
            )
        gpu = _exact_keys(raw["gpu"], {"index", "uuid"}, "gpu")
        gpu_index = _required_integer(gpu["index"], "gpu.index")
        gpu_uuid = gpu["uuid"]
        if (
            gpu_index != 7
            or not isinstance(gpu_uuid, str)
            or _GPU_UUID_RE.fullmatch(gpu_uuid) is None
        ):
            raise ProvisionerSpecError(
                "production autonomy requires GPU index 7 and a verified GPU UUID"
            )

        output_values = _exact_keys(raw["outputs"], _OUTPUT_KEYS, "outputs")
        outputs = ProvisionerOutputs(
            prepare_service_unit=_future_path(
                output_values["prepare_service_unit"],
                "outputs.prepare_service_unit",
            ),
            prepare_path_unit=_future_path(
                output_values["prepare_path_unit"], "outputs.prepare_path_unit"
            ),
            status=_future_path(output_values["status"], "outputs.status"),
            receipts_root=_future_path(
                output_values["receipts_root"],
                "outputs.receipts_root",
                directory=True,
            ),
            artifacts_root=_future_path(
                output_values["artifacts_root"],
                "outputs.artifacts_root",
                directory=True,
            ),
            bootstrap_spec=_future_path(
                output_values["bootstrap_spec"], "outputs.bootstrap_spec"
            ),
            bootstrap_service_unit=_future_path(
                output_values["bootstrap_service_unit"],
                "outputs.bootstrap_service_unit",
            ),
            bootstrap_path_unit=_future_path(
                output_values["bootstrap_path_unit"],
                "outputs.bootstrap_path_unit",
            ),
        )
        expected_units = {
            outputs.prepare_service_unit: PREPARE_SERVICE_UNIT_NAME,
            outputs.prepare_path_unit: PREPARE_PATH_UNIT_NAME,
            outputs.bootstrap_service_unit: BOOTSTRAP_SERVICE_UNIT_NAME,
            outputs.bootstrap_path_unit: BOOTSTRAP_PATH_UNIT_NAME,
        }
        for output, expected_name in expected_units.items():
            if output.parent != activation or output.name != expected_name:
                raise ProvisionerSpecError(
                    f"unit output must be {activation / expected_name}"
                )
        if (
            not _strictly_within(outputs.status, state_root)
            or not _strictly_within(outputs.receipts_root, state_root)
            or not _strictly_within(outputs.artifacts_root, state_root)
            or not _strictly_within(outputs.bootstrap_spec, bootstrap_state)
        ):
            raise ProvisionerSpecError(
                "status/receipt/artifact/bootstrap outputs escaped their state roots"
            )
        output_paths = {
            outputs.prepare_service_unit,
            outputs.prepare_path_unit,
            outputs.status,
            outputs.receipts_root,
            outputs.artifacts_root,
            outputs.bootstrap_spec,
            outputs.bootstrap_service_unit,
            outputs.bootstrap_path_unit,
        }
        if len(output_paths) != len(_OUTPUT_KEYS):
            raise ProvisionerSpecError("provisioner outputs contain duplicates")
        state_outputs = [
            outputs.status,
            outputs.receipts_root,
            outputs.artifacts_root,
        ]
        for index, first in enumerate(state_outputs):
            for second in state_outputs[index + 1 :]:
                if _paths_overlap(first, second):
                    raise ProvisionerSpecError(
                        f"provisioner state outputs overlap: {first} and {second}"
                    )
        if _strictly_within(source, state_root) or _strictly_within(
            source, bootstrap_state
        ):
            raise ProvisionerSpecError(
                "provisioner specification may not be stored in mutable state"
            )

        service_user = raw["service_user"]
        if (
            not isinstance(service_user, str)
            or _SERVICE_USER_RE.fullmatch(service_user) is None
        ):
            raise ProvisionerSpecError("service_user is not a valid system user")
        return cls(
            path=source,
            file_sha256=observed_file_hash,
            identity=identity,
            repository=repository,
            source_revision=revision,
            run_root=run_root,
            state_root=state_root,
            bootstrap_state_root=bootstrap_state,
            curation_status=curation,
            suite_manifest=suite,
            candidate_inbox=candidate_inbox,
            activation_destination=activation,
            legacy_path_unit=legacy,
            immutable_inputs=MappingProxyType(inputs),
            models=models,
            extra_inputs=extra,
            runtime_commands=MappingProxyType(commands),
            publisher_config=MappingProxyType(publisher_config),
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
            actor=_required_id(raw["actor"], "actor"),
            service_user=service_user,
            initial_generation_id=_required_id(
                raw["initial_generation_id"], "initial_generation_id"
            ),
            created_at_utc=_parse_created_at(raw["created_at_utc"]),
            poll_interval_seconds=_positive_number(
                raw["poll_interval_seconds"], "poll_interval_seconds"
            ),
            minimum_clean_observations=_required_integer(
                raw["minimum_clean_observations"],
                "minimum_clean_observations",
                minimum=2,
            ),
            outputs=outputs,
            raw=MappingProxyType(raw),
        )

    def verify_immutable_inputs(self) -> None:
        for name, binding in self.immutable_inputs.items():
            binding.verify(f"immutable input {name}")
        for index, binding in enumerate(self.models):
            binding.verify(f"model {index}")
        for index, binding in enumerate(self.extra_inputs):
            binding.verify(f"extra input {index}")
        if self.path.is_symlink() or file_sha256(self.path) != self.file_sha256:
            raise ProvisionerDriftError("provisioner specification file changed")
        for readiness in (self.curation_status, self.suite_manifest):
            if readiness.sha256 is not None and (
                readiness.path.is_symlink()
                or not readiness.path.is_file()
                or file_sha256(readiness.path) != readiness.sha256
            ):
                raise ProvisionerDriftError(
                    f"hash-bound readiness input changed: {readiness.path}"
                )


def _validate_publisher_config(
    value: Any,
    *,
    run_root: Path,
    bootstrap_state_root: Path,
    gpu_index: Any,
    fixed_inputs: Mapping[Path, str],
) -> dict[str, Any]:
    root = dict(_exact_keys(value, _PUBLISHER_CONFIG_KEYS, "publisher_config"))
    scheduler = _required_directory(
        root["scheduler_directory"], "publisher_config.scheduler_directory"
    )
    cluster = dict(
        _exact_keys(
            root["cluster_executor"], _CLUSTER_KEYS, "publisher_config.cluster_executor"
        )
    )
    cluster["state_directory"] = str(
        _future_path(
            cluster["state_directory"],
            "publisher_config.cluster_executor.state_directory",
            directory=True,
        )
    )
    cluster["owner_id"] = _required_id(
        cluster["owner_id"], "publisher_config.cluster_executor.owner_id"
    )
    gpu_ids = cluster["gpu_ids"]
    if (
        not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(item, str) or not item for item in gpu_ids)
        or gpu_ids != sorted(set(gpu_ids))
        or "7" not in gpu_ids
    ):
        raise ProvisionerSpecError(
            "cluster executor GPU inventory must be sorted, unique, and contain 7"
        )
    for key in (
        "poll_interval_seconds",
        "heartbeat_interval_seconds",
        "stale_after_seconds",
        "backoff_initial_seconds",
        "backoff_max_seconds",
        "lease_proof_timeout_seconds",
    ):
        _positive_number(cluster[key], f"publisher_config.cluster_executor.{key}")
    _required_integer(
        cluster["retry_budget"],
        "publisher_config.cluster_executor.retry_budget",
    )
    lease_command = _validate_argv(
        cluster["lease_proof_command"],
        "publisher_config.cluster_executor.lease_proof_command",
        allow_none=True,
    )
    guardian = _validate_argv(
        cluster["guardian_argv_prefix"],
        "publisher_config.cluster_executor.guardian_argv_prefix",
    )
    for role, argv in (("lease proof command", lease_command), ("guardian", guardian)):
        if not argv:
            continue
        _validate_bound_argv_paths(argv, fixed_inputs, role)

    adaptive = dict(
        _exact_keys(
            root["adaptive_training"],
            _ADAPTIVE_KEYS,
            "publisher_config.adaptive_training",
        )
    )
    adaptive["root"] = str(
        _future_path(
            adaptive["root"],
            "publisher_config.adaptive_training.root",
            directory=True,
        )
    )
    adaptive["observation_path"] = str(
        _future_path(
            adaptive["observation_path"],
            "publisher_config.adaptive_training.observation_path",
        )
    )
    adaptive_trial_argv = _validate_argv(
        adaptive["trial_command_argv_template"],
        "publisher_config.adaptive_training.trial_command_argv_template",
    )
    _validate_bound_argv_paths(
        adaptive_trial_argv,
        fixed_inputs,
        "publisher_config.adaptive_training.trial_command_argv_template",
    )
    _positive_number(
        adaptive["poll_interval_seconds"],
        "publisher_config.adaptive_training.poll_interval_seconds",
    )

    suite = dict(
        _exact_keys(
            root["suite_rotation"], _SUITE_KEYS, "publisher_config.suite_rotation"
        )
    )
    for key in ("registry_root", "root"):
        suite[key] = str(
            _future_path(
                suite[key],
                f"publisher_config.suite_rotation.{key}",
                directory=True,
            )
        )
    for key in (
        "materializer_argv_template",
        "curation_argv_template",
        "continuity_argv_template",
    ):
        template = _validate_argv(suite[key], f"publisher_config.suite_rotation.{key}")
        _validate_bound_argv_paths(
            template,
            fixed_inputs,
            f"publisher_config.suite_rotation.{key}",
        )
    if not isinstance(suite["results"], Mapping) or not suite["results"]:
        raise ProvisionerSpecError("suite rotation results must be a nonempty object")
    _ensure_finite(suite["results"], "publisher_config.suite_rotation.results")
    _positive_number(
        suite["poll_interval_seconds"],
        "publisher_config.suite_rotation.poll_interval_seconds",
    )

    topology = dict(
        _exact_keys(
            root["topology_benchmark"],
            _TOPOLOGY_KEYS,
            "publisher_config.topology_benchmark",
        )
    )
    if not isinstance(topology["publisher_options"], Mapping):
        raise ProvisionerSpecError("topology publisher_options must be an object")
    _ensure_finite(topology["publisher_options"], "topology publisher_options")
    _validate_bound_option_paths(
        topology["publisher_options"],
        fixed_inputs,
        "topology publisher_options",
    )
    _positive_number(
        topology["timeout_seconds"],
        "publisher_config.topology_benchmark.timeout_seconds",
    )
    if float(topology["timeout_seconds"]) <= _TOPOLOGY_STARTUP_RESERVE_SECONDS:
        raise ProvisionerSpecError(
            "topology outer timeout must exceed its startup reserve"
        )

    lease = dict(
        _exact_keys(root["lease_drill"], _LEASE_KEYS, "publisher_config.lease_drill")
    )
    if not isinstance(lease["publisher_options"], Mapping):
        raise ProvisionerSpecError("lease publisher_options must be an object")
    _ensure_finite(lease["publisher_options"], "lease publisher_options")
    _validate_bound_option_paths(
        lease["publisher_options"],
        fixed_inputs,
        "lease publisher_options",
    )
    _positive_number(
        lease["probe_timeout_seconds"],
        "publisher_config.lease_drill.probe_timeout_seconds",
    )
    _lease_timeout_budget(float(lease["probe_timeout_seconds"]))
    _required_integer(
        lease["probe_minimum_completed_work"],
        "publisher_config.lease_drill.probe_minimum_completed_work",
        minimum=1,
    )

    promotion = dict(
        _exact_keys(
            root["promotion_drill"],
            _PROMOTION_KEYS,
            "publisher_config.promotion_drill",
        )
    )
    promotion["disposable_root"] = str(
        _future_path(
            promotion["disposable_root"],
            "publisher_config.promotion_drill.disposable_root",
            directory=True,
        )
    )
    for key, minimum in (
        ("command_timeout_seconds", 1),
        ("max_evaluator_rows", 1),
        ("max_worker_games", 1),
        ("max_replay_attempts", 8),
    ):
        _required_integer(
            promotion[key], f"publisher_config.promotion_drill.{key}", minimum=minimum
        )

    shadow = dict(
        _exact_keys(
            root["shadow_runtime"], _SHADOW_KEYS, "publisher_config.shadow_runtime"
        )
    )
    selected = _required_integer(
        shadow["evaluator_process_count"],
        "publisher_config.shadow_runtime.evaluator_process_count",
        minimum=1,
    )
    if selected not in autonomy_bootstrap.TOPOLOGY_CHOICES:
        raise ProvisionerSpecError("shadow evaluator process count must be 4, 8, or 16")
    if gpu_index != 7:
        raise ProvisionerSpecError("publisher configuration is pinned to GPU 7")
    mutable_paths = [
        Path(cluster["state_directory"]),
        Path(adaptive["root"]),
        Path(adaptive["observation_path"]),
        Path(suite["registry_root"]),
        Path(suite["root"]),
        Path(promotion["disposable_root"]),
    ]
    if any(not _strictly_within(path, run_root) for path in mutable_paths):
        raise ProvisionerSpecError(
            "publisher mutable control-plane paths must be inside run_root"
        )
    if any(_paths_overlap(path, bootstrap_state_root) for path in mutable_paths):
        raise ProvisionerSpecError(
            "generated service state must not overlap bootstrap state"
        )
    if _paths_overlap(scheduler, bootstrap_state_root):
        raise ProvisionerSpecError("scheduler directory overlaps bootstrap state")
    major_roots = [
        Path(cluster["state_directory"]),
        Path(adaptive["root"]),
        Path(suite["registry_root"]),
        Path(suite["root"]),
        Path(promotion["disposable_root"]),
    ]
    for index, first in enumerate(major_roots):
        for second in major_roots[index + 1 :]:
            if _paths_overlap(first, second):
                raise ProvisionerSpecError(
                    f"publisher mutable roots overlap: {first} and {second}"
                )
    for path in major_roots:
        if _paths_overlap(path, scheduler):
            raise ProvisionerSpecError(
                f"publisher mutable root overlaps scheduler: {path}"
            )
    return {
        "scheduler_directory": str(scheduler),
        "cluster_executor": cluster,
        "adaptive_training": adaptive,
        "suite_rotation": suite,
        "topology_benchmark": topology,
        "lease_drill": lease,
        "promotion_drill": promotion,
        "shadow_runtime": shadow,
    }


def _validate_bound_option_paths(
    value: Any,
    fixed_inputs: Mapping[Path, str],
    role: str,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_bound_option_paths(child, fixed_inputs, f"{role}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_bound_option_paths(child, fixed_inputs, f"{role}[{index}]")
        return
    if not isinstance(value, str):
        return
    candidate = Path(value)
    if not candidate.is_absolute() or not candidate.exists():
        return
    path = _absolute_path(value, role)
    if path.is_file() and path not in fixed_inputs:
        raise ProvisionerSpecError(f"{role} names an input file that is not hash-bound")


def load_provisioner_spec(
    path: Path, *, expected_spec_sha256: str | None = None
) -> ProvisionerSpec:
    return ProvisionerSpec.load(path, expected_spec_sha256=expected_spec_sha256)


load_spec = load_provisioner_spec


def _binding_value(path: Path) -> Mapping[str, str]:
    source = _required_file(str(Path(path)), "binding source")
    return {"path": str(source), "sha256": file_sha256(source)}


def _coerce_binding_value(value: Any, role: str) -> Mapping[str, str]:
    if isinstance(value, (str, os.PathLike)):
        return _binding_value(Path(value).resolve())
    binding = FileBinding.load(value, role)
    return dict(binding.to_dict())


def publish_provisioner_spec(
    path: Path,
    *,
    repository: Path,
    source_revision: str,
    run_root: Path,
    state_root: Path,
    bootstrap_state_root: Path,
    curation_status: Path,
    suite_manifest: Path,
    candidate_inbox: Path,
    activation_destination: Path,
    immutable_inputs: Mapping[str, Any],
    models: Sequence[Any],
    runtime_commands: Mapping[str, Sequence[str]],
    publisher_config: Mapping[str, Any],
    gpu_index: int,
    gpu_uuid: str,
    actor: str,
    service_user: str,
    initial_generation_id: str,
    created_at_utc: str,
    extra_inputs: Sequence[Any] = (),
    legacy_path_unit: str = LEGACY_PATH_UNIT_NAME,
    poll_interval_seconds: float = 5.0,
    minimum_clean_observations: int = 3,
    outputs: Mapping[str, Path] | None = None,
) -> ProvisionerSpec:
    """Publish one strict self-hashed host-facing provisioner specification."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = destination.resolve()
    activation = Path(activation_destination).resolve()
    provisioner_state = Path(state_root).resolve()
    bootstrap_state = Path(bootstrap_state_root).resolve()
    output_values = {
        "prepare_service_unit": activation / PREPARE_SERVICE_UNIT_NAME,
        "prepare_path_unit": activation / PREPARE_PATH_UNIT_NAME,
        "status": provisioner_state / "status.json",
        "receipts_root": provisioner_state / "receipts",
        "artifacts_root": provisioner_state / "artifacts",
        "bootstrap_spec": bootstrap_state / "bootstrap-spec.json",
        "bootstrap_service_unit": activation / BOOTSTRAP_SERVICE_UNIT_NAME,
        "bootstrap_path_unit": activation / BOOTSTRAP_PATH_UNIT_NAME,
    }
    if outputs is not None:
        if set(outputs) != _OUTPUT_KEYS:
            raise ProvisionerSpecError(
                "publication outputs fields differ from contract"
            )
        output_values = {name: Path(outputs[name]).resolve() for name in _OUTPUT_KEYS}

    status_path = Path(curation_status).resolve()
    manifest_path = Path(suite_manifest).resolve()
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SPEC_CONTRACT,
        "repository": str(Path(repository).resolve()),
        "source_revision": source_revision,
        "run_root": str(Path(run_root).resolve()),
        "state_root": str(provisioner_state),
        "bootstrap_state_root": str(bootstrap_state),
        "readiness": {
            "curation_status": {
                "path": str(status_path),
                "sha256": None,
            },
            "suite_manifest": {
                "path": str(manifest_path),
                "sha256": None,
            },
        },
        "candidate_inbox": str(Path(candidate_inbox).resolve()),
        "activation_destination": str(activation),
        "legacy_path_unit": legacy_path_unit,
        "immutable_inputs": {
            name: _coerce_binding_value(immutable_inputs[name], f"input {name}")
            for name in sorted(immutable_inputs)
        },
        "models": sorted(
            (_coerce_binding_value(item, "model") for item in models),
            key=lambda item: item["path"],
        ),
        "extra_inputs": sorted(
            (_coerce_binding_value(item, "extra input") for item in extra_inputs),
            key=lambda item: item["path"],
        ),
        "runtime_commands": {
            name: list(runtime_commands[name]) for name in _RUNTIME_COMMANDS
        },
        "publisher_config": json.loads(canonical_json(publisher_config)),
        "gpu": {"index": gpu_index, "uuid": gpu_uuid},
        "actor": actor,
        "service_user": service_user,
        "initial_generation_id": initial_generation_id,
        "created_at_utc": created_at_utc,
        "poll_interval_seconds": poll_interval_seconds,
        "minimum_clean_observations": minimum_clean_observations,
        "outputs": {name: str(output_values[name]) for name in sorted(output_values)},
    }
    value["spec_sha256"] = canonical_sha256(value)
    _publish_immutable_json(destination, value, "provisioner specification")
    return ProvisionerSpec.load(destination)


publish_spec = publish_provisioner_spec


def _run_git(repository: Path, command_runner: CommandRunner, *arguments: str) -> str:
    argv = ["git", "-C", str(repository), *arguments]
    completed = command_runner(
        argv,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ProvisionerDriftError(f"Git {' '.join(arguments)} failed: {message}")
    return completed.stdout.strip()


def _verify_clean_checkout(
    spec: ProvisionerSpec, command_runner: CommandRunner
) -> None:
    revision = _run_git(spec.repository, command_runner, "rev-parse", "HEAD")
    status = _run_git(
        spec.repository,
        command_runner,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if revision != spec.source_revision or status:
        raise ProvisionerDriftError(
            "source revision must match a clean deployed checkout"
        )


def _verify_deployment(spec: ProvisionerSpec) -> Mapping[str, Any]:
    from risk_score.build_live_runtime import verify_deployment_manifest

    manifest = verify_deployment_manifest(
        spec.immutable_inputs["deployment_manifest"].path
    )
    expected_source_hash = hashlib.sha256(
        spec.source_revision.encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("source_revision") != spec.source_revision
        or manifest.get("source_sha256") != expected_source_hash
    ):
        raise ProvisionerDriftError(
            "deployment manifest is not bound to the provisioner source revision"
        )
    return manifest


def _verify_environment(
    spec: ProvisionerSpec, command_runner: CommandRunner
) -> Mapping[str, Any]:
    spec.verify_immutable_inputs()
    _verify_clean_checkout(spec, command_runner)
    manifest = _verify_deployment(spec)
    spec.verify_immutable_inputs()
    _verify_clean_checkout(spec, command_runner)
    return manifest


def _systemd_quote(value: str) -> str:
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def render_prepare_service_unit(spec: ProvisionerSpec | Path) -> str:
    loaded = spec if isinstance(spec, ProvisionerSpec) else ProvisionerSpec.load(spec)
    python = loaded.immutable_inputs["python_executable"].path
    argv = [
        str(python),
        "-m",
        "risk_score.autonomy_provisioner",
        "materialize",
        "--spec",
        str(loaded.path),
        "--expected-spec-sha256",
        loaded.file_sha256,
        "--apply",
    ]
    unit = "\n".join(
        [
            "[Unit]",
            "Description=Prepare KataGo revision-bound autonomy bootstrap",
            "Wants=network-online.target",
            "After=network-online.target",
            "RequiresMountsFor=" + _systemd_quote(str(loaded.run_root)),
            "# Retry indefinitely: suite and final curation status may arrive",
            "# in either order and path events are not a bounded retry budget.",
            "StartLimitIntervalSec=0",
            "",
            "[Service]",
            "Type=oneshot",
            "WorkingDirectory=" + _systemd_quote(str(loaded.repository / "python")),
            "Environment="
            + _systemd_quote(f"PYTHONPATH={loaded.repository / 'python'}"),
            "ExecStart=" + " ".join(_systemd_quote(part) for part in argv),
            "Restart=on-failure",
            "RestartSec=30",
            "TimeoutStartSec=0",
            "TimeoutStopSec=300",
            "UMask=0077",
            "",
        ]
    )
    if (
        "PartOf=katago-risk-training.target" in unit
        or "WantedBy=katago-risk-training.target" in unit
        or "systemctl" in unit
    ):
        raise ProvisionerSpecError(
            "prepare service must remain independent and non-activating"
        )
    return unit


def render_prepare_path_unit(spec: ProvisionerSpec | Path) -> str:
    loaded = spec if isinstance(spec, ProvisionerSpec) else ProvisionerSpec.load(spec)
    unit = "\n".join(
        [
            "[Unit]",
            "Description=Prepare KataGo autonomy after authoritative suite publication",
            "# Existence wakes the service; the provisioner validates authority.",
            "",
            "[Path]",
            "PathChanged=" + _systemd_quote(str(loaded.suite_manifest.path.parent)),
            "PathChanged=" + _systemd_quote(str(loaded.curation_status.path.parent)),
            "PathExists=" + _systemd_quote(str(loaded.suite_manifest.path)),
            "PathExists=" + _systemd_quote(str(loaded.curation_status.path)),
            f"Unit={PREPARE_SERVICE_UNIT_NAME}",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )
    if "katago-risk-training.target" in unit or BOOTSTRAP_SERVICE_UNIT_NAME in unit:
        raise ProvisionerSpecError("prepare path unit is not independent")
    return unit


render_systemd_service_unit = render_prepare_service_unit
render_systemd_path_unit = render_prepare_path_unit


def _prepare_actions(spec: ProvisionerSpec) -> list[list[str]]:
    return [
        ["systemctl", "disable", "--now", spec.legacy_path_unit],
        ["systemctl", "disable", "--now", PREPARE_PATH_UNIT_NAME],
        ["systemctl", "disable", "--now", PREPARE_SERVICE_UNIT_NAME],
        ["systemctl", "disable", "--now", BOOTSTRAP_PATH_UNIT_NAME],
        ["systemctl", "disable", "--now", BOOTSTRAP_SERVICE_UNIT_NAME],
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", "--now", PREPARE_PATH_UNIT_NAME],
    ]


def _prepare_rollback_actions(spec: ProvisionerSpec) -> list[list[str]]:
    return [
        ["systemctl", "disable", "--now", PREPARE_PATH_UNIT_NAME],
        ["systemctl", "enable", "--now", spec.legacy_path_unit],
        ["systemctl", "daemon-reload"],
    ]


def _bootstrap_actions(spec: ProvisionerSpec) -> list[list[str]]:
    return [
        ["systemctl", "disable", "--now", spec.legacy_path_unit],
        ["systemctl", "disable", "--now", PREPARE_PATH_UNIT_NAME],
        ["systemctl", "disable", "--now", PREPARE_SERVICE_UNIT_NAME],
        ["systemctl", "disable", "--now", BOOTSTRAP_PATH_UNIT_NAME],
        ["systemctl", "disable", "--now", BOOTSTRAP_SERVICE_UNIT_NAME],
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", "--now", BOOTSTRAP_PATH_UNIT_NAME],
    ]


def _bootstrap_rollback_actions(_spec: ProvisionerSpec) -> list[list[str]]:
    return [
        ["systemctl", "disable", "--now", BOOTSTRAP_PATH_UNIT_NAME],
        ["systemctl", "enable", "--now", PREPARE_PATH_UNIT_NAME],
        ["systemctl", "daemon-reload"],
    ]


def _assert_path_only_actions(actions: Sequence[Sequence[str]]) -> None:
    for argv in actions:
        if "enable" not in argv:
            continue
        enabled = argv[-1]
        if not enabled.endswith(".path") or enabled == BOOTSTRAP_SERVICE_UNIT_NAME:
            raise ProvisionerSpecError(
                "installation plan attempted to enable a service directly"
            )


def plan_provisioning(
    spec_path: Path | ProvisionerSpec,
    *,
    expected_spec_sha256: str | None = None,
    command_runner: CommandRunner = subprocess.run,
) -> Mapping[str, Any]:
    spec = (
        spec_path
        if isinstance(spec_path, ProvisionerSpec)
        else ProvisionerSpec.load(spec_path, expected_spec_sha256=expected_spec_sha256)
    )
    _verify_environment(spec, command_runner)
    service_data = render_prepare_service_unit(spec).encode("utf-8")
    path_data = render_prepare_path_unit(spec).encode("utf-8")
    _check_immutable_destination(
        spec.outputs.prepare_service_unit, service_data, "prepare service unit"
    )
    _check_immutable_destination(
        spec.outputs.prepare_path_unit, path_data, "prepare path unit"
    )
    prepare_actions = _prepare_actions(spec)
    bootstrap_actions = _bootstrap_actions(spec)
    _assert_path_only_actions([*prepare_actions, *bootstrap_actions])
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": PLAN_CONTRACT,
        "spec": _spec_binding(spec),
        "prepare_units": [
            {
                "name": PREPARE_SERVICE_UNIT_NAME,
                "path": str(spec.outputs.prepare_service_unit),
                "sha256": hashlib.sha256(service_data).hexdigest(),
            },
            {
                "name": PREPARE_PATH_UNIT_NAME,
                "path": str(spec.outputs.prepare_path_unit),
                "sha256": hashlib.sha256(path_data).hexdigest(),
            },
        ],
        "legacy_to_prepare": {
            "actions": prepare_actions,
            "rollback": _prepare_rollback_actions(spec),
        },
        "arm_bootstrap": {
            "actions": bootstrap_actions,
            "rollback": _bootstrap_rollback_actions(spec),
        },
        "constraints": {
            "mutation_activated": False,
            "bootstrap_service_enabled": False,
            "maximum_active_queue": 3,
            "topology_choices": [4, 8, 16],
        },
    }
    value["plan_sha256"] = canonical_sha256(value)
    return value


plan = plan_provisioning


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    target = Path(path)
    _reject_symlink_ancestors(target, "publication directory")
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ProvisionerConflictError(f"unsafe publication directory: {target}")
        return
    if target.parent == target:
        raise ProvisionerConflictError(f"cannot create publication root: {target}")
    _ensure_directory(target.parent)
    try:
        target.mkdir()
    except FileExistsError:
        if target.is_symlink() or not target.is_dir():
            raise ProvisionerConflictError(f"unsafe publication directory: {target}")
    _fsync_directory(target.parent)


def _check_immutable_destination(path: Path, data: bytes, role: str) -> None:
    destination = Path(path)
    _reject_symlink_ancestors(destination, role)
    if os.path.lexists(os.fspath(destination)) and (
        destination.is_symlink()
        or not destination.is_file()
        or destination.read_bytes() != data
    ):
        raise ProvisionerConflictError(
            f"{role} conflicts with existing artifact: {destination}"
        )


def _publish_immutable(
    path: Path, data: bytes, role: str, *, mode: int = 0o444
) -> Mapping[str, str]:
    destination = Path(path)
    _check_immutable_destination(destination, data, role)
    if destination.exists():
        return {"path": str(destination), "sha256": file_sha256(destination)}
    _ensure_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=os.fspath(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(os.fspath(temporary), os.fspath(destination))
        except FileExistsError:
            _check_immutable_destination(destination, data, role)
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return {"path": str(destination), "sha256": file_sha256(destination)}


def _publish_immutable_json(
    path: Path, value: Mapping[str, Any], role: str
) -> Mapping[str, str]:
    return _publish_immutable(
        path,
        (canonical_json(dict(value)) + "\n").encode("utf-8"),
        role,
    )


def _atomic_replace_json(path: Path, value: Mapping[str, Any], role: str) -> None:
    destination = Path(path)
    _reject_symlink_ancestors(destination, role)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ProvisionerConflictError(f"unsafe {role}: {destination}")
    _ensure_directory(destination.parent)
    data = (canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=os.fspath(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(os.fspath(temporary), os.fspath(destination))
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _spec_binding(spec: ProvisionerSpec) -> Mapping[str, str]:
    return {
        "path": str(spec.path),
        "sha256": spec.file_sha256,
        "identity": spec.identity,
    }


def _self_hashed_value(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(value)
    result[hash_field] = canonical_sha256(result)
    return result


def _load_self_hashed(
    path: Path,
    *,
    role: str,
    contract: str,
    hash_field: str,
) -> Mapping[str, Any]:
    value = _load_canonical_object(path, role)
    payload = dict(value)
    supplied = payload.pop(hash_field, None)
    if value.get("contract") != contract or supplied != canonical_sha256(payload):
        raise ProvisionerDriftError(f"{role} contract or self-hash is invalid")
    return value


def _status_value(
    spec: ProvisionerSpec,
    *,
    state: str,
    plan_value: Mapping[str, Any] | None,
    readiness: Mapping[str, Any] | None,
    preparation_receipt: Mapping[str, Any] | None,
    materialization_receipt: Mapping[str, Any] | None,
    installation_receipts: Sequence[Mapping[str, Any]] = (),
    error: Exception | None = None,
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": STATUS_CONTRACT,
        "spec": _spec_binding(spec),
        "state": state,
        "plan_sha256": (None if plan_value is None else plan_value.get("plan_sha256")),
        "readiness": None if readiness is None else dict(readiness),
        "preparation_receipt": (
            None
            if preparation_receipt is None
            else _receipt_reference(preparation_receipt)
        ),
        "materialization_receipt": (
            None
            if materialization_receipt is None
            else _receipt_reference(materialization_receipt)
        ),
        "installation_receipts": [
            _receipt_reference(receipt) for receipt in installation_receipts
        ],
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    value["status_sha256"] = canonical_sha256(value)
    return value


def _receipt_reference(value: Mapping[str, Any]) -> Mapping[str, Any]:
    path = value.get("_path")
    clean = {key: child for key, child in value.items() if key != "_path"}
    identity = next(
        (
            clean[key]
            for key in (
                "receipt_sha256",
                "publication_sha256",
                "installation_sha256",
            )
            if isinstance(clean.get(key), str)
        ),
        canonical_sha256(clean),
    )
    return {
        "path": path,
        "sha256": (
            file_sha256(Path(path))
            if isinstance(path, str) and Path(path).is_file()
            else None
        ),
        "identity": identity,
    }


def _publish_receipt(
    path: Path,
    value: Mapping[str, Any],
    *,
    contract: str,
    hash_field: str = "receipt_sha256",
) -> Mapping[str, Any]:
    payload = dict(value)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("contract", contract)
    payload.pop(hash_field, None)
    payload[hash_field] = canonical_sha256(payload)
    _publish_immutable_json(path, payload, contract)
    loaded = _load_self_hashed(
        path,
        role=contract,
        contract=contract,
        hash_field=hash_field,
    )
    return {**loaded, "_path": str(path)}


def _load_receipt(
    path: Path,
    *,
    contract: str,
    hash_field: str = "receipt_sha256",
) -> Mapping[str, Any] | None:
    if not os.path.lexists(os.fspath(path)):
        return None
    value = _load_self_hashed(
        path,
        role=contract,
        contract=contract,
        hash_field=hash_field,
    )
    return {**value, "_path": str(path)}


def _preparation_receipt_path(spec: ProvisionerSpec) -> Path:
    return spec.outputs.receipts_root / "preparation.json"


def _readiness_receipt_path(spec: ProvisionerSpec) -> Path:
    return spec.outputs.receipts_root / "readiness.json"


def _materialization_receipt_path(spec: ProvisionerSpec) -> Path:
    return spec.outputs.receipts_root / "materialization.json"


def _installation_receipt_path(spec: ProvisionerSpec, phase: str) -> Path:
    return spec.outputs.receipts_root / "installation" / f"{phase}.json"


def _publish_preparation(
    spec: ProvisionerSpec, plan_value: Mapping[str, Any]
) -> Mapping[str, Any]:
    service = _publish_immutable(
        spec.outputs.prepare_service_unit,
        render_prepare_service_unit(spec).encode("utf-8"),
        "prepare service unit",
        mode=0o644,
    )
    path_unit = _publish_immutable(
        spec.outputs.prepare_path_unit,
        render_prepare_path_unit(spec).encode("utf-8"),
        "prepare path unit",
        mode=0o644,
    )
    return _publish_receipt(
        _preparation_receipt_path(spec),
        {
            "spec": _spec_binding(spec),
            "plan_sha256": plan_value["plan_sha256"],
            "units": {
                PREPARE_SERVICE_UNIT_NAME: service,
                PREPARE_PATH_UNIT_NAME: path_unit,
            },
            "enable_unit": PREPARE_PATH_UNIT_NAME,
            "bootstrap_service_enabled": False,
            "mutation_activated": False,
        },
        contract=PREPARATION_RECEIPT_CONTRACT,
    )


def _run_systemctl(
    command_runner: CommandRunner, argv: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    completed = command_runner(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if not hasattr(completed, "returncode"):
        raise ProvisionerApplyError("systemctl runner returned no process result")
    if completed.returncode != 0:
        stderr = str(getattr(completed, "stderr", "")).strip()
        stdout = str(getattr(completed, "stdout", "")).strip()
        raise ProvisionerApplyError(
            f"systemctl command failed ({' '.join(argv)}): {stderr or stdout}"
        )
    return completed


def _unit_state(command_runner: CommandRunner, unit: str) -> Mapping[str, Any]:
    states: dict[str, str] = {}
    for verb, key in (("is-enabled", "enabled"), ("is-active", "active")):
        completed = command_runner(
            ["systemctl", verb, unit],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if not hasattr(completed, "returncode"):
            raise ProvisionerApplyError(f"systemctl {verb} returned no result")
        output = str(getattr(completed, "stdout", "")).strip()
        if verb == "is-enabled":
            states[key] = output or (
                "enabled" if completed.returncode == 0 else "disabled"
            )
        else:
            states[key] = output or (
                "active" if completed.returncode == 0 else "inactive"
            )
    return states


def _unit_is_enabled(state: Mapping[str, Any]) -> bool:
    return state.get("enabled") in {"enabled", "enabled-runtime", "linked"}


def _unit_is_active(state: Mapping[str, Any]) -> bool:
    return state.get("active") in {"active", "activating", "reloading"}


def _restore_unit_states(
    command_runner: CommandRunner,
    initial_state: Mapping[str, Mapping[str, Any]],
) -> tuple[list[list[str]], list[str]]:
    completed: list[list[str]] = []
    errors: list[str] = []

    def run(argv: list[str]) -> None:
        try:
            _run_systemctl(command_runner, argv)
            completed.append(argv)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))

    for unit in sorted(initial_state):
        desired = initial_state[unit]
        try:
            current = _unit_state(command_runner, unit)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if _unit_is_active(current) and not _unit_is_active(desired):
            run(["systemctl", "stop", unit])
        try:
            current = _unit_state(command_runner, unit)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if _unit_is_enabled(current) and not _unit_is_enabled(desired):
            run(["systemctl", "disable", unit])

    for unit in sorted(initial_state):
        desired = initial_state[unit]
        try:
            current = _unit_state(command_runner, unit)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if _unit_is_enabled(desired) and not _unit_is_enabled(current):
            run(["systemctl", "enable", unit])
        try:
            current = _unit_state(command_runner, unit)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if _unit_is_active(desired) and not _unit_is_active(current):
            run(["systemctl", "start", unit])

    for unit, desired in sorted(initial_state.items()):
        try:
            observed = _unit_state(command_runner, unit)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if _unit_is_enabled(observed) != _unit_is_enabled(desired) or _unit_is_active(
            observed
        ) != _unit_is_active(desired):
            errors.append(
                f"unit state rollback did not restore {unit}: "
                f"expected={dict(desired)} observed={dict(observed)}"
            )
    return completed, errors


def _validate_real_apply(spec: ProvisionerSpec, runner: CommandRunner) -> None:
    if runner is not subprocess.run:
        return
    if os.geteuid() != 0:
        raise ProvisionerApplyError("real systemd apply requires root")
    if spec.activation_destination != Path("/etc/systemd/system"):
        raise ProvisionerApplyError(
            "real systemd apply requires activation_destination=/etc/systemd/system"
        )
    activation = spec.activation_destination
    if activation.is_symlink() or not activation.is_dir():
        raise ProvisionerApplyError("systemd activation destination is unsafe")
    activation_stat = activation.stat(follow_symlinks=False)
    if activation_stat.st_uid != 0 or stat.S_IMODE(activation_stat.st_mode) != 0o755:
        raise ProvisionerApplyError(
            "systemd activation destination must be root-owned mode 0755"
        )
    for unit_path in (
        spec.outputs.prepare_service_unit,
        spec.outputs.prepare_path_unit,
        spec.outputs.bootstrap_service_unit,
        spec.outputs.bootstrap_path_unit,
    ):
        if not unit_path.exists():
            continue
        unit_stat = unit_path.stat(follow_symlinks=False)
        if (
            unit_path.is_symlink()
            or not unit_path.is_file()
            or unit_stat.st_uid != 0
            or stat.S_IMODE(unit_stat.st_mode) != 0o644
        ):
            raise ProvisionerApplyError(
                f"systemd unit must be root-owned mode 0644: {unit_path}"
            )


def _open_anchored_directory(path: Path, *, create: bool) -> int:
    target = Path(path)
    if not target.is_absolute():
        raise ProvisionerApplyError(f"mutable directory is not absolute: {target}")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for part in target.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o750, dir_fd=descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ProvisionerApplyError(
            f"cannot open service directory safely: {target}: {exc}"
        ) from exc


def _secure_mutable_directory(path: Path, *, uid: int, gid: int) -> None:
    descriptor = _open_anchored_directory(path, create=True)
    try:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o750)
    except OSError as exc:
        raise ProvisionerApplyError(
            f"cannot prepare mutable service directory safely: {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _secure_mutable_file(
    path: Path, *, uid: int, gid: int, prepare_parent: bool = True
) -> None:
    target = Path(path)
    if prepare_parent:
        _secure_mutable_directory(target.parent, uid=uid, gid=gid)
    file_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    parent_descriptor = _open_anchored_directory(
        target.parent,
        create=prepare_parent,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            target.name,
            file_flags,
            mode=0o600,
            dir_fd=parent_descriptor,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProvisionerApplyError(
                f"mutable service file is not regular: {target}"
            )
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o600)
    except OSError as exc:
        raise ProvisionerApplyError(
            f"cannot prepare mutable service file safely: {target}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _make_root_owned_tree_readable(path: Path) -> None:
    root = Path(path)
    if not root.exists():
        return
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_stat = current_path.stat(follow_symlinks=False)
        if current_path.is_symlink() or current_stat.st_uid != 0:
            raise ProvisionerApplyError(
                f"immutable artifact directory is not root-owned and safe: {current_path}"
            )
        os.chmod(current_path, 0o755, follow_symlinks=False)
        for name in directories:
            if (current_path / name).is_symlink():
                raise ProvisionerApplyError(
                    f"immutable artifact directory is symlinked: {current_path / name}"
                )
        for name in files:
            artifact = current_path / name
            artifact_stat = artifact.stat(follow_symlinks=False)
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or artifact_stat.st_uid != 0
            ):
                raise ProvisionerApplyError(
                    f"immutable artifact is not root-owned and safe: {artifact}"
                )
            os.chmod(artifact, 0o644, follow_symlinks=False)


def _prepare_root_owned_publication_roots(spec: ProvisionerSpec) -> None:
    for root in (spec.state_root, spec.bootstrap_state_root):
        _ensure_directory(root)
        _make_root_owned_tree_readable(root)


def _prepare_service_ownership(spec: ProvisionerSpec) -> None:
    try:
        account = pwd.getpwnam(spec.service_user)
    except KeyError as exc:
        raise ProvisionerApplyError(
            f"service user does not exist: {spec.service_user}"
        ) from exc
    directory_paths = {
        Path(spec.publisher_config["scheduler_directory"]),
        Path(spec.publisher_config["cluster_executor"]["state_directory"]),
        Path(spec.publisher_config["adaptive_training"]["root"]),
        Path(spec.publisher_config["suite_rotation"]["registry_root"]),
        Path(spec.publisher_config["suite_rotation"]["root"]),
        Path(spec.publisher_config["promotion_drill"]["disposable_root"]),
        spec.run_root / "promotion" / "events",
    }
    promotion_root = spec.run_root / "promotion"
    if promotion_root.is_symlink() or not promotion_root.is_dir():
        raise ProvisionerApplyError("promotion control root is unsafe")
    for directory in sorted(directory_paths, key=str):
        _secure_mutable_directory(
            directory,
            uid=account.pw_uid,
            gid=account.pw_gid,
        )
    drill_checkpoint = (
        Path(spec.publisher_config["promotion_drill"]["disposable_root"])
        / "trainer"
        / "trainer-checkpoint.bin"
    )
    if drill_checkpoint.exists():
        _secure_mutable_file(
            drill_checkpoint,
            uid=account.pw_uid,
            gid=account.pw_gid,
        )
    controller_lock = promotion_root / "controller.lock"
    _secure_mutable_file(
        controller_lock,
        uid=account.pw_uid,
        gid=account.pw_gid,
        prepare_parent=False,
    )
    if os.geteuid() == 0:
        for immutable_root in (
            spec.state_root,
            spec.bootstrap_state_root,
            spec.outputs.artifacts_root,
        ):
            _make_root_owned_tree_readable(immutable_root)


def _apply_actions_with_rollback(
    spec: ProvisionerSpec,
    *,
    phase: str,
    actions: Sequence[Sequence[str]],
    rollback: Sequence[Sequence[str]],
    command_runner: CommandRunner,
) -> Mapping[str, Any]:
    _assert_path_only_actions(actions)
    _validate_real_apply(spec, command_runner)
    units = sorted(
        {
            argv[-1]
            for argv in actions
            if argv and argv[-1].endswith((".path", ".service"))
        }
    )
    initial_state = {unit: dict(_unit_state(command_runner, unit)) for unit in units}
    completed_actions: list[list[str]] = []
    rollback_completed: list[list[str]] = []
    try:
        for argv in actions:
            unit = argv[-1] if argv else ""
            if "disable" in argv:
                current = _unit_state(command_runner, unit)
                if not _unit_is_enabled(current) and not _unit_is_active(current):
                    continue
            elif "enable" in argv:
                current = _unit_state(command_runner, unit)
                if _unit_is_enabled(current) and _unit_is_active(current):
                    continue
            _run_systemctl(command_runner, argv)
            completed_actions.append(list(argv))
            if "disable" in argv:
                observed = _unit_state(command_runner, unit)
                if _unit_is_enabled(observed) or _unit_is_active(observed):
                    raise ProvisionerApplyError(
                        f"unit remained enabled or active after disable: {unit}"
                    )
            elif "enable" in argv:
                observed = _unit_state(command_runner, unit)
                if not _unit_is_enabled(observed) or not _unit_is_active(observed):
                    raise ProvisionerApplyError(
                        f"unit did not become enabled and active: {unit}"
                    )
        receipt_path = _installation_receipt_path(spec, phase)
        if receipt_path.exists():
            repair_identity = canonical_sha256(
                {
                    "phase": phase,
                    "initial_unit_state": initial_state,
                    "completed_actions": completed_actions,
                }
            )
            receipt_path = receipt_path.with_name(
                f"{phase}.repair-{repair_identity}.json"
            )
        return _publish_receipt(
            receipt_path,
            {
                "spec": _spec_binding(spec),
                "phase": phase,
                "actions": [list(argv) for argv in actions],
                "completed_actions": completed_actions,
                "rollback_actions": [list(argv) for argv in rollback],
                "rollback_completed": rollback_completed,
                "initial_unit_state": initial_state,
                "final_unit_state": {
                    unit: dict(_unit_state(command_runner, unit)) for unit in units
                },
                "bootstrap_service_enabled": False,
                "mutation_activated": False,
            },
            contract=INSTALLATION_RECEIPT_CONTRACT,
            hash_field="installation_sha256",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        rollback_completed, rollback_errors = _restore_unit_states(
            command_runner, initial_state
        )
        for argv in (["systemctl", "daemon-reload"],):
            try:
                _run_systemctl(command_runner, argv)
                rollback_completed.append(list(argv))
            except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        failure = ProvisionerApplyError(
            f"{phase} failed after {len(completed_actions)} actions; "
            f"rollback_errors={rollback_errors}: {exc}"
        )
        with contextlib.suppress(Exception):
            _publish_failure(spec, phase=phase, error=failure)
        raise failure from exc


def apply_legacy_cutover(
    spec: ProvisionerSpec | Path,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> Mapping[str, Any]:
    loaded = spec if isinstance(spec, ProvisionerSpec) else ProvisionerSpec.load(spec)
    existing = _load_receipt(
        _installation_receipt_path(loaded, "legacy-to-prepare"),
        contract=INSTALLATION_RECEIPT_CONTRACT,
        hash_field="installation_sha256",
    )
    if existing is not None:
        prepare = _unit_state(command_runner, PREPARE_PATH_UNIT_NAME)
        others = [
            loaded.legacy_path_unit,
            PREPARE_SERVICE_UNIT_NAME,
            BOOTSTRAP_PATH_UNIT_NAME,
            BOOTSTRAP_SERVICE_UNIT_NAME,
        ]
        if (
            _unit_is_enabled(prepare)
            and _unit_is_active(prepare)
            and all(
                not _unit_is_enabled(_unit_state(command_runner, unit))
                and not _unit_is_active(_unit_state(command_runner, unit))
                for unit in others
            )
        ):
            return existing
    return _apply_actions_with_rollback(
        loaded,
        phase="legacy-to-prepare",
        actions=_prepare_actions(loaded),
        rollback=_prepare_rollback_actions(loaded),
        command_runner=command_runner,
    )


def apply_bootstrap_arm(
    spec: ProvisionerSpec | Path,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> Mapping[str, Any]:
    loaded = spec if isinstance(spec, ProvisionerSpec) else ProvisionerSpec.load(spec)
    if (
        loaded.outputs.bootstrap_service_unit.is_symlink()
        or not loaded.outputs.bootstrap_service_unit.is_file()
        or loaded.outputs.bootstrap_path_unit.is_symlink()
        or not loaded.outputs.bootstrap_path_unit.is_file()
    ):
        raise ProvisionerApplyError(
            "bootstrap units must be materialized before arming the path"
        )
    existing = _load_receipt(
        _installation_receipt_path(loaded, "arm-bootstrap-path"),
        contract=INSTALLATION_RECEIPT_CONTRACT,
        hash_field="installation_sha256",
    )
    if existing is not None:
        bootstrap = _unit_state(command_runner, BOOTSTRAP_PATH_UNIT_NAME)
        others = [
            loaded.legacy_path_unit,
            PREPARE_PATH_UNIT_NAME,
            PREPARE_SERVICE_UNIT_NAME,
            BOOTSTRAP_SERVICE_UNIT_NAME,
        ]
        if (
            _unit_is_enabled(bootstrap)
            and _unit_is_active(bootstrap)
            and all(
                not _unit_is_enabled(_unit_state(command_runner, unit))
                and not _unit_is_active(_unit_state(command_runner, unit))
                for unit in others
            )
        ):
            return existing
    return _apply_actions_with_rollback(
        loaded,
        phase="arm-bootstrap-path",
        actions=_bootstrap_actions(loaded),
        rollback=_bootstrap_rollback_actions(loaded),
        command_runner=command_runner,
    )


apply_cutover = apply_legacy_cutover


def _publish_failure(
    spec: ProvisionerSpec, *, phase: str, error: Exception
) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": FAILURE_RECEIPT_CONTRACT,
        "spec": _spec_binding(spec),
        "phase": phase,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    value["failure_sha256"] = canonical_sha256(value)
    path = spec.outputs.receipts_root / "failures" / f"{value['failure_sha256']}.json"
    _publish_immutable_json(path, value, "provisioner failure receipt")
    return {**value, "_path": str(path)}


def _readiness_observation(spec: ProvisionerSpec) -> Mapping[str, Any]:
    from risk_score.autonomy_gate_runners import _status_readiness, _suite_readiness

    status_present = os.path.lexists(os.fspath(spec.curation_status.path))
    suite_present = os.path.lexists(os.fspath(spec.suite_manifest.path))
    if status_present and (
        spec.curation_status.path.is_symlink()
        or not spec.curation_status.path.is_file()
    ):
        raise ProvisionerDriftError("curation status path is unsafe")
    if suite_present and (
        spec.suite_manifest.path.is_symlink() or not spec.suite_manifest.path.is_file()
    ):
        raise ProvisionerDriftError("suite manifest path is unsafe")
    status_complete = (
        _status_readiness(spec.curation_status.path, spec.suite_manifest.path)
        if status_present
        else False
    )
    suite_complete = (
        _suite_readiness(spec.suite_manifest.path) if suite_present else False
    )
    return {
        "curation_status": {
            "path": str(spec.curation_status.path),
            "present": status_present,
            "sha256": (
                file_sha256(spec.curation_status.path) if status_present else None
            ),
            "complete": status_complete,
        },
        "suite_manifest": {
            "path": str(spec.suite_manifest.path),
            "present": suite_present,
            "sha256": (
                file_sha256(spec.suite_manifest.path) if suite_present else None
            ),
            "complete": suite_complete,
        },
        "decision": ("READY" if status_complete and suite_complete else "WAIT"),
    }


def _validate_inventory(value: Any, spec: ProvisionerSpec) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvisionerDriftError("candidate inventory must be an object")
    payload = dict(value)
    supplied = payload.pop("inventory_sha256", None)
    candidates = value.get("candidates")
    ignored = value.get("ignored")
    if (
        value.get("schema_version") != 1
        or value.get("contract") != "risk-score-live-candidate-inventory-v1"
        or value.get("inbox") != str(spec.candidate_inbox.resolve())
        or supplied != canonical_sha256(payload)
        or not isinstance(candidates, list)
        or value.get("candidate_count") != len(candidates)
        or not isinstance(ignored, list)
        or any(not isinstance(item, str) for item in ignored)
    ):
        raise ProvisionerDriftError("candidate inventory is malformed")
    return value


def _candidate_total(value: Mapping[str, Any]) -> int:
    return int(value["candidate_count"]) + len(value["ignored"])


def _candidate_inventory(
    spec: ProvisionerSpec,
    *,
    output: Path | None,
    adapter: Callable[..., Mapping[str, Any]],
) -> Mapping[str, Any]:
    if output is not None:
        _ensure_directory(output.parent)
        result = adapter(inbox=spec.candidate_inbox, output=output)
        if not output.is_file() or output.is_symlink():
            raise ProvisionerDependencyError(
                "candidate inventory adapter did not publish its output"
            )
        loaded = _load_canonical_object(output, "candidate inventory")
        if result != loaded:
            raise ProvisionerDependencyError(
                "candidate inventory result differs from its artifact"
            )
        return _validate_inventory(loaded, spec)
    _ensure_directory(spec.state_root)
    with tempfile.TemporaryDirectory(
        prefix=".candidate-observation.", dir=os.fspath(spec.state_root)
    ) as temporary:
        path = Path(temporary) / "inventory.json"
        result = adapter(inbox=spec.candidate_inbox, output=path)
        loaded = _load_canonical_object(path, "candidate inventory")
        if result != loaded:
            raise ProvisionerDependencyError(
                "candidate inventory result differs from its artifact"
            )
        return _validate_inventory(loaded, spec)


@dataclass(frozen=True)
class GenerationPaths:
    root: Path
    inventory_before: Path
    inventory_after: Path
    inventory_final: Path
    active_curation_status: Path
    model_probe_config: Path
    cluster_executor_spec: Path
    registry_spec: Path
    adaptive_training_spec: Path
    suite_rotation_spec: Path
    topology_workload_spec: Path
    topology_benchmark_spec: Path
    lease_probe_spec: Path
    lease_drill_spec: Path
    promotion_drill_spec: Path
    shadow_runtime_root: Path
    lease_runtime: Path
    champion_receipt: Path
    backpressure: Path
    publisher_spec: Path


def _generation_paths(spec: ProvisionerSpec) -> GenerationPaths:
    root = spec.outputs.artifacts_root / spec.identity
    return GenerationPaths(
        root=root,
        inventory_before=root / "candidate-inventory-before.json",
        inventory_after=root / "candidate-inventory-after.json",
        inventory_final=root / "candidate-inventory-final.json",
        active_curation_status=root / "active-curation-status.json",
        model_probe_config=root / "model-probe.cfg",
        cluster_executor_spec=root / "cluster-executor-spec.json",
        registry_spec=root / "suite-registry-spec.json",
        adaptive_training_spec=root / "adaptive-training-spec.json",
        suite_rotation_spec=root / "suite-rotation-service-spec.json",
        topology_workload_spec=root / "evaluator-benchmark-workload-spec.json",
        topology_benchmark_spec=root / "evaluator-topology-benchmark-spec.json",
        lease_probe_spec=root / "autonomy-lease-probe-spec.json",
        lease_drill_spec=root / "autonomy-lease-drill-spec.json",
        promotion_drill_spec=root / "autonomy-promotion-drill-spec.json",
        shadow_runtime_root=root / "shadow-runtime",
        lease_runtime=root / "lease-drill-runtime.json",
        champion_receipt=root / "champion-projection-receipt.json",
        backpressure=root / "bootstrap-backpressure.json",
        publisher_spec=root / "bootstrap-publisher-spec.json",
    )


def _default_candidate_inventory(*, inbox: Path, output: Path) -> Mapping[str, Any]:
    from risk_score.promotion_preflight import candidate_inventory

    return candidate_inventory(inbox, output)


def _default_publish_model_probe_config(
    *,
    source: Path,
    destination: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.curate_position_bank import (
        validate_deterministic_analysis_config,
    )

    validate_deterministic_analysis_config(source)
    record = _publish_immutable(
        destination,
        source.read_bytes(),
        "model probe configuration",
    )
    validate_deterministic_analysis_config(destination)
    return record


def _default_publish_cluster_executor_spec(
    *,
    spec: ProvisionerSpec,
    destination: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.cluster_executor import (
        EXECUTOR_SPEC_CONTRACT,
        load_executor_spec,
    )

    config = spec.publisher_config["cluster_executor"]
    value: dict[str, Any] = {
        "schema_version": 1,
        "contract": EXECUTOR_SPEC_CONTRACT,
        "scheduler_directory": spec.publisher_config["scheduler_directory"],
        "state_directory": config["state_directory"],
        "owner_id": config["owner_id"],
        "gpu_ids": list(config["gpu_ids"]),
        "gpu7_id": "7",
        "poll_interval_seconds": config["poll_interval_seconds"],
        "heartbeat_interval_seconds": config["heartbeat_interval_seconds"],
        "stale_after_seconds": config["stale_after_seconds"],
        "retry_budget": config["retry_budget"],
        "backoff_initial_seconds": config["backoff_initial_seconds"],
        "backoff_max_seconds": config["backoff_max_seconds"],
        "lease_proof_command": config["lease_proof_command"],
        "lease_proof_timeout_seconds": config["lease_proof_timeout_seconds"],
        "gpu7_guardian_prefix": config["guardian_argv_prefix"],
    }
    value["spec_sha256"] = canonical_sha256(value)
    _publish_immutable_json(destination, value, "cluster executor specification")
    loaded = load_executor_spec(destination, expected_spec_sha256=value["spec_sha256"])
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "identity": loaded.spec_sha256,
    }


def _default_publish_registry_spec(
    *,
    spec: ProvisionerSpec,
    destination: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.suite_rotation import publish_registry_spec

    loaded = publish_registry_spec(
        destination,
        registry_root=Path(spec.publisher_config["suite_rotation"]["registry_root"]),
        policy_path=spec.immutable_inputs["promotion_policy"].path,
        original_model_path=spec.immutable_inputs["original_model"].path,
        initial_champion_path=spec.immutable_inputs["suite_champion_model"].path,
        initial_generation_id=spec.initial_generation_id,
        curation_champion_path=spec.immutable_inputs["suite_champion_model"].path,
        created_at_utc=spec.created_at_utc,
    )
    if loaded.created_at_utc != spec.created_at_utc:
        raise ProvisionerConflictError(
            "existing suite registry specification has a different creation time"
        )
    return {
        "path": str(loaded.path),
        "sha256": loaded.file_sha256,
        "identity": loaded.identity,
    }


def _default_bootstrap_registry(
    *,
    spec: ProvisionerSpec,
    registry_spec_path: Path,
    suite_manifest: Path,
    controller_transaction: Mapping[str, Any],
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.suite_rotation import SuiteRotationRegistry

    parsed = _datetime.datetime.fromisoformat(spec.created_at_utc[:-1] + "+00:00")
    expected_lock = spec.run_root / "promotion" / "controller.lock"
    if (
        controller_transaction.get("held") is not True
        or Path(str(controller_transaction.get("path", ""))) != expected_lock
    ):
        raise ProvisionerDependencyError(
            "suite registry bootstrap requires the held production ControllerLock"
        )
    registry = SuiteRotationRegistry(
        registry_spec_path,
        clock=lambda: parsed,
    )
    event = registry.bootstrap(suite_manifest)
    state = registry.reconstruct()
    if (
        state.active_suite_id is None
        or state.current_champion is None
        or state.current_champion.sha256
        != spec.immutable_inputs["suite_champion_model"].sha256
        or state.current_champion.generation_id != spec.initial_generation_id
        or registry.active_path.is_symlink()
        or not registry.active_path.is_file()
    ):
        raise ProvisionerDependencyError(
            "suite registry bootstrap did not publish the expected active suite"
        )
    active = _load_canonical_object(registry.active_path, "active suite projection")
    manifest_path = _require_generated_file(
        Path(str(active.get("manifest_path", ""))),
        "registry-owned active suite manifest",
    )
    if file_sha256(manifest_path) != file_sha256(suite_manifest):
        raise ProvisionerDependencyError(
            "registry-owned active suite differs from the authoritative suite"
        )
    return {
        "event_sha256": event.event_sha256,
        "active_suite": _binding_value(registry.active_path),
        "manifest": _binding_value(manifest_path),
        "suite_id": state.active_suite_id,
        "champion_sha256": state.current_champion.sha256,
        "generation_id": state.current_champion.generation_id,
    }


def _default_verify_target_inactive(
    *,
    command_runner: CommandRunner,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    argv = ["systemctl", "is-active", PRODUCTION_TARGET_UNIT_NAME]
    completed = command_runner(
        argv,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if not hasattr(completed, "returncode"):
        raise ProvisionerDependencyError(
            "production target inactivity check returned no process result"
        )
    state = str(getattr(completed, "stdout", "")).strip() or "unknown"
    if completed.returncode == 0 or state == "active":
        raise ProvisionerDependencyError(
            "production target must be inactive before control-plane bootstrap"
        )
    if completed.returncode not in {3, 4} or state not in {
        "inactive",
        "failed",
        "unknown",
        "not-found",
    }:
        detail = str(getattr(completed, "stderr", "")).strip()
        raise ProvisionerDependencyError(
            f"production target inactivity could not be proven: {state}: {detail}"
        )
    return {"unit": PRODUCTION_TARGET_UNIT_NAME, "state": state, "inactive": True}


@contextlib.contextmanager
def _default_controller_transaction(
    *,
    spec: ProvisionerSpec,
    **_kwargs: Any,
) -> Iterator[Any]:
    from risk_score.promotion_state import ControllerLock

    lock_parent = spec.run_root / "promotion"
    _ensure_directory(lock_parent)
    lock_path = lock_parent / "controller.lock"
    with ControllerLock(lock_path, owner=spec.actor):
        yield {"path": str(lock_path), "held": True}


def _default_publish_adaptive_spec(
    *,
    spec: ProvisionerSpec,
    destination: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.adaptive_training import publish_adaptive_service_spec

    config = spec.publisher_config["adaptive_training"]
    cluster = spec.publisher_config["cluster_executor"]
    loaded = publish_adaptive_service_spec(
        destination,
        root=Path(config["root"]),
        autonomy_policy_path=spec.immutable_inputs["autonomy_policy"].path,
        scheduler_directory=Path(spec.publisher_config["scheduler_directory"]),
        observation_path=Path(config["observation_path"]),
        trial_command_argv_template=config["trial_command_argv_template"],
        gpu_lease_guardian_argv_prefix=cluster["guardian_argv_prefix"],
        poll_interval_seconds=config["poll_interval_seconds"],
        actor=spec.actor,
        gpu7_id="7",
    )
    return {
        "path": str(loaded.path),
        "sha256": loaded.file_sha256,
        "identity": loaded.identity,
    }


def _default_publish_suite_service_spec(
    *,
    spec: ProvisionerSpec,
    destination: Path,
    registry_spec_path: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.suite_rotation_service import publish_service_spec

    config = spec.publisher_config["suite_rotation"]
    cluster = spec.publisher_config["cluster_executor"]
    loaded = publish_service_spec(
        destination,
        root=Path(config["root"]),
        registry_spec_path=registry_spec_path,
        scheduler_directory=Path(spec.publisher_config["scheduler_directory"]),
        gpu7_id="7",
        guardian_argv_prefix=cluster["guardian_argv_prefix"],
        materializer_argv_template=config["materializer_argv_template"],
        curation_argv_template=config["curation_argv_template"],
        continuity_argv_template=config["continuity_argv_template"],
        results=config["results"],
        poll_interval_seconds=config["poll_interval_seconds"],
        actor=spec.actor,
    )
    return {
        "path": str(loaded.path),
        "sha256": loaded.file_sha256,
        "identity": loaded.identity,
    }


def _shadow_runtime_complete(root: Path) -> bool:
    return all(
        (root / name).is_file() and not (root / name).is_symlink()
        for name in (
            "promotion-runtime.json",
            "gpu-lease-runtime.json",
            "deployment-manifest.json",
            "promotion-services.json",
        )
    )


def _default_build_shadow_runtime(
    *,
    spec: ProvisionerSpec,
    output_dir: Path,
    suite_manifest_path: Path | None = None,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.build_live_runtime import (
        build_live_runtime,
        verify_deployment_manifest,
    )

    if _shadow_runtime_complete(output_dir):
        deployment = verify_deployment_manifest(output_dir / "deployment-manifest.json")
        promotion = _load_canonical_object(
            output_dir / "promotion-runtime.json", "shadow promotion runtime"
        )
        gpu = _load_canonical_object(
            output_dir / "gpu-lease-runtime.json", "shadow GPU lease runtime"
        )
        if (
            promotion.get("mutationEnabled") is not False
            or gpu.get("mutationEnabled") is not False
            or deployment.get("source_revision") != spec.source_revision
        ):
            raise ProvisionerConflictError(
                "existing shadow runtime is not the requested immutable runtime"
            )
        return {
            "promotion_runtime": str(output_dir / "promotion-runtime.json"),
            "promotion_runtime_sha256": file_sha256(
                output_dir / "promotion-runtime.json"
            ),
            "gpu_lease_runtime": str(output_dir / "gpu-lease-runtime.json"),
            "gpu_lease_runtime_sha256": file_sha256(
                output_dir / "gpu-lease-runtime.json"
            ),
            "deployment_manifest": str(output_dir / "deployment-manifest.json"),
            "deployment_manifest_sha256": file_sha256(
                output_dir / "deployment-manifest.json"
            ),
            "service_spec": str(output_dir / "promotion-services.json"),
            "service_spec_sha256": file_sha256(output_dir / "promotion-services.json"),
            "systemd_units": {},
            "mutation_enabled": False,
            "full_autonomy": False,
            "evaluator_process_count": spec.publisher_config["shadow_runtime"][
                "evaluator_process_count"
            ],
        }
    return build_live_runtime(
        repo=spec.repository,
        run_root=spec.run_root,
        suite_dir=(
            spec.suite_manifest.path
            if suite_manifest_path is None
            else suite_manifest_path
        ).parent,
        katago_binary=spec.immutable_inputs["katago_binary"].path,
        python_executable=spec.immutable_inputs["python_executable"].path,
        trainer_spec=spec.immutable_inputs["trainer_spec"].path,
        consumer_spec=spec.immutable_inputs["consumer_spec"].path,
        original_model=spec.immutable_inputs["original_model"].path,
        trainer_checkpoint=spec.immutable_inputs["trainer_checkpoint"].path,
        gpu_uuid=spec.gpu_uuid,
        actor=spec.actor,
        source_revision=spec.source_revision,
        output_dir=output_dir,
        mutation_enabled=False,
        require_clean_source=True,
        service_user=None,
        evaluator_command=spec.runtime_commands["evaluator"],
        evaluator_process_count=spec.publisher_config["shadow_runtime"][
            "evaluator_process_count"
        ],
    )


def _default_initialize_champion(
    *,
    spec: ProvisionerSpec,
    promotion_runtime_path: Path,
    receipt_path: Path,
    target_observation: Mapping[str, Any],
    controller_transaction: Mapping[str, Any],
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.promotion_controller import RuntimeConfig
    from risk_score.promotion_state import (
        EventProvenance,
        EventRegistry,
        bootstrap_champion,
        load_champion,
    )

    runtime = RuntimeConfig.load(promotion_runtime_path)
    if runtime.controller.mutation_enabled:
        raise ProvisionerDependencyError(
            "champion projection may only be initialized from shadow runtime"
        )
    original_hash = spec.immutable_inputs["original_model"].sha256
    champion_hash = spec.immutable_inputs["suite_champion_model"].sha256
    if (
        runtime.controller.original_hash != original_hash
        or runtime.original_model_path != spec.immutable_inputs["original_model"].path
    ):
        raise ProvisionerDependencyError(
            "shadow runtime immutable original does not match the provisioner"
        )
    provenance = EventProvenance(
        controller_hash=runtime.controller.controller_hash,
        source_hash=runtime.controller.source_hash,
        original_hash=runtime.controller.original_hash,
        config_hash=runtime.controller.powered_config_hash,
        schedule_hash=runtime.controller.discovery_schedule_hash,
        policy_hash=runtime.controller.policy_hash,
    )
    if target_observation.get("inactive") is not True:
        raise ProvisionerDependencyError(
            "production target inactivity evidence is missing"
        )
    expected_lock_path = spec.run_root / "promotion" / "controller.lock"
    if (
        controller_transaction.get("held") is not True
        or Path(str(controller_transaction.get("path", ""))) != expected_lock_path
    ):
        raise ProvisionerDependencyError(
            "champion initialization requires the held production ControllerLock"
        )
    _ensure_directory(runtime.champion_path.parent)
    registry = EventRegistry(runtime.promotion_root)
    if runtime.lock_path != expected_lock_path:
        raise ProvisionerDependencyError(
            "promotion runtime lock differs from the held ControllerLock"
        )
    event_state = registry.reconstruct()
    if event_state.events and (
        event_state.current_champion_hash != champion_hash
        or not runtime.champion_path.is_file()
        or runtime.champion_path.is_symlink()
    ):
        raise ProvisionerConflictError(
            "existing promotion events do not have the suite champion"
        )
    champion = bootstrap_champion(
        runtime.champion_path,
        champion_hash=champion_hash,
        generation_id=spec.initial_generation_id,
        provenance=provenance,
        actor=spec.actor,
        timestamp_utc=spec.created_at_utc,
    )
    event = registry.bootstrap_champion(
        champion_hash=champion_hash,
        generation_id=spec.initial_generation_id,
        provenance=provenance,
        reason="provisioner initial suite champion bootstrap",
        actor=spec.actor,
        timestamp_utc=spec.created_at_utc,
    )
    loaded = load_champion(runtime.champion_path)
    if (
        loaded != champion
        or loaded.champion_hash != champion_hash
        or loaded.previous_champion_hash is not None
        or loaded.bootstrap is not True
    ):
        raise ProvisionerDependencyError(
            "initial suite champion projection is contradictory"
        )
    return _publish_receipt(
        receipt_path,
        {
            "spec": _spec_binding(spec),
            "operation": "initial-suite-champion-control-plane-bootstrap",
            "target_observation": dict(target_observation),
            "promotion_event": {
                "path": str(registry.events_dir / f"{event.sequence:020d}.json"),
                "event_sha256": event.event_hash,
                "sequence": event.sequence,
            },
            "champion": {
                "path": str(runtime.champion_path),
                "sha256": file_sha256(runtime.champion_path),
                "record_hash": loaded.record_hash,
                "champion_hash": loaded.champion_hash,
                "generation_id": loaded.generation_id,
            },
            "candidate_admission": False,
            "training_started": False,
            "export_started": False,
            "mutation_activated": False,
        },
        contract="risk-score-autonomy-original-champion-projection-v1",
    )


def _default_publish_promotion_drill_spec(
    *,
    spec: ProvisionerSpec,
    destination: Path,
    promotion_runtime_path: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.autonomy_promotion_drills import (
        load_drill_spec,
        publish_drill_spec,
    )

    config = spec.publisher_config["promotion_drill"]
    evidence_root = spec.bootstrap_state_root / "gate-evidence"
    _ensure_directory(evidence_root)
    if destination.exists():
        loaded = load_drill_spec(destination)
        if (
            loaded.production_runtime_path != promotion_runtime_path
            or loaded.evidence_root != evidence_root
            or loaded.disposable_root != Path(config["disposable_root"])
            or loaded.command_timeout_seconds != config["command_timeout_seconds"]
            or loaded.max_evaluator_rows != config["max_evaluator_rows"]
            or loaded.max_worker_games != config["max_worker_games"]
            or loaded.max_replay_attempts != config["max_replay_attempts"]
        ):
            raise ProvisionerConflictError(
                "existing promotion drill specification conflicts"
            )
        return {
            "path": str(destination),
            "sha256": file_sha256(destination),
            "identity": loaded.spec_sha256,
        }
    value = publish_drill_spec(
        destination,
        production_runtime_path=promotion_runtime_path,
        disposable_root=Path(config["disposable_root"]),
        evidence_root=evidence_root,
        command_timeout_seconds=config["command_timeout_seconds"],
        max_evaluator_rows=config["max_evaluator_rows"],
        max_worker_games=config["max_worker_games"],
        max_replay_attempts=config["max_replay_attempts"],
    )
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
        "identity": value["spec_sha256"],
    }


def _default_publish_topology_specs(
    *,
    spec: ProvisionerSpec,
    workload_spec_path: Path,
    benchmark_spec_path: Path,
    model_probe_config: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.evaluator_benchmark_workload import (
        benchmark_input_bindings,
        build_benchmark_argv_template,
        publish_workload_spec,
    )
    from risk_score.evaluator_topology_benchmark import load_benchmark_spec

    config = spec.publisher_config["topology_benchmark"]
    options = dict(config["publisher_options"])
    queries_value = options.get("queries", options.get("query_path"))
    nvidia_value = options.get("nvidia_smi")
    row_count = options.get("row_count", options.get("query_count"))
    max_visits = options.get("max_visits")
    if (
        queries_value is None
        or nvidia_value is None
        or row_count is None
        or max_visits is None
    ):
        raise ProvisionerDependencyError(
            "topology publisher_options require queries, nvidia_smi, "
            "row_count, and max_visits"
        )
    _ensure_directory(workload_spec_path.parent)
    outer_timeout = float(config["timeout_seconds"])
    maximum_inner_timeout = outer_timeout - _TOPOLOGY_STARTUP_RESERVE_SECONDS
    inner_timeout = float(
        options.get("workload_timeout_seconds", maximum_inner_timeout)
    )
    if inner_timeout <= 0 or inner_timeout > maximum_inner_timeout:
        raise ProvisionerDependencyError(
            "topology workload timeout must preserve the fixed outer startup reserve"
        )
    workload = publish_workload_spec(
        workload_spec_path,
        katago=spec.immutable_inputs["katago_binary"].path,
        analysis_config=model_probe_config,
        model=spec.immutable_inputs["original_model"].path,
        queries=Path(str(queries_value)),
        gpu_index=spec.gpu_index,
        expected_gpu_uuid=spec.gpu_uuid,
        max_visits=int(max_visits),
        timeout_seconds=inner_timeout,
        row_count=int(row_count),
        nvidia_smi=Path(str(nvidia_value)),
    )
    evidence = (
        spec.bootstrap_state_root
        / "gate-evidence"
        / "evaluator-topology-benchmark.json"
    )
    benchmark: dict[str, Any] = {
        "schema_version": 1,
        "contract": "risk-score-evaluator-topology-benchmark-spec-v1",
        "benchmark_argv_template": list(
            build_benchmark_argv_template(
                workload,
                python_executable=spec.immutable_inputs["python_executable"].path,
            )
        ),
        "inputs": [
            dict(binding)
            for binding in benchmark_input_bindings(
                workload,
                python_executable=spec.immutable_inputs["python_executable"].path,
            )
        ],
        "timeout_seconds": outer_timeout,
        "work_root": str(spec.bootstrap_state_root),
        "evidence_output": str(evidence),
    }
    benchmark["spec_sha256"] = canonical_sha256(benchmark)
    _publish_immutable_json(
        benchmark_spec_path,
        benchmark,
        "evaluator topology benchmark specification",
    )
    loaded = load_benchmark_spec(
        benchmark_spec_path,
        expected_spec_sha256=file_sha256(benchmark_spec_path),
    )
    return {
        "workload_spec": _binding_value(workload_spec_path),
        "benchmark_spec": _binding_value(benchmark_spec_path),
        "workload_identity": workload.identity,
        "benchmark_identity": loaded.identity,
    }


def _default_build_lease_runtime(
    *,
    spec: ProvisionerSpec,
    source_gpu_lease_config: Path,
    destination: Path,
    model_probe_config: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    """Build an uninstalled, mutation-enabled runtime used only by gate 7."""

    from risk_score.autonomy_lease_probe import build_probe_spec
    from risk_score.autonomy_lease_worker import (
        evaluator_sentinel_argv,
        lease_worker_module_path,
    )
    from risk_score.gpu_lease import RuntimeConfig

    config = spec.publisher_config["lease_drill"]
    options = dict(config["publisher_options"])
    query_value = options.get("query", options.get("query_path"))
    nvidia_value = options.get("nvidia_smi")
    if query_value is None or nvidia_value is None:
        raise ProvisionerDependencyError(
            "lease publisher_options require query and nvidia_smi"
        )
    worker_module = lease_worker_module_path()
    drill_root = Path(spec.publisher_config["promotion_drill"]["disposable_root"])
    control_root = drill_root / "lease-control-plane"
    (
        probe_inner_timeout,
        cleanup_margin_seconds,
        poll_interval_seconds,
    ) = _lease_timeout_budget(float(config["probe_timeout_seconds"]))
    build_probe_spec(
        nvidia_smi=Path(str(nvidia_value)),
        katago=spec.immutable_inputs["katago_binary"].path,
        model=spec.immutable_inputs["suite_champion_model"].path,
        config=model_probe_config,
        query=Path(str(query_value)),
        expected_gpu_uuid=spec.gpu_uuid,
        expected_gpu_index=spec.gpu_index,
        sentinel_readiness_path=control_root / "evaluator-sentinel-ready.json",
        expected_evaluator_process_count=1,
        timeout_seconds=probe_inner_timeout,
        cleanup_margin_seconds=cleanup_margin_seconds,
        poll_interval_seconds=poll_interval_seconds,
        minimum_completed_work=config["probe_minimum_completed_work"],
        sentinel_module=worker_module,
    )
    source = _load_canonical_object(
        source_gpu_lease_config, "shadow GPU lease runtime template"
    )
    value = json.loads(canonical_json(source))
    checkpoint = drill_root / "trainer" / "trainer-checkpoint.bin"
    _ensure_directory(control_root)
    _ensure_directory(checkpoint.parent)
    if not checkpoint.exists():
        checkpoint.write_bytes(
            spec.immutable_inputs["trainer_checkpoint"].path.read_bytes()
        )
        os.chmod(checkpoint, 0o600)
        _fsync_directory(checkpoint.parent)
    paths = value.get("paths")
    trainer = value.get("trainer")
    evaluator = value.get("evaluator")
    if not all(isinstance(item, dict) for item in (paths, trainer, evaluator)):
        raise ProvisionerDependencyError(
            "shadow GPU lease runtime template is malformed"
        )
    value["mutationEnabled"] = True
    value["ownerId"] = f"{spec.actor}-lease-drill"
    paths.update(
        {
            "runRoot": str(drill_root),
            "promotionRoot": str(control_root),
            "leaseState": str(control_root / "gpu-lease.json"),
            "eventLog": str(control_root / "gpu-lease-events.jsonl"),
        }
    )
    trainer["checkpointPath"] = str(checkpoint)
    evaluator["launchCommand"] = list(
        evaluator_sentinel_argv(
            python_executable=spec.immutable_inputs["python_executable"].path,
            expected_gpu_uuid=spec.gpu_uuid,
            expected_gpu_index=spec.gpu_index,
            ready_path=control_root / "evaluator-sentinel-ready.json",
            worker_module=worker_module,
        )
    )
    evaluator["processCount"] = 1
    _publish_immutable_json(destination, value, "lease drill GPU runtime")
    loaded = RuntimeConfig.from_json_file(destination)
    if (
        loaded.mutation_enabled is not True
        or loaded.evaluator_process_count != 1
        or loaded.evaluator_launch_command != tuple(evaluator["launchCommand"])
    ):
        raise ProvisionerDependencyError(
            "lease drill runtime is not a direct single-process CUDA sentinel"
        )
    return {
        **_binding_value(destination),
        "mutation_enabled": True,
        "evaluator_process_count": 1,
        "installed": False,
        "checkpoint_path": str(checkpoint),
        "worker_module": _binding_value(worker_module),
    }


def _default_publish_lease_specs(
    *,
    spec: ProvisionerSpec,
    probe_spec_path: Path,
    drill_spec_path: Path,
    gpu_lease_config: Path,
    model_probe_config: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.autonomy_lease_drill import load_drill_spec
    from risk_score.autonomy_lease_probe import (
        build_probe_spec,
        evaluator_probe_argv,
        evaluator_probe_outer_timeout_seconds,
        evaluator_sentinel_launch_argv,
        load_probe_spec,
        publish_probe_spec,
    )
    from risk_score.autonomy_lease_worker import lease_worker_module_path
    from risk_score.gpu_lease import RuntimeConfig

    config = spec.publisher_config["lease_drill"]
    options = dict(config["publisher_options"])
    query_value = options.get("query", options.get("query_path"))
    nvidia_value = options.get("nvidia_smi")
    if query_value is None or nvidia_value is None:
        raise ProvisionerDependencyError(
            "lease publisher_options require query and nvidia_smi"
        )
    worker_module = lease_worker_module_path()
    sentinel_readiness = (
        Path(spec.publisher_config["promotion_drill"]["disposable_root"])
        / "lease-control-plane"
        / "evaluator-sentinel-ready.json"
    )
    (
        probe_inner_timeout,
        cleanup_margin_seconds,
        poll_interval_seconds,
    ) = _lease_timeout_budget(float(config["probe_timeout_seconds"]))
    _ensure_directory(probe_spec_path.parent)
    desired_probe = build_probe_spec(
        nvidia_smi=Path(str(nvidia_value)),
        katago=spec.immutable_inputs["katago_binary"].path,
        model=spec.immutable_inputs["suite_champion_model"].path,
        config=model_probe_config,
        query=Path(str(query_value)),
        expected_gpu_uuid=spec.gpu_uuid,
        expected_gpu_index=spec.gpu_index,
        sentinel_readiness_path=sentinel_readiness,
        expected_evaluator_process_count=1,
        timeout_seconds=probe_inner_timeout,
        cleanup_margin_seconds=cleanup_margin_seconds,
        poll_interval_seconds=poll_interval_seconds,
        minimum_completed_work=config["probe_minimum_completed_work"],
        sentinel_module=worker_module,
    )
    if probe_spec_path.exists():
        probe = load_probe_spec(probe_spec_path)
        if dict(probe.raw) != desired_probe:
            raise ProvisionerConflictError(
                "existing lease probe specification conflicts"
            )
    else:
        probe = publish_probe_spec(
            probe_spec_path,
            nvidia_smi=Path(str(nvidia_value)),
            katago=spec.immutable_inputs["katago_binary"].path,
            model=spec.immutable_inputs["suite_champion_model"].path,
            config=model_probe_config,
            query=Path(str(query_value)),
            expected_gpu_uuid=spec.gpu_uuid,
            expected_gpu_index=spec.gpu_index,
            sentinel_readiness_path=sentinel_readiness,
            expected_evaluator_process_count=1,
            timeout_seconds=probe_inner_timeout,
            cleanup_margin_seconds=cleanup_margin_seconds,
            poll_interval_seconds=poll_interval_seconds,
            minimum_completed_work=config["probe_minimum_completed_work"],
            sentinel_module=worker_module,
        )
    runtime = RuntimeConfig.from_json_file(gpu_lease_config)
    expected_sentinel_argv = evaluator_sentinel_launch_argv(
        probe,
        python_executable=spec.immutable_inputs["python_executable"].path,
    )
    if (
        runtime.mutation_enabled is not True
        or runtime.evaluator_process_count != 1
        or runtime.evaluator_launch_command != expected_sentinel_argv
    ):
        raise ProvisionerDependencyError(
            "lease drill runtime does not launch the bound direct CUDA sentinel"
        )
    probe_argv = evaluator_probe_argv(
        probe,
        python_executable=spec.immutable_inputs["python_executable"].path,
    )
    probe_outer_timeout = evaluator_probe_outer_timeout_seconds(probe)
    if not math.isclose(
        probe_outer_timeout + cleanup_margin_seconds + poll_interval_seconds,
        float(config["probe_timeout_seconds"]),
    ):
        raise ProvisionerDependencyError(
            "lease probe publisher did not preserve the full outer drill reserve"
        )
    drill_value: dict[str, Any] = {
        "schema_version": 1,
        "contract": "risk-score-autonomy-lease-drill-spec-v1",
        "gpu_lease_config": _binding_value(gpu_lease_config),
        "trainer_source": {"kind": "launch"},
        "expected_gpu_uuid": spec.gpu_uuid,
        "minimum_clean_observations": spec.minimum_clean_observations,
        "evaluator_probe": {
            "argv": list(probe_argv),
            "timeout_seconds": float(config["probe_timeout_seconds"]),
            "executable_sha256": file_sha256(
                spec.immutable_inputs["python_executable"].path
            ),
            "model_sha256": spec.immutable_inputs["suite_champion_model"].sha256,
            "config_sha256": file_sha256(model_probe_config),
            "minimum_completed_work": config["probe_minimum_completed_work"],
        },
        "work_root": str(spec.bootstrap_state_root),
        "evidence_output": str(
            spec.bootstrap_state_root
            / "gate-evidence"
            / "trainer-evaluator-lease-drill.json"
        ),
    }
    drill_value["spec_sha256"] = canonical_sha256(drill_value)
    _publish_immutable_json(
        drill_spec_path,
        drill_value,
        "autonomy lease drill specification",
    )
    drill = load_drill_spec(
        drill_spec_path,
        expected_spec_sha256=file_sha256(drill_spec_path),
    )
    if drill.trainer_source_kind != "launch":
        raise ProvisionerDependencyError(
            "lease drill publisher did not use trainer_source kind launch"
        )
    return {
        "probe_spec": _binding_value(probe_spec_path),
        "drill_spec": _binding_value(drill_spec_path),
        "probe_identity": probe.identity,
        "drill_identity": drill.identity,
    }


def _default_publish_backpressure(
    *,
    spec: ProvisionerSpec,
    destination: Path,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from risk_score.adaptive_training import POLICY_HASH
    from risk_score.promotion_preflight import bootstrap_backpressure

    value = bootstrap_backpressure(destination, POLICY_HASH)
    if (
        value.get("allowExport") is not False
        or value.get("allowEvaluation") is not False
        or value.get("maximumActiveEvaluatorEntries") != 3
    ):
        raise ProvisionerDependencyError(
            "bootstrap backpressure is not fail-closed at queue depth 3"
        )
    return value


def _default_materialize_bootstrap(
    publisher_spec_path: Path,
    *,
    expected_spec_sha256: str,
    command_runner: CommandRunner,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    return autonomy_bootstrap_spec.materialize_bootstrap(
        publisher_spec_path,
        expected_spec_sha256=expected_spec_sha256,
        command_runner=command_runner,
    )


@dataclass(frozen=True)
class ProvisionerAdapters:
    candidate_inventory: Callable[..., Mapping[str, Any]] = _default_candidate_inventory
    publish_model_probe_config: Callable[..., Mapping[str, Any]] = (
        _default_publish_model_probe_config
    )
    publish_cluster_executor_spec: Callable[..., Mapping[str, Any]] = (
        _default_publish_cluster_executor_spec
    )
    publish_registry_spec: Callable[..., Mapping[str, Any]] = (
        _default_publish_registry_spec
    )
    verify_target_inactive: Callable[..., Mapping[str, Any]] = (
        _default_verify_target_inactive
    )
    controller_transaction: Callable[..., Any] = _default_controller_transaction
    bootstrap_registry: Callable[..., Mapping[str, Any]] = _default_bootstrap_registry
    publish_adaptive_spec: Callable[..., Mapping[str, Any]] = (
        _default_publish_adaptive_spec
    )
    publish_suite_service_spec: Callable[..., Mapping[str, Any]] = (
        _default_publish_suite_service_spec
    )
    build_shadow_runtime: Callable[..., Mapping[str, Any]] = (
        _default_build_shadow_runtime
    )
    initialize_champion: Callable[..., Mapping[str, Any]] = _default_initialize_champion
    publish_promotion_drill_spec: Callable[..., Mapping[str, Any]] = (
        _default_publish_promotion_drill_spec
    )
    publish_topology_specs: Callable[..., Mapping[str, Any]] = (
        _default_publish_topology_specs
    )
    build_lease_runtime: Callable[..., Mapping[str, Any]] = _default_build_lease_runtime
    publish_lease_specs: Callable[..., Mapping[str, Any]] = _default_publish_lease_specs
    publish_backpressure: Callable[..., Mapping[str, Any]] = (
        _default_publish_backpressure
    )
    materialize_bootstrap: Callable[..., Mapping[str, Any]] = (
        _default_materialize_bootstrap
    )


def _require_generated_file(path: Path, role: str) -> Path:
    _reject_symlink_ancestors(path, role)
    if path.is_symlink() or not path.is_file():
        raise ProvisionerDependencyError(f"{role} was not published: {path}")
    return path


def _validate_generated_spec(
    path: Path, *, role: str, contract: str
) -> Mapping[str, Any]:
    value = _load_canonical_object(path, role)
    payload = dict(value)
    supplied = payload.pop("spec_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("contract") != contract
        or supplied != canonical_sha256(payload)
    ):
        raise ProvisionerDependencyError(f"{role} contract or self-hash is invalid")
    return value


def _runtime_commands(
    spec: ProvisionerSpec, paths: GenerationPaths
) -> Mapping[str, Sequence[str]]:
    python = str(spec.immutable_inputs["python_executable"].path)
    return {
        "shuffler": list(spec.runtime_commands["shuffler"]),
        "exporter": list(spec.runtime_commands["exporter"]),
        "evaluator": list(spec.runtime_commands["evaluator"]),
        "cluster_executor": [
            python,
            "-m",
            "risk_score.cluster_executor",
            "--spec",
            str(paths.cluster_executor_spec),
            "watch",
        ],
        "adaptive_training": [
            python,
            "-m",
            "risk_score.adaptive_training",
            "--spec",
            str(paths.adaptive_training_spec),
            "watch",
        ],
        "suite_rotation": [
            python,
            "-m",
            "risk_score.suite_rotation_service",
            "--spec",
            str(paths.suite_rotation_spec),
            "watch",
        ],
    }


def _publish_active_curation_status(
    spec: ProvisionerSpec,
    *,
    active_manifest: Path,
    destination: Path,
) -> Mapping[str, str]:
    source = _load_canonical_object(
        spec.curation_status.path, "authoritative curation status"
    )
    projected = json.loads(canonical_json(source))
    artifacts = projected.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("suite"), dict):
        raise ProvisionerDependencyError(
            "authoritative curation status has no suite projection"
        )
    artifacts["suite"]["path"] = str(active_manifest.parent)
    projected.pop("status_sha256", None)
    projected["status_sha256"] = canonical_sha256(projected)
    _publish_immutable_json(
        destination,
        projected,
        "registry-owned active curation status",
    )
    return _binding_value(destination)


def _gate_templates(
    spec: ProvisionerSpec,
    paths: GenerationPaths,
    *,
    maximum_candidates: int,
) -> Mapping[str, Sequence[str]]:
    models = sorted(f"{binding.path}={binding.sha256}" for binding in spec.models)
    templates: dict[str, Sequence[str]] = {
        "curation-suite-readiness": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "curation-suite-readiness",
            "--curation-status",
            "{curation_status}",
            "--suite-manifest",
            "{suite_manifest}",
            "--suite-registry-spec",
            str(paths.registry_spec),
            "--output",
            "{evidence}",
        ],
        "filesystem-rename-fsync": [
            "{python}",
            "-m",
            "risk_score.promotion_preflight",
            "filesystem-test",
            "--root",
            "{run_root}",
            "--output",
            "{evidence}",
        ],
        "deployment-hash-validation": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "deployment-hash-validation",
            "--manifest",
            "{deployment_manifest}",
            "--output",
            "{evidence}",
        ],
        "candidate-inventory": [
            "{python}",
            "-m",
            "risk_score.promotion_preflight",
            "candidate-inventory",
            "--inbox",
            "{candidate_inbox}",
            "--output",
            "{evidence}",
        ],
        "cuda-model-probes": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "cuda-model-probes",
            "--katago",
            "{katago_binary}",
            "--config",
            "{model_probe_config}",
            "--gpu-index",
            str(spec.gpu_index),
            "--expected-gpu-uuid",
            spec.gpu_uuid,
            *[argument for model in models for argument in ("--model", model)],
            "--output",
            "{evidence}",
        ],
        "evaluator-topology-benchmark": [
            "{python}",
            "-m",
            "risk_score.evaluator_topology_benchmark",
            "--spec",
            str(paths.topology_benchmark_spec),
            "--expected-spec-sha256",
            file_sha256(paths.topology_benchmark_spec),
        ],
        "trainer-evaluator-lease-drill": [
            "{python}",
            "-m",
            "risk_score.autonomy_lease_drill",
            "--spec",
            str(paths.lease_drill_spec),
            "--expected-spec-sha256",
            file_sha256(paths.lease_drill_spec),
        ],
        "backlog-bound": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "backlog-bound",
            "--candidate-inbox",
            "{candidate_inbox}",
            "--backpressure",
            str(paths.backpressure),
            "--maximum-candidates",
            str(maximum_candidates),
            "--maximum-active-queue",
            "3",
            "--training-target-unit",
            "katago-risk-training.target",
            "--output",
            "{evidence}",
        ],
    }
    for gate_id in (
        "disposable-canary-drill",
        "crash-replay-drill",
        "rollback-before-admission-drill",
        "rollback-after-admission-drill",
        "shadow-controller-replay",
    ):
        templates[gate_id] = [
            "{python}",
            "-m",
            "risk_score.autonomy_promotion_drills",
            gate_id,
            "--spec",
            str(paths.promotion_drill_spec),
        ]
    if set(templates) != set(autonomy_bootstrap.GATE_ORDER):
        raise ProvisionerDependencyError("gate template inventory is incomplete")
    return {
        gate_id: list(templates[gate_id]) for gate_id in autonomy_bootstrap.GATE_ORDER
    }


def _walk_regular_files(
    root: Path,
    *,
    exclude_names: Iterable[str] = (),
) -> Iterable[Path]:
    target = Path(root)
    if not target.exists():
        return ()
    if target.is_symlink() or not target.is_dir():
        raise ProvisionerConflictError(f"unsafe generated directory: {target}")
    excluded = set(exclude_names)
    result: list[Path] = []
    for current_text, directory_names, file_names in os.walk(
        target, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        if current.is_symlink():
            raise ProvisionerConflictError(
                f"generated tree has a symlinked directory: {current}"
            )
        safe_directories = []
        for name in sorted(directory_names):
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ProvisionerConflictError(
                    f"generated tree has an unsafe directory: {child}"
                )
            if name.startswith("."):
                continue
            safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in sorted(file_names):
            path = current / name
            if name in excluded or name.startswith("."):
                continue
            if path.is_symlink() or not path.is_file():
                raise ProvisionerConflictError(
                    f"generated tree has an unsafe file: {path}"
                )
            result.append(path)
    return tuple(result)


def _generated_fixed_inputs(
    spec: ProvisionerSpec,
    paths: GenerationPaths,
    *,
    champion_path: Path,
    publisher_curation_status: Path,
    publisher_suite_manifest: Path,
) -> Sequence[Mapping[str, str]]:
    fixed_paths: list[Path] = [
        binding.path
        for binding in [
            *spec.immutable_inputs.values(),
            *spec.models,
            *spec.extra_inputs,
        ]
    ]
    fixed_paths.append(spec.path)
    fixed_paths.extend(_walk_regular_files(paths.root))
    registry_root = Path(spec.publisher_config["suite_rotation"]["registry_root"])
    fixed_paths.extend(
        _walk_regular_files(
            registry_root,
            exclude_names={
                "status.json",
                ".suite-rotation.lock",
            },
        )
    )
    fixed_paths.extend(
        [
            champion_path,
            spec.curation_status.path,
            spec.suite_manifest.path,
        ]
    )
    required_file_names = {
        "python_executable",
        "katago_binary",
        "trainer_spec",
        "consumer_spec",
        "original_model",
        "trainer_checkpoint",
        "deployment_manifest",
        "autonomy_policy",
    }
    required_paths = {
        spec.immutable_inputs[name].path for name in required_file_names
    } | {binding.path for binding in spec.models}
    readiness_paths = {
        publisher_curation_status,
        publisher_suite_manifest,
    }
    by_path: dict[Path, str] = {
        binding.path: binding.sha256 for binding in spec.extra_inputs
    }
    for path in fixed_paths:
        if not isinstance(path, Path):
            continue
        if path in required_paths or path in readiness_paths:
            continue
        if _paths_overlap(path, spec.bootstrap_state_root):
            raise ProvisionerConflictError(
                f"fixed generated input overlaps bootstrap state: {path}"
            )
        _require_generated_file(path, "generated fixed input")
        digest = file_sha256(path)
        previous = by_path.get(path)
        if previous is not None and previous != digest:
            raise ProvisionerDriftError(
                f"generated fixed input changed while collecting: {path}"
            )
        by_path[path] = digest
    return [
        {"path": str(path), "sha256": digest}
        for path, digest in sorted(by_path.items(), key=lambda item: str(item[0]))
    ]


def _publisher_spec_value(
    spec: ProvisionerSpec,
    paths: GenerationPaths,
    *,
    inventory: Mapping[str, Any],
    champion_path: Path,
    publisher_curation_status: Path,
    publisher_suite_manifest: Path,
) -> Mapping[str, Any]:
    inputs = spec.immutable_inputs
    required_files = {
        "python_executable": dict(inputs["python_executable"].to_dict()),
        "katago_binary": dict(inputs["katago_binary"].to_dict()),
        "trainer_spec": dict(inputs["trainer_spec"].to_dict()),
        "consumer_spec": dict(inputs["consumer_spec"].to_dict()),
        "original_model": dict(inputs["original_model"].to_dict()),
        "trainer_checkpoint": dict(inputs["trainer_checkpoint"].to_dict()),
        "deployment_manifest": dict(inputs["deployment_manifest"].to_dict()),
        "model_probe_config": _binding_value(paths.model_probe_config),
        "autonomy_policy": dict(inputs["autonomy_policy"].to_dict()),
        "cluster_executor_spec": _binding_value(paths.cluster_executor_spec),
        "adaptive_training_spec": _binding_value(paths.adaptive_training_spec),
        "suite_registry_spec": _binding_value(paths.registry_spec),
        "suite_rotation_spec": _binding_value(paths.suite_rotation_spec),
    }
    maximum_candidates = _candidate_total(inventory)
    value: dict[str, Any] = {
        "schema_version": 2,
        "contract": autonomy_bootstrap_spec.PUBLISHER_SPEC_CONTRACT,
        "repository": str(spec.repository),
        "source_revision": spec.source_revision,
        "run_root": str(spec.run_root),
        "state_root": str(spec.bootstrap_state_root),
        "curation_status": {
            "path": str(publisher_curation_status),
            "sha256": file_sha256(publisher_curation_status),
        },
        "suite_manifest": {
            "path": str(publisher_suite_manifest),
            "sha256": file_sha256(publisher_suite_manifest),
        },
        "candidate_inbox": str(spec.candidate_inbox),
        "activation_destination": str(spec.activation_destination),
        "files": required_files,
        "models": [
            dict(binding.to_dict())
            for binding in sorted(spec.models, key=lambda item: str(item.path))
        ],
        "extra_inputs": list(
            _generated_fixed_inputs(
                spec,
                paths,
                champion_path=champion_path,
                publisher_curation_status=publisher_curation_status,
                publisher_suite_manifest=publisher_suite_manifest,
            )
        ),
        "gate_argv_templates": _gate_templates(
            spec,
            paths,
            maximum_candidates=maximum_candidates,
        ),
        "runtime_commands": {
            name: list(argv) for name, argv in _runtime_commands(spec, paths).items()
        },
        "gpu": {"index": spec.gpu_index, "uuid": spec.gpu_uuid},
        "actor": spec.actor,
        "service_user": spec.service_user,
        "poll_interval_seconds": spec.poll_interval_seconds,
        "minimum_clean_observations": spec.minimum_clean_observations,
        "maximum_candidates": maximum_candidates,
        "maximum_active_queue": 3,
        "outputs": {
            "bootstrap_spec": str(spec.outputs.bootstrap_spec),
            "bootstrap_service_unit": str(spec.outputs.bootstrap_service_unit),
            "bootstrap_path_unit": str(spec.outputs.bootstrap_path_unit),
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    return value


def _artifact_bindings(
    spec: ProvisionerSpec, paths: GenerationPaths
) -> Sequence[Mapping[str, str]]:
    files = list(_walk_regular_files(paths.root))
    files.extend(
        path
        for path in (
            spec.outputs.bootstrap_spec,
            spec.outputs.bootstrap_service_unit,
            spec.outputs.bootstrap_path_unit,
        )
        if path.is_file() and not path.is_symlink()
    )
    by_path: dict[Path, str] = {}
    for path in files:
        digest = file_sha256(path)
        if path in by_path and by_path[path] != digest:
            raise ProvisionerDriftError(f"artifact changed during receipt: {path}")
        by_path[path] = digest
    return [
        {"path": str(path), "sha256": digest}
        for path, digest in sorted(by_path.items(), key=lambda item: str(item[0]))
    ]


def _verify_artifact_bindings(
    bindings: Any, *, role: str
) -> Sequence[Mapping[str, str]]:
    if not isinstance(bindings, list) or not bindings:
        raise ProvisionerDriftError(f"{role} artifact bindings are missing")
    paths: list[str] = []
    for index, raw in enumerate(bindings):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ProvisionerDriftError(f"{role} artifact binding {index} is malformed")
        path = Path(str(raw["path"]))
        expected = raw["sha256"]
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not isinstance(expected, str)
            or _SHA256_RE.fullmatch(expected) is None
            or file_sha256(path) != expected
        ):
            raise ProvisionerDriftError(f"{role} artifact changed: {path}")
        paths.append(str(path))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ProvisionerDriftError(
            f"{role} artifact bindings are not sorted and unique"
        )
    return bindings


def _publish_or_validate_readiness_receipt(
    spec: ProvisionerSpec,
    readiness: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = _readiness_receipt_path(spec)
    existing = _load_receipt(path, contract=READINESS_RECEIPT_CONTRACT)
    expected_body = {
        "spec": _spec_binding(spec),
        "readiness": dict(readiness),
        "candidate_inventory": dict(inventory),
        "maximum_candidates": _candidate_total(inventory),
        "maximum_active_queue": 3,
        "topology_choices": [4, 8, 16],
    }
    if existing is not None:
        comparable = {
            key: value
            for key, value in existing.items()
            if key
            not in {
                "_path",
                "schema_version",
                "contract",
                "receipt_sha256",
            }
        }
        if comparable != expected_body:
            raise ProvisionerDriftError(
                "suite/hash or candidate inventory changed after readiness lock"
            )
        return existing
    return _publish_receipt(
        path,
        expected_body,
        contract=READINESS_RECEIPT_CONTRACT,
    )


def _verify_materialization_receipt(
    spec: ProvisionerSpec,
    receipt: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    if receipt.get("spec") != _spec_binding(spec):
        raise ProvisionerDriftError("materialization receipt binds another spec")
    if receipt.get("readiness") != readiness:
        raise ProvisionerDriftError(
            "suite/hash drifted after bootstrap materialization"
        )
    if receipt.get("candidate_inventory") != inventory:
        raise ProvisionerDriftError(
            "candidate inventory drifted after bootstrap materialization"
        )
    if receipt.get("maximum_candidates") != _candidate_total(inventory):
        raise ProvisionerDriftError("materialization candidate bound is contradictory")
    _verify_artifact_bindings(receipt.get("artifacts"), role="materialization")
    loaded = autonomy_bootstrap.BootstrapSpec.load(spec.outputs.bootstrap_spec)
    publication = receipt.get("bootstrap_publication")
    if not isinstance(publication, Mapping):
        raise ProvisionerDriftError(
            "materialization receipt bootstrap publication is contradictory"
        )
    _validate_bootstrap_publication(publication, spec)
    if publication["bootstrap_spec"]["identity"] != loaded.identity:
        raise ProvisionerDriftError(
            "materialization receipt bootstrap identity changed"
        )


def _validate_bootstrap_publication(
    publication: Mapping[str, Any],
    spec: ProvisionerSpec,
) -> None:
    payload = dict(publication)
    supplied = payload.pop("publication_sha256", None)
    bootstrap_record = publication.get("bootstrap_spec")
    if (
        publication.get("schema_version") != 1
        or publication.get("contract") != autonomy_bootstrap_spec.PUBLICATION_CONTRACT
        or supplied != canonical_sha256(payload)
        or not isinstance(bootstrap_record, Mapping)
        or bootstrap_record.get("path") != str(spec.outputs.bootstrap_spec)
        or bootstrap_record.get("sha256") != file_sha256(spec.outputs.bootstrap_spec)
        or publication.get("suite_manifest_present") is not True
        or publication.get("enable_unit") != BOOTSTRAP_PATH_UNIT_NAME
    ):
        raise ProvisionerDriftError(
            "bootstrap publication contract or self-hash is invalid"
        )
    expected_records = {
        "bootstrap_service_unit": spec.outputs.bootstrap_service_unit,
        "bootstrap_path_unit": spec.outputs.bootstrap_path_unit,
    }
    for name, path in expected_records.items():
        record = publication.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != str(path)
            or path.is_symlink()
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
        ):
            raise ProvisionerDriftError(f"bootstrap publication {name} changed")


def _generate_downstream(
    spec: ProvisionerSpec,
    *,
    readiness: Mapping[str, Any],
    readiness_inventory: Mapping[str, Any],
    readiness_receipt: Mapping[str, Any],
    command_runner: CommandRunner,
    adapters: ProvisionerAdapters,
) -> Mapping[str, Any]:
    paths = _generation_paths(spec)
    _ensure_directory(paths.root)
    if os.geteuid() == 0:
        _make_root_owned_tree_readable(paths.root)
    _publish_immutable_json(
        paths.inventory_before,
        readiness_inventory,
        "candidate inventory before generation",
    )
    adapters.publish_model_probe_config(
        spec=spec,
        source=spec.immutable_inputs["model_probe_config_source"].path,
        destination=paths.model_probe_config,
    )
    _require_generated_file(paths.model_probe_config, "model probe configuration")

    adapters.publish_cluster_executor_spec(
        spec=spec,
        destination=paths.cluster_executor_spec,
    )
    _validate_generated_spec(
        paths.cluster_executor_spec,
        role="cluster executor specification",
        contract="risk-score-cluster-executor-spec-v1",
    )
    adapters.publish_registry_spec(
        spec=spec,
        destination=paths.registry_spec,
        suite_manifest=spec.suite_manifest.path,
    )
    _validate_generated_spec(
        paths.registry_spec,
        role="suite registry specification",
        contract="risk-score-evaluation-suite-registry-spec-v1",
    )
    adapters.publish_adaptive_spec(
        spec=spec,
        destination=paths.adaptive_training_spec,
        cluster_executor_spec_path=paths.cluster_executor_spec,
    )
    _validate_generated_spec(
        paths.adaptive_training_spec,
        role="adaptive training specification",
        contract="risk-score-adaptive-training-service-spec-v1",
    )
    adapters.publish_suite_service_spec(
        spec=spec,
        destination=paths.suite_rotation_spec,
        registry_spec_path=paths.registry_spec,
        cluster_executor_spec_path=paths.cluster_executor_spec,
    )
    _validate_generated_spec(
        paths.suite_rotation_spec,
        role="suite rotation service specification",
        contract="risk-score-suite-rotation-service-spec-v1",
    )
    shadow = adapters.build_shadow_runtime(
        spec=spec,
        output_dir=paths.shadow_runtime_root,
        registry_spec_path=paths.registry_spec,
        adaptive_spec_path=paths.adaptive_training_spec,
        suite_service_spec_path=paths.suite_rotation_spec,
        suite_manifest_path=spec.suite_manifest.path,
    )
    if not isinstance(shadow, Mapping) or shadow.get("mutation_enabled") is not False:
        raise ProvisionerDependencyError(
            "shadow runtime builder did not prove mutation disabled"
        )
    expected_shadow_paths = {
        "promotion_runtime": paths.shadow_runtime_root / "promotion-runtime.json",
        "gpu_lease_runtime": paths.shadow_runtime_root / "gpu-lease-runtime.json",
        "deployment_manifest": paths.shadow_runtime_root / "deployment-manifest.json",
        "service_spec": paths.shadow_runtime_root / "promotion-services.json",
    }
    for name, expected_path in expected_shadow_paths.items():
        if Path(str(shadow.get(name, ""))) != expected_path:
            raise ProvisionerDependencyError(
                f"shadow runtime reported the wrong {name} path"
            )
        _require_generated_file(expected_path, f"shadow {name}")
        if shadow.get(f"{name}_sha256") != file_sha256(expected_path):
            raise ProvisionerDependencyError(
                f"shadow runtime reported the wrong {name} hash"
            )
    promotion_runtime = expected_shadow_paths["promotion_runtime"]
    gpu_runtime = expected_shadow_paths["gpu_lease_runtime"]
    with adapters.controller_transaction(spec=spec) as controller_transaction:
        target_observation = adapters.verify_target_inactive(
            spec=spec,
            command_runner=command_runner,
        )
        if (
            not isinstance(target_observation, Mapping)
            or target_observation.get("inactive") is not True
        ):
            raise ProvisionerDependencyError(
                "production target inactivity was not proven"
            )
        registry_result = adapters.bootstrap_registry(
            spec=spec,
            registry_spec_path=paths.registry_spec,
            suite_manifest=spec.suite_manifest.path,
            controller_transaction=controller_transaction,
        )
        if not isinstance(registry_result, Mapping):
            raise ProvisionerDependencyError(
                "suite registry bootstrap returned no evidence"
            )
        champion_result = adapters.initialize_champion(
            spec=spec,
            promotion_runtime_path=promotion_runtime,
            receipt_path=paths.champion_receipt,
            target_observation=target_observation,
            controller_transaction=controller_transaction,
        )
        if not isinstance(champion_result, Mapping):
            raise ProvisionerDependencyError(
                "champion projection initializer returned no evidence"
            )
        final_target_observation = adapters.verify_target_inactive(
            spec=spec,
            command_runner=command_runner,
        )
        if (
            not isinstance(final_target_observation, Mapping)
            or final_target_observation.get("inactive") is not True
        ):
            raise ProvisionerDependencyError(
                "production target became active during champion transaction"
            )
    manifest_binding = registry_result.get("manifest")
    if not isinstance(manifest_binding, Mapping) or set(manifest_binding) != {
        "path",
        "sha256",
    }:
        raise ProvisionerDependencyError(
            "suite registry bootstrap did not bind its active manifest"
        )
    active_manifest = _require_generated_file(
        Path(str(manifest_binding["path"])),
        "registry-owned active suite manifest",
    )
    registry_root = Path(spec.publisher_config["suite_rotation"]["registry_root"])
    if (
        manifest_binding["sha256"] != file_sha256(active_manifest)
        or file_sha256(active_manifest) != file_sha256(spec.suite_manifest.path)
        or not _strictly_within(active_manifest, registry_root)
        or active_manifest.name != "manifest.json"
    ):
        raise ProvisionerDependencyError(
            "registry-owned active suite changed or differs from authority"
        )
    _publish_active_curation_status(
        spec,
        active_manifest=active_manifest,
        destination=paths.active_curation_status,
    )
    champion_record = champion_result.get("champion")
    champion_path = (
        Path(str(champion_record.get("path", "")))
        if isinstance(champion_record, Mapping)
        else spec.run_root / "promotion" / "champion.json"
    )
    if champion_path != spec.run_root / "promotion" / "champion.json":
        raise ProvisionerDependencyError(
            "champion projection escaped the production control-plane path"
        )
    _require_generated_file(champion_path, "initial suite champion projection")
    expected_champion_hash = spec.immutable_inputs["suite_champion_model"].sha256
    if (
        not isinstance(champion_record, Mapping)
        or champion_record.get("champion_hash") != expected_champion_hash
        or champion_record.get("generation_id") != spec.initial_generation_id
        or registry_result.get("champion_sha256") != expected_champion_hash
        or registry_result.get("generation_id") != spec.initial_generation_id
    ):
        raise ProvisionerDependencyError(
            "suite and promotion control planes disagree on initial champion"
        )

    adapters.publish_promotion_drill_spec(
        spec=spec,
        destination=paths.promotion_drill_spec,
        promotion_runtime_path=promotion_runtime,
        champion_path=champion_path,
    )
    _validate_generated_spec(
        paths.promotion_drill_spec,
        role="autonomy promotion drill specification",
        contract="risk-score-autonomy-promotion-drill-spec-v1",
    )
    adapters.publish_topology_specs(
        spec=spec,
        workload_spec_path=paths.topology_workload_spec,
        benchmark_spec_path=paths.topology_benchmark_spec,
        model_probe_config=paths.model_probe_config,
    )
    _require_generated_file(
        paths.topology_workload_spec,
        "evaluator benchmark workload specification",
    )
    topology_value = _validate_generated_spec(
        paths.topology_benchmark_spec,
        role="evaluator topology benchmark specification",
        contract="risk-score-evaluator-topology-benchmark-spec-v1",
    )
    if topology_value.get("evidence_output") != str(
        spec.bootstrap_state_root
        / "gate-evidence"
        / "evaluator-topology-benchmark.json"
    ):
        raise ProvisionerDependencyError(
            "topology specification has the wrong gate evidence output"
        )
    lease_runtime = adapters.build_lease_runtime(
        spec=spec,
        source_gpu_lease_config=gpu_runtime,
        destination=paths.lease_runtime,
        model_probe_config=paths.model_probe_config,
    )
    if (
        not isinstance(lease_runtime, Mapping)
        or lease_runtime.get("mutation_enabled") is not True
        or lease_runtime.get("evaluator_process_count") != 1
        or lease_runtime.get("installed") is not False
        or Path(str(lease_runtime.get("path", ""))) != paths.lease_runtime
        or lease_runtime.get("sha256") != file_sha256(paths.lease_runtime)
    ):
        raise ProvisionerDependencyError(
            "dedicated lease drill runtime evidence is contradictory"
        )
    adapters.publish_lease_specs(
        spec=spec,
        probe_spec_path=paths.lease_probe_spec,
        drill_spec_path=paths.lease_drill_spec,
        gpu_lease_config=paths.lease_runtime,
        model_probe_config=paths.model_probe_config,
    )
    _require_generated_file(paths.lease_probe_spec, "autonomy lease probe spec")
    lease_value = _validate_generated_spec(
        paths.lease_drill_spec,
        role="autonomy lease drill specification",
        contract="risk-score-autonomy-lease-drill-spec-v1",
    )
    if lease_value.get("evidence_output") != str(
        spec.bootstrap_state_root
        / "gate-evidence"
        / "trainer-evaluator-lease-drill.json"
    ):
        raise ProvisionerDependencyError(
            "lease drill specification has the wrong gate evidence output"
        )
    trainer_source = lease_value.get("trainer_source")
    if trainer_source != {"kind": "launch"}:
        raise ProvisionerDependencyError(
            "lease drill trainer_source must be exactly kind launch"
        )
    adapters.publish_backpressure(
        spec=spec,
        destination=paths.backpressure,
    )
    _require_generated_file(paths.backpressure, "bootstrap backpressure")

    inventory_after = _candidate_inventory(
        spec,
        output=paths.inventory_after,
        adapter=adapters.candidate_inventory,
    )
    if inventory_after != readiness_inventory:
        raise ProvisionerDriftError(
            "candidate inventory changed while generating gate specifications"
        )
    publisher_value = _publisher_spec_value(
        spec,
        paths,
        inventory=inventory_after,
        champion_path=champion_path,
        publisher_curation_status=paths.active_curation_status,
        publisher_suite_manifest=active_manifest,
    )
    _publish_immutable_json(
        paths.publisher_spec,
        publisher_value,
        "bootstrap publisher specification",
    )
    loaded_publisher = autonomy_bootstrap_spec.BootstrapPublisherSpec.load(
        paths.publisher_spec
    )
    if loaded_publisher.maximum_active_queue != 3 or (
        loaded_publisher.maximum_candidates != _candidate_total(inventory_after)
    ):
        raise ProvisionerDependencyError(
            "bootstrap publisher candidate bounds are contradictory"
        )
    _verify_clean_checkout(spec, command_runner)
    publication = adapters.materialize_bootstrap(
        paths.publisher_spec,
        expected_spec_sha256=file_sha256(paths.publisher_spec),
        command_runner=command_runner,
    )
    if not isinstance(publication, Mapping):
        raise ProvisionerDependencyError(
            "bootstrap materializer returned a contradictory publication"
        )
    try:
        _validate_bootstrap_publication(publication, spec)
    except ProvisionerDriftError as exc:
        raise ProvisionerDependencyError(str(exc)) from exc
    loaded_bootstrap = autonomy_bootstrap.BootstrapSpec.load(
        spec.outputs.bootstrap_spec
    )
    if [gate.gate_id for gate in loaded_bootstrap.gates] != list(
        autonomy_bootstrap.GATE_ORDER
    ):
        raise ProvisionerDependencyError(
            "materialized bootstrap gate inventory is not canonical"
        )
    inventory_final = _candidate_inventory(
        spec,
        output=paths.inventory_final,
        adapter=adapters.candidate_inventory,
    )
    if inventory_final != readiness_inventory:
        raise ProvisionerDriftError(
            "candidate inventory changed during bootstrap materialization"
        )
    final_readiness = _readiness_observation(spec)
    if final_readiness != readiness:
        raise ProvisionerDriftError(
            "authoritative curation status or suite changed during materialization"
        )
    _verify_clean_checkout(spec, command_runner)
    spec.verify_immutable_inputs()
    receipt = _publish_receipt(
        _materialization_receipt_path(spec),
        {
            "spec": _spec_binding(spec),
            "readiness": dict(readiness),
            "readiness_receipt": _receipt_reference(readiness_receipt),
            "candidate_inventory": dict(inventory_final),
            "maximum_candidates": _candidate_total(inventory_final),
            "maximum_active_queue": 3,
            "topology_choices": [4, 8, 16],
            "gpu": {"index": spec.gpu_index, "uuid": spec.gpu_uuid},
            "champion_projection": dict(champion_result),
            "registry_bootstrap": dict(registry_result),
            "bootstrap_publication": dict(publication),
            "artifacts": list(_artifact_bindings(spec, paths)),
            "enable_unit": BOOTSTRAP_PATH_UNIT_NAME,
            "bootstrap_service_enabled": False,
            "mutation_activated": False,
        },
        contract=MATERIALIZATION_RECEIPT_CONTRACT,
    )
    return receipt


def _load_installation_receipts(
    spec: ProvisionerSpec,
) -> list[Mapping[str, Any]]:
    receipts = []
    for phase in ("legacy-to-prepare", "arm-bootstrap-path"):
        installation_root = _installation_receipt_path(spec, phase).parent
        paths = (
            sorted(installation_root.glob(f"{phase}*.json"), key=str)
            if installation_root.is_dir()
            else []
        )
        for path in paths:
            receipt = _load_receipt(
                path,
                contract=INSTALLATION_RECEIPT_CONTRACT,
                hash_field="installation_sha256",
            )
            if receipt is None:
                continue
            if (
                receipt.get("spec") != _spec_binding(spec)
                or receipt.get("phase") != phase
                or receipt.get("bootstrap_service_enabled") is not False
                or receipt.get("mutation_activated") is not False
            ):
                raise ProvisionerDriftError(
                    f"installation receipt is contradictory: {phase}"
                )
            _assert_path_only_actions(receipt.get("actions", []))
            receipts.append(receipt)
    return receipts


def _materialize_provisioner_locked(
    spec_path: Path | ProvisionerSpec,
    *,
    expected_spec_sha256: str | None = None,
    apply: bool = False,
    command_runner: CommandRunner = subprocess.run,
    systemctl_runner: CommandRunner = subprocess.run,
    adapters: ProvisionerAdapters | None = None,
) -> Mapping[str, Any]:
    """Install preparation units and materialize the bootstrap when ready."""

    spec = (
        spec_path
        if isinstance(spec_path, ProvisionerSpec)
        else ProvisionerSpec.load(spec_path, expected_spec_sha256=expected_spec_sha256)
    )
    selected_adapters = adapters or ProvisionerAdapters()
    plan_value = plan_provisioning(spec, command_runner=command_runner)
    preparation = _publish_preparation(spec, plan_value)
    if apply:
        _validate_real_apply(spec, systemctl_runner)
        if systemctl_runner is subprocess.run:
            _prepare_root_owned_publication_roots(spec)
    installations = _load_installation_receipts(spec)

    readiness = _readiness_observation(spec)
    if readiness["decision"] == "WAIT":
        if apply:
            _validate_real_apply(spec, systemctl_runner)
            if systemctl_runner is subprocess.run:
                _prepare_service_ownership(spec)
            installations.append(
                apply_legacy_cutover(spec, command_runner=systemctl_runner)
            )
        status_value = _status_value(
            spec,
            state="WAIT",
            plan_value=plan_value,
            readiness=readiness,
            preparation_receipt=preparation,
            materialization_receipt=None,
            installation_receipts=installations,
        )
        _atomic_replace_json(spec.outputs.status, status_value, "provisioner status")
        return status_value

    inventory = _candidate_inventory(
        spec,
        output=None,
        adapter=selected_adapters.candidate_inventory,
    )
    readiness_receipt = _publish_or_validate_readiness_receipt(
        spec, readiness, inventory
    )
    materialization = _load_receipt(
        _materialization_receipt_path(spec),
        contract=MATERIALIZATION_RECEIPT_CONTRACT,
    )
    try:
        if materialization is None:
            materialization = _generate_downstream(
                spec,
                readiness=readiness,
                readiness_inventory=inventory,
                readiness_receipt=readiness_receipt,
                command_runner=command_runner,
                adapters=selected_adapters,
            )
        else:
            _verify_materialization_receipt(
                spec,
                materialization,
                readiness=readiness,
                inventory=inventory,
            )
        if apply:
            _validate_real_apply(spec, systemctl_runner)
            if systemctl_runner is subprocess.run:
                _prepare_service_ownership(spec)
            arm_readiness = _readiness_observation(spec)
            arm_inventory = _candidate_inventory(
                spec,
                output=None,
                adapter=selected_adapters.candidate_inventory,
            )
            _verify_materialization_receipt(
                spec,
                materialization,
                readiness=arm_readiness,
                inventory=arm_inventory,
            )
            arm = apply_bootstrap_arm(spec, command_runner=systemctl_runner)
            installations.append(arm)
        state = "APPLIED" if apply else "MATERIALIZED"
        status_value = _status_value(
            spec,
            state=state,
            plan_value=plan_value,
            readiness=readiness,
            preparation_receipt=preparation,
            materialization_receipt=materialization,
            installation_receipts=installations,
        )
        _atomic_replace_json(spec.outputs.status, status_value, "provisioner status")
        return status_value
    except Exception as exc:
        with contextlib.suppress(Exception):
            _publish_failure(spec, phase="materialize", error=exc)
        with contextlib.suppress(Exception):
            failed = _status_value(
                spec,
                state="ERROR",
                plan_value=plan_value,
                readiness=readiness,
                preparation_receipt=preparation,
                materialization_receipt=materialization,
                installation_receipts=installations,
                error=exc,
            )
            _atomic_replace_json(
                spec.outputs.status, failed, "provisioner failure status"
            )
        raise


def materialize_provisioner(
    spec_path: Path | ProvisionerSpec,
    *,
    expected_spec_sha256: str | None = None,
    apply: bool = False,
    command_runner: CommandRunner = subprocess.run,
    systemctl_runner: CommandRunner = subprocess.run,
    adapters: ProvisionerAdapters | None = None,
) -> Mapping[str, Any]:
    """Materialize under one exclusive revision-bound provisioner lock."""

    spec = (
        spec_path
        if isinstance(spec_path, ProvisionerSpec)
        else ProvisionerSpec.load(spec_path, expected_spec_sha256=expected_spec_sha256)
    )
    _verify_environment(spec, command_runner)
    _ensure_directory(spec.state_root)
    from risk_score.promotion_state import ControllerLock

    with ControllerLock(
        spec.state_root / ".autonomy-provisioner.lock",
        owner=spec.actor,
    ):
        return _materialize_provisioner_locked(
            spec,
            apply=apply,
            command_runner=command_runner,
            systemctl_runner=systemctl_runner,
            adapters=adapters,
        )


materialize = materialize_provisioner
provision = materialize_provisioner


def _validate_persisted_status(spec: ProvisionerSpec) -> Mapping[str, Any] | None:
    path = spec.outputs.status
    if not os.path.lexists(os.fspath(path)):
        return None
    value = _load_canonical_object(path, "provisioner status")
    payload = dict(value)
    supplied = payload.pop("status_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != STATUS_CONTRACT
        or value.get("spec") != _spec_binding(spec)
        or supplied != canonical_sha256(payload)
    ):
        raise ProvisionerDriftError("persisted provisioner status is invalid")
    return value


def provisioner_status(
    spec_path: Path | ProvisionerSpec,
    *,
    expected_spec_sha256: str | None = None,
    command_runner: CommandRunner = subprocess.run,
    adapters: ProvisionerAdapters | None = None,
) -> Mapping[str, Any]:
    """Revalidate receipts and derive current provisioner status without writes."""

    spec = (
        spec_path
        if isinstance(spec_path, ProvisionerSpec)
        else ProvisionerSpec.load(spec_path, expected_spec_sha256=expected_spec_sha256)
    )
    selected_adapters = adapters or ProvisionerAdapters()
    _validate_persisted_status(spec)
    plan_value = plan_provisioning(spec, command_runner=command_runner)
    preparation = _load_receipt(
        _preparation_receipt_path(spec),
        contract=PREPARATION_RECEIPT_CONTRACT,
    )
    if preparation is not None:
        if (
            preparation.get("spec") != _spec_binding(spec)
            or preparation.get("plan_sha256") != plan_value["plan_sha256"]
        ):
            raise ProvisionerDriftError("preparation receipt changed")
        units = preparation.get("units")
        if not isinstance(units, Mapping) or set(units) != {
            PREPARE_SERVICE_UNIT_NAME,
            PREPARE_PATH_UNIT_NAME,
        }:
            raise ProvisionerDriftError("preparation receipt unit inventory is invalid")
        _verify_artifact_bindings(
            sorted(
                [
                    units[PREPARE_SERVICE_UNIT_NAME],
                    units[PREPARE_PATH_UNIT_NAME],
                ],
                key=lambda item: item["path"],
            ),
            role="preparation",
        )
    installations = _load_installation_receipts(spec)
    readiness = _readiness_observation(spec)
    materialization = _load_receipt(
        _materialization_receipt_path(spec),
        contract=MATERIALIZATION_RECEIPT_CONTRACT,
    )
    if materialization is not None:
        if readiness["decision"] != "READY":
            raise ProvisionerDriftError(
                "materialized readiness artifacts are no longer present"
            )
        inventory = _candidate_inventory(
            spec,
            output=None,
            adapter=selected_adapters.candidate_inventory,
        )
        _verify_materialization_receipt(
            spec,
            materialization,
            readiness=readiness,
            inventory=inventory,
        )
        state = (
            "APPLIED"
            if any(
                receipt.get("phase") == "arm-bootstrap-path"
                for receipt in installations
            )
            else "MATERIALIZED"
        )
    elif readiness["decision"] == "WAIT":
        state = "WAIT" if preparation is not None else "UNPREPARED"
    else:
        state = "READY" if preparation is not None else "UNPREPARED"
    return _status_value(
        spec,
        state=state,
        plan_value=plan_value,
        readiness=readiness,
        preparation_receipt=preparation,
        materialization_receipt=materialization,
        installation_receipts=installations,
    )


status = provisioner_status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "materialize", "status"):
        child = subparsers.add_parser(mode)
        child.add_argument("--spec", required=True, type=Path)
        child.add_argument("--expected-spec-sha256")
        if mode == "materialize":
            child.add_argument("--apply", action="store_true")
    runtime = subparsers.add_parser("suite-rotation-runtime")
    runtime.add_argument("--service-spec", required=True, type=Path)
    runtime.add_argument("--suite-registry-spec", required=True, type=Path)
    return parser.parse_args(argv)


def _run_suite_rotation_runtime(args: argparse.Namespace) -> int:
    from risk_score.suite_rotation_service import load_service_spec
    from risk_score.suite_rotation_service import main as service_main

    registry = _validate_generated_spec(
        args.suite_registry_spec,
        role="suite rotation runtime registry specification",
        contract="risk-score-evaluation-suite-registry-spec-v1",
    )
    service = load_service_spec(args.service_spec)
    if (
        service.registry_spec.path != args.suite_registry_spec.resolve()
        or service.registry_spec.sha256 != file_sha256(args.suite_registry_spec)
        or not isinstance(registry.get("spec_sha256"), str)
    ):
        raise ProvisionerDriftError(
            "suite rotation service and registry specifications disagree"
        )
    return service_main(
        [
            "watch",
            "--spec",
            str(service.path),
            "--expected-spec-sha256",
            service.file_sha256,
        ]
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    command_runner: CommandRunner = subprocess.run,
    systemctl_runner: CommandRunner = subprocess.run,
    adapters: ProvisionerAdapters | None = None,
) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "suite-rotation-runtime":
            return _run_suite_rotation_runtime(args)
        if args.mode == "plan":
            result = plan_provisioning(
                args.spec,
                expected_spec_sha256=args.expected_spec_sha256,
                command_runner=command_runner,
            )
        elif args.mode == "materialize":
            result = materialize_provisioner(
                args.spec,
                expected_spec_sha256=args.expected_spec_sha256,
                apply=args.apply,
                command_runner=command_runner,
                systemctl_runner=systemctl_runner,
                adapters=adapters,
            )
        else:
            result = provisioner_status(
                args.spec,
                expected_spec_sha256=args.expected_spec_sha256,
                command_runner=command_runner,
                adapters=adapters,
            )
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        TypeError,
        ValueError,
        ProvisionerError,
        autonomy_bootstrap.BootstrapError,
        autonomy_bootstrap_spec.BootstrapSpecPublicationError,
    ) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 1 if args.mode == "materialize" and result.get("state") == "WAIT" else 0


__all__ = [
    "BOOTSTRAP_PATH_UNIT_NAME",
    "BOOTSTRAP_SERVICE_UNIT_NAME",
    "FAILURE_RECEIPT_CONTRACT",
    "INSTALLATION_RECEIPT_CONTRACT",
    "LEGACY_PATH_UNIT_NAME",
    "MATERIALIZATION_RECEIPT_CONTRACT",
    "PATH_UNIT_NAME",
    "PLAN_CONTRACT",
    "PREPARATION_RECEIPT_CONTRACT",
    "PREPARE_PATH_UNIT_NAME",
    "PREPARE_SERVICE_UNIT_NAME",
    "PREPARE_UNIT_NAME",
    "PROVISIONER_SPEC_CONTRACT",
    "PROVISIONER_STATUS_CONTRACT",
    "READINESS_RECEIPT_CONTRACT",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "STATUS_CONTRACT",
    "ProvisionerAdapters",
    "ProvisionerApplyError",
    "ProvisionerConflictError",
    "ProvisionerDependencyError",
    "ProvisionerDriftError",
    "ProvisionerError",
    "ProvisionerSpec",
    "ProvisionerSpecError",
    "apply_bootstrap_arm",
    "apply_cutover",
    "apply_legacy_cutover",
    "load_provisioner_spec",
    "load_spec",
    "main",
    "materialize",
    "materialize_provisioner",
    "parse_args",
    "plan",
    "plan_provisioning",
    "provision",
    "provisioner_status",
    "publish_provisioner_spec",
    "publish_spec",
    "render_prepare_path_unit",
    "render_prepare_service_unit",
    "render_systemd_path_unit",
    "render_systemd_service_unit",
    "status",
]


if __name__ == "__main__":
    raise SystemExit(main())
