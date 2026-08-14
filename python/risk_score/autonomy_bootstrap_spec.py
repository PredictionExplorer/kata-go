#!/usr/bin/env python3
"""Materialize the immutable production autonomy-bootstrap specification.

The publisher consumes a canonical, self-hashed input specification.  Gate
commands are argv templates, not shell fragments, and are restricted to the
production executor modules.  ``{evidence}`` is expanded to a unique path below
the bootstrap state root; the remaining supported placeholders name frozen
publisher inputs.

The companion systemd path unit is deliberately independent of
``katago-risk-training.target``.  It watches only the fixed authoritative suite
manifest path and starts the hash-pinned bootstrap service when that file
appears.  Existence is merely the wake-up condition: the readiness gate still
validates the manifest contract and self-hash before any later gate can run.
Operators should enable the path unit, not the service directly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from risk_score.autonomy_bootstrap import (
    BOOTSTRAP_UNIT_NAME,
    GATE_ORDER,
    REQUIRED_ACTIVATION_UNITS,
    SPEC_CONTRACT,
    BootstrapError,
    BootstrapSpec,
    render_bootstrap_systemd_unit,
)
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256

LEGACY_PUBLISHER_SPEC_CONTRACT = "risk-score-autonomy-bootstrap-publisher-spec-v1"
PUBLISHER_SPEC_CONTRACT = "risk-score-autonomy-bootstrap-publisher-spec-v2"
PUBLICATION_CONTRACT = "risk-score-autonomy-bootstrap-publication-v1"
BOOTSTRAP_PATH_UNIT_NAME = "katago-risk-autonomy-bootstrap.path"
PATH_UNIT_NAME = BOOTSTRAP_PATH_UNIT_NAME

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_MAX_SPEC_BYTES = 4 * 1024 * 1024

_REQUIRED_FILES = frozenset(
    {
        "python_executable",
        "katago_binary",
        "trainer_spec",
        "consumer_spec",
        "original_model",
        "trainer_checkpoint",
        "deployment_manifest",
        "model_probe_config",
        "autonomy_policy",
        "cluster_executor_spec",
        "adaptive_training_spec",
        "suite_registry_spec",
        "suite_rotation_spec",
    }
)
_RUNTIME_COMMANDS = (
    "shuffler",
    "exporter",
    "evaluator",
    "cluster_executor",
    "adaptive_training",
    "suite_rotation",
)
_EXPECTED_GATE_MODULES = {
    "curation-suite-readiness": "risk_score.autonomy_gate_runners",
    "filesystem-rename-fsync": "risk_score.promotion_preflight",
    "deployment-hash-validation": "risk_score.autonomy_gate_runners",
    "candidate-inventory": "risk_score.promotion_preflight",
    "cuda-model-probes": "risk_score.autonomy_gate_runners",
    "evaluator-topology-benchmark": "risk_score.evaluator_topology_benchmark",
    "trainer-evaluator-lease-drill": "risk_score.autonomy_lease_drill",
    "disposable-canary-drill": "risk_score.autonomy_promotion_drills",
    "crash-replay-drill": "risk_score.autonomy_promotion_drills",
    "rollback-before-admission-drill": "risk_score.autonomy_promotion_drills",
    "rollback-after-admission-drill": "risk_score.autonomy_promotion_drills",
    "shadow-controller-replay": "risk_score.autonomy_promotion_drills",
    "backlog-bound": "risk_score.autonomy_gate_runners",
}
_EXPECTED_GATE_SUBCOMMANDS = {
    "curation-suite-readiness": "curation-suite-readiness",
    "filesystem-rename-fsync": "filesystem-test",
    "deployment-hash-validation": "deployment-hash-validation",
    "candidate-inventory": "candidate-inventory",
    "cuda-model-probes": "cuda-model-probes",
    "disposable-canary-drill": "disposable-canary-drill",
    "crash-replay-drill": "crash-replay-drill",
    "rollback-before-admission-drill": "rollback-before-admission-drill",
    "rollback-after-admission-drill": "rollback-after-admission-drill",
    "shadow-controller-replay": "shadow-controller-replay",
    "backlog-bound": "backlog-bound",
}
_EXPECTED_EXECUTOR_SPEC_CONTRACTS = {
    "evaluator-topology-benchmark": "risk-score-evaluator-topology-benchmark-spec-v1",
    "trainer-evaluator-lease-drill": "risk-score-autonomy-lease-drill-spec-v1",
}
_CONTROL_MODULES = frozenset(
    {
        "risk_score.autonomy_bootstrap",
        "risk_score.autonomy_bootstrap_spec",
        "risk_score.build_live_runtime",
        "risk_score.service_activation",
        *_EXPECTED_GATE_MODULES.values(),
    }
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract",
    "repository",
    "source_revision",
    "run_root",
    "state_root",
    "curation_status",
    "suite_manifest",
    "candidate_inbox",
    "activation_destination",
    "files",
    "models",
    "extra_inputs",
    "gate_argv_templates",
    "runtime_commands",
    "gpu",
    "actor",
    "service_user",
    "poll_interval_seconds",
    "minimum_clean_observations",
    "maximum_candidates",
    "maximum_active_queue",
    "outputs",
    "spec_sha256",
}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class BootstrapSpecPublicationError(ValueError):
    """The publisher input or immutable output set is unsafe."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise BootstrapSpecPublicationError(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _ensure_finite(value: Any, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BootstrapSpecPublicationError(f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise BootstrapSpecPublicationError(f"{role} contains a non-string key")
            _ensure_finite(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite(child, f"{role}[{index}]")


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    _reject_symlink_ancestors(source, role)
    if source.is_symlink() or not source.is_file():
        raise BootstrapSpecPublicationError(
            f"{role} must be a regular non-symlink file"
        )
    if source.stat().st_size > _MAX_SPEC_BYTES:
        raise BootstrapSpecPublicationError(f"{role} exceeds the size limit")
    data = source.read_bytes()
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BootstrapSpecPublicationError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BootstrapSpecPublicationError(f"{role} root must be an object")
    _ensure_finite(value, role)
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise BootstrapSpecPublicationError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise BootstrapSpecPublicationError(
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
        raise BootstrapSpecPublicationError(f"{role} must be a nonempty absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise BootstrapSpecPublicationError(
            f"{role} must be absolute and lexically normalized"
        )
    _reject_symlink_ancestors(path, role)
    return path


def _existing_directory(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.is_symlink() or not path.is_dir():
        raise BootstrapSpecPublicationError(
            f"{role} must be an existing non-symlink directory"
        )
    return path


def _future_file(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise BootstrapSpecPublicationError(
                f"{role} must be a regular non-symlink file"
            )
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


def _required_string(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise BootstrapSpecPublicationError(f"{role} must be a nonempty string")
    return value


def _required_integer(value: Any, role: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BootstrapSpecPublicationError(f"{role} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise BootstrapSpecPublicationError(f"{role} must be a positive finite number")
    return float(value)


@dataclass(frozen=True)
class InputBinding:
    path: Path
    sha256: str

    @classmethod
    def from_value(cls, value: Any, role: str) -> "InputBinding":
        if not isinstance(value, Mapping):
            raise BootstrapSpecPublicationError(f"{role} must be an object")
        _exact_keys(value, {"path", "sha256"}, role)
        path = _absolute_path(value["path"], f"{role}.path")
        supplied = value["sha256"]
        if not isinstance(supplied, str) or _SHA256_RE.fullmatch(supplied) is None:
            raise BootstrapSpecPublicationError(
                f"{role}.sha256 must be a lowercase SHA-256"
            )
        if path.is_symlink() or not path.is_file():
            raise BootstrapSpecPublicationError(
                f"{role}.path must be a regular non-symlink file"
            )
        if file_sha256(path) != supplied:
            raise BootstrapSpecPublicationError(f"{role}.path hash changed")
        return cls(path=path, sha256=supplied)

    def value(self) -> Mapping[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class ReadinessInputBinding:
    path: Path
    sha256: Optional[str]

    @classmethod
    def from_value(cls, value: Any, role: str) -> "ReadinessInputBinding":
        if not isinstance(value, Mapping):
            raise BootstrapSpecPublicationError(f"{role} must be an object")
        _exact_keys(value, {"path", "sha256"}, role)
        path = _absolute_path(value["path"], f"{role}.path")
        supplied = value["sha256"]
        if supplied is not None and (
            not isinstance(supplied, str) or _SHA256_RE.fullmatch(supplied) is None
        ):
            raise BootstrapSpecPublicationError(
                f"{role}.sha256 must be null or a lowercase SHA-256"
            )
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise BootstrapSpecPublicationError(
                    f"{role}.path must be a regular non-symlink file"
                )
            if supplied is not None and file_sha256(path) != supplied:
                raise BootstrapSpecPublicationError(f"{role}.path hash changed")
        elif supplied is not None:
            raise BootstrapSpecPublicationError(
                f"{role}.path is missing despite a bound hash"
            )
        return cls(path=path, sha256=supplied)

    def value(self) -> Mapping[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256}


def _validate_argv(value: Any, role: str) -> Tuple[str, ...]:
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
        raise BootstrapSpecPublicationError(
            f"{role} must be a nonempty argv string array"
        )
    return tuple(value)


def _binding_map(bindings: Sequence[InputBinding], role: str) -> Dict[Path, str]:
    result: Dict[Path, str] = {}
    for binding in bindings:
        previous = result.get(binding.path)
        if previous is not None and previous != binding.sha256:
            raise BootstrapSpecPublicationError(
                f"{role} binds one path to conflicting hashes: {binding.path}"
            )
        result[binding.path] = binding.sha256
    return result


@dataclass(frozen=True)
class BootstrapPublisherSpec:
    path: Path
    file_sha256: str
    identity: str
    repository: Path
    source_revision: str
    run_root: Path
    state_root: Path
    curation_status_binding: ReadinessInputBinding
    suite_manifest_binding: ReadinessInputBinding
    candidate_inbox: Path
    activation_destination: Path
    files: Mapping[str, InputBinding]
    models: Tuple[InputBinding, ...]
    extra_inputs: Tuple[InputBinding, ...]
    gate_argv_templates: Mapping[str, Tuple[str, ...]]
    runtime_commands: Mapping[str, Tuple[str, ...]]
    gpu_index: int
    expected_gpu_uuid: str
    actor: str
    service_user: str
    poll_interval_seconds: float
    minimum_clean_observations: int
    maximum_candidates: int
    maximum_active_queue: int
    bootstrap_spec_path: Path
    bootstrap_service_unit_path: Path
    bootstrap_path_unit_path: Path

    @property
    def curation_status(self) -> Path:
        return self.curation_status_binding.path

    @property
    def suite_manifest(self) -> Path:
        return self.suite_manifest_binding.path

    @classmethod
    def load(cls, path: Path) -> "BootstrapPublisherSpec":
        source = Path(path)
        if not source.is_absolute():
            source = source.resolve()
        raw = _load_canonical_object(source, "bootstrap publisher specification")
        if raw.get("contract") == LEGACY_PUBLISHER_SPEC_CONTRACT:
            raise BootstrapSpecPublicationError(
                "bootstrap publisher specification v1 is legacy-unsatisfiable: "
                "it cannot bind separate readiness registry and suite service specs"
            )
        _exact_keys(raw, _TOP_LEVEL_KEYS, "bootstrap publisher specification")
        if raw["schema_version"] != 2 or raw["contract"] != PUBLISHER_SPEC_CONTRACT:
            raise BootstrapSpecPublicationError(
                "bootstrap publisher specification contract is unsupported"
            )
        payload = dict(raw)
        identity = payload.pop("spec_sha256")
        if (
            not isinstance(identity, str)
            or _SHA256_RE.fullmatch(identity) is None
            or identity != canonical_sha256(payload)
        ):
            raise BootstrapSpecPublicationError(
                "bootstrap publisher specification self-hash is invalid"
            )

        repository = _existing_directory(raw["repository"], "repository")
        python_root = _existing_directory(
            str(repository / "python"), "repository Python root"
        )
        if python_root.parent != repository:
            raise BootstrapSpecPublicationError(
                "repository Python root is not canonical"
            )
        revision = raw["source_revision"]
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise BootstrapSpecPublicationError(
                "source_revision must be a lowercase Git object id"
            )

        run_root = _existing_directory(raw["run_root"], "run_root")
        if _paths_overlap(repository, run_root):
            raise BootstrapSpecPublicationError(
                "run_root must be outside the deployed source checkout"
            )
        state_root = _absolute_path(raw["state_root"], "state_root")
        if not _strictly_within(state_root, run_root):
            raise BootstrapSpecPublicationError(
                "state_root must be strictly inside run_root"
            )
        if state_root.exists() and (state_root.is_symlink() or not state_root.is_dir()):
            raise BootstrapSpecPublicationError(
                "state_root must be a non-symlink directory"
            )
        if not state_root.exists() and (
            state_root.parent.is_symlink() or not state_root.parent.is_dir()
        ):
            raise BootstrapSpecPublicationError(
                "state_root parent must already be a safe directory"
            )
        curation_binding = ReadinessInputBinding.from_value(
            raw["curation_status"], "curation_status"
        )
        suite_binding = ReadinessInputBinding.from_value(
            raw["suite_manifest"], "suite_manifest"
        )
        curation_status = curation_binding.path
        suite_manifest = suite_binding.path
        if not curation_status.parent.is_dir() or not suite_manifest.parent.is_dir():
            raise BootstrapSpecPublicationError(
                "curation and suite manifest parents must already exist"
            )
        if not suite_manifest.is_file() and suite_binding.sha256 is not None:
            raise BootstrapSpecPublicationError(
                "a missing suite manifest may not have a bound hash"
            )
        if suite_binding.sha256 is not None and (
            not curation_status.is_file() or curation_binding.sha256 is None
        ):
            raise BootstrapSpecPublicationError(
                "a published suite requires a hash-bound curation status"
            )
        candidate_inbox = _existing_directory(raw["candidate_inbox"], "candidate_inbox")
        activation_destination = _existing_directory(
            raw["activation_destination"], "activation_destination"
        )
        if _paths_overlap(repository, activation_destination):
            raise BootstrapSpecPublicationError(
                "activation_destination must be outside the deployed source checkout"
            )

        raw_files = raw["files"]
        if not isinstance(raw_files, Mapping):
            raise BootstrapSpecPublicationError("files must be an object")
        _exact_keys(raw_files, set(_REQUIRED_FILES), "files")
        files = {
            name: InputBinding.from_value(value, f"files.{name}")
            for name, value in raw_files.items()
        }
        _validate_named_service_spec(
            files["suite_registry_spec"],
            role="suite registry specification",
            contract="risk-score-evaluation-suite-registry-spec-v1",
        )
        _validate_named_service_spec(
            files["suite_rotation_spec"],
            role="suite rotation service specification",
            contract="risk-score-suite-rotation-service-spec-v1",
        )

        raw_models = raw["models"]
        if not isinstance(raw_models, list) or not raw_models:
            raise BootstrapSpecPublicationError("models must be a nonempty array")
        models = tuple(
            InputBinding.from_value(value, f"models[{index}]")
            for index, value in enumerate(raw_models)
        )
        if [str(model.path) for model in models] != sorted(
            str(model.path) for model in models
        ):
            raise BootstrapSpecPublicationError("models must be sorted by path")
        if len({model.path for model in models}) != len(models) or len(
            {model.sha256 for model in models}
        ) != len(models):
            raise BootstrapSpecPublicationError(
                "models must have unique paths and content hashes"
            )
        original = files["original_model"]
        if not any(model.path == original.path for model in models):
            raise BootstrapSpecPublicationError(
                "models must include files.original_model"
            )

        raw_extra = raw["extra_inputs"]
        if not isinstance(raw_extra, list):
            raise BootstrapSpecPublicationError("extra_inputs must be an array")
        extra = tuple(
            InputBinding.from_value(value, f"extra_inputs[{index}]")
            for index, value in enumerate(raw_extra)
        )
        if [str(item.path) for item in extra] != sorted(
            str(item.path) for item in extra
        ) or len({item.path for item in extra}) != len(extra):
            raise BootstrapSpecPublicationError(
                "extra_inputs must have unique paths sorted by path"
            )
        fixed_binding_paths = _binding_map(
            [*files.values(), *models, *extra], "publisher inputs"
        )
        mutable_aliases = sorted(
            str(path)
            for path in [source, *fixed_binding_paths]
            if path == state_root or _strictly_within(path, state_root)
        )
        if mutable_aliases:
            raise BootstrapSpecPublicationError(
                "publisher and fixed inputs must be outside mutable state_root: "
                f"{mutable_aliases}"
            )
        readiness_paths = {curation_status, suite_manifest}
        overlap = sorted(
            (str(path) for path in readiness_paths.intersection(fixed_binding_paths))
        )
        if overlap:
            raise BootstrapSpecPublicationError(
                f"readiness paths may not be duplicated in fixed inputs: {overlap}"
            )

        raw_templates = raw["gate_argv_templates"]
        if not isinstance(raw_templates, Mapping):
            raise BootstrapSpecPublicationError("gate_argv_templates must be an object")
        _exact_keys(raw_templates, set(GATE_ORDER), "gate_argv_templates")
        templates = {
            gate_id: _validate_argv(
                raw_templates[gate_id], f"gate_argv_templates.{gate_id}"
            )
            for gate_id in GATE_ORDER
        }

        raw_commands = raw["runtime_commands"]
        if not isinstance(raw_commands, Mapping):
            raise BootstrapSpecPublicationError("runtime_commands must be an object")
        _exact_keys(raw_commands, set(_RUNTIME_COMMANDS), "runtime_commands")
        runtime_commands = {
            name: _validate_argv(raw_commands[name], f"runtime_commands.{name}")
            for name in _RUNTIME_COMMANDS
        }
        bound_paths = fixed_binding_paths
        for name, argv in runtime_commands.items():
            executable = _absolute_path(argv[0], f"runtime_commands.{name}[0]")
            if (
                executable.is_symlink()
                or not executable.is_file()
                or executable not in bound_paths
            ):
                raise BootstrapSpecPublicationError(
                    f"runtime_commands.{name} executable must be hash-bound"
                )
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
                    raise BootstrapSpecPublicationError(
                        f"runtime_commands.{name}[{index}] contains an "
                        "unsupported relative path"
                    )
                if not candidate.is_absolute() or not candidate.exists():
                    continue
                path = _absolute_path(
                    candidate_text,
                    f"runtime_commands.{name}[{index}]",
                )
                if path.is_file() and path not in bound_paths:
                    raise BootstrapSpecPublicationError(
                        f"runtime_commands.{name}[{index}] file must be hash-bound"
                    )
        command_spec_requirements = {
            "cluster_executor": files["cluster_executor_spec"].path,
            "adaptive_training": files["adaptive_training_spec"].path,
            "suite_rotation": files["suite_rotation_spec"].path,
        }
        for name, required_path in command_spec_requirements.items():
            if str(required_path) not in runtime_commands[name]:
                raise BootstrapSpecPublicationError(
                    f"runtime_commands.{name} must include its frozen spec path"
                )

        gpu = raw["gpu"]
        if not isinstance(gpu, Mapping):
            raise BootstrapSpecPublicationError("gpu must be an object")
        _exact_keys(gpu, {"index", "uuid"}, "gpu")
        gpu_index = _required_integer(gpu["index"], "gpu.index", minimum=0)
        gpu_uuid = gpu["uuid"]
        if not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
            raise BootstrapSpecPublicationError("gpu.uuid is malformed")

        outputs = raw["outputs"]
        if not isinstance(outputs, Mapping):
            raise BootstrapSpecPublicationError("outputs must be an object")
        _exact_keys(
            outputs,
            {
                "bootstrap_spec",
                "bootstrap_service_unit",
                "bootstrap_path_unit",
            },
            "outputs",
        )
        bootstrap_spec_path = _future_file(
            outputs["bootstrap_spec"], "outputs.bootstrap_spec"
        )
        service_path = _future_file(
            outputs["bootstrap_service_unit"], "outputs.bootstrap_service_unit"
        )
        path_path = _future_file(
            outputs["bootstrap_path_unit"], "outputs.bootstrap_path_unit"
        )
        if not _strictly_within(bootstrap_spec_path, state_root):
            raise BootstrapSpecPublicationError(
                "bootstrap specification output must be inside state_root"
            )
        if (
            service_path.parent != activation_destination
            or service_path.name != BOOTSTRAP_UNIT_NAME
        ):
            raise BootstrapSpecPublicationError(
                f"bootstrap service output must be "
                f"{activation_destination / BOOTSTRAP_UNIT_NAME}"
            )
        if (
            path_path.parent != activation_destination
            or path_path.name != BOOTSTRAP_PATH_UNIT_NAME
        ):
            raise BootstrapSpecPublicationError(
                f"bootstrap path output must be "
                f"{activation_destination / BOOTSTRAP_PATH_UNIT_NAME}"
            )
        if len({bootstrap_spec_path, service_path, path_path}) != 3:
            raise BootstrapSpecPublicationError("publisher outputs must be unique")

        return cls(
            path=source,
            file_sha256=file_sha256(source),
            identity=identity,
            repository=repository,
            source_revision=revision,
            run_root=run_root,
            state_root=state_root,
            curation_status_binding=curation_binding,
            suite_manifest_binding=suite_binding,
            candidate_inbox=candidate_inbox,
            activation_destination=activation_destination,
            files=files,
            models=models,
            extra_inputs=extra,
            gate_argv_templates=templates,
            runtime_commands=runtime_commands,
            gpu_index=gpu_index,
            expected_gpu_uuid=gpu_uuid,
            actor=_required_string(raw["actor"], "actor"),
            service_user=_required_string(raw["service_user"], "service_user"),
            poll_interval_seconds=_positive_number(
                raw["poll_interval_seconds"], "poll_interval_seconds"
            ),
            minimum_clean_observations=_required_integer(
                raw["minimum_clean_observations"],
                "minimum_clean_observations",
                minimum=2,
            ),
            maximum_candidates=_required_integer(
                raw["maximum_candidates"], "maximum_candidates", minimum=0
            ),
            maximum_active_queue=_required_integer(
                raw["maximum_active_queue"], "maximum_active_queue", minimum=1
            ),
            bootstrap_spec_path=bootstrap_spec_path,
            bootstrap_service_unit_path=service_path,
            bootstrap_path_unit_path=path_path,
        )


def load_publisher_spec(path: Path) -> BootstrapPublisherSpec:
    return BootstrapPublisherSpec.load(path)


def _run_git(
    repository: Path,
    command_runner: CommandRunner,
    *arguments: str,
) -> str:
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
        raise BootstrapSpecPublicationError(
            f"Git {' '.join(arguments)} failed: {message}"
        )
    return completed.stdout.strip()


def _verify_clean_checkout(
    spec: BootstrapPublisherSpec, command_runner: CommandRunner
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
        raise BootstrapSpecPublicationError(
            "source revision must match a clean deployed checkout"
        )


def _module_path(repository: Path, module: str) -> Path:
    prefix = "risk_score."
    if not module.startswith(prefix):
        raise BootstrapSpecPublicationError(f"unsupported Python module: {module}")
    path = (
        repository
        / "python"
        / "risk_score"
        / (module[len(prefix) :].replace(".", "/") + ".py")
    )
    _reject_symlink_ancestors(path, f"module {module}")
    if path.is_symlink() or not path.is_file():
        raise BootstrapSpecPublicationError(
            f"required deployed module is missing: {module}"
        )
    return path


def _template_fields(template: Sequence[str], role: str) -> Tuple[str, ...]:
    formatter = string.Formatter()
    fields: List[str] = []
    for argument in template:
        try:
            pieces = list(formatter.parse(argument))
        except ValueError as exc:
            raise BootstrapSpecPublicationError(
                f"{role} has malformed placeholders: {exc}"
            ) from exc
        for _literal, field, format_spec, conversion in pieces:
            if field is None:
                continue
            if not field or format_spec or conversion:
                raise BootstrapSpecPublicationError(
                    f"{role} placeholders may not use formatting or conversion"
                )
            fields.append(field)
    return tuple(fields)


def _flag_values(argv: Sequence[str], flag: str, role: str) -> Tuple[str, ...]:
    values = []
    for index, argument in enumerate(argv):
        if argument != flag:
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise BootstrapSpecPublicationError(f"{role} has incomplete {flag}")
        values.append(argv[index + 1])
    return tuple(values)


def _one_flag(argv: Sequence[str], flag: str, role: str) -> str:
    values = _flag_values(argv, flag, role)
    if len(values) != 1:
        raise BootstrapSpecPublicationError(f"{role} must contain exactly one {flag}")
    return values[0]


def _bound_input_hash(spec: BootstrapPublisherSpec, path: Path, role: str) -> str:
    bindings = _binding_map(
        [*spec.files.values(), *spec.models, *spec.extra_inputs],
        "publisher inputs",
    )
    digest = bindings.get(path)
    if digest is None:
        raise BootstrapSpecPublicationError(f"{role} must be a hash-bound input")
    return digest


def _validate_named_service_spec(
    binding: InputBinding, *, role: str, contract: str
) -> None:
    value = _load_canonical_object(binding.path, role)
    payload = dict(value)
    supplied = payload.pop("spec_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("contract") != contract
        or supplied != canonical_sha256(payload)
    ):
        raise BootstrapSpecPublicationError(f"{role} contract or self-hash is invalid")


def _validate_spec_driven_gate(
    spec: BootstrapPublisherSpec,
    gate_id: str,
    argv: Sequence[str],
    evidence: Path,
) -> None:
    role = f"{gate_id} argv"
    if len(argv) != 7:
        raise BootstrapSpecPublicationError(
            f"{role} must contain only --spec and --expected-spec-sha256"
        )
    spec_path = _absolute_path(_one_flag(argv, "--spec", role), f"{role} --spec")
    expected_hash = _one_flag(argv, "--expected-spec-sha256", role)
    bound_hash = _bound_input_hash(spec, spec_path, f"{role} --spec")
    if expected_hash != bound_hash:
        raise BootstrapSpecPublicationError(
            f"{role} expected spec hash contradicts its input binding"
        )
    value = _load_canonical_object(spec_path, f"{gate_id} executor specification")
    payload = dict(value)
    supplied_identity = payload.pop("spec_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("contract") != _EXPECTED_EXECUTOR_SPEC_CONTRACTS[gate_id]
        or supplied_identity != canonical_sha256(payload)
    ):
        raise BootstrapSpecPublicationError(
            f"{gate_id} executor specification contract or self-hash is invalid"
        )
    if value.get("evidence_output") != str(evidence):
        raise BootstrapSpecPublicationError(
            f"{gate_id} executor specification must bind evidence_output to {evidence}"
        )


def _validate_promotion_gate(
    spec: BootstrapPublisherSpec,
    gate_id: str,
    argv: Sequence[str],
    evidence: Path,
) -> None:
    role = f"{gate_id} argv"
    if len(argv) != 6:
        raise BootstrapSpecPublicationError(
            f"{role} must contain only its subcommand and --spec binding"
        )
    spec_path = _absolute_path(_one_flag(argv, "--spec", role), f"{role} --spec")
    _bound_input_hash(spec, spec_path, f"{role} --spec")
    value = _load_canonical_object(spec_path, f"{gate_id} executor specification")
    payload = dict(value)
    supplied_identity = payload.pop("spec_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("contract") != "risk-score-autonomy-promotion-drill-spec-v1"
        or supplied_identity != canonical_sha256(payload)
    ):
        raise BootstrapSpecPublicationError(
            f"{gate_id} executor specification contract or self-hash is invalid"
        )
    evidence_root = _absolute_path(
        value.get("evidence_root"), f"{gate_id} executor evidence_root"
    )
    if evidence_root != evidence.parent or evidence.name != f"{gate_id}.json":
        raise BootstrapSpecPublicationError(
            f"{gate_id} executor specification does not bind its gate evidence path"
        )


def _validate_direct_gate_cli(
    spec: BootstrapPublisherSpec,
    gate_id: str,
    argv: Sequence[str],
    evidence: Path,
) -> None:
    role = f"{gate_id} argv"
    allowed_flags = {
        "curation-suite-readiness": {
            "--curation-status",
            "--suite-manifest",
            "--suite-registry-spec",
            "--output",
        },
        "filesystem-rename-fsync": {"--root", "--output"},
        "deployment-hash-validation": {"--manifest", "--output"},
        "candidate-inventory": {"--inbox", "--output"},
        "cuda-model-probes": {
            "--katago",
            "--config",
            "--gpu-index",
            "--expected-gpu-uuid",
            "--model",
            "--model-binding",
            "--output",
        },
        "backlog-bound": {
            "--inbox",
            "--candidate-inbox",
            "--controller-status",
            "--backpressure",
            "--backpressure-status",
            "--maximum-candidates",
            "--maximum-active-queue",
            "--training-target-unit",
            "--output",
        },
    }[gate_id]
    tail = tuple(argv[4:])
    if (
        len(tail) % 2
        or any(tail[index] not in allowed_flags for index in range(0, len(tail), 2))
        or any(
            not tail[index] or tail[index].startswith("--")
            for index in range(1, len(tail), 2)
        )
    ):
        raise BootstrapSpecPublicationError(
            f"{role} contains unsupported or incomplete flags"
        )
    if _one_flag(argv, "--output", role) != str(evidence):
        raise BootstrapSpecPublicationError(
            f"{role} output contradicts the generated evidence path"
        )
    if gate_id == "curation-suite-readiness":
        expected = {
            "--curation-status": str(spec.curation_status),
            "--suite-manifest": str(spec.suite_manifest),
            "--suite-registry-spec": str(spec.files["suite_registry_spec"].path),
        }
    elif gate_id == "filesystem-rename-fsync":
        expected = {"--root": str(spec.run_root)}
    elif gate_id == "deployment-hash-validation":
        expected = {"--manifest": str(spec.files["deployment_manifest"].path)}
    elif gate_id == "candidate-inventory":
        expected = {"--inbox": str(spec.candidate_inbox)}
    elif gate_id == "cuda-model-probes":
        expected = {
            "--katago": str(spec.files["katago_binary"].path),
            "--config": str(spec.files["model_probe_config"].path),
            "--gpu-index": str(spec.gpu_index),
            "--expected-gpu-uuid": spec.expected_gpu_uuid,
        }
        model_values = [
            *_flag_values(argv, "--model", role),
            *_flag_values(argv, "--model-binding", role),
        ]
        expected_models = sorted(
            f"{model.path}={model.sha256}" for model in spec.models
        )
        if sorted(model_values) != expected_models:
            raise BootstrapSpecPublicationError(
                "cuda-model-probes argv must contain every hash-bound model"
            )
    elif gate_id == "backlog-bound":
        inbox_values = [
            *_flag_values(argv, "--inbox", role),
            *_flag_values(argv, "--candidate-inbox", role),
        ]
        if inbox_values != [str(spec.candidate_inbox)]:
            raise BootstrapSpecPublicationError(
                "backlog-bound argv must contain the candidate inbox exactly once"
            )
        expected = {
            "--maximum-candidates": str(spec.maximum_candidates),
            "--maximum-active-queue": str(spec.maximum_active_queue),
        }
        controller_values = _flag_values(argv, "--controller-status", role)
        if len(controller_values) > 1:
            raise BootstrapSpecPublicationError(
                "backlog-bound argv may contain at most one controller status"
            )
        if controller_values:
            path = _absolute_path(
                controller_values[0],
                f"{role} --controller-status",
            )
            _bound_input_hash(spec, path, f"{role} --controller-status")
        for flag_names in (("--backpressure", "--backpressure-status"),):
            values = [
                value
                for flag_name in flag_names
                for value in _flag_values(argv, flag_name, role)
            ]
            if len(values) != 1:
                raise BootstrapSpecPublicationError(
                    f"backlog-bound argv must contain one of {flag_names}"
                )
            path = _absolute_path(values[0], f"{role} {flag_names[0]}")
            _bound_input_hash(spec, path, f"{role} {flag_names[0]}")
        target_values = _flag_values(argv, "--training-target-unit", role)
        if len(target_values) > 1:
            raise BootstrapSpecPublicationError(
                "backlog-bound argv may contain at most one training target unit"
            )
    else:  # pragma: no cover - caller restricts this helper.
        raise AssertionError(f"unexpected direct gate: {gate_id}")
    for flag_name, expected_value in expected.items():
        if _one_flag(argv, flag_name, role) != expected_value:
            raise BootstrapSpecPublicationError(
                f"{role} {flag_name} contradicts the publisher specification"
            )


def _expand_gate_argv(
    spec: BootstrapPublisherSpec,
    gate_id: str,
    evidence: Path,
) -> Tuple[str, ...]:
    files = spec.files
    context = {
        "python": str(files["python_executable"].path),
        "repository": str(spec.repository),
        "run_root": str(spec.run_root),
        "state_root": str(spec.state_root),
        "curation_status": str(spec.curation_status),
        "suite_manifest": str(spec.suite_manifest),
        "candidate_inbox": str(spec.candidate_inbox),
        "deployment_manifest": str(files["deployment_manifest"].path),
        "katago_binary": str(files["katago_binary"].path),
        "model_probe_config": str(files["model_probe_config"].path),
        "evidence": str(evidence),
    }
    template = spec.gate_argv_templates[gate_id]
    fields = _template_fields(template, f"{gate_id} argv template")
    unknown = sorted(set(fields).difference(context))
    if unknown:
        raise BootstrapSpecPublicationError(
            f"{gate_id} argv template has unsupported placeholders: {unknown}"
        )
    direct_gate = gate_id in {
        "curation-suite-readiness",
        "filesystem-rename-fsync",
        "deployment-hash-validation",
        "candidate-inventory",
        "cuda-model-probes",
        "backlog-bound",
    }
    evidence_count = fields.count("evidence")
    if (direct_gate and evidence_count != 1) or (
        not direct_gate and evidence_count not in {0, 1}
    ):
        raise BootstrapSpecPublicationError(
            f"{gate_id} argv template has an invalid {{evidence}} placeholder count"
        )
    if template[0] != "{python}":
        raise BootstrapSpecPublicationError(
            f"{gate_id} argv template must use {{python}} as its executable"
        )
    try:
        argv = tuple(argument.format_map(context) for argument in template)
    except (KeyError, ValueError) as exc:
        raise BootstrapSpecPublicationError(
            f"{gate_id} argv template could not be expanded: {exc}"
        ) from exc
    module = _EXPECTED_GATE_MODULES[gate_id]
    if len(argv) < 3 or argv[1:3] != ("-m", module):
        raise BootstrapSpecPublicationError(
            f"{gate_id} must invoke {module} with Python -m"
        )
    expected_subcommand = _EXPECTED_GATE_SUBCOMMANDS.get(gate_id)
    if expected_subcommand is not None and (
        len(argv) < 4 or argv[3] != expected_subcommand
    ):
        raise BootstrapSpecPublicationError(
            f"{gate_id} must use subcommand {expected_subcommand}"
        )
    if direct_gate:
        _validate_direct_gate_cli(spec, gate_id, argv, evidence)
    elif gate_id in {
        "evaluator-topology-benchmark",
        "trainer-evaluator-lease-drill",
    }:
        if evidence_count != 0:
            raise BootstrapSpecPublicationError(
                f"{gate_id} evidence is bound by its executor specification, "
                "not a CLI output override"
            )
        _validate_spec_driven_gate(spec, gate_id, argv, evidence)
    elif gate_id in {
        "disposable-canary-drill",
        "crash-replay-drill",
        "rollback-before-admission-drill",
        "rollback-after-admission-drill",
        "shadow-controller-replay",
    }:
        if evidence_count != 0:
            raise BootstrapSpecPublicationError(
                f"{gate_id} evidence is bound by its executor specification, "
                "not a CLI output override"
            )
        _validate_promotion_gate(spec, gate_id, argv, evidence)
    return argv


def _validate_curation_status(path: Path) -> None:
    if not path.exists():
        return
    value = _load_canonical_object(path, "curation status")
    body = dict(value)
    supplied = body.pop("status_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("contract") != "risk-score-curation-pipeline-status-v1"
        or supplied != canonical_sha256(body)
    ):
        raise BootstrapSpecPublicationError(
            "curation status contract or self-hash is invalid"
        )


def _validate_authoritative_suite(path: Path) -> None:
    if not path.exists():
        return
    value = _load_canonical_object(path, "authoritative suite manifest")
    body = dict(value)
    supplied = body.pop("manifestPayloadSha256", None)
    if (
        value.get("manifestContract")
        != "risk-score-authoritative-evaluation-manifest-v3"
        or value.get("machineReviewOnly") is not True
        or supplied != canonical_sha256(body)
    ):
        raise BootstrapSpecPublicationError("suite manifest is not authoritative v3")


def _deduplicated_binding_values(
    bindings: Sequence[InputBinding],
) -> List[Mapping[str, str]]:
    by_path = _binding_map(bindings, "bootstrap fixed inputs")
    return [
        {"path": str(path), "sha256": digest}
        for path, digest in sorted(by_path.items(), key=lambda item: str(item[0]))
    ]


def _fixed_bootstrap_bindings(
    spec: BootstrapPublisherSpec,
) -> Tuple[InputBinding, ...]:
    bindings: List[InputBinding] = [
        *spec.files.values(),
        *spec.models,
        *spec.extra_inputs,
        InputBinding(spec.path, spec.file_sha256),
    ]
    for module in sorted(_CONTROL_MODULES):
        path = _module_path(spec.repository, module)
        bindings.append(InputBinding(path, file_sha256(path)))
    runtime_modules = {
        argv[2]
        for argv in spec.runtime_commands.values()
        if len(argv) >= 3 and argv[1] == "-m" and argv[2].startswith("risk_score.")
    }
    for module in sorted(runtime_modules):
        path = _module_path(spec.repository, module)
        bindings.append(InputBinding(path, file_sha256(path)))
    for path in sorted(
        (spec.repository / "python" / "risk_score").glob("*.py"),
        key=str,
    ):
        _reject_symlink_ancestors(path, "deployed risk_score module")
        if path.is_symlink() or not path.is_file():
            raise BootstrapSpecPublicationError(
                f"deployed risk_score module is unsafe: {path}"
            )
        bindings.append(InputBinding(path, file_sha256(path)))
    _binding_map(bindings, "bootstrap fixed inputs")
    return tuple(bindings)


def _gate_requirements(spec: BootstrapPublisherSpec) -> Mapping[str, Mapping[str, Any]]:
    return {
        "curation-suite-readiness": {
            "curation_status": str(spec.curation_status),
            "suite_manifest": str(spec.suite_manifest),
        },
        "filesystem-rename-fsync": {"root": str(spec.run_root)},
        "deployment-hash-validation": {
            "manifest": str(spec.files["deployment_manifest"].path)
        },
        "candidate-inventory": {"inbox": str(spec.candidate_inbox)},
        "cuda-model-probes": {
            "expected_gpu_uuid": spec.expected_gpu_uuid,
            "model_sha256s": sorted(model.sha256 for model in spec.models),
        },
        "evaluator-topology-benchmark": {"choices": [4, 8, 16]},
        "trainer-evaluator-lease-drill": {
            "expected_gpu_uuid": spec.expected_gpu_uuid,
            "minimum_clean_observations": spec.minimum_clean_observations,
        },
        "disposable-canary-drill": {},
        "crash-replay-drill": {},
        "rollback-before-admission-drill": {},
        "rollback-after-admission-drill": {},
        "shadow-controller-replay": {},
        "backlog-bound": {
            "maximum_candidates": spec.maximum_candidates,
            "maximum_active_queue": spec.maximum_active_queue,
        },
    }


def _runtime_argv(spec: BootstrapPublisherSpec, output_dir: Path) -> List[str]:
    files = spec.files
    argv = [
        str(files["python_executable"].path),
        "-m",
        "risk_score.build_live_runtime",
        "--repo",
        str(spec.repository),
        "--run-root",
        str(spec.run_root),
        "--suite-dir",
        str(spec.suite_manifest.parent),
        "--katago-binary",
        str(files["katago_binary"].path),
        "--python-executable",
        str(files["python_executable"].path),
        "--trainer-spec",
        str(files["trainer_spec"].path),
        "--consumer-spec",
        str(files["consumer_spec"].path),
        "--original-model",
        str(files["original_model"].path),
        "--trainer-checkpoint",
        str(files["trainer_checkpoint"].path),
        "--gpu-uuid",
        spec.expected_gpu_uuid,
        "--actor",
        spec.actor,
        "--source-revision",
        spec.source_revision,
        "--evaluator-process-count",
        "{selected_evaluator_processes}",
        "--output-dir",
        str(output_dir),
        "--mutation-enabled",
        "--full-autonomy",
        "--service-user",
        spec.service_user,
    ]
    command_flags = {
        "shuffler": "--shuffler-command-json",
        "exporter": "--exporter-command-json",
        "evaluator": "--evaluator-command-json",
        "cluster_executor": "--cluster-executor-command-json",
        "adaptive_training": "--adaptive-training-command-json",
        "suite_rotation": "--suite-rotation-command-json",
    }
    for name in _RUNTIME_COMMANDS:
        argv.extend(
            [
                command_flags[name],
                canonical_json(list(spec.runtime_commands[name])),
            ]
        )
    argv.extend(
        [
            "--autonomy-policy",
            str(files["autonomy_policy"].path),
            "--cluster-executor-spec",
            str(files["cluster_executor_spec"].path),
            "--adaptive-training-spec",
            str(files["adaptive_training_spec"].path),
            "--suite-rotation-spec",
            str(files["suite_rotation_spec"].path),
        ]
    )
    return argv


def _bootstrap_value(spec: BootstrapPublisherSpec) -> Mapping[str, Any]:
    _validate_curation_status(spec.curation_status)
    _validate_authoritative_suite(spec.suite_manifest)
    fixed_bindings = _fixed_bootstrap_bindings(spec)
    # The lease drill is required to advance the trainer checkpoint. Bind its
    # path in the runtime argv, but do not freeze its pre-drill contents into
    # every gate receipt.
    mutable_gate_paths = {spec.files["trainer_checkpoint"].path}
    fixed_values = _deduplicated_binding_values(
        [
            binding
            for binding in fixed_bindings
            if binding.path not in mutable_gate_paths
        ]
    )
    # Readiness hashes come from the immutable publisher input rather than the
    # current filesystem.  A spec prepared while the suite is absent therefore
    # remains byte-for-byte replayable after curation atomically publishes it.
    readiness_bindings = [
        dict(spec.curation_status_binding.value()),
        dict(spec.suite_manifest_binding.value()),
    ]
    requirements = _gate_requirements(spec)
    promotion_gates = {
        "disposable-canary-drill",
        "crash-replay-drill",
        "rollback-before-admission-drill",
        "rollback-after-admission-drill",
        "shadow-controller-replay",
    }
    gates = []
    for gate_id in GATE_ORDER:
        evidence = spec.state_root / "gate-evidence" / f"{gate_id}.json"
        argv = _expand_gate_argv(spec, gate_id, evidence)
        inputs: List[Mapping[str, Any]] = list(fixed_values)
        if gate_id == GATE_ORDER[0]:
            inputs.extend(readiness_bindings)
        outputs = [evidence]
        if gate_id in promotion_gates:
            outputs.append(evidence.with_name(f"{gate_id}.detail.json"))
        gates.append(
            {
                "id": gate_id,
                "argv": list(argv),
                "evidence": str(evidence),
                "inputs": sorted(inputs, key=lambda item: item["path"]),
                "outputs": [str(path) for path in outputs],
                "requirements": dict(requirements[gate_id]),
            }
        )

    runtime_output = spec.state_root / "generated-runtime"
    activation_receipt = spec.state_root / "activation" / "receipt.json"
    value: Dict[str, Any] = {
        "schema_version": 1,
        "contract": SPEC_CONTRACT,
        "state_root": str(spec.state_root),
        "poll_interval_seconds": spec.poll_interval_seconds,
        "gates": gates,
        "runtime": {
            "argv": _runtime_argv(spec, runtime_output),
            "output_dir": str(runtime_output),
        },
        "activation": {
            "argv": [
                str(spec.files["python_executable"].path),
                "-m",
                "risk_score.service_activation",
                "--spec",
                str(runtime_output / "promotion-services.json"),
                "--destination",
                str(spec.activation_destination),
                "--receipt",
                str(activation_receipt),
                "--apply",
            ],
            "destination": str(spec.activation_destination),
            "receipt": str(activation_receipt),
            "required_units": list(REQUIRED_ACTIVATION_UNITS),
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    return value


def _systemd_quote(value: str) -> str:
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def _systemd_path_value(value: Path) -> str:
    text = str(value)
    if not text.startswith("/") or any(
        character.isspace() or character in {'"', "'", "\\"} for character in text
    ):
        raise BootstrapSpecPublicationError(
            "systemd directive path is not safely representable"
        )
    return text.replace("%", "%%")


def render_bootstrap_path_unit(*, suite_manifest: Path) -> str:
    """Render the suite-publication trigger, independent of the runtime target."""

    manifest = _absolute_path(str(Path(suite_manifest)), "suite manifest trigger")
    if not manifest.parent.is_dir():
        raise BootstrapSpecPublicationError(
            "suite manifest trigger parent must already exist"
        )
    unit = "\n".join(
        [
            "[Unit]",
            "Description=Start KataGo autonomy bootstrap after suite publication",
            "# Existence wakes the service; the readiness gate validates authority.",
            "",
            "[Path]",
            "PathExists=" + _systemd_path_value(manifest),
            f"Unit={BOOTSTRAP_UNIT_NAME}",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )
    if (
        "PartOf=katago-risk-training.target" in unit
        or "WantedBy=katago-risk-training.target" in unit
    ):
        raise BootstrapSpecPublicationError(
            "bootstrap path unit must be independent of the runtime target"
        )
    return unit


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
            raise BootstrapSpecPublicationError(
                f"unsafe publication directory: {target}"
            )
        return
    parent = target.parent
    _ensure_directory(parent)
    try:
        target.mkdir()
    except FileExistsError:
        if target.is_symlink() or not target.is_dir():
            raise BootstrapSpecPublicationError(
                f"unsafe publication directory: {target}"
            )
    _fsync_directory(parent)


def _check_immutable_destination(path: Path, data: bytes, role: str) -> None:
    destination = Path(path)
    _reject_symlink_ancestors(destination, role)
    if os.path.lexists(os.fspath(destination)) and (
        destination.is_symlink()
        or not destination.is_file()
        or destination.read_bytes() != data
    ):
        raise BootstrapSpecPublicationError(
            f"{role} conflicts with existing artifact: {destination}"
        )


def _publish_immutable(path: Path, data: bytes, role: str) -> Mapping[str, str]:
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
        try:
            os.link(os.fspath(temporary), os.fspath(destination))
        except FileExistsError:
            _check_immutable_destination(destination, data, role)
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return {"path": str(destination), "sha256": file_sha256(destination)}


def _render_unpublished_bootstrap_service(
    spec: BootstrapPublisherSpec,
    bootstrap_data: bytes,
) -> bytes:
    """Use the canonical renderer before exposing the final specification.

    The renderer requires an existing spec file.  A same-content staging file
    lets the publisher preflight all three immutable destinations before it
    creates any of them.  The only staging-dependent token in renderer output
    is the quoted ``--spec`` value, which is replaced exactly once.
    """

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".autonomy-bootstrap-spec-render.",
        suffix=".json",
        dir=os.fspath(spec.activation_destination),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(bootstrap_data)
            output.flush()
            os.fsync(output.fileno())
        rendered = render_bootstrap_systemd_unit(
            python_executable=spec.files["python_executable"].path,
            working_directory=spec.repository / "python",
            spec_path=temporary,
            run_root=spec.run_root,
        )
        staging_token = _systemd_quote(str(temporary))
        final_token = _systemd_quote(str(spec.bootstrap_spec_path))
        if rendered.count(staging_token) != 1:
            raise BootstrapSpecPublicationError(
                "bootstrap renderer did not produce one pinned specification path"
            )
        return rendered.replace(staging_token, final_token, 1).encode("utf-8")
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def materialize_bootstrap(
    publisher_spec_path: Path,
    *,
    expected_spec_sha256: Optional[str] = None,
    command_runner: CommandRunner = subprocess.run,
) -> Mapping[str, Any]:
    """Publish the bootstrap spec and its service/path units exactly once."""

    spec = BootstrapPublisherSpec.load(publisher_spec_path)
    if expected_spec_sha256 is not None and expected_spec_sha256 != spec.file_sha256:
        raise BootstrapSpecPublicationError(
            "bootstrap publisher specification installed hash changed"
        )
    _verify_clean_checkout(spec, command_runner)

    bootstrap_value = _bootstrap_value(spec)
    bootstrap_data = (canonical_json(bootstrap_value) + "\n").encode("utf-8")
    path_data = render_bootstrap_path_unit(suite_manifest=spec.suite_manifest).encode(
        "utf-8"
    )
    service_data = _render_unpublished_bootstrap_service(spec, bootstrap_data)
    _verify_clean_checkout(spec, command_runner)

    # Preflight the complete immutable set before creating any output.
    _check_immutable_destination(
        spec.bootstrap_spec_path, bootstrap_data, "bootstrap specification"
    )
    _check_immutable_destination(
        spec.bootstrap_service_unit_path, service_data, "bootstrap service unit"
    )
    _check_immutable_destination(
        spec.bootstrap_path_unit_path, path_data, "bootstrap path unit"
    )

    _ensure_directory(spec.state_root)
    _ensure_directory(spec.state_root / "gate-evidence")
    bootstrap_record = _publish_immutable(
        spec.bootstrap_spec_path, bootstrap_data, "bootstrap specification"
    )
    loaded = BootstrapSpec.load(spec.bootstrap_spec_path)
    installed_service_data = render_bootstrap_systemd_unit(
        python_executable=spec.files["python_executable"].path,
        working_directory=spec.repository / "python",
        spec_path=spec.bootstrap_spec_path,
        run_root=spec.run_root,
    ).encode("utf-8")
    if installed_service_data != service_data:
        raise BootstrapSpecPublicationError(
            "bootstrap service staging render changed after spec publication"
        )
    service_record = _publish_immutable(
        spec.bootstrap_service_unit_path, service_data, "bootstrap service unit"
    )
    path_record = _publish_immutable(
        spec.bootstrap_path_unit_path, path_data, "bootstrap path unit"
    )

    result: Dict[str, Any] = {
        "schema_version": 1,
        "contract": PUBLICATION_CONTRACT,
        "publisher_spec": {
            "path": str(spec.path),
            "sha256": spec.file_sha256,
            "identity": spec.identity,
        },
        "bootstrap_spec": {
            **bootstrap_record,
            "identity": loaded.identity,
        },
        "bootstrap_service_unit": service_record,
        "bootstrap_path_unit": path_record,
        "suite_manifest_present": spec.suite_manifest.is_file(),
        "enable_unit": BOOTSTRAP_PATH_UNIT_NAME,
    }
    result["publication_sha256"] = canonical_sha256(result)
    return result


load_spec = load_publisher_spec
materialize = materialize_bootstrap
publish_bootstrap_spec = materialize_bootstrap
publish = materialize_bootstrap
render_systemd_path_unit = render_bootstrap_path_unit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("materialize",),
        default="materialize",
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256")
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> int:
    args = parse_args(argv)
    try:
        result = materialize_bootstrap(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
            command_runner=command_runner,
        )
    except (
        BootstrapError,
        BootstrapSpecPublicationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
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
