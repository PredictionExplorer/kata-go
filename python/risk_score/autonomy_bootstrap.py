#!/usr/bin/env python3
"""Hash-bound, restart-safe gate runner for automatic promotion activation.

The runner intentionally does not implement the destructive live drills.  Each
drill is an immutable, argv-only command in a canonical bootstrap
specification.  The runner validates the command's machine evidence, publishes
its own hash-bound PASS receipt, and advances an fsync'd journal.  Activation
is reachable only after every receipt and the generated mutation-enabled
runtime have been revalidated.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are Unix.
    fcntl = None  # type: ignore[assignment]

from risk_score.build_live_runtime import verify_deployment_manifest
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_host import (
    HostCommandError,
    atomic_replace_json,
    atomic_write_json,
)
from risk_score.promotion_state import atomic_write_bytes
from risk_score.service_activation import (
    FULL_SERVICE_UNIT_NAMES,
    TARGET_UNIT,
    plan_service_activation,
)

BOOTSTRAP_UNIT_NAME = "katago-risk-autonomy-bootstrap.service"
SPEC_CONTRACT = "risk-score-autonomy-bootstrap-spec-v1"
GATE_EVIDENCE_CONTRACT = "risk-score-autonomy-gate-evidence-v1"
GATE_RECEIPT_CONTRACT = "risk-score-autonomy-gate-receipt-v1"
JOURNAL_CONTRACT = "risk-score-autonomy-bootstrap-journal-v1"
RUNTIME_RECEIPT_CONTRACT = "risk-score-autonomy-runtime-receipt-v1"
ACTIVATION_VERIFICATION_CONTRACT = "risk-score-autonomy-activation-verification-v1"
SAFETY_HALT_CONTRACT = "risk-score-autonomy-safety-halt-v1"
STATUS_CONTRACT = "risk-score-autonomy-bootstrap-status-v1"

GATE_ORDER = (
    "curation-suite-readiness",
    "filesystem-rename-fsync",
    "deployment-hash-validation",
    "candidate-inventory",
    "cuda-model-probes",
    "evaluator-topology-benchmark",
    "trainer-evaluator-lease-drill",
    "disposable-canary-drill",
    "crash-replay-drill",
    "rollback-before-admission-drill",
    "rollback-after-admission-drill",
    "shadow-controller-replay",
    "backlog-bound",
)
TOPOLOGY_CHOICES = (4, 8, 16)
REQUIRED_ACTIVATION_UNITS = tuple(
    sorted({TARGET_UNIT, *FULL_SERVICE_UNIT_NAMES.values()})
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
FailureHook = Callable[[str], None]


class BootstrapError(HostCommandError):
    """A bootstrap specification, evidence, or state contradiction."""


class BootstrapSafetyHalt(BootstrapError):
    """The bootstrap entered a terminal, local safety halt."""


class BootstrapBusy(BootstrapError):
    """Another bootstrap process owns the exclusive state lock."""


class BootstrapInterrupted(RuntimeError):
    """Test/integration hook used to model a process crash without a halt."""


class _GateWaiting(RuntimeError):
    pass


def _exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise BootstrapError(
            f"{role} keys differ from contract; missing={missing}, extra={extra}"
        )


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BootstrapError(f"{role} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BootstrapError(f"{role} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, role: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise BootstrapError(f"{role} must be a {qualifier} number")
    return float(value)


def _ensure_finite_json(value: Any, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BootstrapError(f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise BootstrapError(f"{role} contains a non-string object key")
            _ensure_finite_json(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{role}[{index}]")


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise BootstrapError(f"{role} must be a regular non-symlink file")
    data = source.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{role} root must be an object")
    _ensure_finite_json(value, role)
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise BootstrapError(f"{role} must be canonical newline-terminated JSON")
    return value


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise BootstrapError(f"{role} has a symlinked path component: {current}")
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"{role} must be a nonempty absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or normalized != path:
        raise BootstrapError(f"{role} must be an absolute lexically-normal path")
    _reject_symlink_ancestors(path, role)
    return path


def _strictly_within(path: Path, parent: Path) -> bool:
    if path == parent:
        return False
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_argv(value: Any, role: str) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(part, str)
            or not part
            or "\0" in part
            or "\n" in part
            or "\r" in part
            for part in value
        )
    ):
        raise BootstrapError(f"{role} must be a nonempty argv string array")
    executable = Path(value[0])
    if not executable.is_absolute():
        raise BootstrapError(f"{role} executable must be absolute")
    try:
        resolved_executable = executable.resolve(strict=True)
    except OSError as exc:
        raise BootstrapError(f"{role} executable is missing") from exc
    if not resolved_executable.is_file():
        raise BootstrapError(f"{role} executable must resolve to a regular file")
    return tuple(value)


def _parse_flag_argv(
    argv: Sequence[str],
    *,
    module: str,
    value_flags: set[str],
    switch_flags: set[str],
    required_value_flags: set[str],
    required_switch_flags: set[str],
    role: str,
) -> Mapping[str, Any]:
    if len(argv) < 3 or tuple(argv[1:3]) != ("-m", module):
        raise BootstrapError(f"{role} must invoke {module} with Python -m")
    parsed: Dict[str, Any] = {}
    index = 3
    while index < len(argv):
        flag = argv[index]
        if flag in switch_flags:
            if flag in parsed:
                raise BootstrapError(f"{role} repeats {flag}")
            parsed[flag] = True
            index += 1
            continue
        if flag not in value_flags or index + 1 >= len(argv):
            raise BootstrapError(f"{role} has unsupported or incomplete flag {flag!r}")
        if flag in parsed:
            raise BootstrapError(f"{role} repeats {flag}")
        parsed[flag] = argv[index + 1]
        index += 2
    missing_values = sorted(required_value_flags.difference(parsed))
    missing_switches = sorted(required_switch_flags.difference(parsed))
    if missing_values or missing_switches:
        raise BootstrapError(
            f"{role} is incomplete; missing={missing_values + missing_switches}"
        )
    return parsed


@dataclass(frozen=True)
class FileBinding:
    path: Path
    expected_sha256: Optional[str]

    @classmethod
    def from_value(cls, value: Any, role: str, *, allow_unbound: bool) -> "FileBinding":
        if not isinstance(value, Mapping):
            raise BootstrapError(f"{role} must be an object")
        _exact_keys(value, {"path", "sha256"}, role)
        path = _absolute_path(value["path"], f"{role}.path")
        raw_hash = value["sha256"]
        if raw_hash is None:
            if not allow_unbound:
                raise BootstrapError(f"{role}.sha256 may not be null")
            expected = None
        else:
            expected = _require_sha256(raw_hash, f"{role}.sha256")
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise BootstrapError(f"{role}.path must be a regular file")
            if expected is not None and file_sha256(path) != expected:
                raise BootstrapError(f"{role}.path does not match its bound hash")
        elif expected is not None:
            raise BootstrapError(f"{role}.path is missing")
        return cls(path, expected)

    def spec_value(self) -> Mapping[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.expected_sha256,
        }

    def observe(self, role: str) -> Mapping[str, str]:
        if self.path.is_symlink() or not self.path.is_file():
            raise BootstrapError(f"{role} is missing or unsafe: {self.path}")
        observed = file_sha256(self.path)
        if self.expected_sha256 is not None and observed != self.expected_sha256:
            raise BootstrapError(f"{role} changed: {self.path}")
        return {"path": str(self.path), "sha256": observed}


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    argv: Tuple[str, ...]
    evidence: Path
    inputs: Tuple[FileBinding, ...]
    outputs: Tuple[Path, ...]
    requirements: Mapping[str, Any]

    def spec_value(self) -> Mapping[str, Any]:
        return {
            "id": self.gate_id,
            "argv": list(self.argv),
            "evidence": str(self.evidence),
            "inputs": [binding.spec_value() for binding in self.inputs],
            "outputs": [str(path) for path in self.outputs],
            "requirements": dict(self.requirements),
        }

    @property
    def definition_sha256(self) -> str:
        return canonical_sha256(self.spec_value())


@dataclass(frozen=True)
class RuntimeBuildSpec:
    argv: Tuple[str, ...]
    output_dir: Path


@dataclass(frozen=True)
class ActivationSpec:
    argv: Tuple[str, ...]
    destination: Path
    receipt: Path
    required_units: Tuple[str, ...]


@dataclass(frozen=True)
class BootstrapSpec:
    path: Path
    file_sha256: str
    identity: str
    state_root: Path
    poll_interval_seconds: float
    gates: Tuple[GateSpec, ...]
    runtime: RuntimeBuildSpec
    activation: ActivationSpec

    @classmethod
    def load(cls, path: Path) -> "BootstrapSpec":
        source = Path(path)
        if not source.is_absolute():
            source = source.resolve()
        _reject_symlink_ancestors(source, "bootstrap specification")
        raw = _load_canonical_object(source, "bootstrap specification")
        _exact_keys(
            raw,
            {
                "schema_version",
                "contract",
                "state_root",
                "poll_interval_seconds",
                "gates",
                "runtime",
                "activation",
                "spec_sha256",
            },
            "bootstrap specification",
        )
        if raw["schema_version"] != 1 or raw["contract"] != SPEC_CONTRACT:
            raise BootstrapError("bootstrap specification contract is unsupported")
        payload = dict(raw)
        supplied_identity = payload.pop("spec_sha256")
        identity = _require_sha256(supplied_identity, "bootstrap spec identity")
        if identity != canonical_sha256(payload):
            raise BootstrapError("bootstrap specification self-hash is invalid")

        state_root = _absolute_path(raw["state_root"], "state_root")
        if state_root.exists() and not state_root.is_dir():
            raise BootstrapError("state_root must be a directory")
        if not state_root.exists() and not state_root.parent.is_dir():
            raise BootstrapError("state_root parent must already exist")
        interval = _require_number(
            raw["poll_interval_seconds"],
            "poll_interval_seconds",
            positive=True,
        )

        raw_gates = raw["gates"]
        if not isinstance(raw_gates, list) or len(raw_gates) != len(GATE_ORDER):
            raise BootstrapError(
                "bootstrap gates must contain the complete fixed order"
            )
        gates = []
        seen_outputs: set[Path] = set()
        for index, (raw_gate, expected_id) in enumerate(zip(raw_gates, GATE_ORDER)):
            role = f"gates[{index}]"
            if not isinstance(raw_gate, Mapping):
                raise BootstrapError(f"{role} must be an object")
            _exact_keys(
                raw_gate,
                {"id", "argv", "evidence", "inputs", "outputs", "requirements"},
                role,
            )
            if raw_gate["id"] != expected_id:
                raise BootstrapError(
                    f"gate {index} must be {expected_id!r}, found {raw_gate['id']!r}"
                )
            argv = _validate_argv(raw_gate["argv"], f"{role}.argv")
            evidence = _absolute_path(raw_gate["evidence"], f"{role}.evidence")
            if not _strictly_within(evidence, state_root):
                raise BootstrapError(f"{role}.evidence must be inside state_root")
            raw_inputs = raw_gate["inputs"]
            if not isinstance(raw_inputs, list):
                raise BootstrapError(f"{role}.inputs must be an array")
            allow_unbound = expected_id == GATE_ORDER[0]
            inputs = tuple(
                FileBinding.from_value(
                    value, f"{role}.inputs[{input_index}]", allow_unbound=allow_unbound
                )
                for input_index, value in enumerate(raw_inputs)
            )
            if len({binding.path for binding in inputs}) != len(inputs):
                raise BootstrapError(f"{role}.inputs contains duplicate paths")
            raw_outputs = raw_gate["outputs"]
            if not isinstance(raw_outputs, list) or not raw_outputs:
                raise BootstrapError(f"{role}.outputs must be a nonempty array")
            outputs = tuple(
                _absolute_path(value, f"{role}.outputs[{output_index}]")
                for output_index, value in enumerate(raw_outputs)
            )
            if len(set(outputs)) != len(outputs) or evidence not in outputs:
                raise BootstrapError(
                    f"{role}.outputs must be unique and include evidence"
                )
            for output in outputs:
                if not _strictly_within(output, state_root):
                    raise BootstrapError(f"{role} output must be inside state_root")
                if output in seen_outputs:
                    raise BootstrapError("gate output paths must be globally unique")
                seen_outputs.add(output)
                if output.exists() and (output.is_symlink() or not output.is_file()):
                    raise BootstrapError(f"unsafe existing gate output: {output}")
            requirements = raw_gate["requirements"]
            if not isinstance(requirements, Mapping):
                raise BootstrapError(f"{role}.requirements must be an object")
            gate = GateSpec(
                expected_id,
                argv,
                evidence,
                inputs,
                outputs,
                dict(requirements),
            )
            _validate_gate_requirements(gate)
            gates.append(gate)

        runtime_raw = raw["runtime"]
        if not isinstance(runtime_raw, Mapping):
            raise BootstrapError("runtime must be an object")
        _exact_keys(runtime_raw, {"argv", "output_dir"}, "runtime")
        runtime_argv = _validate_argv(runtime_raw["argv"], "runtime.argv")
        output_dir = _absolute_path(runtime_raw["output_dir"], "runtime.output_dir")
        if not _strictly_within(output_dir, state_root):
            raise BootstrapError("runtime.output_dir must be inside state_root")
        if output_dir.exists() and not output_dir.is_dir():
            raise BootstrapError("runtime.output_dir must be a directory")
        runtime_flags = _parse_flag_argv(
            runtime_argv,
            module="risk_score.build_live_runtime",
            value_flags={
                "--repo",
                "--run-root",
                "--suite-dir",
                "--katago-binary",
                "--python-executable",
                "--trainer-spec",
                "--consumer-spec",
                "--original-model",
                "--trainer-checkpoint",
                "--gpu-uuid",
                "--actor",
                "--source-revision",
                "--output-dir",
                "--service-user",
                "--shuffler-command-json",
                "--exporter-command-json",
                "--evaluator-command-json",
                "--evaluator-process-count",
                "--cluster-executor-command-json",
                "--adaptive-training-command-json",
                "--suite-rotation-command-json",
                "--autonomy-policy",
                "--cluster-executor-spec",
                "--adaptive-training-spec",
                "--suite-rotation-spec",
                "--suite-registry-spec",
            },
            switch_flags={"--mutation-enabled", "--full-autonomy"},
            required_value_flags={
                "--repo",
                "--run-root",
                "--suite-dir",
                "--katago-binary",
                "--python-executable",
                "--trainer-spec",
                "--consumer-spec",
                "--original-model",
                "--trainer-checkpoint",
                "--gpu-uuid",
                "--actor",
                "--source-revision",
                "--output-dir",
                "--service-user",
                "--shuffler-command-json",
                "--exporter-command-json",
                "--cluster-executor-command-json",
                "--adaptive-training-command-json",
                "--suite-rotation-command-json",
                "--autonomy-policy",
                "--cluster-executor-spec",
                "--adaptive-training-spec",
            },
            required_switch_flags={"--mutation-enabled", "--full-autonomy"},
            role="runtime.argv",
        )
        suite_rotation_spec = runtime_flags.get("--suite-rotation-spec")
        deprecated_suite_registry_spec = runtime_flags.pop(
            "--suite-registry-spec", None
        )
        if suite_rotation_spec is None and deprecated_suite_registry_spec is None:
            raise BootstrapError(
                "runtime.argv is incomplete; missing=['--suite-rotation-spec']"
            )
        if (
            suite_rotation_spec is not None
            and deprecated_suite_registry_spec is not None
            and suite_rotation_spec != deprecated_suite_registry_spec
        ):
            raise BootstrapError(
                "runtime suite rotation specification contradicts deprecated "
                "--suite-registry-spec alias"
            )
        runtime_flags["--suite-rotation-spec"] = (
            suite_rotation_spec or deprecated_suite_registry_spec
        )
        if runtime_flags["--output-dir"] != str(output_dir):
            raise BootstrapError("runtime argv output directory contradicts the spec")
        if (
            "--evaluator-process-count" in runtime_flags
            and runtime_flags["--evaluator-process-count"]
            != "{selected_evaluator_processes}"
        ):
            raise BootstrapError(
                "runtime evaluator process count must use the "
                "selected-topology placeholder"
            )
        runtime_commands: Dict[str, Sequence[str]] = {}
        for command_flag in (
            "--shuffler-command-json",
            "--exporter-command-json",
            "--evaluator-command-json",
            "--cluster-executor-command-json",
            "--adaptive-training-command-json",
            "--suite-rotation-command-json",
        ):
            if command_flag not in runtime_flags:
                continue
            try:
                command = json.loads(runtime_flags[command_flag])
            except json.JSONDecodeError as exc:
                raise BootstrapError(f"{command_flag} is invalid JSON") from exc
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(part, str) or not part for part in command)
                or not Path(command[0]).is_absolute()
            ):
                raise BootstrapError(f"{command_flag} must encode an absolute argv")
            runtime_commands[command_flag] = command
        for input_flag in (
            "--autonomy-policy",
            "--cluster-executor-spec",
            "--adaptive-training-spec",
            "--suite-rotation-spec",
        ):
            input_path = _absolute_path(runtime_flags[input_flag], input_flag)
            if input_path.is_symlink() or not input_path.is_file():
                raise BootstrapError(f"{input_flag} must name a regular file")
        if (
            runtime_flags["--suite-rotation-spec"]
            not in runtime_commands["--suite-rotation-command-json"]
        ):
            raise BootstrapError(
                "suite rotation command does not bind --suite-rotation-spec"
            )

        activation_raw = raw["activation"]
        if not isinstance(activation_raw, Mapping):
            raise BootstrapError("activation must be an object")
        _exact_keys(
            activation_raw,
            {"argv", "destination", "receipt", "required_units"},
            "activation",
        )
        activation_argv = _validate_argv(activation_raw["argv"], "activation.argv")
        destination = _absolute_path(
            activation_raw["destination"], "activation.destination"
        )
        if not destination.is_dir():
            raise BootstrapError("activation.destination must already be a directory")
        receipt = _absolute_path(activation_raw["receipt"], "activation.receipt")
        if not _strictly_within(receipt, state_root):
            raise BootstrapError("activation.receipt must be inside state_root")
        required_units = activation_raw["required_units"]
        if (
            not isinstance(required_units, list)
            or tuple(required_units) != REQUIRED_ACTIVATION_UNITS
        ):
            raise BootstrapError(
                "activation.required_units must be the complete sorted "
                "automatic inventory"
            )
        activation_flags = _parse_flag_argv(
            activation_argv,
            module="risk_score.service_activation",
            value_flags={"--spec", "--destination", "--receipt"},
            switch_flags={"--apply"},
            required_value_flags={"--spec", "--destination", "--receipt"},
            required_switch_flags={"--apply"},
            role="activation.argv",
        )
        expected_service_spec = output_dir / "promotion-services.json"
        if (
            activation_flags["--spec"] != str(expected_service_spec)
            or activation_flags["--destination"] != str(destination)
            or activation_flags["--receipt"] != str(receipt)
        ):
            raise BootstrapError("activation argv paths contradict the bootstrap spec")

        return cls(
            path=source,
            file_sha256=file_sha256(source),
            identity=identity,
            state_root=state_root,
            poll_interval_seconds=interval,
            gates=tuple(gates),
            runtime=RuntimeBuildSpec(runtime_argv, output_dir),
            activation=ActivationSpec(
                activation_argv,
                destination,
                receipt,
                tuple(required_units),
            ),
        )


def _validate_gate_requirements(gate: GateSpec) -> None:
    requirements = gate.requirements
    gate_id = gate.gate_id
    expected_keys: Mapping[str, set[str]] = {
        "curation-suite-readiness": {"curation_status", "suite_manifest"},
        "filesystem-rename-fsync": {"root"},
        "deployment-hash-validation": {"manifest"},
        "candidate-inventory": {"inbox"},
        "cuda-model-probes": {"expected_gpu_uuid", "model_sha256s"},
        "evaluator-topology-benchmark": {"choices"},
        "trainer-evaluator-lease-drill": {
            "expected_gpu_uuid",
            "minimum_clean_observations",
        },
        "disposable-canary-drill": set(),
        "crash-replay-drill": set(),
        "rollback-before-admission-drill": set(),
        "rollback-after-admission-drill": set(),
        "shadow-controller-replay": set(),
        "backlog-bound": {"maximum_candidates", "maximum_active_queue"},
    }
    _exact_keys(requirements, expected_keys[gate_id], f"{gate_id}.requirements")
    input_paths = {binding.path for binding in gate.inputs}
    if gate_id == "curation-suite-readiness":
        for key in ("curation_status", "suite_manifest"):
            path = _absolute_path(requirements[key], f"{gate_id}.{key}")
            if path not in input_paths:
                raise BootstrapError(f"{gate_id}.{key} must be a declared input")
    elif gate_id == "filesystem-rename-fsync":
        root = _absolute_path(requirements["root"], f"{gate_id}.root")
        if not root.is_dir():
            raise BootstrapError("filesystem gate root must be a directory")
    elif gate_id == "deployment-hash-validation":
        manifest = _absolute_path(requirements["manifest"], f"{gate_id}.manifest")
        if manifest not in input_paths:
            raise BootstrapError("deployment manifest must be a declared input")
    elif gate_id == "candidate-inventory":
        inbox = _absolute_path(requirements["inbox"], f"{gate_id}.inbox")
        if not inbox.is_dir():
            raise BootstrapError("candidate inventory inbox must be a directory")
    elif gate_id == "cuda-model-probes":
        uuid = requirements["expected_gpu_uuid"]
        if not isinstance(uuid, str) or _GPU_UUID_RE.fullmatch(uuid) is None:
            raise BootstrapError("CUDA gate expected_gpu_uuid is malformed")
        hashes = requirements["model_sha256s"]
        if not isinstance(hashes, list) or not hashes or hashes != sorted(set(hashes)):
            raise BootstrapError(
                "CUDA gate model_sha256s must be a sorted unique nonempty array"
            )
        for value in hashes:
            _require_sha256(value, "CUDA gate model hash")
        bound_hashes = {
            binding.expected_sha256
            for binding in gate.inputs
            if binding.expected_sha256 is not None
        }
        if not set(hashes).issubset(bound_hashes):
            raise BootstrapError("every probed model must be a hash-bound gate input")
    elif gate_id == "evaluator-topology-benchmark":
        if requirements["choices"] != list(TOPOLOGY_CHOICES):
            raise BootstrapError("topology choices must be exactly [4,8,16]")
    elif gate_id == "trainer-evaluator-lease-drill":
        uuid = requirements["expected_gpu_uuid"]
        if not isinstance(uuid, str) or _GPU_UUID_RE.fullmatch(uuid) is None:
            raise BootstrapError("lease drill expected_gpu_uuid is malformed")
        _require_int(
            requirements["minimum_clean_observations"],
            "lease minimum clean observations",
            minimum=2,
        )
    elif gate_id == "backlog-bound":
        _require_int(
            requirements["maximum_candidates"],
            "backlog maximum_candidates",
        )
        _require_int(
            requirements["maximum_active_queue"],
            "backlog maximum_active_queue",
            minimum=1,
        )


def load_bootstrap_spec(path: Path) -> BootstrapSpec:
    return BootstrapSpec.load(path)


def publish_gate_evidence(
    path: Path,
    gate_id: str,
    checks: Mapping[str, Any],
    *,
    decision: str = "PASS",
) -> Mapping[str, Any]:
    """Publish canonical generic evidence for a composite live drill."""

    if gate_id not in GATE_ORDER:
        raise BootstrapError(f"unknown bootstrap gate: {gate_id}")
    if decision not in {"PASS", "WAIT", "FAIL"}:
        raise BootstrapError("gate evidence decision is invalid")
    if not isinstance(checks, Mapping):
        raise BootstrapError("gate checks must be an object")
    value: Dict[str, Any] = {
        "schema_version": 1,
        "contract": GATE_EVIDENCE_CONTRACT,
        "gate_id": gate_id,
        "decision": decision,
        "checks": dict(checks),
    }
    _ensure_finite_json(value, "gate evidence")
    value["evidence_sha256"] = canonical_sha256(value)
    atomic_replace_json(Path(path), value)
    return value


def _load_generic_evidence(path: Path, gate_id: str) -> Mapping[str, Any]:
    value = _load_canonical_object(path, f"{gate_id} evidence")
    _exact_keys(
        value,
        {
            "schema_version",
            "contract",
            "gate_id",
            "decision",
            "checks",
            "evidence_sha256",
        },
        f"{gate_id} evidence",
    )
    payload = dict(value)
    supplied = payload.pop("evidence_sha256")
    if (
        value["schema_version"] != 1
        or value["contract"] != GATE_EVIDENCE_CONTRACT
        or value["gate_id"] != gate_id
        or supplied != canonical_sha256(payload)
        or not isinstance(value["checks"], Mapping)
    ):
        raise BootstrapError(f"{gate_id} evidence identity is invalid")
    if value["decision"] == "WAIT" and gate_id == GATE_ORDER[0]:
        checks = value["checks"]
        _exact_keys(
            checks,
            {"curation_complete", "suite_complete"},
            f"{gate_id} WAIT checks",
        )
        if (
            type(checks["curation_complete"]) is not bool
            or type(checks["suite_complete"]) is not bool
            or (checks["curation_complete"] and checks["suite_complete"])
        ):
            raise BootstrapError("readiness WAIT evidence is contradictory")
        raise _GateWaiting("curation or suite artifacts are not complete")
    if value["decision"] != "PASS":
        raise BootstrapError(
            f"{gate_id} evidence decision is {value['decision']!r}, not PASS"
        )
    return value["checks"]


def _validate_self_hash(value: Mapping[str, Any], field: str, role: str) -> None:
    payload = dict(value)
    supplied = payload.pop(field, None)
    if supplied != canonical_sha256(payload):
        raise BootstrapError(f"{role} self-hash is invalid")


def _validate_readiness(gate: GateSpec) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {"curation_complete", "suite_complete"},
        "curation readiness checks",
    )
    if checks != {"curation_complete": True, "suite_complete": True}:
        raise BootstrapError("curation and suite readiness did not PASS")
    status_path = Path(gate.requirements["curation_status"])
    suite_path = Path(gate.requirements["suite_manifest"])
    status = _load_canonical_object(status_path, "curation pipeline status")
    if (
        status.get("contract") != "risk-score-curation-pipeline-status-v1"
        or status.get("state") != "complete"
        or status.get("error") is not None
    ):
        raise BootstrapError("curation pipeline is not canonically complete")
    _validate_self_hash(status, "status_sha256", "curation pipeline status")
    artifacts = status.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(artifacts.get("reviewed_bank"), Mapping)
        or artifacts["reviewed_bank"].get("complete") is not True
        or not isinstance(artifacts.get("suite"), Mapping)
        or artifacts["suite"].get("complete") is not True
        or artifacts["suite"].get("path") != str(suite_path.parent)
    ):
        raise BootstrapError("curation status lacks complete reviewed/suite artifacts")
    suite = _load_canonical_object(suite_path, "evaluation suite manifest")
    payload = dict(suite)
    supplied = payload.pop("manifestPayloadSha256", None)
    if (
        suite.get("manifestContract")
        != "risk-score-authoritative-evaluation-manifest-v3"
        or suite.get("machineReviewOnly") is not True
        or supplied != canonical_sha256(payload)
    ):
        raise BootstrapError("evaluation suite manifest is not authoritative v3")
    return {"verified": True}


def _validate_filesystem(gate: GateSpec) -> Mapping[str, Any]:
    value = _load_canonical_object(gate.evidence, "filesystem durability evidence")
    _exact_keys(
        value,
        {
            "schema_version",
            "contract",
            "root",
            "device",
            "atomic_rename_preserved_inode",
            "directory_fsync_succeeded",
            "payload_sha256",
        },
        "filesystem durability evidence",
    )
    if (
        value["schema_version"] != 1
        or value["contract"] != "risk-score-live-filesystem-test-v1"
        or value["root"] != str(Path(gate.requirements["root"]).resolve())
        or type(value["device"]) is not int
        or value["device"] < 0
        or value["atomic_rename_preserved_inode"] is not True
        or value["directory_fsync_succeeded"] is not True
    ):
        raise BootstrapError("filesystem rename/fsync evidence did not PASS")
    _require_sha256(value["payload_sha256"], "filesystem payload hash")
    return {"verified": True}


def _validate_deployment(gate: GateSpec) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(checks, {"deployment_valid"}, "deployment validation checks")
    if checks["deployment_valid"] is not True:
        raise BootstrapError("deployment validation evidence did not PASS")
    manifest = Path(gate.requirements["manifest"])
    value = verify_deployment_manifest(manifest)
    return {
        "verified": True,
        "manifest_identity": value["manifest_sha256"],
    }


def _validate_candidate_inventory(gate: GateSpec) -> Mapping[str, Any]:
    value = _load_canonical_object(gate.evidence, "candidate inventory")
    _exact_keys(
        value,
        {
            "schema_version",
            "contract",
            "inbox",
            "candidate_count",
            "ignored",
            "candidates",
            "inventory_sha256",
        },
        "candidate inventory",
    )
    _validate_self_hash(value, "inventory_sha256", "candidate inventory")
    candidates = value["candidates"]
    ignored = value["ignored"]
    if (
        value["schema_version"] != 1
        or value["contract"] != "risk-score-live-candidate-inventory-v1"
        or value["inbox"] != str(Path(gate.requirements["inbox"]).resolve())
        or not isinstance(candidates, list)
        or value["candidate_count"] != len(candidates)
        or not isinstance(ignored, list)
        or any(not isinstance(item, str) for item in ignored)
    ):
        raise BootstrapError("candidate inventory evidence is malformed")
    names = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise BootstrapError(f"candidate inventory row {index} is malformed")
        required = {
            "name",
            "path",
            "model_sha256",
            "checkpoint_sha256",
            "directory_manifest_sha256",
            "sample_count",
            "data_count",
            "size_bytes",
        }
        _exact_keys(candidate, required, f"candidate inventory row {index}")
        if not isinstance(candidate["name"], str) or not candidate["name"]:
            raise BootstrapError("candidate inventory contains an empty name")
        names.append(candidate["name"])
        for key in (
            "model_sha256",
            "checkpoint_sha256",
            "directory_manifest_sha256",
        ):
            _require_sha256(candidate[key], f"candidate {key}")
        for key in ("sample_count", "data_count", "size_bytes"):
            _require_int(candidate[key], f"candidate {key}")
    if names != sorted(set(names)):
        raise BootstrapError("candidate inventory names are not sorted and unique")
    return {"verified": True, "candidate_count": len(candidates)}


def _validate_cuda_models(gate: GateSpec) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {"cuda_available", "gpu_exclusive", "gpu_uuid", "models"},
        "CUDA/model checks",
    )
    models = checks["models"]
    if (
        checks["cuda_available"] is not True
        or checks["gpu_exclusive"] is not True
        or checks["gpu_uuid"] != gate.requirements["expected_gpu_uuid"]
        or not isinstance(models, list)
    ):
        raise BootstrapError("CUDA/model probes did not PASS")
    observed_hashes = []
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise BootstrapError(f"model probe {index} is malformed")
        _exact_keys(
            model,
            {"model_sha256", "finite", "deterministic"},
            f"model probe {index}",
        )
        observed_hashes.append(
            _require_sha256(model["model_sha256"], f"model probe {index} hash")
        )
        if model["finite"] is not True or model["deterministic"] is not True:
            raise BootstrapError(f"model probe {index} did not PASS")
    if observed_hashes != gate.requirements["model_sha256s"]:
        raise BootstrapError("model probe inventory does not match the bootstrap spec")
    return {"verified": True, "gpu_uuid": checks["gpu_uuid"]}


def select_evaluator_topology(
    benchmarks: Sequence[Mapping[str, Any]],
    *,
    choices: Sequence[int] = TOPOLOGY_CHOICES,
) -> int:
    """Select fastest deterministic topology, breaking ties toward fewer workers."""

    if tuple(choices) != TOPOLOGY_CHOICES:
        raise BootstrapError("topology choices must be exactly 4, 8, and 16")
    if not isinstance(benchmarks, (list, tuple)) or len(benchmarks) != len(choices):
        raise BootstrapError("topology benchmark inventory is incomplete")
    observed: Dict[int, float] = {}
    output_hashes = set()
    for index, benchmark in enumerate(benchmarks):
        if not isinstance(benchmark, Mapping):
            raise BootstrapError(f"topology benchmark {index} is malformed")
        _exact_keys(
            benchmark,
            {
                "process_count",
                "throughput_per_second",
                "output_sha256",
                "repeat_output_sha256",
            },
            f"topology benchmark {index}",
        )
        count = _require_int(
            benchmark["process_count"],
            f"topology benchmark {index} process_count",
            minimum=1,
        )
        if count in observed:
            raise BootstrapError("topology benchmark process counts are duplicated")
        throughput = _require_number(
            benchmark["throughput_per_second"],
            f"topology benchmark {index} throughput",
            positive=True,
        )
        first = _require_sha256(
            benchmark["output_sha256"],
            f"topology benchmark {index} output hash",
        )
        repeated = _require_sha256(
            benchmark["repeat_output_sha256"],
            f"topology benchmark {index} repeat hash",
        )
        if first != repeated:
            raise BootstrapError("topology benchmark output is not deterministic")
        output_hashes.add(first)
        observed[count] = throughput
    if set(observed) != set(choices):
        raise BootstrapError("topology benchmark must cover exactly 4, 8, and 16")
    if len(output_hashes) != 1:
        raise BootstrapError("deterministic output changed across topology choices")
    return max(observed, key=lambda count: (observed[count], -count))


def _validate_topology(gate: GateSpec) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {"benchmarks", "selected_process_count"},
        "topology benchmark checks",
    )
    if not isinstance(checks["benchmarks"], list):
        raise BootstrapError("topology benchmarks must be an array")
    selected = select_evaluator_topology(checks["benchmarks"])
    if checks["selected_process_count"] != selected:
        raise BootstrapError("reported topology selection is not deterministic")
    return {"verified": True, "selected_evaluator_processes": selected}


def _validate_lease_drill(gate: GateSpec) -> Mapping[str, Any]:
    from risk_score.gpu_lease import SCHEMA_VERSION as GPU_LEASE_SCHEMA_VERSION

    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {
            "gpu_uuid",
            "lease_schema_version",
            "trainer_drained",
            "checkpoint_handoff_verified",
            "evaluator_exclusive",
            "process_overlap_observed",
            "trainer_restored",
            "lease_clean_observations",
            "release_clean_observations",
            "safety_halt",
        },
        "trainer/evaluator lease checks",
    )
    minimum = gate.requirements["minimum_clean_observations"]
    if (
        checks["gpu_uuid"] != gate.requirements["expected_gpu_uuid"]
        or checks["lease_schema_version"] != GPU_LEASE_SCHEMA_VERSION
        or checks["trainer_drained"] is not True
        or checks["checkpoint_handoff_verified"] is not True
        or checks["evaluator_exclusive"] is not True
        or checks["process_overlap_observed"] is not False
        or checks["trainer_restored"] is not True
        or checks["safety_halt"] is not False
        or _require_int(
            checks["lease_clean_observations"],
            "lease clean observation count",
        )
        < minimum
        or _require_int(
            checks["release_clean_observations"],
            "release clean observation count",
        )
        < minimum
    ):
        raise BootstrapError("trainer/evaluator lease drill did not PASS")
    return {"verified": True}


def _require_true_checks(gate: GateSpec, expected_keys: set[str]) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(checks, expected_keys, f"{gate.gate_id} checks")
    if any(checks[key] is not True for key in expected_keys):
        raise BootstrapError(f"{gate.gate_id} did not PASS every required check")
    return {"verified": True}


def _validate_crash_replay(gate: GateSpec) -> Mapping[str, Any]:
    from risk_score.promotion_controller import PROMOTION_FAILURE_STEPS

    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {"boundaries", "production_unchanged"},
        "crash replay checks",
    )
    boundaries = checks["boundaries"]
    if not isinstance(boundaries, list):
        raise BootstrapError("crash replay boundaries must be an array")
    expected = list(PROMOTION_FAILURE_STEPS)
    observed = []
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping):
            raise BootstrapError(f"crash boundary {index} is malformed")
        _exact_keys(
            boundary,
            {"step", "crash_injected", "replay_converged"},
            f"crash boundary {index}",
        )
        observed.append(boundary["step"])
        if (
            boundary["crash_injected"] is not True
            or boundary["replay_converged"] is not True
        ):
            raise BootstrapError(f"crash boundary {index} did not converge")
    if observed != expected or checks["production_unchanged"] is not True:
        raise BootstrapError("crash replay coverage is incomplete")
    return {"verified": True, "boundary_count": len(observed)}


def _validate_rollback_before(gate: GateSpec) -> Mapping[str, Any]:
    return _require_true_checks(
        gate,
        {
            "rollback_requested",
            "refused_without_forensic_flow",
            "staged_data_preserved",
            "champion_unchanged",
            "production_unchanged",
        },
    )


def _validate_rollback_after(gate: GateSpec) -> Mapping[str, Any]:
    return _require_true_checks(
        gate,
        {
            "rollback_complete",
            "champion_restored",
            "checkpoint_restored",
            "admitted_data_quarantined",
            "derived_data_removed",
            "watermarks_restored",
            "production_unchanged",
        },
    )


def _validate_shadow_replay(gate: GateSpec) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {
            "mutation_enabled",
            "first_replay_sha256",
            "second_replay_sha256",
            "event_log_sha256",
            "production_unchanged",
        },
        "shadow replay checks",
    )
    first = _require_sha256(checks["first_replay_sha256"], "first replay hash")
    second = _require_sha256(checks["second_replay_sha256"], "second replay hash")
    _require_sha256(checks["event_log_sha256"], "shadow event log hash")
    if (
        checks["mutation_enabled"] is not False
        or first != second
        or checks["production_unchanged"] is not True
    ):
        raise BootstrapError("shadow controller replay is inconsistent")
    return {"verified": True, "replay_sha256": first}


def _validate_backlog(gate: GateSpec) -> Mapping[str, Any]:
    checks = _load_generic_evidence(gate.evidence, gate.gate_id)
    _exact_keys(
        checks,
        {
            "candidate_count",
            "maximum_candidates",
            "active_queue_depth",
            "maximum_active_queue",
            "backpressure_enforced",
            "evidence_preserved",
        },
        "backlog checks",
    )
    candidates = _require_int(checks["candidate_count"], "backlog candidate count")
    active = _require_int(checks["active_queue_depth"], "active queue depth")
    if (
        checks["maximum_candidates"] != gate.requirements["maximum_candidates"]
        or checks["maximum_active_queue"] != gate.requirements["maximum_active_queue"]
        or candidates > checks["maximum_candidates"]
        or active > checks["maximum_active_queue"]
        or checks["backpressure_enforced"] is not True
        or checks["evidence_preserved"] is not True
    ):
        raise BootstrapError("candidate backlog is not safely bounded")
    return {
        "verified": True,
        "candidate_count": candidates,
        "active_queue_depth": active,
    }


def _validate_gate_evidence(gate: GateSpec) -> Mapping[str, Any]:
    validators: Mapping[str, Callable[[GateSpec], Mapping[str, Any]]] = {
        "curation-suite-readiness": _validate_readiness,
        "filesystem-rename-fsync": _validate_filesystem,
        "deployment-hash-validation": _validate_deployment,
        "candidate-inventory": _validate_candidate_inventory,
        "cuda-model-probes": _validate_cuda_models,
        "evaluator-topology-benchmark": _validate_topology,
        "trainer-evaluator-lease-drill": _validate_lease_drill,
        "disposable-canary-drill": lambda item: _require_true_checks(
            item,
            {
                "canary_passed",
                "fresh_audit_passed",
                "disposable_root_removed",
                "production_unchanged",
            },
        ),
        "crash-replay-drill": _validate_crash_replay,
        "rollback-before-admission-drill": _validate_rollback_before,
        "rollback-after-admission-drill": _validate_rollback_after,
        "shadow-controller-replay": _validate_shadow_replay,
        "backlog-bound": _validate_backlog,
    }
    return validators[gate.gate_id](gate)


class AutonomyBootstrap:
    """Execute and reconcile one immutable bootstrap specification."""

    def __init__(
        self,
        spec: BootstrapSpec | Path,
        *,
        expected_spec_sha256: Optional[str] = None,
        command_runner: CommandRunner = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        failure_hook: Optional[FailureHook] = None,
    ) -> None:
        self.spec = (
            spec if isinstance(spec, BootstrapSpec) else BootstrapSpec.load(spec)
        )
        if expected_spec_sha256 is not None:
            expected = _require_sha256(
                expected_spec_sha256, "expected bootstrap specification hash"
            )
            if self.spec.file_sha256 != expected:
                raise BootstrapError(
                    "bootstrap specification file hash is not the installed hash"
                )
        self.command_runner = command_runner
        self.sleep = sleep
        self.failure_hook = failure_hook
        self._current_stage = "initialization"
        self._activation_invoked = False

    @property
    def lock_path(self) -> Path:
        return self.spec.state_root / "bootstrap.lock"

    @property
    def journal_path(self) -> Path:
        return self.spec.state_root / "journal.json"

    @property
    def safety_halt_path(self) -> Path:
        return self.spec.state_root / "safety-halt.json"

    @property
    def status_path(self) -> Path:
        return self.spec.state_root / "status.json"

    @property
    def runtime_receipt_path(self) -> Path:
        return self.spec.state_root / "receipts" / "runtime.json"

    @property
    def activation_verification_path(self) -> Path:
        return self.spec.state_root / "receipts" / "activation-verification.json"

    def gate_receipt_path(self, index: int) -> Path:
        return (
            self.spec.state_root
            / "receipts"
            / f"{index:02d}-{self.spec.gates[index].gate_id}.json"
        )

    def _spec_binding(self) -> Mapping[str, str]:
        return {
            "path": str(self.spec.path),
            "file_sha256": self.spec.file_sha256,
            "identity": self.spec.identity,
        }

    def _checkpoint(self, stage: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(stage)

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise BootstrapError("POSIX advisory locking is unavailable")
        self.spec.state_root.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(self.spec.state_root, "state_root")
        lock = self.lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise BootstrapBusy(
                        f"another bootstrap owns {self.lock_path}"
                    ) from exc
                raise
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _run_command(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        result = self.command_runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if not hasattr(result, "returncode"):
            raise BootstrapError("command runner returned no process result")
        if result.returncode != 0:
            stderr = getattr(result, "stderr", "")
            raise BootstrapError(
                f"command failed with return code {result.returncode}: "
                f"{str(stderr).strip()}"
            )
        return result

    def _base_journal(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": JOURNAL_CONTRACT,
            "spec": self._spec_binding(),
            "state": "gates",
            "completed_gate_receipts": [],
            "selected_evaluator_processes": None,
            "runtime_receipt": None,
            "activation_verification": None,
            "safety_halt": None,
            "waiting_gate": None,
        }
        value["journal_sha256"] = canonical_sha256(value)
        return value

    def _load_journal(self) -> Dict[str, Any]:
        if not self.journal_path.exists() and not self.journal_path.is_symlink():
            return self._base_journal()
        value = _load_canonical_object(self.journal_path, "bootstrap journal")
        _exact_keys(
            value,
            {
                "schema_version",
                "contract",
                "spec",
                "state",
                "completed_gate_receipts",
                "selected_evaluator_processes",
                "runtime_receipt",
                "activation_verification",
                "safety_halt",
                "waiting_gate",
                "journal_sha256",
            },
            "bootstrap journal",
        )
        payload = dict(value)
        supplied = payload.pop("journal_sha256")
        if (
            value["schema_version"] != 1
            or value["contract"] != JOURNAL_CONTRACT
            or value["spec"] != self._spec_binding()
            or supplied != canonical_sha256(payload)
            or value["state"]
            not in {
                "gates",
                "waiting",
                "runtime",
                "activation",
                "active",
                "safety-halt",
            }
            or not isinstance(value["completed_gate_receipts"], list)
        ):
            raise BootstrapError("bootstrap journal identity is invalid")
        return value

    def _write_journal(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(value)
        payload.pop("journal_sha256", None)
        payload["journal_sha256"] = canonical_sha256(payload)
        atomic_replace_json(self.journal_path, payload)
        atomic_replace_json(
            self.status_path,
            self._status_from_journal(payload),
        )
        return payload

    def _observe_inputs(self, gate: GateSpec) -> list[Mapping[str, str]]:
        return [binding.observe(f"{gate.gate_id} input") for binding in gate.inputs]

    def _observe_outputs(self, gate: GateSpec) -> list[Mapping[str, str]]:
        observed = []
        for path in gate.outputs:
            if path.is_symlink() or not path.is_file():
                raise BootstrapError(
                    f"{gate.gate_id} did not publish required output: {path}"
                )
            observed.append({"path": str(path), "sha256": file_sha256(path)})
        return observed

    def _expected_gate_receipt(
        self,
        index: int,
        *,
        previous_receipt_sha256: Optional[str],
    ) -> Mapping[str, Any]:
        gate = self.spec.gates[index]
        inputs = self._observe_inputs(gate)
        validation = dict(_validate_gate_evidence(gate))
        outputs = self._observe_outputs(gate)
        evidence_record = next(
            record for record in outputs if record["path"] == str(gate.evidence)
        )
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": GATE_RECEIPT_CONTRACT,
            "gate_id": gate.gate_id,
            "gate_index": index,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "gate_definition_sha256": gate.definition_sha256,
            "argv_sha256": canonical_sha256(list(gate.argv)),
            "previous_receipt_sha256": previous_receipt_sha256,
            "inputs": inputs,
            "input_set_sha256": canonical_sha256(inputs),
            "outputs": outputs,
            "output_set_sha256": canonical_sha256(outputs),
            "evidence": evidence_record,
            "validation": validation,
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _load_valid_gate_receipt(
        self,
        index: int,
        *,
        previous_receipt_sha256: Optional[str],
    ) -> Mapping[str, Any]:
        path = self.gate_receipt_path(index)
        existing = _load_canonical_object(
            path, f"{self.spec.gates[index].gate_id} receipt"
        )
        payload = dict(existing)
        supplied = payload.pop("receipt_sha256", None)
        if existing.get(
            "contract"
        ) != GATE_RECEIPT_CONTRACT or supplied != canonical_sha256(payload):
            raise BootstrapError("gate receipt self-hash or contract is invalid")
        expected = self._expected_gate_receipt(
            index, previous_receipt_sha256=previous_receipt_sha256
        )
        if existing != expected:
            raise BootstrapError(
                f"gate receipt no longer binds current inputs/outputs: "
                f"{self.spec.gates[index].gate_id}"
            )
        return existing

    def _append_gate(
        self,
        journal: Mapping[str, Any],
        index: int,
        receipt: Mapping[str, Any],
    ) -> Dict[str, Any]:
        updated = dict(journal)
        completed = list(updated["completed_gate_receipts"])
        completed.append(
            {
                "gate_id": self.spec.gates[index].gate_id,
                "path": str(self.gate_receipt_path(index)),
                "sha256": file_sha256(self.gate_receipt_path(index)),
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
        updated["completed_gate_receipts"] = completed
        selected = receipt["validation"].get("selected_evaluator_processes")
        if selected is not None:
            updated["selected_evaluator_processes"] = selected
        updated["state"] = (
            "runtime" if len(completed) == len(self.spec.gates) else "gates"
        )
        updated["waiting_gate"] = None
        return self._write_journal(updated)

    def _validate_completed_gates(self, journal: Mapping[str, Any]) -> Optional[str]:
        completed = journal["completed_gate_receipts"]
        if len(completed) > len(self.spec.gates):
            raise BootstrapError("journal has too many completed gates")
        previous: Optional[str] = None
        selected: Optional[int] = None
        for index, record in enumerate(completed):
            if not isinstance(record, Mapping):
                raise BootstrapError("journal gate receipt binding is malformed")
            _exact_keys(
                record,
                {"gate_id", "path", "sha256", "receipt_sha256"},
                "journal gate receipt binding",
            )
            path = self.gate_receipt_path(index)
            if (
                record["gate_id"] != self.spec.gates[index].gate_id
                or record["path"] != str(path)
                or path.is_symlink()
                or not path.is_file()
                or file_sha256(path) != record["sha256"]
            ):
                raise BootstrapError("journal gate receipt is missing or changed")
            receipt = self._load_valid_gate_receipt(
                index, previous_receipt_sha256=previous
            )
            if receipt["receipt_sha256"] != record["receipt_sha256"]:
                raise BootstrapError("journal gate receipt identity changed")
            previous = receipt["receipt_sha256"]
            if "selected_evaluator_processes" in receipt["validation"]:
                selected = receipt["validation"]["selected_evaluator_processes"]
        if journal["selected_evaluator_processes"] != selected:
            raise BootstrapError("journal topology selection contradicts gate receipt")
        return previous

    def _execute_gate(
        self,
        index: int,
        previous_receipt_sha256: Optional[str],
    ) -> Mapping[str, Any]:
        gate = self.spec.gates[index]
        self._current_stage = gate.gate_id
        before = None
        if gate.gate_id != GATE_ORDER[0]:
            before = self._observe_inputs(gate)
        self._run_command(gate.argv)
        if gate.gate_id != GATE_ORDER[0]:
            after = self._observe_inputs(gate)
            if before != after:
                raise BootstrapError(f"{gate.gate_id} inputs changed during execution")
        receipt = self._expected_gate_receipt(
            index, previous_receipt_sha256=previous_receipt_sha256
        )
        atomic_write_json(self.gate_receipt_path(index), receipt)
        self._checkpoint(f"after-gate-receipt:{gate.gate_id}")
        return receipt

    def _runtime_paths(self) -> Mapping[str, Path]:
        root = self.spec.runtime.output_dir
        return {
            "promotion_runtime": root / "promotion-runtime.json",
            "gpu_lease_runtime": root / "gpu-lease-runtime.json",
            "deployment_manifest": root / "deployment-manifest.json",
            "service_spec": root / "promotion-services.json",
        }

    def _derive_runtime_result(
        self, selected: int
    ) -> Tuple[Mapping[str, Any], list[Mapping[str, str]]]:
        paths = self._runtime_paths()
        values = {
            name: _load_canonical_object(path, f"generated {name}")
            for name, path in paths.items()
        }
        promotion = values["promotion_runtime"]
        gpu = values["gpu_lease_runtime"]
        service = values["service_spec"]
        if promotion.get("mutationEnabled") is not True:
            raise BootstrapError("generated promotion runtime is not mutation-enabled")
        gpu_evaluator = gpu.get("evaluator")
        if (
            gpu.get("mutationEnabled") is not True
            or not isinstance(gpu_evaluator, Mapping)
            or gpu_evaluator.get("processCount") != selected
        ):
            raise BootstrapError(
                "generated GPU lease topology differs from benchmark selection"
            )
        promotion_hashes = promotion.get("hashes")
        promotion_paths = promotion.get("paths")
        if (
            not isinstance(promotion_hashes, Mapping)
            or not isinstance(promotion_paths, Mapping)
            or promotion_hashes.get("gpuLeaseConfig")
            != file_sha256(paths["gpu_lease_runtime"])
            or promotion_paths.get("gpuLeaseConfig") != str(paths["gpu_lease_runtime"])
        ):
            raise BootstrapError("promotion runtime does not bind the GPU lease config")
        if (
            service.get("schema_version") != 3
            or service.get("contract") != "risk-score-host-services-v3"
            or service.get("mutation_enabled") is not True
            or service.get("full_autonomy") is not True
            or service.get("evaluator_process_count") != selected
        ):
            raise BootstrapError("generated service specification is not full autonomy")
        plan = plan_service_activation(
            spec_path=paths["service_spec"],
            destination=self.spec.activation.destination,
        )
        if tuple(plan["unit_inventory"]) != self.spec.activation.required_units:
            raise BootstrapError("generated service unit inventory is incomplete")
        deployment = verify_deployment_manifest(paths["deployment_manifest"])
        deployment_files = deployment.get("files")
        module_record = (
            deployment_files.get("module:autonomy_bootstrap.py")
            if isinstance(deployment_files, Mapping)
            else None
        )
        if not isinstance(module_record, Mapping) or module_record.get(
            "sha256"
        ) != file_sha256(Path(__file__)):
            raise BootstrapError(
                "deployment manifest does not bind autonomy_bootstrap.py"
            )
        unit_records = service.get("systemd_units")
        if not isinstance(unit_records, Mapping):
            raise BootstrapError("generated service spec has no systemd units")
        systemd_units: Dict[str, Mapping[str, str]] = {}
        output_paths = list(paths.values())
        for name, record in sorted(unit_records.items()):
            if not isinstance(record, Mapping):
                raise BootstrapError(f"generated systemd unit {name} is malformed")
            path = Path(str(record.get("path", "")))
            if (
                not path.is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or file_sha256(path) != record.get("sha256")
            ):
                raise BootstrapError(f"generated systemd unit {name} changed")
            systemd_units[name] = {
                "path": str(path),
                "sha256": record["sha256"],
            }
            output_paths.append(path)
        result = {
            "promotion_runtime": str(paths["promotion_runtime"]),
            "promotion_runtime_sha256": file_sha256(paths["promotion_runtime"]),
            "gpu_lease_runtime": str(paths["gpu_lease_runtime"]),
            "gpu_lease_runtime_sha256": file_sha256(paths["gpu_lease_runtime"]),
            "deployment_manifest": str(paths["deployment_manifest"]),
            "deployment_manifest_sha256": file_sha256(paths["deployment_manifest"]),
            "service_spec": str(paths["service_spec"]),
            "service_spec_sha256": file_sha256(paths["service_spec"]),
            "systemd_units": systemd_units,
            "mutation_enabled": True,
            "full_autonomy": True,
            "evaluator_process_count": selected,
        }
        unique_paths = sorted(set(output_paths), key=str)
        outputs = [
            {"path": str(path), "sha256": file_sha256(path)} for path in unique_paths
        ]
        return result, outputs

    def _parse_canonical_stdout(
        self, result: subprocess.CompletedProcess[str], role: str
    ) -> Mapping[str, Any]:
        stdout = getattr(result, "stdout", "")
        if not isinstance(stdout, str):
            raise BootstrapError(f"{role} stdout must be text")
        try:
            value = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise BootstrapError(f"{role} did not print canonical JSON") from exc
        if not isinstance(value, dict) or stdout != canonical_json(value) + "\n":
            raise BootstrapError(f"{role} stdout is not canonical JSON")
        return value

    def _expected_runtime_receipt(self, selected: int) -> Mapping[str, Any]:
        result, outputs = self._derive_runtime_result(selected)
        argv = self._expanded_runtime_argv(selected)
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": RUNTIME_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "selected_evaluator_processes": selected,
            "argv_template_sha256": canonical_sha256(list(self.spec.runtime.argv)),
            "argv": list(argv),
            "argv_sha256": canonical_sha256(list(argv)),
            "result": result,
            "result_sha256": canonical_sha256(result),
            "outputs": outputs,
            "output_set_sha256": canonical_sha256(outputs),
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _load_valid_runtime_receipt(self, selected: int) -> Mapping[str, Any]:
        existing = _load_canonical_object(self.runtime_receipt_path, "runtime receipt")
        payload = dict(existing)
        supplied = payload.pop("receipt_sha256", None)
        if (
            existing.get("contract") != RUNTIME_RECEIPT_CONTRACT
            or supplied != canonical_sha256(payload)
            or existing != self._expected_runtime_receipt(selected)
        ):
            raise BootstrapError("runtime receipt or generated artifacts changed")
        return existing

    def _runtime_artifacts_complete(self) -> bool:
        return all(
            path.is_file() and not path.is_symlink()
            for path in self._runtime_paths().values()
        )

    def _expanded_runtime_argv(self, selected: int) -> Tuple[str, ...]:
        if selected not in TOPOLOGY_CHOICES:
            raise BootstrapError("runtime topology selection is invalid")
        return tuple(
            part.replace("{selected_evaluator_processes}", str(selected))
            for part in self.spec.runtime.argv
        )

    def _make_runtime_service_readable(self) -> None:
        root = self.spec.runtime.output_dir
        if root.is_symlink() or not root.is_dir():
            raise BootstrapError("generated runtime root is missing or unsafe")
        for current_text, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            current = Path(current_text)
            if current.is_symlink():
                raise BootstrapError("generated runtime contains a symlinked directory")
            os.chmod(current, 0o755)
            for name in directory_names:
                child = current / name
                if child.is_symlink() or not child.is_dir():
                    raise BootstrapError(
                        f"generated runtime directory is unsafe: {child}"
                    )
            for name in file_names:
                child = current / name
                if child.is_symlink() or not child.is_file():
                    raise BootstrapError(f"generated runtime file is unsafe: {child}")
                os.chmod(child, 0o644)

    def _prepare_runtime(self, selected: int) -> Mapping[str, Any]:
        self._current_stage = "mutation-enabled-runtime"
        if self.runtime_receipt_path.exists() or self.runtime_receipt_path.is_symlink():
            return self._load_valid_runtime_receipt(selected)
        if not self._runtime_artifacts_complete():
            result = self._run_command(self._expanded_runtime_argv(selected))
            reported = self._parse_canonical_stdout(result, "runtime builder")
            derived, _ = self._derive_runtime_result(selected)
            if reported != derived:
                raise BootstrapError(
                    "runtime builder stdout does not match generated artifact hashes"
                )
        self._make_runtime_service_readable()
        receipt = self._expected_runtime_receipt(selected)
        atomic_write_json(self.runtime_receipt_path, receipt)
        self._checkpoint("after-runtime-receipt")
        return receipt

    def _validate_activation_receipt(self) -> Mapping[str, Any]:
        path = self.spec.activation.receipt
        value = _load_canonical_object(path, "service activation receipt")
        payload = dict(value)
        supplied = payload.pop("receipt_sha256", None)
        required = set(self.spec.activation.required_units)
        installed = value.get("installed_units")
        active = value.get("active")
        if (
            value.get("schema_version") != 1
            or value.get("contract") != "risk-score-systemd-activation-receipt-v1"
            or supplied != canonical_sha256(payload)
            or value.get("service_spec_sha256")
            != file_sha256(self._runtime_paths()["service_spec"])
            or value.get("target_unit") != TARGET_UNIT
            or value.get("unit_inventory") != list(self.spec.activation.required_units)
            or not isinstance(installed, Mapping)
            or set(installed) != required
            or not isinstance(active, Mapping)
            or set(active) != required
            or any(status != "active" for status in active.values())
        ):
            raise BootstrapError(
                "activation receipt does not prove every required unit active"
            )
        for unit, record in installed.items():
            destination = self.spec.activation.destination / unit
            if (
                not isinstance(record, Mapping)
                or record.get("path") != str(destination)
                or destination.is_symlink()
                or not destination.is_file()
                or record.get("sha256") != file_sha256(destination)
            ):
                raise BootstrapError(f"activation receipt unit changed: {unit}")
        if not isinstance(value.get("restart_occurred"), bool):
            raise BootstrapError("activation receipt restart field is malformed")
        return value

    def _expected_activation_verification(
        self, runtime_receipt: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        activation = self._validate_activation_receipt()
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": ACTIVATION_VERIFICATION_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
            "activation_receipt": {
                "path": str(self.spec.activation.receipt),
                "sha256": file_sha256(self.spec.activation.receipt),
                "identity": activation["receipt_sha256"],
            },
            "required_units": list(self.spec.activation.required_units),
            "active": dict(activation["active"]),
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _load_valid_activation_verification(
        self, runtime_receipt: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        existing = _load_canonical_object(
            self.activation_verification_path,
            "activation verification receipt",
        )
        payload = dict(existing)
        supplied = payload.pop("receipt_sha256", None)
        if (
            existing.get("contract") != ACTIVATION_VERIFICATION_CONTRACT
            or supplied != canonical_sha256(payload)
            or existing != self._expected_activation_verification(runtime_receipt)
        ):
            raise BootstrapError("activation verification receipt changed")
        return existing

    def _activate(self, runtime_receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        self._current_stage = "service-activation"
        if (
            self.activation_verification_path.exists()
            or self.activation_verification_path.is_symlink()
        ):
            return self._load_valid_activation_verification(runtime_receipt)
        if not (
            self.spec.activation.receipt.exists()
            or self.spec.activation.receipt.is_symlink()
        ):
            self._activation_invoked = True
            result = self._run_command(self.spec.activation.argv)
            self._checkpoint("after-activation-command")
            reported = self._parse_canonical_stdout(result, "service activation")
            verified = self._validate_activation_receipt()
            if reported != verified:
                raise BootstrapError(
                    "activation stdout does not match canonical activation receipt"
                )
        verification = self._expected_activation_verification(runtime_receipt)
        atomic_write_json(self.activation_verification_path, verification)
        self._checkpoint("after-activation-verification")
        return verification

    def _receipt_binding(self, path: Path, identity: str) -> Mapping[str, str]:
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "identity": identity,
        }

    def _validate_runtime_and_activation_journal(
        self, journal: Mapping[str, Any]
    ) -> None:
        selected = journal["selected_evaluator_processes"]
        if selected is None:
            if journal["runtime_receipt"] is not None:
                raise BootstrapError("journal has runtime without topology selection")
            return
        runtime_binding = journal["runtime_receipt"]
        runtime_receipt = None
        if runtime_binding is not None:
            runtime_receipt = self._load_valid_runtime_receipt(selected)
            expected_binding = self._receipt_binding(
                self.runtime_receipt_path,
                runtime_receipt["receipt_sha256"],
            )
            if runtime_binding != expected_binding:
                raise BootstrapError("journal runtime receipt binding changed")
        activation_binding = journal["activation_verification"]
        if activation_binding is not None:
            if runtime_receipt is None:
                raise BootstrapError("journal activation has no runtime receipt")
            activation = self._load_valid_activation_verification(runtime_receipt)
            expected_binding = self._receipt_binding(
                self.activation_verification_path,
                activation["receipt_sha256"],
            )
            if activation_binding != expected_binding:
                raise BootstrapError("journal activation receipt binding changed")

    def _validate_journal_progress(self, journal: Mapping[str, Any]) -> None:
        if journal["state"] == "safety-halt":
            return
        completed = len(journal["completed_gate_receipts"])
        total = len(self.spec.gates)
        runtime_ready = journal["runtime_receipt"] is not None
        activation_ready = journal["activation_verification"] is not None
        if completed < total:
            if runtime_ready or activation_ready:
                raise BootstrapError(
                    "journal reached runtime before every gate completed"
                )
            if journal["state"] not in {"gates", "waiting"}:
                raise BootstrapError("journal gate state is contradictory")
            if (
                journal["state"] == "waiting"
                and journal["waiting_gate"] != self.spec.gates[completed].gate_id
            ) or (journal["state"] == "gates" and journal["waiting_gate"] is not None):
                raise BootstrapError("journal waiting gate is contradictory")
            return
        if completed != total or journal["waiting_gate"] is not None:
            raise BootstrapError("journal completed gate inventory is contradictory")
        if journal["selected_evaluator_processes"] not in TOPOLOGY_CHOICES:
            raise BootstrapError("journal has no valid completed topology")
        expected_state = (
            "active"
            if activation_ready
            else "activation"
            if runtime_ready
            else "runtime"
        )
        if journal["state"] != expected_state:
            raise BootstrapError("journal runtime/activation state is contradictory")

    def _load_valid_safety_halt(self, journal: Mapping[str, Any]) -> Mapping[str, Any]:
        value = _load_canonical_object(self.safety_halt_path, "bootstrap safety halt")
        _exact_keys(
            value,
            {
                "schema_version",
                "contract",
                "state",
                "spec",
                "failed_stage",
                "completed_gates",
                "activation_invoked",
                "error",
                "halt_sha256",
            },
            "bootstrap safety halt",
        )
        payload = dict(value)
        supplied = payload.pop("halt_sha256")
        binding = journal.get("safety_halt")
        if (
            value["schema_version"] != 1
            or value["contract"] != SAFETY_HALT_CONTRACT
            or value["state"] != "safety-halt"
            or value["spec"] != self._spec_binding()
            or supplied != canonical_sha256(payload)
            or binding
            != self._receipt_binding(self.safety_halt_path, value["halt_sha256"])
        ):
            raise BootstrapError("bootstrap safety halt receipt changed")
        return value

    def _advance_locked(self) -> Mapping[str, Any]:
        journal = self._load_journal()
        if journal["state"] == "safety-halt":
            return self._status_from_journal(journal)
        previous = self._validate_completed_gates(journal)
        self._validate_runtime_and_activation_journal(journal)
        self._validate_journal_progress(journal)
        if journal["state"] == "active":
            return self._status_from_journal(journal)

        while len(journal["completed_gate_receipts"]) < len(self.spec.gates):
            index = len(journal["completed_gate_receipts"])
            for future in range(index + 1, len(self.spec.gates)):
                if self.gate_receipt_path(future).exists():
                    raise BootstrapError("gate receipts exist out of order")
            receipt_path = self.gate_receipt_path(index)
            if receipt_path.exists() or receipt_path.is_symlink():
                receipt = self._load_valid_gate_receipt(
                    index, previous_receipt_sha256=previous
                )
            else:
                try:
                    receipt = self._execute_gate(index, previous)
                except _GateWaiting:
                    waiting = dict(journal)
                    waiting["state"] = "waiting"
                    waiting["waiting_gate"] = self.spec.gates[index].gate_id
                    journal = self._write_journal(waiting)
                    return self._status_from_journal(journal)
            previous = receipt["receipt_sha256"]
            journal = self._append_gate(journal, index, receipt)
            self._checkpoint(f"after-gate-journal:{self.spec.gates[index].gate_id}")

        selected = journal["selected_evaluator_processes"]
        if selected not in TOPOLOGY_CHOICES:
            raise BootstrapError("completed gates have no valid topology selection")
        runtime_receipt = self._prepare_runtime(selected)
        expected_runtime_binding = self._receipt_binding(
            self.runtime_receipt_path,
            runtime_receipt["receipt_sha256"],
        )
        if journal["runtime_receipt"] is None:
            updated = dict(journal)
            updated["runtime_receipt"] = expected_runtime_binding
            updated["state"] = "activation"
            journal = self._write_journal(updated)
            self._checkpoint("after-runtime-journal")
        elif journal["runtime_receipt"] != expected_runtime_binding:
            raise BootstrapError("journal runtime receipt is contradictory")

        activation = self._activate(runtime_receipt)
        expected_activation_binding = self._receipt_binding(
            self.activation_verification_path,
            activation["receipt_sha256"],
        )
        if journal["activation_verification"] is None:
            updated = dict(journal)
            updated["activation_verification"] = expected_activation_binding
            updated["state"] = "active"
            journal = self._write_journal(updated)
            self._checkpoint("after-activation-journal")
        elif journal["activation_verification"] != expected_activation_binding:
            raise BootstrapError("journal activation receipt is contradictory")
        return self._status_from_journal(journal)

    def _record_safety_halt(self, exc: Exception) -> None:
        try:
            journal = self._load_journal()
        except Exception:
            journal = self._base_journal()
        if journal.get("state") == "safety-halt":
            return
        completed = [
            record.get("gate_id")
            for record in journal.get("completed_gate_receipts", [])
            if isinstance(record, Mapping)
        ]
        halt: Dict[str, Any] = {
            "schema_version": 1,
            "contract": SAFETY_HALT_CONTRACT,
            "state": "safety-halt",
            "spec": self._spec_binding(),
            "failed_stage": self._current_stage,
            "completed_gates": completed,
            "activation_invoked": self._activation_invoked,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        halt["halt_sha256"] = canonical_sha256(halt)
        if self.safety_halt_path.exists() or self.safety_halt_path.is_symlink():
            existing = _load_canonical_object(
                self.safety_halt_path, "bootstrap safety halt"
            )
            halt = existing
        else:
            atomic_write_json(self.safety_halt_path, halt)
        repaired = dict(journal)
        repaired["state"] = "safety-halt"
        repaired["waiting_gate"] = None
        repaired["safety_halt"] = self._receipt_binding(
            self.safety_halt_path,
            halt["halt_sha256"],
        )
        self._write_journal(repaired)

    def _run_locked(self) -> Mapping[str, Any]:
        try:
            return self._advance_locked()
        except BootstrapInterrupted:
            raise
        except Exception as exc:
            self._record_safety_halt(exc)
            raise BootstrapSafetyHalt(str(exc)) from exc

    def run_once(self) -> Mapping[str, Any]:
        with self._exclusive_lock():
            return self._run_locked()

    once = run_once

    def watch(self, *, poll_interval: Optional[float] = None) -> Mapping[str, Any]:
        interval = (
            self.spec.poll_interval_seconds
            if poll_interval is None
            else _require_number(poll_interval, "watch poll interval", positive=True)
        )
        with self._exclusive_lock():
            while True:
                result = self._run_locked()
                if result["state"] in {"active", "safety-halt"}:
                    return result
                self.sleep(interval)

    def _status_from_journal(self, journal: Mapping[str, Any]) -> Mapping[str, Any]:
        completed = [
            record["gate_id"]
            for record in journal["completed_gate_receipts"]
            if isinstance(record, Mapping) and isinstance(record.get("gate_id"), str)
        ]
        next_gate = (
            self.spec.gates[len(completed)].gate_id
            if len(completed) < len(self.spec.gates)
            else None
        )
        halt = None
        if journal["state"] == "safety-halt":
            halt = self._load_valid_safety_halt(journal)
        value: Dict[str, Any] = {
            "schema_version": 1,
            "contract": STATUS_CONTRACT,
            "spec": self._spec_binding(),
            "state": journal["state"],
            "completed_gates": completed,
            "total_gates": len(self.spec.gates),
            "next_gate": next_gate,
            "waiting_gate": journal["waiting_gate"],
            "selected_evaluator_processes": journal["selected_evaluator_processes"],
            "runtime_ready": journal["runtime_receipt"] is not None,
            "activation_verified": journal["activation_verification"] is not None,
            "safety_halt": halt,
        }
        value["status_sha256"] = canonical_sha256(value)
        return value

    def status(self) -> Mapping[str, Any]:
        journal = self._load_journal()
        if journal["state"] != "safety-halt":
            self._validate_completed_gates(journal)
            self._validate_runtime_and_activation_journal(journal)
            self._validate_journal_progress(journal)
        return self._status_from_journal(journal)


BootstrapRunner = AutonomyBootstrap


def _systemd_quote(value: str) -> str:
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def _systemd_path_value(value: Path) -> str:
    text = str(value)
    if not text.startswith("/") or any(
        character.isspace() or character in {'"', "'", "\\"} for character in text
    ):
        raise BootstrapError("systemd directive path is not safely representable")
    return text.replace("%", "%%")


def render_bootstrap_systemd_unit(
    *,
    python_executable: Path,
    working_directory: Path,
    spec_path: Path,
    run_root: Path,
) -> str:
    python = _absolute_path(str(python_executable), "bootstrap Python executable")
    working = _absolute_path(str(working_directory), "bootstrap working directory")
    spec = _absolute_path(str(spec_path), "bootstrap specification")
    root = _absolute_path(str(run_root), "bootstrap run root")
    if (
        python.is_symlink()
        or not python.is_file()
        or working.is_symlink()
        or not working.is_dir()
        or spec.is_symlink()
        or not spec.is_file()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise BootstrapError(
            "bootstrap unit inputs must be existing non-symlink files/directories"
        )
    BootstrapSpec.load(spec)
    argv = (
        str(python),
        "-m",
        "risk_score.autonomy_bootstrap",
        "watch",
        "--spec",
        str(spec),
        "--expected-spec-sha256",
        file_sha256(spec),
    )
    return "\n".join(
        [
            "[Unit]",
            "Description=KataGo hash-bound autonomy bootstrap",
            "Wants=network-online.target",
            "After=network-online.target",
            "# Do not order this unit before the runtime target: the final",
            "# activation command synchronously restarts that target.",
            "RequiresMountsFor=" + _systemd_path_value(root),
            "StartLimitIntervalSec=600",
            "StartLimitBurst=3",
            "",
            "[Service]",
            "Type=oneshot",
            "WorkingDirectory=" + _systemd_path_value(working),
            "Environment=" + _systemd_quote(f"PYTHONPATH={working}"),
            "ExecStart=" + " ".join(_systemd_quote(item) for item in argv),
            "RemainAfterExit=yes",
            "Restart=on-failure",
            "RestartSec=30",
            "KillSignal=SIGINT",
            "KillMode=control-group",
            "TimeoutStartSec=0",
            "TimeoutStopSec=300",
            "UMask=0077",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )


def publish_bootstrap_systemd_unit(
    path: Path,
    *,
    python_executable: Path,
    working_directory: Path,
    spec_path: Path,
    run_root: Path,
) -> Mapping[str, str]:
    destination = Path(path)
    if (
        not destination.is_absolute()
        or destination.name != BOOTSTRAP_UNIT_NAME
        or destination.is_symlink()
    ):
        raise BootstrapError(
            f"bootstrap unit path must be absolute and named {BOOTSTRAP_UNIT_NAME}"
        )
    unit = render_bootstrap_systemd_unit(
        python_executable=python_executable,
        working_directory=working_directory,
        spec_path=spec_path,
        run_root=run_root,
    ).encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != unit:
            raise BootstrapError("bootstrap unit conflicts with existing artifact")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(destination, unit)
    return {
        "path": str(destination),
        "sha256": file_sha256(destination),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "once", "watch"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256")
    parser.add_argument("--poll-interval", type=float)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    command_runner: CommandRunner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = parse_args(argv)
    try:
        runner = AutonomyBootstrap(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
            command_runner=command_runner,
            sleep=sleep,
        )
        if args.mode == "status":
            result = runner.status()
        elif args.mode == "once":
            result = runner.run_once()
        else:
            result = runner.watch(poll_interval=args.poll_interval)
        print(canonical_json(result))
        if result["state"] == "safety-halt":
            return 2
        return 0 if result["state"] == "active" or args.mode == "status" else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError, BootstrapError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
