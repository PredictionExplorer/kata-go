#!/usr/bin/env python3
"""Privileged, restart-safe evaluation-suite and runtime deployment handshake.

The suite-rotation service can prepare an immutable deployment request, but it
is deliberately unable to mutate the active-suite pointer or install systemd
units.  This module is the separately installed root-side consumer of those
requests.  It fences the old controller, builds and validates a content-
addressed runtime, performs the registry compare-and-swap, and only then
installs the exact validated service inventory.

The deployer unit is intentionally independent from the runtime target.  A
failed target restart must never restart this process into the old runtime
after the active-suite pointer has advanced.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import os
import re
import stat
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
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
except ImportError:  # pragma: no cover - the installed service targets Linux.
    fcntl = None  # type: ignore[assignment]

from risk_score.build_live_runtime import verify_deployment_manifest
from risk_score.service_activation import (
    AUTONOMY_SERVICE_SPEC_CONTRACT,
    FULL_EXPECTED_SERVICE_UNITS,
    FULL_SERVICE_UNIT_NAMES,
    TARGET_UNIT,
    apply_service_activation,
    plan_service_activation,
)
from risk_score.suite_rotation import (
    CONTINUITY_CONTRACT,
    SuiteRotationRegistry,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    load_registry_spec,
    validate_suite_manifest,
)
from risk_score.suite_rotation_service import (
    DEPLOYMENT_REQUEST_CONTRACT,
    SERVICE_SPEC_CONTRACT,
    load_service_spec,
)

SCHEMA_VERSION = 1
DEPLOYER_UNIT_NAME = "katago-risk-autonomy-deployer.service"
SPEC_CONTRACT = "risk-score-autonomy-deployer-spec-v1"
AUTONOMY_DEPLOYER_SPEC_CONTRACT = SPEC_CONTRACT
REQUEST_RECEIPT_CONTRACT = "risk-score-autonomy-deployment-request-receipt-v1"
FENCE_PROOF_CONTRACT = "risk-score-controller-fence-quiescence-proof-v1"
FENCE_RECEIPT_CONTRACT = "risk-score-autonomy-controller-fence-receipt-v1"
RUNTIME_RECEIPT_CONTRACT = "risk-score-autonomy-deployment-runtime-receipt-v1"
POINTER_RECEIPT_CONTRACT = "risk-score-autonomy-suite-pointer-receipt-v1"
ACTIVATION_RECEIPT_CONTRACT = "risk-score-autonomy-service-activation-receipt-v1"
COMPLETION_RECEIPT_CONTRACT = "risk-score-autonomy-deployment-completion-v1"
RETRYABLE_HALT_CONTRACT = "risk-score-autonomy-deployment-retryable-halt-v1"
SAFETY_HALT_CONTRACT = "risk-score-autonomy-deployment-safety-halt-v1"
JOURNAL_CONTRACT = "risk-score-autonomy-deployer-journal-v1"
STATUS_CONTRACT = "risk-score-autonomy-deployer-status-v1"
DEPLOYER_STATUS_CONTRACT = STATUS_CONTRACT
CURRENT_DEPLOYMENT_CONTRACT = "risk-score-autonomy-current-deployment-v1"
ROOT_UNIT_NAME = DEPLOYER_UNIT_NAME

REQUIRED_ACTIVATION_UNITS = tuple(sorted({TARGET_UNIT, *FULL_EXPECTED_SERVICE_UNITS}))
QUIESCENT_ROLES = (
    "controller",
    "selfplay",
    "shuffler",
    "trainer",
    "exporter",
    "evaluator",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,254})$")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SPEC_KEYS = {
    "schema_version",
    "contract",
    "root",
    "request_inbox",
    "registry_spec",
    "current_deployment",
    "activation_destination",
    "controller_fence_argv",
    "activation_argv_template",
    "activation_receipt_root",
    "required_units",
    "source",
    "actor",
    "poll_interval_seconds",
    "spec_sha256",
}
_REQUEST_KEYS = {
    "schema_version",
    "contract",
    "request_id",
    "actor",
    "created_at_utc",
    "service_spec",
    "registry_spec",
    "rotation_request",
    "candidate_suite",
    "continuity",
    "compare_and_swap",
    "proposed_frozen_commands",
    "proposed_frozen_commands_sha256",
    "privilege_boundary",
    "request_sha256",
}
_CAS_KEYS = {
    "expected_active_suite_id",
    "expected_champion_sha256",
    "expected_generation_id",
    "expected_pin_count",
    "require_clean_generation_boundary",
    "boundary_must_follow_continuity",
    "continuity_event_sequence",
}
_COMMAND_KEYS = {
    "argv",
    "argv_sha256",
    "frozen_inputs",
    "frozen_inputs_sha256",
}
_SERVICE_COMMAND_KEYS = _COMMAND_KEYS | {"executable_sha256"}
_RUNTIME_INPUT_KEYS = {
    "candidate_suite_id",
    "candidate_version_sha256",
    "candidate_manifest_path",
    "candidate_manifest_sha256",
    "candidate_manifest_identity",
    "materialization_receipt_sha256",
}
_SERVICE_INPUT_KEYS = {
    "service_spec_path",
    "service_spec_sha256",
    "service_spec_identity",
    "registry_spec_path",
    "registry_spec_sha256",
}
_ACTIVATION_TEMPLATE_FIELDS = frozenset(
    {"service_spec", "activation_destination", "activation_receipt"}
)
_RECEIPT_NAMES = (
    "request",
    "fence",
    "runtime",
    "pointer",
    "activation",
    "completion",
)

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
SystemctlRunner = Callable[..., subprocess.CompletedProcess[str]]


class DeployerError(RuntimeError):
    """Base error for a malformed or contradictory deployment handshake."""


class DeployerSpecError(DeployerError, ValueError):
    """The immutable deployer specification is invalid or changed."""


class DeploymentRequestError(DeployerError, ValueError):
    """A privileged deployment request is malformed, stale, or tampered."""


class DeploymentStateError(DeployerError):
    """Durable local state contradicts immutable inputs or the registry."""


class DeployerBusyError(DeployerError):
    """Another deployer process owns the root writer lock."""


class DeployerSafetyHalt(DeployerError):
    """The fenced deployment entered a non-retryable local safety halt."""


class ActivationRetryableHalt(DeployerError):
    """Pointer CAS succeeded but the exact new runtime still needs applying."""


class DeployerInterrupted(RuntimeError):
    """A test/integration crash checkpoint; no durable halt is written."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentRequestError(f"{role} must be an object")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise DeploymentRequestError(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _spec_exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeployerSpecError(f"{role} must be an object")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise DeployerSpecError(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, role: str, *, spec: bool = False) -> str:
    error = DeployerSpecError if spec else DeploymentRequestError
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise error(f"{role} must be a lowercase SHA-256")
    return value


def _require_id(value: Any, role: str, *, spec: bool = False) -> str:
    error = DeployerSpecError if spec else DeploymentRequestError
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise error(f"{role} must be a safe nonempty identifier")
    return value


def _positive_number(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise DeployerSpecError(f"{role} must be a positive finite number")
    return float(value)


def _reject_symlink_ancestors(
    path: Path, role: str, *, error_type: type[DeployerError] = DeployerSpecError
) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise error_type(f"{role} has a symlinked path component: {current}")
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(
    value: Any,
    role: str,
    *,
    error_type: type[DeployerError] = DeployerSpecError,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise error_type(f"{role} must be a nonempty absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise error_type(f"{role} must be absolute and lexically normalized")
    _reject_symlink_ancestors(path, role, error_type=error_type)
    return path


def _required_file(
    value: Any,
    role: str,
    expected_hash: Optional[str] = None,
    *,
    error_type: type[DeployerError] = DeployerSpecError,
) -> Path:
    path = _absolute_path(value, role, error_type=error_type)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise error_type(f"{role} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise error_type(f"{role} must be a regular non-symlink file")
    if expected_hash is not None and file_sha256(path) != expected_hash:
        raise error_type(f"{role} hash changed")
    return path


def _required_directory(
    value: Any,
    role: str,
    *,
    error_type: type[DeployerError] = DeployerSpecError,
) -> Path:
    path = _absolute_path(value, role, error_type=error_type)
    if path.is_symlink() or not path.is_dir():
        raise error_type(f"{role} must be an existing non-symlink directory")
    return path


def _future_directory(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise DeployerSpecError(f"{role} must be a non-symlink directory")
    elif path.parent.is_symlink() or not path.parent.is_dir():
        raise DeployerSpecError(f"{role} parent must already be a safe directory")
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


def _load_canonical_object(
    path: Path,
    role: str,
    *,
    error_type: type[DeployerError] = DeploymentStateError,
) -> Dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError as exc:
        raise error_type(f"{role} is missing: {source}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise error_type(f"{role} must be a regular non-symlink file")
    if metadata.st_size > _MAX_JSON_BYTES:
        raise error_type(f"{role} exceeds the size limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise error_type(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise error_type(f"{role} root must be an object")
    if data != canonical_json_bytes(value) + b"\n":
        raise error_type(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_self_hashed(
    path: Path,
    *,
    role: str,
    contract: str,
    hash_field: str,
    exact_keys: Optional[set[str]] = None,
    error_type: type[DeployerError] = DeploymentRequestError,
) -> Tuple[Dict[str, Any], str]:
    value = _load_canonical_object(path, role, error_type=error_type)
    if exact_keys is not None and set(value) != exact_keys:
        missing = sorted(exact_keys.difference(value))
        extra = sorted(set(value).difference(exact_keys))
        raise error_type(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )
    payload = dict(value)
    identity = payload.pop(hash_field, None)
    if not isinstance(identity, str) or _SHA256_RE.fullmatch(identity) is None:
        raise error_type(f"{role} {hash_field} must be a lowercase SHA-256")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != contract
        or identity != canonical_sha256(payload)
    ):
        raise error_type(f"{role} contract or self-hash is invalid")
    return value, identity


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
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise DeploymentStateError(f"unsafe state directory: {target}")
        return
    target.mkdir(parents=True, exist_ok=False)
    _fsync_directory(target.parent)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    _ensure_directory(target.parent)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise DeploymentStateError(f"unsafe mutable projection path: {target}")
    data = canonical_json_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(os.fspath(temporary), os.fspath(target))
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    _ensure_directory(target.parent)
    data = canonical_json_bytes(dict(value)) + b"\n"
    if os.path.lexists(os.fspath(target)):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise DeploymentStateError(f"immutable artifact conflicts: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(os.fspath(temporary), os.fspath(target))
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != data
            ):
                raise DeploymentStateError(f"immutable artifact conflicts: {target}")
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _validate_argv(value: Any, role: str, *, template: bool = False) -> Tuple[str, ...]:
    error = DeployerSpecError if template else DeploymentRequestError
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
        raise error(f"{role} must be a nonempty argv string array")
    if "{" not in value[0]:
        executable = _required_file(value[0], f"{role} executable", error_type=error)
        if executable != Path(value[0]):
            raise error(f"{role} executable path changed")
    return tuple(value)


def _validate_activation_template(value: Any) -> Tuple[str, ...]:
    argv = _validate_argv(value, "activation argv template", template=True)
    formatter = string.Formatter()
    fields = set()
    for part in argv:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise DeployerSpecError(
                "activation argv template formatting is invalid"
            ) from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                field_name not in _ACTIVATION_TEMPLATE_FIELDS
                or format_spec
                or conversion is not None
            ):
                raise DeployerSpecError(
                    f"activation argv template uses unsupported placeholder "
                    f"{field_name!r}"
                )
            fields.add(field_name)
    if fields != set(_ACTIVATION_TEMPLATE_FIELDS):
        raise DeployerSpecError(
            "activation argv template must bind service_spec, "
            "activation_destination, and activation_receipt"
        )
    return argv


def _parse_module_flags(
    argv: Sequence[str],
    *,
    module: str,
    value_flags: set[str],
    switch_flags: set[str],
    required_values: set[str],
    required_switches: set[str],
    role: str,
) -> Mapping[str, Any]:
    if len(argv) < 3 or tuple(argv[1:3]) != ("-m", module):
        raise DeploymentRequestError(f"{role} must invoke {module} with Python -m")
    parsed: Dict[str, Any] = {}
    index = 3
    while index < len(argv):
        flag = argv[index]
        if flag in switch_flags:
            if flag in parsed:
                raise DeploymentRequestError(f"{role} repeats {flag}")
            parsed[flag] = True
            index += 1
            continue
        if flag not in value_flags or index + 1 >= len(argv):
            raise DeploymentRequestError(
                f"{role} has unsupported or incomplete flag {flag!r}"
            )
        if flag in parsed:
            raise DeploymentRequestError(f"{role} repeats {flag}")
        parsed[flag] = argv[index + 1]
        index += 2
    missing = sorted(
        required_values.difference(parsed) | required_switches.difference(parsed)
    )
    if missing:
        raise DeploymentRequestError(f"{role} is incomplete; missing={missing}")
    return MappingProxyType(parsed)


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str
    identity: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        value = {"path": str(self.path), "sha256": self.sha256}
        if self.identity is not None:
            value["identity"] = self.identity
        return value


@dataclass(frozen=True)
class DeployerSpec:
    path: Path
    file_sha256: str
    identity: str
    root: Path
    request_inbox: Path
    registry_spec: FileBinding
    current_deployment: Path
    activation_destination: Path
    controller_fence_argv: Tuple[str, ...]
    activation_argv_template: Tuple[str, ...]
    activation_receipt_root: Path
    required_units: Tuple[str, ...]
    source: FileBinding
    actor: str
    poll_interval_seconds: float
    raw: Mapping[str, Any]


def _binding_from_spec(value: Any, role: str) -> FileBinding:
    checked = _spec_exact_keys(value, {"path", "sha256"}, role)
    expected = _require_sha256(checked["sha256"], f"{role} hash", spec=True)
    path = _required_file(checked["path"], role, expected)
    return FileBinding(path, expected)


def _expanded_activation_argv(
    template: Sequence[str],
    *,
    service_spec: Path,
    destination: Path,
    receipt: Path,
) -> Tuple[str, ...]:
    values = {
        "service_spec": str(service_spec),
        "activation_destination": str(destination),
        "activation_receipt": str(receipt),
    }
    try:
        argv = tuple(part.format_map(values) for part in template)
    except (KeyError, ValueError) as exc:  # pragma: no cover - load validates.
        raise DeploymentStateError(f"cannot expand activation argv: {exc}") from exc
    if any(not part or "\x00" in part for part in argv):
        raise DeploymentStateError("activation argv expanded unsafely")
    return argv


def _validate_activation_argv(
    argv: Sequence[str],
    *,
    service_spec: Path,
    destination: Path,
    receipt: Path,
) -> None:
    try:
        parsed = _parse_module_flags(
            argv,
            module="risk_score.service_activation",
            value_flags={"--spec", "--destination", "--receipt"},
            switch_flags={"--apply"},
            required_values={"--spec", "--destination", "--receipt"},
            required_switches={"--apply"},
            role="activation argv",
        )
    except DeploymentRequestError as exc:
        raise DeployerSpecError(str(exc)) from exc
    if (
        parsed["--spec"] != str(service_spec)
        or parsed["--destination"] != str(destination)
        or parsed["--receipt"] != str(receipt)
    ):
        raise DeployerSpecError("activation argv paths contradict the deployer spec")


def load_deployer_spec(
    path: Path, *, expected_spec_sha256: Optional[str] = None
) -> DeployerSpec:
    """Load a strict canonical, self-hashed privileged deployer specification."""

    source = Path(path).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    _reject_symlink_ancestors(source, "deployer specification")
    raw = _load_canonical_object(
        source, "deployer specification", error_type=DeployerSpecError
    )
    _spec_exact_keys(raw, _SPEC_KEYS, "deployer specification")
    if raw["schema_version"] != SCHEMA_VERSION or raw["contract"] != SPEC_CONTRACT:
        raise DeployerSpecError("deployer specification contract is unsupported")
    payload = dict(raw)
    identity = _require_sha256(
        payload.pop("spec_sha256", None), "deployer specification identity", spec=True
    )
    if identity != canonical_sha256(payload):
        raise DeployerSpecError("deployer specification self-hash is invalid")
    observed_file_hash = file_sha256(source)
    if expected_spec_sha256 is not None:
        expected = _require_sha256(
            expected_spec_sha256,
            "expected deployer specification hash",
            spec=True,
        )
        if expected not in {observed_file_hash, identity}:
            raise DeployerSpecError("deployer specification hash is not expected")

    root = _future_directory(raw["root"], "deployer root")
    inbox = _required_directory(raw["request_inbox"], "deployment request inbox")
    registry = _binding_from_spec(raw["registry_spec"], "registry specification")
    source_binding = _binding_from_spec(raw["source"], "deployer source")
    current = _absolute_path(raw["current_deployment"], "current deployment projection")
    activation_destination = _required_directory(
        raw["activation_destination"], "service activation destination"
    )
    receipt_root = _absolute_path(
        raw["activation_receipt_root"], "activation receipt root"
    )
    if receipt_root.exists() or receipt_root.is_symlink():
        if receipt_root.is_symlink() or not receipt_root.is_dir():
            raise DeployerSpecError(
                "activation receipt root must be a non-symlink directory"
            )
    if not _strictly_within(current, root):
        raise DeployerSpecError(
            "current deployment projection must be strictly beneath deployer root"
        )
    if not _strictly_within(receipt_root, root):
        raise DeployerSpecError(
            "activation receipt root must be strictly beneath deployer root"
        )
    if _paths_overlap(root, inbox) or _paths_overlap(root, activation_destination):
        raise DeployerSpecError(
            "deployer state may not overlap the request inbox or systemd destination"
        )
    if _paths_overlap(receipt_root, current):
        raise DeployerSpecError(
            "activation receipt root overlaps current deployment projection"
        )
    controller = _validate_argv(
        raw["controller_fence_argv"], "controller fence argv", template=True
    )
    activation = _validate_activation_template(raw["activation_argv_template"])
    required = raw["required_units"]
    if not isinstance(required, list) or tuple(required) != REQUIRED_ACTIVATION_UNITS:
        raise DeployerSpecError(
            "required_units must be the complete sorted full-v3 inventory"
        )
    sample_runtime = root / "runtimes" / ("a" * 64)
    sample_receipt = receipt_root / f"{'a' * 64}.json"
    _validate_activation_argv(
        _expanded_activation_argv(
            activation,
            service_spec=sample_runtime / "promotion-services.json",
            destination=activation_destination,
            receipt=sample_receipt,
        ),
        service_spec=sample_runtime / "promotion-services.json",
        destination=activation_destination,
        receipt=sample_receipt,
    )
    actor = _require_id(raw["actor"], "deployer actor", spec=True)
    poll = _positive_number(raw["poll_interval_seconds"], "poll interval")
    return DeployerSpec(
        path=source,
        file_sha256=observed_file_hash,
        identity=identity,
        root=root,
        request_inbox=inbox,
        registry_spec=registry,
        current_deployment=current,
        activation_destination=activation_destination,
        controller_fence_argv=controller,
        activation_argv_template=activation,
        activation_receipt_root=receipt_root,
        required_units=tuple(required),
        source=source_binding,
        actor=actor,
        poll_interval_seconds=poll,
        raw=MappingProxyType(raw),
    )


def publish_deployer_spec(
    path: Path,
    *,
    root: Path,
    request_inbox: Path,
    registry_spec_path: Path,
    controller_fence_argv: Sequence[str],
    activation_argv_template: Sequence[str],
    actor: str,
    poll_interval_seconds: float,
    activation_destination: Optional[Path] = None,
    service_activation_destination: Optional[Path] = None,
    current_deployment: Optional[Path] = None,
    current_deployment_path: Optional[Path] = None,
    activation_receipt_root: Optional[Path] = None,
    required_units: Sequence[str] = REQUIRED_ACTIVATION_UNITS,
    source_path: Optional[Path] = None,
    source: Optional[Path] = None,
) -> DeployerSpec:
    """Publish one immutable deployer specification.

    The alias keyword pairs are accepted only at publication time; the emitted
    JSON contract has one unambiguous field for each value.
    """

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = destination.resolve()
    state_root = Path(root).resolve()
    activation_target = activation_destination or service_activation_destination
    if activation_target is None:
        raise DeployerSpecError("service activation destination is required")
    if (
        activation_destination is not None
        and service_activation_destination is not None
        and Path(activation_destination).resolve()
        != Path(service_activation_destination).resolve()
    ):
        raise DeployerSpecError("activation destination aliases conflict")
    current_target = current_deployment or current_deployment_path
    if current_target is None:
        current_target = state_root / "current-deployment.json"
    if (
        current_deployment is not None
        and current_deployment_path is not None
        and Path(current_deployment).resolve()
        != Path(current_deployment_path).resolve()
    ):
        raise DeployerSpecError("current deployment aliases conflict")
    source_target = source_path or source or Path(__file__).resolve()
    if (
        source_path is not None
        and source is not None
        and Path(source_path).resolve() != Path(source).resolve()
    ):
        raise DeployerSpecError("source aliases conflict")
    registry_path = Path(registry_spec_path).resolve()
    source_file = Path(source_target).resolve()
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SPEC_CONTRACT,
        "root": str(state_root),
        "request_inbox": str(Path(request_inbox).resolve()),
        "registry_spec": {
            "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        },
        "current_deployment": str(Path(current_target).resolve()),
        "activation_destination": str(Path(activation_target).resolve()),
        "controller_fence_argv": list(controller_fence_argv),
        "activation_argv_template": list(activation_argv_template),
        "activation_receipt_root": str(
            Path(
                activation_receipt_root or state_root / "activation-receipts"
            ).resolve()
        ),
        "required_units": list(required_units),
        "source": {
            "path": str(source_file),
            "sha256": file_sha256(source_file),
        },
        "actor": actor,
        "poll_interval_seconds": poll_interval_seconds,
    }
    value["spec_sha256"] = canonical_sha256(value)
    _atomic_immutable_json(destination, value)
    return load_deployer_spec(destination)


@dataclass(frozen=True)
class RequestContext:
    path: Path
    file_sha256: str
    identity: str
    raw: Mapping[str, Any]
    request_id: str
    candidate_suite_id: str
    runtime_id: str
    runtime_output: Path
    activation_receipt: Path
    runtime_argv: Tuple[str, ...]
    service_spec: FileBinding
    rotation_request: FileBinding
    suite_manifest: FileBinding
    continuity: FileBinding

    @property
    def state_root(self) -> Path:
        raise AttributeError("request state root belongs to the deployer")


def _request_binding(value: Any, role: str, *, identity: bool = True) -> FileBinding:
    keys = {"path", "sha256", "identity"} if identity else {"path", "sha256"}
    checked = _exact_keys(value, keys, f"{role} binding")
    expected = _require_sha256(checked["sha256"], f"{role} file hash")
    path = _required_file(
        checked["path"], role, expected, error_type=DeploymentRequestError
    )
    raw_identity = (
        _require_sha256(checked["identity"], f"{role} identity") if identity else None
    )
    return FileBinding(path, expected, raw_identity)


def _validate_nested_bindings(value: Any, role: str) -> None:
    if isinstance(value, Mapping):
        if {"path", "sha256"}.issubset(value):
            raw_path = value.get("path")
            raw_hash = value.get("sha256")
            if isinstance(raw_path, str) and Path(raw_path).is_absolute():
                expected = _require_sha256(raw_hash, f"{role} hash")
                _required_file(
                    raw_path,
                    role,
                    expected,
                    error_type=DeploymentRequestError,
                )
        for key, child in value.items():
            _validate_nested_bindings(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_bindings(child, f"{role}[{index}]")


def build_fence_proof(
    request: Mapping[str, Any],
    *,
    controller_fenced: bool = True,
    quiescent: bool = True,
    quiescent_roles: Sequence[str] = QUIESCENT_ROLES,
) -> Mapping[str, Any]:
    """Build the canonical proof expected on the reviewed fence command stdout."""

    cas = request["compare_and_swap"]
    candidate = request["candidate_suite"]
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": FENCE_PROOF_CONTRACT,
        "request_id": request["request_id"],
        "deployment_request_sha256": request["request_sha256"],
        "expected_active_suite_id": cas["expected_active_suite_id"],
        "expected_champion_sha256": cas["expected_champion_sha256"],
        "expected_generation_id": cas["expected_generation_id"],
        "candidate_suite_id": candidate["suite_id"],
        "controller_fenced": controller_fenced,
        "quiescent": quiescent,
        "quiescent_roles": list(quiescent_roles),
    }
    value["proof_sha256"] = canonical_sha256(value)
    return value


class AutonomyDeployer:
    """Reconcile immutable suite deployment requests under a root-side lock."""

    def __init__(
        self,
        spec: DeployerSpec | Path | str,
        *,
        expected_spec_sha256: Optional[str] = None,
        registry: Optional[Any] = None,
        registry_factory: Callable[[Path], Any] = SuiteRotationRegistry,
        registry_spec_loader: Callable[[Path], Any] = load_registry_spec,
        service_spec_loader: Callable[[Path], Any] = load_service_spec,
        suite_validator: Callable[..., Any] = validate_suite_manifest,
        promotion_runtime_loader: Optional[Callable[[Path], Any]] = None,
        deployment_verifier: Callable[[Path], Mapping[str, Any]] = (
            verify_deployment_manifest
        ),
        activation_planner: Callable[..., Mapping[str, Any]] = (
            plan_service_activation
        ),
        activation_applier: Callable[..., Mapping[str, Any]] = (
            apply_service_activation
        ),
        command_runner: CommandRunner = subprocess.run,
        systemctl_runner: SystemctlRunner = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        failure_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.spec = (
            spec
            if isinstance(spec, DeployerSpec)
            else load_deployer_spec(
                Path(spec), expected_spec_sha256=expected_spec_sha256
            )
        )
        if isinstance(spec, DeployerSpec) and expected_spec_sha256 is not None:
            expected = _require_sha256(
                expected_spec_sha256, "expected deployer specification hash", spec=True
            )
            if expected not in {self.spec.file_sha256, self.spec.identity}:
                raise DeployerSpecError("deployer specification hash is not expected")
        self.registry_spec_loader = registry_spec_loader
        self.registry_spec = registry_spec_loader(self.spec.registry_spec.path)
        if (
            getattr(self.registry_spec, "file_sha256", self.spec.registry_spec.sha256)
            != self.spec.registry_spec.sha256
            or Path(getattr(self.registry_spec, "path", self.spec.registry_spec.path))
            != self.spec.registry_spec.path
        ):
            raise DeployerSpecError(
                "registry loader returned a different specification"
            )
        self.registry = registry or registry_factory(self.spec.registry_spec.path)
        registry_path = getattr(getattr(self.registry, "spec", None), "path", None)
        if (
            registry_path is not None
            and Path(registry_path) != self.spec.registry_spec.path
        ):
            raise DeployerSpecError("registry instance uses a different specification")
        self.service_spec_loader = service_spec_loader
        self.suite_validator = suite_validator
        self.promotion_runtime_loader = promotion_runtime_loader
        self.deployment_verifier = deployment_verifier
        self.activation_planner = activation_planner
        self.activation_applier = activation_applier
        self.command_runner = command_runner
        self.systemctl_runner = systemctl_runner
        self.sleeper = sleeper
        self.failure_hook = failure_hook
        self._current_stage = "initialization"
        self._may_be_fenced = False
        self._current_context: Optional[RequestContext] = None
        self._assert_frozen()

    @property
    def lock_path(self) -> Path:
        return self.spec.root / "deployer.lock"

    @property
    def status_path(self) -> Path:
        return self.spec.root / "status.json"

    @property
    def requests_state_root(self) -> Path:
        return self.spec.root / "requests"

    @property
    def runtimes_root(self) -> Path:
        return self.spec.root / "runtimes"

    def _spec_binding(self) -> Mapping[str, str]:
        return {
            "path": str(self.spec.path),
            "sha256": self.spec.file_sha256,
            "identity": self.spec.identity,
        }

    def _assert_frozen(self) -> None:
        frozen = (
            (self.spec.path, self.spec.file_sha256, "deployer specification"),
            (
                self.spec.registry_spec.path,
                self.spec.registry_spec.sha256,
                "registry specification",
            ),
            (self.spec.source.path, self.spec.source.sha256, "deployer source"),
        )
        for path, expected, role in frozen:
            if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
                raise DeploymentStateError(f"frozen {role} changed")
        if (
            self.spec.request_inbox.is_symlink()
            or not self.spec.request_inbox.is_dir()
            or self.spec.activation_destination.is_symlink()
            or not self.spec.activation_destination.is_dir()
        ):
            raise DeploymentStateError("frozen deployment directory changed")

    def _checkpoint(self, stage: str) -> None:
        if self.failure_hook is None:
            return
        try:
            self.failure_hook(stage)
        except BaseException as exc:
            raise DeployerInterrupted(str(exc)) from exc

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise DeployerBusyError("POSIX advisory locking is unavailable")
        _ensure_directory(self.spec.root)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(self.lock_path), flags, 0o600)
        locked = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise DeployerBusyError("deployer lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise DeployerBusyError(
                        "another autonomy deployer is active"
                    ) from exc
                raise
            locked = True
            yield
        finally:
            if locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _run_command(
        self, argv: Sequence[str], role: str
    ) -> subprocess.CompletedProcess[str]:
        result = self.command_runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        returncode = getattr(result, "returncode", None)
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise DeploymentStateError(f"{role} runner returned no integer status")
        if returncode != 0:
            stderr = getattr(result, "stderr", "")
            raise DeploymentStateError(
                f"{role} returned {returncode}: {str(stderr).strip()}"
            )
        return result

    @staticmethod
    def _canonical_stdout(
        result: subprocess.CompletedProcess[str], role: str
    ) -> Mapping[str, Any]:
        stdout = getattr(result, "stdout", "")
        if not isinstance(stdout, str):
            raise DeploymentStateError(f"{role} stdout must be text")
        try:
            value = json.loads(
                stdout,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DeploymentStateError(f"{role} did not print canonical JSON") from exc
        if not isinstance(value, dict) or stdout != canonical_json(value) + "\n":
            raise DeploymentStateError(f"{role} stdout is not canonical JSON")
        return value

    def _request_paths(self) -> Tuple[Path, ...]:
        paths = []
        for root, directories, filenames in os.walk(
            self.spec.request_inbox, topdown=True, followlinks=False
        ):
            root_path = Path(root)
            safe_directories = []
            for name in sorted(directories):
                child = root_path / name
                if child.is_symlink():
                    raise DeploymentRequestError(
                        f"request inbox contains a symlinked directory: {child}"
                    )
                safe_directories.append(name)
            directories[:] = safe_directories
            for name in sorted(filenames):
                if not name.endswith(".json"):
                    continue
                path = root_path / name
                if path.is_symlink() or not path.is_file():
                    raise DeploymentRequestError(
                        f"request inbox contains an unsafe JSON path: {path}"
                    )
                paths.append(path)
        return tuple(
            sorted(
                paths,
                key=lambda item: item.relative_to(self.spec.request_inbox).as_posix(),
            )
        )

    def _validate_service_command(
        self,
        command: Mapping[str, Any],
        *,
        request: Mapping[str, Any],
        service_binding: FileBinding,
    ) -> Tuple[str, ...]:
        _exact_keys(command, _SERVICE_COMMAND_KEYS, "proposed service command")
        argv = _validate_argv(command["argv"], "proposed service command")
        if command["argv_sha256"] != canonical_sha256(list(argv)):
            raise DeploymentRequestError(
                "proposed service command argv hash is invalid"
            )
        executable_hash = _require_sha256(
            command["executable_sha256"], "service command executable hash"
        )
        if file_sha256(Path(argv[0])) != executable_hash:
            raise DeploymentRequestError("service command executable changed")
        inputs = _exact_keys(
            command["frozen_inputs"],
            _SERVICE_INPUT_KEYS,
            "service command frozen inputs",
        )
        if command["frozen_inputs_sha256"] != canonical_sha256(dict(inputs)):
            raise DeploymentRequestError("service command frozen input hash is invalid")
        registry = request["registry_spec"]
        expected_inputs = {
            "service_spec_path": str(service_binding.path),
            "service_spec_sha256": service_binding.sha256,
            "service_spec_identity": service_binding.identity,
            "registry_spec_path": registry["path"],
            "registry_spec_sha256": registry["sha256"],
        }
        if dict(inputs) != expected_inputs:
            raise DeploymentRequestError(
                "service command frozen inputs contradict request bindings"
            )
        if (
            len(argv) != 6
            or tuple(argv[1:3]) != ("-m", "risk_score.suite_rotation_service")
            or tuple(argv[3:6]) != ("watch", "--spec", str(service_binding.path))
        ):
            raise DeploymentRequestError(
                "proposed service command is not the bound suite-rotation watcher"
            )
        return argv

    def _validate_runtime_command(
        self,
        command: Mapping[str, Any],
        *,
        request: Mapping[str, Any],
        service_argv: Sequence[str],
    ) -> Tuple[Tuple[str, ...], str]:
        _exact_keys(command, _COMMAND_KEYS, "proposed runtime command")
        argv = _validate_argv(command["argv"], "proposed runtime command")
        if command["argv_sha256"] != canonical_sha256(list(argv)):
            raise DeploymentRequestError(
                "proposed runtime command argv hash is invalid"
            )
        if argv[0] != service_argv[0]:
            raise DeploymentRequestError(
                "runtime and service commands must use the same hash-bound Python"
            )
        inputs = _exact_keys(
            command["frozen_inputs"],
            _RUNTIME_INPUT_KEYS,
            "runtime command frozen inputs",
        )
        if command["frozen_inputs_sha256"] != canonical_sha256(dict(inputs)):
            raise DeploymentRequestError("runtime command frozen input hash is invalid")
        candidate = request["candidate_suite"]
        manifest = candidate["manifest"]
        expected_inputs = {
            "candidate_suite_id": candidate["suite_id"],
            "candidate_version_sha256": candidate["version_sha256"],
            "candidate_manifest_path": manifest["path"],
            "candidate_manifest_sha256": manifest["sha256"],
            "candidate_manifest_identity": manifest["identity"],
            "materialization_receipt_sha256": inputs["materialization_receipt_sha256"],
        }
        _require_sha256(
            inputs["materialization_receipt_sha256"],
            "runtime materialization receipt identity",
        )
        if dict(inputs) != expected_inputs:
            raise DeploymentRequestError(
                "runtime command frozen inputs contradict candidate suite"
            )
        flags = _parse_module_flags(
            argv,
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
                "--cluster-executor-command-json",
                "--adaptive-training-command-json",
                "--suite-rotation-command-json",
                "--autonomy-policy",
                "--cluster-executor-spec",
                "--adaptive-training-spec",
                "--suite-registry-spec",
                "--evaluator-process-count",
            },
            switch_flags={"--mutation-enabled", "--full-autonomy"},
            required_values={
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
                "--suite-registry-spec",
            },
            required_switches={"--mutation-enabled", "--full-autonomy"},
            role="proposed runtime command",
        )
        if flags["--output-dir"] != "{output_dir}":
            raise DeploymentRequestError(
                "runtime command output directory must be {output_dir}"
            )
        if flags["--suite-dir"] != str(
            Path(request["candidate_suite"]["manifest"]["path"]).parent
        ):
            raise DeploymentRequestError(
                "runtime command suite directory differs from candidate suite"
            )
        if flags["--suite-registry-spec"] != request["service_spec"]["path"]:
            raise DeploymentRequestError(
                "runtime command uses a different suite rotation service "
                "specification"
            )
        runtime_id = canonical_sha256(
            {
                "deployment_request_sha256": request["request_sha256"],
                "candidate_suite_id": request["candidate_suite"]["suite_id"],
                "runtime_command_sha256": command["argv_sha256"],
                "runtime_inputs_sha256": command["frozen_inputs_sha256"],
            }
        )
        return argv, runtime_id

    def _validate_request_static(self, path: Path) -> RequestContext:
        _reject_symlink_ancestors(
            path,
            "deployment request",
            error_type=DeploymentRequestError,
        )
        value, identity = _load_self_hashed(
            path,
            role="privileged deployment request",
            contract=DEPLOYMENT_REQUEST_CONTRACT,
            hash_field="request_sha256",
            exact_keys=_REQUEST_KEYS,
            error_type=DeploymentRequestError,
        )
        request_id = _require_id(value["request_id"], "deployment request ID")
        _require_id(value["actor"], "deployment request actor")
        created = value["created_at_utc"]
        if not isinstance(created, str):
            raise DeploymentRequestError("deployment request timestamp is missing")
        try:
            parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DeploymentRequestError(
                "deployment request timestamp is invalid"
            ) from exc
        if parsed_created.tzinfo is None:
            raise DeploymentRequestError("deployment request timestamp must be aware")

        service_binding = _request_binding(
            value["service_spec"], "suite-rotation service specification"
        )
        loaded_service = self.service_spec_loader(service_binding.path)
        if (
            getattr(loaded_service, "file_sha256", file_sha256(service_binding.path))
            != service_binding.sha256
            or getattr(loaded_service, "identity", service_binding.identity)
            != service_binding.identity
        ):
            raise DeploymentRequestError(
                "suite-rotation service loader returned different coordinates"
            )
        raw_service = getattr(loaded_service, "raw", None)
        if isinstance(raw_service, Mapping):
            if (
                raw_service.get("contract") != SERVICE_SPEC_CONTRACT
                or raw_service.get("actor") != value["actor"]
            ):
                raise DeploymentRequestError(
                    "deployment request actor or service contract changed"
                )

        registry_binding = _request_binding(
            value["registry_spec"], "suite rotation registry specification"
        )
        if (
            registry_binding.path != self.spec.registry_spec.path
            or registry_binding.sha256 != self.spec.registry_spec.sha256
            or registry_binding.identity
            != getattr(self.registry_spec, "identity", registry_binding.identity)
        ):
            raise DeploymentRequestError(
                "deployment request uses a different registry specification"
            )

        rotation_binding = _request_binding(
            value["rotation_request"], "rotation request manifest"
        )
        rotation, rotation_identity = _load_self_hashed(
            rotation_binding.path,
            role="rotation request manifest",
            contract="risk-score-evaluation-suite-rotation-request-v1",
            hash_field="request_sha256",
            error_type=DeploymentRequestError,
        )
        if rotation_identity != rotation_binding.identity:
            raise DeploymentRequestError("rotation request identity changed")
        _validate_nested_bindings(rotation, "rotation request")
        if rotation.get("request_id") != request_id:
            raise DeploymentRequestError("rotation request ID changed")

        candidate = _exact_keys(
            value["candidate_suite"],
            {"suite_id", "version_sha256", "manifest"},
            "candidate suite",
        )
        candidate_id = _require_sha256(candidate["suite_id"], "candidate suite ID")
        _require_sha256(candidate["version_sha256"], "candidate suite version")
        suite_binding = _request_binding(
            candidate["manifest"], "candidate suite manifest"
        )
        validated_suite = self.suite_validator(
            suite_binding.path,
            self.registry_spec,
            expected_champion_sha256=value["compare_and_swap"][
                "expected_champion_sha256"
            ],
        )
        if (
            getattr(validated_suite, "suite_id", None) != candidate_id
            or getattr(validated_suite, "manifest_sha256", suite_binding.sha256)
            != suite_binding.sha256
            or getattr(validated_suite, "manifest_identity", suite_binding.identity)
            != suite_binding.identity
        ):
            raise DeploymentRequestError(
                "candidate suite identity differs from its validated manifest"
            )

        continuity_binding = _request_binding(
            value["continuity"], "continuity manifest"
        )
        _, continuity_identity = _load_self_hashed(
            continuity_binding.path,
            role="continuity manifest",
            contract=CONTINUITY_CONTRACT,
            hash_field="manifest_sha256",
            error_type=DeploymentRequestError,
        )
        if continuity_identity != continuity_binding.identity:
            raise DeploymentRequestError("continuity manifest identity changed")
        continuity_value = _load_canonical_object(
            continuity_binding.path,
            "continuity manifest",
            error_type=DeploymentRequestError,
        )
        _validate_nested_bindings(continuity_value, "continuity manifest")

        cas = _exact_keys(value["compare_and_swap"], _CAS_KEYS, "compare-and-swap")
        _require_sha256(cas["expected_active_suite_id"], "expected active suite")
        _require_sha256(cas["expected_champion_sha256"], "expected champion")
        _require_id(cas["expected_generation_id"], "expected generation")
        if (
            cas["expected_pin_count"] != 0
            or cas["require_clean_generation_boundary"] is not True
            or cas["boundary_must_follow_continuity"] is not True
            or isinstance(cas["continuity_event_sequence"], bool)
            or not isinstance(cas["continuity_event_sequence"], int)
            or cas["continuity_event_sequence"] < 1
        ):
            raise DeploymentRequestError(
                "deployment request weakens the required clean-boundary CAS"
            )

        commands = _exact_keys(
            value["proposed_frozen_commands"],
            {"runtime", "service"},
            "proposed frozen commands",
        )
        if value["proposed_frozen_commands_sha256"] != canonical_sha256(dict(commands)):
            raise DeploymentRequestError("proposed frozen command hash is invalid")
        service_argv = self._validate_service_command(
            commands["service"],
            request=value,
            service_binding=service_binding,
        )
        runtime_argv, runtime_id = self._validate_runtime_command(
            commands["runtime"],
            request=value,
            service_argv=service_argv,
        )
        privilege = _exact_keys(
            value["privilege_boundary"],
            {
                "privileged_deployer_required",
                "service_may_activate_suite",
                "service_may_mutate_active_suite_pointer",
                "activation_api",
            },
            "privilege boundary",
        )
        if dict(privilege) != {
            "privileged_deployer_required": True,
            "service_may_activate_suite": False,
            "service_may_mutate_active_suite_pointer": False,
            "activation_api": "SuiteRotationRegistry.activate_suite",
        }:
            raise DeploymentRequestError(
                "deployment request privilege boundary changed"
            )

        runtime_output = self.runtimes_root / runtime_id
        activation_receipt = self.spec.activation_receipt_root / f"{runtime_id}.json"
        return RequestContext(
            path=Path(path),
            file_sha256=file_sha256(path),
            identity=identity,
            raw=MappingProxyType(value),
            request_id=request_id,
            candidate_suite_id=candidate_id,
            runtime_id=runtime_id,
            runtime_output=runtime_output,
            activation_receipt=activation_receipt,
            runtime_argv=runtime_argv,
            service_spec=service_binding,
            rotation_request=rotation_binding,
            suite_manifest=suite_binding,
            continuity=continuity_binding,
        )

    def _request_state_root(self, context: RequestContext) -> Path:
        return self.requests_state_root / context.identity

    def _receipt_path(self, context: RequestContext, name: str) -> Path:
        index = _RECEIPT_NAMES.index(name)
        return (
            self._request_state_root(context) / "receipts" / f"{index:02d}-{name}.json"
        )

    def _journal_path(self, context: RequestContext) -> Path:
        return self._request_state_root(context) / "journal.json"

    def _retryable_halt_path(self, context: RequestContext) -> Path:
        return self._request_state_root(context) / "retryable-halt.json"

    def _safety_halt_path(self, context: RequestContext) -> Path:
        return self._request_state_root(context) / "safety-halt.json"

    def _context_request_binding(self, context: RequestContext) -> Mapping[str, str]:
        return {
            "path": str(context.path),
            "sha256": context.file_sha256,
            "identity": context.identity,
        }

    def _base_journal(self, context: RequestContext) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": JOURNAL_CONTRACT,
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "request_id": context.request_id,
            "runtime_id": context.runtime_id,
            "state": "pending",
            "receipts": {name: None for name in _RECEIPT_NAMES},
            "retryable_halt": None,
            "safety_halt": None,
        }
        value["journal_sha256"] = canonical_sha256(value)
        return value

    def _load_journal(self, context: RequestContext) -> Dict[str, Any]:
        path = self._journal_path(context)
        if not os.path.lexists(os.fspath(path)):
            return self._base_journal(context)
        value, _ = _load_self_hashed(
            path,
            role="deployment journal",
            contract=JOURNAL_CONTRACT,
            hash_field="journal_sha256",
            error_type=DeploymentStateError,
        )
        expected_keys = {
            "schema_version",
            "contract",
            "spec",
            "request",
            "request_id",
            "runtime_id",
            "state",
            "receipts",
            "retryable_halt",
            "safety_halt",
            "journal_sha256",
        }
        if (
            set(value) != expected_keys
            or value["spec"] != self._spec_binding()
            or value["request"] != self._context_request_binding(context)
            or value["request_id"] != context.request_id
            or value["runtime_id"] != context.runtime_id
            or value["state"]
            not in {
                "pending",
                "request-validated",
                "fenced",
                "runtime-ready",
                "pointer-activated",
                "activation-halt",
                "active",
                "safety-halt",
            }
            or not isinstance(value["receipts"], Mapping)
            or set(value["receipts"]) != set(_RECEIPT_NAMES)
        ):
            raise DeploymentStateError("deployment journal identity is invalid")
        return value

    def _write_journal(
        self, context: RequestContext, journal: Mapping[str, Any]
    ) -> Dict[str, Any]:
        value = dict(journal)
        value.pop("journal_sha256", None)
        value["journal_sha256"] = canonical_sha256(value)
        _atomic_replace_json(self._journal_path(context), value)
        return value

    @staticmethod
    def _receipt_binding(path: Path, identity: str) -> Mapping[str, str]:
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "identity": identity,
        }

    def _adopt_receipt(
        self,
        context: RequestContext,
        journal: Mapping[str, Any],
        *,
        name: str,
        receipt: Mapping[str, Any],
        state: str,
    ) -> Dict[str, Any]:
        path = self._receipt_path(context, name)
        binding = self._receipt_binding(path, str(receipt["receipt_sha256"]))
        existing = journal["receipts"][name]
        if existing is not None and existing != binding:
            raise DeploymentStateError(f"journal {name} receipt binding changed")
        updated = dict(journal)
        receipts = dict(updated["receipts"])
        receipts[name] = binding
        updated["receipts"] = receipts
        current_state = str(journal["state"])
        ranks = {
            "pending": 0,
            "request-validated": 1,
            "fenced": 2,
            "runtime-ready": 3,
            "pointer-activated": 4,
            "active": 5,
        }
        if current_state == "safety-halt":
            updated["state"] = current_state
        elif current_state == "activation-halt" and state != "active":
            # A pointer-advanced deployment remains the sole retryable request
            # even if the process crashes while revalidating earlier receipts.
            updated["state"] = current_state
        elif ranks.get(state, -1) > ranks.get(current_state, -1):
            updated["state"] = state
        else:
            updated["state"] = current_state
        if updated == dict(journal):
            return dict(journal)
        return self._write_journal(context, updated)

    def _load_receipt(
        self, context: RequestContext, name: str, contract: str
    ) -> Optional[Mapping[str, Any]]:
        path = self._receipt_path(context, name)
        if not os.path.lexists(os.fspath(path)):
            return None
        value, _ = _load_self_hashed(
            path,
            role=f"{name} deployment receipt",
            contract=contract,
            hash_field="receipt_sha256",
            error_type=DeploymentStateError,
        )
        if value.get("spec") != self._spec_binding() or value.get(
            "request"
        ) != self._context_request_binding(context):
            raise DeploymentStateError(f"{name} receipt ancestry changed")
        return value

    def _registry_preconditions(
        self, context: RequestContext, *, boundary_id: Optional[str] = None
    ) -> Tuple[Any, str]:
        state = self.registry.reconstruct()
        request = context.raw
        cas = request["compare_and_swap"]
        current = getattr(state, "current_champion", None)
        if state.active_suite_id != cas["expected_active_suite_id"]:
            raise DeploymentRequestError("expected active suite is stale")
        if (
            current is None
            or current.sha256 != cas["expected_champion_sha256"]
            or current.generation_id != cas["expected_generation_id"]
        ):
            raise DeploymentRequestError("expected champion or generation is stale")
        if state.pins or len(state.pins) != cas["expected_pin_count"]:
            raise DeploymentRequestError("in-flight evaluation pins block deployment")
        registry_request = state.requests.get(context.request_id)
        if registry_request is None:
            raise DeploymentRequestError("rotation request is absent from registry")
        registry_manifest = registry_request.get("_manifest")
        request_manifest_binding = registry_request.get("request_manifest")
        if (
            not isinstance(registry_manifest, Mapping)
            or registry_manifest.get("request_sha256")
            != context.rotation_request.identity
            or not isinstance(request_manifest_binding, Mapping)
            or request_manifest_binding.get("path")
            != str(context.rotation_request.path)
            or request_manifest_binding.get("sha256") != context.rotation_request.sha256
            or request_manifest_binding.get("identity")
            != context.rotation_request.identity
        ):
            raise DeploymentRequestError("registry rotation request binding changed")
        registration = state.registrations.get(context.candidate_suite_id)
        version = state.versions.get(context.candidate_suite_id)
        if (
            registration is None
            or registration.get("request_id") != context.request_id
            or version is None
            or version.version_sha256
            != context.raw["candidate_suite"]["version_sha256"]
            or Path(version.manifest_path) != context.suite_manifest.path
            or version.manifest_sha256 != context.suite_manifest.sha256
            or version.manifest_identity != context.suite_manifest.identity
        ):
            raise DeploymentRequestError(
                "candidate suite is not the registered request-bound version"
            )
        continuity = state.continuity.get(context.candidate_suite_id)
        if (
            continuity is None
            or continuity.get("request_id") != context.request_id
            or continuity.get("_sequence") != cas["continuity_event_sequence"]
            or continuity.get("manifest", {}).get("path")
            != str(context.continuity.path)
            or continuity.get("manifest", {}).get("sha256") != context.continuity.sha256
            or continuity.get("manifest", {}).get("identity")
            != context.continuity.identity
        ):
            raise DeploymentRequestError("registered continuity binding changed")
        eligible = [
            (candidate_id, boundary)
            for candidate_id, boundary in state.boundaries.items()
            if boundary.get("clean") is True
            and boundary.get("champion_sha256") == cas["expected_champion_sha256"]
            and boundary.get("generation_id") == cas["expected_generation_id"]
            and isinstance(boundary.get("_sequence"), int)
            and boundary["_sequence"] > continuity["_sequence"]
        ]
        eligible.sort(key=lambda item: (item[1]["_sequence"], item[0]))
        if not eligible:
            raise DeploymentRequestError(
                "no clean generation boundary follows registered continuity"
            )
        selected = eligible[0][0]
        if boundary_id is not None and selected != boundary_id:
            raise DeploymentRequestError("selected generation boundary changed")
        return state, selected

    def _request_receipt_value(
        self, context: RequestContext, state: Any, boundary_id: str
    ) -> Dict[str, Any]:
        continuity = state.continuity[context.candidate_suite_id]
        boundary = state.boundaries[boundary_id]
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": REQUEST_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "request_id": context.request_id,
            "candidate_suite_id": context.candidate_suite_id,
            "runtime_id": context.runtime_id,
            "registry_snapshot": {
                "last_sequence": state.last_sequence,
                "last_event_sha256": state.last_event_sha256,
                "active_suite_id": state.active_suite_id,
                "champion_sha256": state.current_champion.sha256,
                "generation_id": state.current_champion.generation_id,
                "pin_count": len(state.pins),
            },
            "continuity_sequence": continuity["_sequence"],
            "boundary_id": boundary_id,
            "boundary_sequence": boundary["_sequence"],
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_request_receipt(self, context: RequestContext) -> Mapping[str, Any]:
        existing = self._load_receipt(context, "request", REQUEST_RECEIPT_CONTRACT)
        if existing is not None:
            if (
                existing.get("decision") != "PASS"
                or existing.get("request_id") != context.request_id
                or existing.get("candidate_suite_id") != context.candidate_suite_id
                or existing.get("runtime_id") != context.runtime_id
                or not isinstance(existing.get("boundary_id"), str)
                or not isinstance(existing.get("boundary_sequence"), int)
            ):
                raise DeploymentStateError("request validation receipt changed")
            return existing
        state, boundary_id = self._registry_preconditions(context)
        value = self._request_receipt_value(context, state, boundary_id)
        _atomic_immutable_json(self._receipt_path(context, "request"), value)
        self._checkpoint("after-request-receipt")
        return value

    def _validate_fence_proof(
        self, proof: Mapping[str, Any], context: RequestContext
    ) -> None:
        expected_keys = {
            "schema_version",
            "contract",
            "request_id",
            "deployment_request_sha256",
            "expected_active_suite_id",
            "expected_champion_sha256",
            "expected_generation_id",
            "candidate_suite_id",
            "controller_fenced",
            "quiescent",
            "quiescent_roles",
            "proof_sha256",
        }
        if set(proof) != expected_keys:
            raise DeploymentStateError("controller fence proof fields differ")
        payload = dict(proof)
        supplied = payload.pop("proof_sha256")
        expected = build_fence_proof(context.raw)
        if (
            supplied != canonical_sha256(payload)
            or proof != expected
            or proof["controller_fenced"] is not True
            or proof["quiescent"] is not True
            or proof["quiescent_roles"] != list(QUIESCENT_ROLES)
        ):
            raise DeploymentStateError(
                "controller fence did not prove controller and all roles quiescent"
            )

    def _fence_receipt_value(
        self, context: RequestContext, proof: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._validate_fence_proof(proof, context)
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": FENCE_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "argv": list(self.spec.controller_fence_argv),
            "argv_sha256": canonical_sha256(list(self.spec.controller_fence_argv)),
            "proof": dict(proof),
            "proof_sha256": proof["proof_sha256"],
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_fence(
        self, context: RequestContext, boundary_id: str
    ) -> Mapping[str, Any]:
        existing = self._load_receipt(context, "fence", FENCE_RECEIPT_CONTRACT)
        if existing is not None:
            expected = self._fence_receipt_value(context, existing.get("proof", {}))
            if existing != expected:
                raise DeploymentStateError("controller fence receipt changed")
            self._may_be_fenced = True
            return existing
        self._registry_preconditions(context, boundary_id=boundary_id)
        self._current_stage = "controller-fence"
        self._may_be_fenced = True
        result = self._run_command(
            self.spec.controller_fence_argv, "controller fence command"
        )
        proof = self._canonical_stdout(result, "controller fence command")
        value = self._fence_receipt_value(context, proof)
        _atomic_immutable_json(self._receipt_path(context, "fence"), value)
        self._checkpoint("after-fence-receipt")
        return value

    def _expanded_runtime_argv(self, context: RequestContext) -> Tuple[str, ...]:
        result = tuple(
            part.replace("{output_dir}", str(context.runtime_output))
            for part in context.runtime_argv
        )
        if any("{output_dir}" in part for part in result):
            raise DeploymentStateError("runtime output placeholder was not expanded")
        return result

    def _runtime_paths(self, context: RequestContext) -> Mapping[str, Path]:
        root = context.runtime_output
        return {
            "promotion_runtime": root / "promotion-runtime.json",
            "gpu_lease_runtime": root / "gpu-lease-runtime.json",
            "deployment_manifest": root / "deployment-manifest.json",
            "service_spec": root / "promotion-services.json",
        }

    def _load_promotion_runtime(self, path: Path) -> Any:
        if self.promotion_runtime_loader is not None:
            return self.promotion_runtime_loader(path)
        from risk_score.promotion_controller import RuntimeConfig

        return RuntimeConfig.load(path)

    @staticmethod
    def _candidate_schedule_map(
        manifest: Mapping[str, Any],
    ) -> Mapping[str, Mapping[str, Any]]:
        banks = manifest.get("banks")
        if not isinstance(banks, list):
            raise DeploymentStateError("candidate suite has no schedule inventory")
        by_name: Dict[str, Mapping[str, Any]] = {}
        for bank in banks:
            if not isinstance(bank, Mapping):
                raise DeploymentStateError("candidate suite schedule bank is malformed")
            name = bank.get("qualifiedName")
            schedule = bank.get("schedule")
            if (
                not isinstance(name, str)
                or name in by_name
                or not isinstance(schedule, Mapping)
                or not isinstance(schedule.get("path"), str)
                or not isinstance(schedule.get("sha256"), str)
            ):
                raise DeploymentStateError(
                    "candidate suite schedule binding is malformed"
                )
            by_name[name] = schedule
        return MappingProxyType(by_name)

    def _validate_runtime_schedules(
        self,
        promotion: Mapping[str, Any],
        context: RequestContext,
        suite_manifest: Mapping[str, Any],
    ) -> None:
        paths = promotion.get("paths")
        hashes = promotion.get("hashes")
        if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
            raise DeploymentStateError(
                "promotion runtime path/hash inventory is malformed"
            )
        if (
            paths.get("suites") != str(context.suite_manifest.path.parent)
            or hashes.get("suiteManifest") != context.suite_manifest.sha256
        ):
            raise DeploymentStateError(
                "promotion runtime does not bind the candidate suite manifest"
            )
        bank_map = self._candidate_schedule_map(suite_manifest)
        expected = {
            "discoveryOrdinarySchedule": "discovery",
            "confirmationOrdinarySchedule": "confirmation",
            "auditSchedule": "audit",
            "lead40Schedule": "lead-40-confirmation",
            "lead80Schedule": "lead-80-confirmation",
        }
        for runtime_key, bank_name in expected.items():
            binding = bank_map.get(bank_name)
            if binding is None:
                raise DeploymentStateError(
                    f"candidate suite is missing {bank_name} schedule"
                )
            schedule = context.suite_manifest.path.parent / binding["path"]
            if (
                paths.get(runtime_key) != str(schedule)
                or hashes.get(runtime_key) != binding["sha256"]
                or schedule.is_symlink()
                or not schedule.is_file()
                or file_sha256(schedule) != binding["sha256"]
            ):
                raise DeploymentStateError(
                    f"promotion runtime {runtime_key} schedule binding changed"
                )
        standard = Path(str(paths.get("standardConfirmationSchedule", "")))
        confirmation = (
            context.suite_manifest.path.parent / bank_map["confirmation"]["path"]
        )
        if (
            not standard.is_absolute()
            or standard.is_symlink()
            or not standard.is_file()
            or file_sha256(standard) != file_sha256(confirmation)
            or hashes.get("standardConfirmationSchedule") != file_sha256(standard)
        ):
            raise DeploymentStateError(
                "standard confirmation copy differs from candidate suite"
            )

    def _derive_runtime_result(
        self, context: RequestContext
    ) -> Tuple[Mapping[str, Any], list[Mapping[str, str]]]:
        paths = self._runtime_paths(context)
        values = {
            name: _load_canonical_object(path, f"generated {name}")
            for name, path in paths.items()
        }
        promotion = values["promotion_runtime"]
        gpu = values["gpu_lease_runtime"]
        service = values["service_spec"]
        if promotion.get("mutationEnabled") is not True:
            raise DeploymentStateError("generated promotion runtime is not automatic")
        if gpu.get("mutationEnabled") is not True:
            raise DeploymentStateError("generated GPU lease runtime is not automatic")
        self._load_promotion_runtime(paths["promotion_runtime"])

        candidate_manifest = _load_canonical_object(
            context.suite_manifest.path,
            "candidate suite manifest",
            error_type=DeploymentStateError,
        )
        validated = self.suite_validator(
            context.suite_manifest.path,
            self.registry_spec,
            expected_champion_sha256=context.raw["compare_and_swap"][
                "expected_champion_sha256"
            ],
        )
        if getattr(validated, "suite_id", None) != context.candidate_suite_id:
            raise DeploymentStateError(
                "generated runtime candidate suite identity changed"
            )
        self._validate_runtime_schedules(promotion, context, candidate_manifest)

        if (
            service.get("schema_version") != 3
            or service.get("contract") != AUTONOMY_SERVICE_SPEC_CONTRACT
            or service.get("mutation_enabled") is not True
            or service.get("full_autonomy") is not True
        ):
            raise DeploymentStateError(
                "generated service specification is not full autonomy v3"
            )
        inputs = service.get("service_inputs")
        expected_input_names = {
            "autonomy_policy",
            "executor_spec",
            "adaptive_spec",
            "suite_registry_spec",
        }
        if not isinstance(inputs, Mapping) or set(inputs) != expected_input_names:
            raise DeploymentStateError(
                "generated v3 service input inventory is incomplete"
            )
        for name, binding in inputs.items():
            if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                raise DeploymentStateError(
                    f"generated service input {name} is malformed"
                )
            expected_hash = _require_sha256(
                binding["sha256"], f"generated service input {name} hash"
            )
            path = _required_file(
                binding["path"],
                f"generated service input {name}",
                expected_hash,
                error_type=DeploymentStateError,
            )
            if name == "suite_registry_spec" and (
                path != self.spec.registry_spec.path
                or expected_hash != self.spec.registry_spec.sha256
            ):
                raise DeploymentStateError(
                    "generated service spec uses a different suite registry"
                )

        services = service.get("services")
        units = service.get("systemd_units")
        if (
            not isinstance(services, Mapping)
            or set(services) != set(FULL_SERVICE_UNIT_NAMES)
            or not isinstance(units, Mapping)
            or set(units) != set(FULL_SERVICE_UNIT_NAMES) | {"target"}
        ):
            raise DeploymentStateError("generated full-v3 unit inventory is incomplete")
        expected_unit_names = set(self.spec.required_units)
        observed_units: Dict[str, Mapping[str, str]] = {}
        output_paths = list(paths.values())
        for key, binding in sorted(units.items()):
            if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                raise DeploymentStateError(f"generated systemd unit {key} is malformed")
            expected_hash = _require_sha256(
                binding["sha256"], f"generated systemd unit {key} hash"
            )
            unit = _required_file(
                binding["path"],
                f"generated systemd unit {key}",
                expected_hash,
                error_type=DeploymentStateError,
            )
            expected_name = (
                TARGET_UNIT if key == "target" else FULL_SERVICE_UNIT_NAMES[key]
            )
            if unit.name != expected_name or unit.name == DEPLOYER_UNIT_NAME:
                raise DeploymentStateError(
                    f"generated systemd unit name is unexpected: {unit.name}"
                )
            observed_units[key] = {"path": str(unit), "sha256": expected_hash}
            output_paths.append(unit)
        if {Path(record["path"]).name for record in observed_units.values()} != (
            expected_unit_names
        ):
            raise DeploymentStateError("generated required unit names changed")
        plan = self.activation_planner(
            spec_path=paths["service_spec"],
            destination=self.spec.activation_destination,
        )
        if tuple(plan.get("unit_inventory", ())) != self.spec.required_units:
            raise DeploymentStateError("activation plan omits required full-v3 units")

        deployment = self.deployment_verifier(paths["deployment_manifest"])
        deployment_files = deployment.get("files")
        if not isinstance(deployment_files, Mapping):
            raise DeploymentStateError(
                "deployment manifest file inventory is malformed"
            )
        required_deployment_paths = {
            paths["promotion_runtime"]: file_sha256(paths["promotion_runtime"]),
            paths["gpu_lease_runtime"]: file_sha256(paths["gpu_lease_runtime"]),
            paths["service_spec"]: file_sha256(paths["service_spec"]),
            context.suite_manifest.path: context.suite_manifest.sha256,
            self.spec.source.path: self.spec.source.sha256,
        }
        observed_deployment_paths = {
            Path(record["path"]): record.get("sha256")
            for record in deployment_files.values()
            if isinstance(record, Mapping) and isinstance(record.get("path"), str)
        }
        for required_path, required_hash in required_deployment_paths.items():
            if observed_deployment_paths.get(required_path) != required_hash:
                raise DeploymentStateError(
                    f"deployment manifest does not bind {required_path}"
                )

        result = {
            "promotion_runtime": str(paths["promotion_runtime"]),
            "promotion_runtime_sha256": file_sha256(paths["promotion_runtime"]),
            "gpu_lease_runtime": str(paths["gpu_lease_runtime"]),
            "gpu_lease_runtime_sha256": file_sha256(paths["gpu_lease_runtime"]),
            "deployment_manifest": str(paths["deployment_manifest"]),
            "deployment_manifest_sha256": file_sha256(paths["deployment_manifest"]),
            "service_spec": str(paths["service_spec"]),
            "service_spec_sha256": file_sha256(paths["service_spec"]),
            "systemd_units": observed_units,
            "mutation_enabled": True,
            "full_autonomy": True,
            "evaluator_process_count": service.get("evaluator_process_count"),
        }
        outputs = [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in sorted(set(output_paths), key=str)
        ]
        return MappingProxyType(result), outputs

    def _runtime_receipt_value(self, context: RequestContext) -> Mapping[str, Any]:
        result, outputs = self._derive_runtime_result(context)
        expanded = self._expanded_runtime_argv(context)
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": RUNTIME_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "runtime_id": context.runtime_id,
            "output": str(context.runtime_output),
            "argv_template": list(context.runtime_argv),
            "argv_template_sha256": canonical_sha256(list(context.runtime_argv)),
            "argv": list(expanded),
            "argv_sha256": canonical_sha256(list(expanded)),
            "result": dict(result),
            "result_sha256": canonical_sha256(dict(result)),
            "outputs": outputs,
            "output_set_sha256": canonical_sha256(outputs),
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_runtime(
        self, context: RequestContext, boundary_id: str
    ) -> Mapping[str, Any]:
        existing = self._load_receipt(context, "runtime", RUNTIME_RECEIPT_CONTRACT)
        if existing is not None:
            expected = self._runtime_receipt_value(context)
            if existing != expected:
                raise DeploymentStateError("runtime receipt or outputs changed")
            return existing
        self._registry_preconditions(context, boundary_id=boundary_id)
        self._current_stage = "runtime-build"
        _ensure_directory(self.runtimes_root)
        if not context.runtime_output.exists():
            context.runtime_output.mkdir(parents=False, exist_ok=False)
            _fsync_directory(context.runtime_output.parent)
        elif context.runtime_output.is_symlink() or not context.runtime_output.is_dir():
            raise DeploymentStateError("content-addressed runtime output is unsafe")
        result = self._run_command(
            self._expanded_runtime_argv(context), "runtime builder"
        )
        reported = self._canonical_stdout(result, "runtime builder")
        derived, _ = self._derive_runtime_result(context)
        if reported != derived:
            raise DeploymentStateError(
                "runtime builder stdout does not match generated runtime"
            )
        value = self._runtime_receipt_value(context)
        _atomic_immutable_json(self._receipt_path(context, "runtime"), value)
        self._checkpoint("after-runtime-receipt")
        return value

    @staticmethod
    def _activation_event(state: Any, context: RequestContext) -> Any:
        for event in reversed(tuple(state.events)):
            if event.event_type != "suite.activated":
                continue
            payload = event.payload
            if payload.get("suite_id") != context.candidate_suite_id:
                continue
            if (
                payload.get("request_id") == context.request_id
                and payload.get("previous_suite_id")
                == context.raw["compare_and_swap"]["expected_active_suite_id"]
                and payload.get("expected_champion_sha256")
                == context.raw["compare_and_swap"]["expected_champion_sha256"]
            ):
                return event
            raise DeploymentStateError(
                "candidate suite is active with mismatched activation metadata"
            )
        raise DeploymentStateError("active candidate has no registry activation event")

    def _pointer_receipt_value(
        self, context: RequestContext, boundary_id: str
    ) -> Mapping[str, Any]:
        state = self.registry.reconstruct()
        if state.active_suite_id != context.candidate_suite_id or state.pins:
            raise DeploymentStateError(
                "registry pointer CAS did not activate the exact candidate"
            )
        event = self._activation_event(state, context)
        if event.payload.get("boundary_id") != boundary_id:
            raise DeploymentStateError("registry activation boundary changed")
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": POINTER_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "request_id": context.request_id,
            "candidate_suite_id": context.candidate_suite_id,
            "previous_suite_id": context.raw["compare_and_swap"][
                "expected_active_suite_id"
            ],
            "expected_champion_sha256": context.raw["compare_and_swap"][
                "expected_champion_sha256"
            ],
            "expected_generation_id": context.raw["compare_and_swap"][
                "expected_generation_id"
            ],
            "boundary_id": boundary_id,
            "event_sequence": event.sequence,
            "event_sha256": event.event_sha256,
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_pointer(
        self, context: RequestContext, boundary_id: str
    ) -> Mapping[str, Any]:
        existing = self._load_receipt(context, "pointer", POINTER_RECEIPT_CONTRACT)
        if existing is not None:
            expected = self._pointer_receipt_value(context, boundary_id)
            if existing != expected:
                raise DeploymentStateError("suite pointer receipt changed")
            return existing
        self._current_stage = "suite-pointer-cas"
        self._checkpoint("before-pointer-cas")
        cas = context.raw["compare_and_swap"]
        try:
            self.registry.activate_suite(
                context.request_id,
                context.candidate_suite_id,
                expected_active_suite_id=cas["expected_active_suite_id"],
                expected_champion_sha256=cas["expected_champion_sha256"],
                boundary_id=boundary_id,
            )
        except BaseException as exc:
            # SuiteRotationRegistry can be interrupted after its immutable
            # activation event is linked but before it repairs the mutable
            # active projection or returns.  Detect that exact committed CAS
            # and leave it for the normal idempotent replay path; never turn it
            # into a terminal halt or attempt a pointer rollback.
            try:
                self._pointer_receipt_value(context, boundary_id)
            except BaseException:
                raise exc
            raise DeployerInterrupted(
                "suite pointer CAS committed before registry call returned"
            ) from exc
        self._checkpoint("after-pointer-cas")
        value = self._pointer_receipt_value(context, boundary_id)
        _atomic_immutable_json(self._receipt_path(context, "pointer"), value)
        self._checkpoint("after-pointer-receipt")
        return value

    def _validate_system_activation_receipt(
        self, context: RequestContext
    ) -> Mapping[str, Any]:
        value, _ = _load_self_hashed(
            context.activation_receipt,
            role="systemd activation receipt",
            contract="risk-score-systemd-activation-receipt-v1",
            hash_field="receipt_sha256",
            error_type=DeploymentStateError,
        )
        required = set(self.spec.required_units)
        installed = value.get("installed_units")
        active = value.get("active")
        service_spec = self._runtime_paths(context)["service_spec"]
        if (
            value.get("service_spec_sha256") != file_sha256(service_spec)
            or value.get("target_unit") != TARGET_UNIT
            or value.get("unit_inventory") != list(self.spec.required_units)
            or not isinstance(installed, Mapping)
            or set(installed) != required
            or not isinstance(active, Mapping)
            or set(active) != required
            or any(status != "active" for status in active.values())
            or not isinstance(value.get("restart_occurred"), bool)
        ):
            raise DeploymentStateError(
                "activation receipt does not prove all required units active"
            )
        for unit, binding in installed.items():
            destination = self.spec.activation_destination / unit
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"path", "sha256"}
                or binding["path"] != str(destination)
                or destination.is_symlink()
                or not destination.is_file()
                or binding["sha256"] != file_sha256(destination)
            ):
                raise DeploymentStateError(
                    f"activation receipt installed unit changed: {unit}"
                )
        return value

    def _activation_receipt_value(
        self,
        context: RequestContext,
        runtime_receipt: Mapping[str, Any],
        pointer_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        system_receipt = self._validate_system_activation_receipt(context)
        service_spec = self._runtime_paths(context)["service_spec"]
        argv = _expanded_activation_argv(
            self.spec.activation_argv_template,
            service_spec=service_spec,
            destination=self.spec.activation_destination,
            receipt=context.activation_receipt,
        )
        _validate_activation_argv(
            argv,
            service_spec=service_spec,
            destination=self.spec.activation_destination,
            receipt=context.activation_receipt,
        )
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": ACTIVATION_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
            "pointer_receipt_sha256": pointer_receipt["receipt_sha256"],
            "argv": list(argv),
            "argv_sha256": canonical_sha256(list(argv)),
            "systemd_receipt": {
                "path": str(context.activation_receipt),
                "sha256": file_sha256(context.activation_receipt),
                "identity": system_receipt["receipt_sha256"],
            },
            "required_units": list(self.spec.required_units),
            "active": dict(system_receipt["active"]),
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_activation(
        self,
        context: RequestContext,
        runtime_receipt: Mapping[str, Any],
        pointer_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        existing = self._load_receipt(
            context, "activation", ACTIVATION_RECEIPT_CONTRACT
        )
        if existing is not None:
            expected = self._activation_receipt_value(
                context, runtime_receipt, pointer_receipt
            )
            if existing != expected:
                raise DeploymentStateError("service activation receipt changed")
            return existing
        self._current_stage = "service-activation"
        service_spec = self._runtime_paths(context)["service_spec"]
        argv = _expanded_activation_argv(
            self.spec.activation_argv_template,
            service_spec=service_spec,
            destination=self.spec.activation_destination,
            receipt=context.activation_receipt,
        )
        _validate_activation_argv(
            argv,
            service_spec=service_spec,
            destination=self.spec.activation_destination,
            receipt=context.activation_receipt,
        )
        result = self.activation_applier(
            spec_path=service_spec,
            destination=self.spec.activation_destination,
            receipt_path=context.activation_receipt,
            command_runner=self.systemctl_runner,
        )
        verified = self._validate_system_activation_receipt(context)
        if not isinstance(result, Mapping) or dict(result) != dict(verified):
            raise DeploymentStateError(
                "service activation result differs from canonical receipt"
            )
        value = self._activation_receipt_value(
            context, runtime_receipt, pointer_receipt
        )
        _atomic_immutable_json(self._receipt_path(context, "activation"), value)
        self._checkpoint("after-activation-receipt")
        return value

    def _current_projection_value(
        self,
        context: RequestContext,
        runtime_receipt: Mapping[str, Any],
        pointer_receipt: Mapping[str, Any],
        activation_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": CURRENT_DEPLOYMENT_CONTRACT,
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "request_id": context.request_id,
            "runtime_id": context.runtime_id,
            "runtime_output": str(context.runtime_output),
            "candidate_suite_id": context.candidate_suite_id,
            "runtime_receipt": self._receipt_binding(
                self._receipt_path(context, "runtime"),
                runtime_receipt["receipt_sha256"],
            ),
            "pointer_receipt": self._receipt_binding(
                self._receipt_path(context, "pointer"),
                pointer_receipt["receipt_sha256"],
            ),
            "activation_receipt": self._receipt_binding(
                self._receipt_path(context, "activation"),
                activation_receipt["receipt_sha256"],
            ),
            "actor": self.spec.actor,
        }
        value["record_sha256"] = canonical_sha256(value)
        return value

    def _completion_receipt_value(
        self,
        context: RequestContext,
        projection: Mapping[str, Any],
        activation_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": COMPLETION_RECEIPT_CONTRACT,
            "decision": "PASS",
            "spec": self._spec_binding(),
            "request": self._context_request_binding(context),
            "request_id": context.request_id,
            "runtime_id": context.runtime_id,
            "candidate_suite_id": context.candidate_suite_id,
            "activation_receipt_sha256": activation_receipt["receipt_sha256"],
            "current_deployment": {
                "path": str(self.spec.current_deployment),
                "sha256": file_sha256(self.spec.current_deployment),
                "identity": projection["record_sha256"],
            },
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value

    def _ensure_completion(
        self,
        context: RequestContext,
        runtime_receipt: Mapping[str, Any],
        pointer_receipt: Mapping[str, Any],
        activation_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        projection = self._current_projection_value(
            context, runtime_receipt, pointer_receipt, activation_receipt
        )
        if os.path.lexists(os.fspath(self.spec.current_deployment)):
            existing_projection = _load_canonical_object(
                self.spec.current_deployment, "current deployment projection"
            )
            if existing_projection != projection:
                _atomic_replace_json(self.spec.current_deployment, projection)
        else:
            _atomic_replace_json(self.spec.current_deployment, projection)
        existing = self._load_receipt(
            context, "completion", COMPLETION_RECEIPT_CONTRACT
        )
        expected = self._completion_receipt_value(
            context, projection, activation_receipt
        )
        if existing is not None:
            if existing != expected:
                raise DeploymentStateError("completion receipt changed")
            return existing
        _atomic_immutable_json(self._receipt_path(context, "completion"), expected)
        self._checkpoint("after-completion-receipt")
        return expected

    def _record_retryable_halt(
        self,
        context: RequestContext,
        journal: Mapping[str, Any],
        runtime_receipt: Mapping[str, Any],
        pointer_receipt: Mapping[str, Any],
        exc: BaseException,
    ) -> Dict[str, Any]:
        path = self._retryable_halt_path(context)
        if os.path.lexists(os.fspath(path)):
            halt, _ = _load_self_hashed(
                path,
                role="retryable activation halt",
                contract=RETRYABLE_HALT_CONTRACT,
                hash_field="halt_sha256",
                error_type=DeploymentStateError,
            )
        else:
            halt = {
                "schema_version": SCHEMA_VERSION,
                "contract": RETRYABLE_HALT_CONTRACT,
                "state": "activation-halt",
                "retryable": True,
                "retry_scope": "exact-runtime-only",
                "spec": self._spec_binding(),
                "request": self._context_request_binding(context),
                "runtime_id": context.runtime_id,
                "runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
                "pointer_receipt_sha256": pointer_receipt["receipt_sha256"],
                "old_controller_remains_fenced": True,
                "pointer_revert_permitted": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            halt["halt_sha256"] = canonical_sha256(halt)
            _atomic_immutable_json(path, halt)
        updated = dict(journal)
        updated["state"] = "activation-halt"
        updated["retryable_halt"] = self._receipt_binding(path, halt["halt_sha256"])
        return self._write_journal(context, updated)

    def _record_safety_halt(
        self,
        context: RequestContext,
        journal: Mapping[str, Any],
        exc: BaseException,
    ) -> Dict[str, Any]:
        path = self._safety_halt_path(context)
        if os.path.lexists(os.fspath(path)):
            halt, _ = _load_self_hashed(
                path,
                role="deployment safety halt",
                contract=SAFETY_HALT_CONTRACT,
                hash_field="halt_sha256",
                error_type=DeploymentStateError,
            )
        else:
            halt = {
                "schema_version": SCHEMA_VERSION,
                "contract": SAFETY_HALT_CONTRACT,
                "state": "safety-halt",
                "retryable": False,
                "spec": self._spec_binding(),
                "request": self._context_request_binding(context),
                "runtime_id": context.runtime_id,
                "failed_stage": self._current_stage,
                "controller_may_be_fenced": self._may_be_fenced,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            halt["halt_sha256"] = canonical_sha256(halt)
            _atomic_immutable_json(path, halt)
        updated = dict(journal)
        updated["state"] = "safety-halt"
        updated["safety_halt"] = self._receipt_binding(path, halt["halt_sha256"])
        return self._write_journal(context, updated)

    def _validate_completed_history(
        self, context: RequestContext, journal: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        contracts = {
            "request": REQUEST_RECEIPT_CONTRACT,
            "fence": FENCE_RECEIPT_CONTRACT,
            "runtime": RUNTIME_RECEIPT_CONTRACT,
            "pointer": POINTER_RECEIPT_CONTRACT,
            "activation": ACTIVATION_RECEIPT_CONTRACT,
            "completion": COMPLETION_RECEIPT_CONTRACT,
        }
        receipts: Dict[str, Mapping[str, Any]] = {}
        for name in _RECEIPT_NAMES:
            receipt = self._load_receipt(context, name, contracts[name])
            if receipt is None:
                raise DeploymentStateError(
                    f"completed deployment is missing its {name} receipt"
                )
            binding = self._receipt_binding(
                self._receipt_path(context, name), receipt["receipt_sha256"]
            )
            if journal["receipts"].get(name) != binding:
                raise DeploymentStateError(
                    f"completed deployment {name} receipt binding changed"
                )
            receipts[name] = receipt
        expected_fence = self._fence_receipt_value(
            context, receipts["fence"].get("proof", {})
        )
        if receipts["fence"] != expected_fence:
            raise DeploymentStateError("completed controller fence receipt changed")
        expected_runtime = self._runtime_receipt_value(context)
        if receipts["runtime"] != expected_runtime:
            raise DeploymentStateError("completed runtime artifacts changed")
        systemd_binding = receipts["activation"].get("systemd_receipt")
        if not isinstance(systemd_binding, Mapping):
            raise DeploymentStateError(
                "completed activation has no systemd receipt binding"
            )
        systemd_path = Path(str(systemd_binding.get("path", "")))
        systemd_receipt, systemd_identity = _load_self_hashed(
            systemd_path,
            role="historical systemd activation receipt",
            contract="risk-score-systemd-activation-receipt-v1",
            hash_field="receipt_sha256",
            error_type=DeploymentStateError,
        )
        if (
            systemd_binding.get("sha256") != file_sha256(systemd_path)
            or systemd_binding.get("identity") != systemd_identity
            or systemd_receipt.get("service_spec_sha256")
            != receipts["runtime"]["result"]["service_spec_sha256"]
        ):
            raise DeploymentStateError("historical systemd activation receipt changed")
        if (
            receipts["completion"].get("runtime_id") != context.runtime_id
            or receipts["completion"].get("candidate_suite_id")
            != context.candidate_suite_id
            or receipts["completion"].get("activation_receipt_sha256")
            != receipts["activation"]["receipt_sha256"]
        ):
            raise DeploymentStateError("completed deployment receipt chain changed")
        return journal

    def _advance_request(self, context: RequestContext) -> Mapping[str, Any]:
        self._current_context = context
        self._may_be_fenced = False
        journal = self._load_journal(context)
        if journal["state"] == "safety-halt":
            raise DeployerSafetyHalt(
                f"request {context.request_id} is in a terminal safety halt"
            )
        if journal["state"] == "active":
            return self._validate_completed_history(context, journal)
        try:
            request_receipt = self._ensure_request_receipt(context)
            journal = self._adopt_receipt(
                context,
                journal,
                name="request",
                receipt=request_receipt,
                state="request-validated",
            )
            boundary_id = str(request_receipt["boundary_id"])

            fence_receipt = self._ensure_fence(context, boundary_id)
            journal = self._adopt_receipt(
                context,
                journal,
                name="fence",
                receipt=fence_receipt,
                state="fenced",
            )

            runtime_receipt = self._ensure_runtime(context, boundary_id)
            journal = self._adopt_receipt(
                context,
                journal,
                name="runtime",
                receipt=runtime_receipt,
                state="runtime-ready",
            )

            pointer_receipt = self._ensure_pointer(context, boundary_id)
            journal = self._adopt_receipt(
                context,
                journal,
                name="pointer",
                receipt=pointer_receipt,
                state="pointer-activated",
            )
            try:
                activation_receipt = self._ensure_activation(
                    context, runtime_receipt, pointer_receipt
                )
            except DeployerInterrupted:
                raise
            except BaseException as exc:
                self._record_retryable_halt(
                    context,
                    journal,
                    runtime_receipt,
                    pointer_receipt,
                    exc,
                )
                raise ActivationRetryableHalt(
                    "active suite advanced; exact new runtime remains fenced "
                    "and must be replayed"
                ) from exc
            journal = self._adopt_receipt(
                context,
                journal,
                name="activation",
                receipt=activation_receipt,
                state="pointer-activated",
            )
            completion = self._ensure_completion(
                context,
                runtime_receipt,
                pointer_receipt,
                activation_receipt,
            )
            journal = self._adopt_receipt(
                context,
                journal,
                name="completion",
                receipt=completion,
                state="active",
            )
            if journal.get("retryable_halt") is not None:
                repaired = dict(journal)
                repaired["retryable_halt"] = None
                journal = self._write_journal(context, repaired)
            return journal
        except (DeployerInterrupted, ActivationRetryableHalt):
            raise
        except DeployerSafetyHalt:
            raise
        except BaseException as exc:
            current = self._load_journal(context)
            if self._may_be_fenced:
                self._record_safety_halt(context, current, exc)
                raise DeployerSafetyHalt(str(exc)) from exc
            raise

    def _raw_journals(self) -> Tuple[Tuple[Path, Mapping[str, Any]], ...]:
        if not self.requests_state_root.exists():
            return ()
        if (
            self.requests_state_root.is_symlink()
            or not self.requests_state_root.is_dir()
        ):
            raise DeploymentStateError("request state root is unsafe")
        result = []
        for directory in sorted(
            self.requests_state_root.iterdir(), key=lambda p: p.name
        ):
            if directory.is_symlink() or not directory.is_dir():
                raise DeploymentStateError(
                    f"unsafe request state directory: {directory}"
                )
            path = directory / "journal.json"
            if not path.exists():
                continue
            value, _ = _load_self_hashed(
                path,
                role="deployment journal",
                contract=JOURNAL_CONTRACT,
                hash_field="journal_sha256",
                error_type=DeploymentStateError,
            )
            result.append((path, value))
        return tuple(result)

    def _priority_halt_context(self) -> Optional[RequestContext]:
        halted = []
        for _, journal in self._raw_journals():
            if (
                journal.get("state") not in {"activation-halt", "safety-halt"}
                and journal.get("retryable_halt") is None
                and journal.get("safety_halt") is None
            ):
                continue
            request = journal.get("request")
            if not isinstance(request, Mapping) or not isinstance(
                request.get("path"), str
            ):
                raise DeploymentStateError(
                    "halted journal request binding is malformed"
                )
            context = self._validate_request_static(Path(request["path"]))
            if request != self._context_request_binding(context):
                raise DeploymentStateError("halted request changed in its inbox")
            halted.append((journal["state"], context))
        if len(halted) > 1:
            raise DeploymentStateError("multiple deployment requests are halted")
        return halted[0][1] if halted else None

    def _journal_summaries(self) -> list[Mapping[str, Any]]:
        summaries = []
        for _, journal in self._raw_journals():
            request = journal.get("request", {})
            summaries.append(
                {
                    "request_id": journal.get("request_id"),
                    "request_sha256": (
                        request.get("identity")
                        if isinstance(request, Mapping)
                        else None
                    ),
                    "path": (
                        request.get("path") if isinstance(request, Mapping) else None
                    ),
                    "runtime_id": journal.get("runtime_id"),
                    "state": journal.get("state"),
                }
            )
        return summaries

    def _status_value(
        self, *, error: Optional[BaseException] = None
    ) -> Mapping[str, Any]:
        summaries = self._journal_summaries()
        states = {item["state"] for item in summaries}
        inbox_paths = self._request_paths()
        known_paths = {item["path"] for item in summaries}
        pending = [str(path) for path in inbox_paths if str(path) not in known_paths]
        state = (
            "safety-halt"
            if "safety-halt" in states
            else (
                "activation-halt"
                if "activation-halt" in states
                else (
                    "pending"
                    if pending or any(item != "active" for item in states)
                    else "active" if summaries else "idle"
                )
            )
        )
        current = None
        if os.path.lexists(os.fspath(self.spec.current_deployment)):
            current = _load_canonical_object(
                self.spec.current_deployment, "current deployment projection"
            )
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": STATUS_CONTRACT,
            "spec": self._spec_binding(),
            "actor": self.spec.actor,
            "state": state,
            "pending_request_paths": pending,
            "requests": summaries,
            "current_deployment": current,
            "error": (
                None
                if error is None
                else {"type": type(error).__name__, "message": str(error)}
            ),
        }
        value["status_sha256"] = canonical_sha256(value)
        return value

    def _persist_status(self, status: Mapping[str, Any]) -> None:
        _atomic_replace_json(self.status_path, status)

    def status(self) -> Mapping[str, Any]:
        self._assert_frozen()
        return self._status_value()

    def _once_locked(self) -> Mapping[str, Any]:
        self._assert_frozen()
        _ensure_directory(self.spec.root)
        _ensure_directory(self.requests_state_root)
        _ensure_directory(self.runtimes_root)
        _ensure_directory(self.spec.activation_receipt_root)
        halted = self._priority_halt_context()
        if halted is not None:
            journal = self._load_journal(halted)
            if journal["state"] == "safety-halt":
                raise DeployerSafetyHalt(
                    f"request {halted.request_id} is in a terminal safety halt"
                )
            self._advance_request(halted)
        else:
            historical_paths: Dict[str, Mapping[str, Any]] = {}
            for _, journal in self._raw_journals():
                request_binding = journal.get("request")
                if not isinstance(request_binding, Mapping) or not isinstance(
                    request_binding.get("path"), str
                ):
                    raise DeploymentStateError(
                        "historical journal request binding is malformed"
                    )
                request_path = str(request_binding["path"])
                prior = historical_paths.get(request_path)
                if prior is not None and prior != request_binding:
                    raise DeploymentStateError(
                        "one inbox path has conflicting deployment histories"
                    )
                historical_paths[request_path] = request_binding
            for path in self._request_paths():
                context = self._validate_request_static(path)
                historical = historical_paths.get(str(path))
                if historical is not None and historical != (
                    self._context_request_binding(context)
                ):
                    raise DeploymentStateError(
                        "immutable deployment request path was replaced"
                    )
                self._advance_request(context)
        status = self._status_value()
        self._persist_status(status)
        return status

    def once(self) -> Mapping[str, Any]:
        with self._exclusive_lock():
            try:
                return self._once_locked()
            except DeployerInterrupted:
                raise
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    failed = self._status_value(error=exc)
                    self._persist_status(failed)
                raise

    reconcile_once = once

    def watch(self, *, poll_interval: Optional[float] = None) -> None:
        interval = (
            self.spec.poll_interval_seconds
            if poll_interval is None
            else _positive_number(poll_interval, "watch poll interval")
        )
        while True:
            try:
                self.once()
            except ActivationRetryableHalt:
                pass
            self.sleeper(interval)


PrivilegedAutonomyDeployer = AutonomyDeployer
Deployer = AutonomyDeployer


def status(
    spec_path: Path,
    **deployer_kwargs: Any,
) -> Mapping[str, Any]:
    return AutonomyDeployer(spec_path, **deployer_kwargs).status()


def once(
    spec_path: Path,
    **deployer_kwargs: Any,
) -> Mapping[str, Any]:
    return AutonomyDeployer(spec_path, **deployer_kwargs).once()


def watch(
    spec_path: Path,
    *,
    poll_interval: Optional[float] = None,
    **deployer_kwargs: Any,
) -> None:
    AutonomyDeployer(spec_path, **deployer_kwargs).watch(poll_interval=poll_interval)


def _systemd_quote(value: str) -> str:
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def render_deployer_systemd_unit(
    *,
    python_executable: Path,
    working_directory: Path,
    spec_path: Path,
    run_root: Optional[Path] = None,
) -> str:
    """Render the separately installed root unit pinned to the spec file hash."""

    python = _required_file(
        str(Path(python_executable).resolve()), "deployer Python executable"
    )
    working = _required_directory(
        str(Path(working_directory).resolve()), "deployer working directory"
    )
    spec_source = _required_file(
        str(Path(spec_path).resolve()), "deployer specification"
    )
    spec = load_deployer_spec(spec_source)
    root = (
        spec.root
        if run_root is None
        else _future_directory(str(Path(run_root).resolve()), "deployer run root")
    )
    if root != spec.root:
        raise DeployerSpecError("deployer unit run root differs from specification")
    argv = (
        str(python),
        "-m",
        "risk_score.autonomy_deployer",
        "watch",
        "--spec",
        str(spec_source),
        "--expected-spec-sha256",
        file_sha256(spec_source),
    )
    unit = "\n".join(
        [
            "[Unit]",
            "Description=KataGo privileged suite/runtime deployer",
            "Wants=network-online.target",
            "After=network-online.target",
            f"Before={TARGET_UNIT}",
            "RequiresMountsFor=" + _systemd_quote(str(root)),
            "StartLimitIntervalSec=600",
            "StartLimitBurst=3",
            "",
            "[Service]",
            "Type=simple",
            "User=root",
            "WorkingDirectory=" + _systemd_quote(str(working)),
            "Environment=" + _systemd_quote(f"PYTHONPATH={working}"),
            "ExecStart=" + " ".join(_systemd_quote(item) for item in argv),
            "Restart=always",
            "RestartSec=30",
            "KillSignal=SIGINT",
            "KillMode=control-group",
            "TimeoutStopSec=300",
            "UMask=0077",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )
    if "PartOf=" in unit:
        raise DeployerSpecError("deployer unit must never be PartOf the runtime target")
    return unit


def publish_deployer_systemd_unit(
    path: Path,
    *,
    python_executable: Path,
    working_directory: Path,
    spec_path: Path,
    run_root: Optional[Path] = None,
) -> Mapping[str, str]:
    destination = Path(path)
    if (
        not destination.is_absolute()
        or destination.name != DEPLOYER_UNIT_NAME
        or destination.is_symlink()
    ):
        raise DeployerSpecError(
            f"deployer unit path must be absolute and named {DEPLOYER_UNIT_NAME}"
        )
    data = render_deployer_systemd_unit(
        python_executable=python_executable,
        working_directory=working_directory,
        spec_path=spec_path,
        run_root=run_root,
    ).encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != data:
            raise DeployerSpecError("deployer unit conflicts with existing artifact")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=os.fspath(destination.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    return {"path": str(destination), "sha256": file_sha256(destination)}


render_systemd_unit = render_deployer_systemd_unit
publish_systemd_unit = publish_deployer_systemd_unit
load_spec = load_deployer_spec
publish_spec = publish_deployer_spec


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
    systemctl_runner: SystemctlRunner = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    args = parse_args(argv)
    try:
        deployer = AutonomyDeployer(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
            command_runner=command_runner,
            systemctl_runner=systemctl_runner,
            sleeper=sleeper,
        )
        if args.mode == "status":
            result = deployer.status()
        elif args.mode == "once":
            result = deployer.once()
        else:
            deployer.watch(poll_interval=args.poll_interval)
            return 0
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        TypeError,
        ValueError,
        DeployerError,
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


__all__ = [
    "ACTIVATION_RECEIPT_CONTRACT",
    "AUTONOMY_DEPLOYER_SPEC_CONTRACT",
    "ActivationRetryableHalt",
    "AutonomyDeployer",
    "COMPLETION_RECEIPT_CONTRACT",
    "CURRENT_DEPLOYMENT_CONTRACT",
    "DEPLOYER_UNIT_NAME",
    "DEPLOYER_STATUS_CONTRACT",
    "Deployer",
    "DeployerBusyError",
    "DeployerError",
    "DeployerInterrupted",
    "DeployerSafetyHalt",
    "DeployerSpec",
    "DeployerSpecError",
    "DeploymentRequestError",
    "DeploymentStateError",
    "FENCE_PROOF_CONTRACT",
    "FENCE_RECEIPT_CONTRACT",
    "JOURNAL_CONTRACT",
    "POINTER_RECEIPT_CONTRACT",
    "PrivilegedAutonomyDeployer",
    "QUIESCENT_ROLES",
    "REQUEST_RECEIPT_CONTRACT",
    "REQUIRED_ACTIVATION_UNITS",
    "RETRYABLE_HALT_CONTRACT",
    "RUNTIME_RECEIPT_CONTRACT",
    "ROOT_UNIT_NAME",
    "SAFETY_HALT_CONTRACT",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "STATUS_CONTRACT",
    "build_fence_proof",
    "load_deployer_spec",
    "load_spec",
    "main",
    "once",
    "parse_args",
    "publish_deployer_spec",
    "publish_deployer_systemd_unit",
    "publish_spec",
    "publish_systemd_unit",
    "render_deployer_systemd_unit",
    "render_systemd_unit",
    "status",
    "watch",
]


if __name__ == "__main__":
    raise SystemExit(main())
