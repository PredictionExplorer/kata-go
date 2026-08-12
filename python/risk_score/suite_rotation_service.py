#!/usr/bin/env python3
"""Unattended, fail-closed execution of evaluation-suite rotation requests.

The registry remains the only authority for cadence, suite validation, and the
active-suite pointer.  This service turns a current registry request into
restartable low-priority cluster work and stops at an immutable privileged
deployment request.  It never calls ``SuiteRotationRegistry.activate_suite``.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
import re
import stat
import string
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    Union,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production service targets are Unix.
    fcntl = None  # type: ignore[assignment]

from risk_score.build_evaluation_suites import MACHINE_MANIFEST_CONTRACT
from risk_score.cluster_executor import WORK_SPEC_CONTRACT
from risk_score.cluster_scheduler import (
    ClusterScheduler,
    SchedulerError,
    WorkItem,
    WorkKind,
    WorkRecord,
    WorkState,
)
from risk_score.curation_pipeline import (
    SPEC_CONTRACT as PIPELINE_SPEC_CONTRACT,
)
from risk_score.curation_pipeline import (
    load_pipeline_spec,
)
from risk_score.curation_supplement import (
    SPEC_CONTRACT as SUPPLEMENT_SPEC_CONTRACT,
)
from risk_score.curation_supplement import (
    load_supplement_spec,
)
from risk_score.suite_rotation import (
    CONTINUITY_CONTRACT,
    SuiteRotationError,
    SuiteRotationRegistry,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    publish_continuity_manifest,
    validate_suite_manifest,
)

SCHEMA_VERSION = 1
SERVICE_SPEC_CONTRACT = "risk-score-suite-rotation-service-spec-v1"
MATERIALIZATION_RECEIPT_CONTRACT = (
    "risk-score-suite-rotation-materialization-receipt-v1"
)
CONTINUITY_EVIDENCE_CONTRACT = "risk-score-suite-rotation-continuity-evidence-v1"
DEPLOYMENT_REQUEST_CONTRACT = (
    "risk-score-suite-rotation-privileged-deployment-request-v1"
)
STATUS_CONTRACT = "risk-score-suite-rotation-service-status-v1"
WORK_PRODUCER_CONTRACT = "risk-score-suite-rotation-cluster-work-v1"
SPEC_CONTRACT = SERVICE_SPEC_CONTRACT
SERVICE_STATUS_CONTRACT = STATUS_CONTRACT

LOCK_FILENAME = ".suite-rotation-service.lock"
MAX_JSON_BYTES = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:@+-]{0,254})$")
_SERVICE_SPEC_KEYS = {
    "schema_version",
    "contract",
    "root",
    "registry_spec",
    "scheduler_directory",
    "gpu7_id",
    "guardian_argv_prefix",
    "materializer_argv_template",
    "curation_argv_template",
    "continuity_argv_template",
    "results",
    "poll_interval_seconds",
    "actor",
    "spec_sha256",
}
_RESULT_NAMES = {
    "supplement_spec",
    "pipeline_spec",
    "suite_manifest",
    "continuity_evidence",
    "continuity_manifest",
    "deployment_request",
    "status",
}
_RESULT_CONTRACTS = {
    "supplement_spec": SUPPLEMENT_SPEC_CONTRACT,
    "pipeline_spec": PIPELINE_SPEC_CONTRACT,
    "suite_manifest": MACHINE_MANIFEST_CONTRACT,
    "continuity_evidence": CONTINUITY_EVIDENCE_CONTRACT,
    "continuity_manifest": CONTINUITY_CONTRACT,
    "deployment_request": DEPLOYMENT_REQUEST_CONTRACT,
    "status": STATUS_CONTRACT,
}
_PATH_TEMPLATE_FIELDS = frozenset({"request_id", "role"})
_EXECUTOR_FIELDS = frozenset(
    {
        "claim_id",
        "gpu_id",
        "guardian_receipt",
        "log_path",
        "state_directory",
        "work_id",
    }
)
_COMMAND_FIELDS = (
    frozenset(
        {
            "actor",
            "base_suite_id",
            "candidate_suite_id",
            "candidate_suite_manifest",
            "continuity_evidence",
            "continuity_manifest",
            "deployment_request",
            "gpu7_id",
            "model_path",
            "model_sha256",
            "pipeline_output_root",
            "pipeline_request",
            "pipeline_request_identity",
            "pipeline_request_sha256",
            "pipeline_spec",
            "policy_identity",
            "policy_path",
            "policy_sha256",
            "previous_champion_sha256",
            "registry_spec",
            "registry_spec_identity",
            "registry_spec_sha256",
            "request_id",
            "role",
            "rotation_request",
            "rotation_request_identity",
            "rotation_request_sha256",
            "scheduler_directory",
            "service_root",
            "service_spec",
            "service_spec_identity",
            "service_spec_sha256",
            "suite_manifest",
            "supplement_output_root",
            "supplement_request",
            "supplement_request_identity",
            "supplement_request_sha256",
            "supplement_spec",
        }
    )
    | _EXECUTOR_FIELDS
)
_MATERIALIZER_REQUIRED_FIELDS = frozenset(
    {
        "request_id",
        "rotation_request",
        "supplement_request",
        "pipeline_request",
        "supplement_spec",
        "pipeline_spec",
    }
)
_CURATION_REQUIRED_FIELDS = frozenset(
    {"request_id", "supplement_spec", "pipeline_spec", "suite_manifest"}
)
_CONTINUITY_REQUIRED_FIELDS = frozenset(
    {
        "request_id",
        "role",
        "model_path",
        "model_sha256",
        "candidate_suite_id",
        "continuity_evidence",
    }
)
_GUARDIAN_HASH_FLAGS = frozenset(
    {
        "--config-sha256",
        "--expected-config-sha256",
        "--expected-spec-sha256",
        "--spec-sha256",
    }
)
_GUARDIAN_COMMAND_MARKERS = frozenset({"--", "--command-json"})


class SuiteRotationServiceError(RuntimeError):
    """Base class for unattended suite-rotation failures."""


class ServiceSpecError(SuiteRotationServiceError, ValueError):
    """The canonical service specification is malformed or stale."""


class ServiceStateError(SuiteRotationServiceError, ValueError):
    """Durable service or child output contradicts its immutable ancestry."""


class ServiceConflictError(SuiteRotationServiceError):
    """A replay conflicts with already durable work or output."""


class ServiceStaleError(ServiceConflictError):
    """The active suite or champion changed while a request was being handled."""


class ServiceBusyError(SuiteRotationServiceError):
    """Another service process owns the process-lifetime writer lock."""


class MaterializerError(SuiteRotationServiceError):
    """The finite materializer command did not produce exact frozen specs."""


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
class ResultBinding:
    path_template: str
    contract: str

    def path(self, *, request_id: str = "", role: str = "") -> Path:
        values = {"request_id": request_id, "role": role}
        try:
            rendered = self.path_template.format_map(values)
        except (KeyError, ValueError) as exc:  # pragma: no cover - load validates.
            raise ServiceSpecError(f"cannot render result path: {exc}") from exc
        return Path(rendered)

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path_template, "contract": self.contract}


@dataclass(frozen=True)
class ServiceSpec:
    path: Path
    file_sha256: str
    identity: str
    root: Path
    registry_spec: FileBinding
    scheduler_directory: Path
    gpu7_id: str
    guardian_argv_prefix: Tuple[str, ...]
    materializer_argv_template: Tuple[str, ...]
    curation_argv_template: Tuple[str, ...]
    continuity_argv_template: Tuple[str, ...]
    results: Mapping[str, ResultBinding]
    poll_interval_seconds: float
    actor: str
    raw: Mapping[str, Any]

    @property
    def status_path(self) -> Path:
        return self.results["status"].path()


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    request: Mapping[str, Any]
    request_binding: FileBinding
    supplement_request: Mapping[str, Any]
    supplement_binding: FileBinding
    pipeline_request: Mapping[str, Any]
    pipeline_binding: FileBinding
    base_suite_id: str
    champion_sha256: str
    generation_id: str


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
        raise ServiceSpecError(f"{role} must be an object")
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise ServiceSpecError(
            f"{role} fields differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ServiceSpecError(f"{role} must be a lowercase SHA-256")
    return value


def _require_id(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ServiceSpecError(f"{role} must be a safe nonempty identifier")
    return value


def _require_number(value: Any, role: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ServiceSpecError(f"{role} must be a {qualifier} number")
    return float(value)


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise ServiceSpecError(f"{role} has a symlinked path component: {current}")
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ServiceSpecError(f"{role} must be a nonempty absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise ServiceSpecError(f"{role} must be absolute and lexically normalized")
    _reject_symlink_ancestors(path, role)
    return path


def _required_file(value: Any, role: str, expected_hash: Optional[str] = None) -> Path:
    path = _absolute_path(value, role)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ServiceSpecError(f"{role} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServiceSpecError(f"{role} must be a regular non-symlink file")
    if expected_hash is not None and file_sha256(path) != expected_hash:
        raise ServiceSpecError(f"{role} hash changed")
    return path


def _required_directory(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.is_symlink() or not path.is_dir():
        raise ServiceSpecError(f"{role} must be an existing non-symlink directory")
    return path


def _future_directory(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ServiceSpecError(f"{role} must be a non-symlink directory")
    elif not path.parent.is_dir() or path.parent.is_symlink():
        raise ServiceSpecError(f"{role} parent must already be a safe directory")
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


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except FileNotFoundError as exc:
        raise ServiceStateError(f"{role} is missing: {source}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ServiceStateError(f"{role} must be a regular non-symlink file")
    if metadata.st_size > MAX_JSON_BYTES:
        raise ServiceStateError(f"{role} exceeds the size limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ServiceStateError):
            raise
        raise ServiceStateError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ServiceStateError(f"{role} root must be an object")
    if data != canonical_json_bytes(value) + b"\n":
        raise ServiceStateError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_self_hashed(
    path: Path,
    *,
    role: str,
    contract: str,
    hash_field: str,
) -> Tuple[Dict[str, Any], str]:
    value = _load_canonical_object(path, role)
    payload = dict(value)
    supplied = payload.pop(hash_field, None)
    try:
        identity = _require_sha256(supplied, f"{role} {hash_field}")
    except ServiceSpecError as exc:
        raise ServiceStateError(str(exc)) from exc
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != contract
    ):
        raise ServiceStateError(f"{role} contract is unsupported")
    if identity != canonical_sha256(payload):
        raise ServiceStateError(f"{role} self-hash is invalid")
    return value, identity


def _template_fields(
    value: Any,
    role: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> Tuple[Tuple[str, ...], frozenset[str]]:
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
        raise ServiceSpecError(f"{role} must be a nonempty argv string array")
    formatter = string.Formatter()
    fields = set()
    for part in value:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise ServiceSpecError(f"{role} has invalid formatting") from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name not in allowed or format_spec or conversion is not None:
                raise ServiceSpecError(
                    f"{role} uses an unsupported placeholder: {field_name!r}"
                )
            fields.add(field_name)
    missing = sorted(required.difference(fields))
    if missing:
        raise ServiceSpecError(f"{role} does not bind required placeholders: {missing}")
    return tuple(value), frozenset(fields)


def _path_template(value: Any, role: str) -> Tuple[str, frozenset[str]]:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
    ):
        raise ServiceSpecError(f"{role} must be a nonempty absolute path template")
    formatter = string.Formatter()
    fields = set()
    try:
        parsed = formatter.parse(value)
    except ValueError as exc:
        raise ServiceSpecError(f"{role} has invalid formatting") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if (
            field_name not in _PATH_TEMPLATE_FIELDS
            or format_spec
            or conversion is not None
        ):
            raise ServiceSpecError(
                f"{role} uses an unsupported placeholder: {field_name!r}"
            )
        fields.add(field_name)
    rendered = value.format_map(
        {"request_id": "rotation-" + ("a" * 64), "role": "current_champion"}
    )
    _absolute_path(rendered, role)
    return value, frozenset(fields)


class _PreserveMissing(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _expand_argv(
    template: Sequence[str],
    values: Mapping[str, str],
    role: str,
    *,
    preserve_executor_fields: bool,
) -> Tuple[str, ...]:
    mapping: Mapping[str, str]
    if preserve_executor_fields:
        mapping = _PreserveMissing(values)
    else:
        mapping = values
    try:
        result = tuple(part.format_map(mapping) for part in template)
    except (KeyError, ValueError) as exc:
        raise ServiceStateError(f"cannot expand {role}: {exc}") from exc
    if any(not part or "\x00" in part for part in result):
        raise ServiceStateError(f"{role} expanded to an unsafe argv")
    if not preserve_executor_fields and any(
        field in _template_fields_from_sequence(result) for field in _EXECUTOR_FIELDS
    ):
        raise ServiceStateError(f"{role} retained an executor-only placeholder")
    return result


def _template_fields_from_sequence(value: Sequence[str]) -> frozenset[str]:
    formatter = string.Formatter()
    fields = set()
    for part in value:
        for _, field_name, _, _ in formatter.parse(part):
            if field_name is not None:
                fields.add(field_name)
    return frozenset(fields)


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
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ServiceStateError(f"unsafe service directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)
    _fsync_directory(path.parent)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    _ensure_directory(target.parent)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ServiceStateError(f"unsafe mutable projection path: {target}")
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
            raise ServiceConflictError(f"immutable artifact conflicts: {target}")
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
                raise ServiceConflictError(f"immutable artifact conflicts: {target}")
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def load_service_spec(
    path: Path,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> ServiceSpec:
    """Load and completely validate a canonical, self-hashed service spec."""

    source = Path(path).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    _reject_symlink_ancestors(source, "service specification")
    try:
        raw = _load_canonical_object(source, "suite-rotation service specification")
    except ServiceStateError as exc:
        raise ServiceSpecError(str(exc)) from exc
    _exact_keys(raw, _SERVICE_SPEC_KEYS, "service specification")
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or raw["contract"] != SERVICE_SPEC_CONTRACT
    ):
        raise ServiceSpecError("service specification contract is unsupported")
    payload = dict(raw)
    identity = _require_sha256(
        payload.pop("spec_sha256"), "service specification identity"
    )
    if identity != canonical_sha256(payload):
        raise ServiceSpecError("service specification self-hash is invalid")
    if expected_spec_sha256 is not None and identity != _require_sha256(
        expected_spec_sha256, "expected service specification identity"
    ):
        raise ServiceSpecError("service specification identity is not expected")

    root = _future_directory(raw["root"], "service root")
    registry_value = _exact_keys(
        raw["registry_spec"], {"path", "sha256"}, "registry specification binding"
    )
    registry_hash = _require_sha256(
        registry_value["sha256"], "registry specification file hash"
    )
    registry_path = _required_file(
        registry_value["path"], "registry specification", registry_hash
    )
    scheduler_directory = _required_directory(
        raw["scheduler_directory"], "scheduler directory"
    )
    gpu7_id = _require_id(raw["gpu7_id"], "GPU7 ID")
    actor = _require_id(raw["actor"], "service actor")
    poll = _require_number(raw["poll_interval_seconds"], "poll interval", positive=True)

    guardian, guardian_fields = _template_fields(
        raw["guardian_argv_prefix"],
        "guardian argv prefix",
        allowed=_EXECUTOR_FIELDS,
    )
    if not {"claim_id", "work_id", "guardian_receipt"}.issubset(guardian_fields):
        raise ServiceSpecError(
            "guardian argv prefix must bind claim_id, work_id, and "
            "guardian_receipt for its lifetime"
        )
    if guardian[-1] not in _GUARDIAN_COMMAND_MARKERS:
        raise ServiceSpecError(
            "guardian argv prefix must end with -- or --command-json"
        )
    hash_bound = False
    for index, part in enumerate(guardian):
        if (
            part in _GUARDIAN_HASH_FLAGS
            and index + 1 < len(guardian)
            and _SHA256_RE.fullmatch(guardian[index + 1]) is not None
        ):
            hash_bound = True
            break
        for flag in _GUARDIAN_HASH_FLAGS:
            marker = flag + "="
            if (
                part.startswith(marker)
                and _SHA256_RE.fullmatch(part[len(marker) :]) is not None
            ):
                hash_bound = True
                break
        if hash_bound:
            break
    if not hash_bound:
        raise ServiceSpecError(
            "guardian argv prefix must include a literal specification hash"
        )
    materializer, _ = _template_fields(
        raw["materializer_argv_template"],
        "materializer argv template",
        allowed=_COMMAND_FIELDS,
        required=_MATERIALIZER_REQUIRED_FIELDS,
    )
    curation, _ = _template_fields(
        raw["curation_argv_template"],
        "curation argv template",
        allowed=_COMMAND_FIELDS,
        required=_CURATION_REQUIRED_FIELDS,
    )
    continuity, _ = _template_fields(
        raw["continuity_argv_template"],
        "continuity argv template",
        allowed=_COMMAND_FIELDS,
        required=_CONTINUITY_REQUIRED_FIELDS,
    )

    results_value = _exact_keys(raw["results"], _RESULT_NAMES, "result bindings")
    results: Dict[str, ResultBinding] = {}
    rendered: Dict[str, Tuple[Path, ...]] = {}
    for name in sorted(_RESULT_NAMES):
        value = _exact_keys(results_value[name], {"path", "contract"}, f"result {name}")
        if value["contract"] != _RESULT_CONTRACTS[name]:
            raise ServiceSpecError(f"result {name} contract is unsupported")
        template, fields = _path_template(value["path"], f"result {name} path")
        if name == "status":
            if fields:
                raise ServiceSpecError("status result path may not be templated")
            paths = (Path(template),)
        else:
            if "request_id" not in fields:
                raise ServiceSpecError(
                    f"result {name} path must isolate each request_id"
                )
            if name == "continuity_evidence":
                if "role" not in fields:
                    raise ServiceSpecError(
                        "continuity evidence path must isolate each role"
                    )
                paths = tuple(
                    Path(
                        template.format_map(
                            {
                                "request_id": "rotation-" + ("a" * 64),
                                "role": role,
                            }
                        )
                    )
                    for role in ("current_champion", "previous_champion")
                )
            else:
                if "role" in fields:
                    raise ServiceSpecError(
                        f"result {name} path may not vary by continuity role"
                    )
                paths = (
                    Path(
                        template.format_map(
                            {
                                "request_id": "rotation-" + ("a" * 64),
                                "role": "",
                            }
                        )
                    ),
                )
        rendered[name] = paths
        results[name] = ResultBinding(template, value["contract"])

    flattened = [(name, path) for name, paths in rendered.items() for path in paths]
    for index, (name, path) in enumerate(flattened):
        for other_name, other in flattened[index + 1 :]:
            if _paths_overlap(path, other):
                raise ServiceSpecError(f"result paths overlap: {name} and {other_name}")
    status_path = results["status"].path()
    if not _strictly_within(status_path, root):
        raise ServiceSpecError("status result must be strictly beneath service root")
    if status_path == root / LOCK_FILENAME:
        raise ServiceSpecError("status result may not replace the service lock")
    if _paths_overlap(root, scheduler_directory):
        raise ServiceSpecError("service root may not overlap scheduler directory")
    for name, result_path in flattened:
        if (
            result_path == source
            or result_path == registry_path
            or _paths_overlap(result_path, scheduler_directory)
        ):
            raise ServiceSpecError(
                f"result {name} overlaps a frozen control-plane input"
            )

    return ServiceSpec(
        path=source.resolve(),
        file_sha256=file_sha256(source),
        identity=identity,
        root=root,
        registry_spec=FileBinding(registry_path, registry_hash),
        scheduler_directory=scheduler_directory,
        gpu7_id=gpu7_id,
        guardian_argv_prefix=guardian,
        materializer_argv_template=materializer,
        curation_argv_template=curation,
        continuity_argv_template=continuity,
        results=MappingProxyType(results),
        poll_interval_seconds=poll,
        actor=actor,
        raw=MappingProxyType(raw),
    )


def publish_service_spec(
    path: Path,
    *,
    root: Path,
    registry_spec_path: Path,
    scheduler_directory: Path,
    gpu7_id: str,
    guardian_argv_prefix: Sequence[str],
    materializer_argv_template: Sequence[str],
    curation_argv_template: Sequence[str],
    continuity_argv_template: Sequence[str],
    results: Mapping[str, Mapping[str, str]],
    poll_interval_seconds: float,
    actor: str,
) -> ServiceSpec:
    """Publish one immutable canonical service specification."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = destination.resolve()
    registry = _required_file(
        str(Path(registry_spec_path).resolve()), "registry specification"
    )
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SERVICE_SPEC_CONTRACT,
        "root": str(Path(root).resolve()),
        "registry_spec": {
            "path": str(registry),
            "sha256": file_sha256(registry),
        },
        "scheduler_directory": str(Path(scheduler_directory).resolve()),
        "gpu7_id": gpu7_id,
        "guardian_argv_prefix": list(guardian_argv_prefix),
        "materializer_argv_template": list(materializer_argv_template),
        "curation_argv_template": list(curation_argv_template),
        "continuity_argv_template": list(continuity_argv_template),
        "results": {name: dict(results[name]) for name in sorted(results)},
        "poll_interval_seconds": poll_interval_seconds,
        "actor": actor,
    }
    value["spec_sha256"] = canonical_sha256(value)
    _atomic_immutable_json(destination, value)
    return load_service_spec(destination)


_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.Lock] = {}


def _process_lock(path: Path) -> threading.Lock:
    key = os.fspath(path)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PROCESS_LOCKS[key] = lock
        return lock


class SuiteRotationService:
    """Restart-safe producer for one current suite-rotation request."""

    def __init__(
        self,
        spec: Union[ServiceSpec, Path, str],
        *,
        expected_spec_sha256: Optional[str] = None,
        registry: Optional[SuiteRotationRegistry] = None,
        scheduler: Optional[ClusterScheduler] = None,
        clock: Callable[[], Any] = lambda: datetime.now(timezone.utc),
        runner: Callable[..., Any] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        supplement_loader: Callable[[Path], Any] = load_supplement_spec,
        pipeline_loader: Callable[[Path], Any] = load_pipeline_spec,
        suite_validator: Callable[..., Any] = validate_suite_manifest,
        failure_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.spec = (
            spec
            if isinstance(spec, ServiceSpec)
            else load_service_spec(
                Path(spec), expected_spec_sha256=expected_spec_sha256
            )
        )
        if (
            isinstance(spec, ServiceSpec)
            and expected_spec_sha256 is not None
            and self.spec.identity != expected_spec_sha256
        ):
            raise ServiceSpecError("service specification identity is not expected")
        self.clock = clock
        self.runner = runner
        self.sleeper = sleeper
        self.supplement_loader = supplement_loader
        self.pipeline_loader = pipeline_loader
        self.suite_validator = suite_validator
        self.failure_hook = failure_hook or (lambda _: None)
        self.registry = registry or SuiteRotationRegistry(
            self.spec.registry_spec.path, clock=clock
        )
        self.scheduler = scheduler or ClusterScheduler(self.spec.scheduler_directory)
        if Path(self.scheduler.directory) != self.spec.scheduler_directory:
            raise ServiceSpecError("injected scheduler uses a different directory")
        self._assert_frozen()

    @property
    def lock_path(self) -> Path:
        return self.spec.root / LOCK_FILENAME

    def _now(self) -> datetime:
        value = self.clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ServiceStateError("clock returned a naive datetime")
            return value.astimezone(timezone.utc)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ServiceStateError("clock must return datetime or Unix seconds")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ServiceStateError("clock returned a non-finite value")
        return datetime.fromtimestamp(numeric, tz=timezone.utc)

    @staticmethod
    def _format_time(value: datetime) -> str:
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    def _assert_frozen(self) -> None:
        for path, expected, role in (
            (self.spec.path, self.spec.file_sha256, "service specification"),
            (
                self.spec.registry_spec.path,
                self.spec.registry_spec.sha256,
                "registry specification",
            ),
        ):
            if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
                raise ServiceStateError(f"frozen {role} changed")
        if (
            self.spec.scheduler_directory.is_symlink()
            or not self.spec.scheduler_directory.is_dir()
            or Path(self.scheduler.directory) != self.spec.scheduler_directory
        ):
            raise ServiceStateError("scheduler directory changed")
        snapshot = self.scheduler.reconstruct()
        if snapshot.dynamic_gpus or self.spec.gpu7_id not in snapshot.gpu_ids:
            raise ServiceStateError(
                "scheduler must expose a fixed inventory containing GPU7"
            )
        registry_spec = getattr(getattr(self.registry, "spec", None), "path", None)
        if (
            registry_spec is not None
            and Path(registry_spec) != self.spec.registry_spec.path
        ):
            raise ServiceStateError("registry instance uses a different specification")

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise ServiceBusyError("POSIX advisory locking is unavailable")
        _ensure_directory(self.spec.root)
        process_lock = _process_lock(self.lock_path)
        if not process_lock.acquire(blocking=False):
            raise ServiceBusyError("another suite-rotation service is active")
        descriptor = -1
        locked = False
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(os.fspath(self.lock_path), flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ServiceBusyError("service lock is not a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise ServiceBusyError(
                        "another process holds the suite-rotation service lock"
                    ) from exc
                raise
            locked = True
            yield
        finally:
            if descriptor >= 0:
                if locked:
                    with contextlib.suppress(OSError):
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            process_lock.release()

    def _result_path(
        self,
        name: str,
        request_id: str,
        *,
        role: str = "",
    ) -> Path:
        path = self.spec.results[name].path(request_id=request_id, role=role)
        _reject_symlink_ancestors(path, f"{name} result")
        return path

    def _request_state_dir(self, request_id: str) -> Path:
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return self.spec.root / "requests" / digest

    def _materialization_receipt_path(self, request_id: str) -> Path:
        return self._request_state_dir(request_id) / "materialization.json"

    @staticmethod
    def _binding_from_value(value: Any, role: str) -> FileBinding:
        checked = _exact_keys(value, {"path", "sha256", "identity"}, f"{role} binding")
        sha256 = _require_sha256(checked["sha256"], f"{role} file hash")
        identity = _require_sha256(checked["identity"], f"{role} identity")
        path = _required_file(checked["path"], role, sha256)
        return FileBinding(path, sha256, identity)

    def _request_context(
        self,
        registry_status: Mapping[str, Any],
        state: Any,
    ) -> Optional[RequestContext]:
        request_id = registry_status.get("current_request_id")
        if request_id is None:
            return None
        request_id = _require_id(request_id, "current rotation request ID")
        raw_state_request = state.requests.get(request_id)
        if raw_state_request is None:
            raise ServiceStateError("registry status names a missing current request")
        request_manifest = raw_state_request.get("_manifest")
        if not isinstance(request_manifest, Mapping):
            raise ServiceStateError("registry request has no validated manifest")
        request_binding = self._binding_from_value(
            raw_state_request.get("request_manifest"), "rotation request manifest"
        )
        if request_binding.identity != request_manifest.get("request_sha256"):
            raise ServiceStateError("rotation request binding identity changed")
        if _load_canonical_object(
            request_binding.path, "rotation request manifest"
        ) != dict(request_manifest):
            raise ServiceStateError(
                "rotation request manifest changed after validation"
            )

        children = request_manifest.get("requests")
        if not isinstance(children, Mapping) or set(children) != {
            "curation_supplement",
            "curation_pipeline",
        }:
            raise ServiceStateError("rotation request child inventory is malformed")
        supplement_binding = self._binding_from_value(
            children["curation_supplement"], "curation supplement request"
        )
        pipeline_binding = self._binding_from_value(
            children["curation_pipeline"], "curation pipeline request"
        )
        supplement, supplement_identity = _load_self_hashed(
            supplement_binding.path,
            role="curation supplement request",
            contract="risk-score-suite-rotation-curation-supplement-request-v1",
            hash_field="request_sha256",
        )
        pipeline, pipeline_identity = _load_self_hashed(
            pipeline_binding.path,
            role="curation pipeline request",
            contract="risk-score-suite-rotation-curation-pipeline-request-v1",
            hash_field="request_sha256",
        )
        if (
            supplement_identity != supplement_binding.identity
            or pipeline_identity != pipeline_binding.identity
            or supplement.get("request_id") != request_id
            or pipeline.get("request_id") != request_id
            or supplement.get("models") != request_manifest.get("models")
            or pipeline.get("models") != request_manifest.get("models")
            or supplement.get("policy") != request_manifest.get("policy")
            or pipeline.get("policy") != request_manifest.get("policy")
        ):
            raise ServiceStateError("rotation child request ancestry changed")
        current = state.current_champion
        if current is None:
            raise ServiceStateError("registry has no current champion")
        generation_id = getattr(current, "generation_id", None)
        if not isinstance(generation_id, str):
            generation_id = raw_state_request.get("generation_id")
        return RequestContext(
            request_id=request_id,
            request=MappingProxyType(dict(request_manifest)),
            request_binding=request_binding,
            supplement_request=MappingProxyType(supplement),
            supplement_binding=supplement_binding,
            pipeline_request=MappingProxyType(pipeline),
            pipeline_binding=pipeline_binding,
            base_suite_id=_require_sha256(
                raw_state_request.get("base_suite_id"), "request base suite ID"
            ),
            champion_sha256=_require_sha256(
                raw_state_request.get("champion_sha256"),
                "request champion hash",
            ),
            generation_id=_require_id(generation_id, "request generation ID"),
        )

    def _assert_request_current(self, context: RequestContext) -> Any:
        state = self.registry.reconstruct()
        current = state.current_champion
        if (
            state.active_suite_id != context.base_suite_id
            or current is None
            or current.sha256 != context.champion_sha256
            or current.generation_id != context.generation_id
        ):
            raise ServiceStaleError(
                "active suite or champion changed during suite rotation"
            )
        current_requests = [
            request_id
            for request_id, request in state.requests.items()
            if request["base_suite_id"] == state.active_suite_id
            and request["champion_sha256"] == current.sha256
        ]
        if context.request_id not in current_requests:
            raise ServiceStaleError("rotation request is no longer current")
        return state

    def _command_values(
        self,
        context: RequestContext,
        *,
        candidate_suite_id: str = "",
        candidate_suite_manifest: Optional[Path] = None,
        role: str = "",
        model_path: Optional[Path] = None,
        model_sha256: str = "",
        continuity_evidence: Optional[Path] = None,
    ) -> Dict[str, str]:
        request_id = context.request_id
        supplement_spec = self._result_path("supplement_spec", request_id)
        pipeline_spec = self._result_path("pipeline_spec", request_id)
        suite_manifest = self._result_path("suite_manifest", request_id)
        continuity_manifest = self._result_path("continuity_manifest", request_id)
        deployment_request = self._result_path("deployment_request", request_id)
        policy = context.request["policy"]
        return {
            "actor": self.spec.actor,
            "base_suite_id": context.base_suite_id,
            "candidate_suite_id": candidate_suite_id,
            "candidate_suite_manifest": str(
                suite_manifest
                if candidate_suite_manifest is None
                else candidate_suite_manifest
            ),
            "continuity_evidence": (
                "" if continuity_evidence is None else str(continuity_evidence)
            ),
            "continuity_manifest": str(continuity_manifest),
            "deployment_request": str(deployment_request),
            "gpu7_id": self.spec.gpu7_id,
            "model_path": "" if model_path is None else str(model_path),
            "model_sha256": model_sha256,
            "pipeline_output_root": str(context.pipeline_request["output_root"]),
            "pipeline_request": str(context.pipeline_binding.path),
            "pipeline_request_identity": str(context.pipeline_binding.identity),
            "pipeline_request_sha256": context.pipeline_binding.sha256,
            "pipeline_spec": str(pipeline_spec),
            "policy_identity": str(policy["identity"]),
            "policy_path": str(policy["path"]),
            "policy_sha256": str(policy["sha256"]),
            "previous_champion_sha256": str(
                getattr(self.registry.reconstruct(), "previous_champion_sha256", "")
                or ""
            ),
            "registry_spec": str(self.spec.registry_spec.path),
            "registry_spec_identity": str(
                getattr(getattr(self.registry, "spec", None), "identity", "")
            ),
            "registry_spec_sha256": self.spec.registry_spec.sha256,
            "request_id": request_id,
            "role": role,
            "rotation_request": str(context.request_binding.path),
            "rotation_request_identity": str(context.request_binding.identity),
            "rotation_request_sha256": context.request_binding.sha256,
            "scheduler_directory": str(self.spec.scheduler_directory),
            "service_root": str(self.spec.root),
            "service_spec": str(self.spec.path),
            "service_spec_identity": self.spec.identity,
            "service_spec_sha256": self.spec.file_sha256,
            "suite_manifest": str(suite_manifest),
            "supplement_output_root": str(context.supplement_request["output_root"]),
            "supplement_request": str(context.supplement_binding.path),
            "supplement_request_identity": str(context.supplement_binding.identity),
            "supplement_request_sha256": context.supplement_binding.sha256,
            "supplement_spec": str(supplement_spec),
        }

    @staticmethod
    def _model_spec_binding(value: Any, role: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ServiceStateError(f"{role} model binding is malformed")
        _require_sha256(value["sha256"], f"{role} model hash")
        _absolute_path(value["path"], f"{role} model path")
        return value

    def _validate_materialized_specs(
        self,
        context: RequestContext,
        materializer_argv: Sequence[str],
    ) -> Mapping[str, Any]:
        supplement_path = self._result_path("supplement_spec", context.request_id)
        pipeline_path = self._result_path("pipeline_spec", context.request_id)
        supplement_raw, supplement_identity = _load_self_hashed(
            supplement_path,
            role="materialized curation supplement specification",
            contract=SUPPLEMENT_SPEC_CONTRACT,
            hash_field="spec_sha256",
        )
        pipeline_raw, pipeline_identity = _load_self_hashed(
            pipeline_path,
            role="materialized curation pipeline specification",
            contract=PIPELINE_SPEC_CONTRACT,
            hash_field="spec_sha256",
        )
        child_models = context.request["models"]
        for raw, role in (
            (supplement_raw, "supplement"),
            (pipeline_raw, "pipeline"),
        ):
            models = raw.get("models")
            if not isinstance(models, Mapping) or set(models) != {
                "original",
                "champion",
            }:
                raise ServiceStateError(f"{role} model bindings are incomplete")
            for model_role in ("original", "champion"):
                observed = self._model_spec_binding(
                    models[model_role], f"{role} {model_role}"
                )
                requested = child_models[model_role]
                if (
                    observed["path"] != requested["path"]
                    or observed["sha256"] != requested["sha256"]
                ):
                    raise ServiceStateError(
                        f"{role} {model_role} model is not request-bound"
                    )
            policy = raw.get("policy")
            if (
                not isinstance(policy, Mapping)
                or set(policy) != {"path", "sha256"}
                or policy["path"] != context.request["policy"]["path"]
                or policy["sha256"] != context.request["policy"]["sha256"]
            ):
                raise ServiceStateError(f"{role} policy is not request-bound")
        for field in (
            "deployment",
            "deployment_manifest",
            "run_root",
            "katago",
            "analysis_config",
            "policy",
            "models",
        ):
            if supplement_raw.get(field) != pipeline_raw.get(field):
                raise ServiceStateError(
                    f"supplement and pipeline {field} provenance differs"
                )

        if supplement_raw.get("target_counts") != context.supplement_request.get(
            "target_counts"
        ):
            raise ServiceStateError("supplement targets differ from child request")
        if pipeline_raw.get("quotas") != context.pipeline_request.get("source_quotas"):
            raise ServiceStateError("pipeline quotas differ from child request")
        if pipeline_raw.get("suite_seed") != context.pipeline_request.get("suite_seed"):
            raise ServiceStateError("pipeline seed differs from child request")
        supplement_topology = supplement_raw.get("topology")
        pipeline_topology = pipeline_raw.get("topology")
        if (
            not isinstance(supplement_topology, Mapping)
            or supplement_topology.get("gpus") != [self.spec.gpu7_id]
            or supplement_topology.get("selfplay_gpus") != [self.spec.gpu7_id]
            or not isinstance(pipeline_topology, Mapping)
            or pipeline_topology.get("gpus") != [self.spec.gpu7_id]
        ):
            raise ServiceStateError(
                "curation topology must be pinned exclusively to guarded GPU7"
            )

        supplement_output = _absolute_path(
            context.supplement_request["output_root"],
            "supplement request output root",
        )
        pipeline_output = _absolute_path(
            context.pipeline_request["output_root"], "pipeline request output root"
        )
        supplement_work = _absolute_path(
            supplement_raw.get("work_root"), "supplement work root"
        )
        pipeline_work = _absolute_path(
            pipeline_raw.get("work_root"), "pipeline work root"
        )
        if not _strictly_within(supplement_work, supplement_output):
            raise ServiceStateError(
                "supplement work root escaped the registry child request"
            )
        if not _strictly_within(pipeline_work, pipeline_output):
            raise ServiceStateError(
                "pipeline work root escaped the registry child request"
            )
        if _paths_overlap(supplement_output, pipeline_output) or _paths_overlap(
            supplement_work, pipeline_work
        ):
            raise ServiceStateError("supplement and pipeline curation paths overlap")
        outputs = pipeline_raw.get("outputs")
        if not isinstance(outputs, Mapping) or set(outputs) != {
            "reviewed_bank",
            "reviewed_manifest",
            "suite_directory",
        }:
            raise ServiceStateError("pipeline output bindings are incomplete")
        output_paths = {
            name: _absolute_path(value, f"pipeline output {name}")
            for name, value in outputs.items()
        }
        if any(
            not _strictly_within(path, pipeline_output)
            for path in output_paths.values()
        ):
            raise ServiceStateError("pipeline output escaped its child request root")
        if any(
            _paths_overlap(path, supplement_work) or _paths_overlap(path, pipeline_work)
            for path in output_paths.values()
        ):
            raise ServiceStateError("curation output overlaps a mutable work root")
        expected_manifest = self._result_path("suite_manifest", context.request_id)
        if output_paths["suite_directory"] / "manifest.json" != expected_manifest:
            raise ServiceStateError(
                "pipeline suite output differs from the bound service result"
            )
        protected = (
            self.spec.scheduler_directory,
            self.spec.root,
        )
        for work_root in (supplement_work, pipeline_work):
            if any(_paths_overlap(work_root, root) for root in protected):
                raise ServiceStateError(
                    "curation work overlaps scheduler or service state"
                )

        try:
            loaded_supplement = self.supplement_loader(supplement_path)
            loaded_pipeline = self.pipeline_loader(pipeline_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ServiceStateError(
                f"materialized curation specification validation failed: {exc}"
            ) from exc
        for loaded, raw, identity, path, role in (
            (
                loaded_supplement,
                supplement_raw,
                supplement_identity,
                supplement_path,
                "supplement",
            ),
            (
                loaded_pipeline,
                pipeline_raw,
                pipeline_identity,
                pipeline_path,
                "pipeline",
            ),
        ):
            loaded_raw = getattr(loaded, "raw", raw)
            loaded_identity = getattr(loaded, "identity", identity)
            if dict(loaded_raw) != raw or loaded_identity != identity:
                raise ServiceStateError(
                    f"{role} loader returned contradictory frozen coordinates"
                )
            loaded_file_hash = getattr(loaded, "file_sha256", file_sha256(path))
            if loaded_file_hash != file_sha256(path):
                raise ServiceStateError(
                    f"{role} loader did not bind the exact specification file"
                )

        receipt: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": MATERIALIZATION_RECEIPT_CONTRACT,
            "service_spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "registry_spec": {
                "path": str(self.spec.registry_spec.path),
                "sha256": self.spec.registry_spec.sha256,
            },
            "request_id": context.request_id,
            "rotation_request": context.request_binding.to_dict(),
            "child_requests": {
                "curation_supplement": context.supplement_binding.to_dict(),
                "curation_pipeline": context.pipeline_binding.to_dict(),
            },
            "materializer_argv": list(materializer_argv),
            "materializer_argv_sha256": canonical_sha256(list(materializer_argv)),
            "specifications": {
                "curation_supplement": {
                    "path": str(supplement_path),
                    "sha256": file_sha256(supplement_path),
                    "identity": supplement_identity,
                    "contract": SUPPLEMENT_SPEC_CONTRACT,
                    "inputs_sha256": canonical_sha256(
                        {
                            key: supplement_raw[key]
                            for key in (
                                "deployment",
                                "deployment_manifest",
                                "katago",
                                "analysis_config",
                                "selfplay_config",
                                "selfplay_models_directory",
                                "policy",
                                "models",
                                "topology",
                            )
                        }
                    ),
                },
                "curation_pipeline": {
                    "path": str(pipeline_path),
                    "sha256": file_sha256(pipeline_path),
                    "identity": pipeline_identity,
                    "contract": PIPELINE_SPEC_CONTRACT,
                    "inputs_sha256": canonical_sha256(
                        {
                            key: pipeline_raw[key]
                            for key in (
                                "deployment",
                                "deployment_manifest",
                                "katago",
                                "analysis_config",
                                "policy",
                                "models",
                                "sources",
                                "topology",
                            )
                        }
                    ),
                },
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return receipt

    def _ensure_materialization(self, context: RequestContext) -> Mapping[str, Any]:
        supplement = self._result_path("supplement_spec", context.request_id)
        pipeline = self._result_path("pipeline_spec", context.request_id)
        receipt_path = self._materialization_receipt_path(context.request_id)
        materializer_argv = _expand_argv(
            self.spec.materializer_argv_template,
            self._command_values(context),
            "materializer argv",
            preserve_executor_fields=False,
        )
        exists = (
            os.path.lexists(os.fspath(supplement)),
            os.path.lexists(os.fspath(pipeline)),
        )
        partial_bindings = {
            path: file_sha256(path)
            for path, present in zip((supplement, pipeline), exists)
            if present and path.is_file() and not path.is_symlink()
        }
        if any(exists) and len(partial_bindings) != sum(exists):
            raise ServiceStateError(
                "partly materialized curation specifications are unsafe"
            )
        if not all(exists):
            if os.path.lexists(os.fspath(receipt_path)):
                raise ServiceStateError(
                    "materialization receipt exists without its frozen specifications"
                )
            completed = self.runner(
                list(materializer_argv),
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
            returncode = getattr(completed, "returncode", None)
            if isinstance(returncode, bool) or not isinstance(returncode, int):
                raise MaterializerError(
                    "materializer runner returned no integer status"
                )
            if returncode != 0:
                stderr = getattr(completed, "stderr", "")
                raise MaterializerError(
                    f"materializer returned {returncode}: {str(stderr).strip()}"
                )
            self.failure_hook("materializer-returned")
            if not supplement.is_file() or not pipeline.is_file():
                raise MaterializerError(
                    "materializer succeeded without publishing both specifications"
                )
            for path, expected_hash in partial_bindings.items():
                if file_sha256(path) != expected_hash:
                    raise ServiceConflictError(
                        "materializer changed an already published partial "
                        "specification"
                    )
        receipt = self._validate_materialized_specs(context, materializer_argv)
        _atomic_immutable_json(receipt_path, receipt)
        self.failure_hook("materialization-receipt-published")
        loaded, identity = _load_self_hashed(
            receipt_path,
            role="materialization receipt",
            contract=MATERIALIZATION_RECEIPT_CONTRACT,
            hash_field="receipt_sha256",
        )
        if identity != receipt["receipt_sha256"] or loaded != receipt:
            raise ServiceStateError("materialization receipt replay is contradictory")
        return receipt

    def _stage_records(
        self,
        *,
        request_id: str,
        stage: str,
        role: Optional[str],
    ) -> Tuple[WorkRecord, ...]:
        snapshot = self.scheduler.reconstruct()
        records = []
        for record in snapshot.work.values():
            payload = record.payload
            if (
                payload.get("producer_contract") == WORK_PRODUCER_CONTRACT
                and payload.get("service_spec_identity") == self.spec.identity
                and payload.get("request_id") == request_id
                and payload.get("stage") == stage
                and payload.get("role") == role
            ):
                records.append(record)
        records.sort(key=lambda record: record.enqueue_sequence)
        return tuple(records)

    def _build_work_item(
        self,
        *,
        context: RequestContext,
        stage: str,
        role: Optional[str],
        command: Sequence[str],
        inputs: Mapping[str, Any],
        attempt: int,
    ) -> WorkItem:
        stage_identity = canonical_sha256(
            {
                "service_spec_identity": self.spec.identity,
                "request_id": context.request_id,
                "stage": stage,
                "role": role,
                "inputs": dict(inputs),
                "command": list(command),
            }
        )
        work_id = f"suite-rotation-{stage}-{stage_identity[:40]}-attempt-{attempt:04d}"
        child_argv: Tuple[str, ...]
        if self.spec.guardian_argv_prefix[-1] == "--command-json":
            child_argv = (
                json.dumps(
                    list(command),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        else:
            child_argv = tuple(command)
        argv = tuple(self.spec.guardian_argv_prefix) + child_argv
        if tuple(argv[: len(self.spec.guardian_argv_prefix)]) != (
            self.spec.guardian_argv_prefix
        ):
            raise ServiceStateError("GPU7 work lost its frozen guardian prefix")
        argv_fields = _template_fields_from_sequence(argv)
        if not {"claim_id", "work_id"}.issubset(argv_fields):
            raise ServiceStateError(
                "GPU7 guardian work does not bind claim_id and work_id"
            )
        work_spec: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": WORK_SPEC_CONTRACT,
            "work_id": work_id,
            "kind": WorkKind.BACKFILL.value,
            "eligible_gpus": [self.spec.gpu7_id],
            "argv": list(argv),
            "cwd": str(self.spec.root),
            "environment": {},
            "lease_role": "none",
            "safe_drain": None,
        }
        work_spec["spec_sha256"] = canonical_sha256(work_spec)
        payload = {
            "producer_contract": WORK_PRODUCER_CONTRACT,
            "service_spec_identity": self.spec.identity,
            "request_id": context.request_id,
            "stage": stage,
            "role": role,
            "stage_identity": stage_identity,
            "attempt": attempt,
            "inputs": dict(inputs),
            "executor_spec": work_spec,
        }
        return WorkItem(
            work_id=work_id,
            kind=WorkKind.BACKFILL,
            eligible_gpus=(self.spec.gpu7_id,),
            preemptible=True,
            preferred_gpu=self.spec.gpu7_id,
            payload=payload,
        )

    def _ensure_stage_work(
        self,
        *,
        context: RequestContext,
        stage: str,
        role: Optional[str],
        command: Sequence[str],
        inputs: Mapping[str, Any],
        output_path: Path,
    ) -> Tuple[bool, WorkRecord]:
        expected_stage_identity = canonical_sha256(
            {
                "service_spec_identity": self.spec.identity,
                "request_id": context.request_id,
                "stage": stage,
                "role": role,
                "inputs": dict(inputs),
                "command": list(command),
            }
        )
        records = self._stage_records(
            request_id=context.request_id, stage=stage, role=role
        )
        for record in records:
            if record.payload.get("stage_identity") != expected_stage_identity:
                raise ServiceConflictError(
                    f"{stage} scheduler history changed its frozen inputs"
                )
        nonterminal = [
            record
            for record in records
            if record.state in {WorkState.QUEUED, WorkState.CLAIMED}
        ]
        if len(nonterminal) > 1:
            raise ServiceConflictError(
                f"overlapping {stage} executions are present in the scheduler"
            )
        latest = records[-1] if records else None
        output_exists = os.path.lexists(os.fspath(output_path))
        if output_exists:
            if (
                latest is not None
                and latest.state == WorkState.CLAIMED
                and len(nonterminal) == 1
            ):
                if output_path.is_symlink() or not output_path.is_file():
                    raise ServiceConflictError(f"{stage} output path is unsafe")
                return False, latest
            if (
                output_path.is_symlink()
                or not output_path.is_file()
                or latest is None
                or latest.state != WorkState.COMPLETED
            ):
                raise ServiceConflictError(
                    f"{stage} output overlaps incomplete or unscheduled work"
                )
            return True, latest
        if latest is not None and latest.state == WorkState.COMPLETED:
            raise ServiceStateError(f"completed {stage} work published no bound output")
        if nonterminal:
            return False, nonterminal[0]
        attempt = 1 if latest is None else int(latest.payload["attempt"]) + 1
        item = self._build_work_item(
            context=context,
            stage=stage,
            role=role,
            command=command,
            inputs=inputs,
            attempt=attempt,
        )
        record = self.scheduler.enqueue(item)
        self.failure_hook(f"{stage}-work-enqueued")
        return False, record

    def _registered_candidate(
        self,
        state: Any,
        context: RequestContext,
    ) -> Optional[str]:
        candidates = [
            suite_id
            for suite_id, registration in state.registrations.items()
            if registration["request_id"] == context.request_id
        ]
        if len(candidates) > 1:
            raise ServiceConflictError(
                "rotation request has multiple registered candidate suites"
            )
        return candidates[0] if candidates else None

    def _validate_candidate_output(self, context: RequestContext) -> Any:
        manifest_path = self._result_path("suite_manifest", context.request_id)
        manifest = _load_canonical_object(manifest_path, "curation suite manifest")
        if (
            manifest.get("schemaVersion") != 3
            or manifest.get("manifestContract") != MACHINE_MANIFEST_CONTRACT
        ):
            raise ServiceStateError("curation output is not an authoritative v3 suite")
        try:
            return self.suite_validator(
                manifest_path,
                self.registry.spec,
                expected_champion_sha256=context.champion_sha256,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ServiceStateError(
                f"candidate suite validation failed: {exc}"
            ) from exc

    def _ensure_candidate(
        self,
        context: RequestContext,
        materialization: Mapping[str, Any],
    ) -> Tuple[Optional[str], WorkRecord]:
        values = self._command_values(context)
        curation_command = _expand_argv(
            self.spec.curation_argv_template,
            values,
            "curation argv",
            preserve_executor_fields=True,
        )
        specifications = materialization["specifications"]
        inputs = {
            "materialization_receipt_sha256": materialization["receipt_sha256"],
            "supplement_spec": specifications["curation_supplement"],
            "pipeline_spec": specifications["curation_pipeline"],
            "suite_manifest_path": str(
                self._result_path("suite_manifest", context.request_id)
            ),
        }
        ready, record = self._ensure_stage_work(
            context=context,
            stage="curation",
            role=None,
            command=curation_command,
            inputs=inputs,
            output_path=self._result_path("suite_manifest", context.request_id),
        )
        state = self._assert_request_current(context)
        registered = self._registered_candidate(state, context)
        if not ready:
            if registered is not None:
                raise ServiceConflictError(
                    "suite was registered before curation work completed"
                )
            return None, record
        validated = self._validate_candidate_output(context)
        if registered is not None:
            if registered != validated.suite_id:
                raise ServiceConflictError(
                    "registered suite differs from immutable curation output"
                )
            return registered, record
        self._assert_request_current(context)
        event = self.registry.register_suite(
            context.request_id,
            self._result_path("suite_manifest", context.request_id),
        )
        self.failure_hook("candidate-suite-registered")
        suite_id = event.payload["suite_id"]
        if suite_id != validated.suite_id:
            raise ServiceStateError("registry registered a different candidate suite")
        return suite_id, record

    def _continuity_model(
        self,
        state: Any,
        context: RequestContext,
        role: str,
    ) -> Any:
        if role == "current_champion":
            binding = state.current_champion
            expected = context.champion_sha256
        else:
            expected = state.previous_champion_sha256
            binding = None if expected is None else state.champion_history.get(expected)
        if binding is None or binding.sha256 != expected:
            raise ServiceStateError(
                f"{role} model binding is unavailable for continuity"
            )
        return binding

    def _expected_continuity_evidence(
        self,
        context: RequestContext,
        *,
        candidate_suite_id: str,
        candidate_version: Any,
        role: str,
        model: Any,
        completed_at_utc: str,
    ) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTINUITY_EVIDENCE_CONTRACT,
            "request_id": context.request_id,
            "base_suite_id": context.base_suite_id,
            "candidate_suite": {
                "suite_id": candidate_suite_id,
                "version_sha256": candidate_version.version_sha256,
                "manifest_path": str(candidate_version.manifest_path),
                "manifest_sha256": candidate_version.manifest_sha256,
                "manifest_identity": candidate_version.manifest_identity,
            },
            "role": role,
            "model": {
                "path": str(model.path),
                "sha256": model.sha256,
            },
            "policy": {
                "path": str(self.registry.spec.policy_path),
                "sha256": self.registry.spec.policy_file_sha256,
                "identity": self.registry.spec.policy_identity,
            },
            "decision": "PASS",
            "completed_at_utc": completed_at_utc,
        }
        value["evidence_sha256"] = canonical_sha256(value)
        return value

    def _validate_continuity_evidence(
        self,
        path: Path,
        context: RequestContext,
        *,
        candidate_suite_id: str,
        candidate_version: Any,
        role: str,
        model: Any,
    ) -> Mapping[str, Any]:
        value, _ = _load_self_hashed(
            path,
            role=f"{role} continuity evidence",
            contract=CONTINUITY_EVIDENCE_CONTRACT,
            hash_field="evidence_sha256",
        )
        completed = value.get("completed_at_utc")
        if not isinstance(completed, str):
            raise ServiceStateError("continuity evidence has no completion timestamp")
        try:
            parsed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ServiceStateError(
                "continuity evidence completion timestamp is invalid"
            ) from exc
        if parsed.tzinfo is None:
            raise ServiceStateError("continuity evidence timestamp is naive")
        expected = self._expected_continuity_evidence(
            context,
            candidate_suite_id=candidate_suite_id,
            candidate_version=candidate_version,
            role=role,
            model=model,
            completed_at_utc=self._format_time(parsed),
        )
        if value != expected:
            raise ServiceStateError(
                f"{role} continuity evidence contradicts frozen inputs"
            )
        return value

    def _ensure_continuity(
        self,
        context: RequestContext,
        candidate_suite_id: str,
    ) -> Tuple[bool, Mapping[str, WorkRecord]]:
        state = self._assert_request_current(context)
        candidate_version = state.versions[candidate_suite_id]
        existing = state.continuity.get(candidate_suite_id)
        evidence_values: Dict[str, Mapping[str, Any]] = {}
        work: Dict[str, WorkRecord] = {}
        all_ready = True
        for role in ("current_champion", "previous_champion"):
            model = self._continuity_model(state, context, role)
            evidence_path = self._result_path(
                "continuity_evidence", context.request_id, role=role
            )
            values = self._command_values(
                context,
                candidate_suite_id=candidate_suite_id,
                candidate_suite_manifest=candidate_version.manifest_path,
                role=role,
                model_path=model.path,
                model_sha256=model.sha256,
                continuity_evidence=evidence_path,
            )
            command = _expand_argv(
                self.spec.continuity_argv_template,
                values,
                f"{role} continuity argv",
                preserve_executor_fields=True,
            )
            inputs = {
                "candidate_suite_id": candidate_suite_id,
                "candidate_version_sha256": candidate_version.version_sha256,
                "candidate_manifest_sha256": candidate_version.manifest_sha256,
                "model": {"path": str(model.path), "sha256": model.sha256},
                "policy_identity": self.registry.spec.policy_identity,
                "evidence_path": str(evidence_path),
            }
            ready, record = self._ensure_stage_work(
                context=context,
                stage="continuity",
                role=role,
                command=command,
                inputs=inputs,
                output_path=evidence_path,
            )
            work[role] = record
            all_ready = all_ready and ready
            if ready:
                evidence_values[role] = self._validate_continuity_evidence(
                    evidence_path,
                    context,
                    candidate_suite_id=candidate_suite_id,
                    candidate_version=candidate_version,
                    role=role,
                    model=model,
                )
        if not all_ready:
            if existing is not None:
                raise ServiceConflictError(
                    "continuity was registered before both shadow replays completed"
                )
            return False, MappingProxyType(work)

        continuity_path = self._result_path("continuity_manifest", context.request_id)
        if os.path.lexists(os.fspath(continuity_path)):
            existing_manifest = _load_canonical_object(
                continuity_path, "continuity manifest"
            )
            completed_at = existing_manifest.get("completed_at_utc")
        else:
            completed_at = max(
                str(value["completed_at_utc"]) for value in evidence_values.values()
            )
        state = self._assert_request_current(context)
        previous_hash = state.previous_champion_sha256
        if previous_hash is None:
            raise ServiceStateError("continuity replay requires a previous champion")
        receipt = publish_continuity_manifest(
            continuity_path,
            request_id=context.request_id,
            candidate_suite_id=candidate_suite_id,
            base_suite_id=context.base_suite_id,
            policy_hash=self.registry.spec.policy_identity,
            current_champion_sha256=context.champion_sha256,
            previous_champion_sha256=previous_hash,
            current_evidence_path=self._result_path(
                "continuity_evidence",
                context.request_id,
                role="current_champion",
            ),
            previous_evidence_path=self._result_path(
                "continuity_evidence",
                context.request_id,
                role="previous_champion",
            ),
            completed_at_utc=completed_at,
        )
        self.failure_hook("continuity-manifest-published")
        if existing is None:
            self._assert_request_current(context)
            self.registry.record_continuity(
                context.request_id, candidate_suite_id, continuity_path
            )
            self.failure_hook("continuity-recorded")
        else:
            identity = existing["manifest"]["identity"]
            if identity != receipt["manifest_sha256"]:
                raise ServiceConflictError(
                    "registry continuity differs from bound service output"
                )
        return True, MappingProxyType(work)

    def _deployment_value(
        self,
        context: RequestContext,
        state: Any,
        candidate_suite_id: str,
        *,
        created_at_utc: str,
    ) -> Dict[str, Any]:
        version = state.versions[candidate_suite_id]
        continuity = state.continuity[candidate_suite_id]
        materialization = _load_canonical_object(
            self._materialization_receipt_path(context.request_id),
            "materialization receipt",
        )
        runtime_argv = tuple(materialization["materializer_argv"])
        python = str(Path(sys.executable).resolve())
        service_argv = (
            python,
            "-m",
            "risk_score.suite_rotation_service",
            "watch",
            "--spec",
            str(self.spec.path),
        )
        runtime_inputs = {
            "candidate_suite_id": candidate_suite_id,
            "candidate_version_sha256": version.version_sha256,
            "candidate_manifest_path": str(version.manifest_path),
            "candidate_manifest_sha256": version.manifest_sha256,
            "candidate_manifest_identity": version.manifest_identity,
            "materialization_receipt_sha256": materialization["receipt_sha256"],
        }
        service_inputs = {
            "service_spec_path": str(self.spec.path),
            "service_spec_sha256": self.spec.file_sha256,
            "service_spec_identity": self.spec.identity,
            "registry_spec_path": str(self.spec.registry_spec.path),
            "registry_spec_sha256": self.spec.registry_spec.sha256,
        }
        commands = {
            "runtime": {
                "argv": list(runtime_argv),
                "argv_sha256": canonical_sha256(list(runtime_argv)),
                "frozen_inputs": runtime_inputs,
                "frozen_inputs_sha256": canonical_sha256(runtime_inputs),
            },
            "service": {
                "argv": list(service_argv),
                "argv_sha256": canonical_sha256(list(service_argv)),
                "executable_sha256": (
                    file_sha256(Path(python)) if Path(python).is_file() else None
                ),
                "frozen_inputs": service_inputs,
                "frozen_inputs_sha256": canonical_sha256(service_inputs),
            },
        }
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": DEPLOYMENT_REQUEST_CONTRACT,
            "request_id": context.request_id,
            "actor": self.spec.actor,
            "created_at_utc": created_at_utc,
            "service_spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "registry_spec": {
                "path": str(self.spec.registry_spec.path),
                "sha256": self.spec.registry_spec.sha256,
                "identity": self.registry.spec.identity,
            },
            "rotation_request": context.request_binding.to_dict(),
            "candidate_suite": {
                "suite_id": candidate_suite_id,
                "version_sha256": version.version_sha256,
                "manifest": {
                    "path": str(version.manifest_path),
                    "sha256": version.manifest_sha256,
                    "identity": version.manifest_identity,
                },
            },
            "continuity": {
                "path": continuity["manifest"]["path"],
                "sha256": continuity["manifest"]["sha256"],
                "identity": continuity["manifest"]["identity"],
            },
            "compare_and_swap": {
                "expected_active_suite_id": context.base_suite_id,
                "expected_champion_sha256": context.champion_sha256,
                "expected_generation_id": context.generation_id,
                "expected_pin_count": 0,
                "require_clean_generation_boundary": True,
                "boundary_must_follow_continuity": True,
                "continuity_event_sequence": continuity["_sequence"],
            },
            "proposed_frozen_commands": commands,
            "proposed_frozen_commands_sha256": canonical_sha256(commands),
            "privilege_boundary": {
                "privileged_deployer_required": True,
                "service_may_activate_suite": False,
                "service_may_mutate_active_suite_pointer": False,
                "activation_api": "SuiteRotationRegistry.activate_suite",
            },
        }
        value["request_sha256"] = canonical_sha256(value)
        return value

    def _ensure_deployment_request(
        self,
        context: RequestContext,
        candidate_suite_id: str,
    ) -> Optional[Mapping[str, Any]]:
        state = self._assert_request_current(context)
        if candidate_suite_id not in state.continuity:
            raise ServiceStateError("candidate continuity was not registered")
        if state.pins:
            return None
        destination = self._result_path("deployment_request", context.request_id)
        if os.path.lexists(os.fspath(destination)):
            existing, _ = _load_self_hashed(
                destination,
                role="privileged deployment request",
                contract=DEPLOYMENT_REQUEST_CONTRACT,
                hash_field="request_sha256",
            )
            created = existing.get("created_at_utc")
            if not isinstance(created, str):
                raise ServiceStateError("deployment request has no creation timestamp")
            expected = self._deployment_value(
                context,
                state,
                candidate_suite_id,
                created_at_utc=created,
            )
            if existing != expected:
                raise ServiceConflictError(
                    "privileged deployment request conflicts with current state"
                )
            return existing
        value = self._deployment_value(
            context,
            state,
            candidate_suite_id,
            created_at_utc=self._format_time(self._now()),
        )
        _atomic_immutable_json(destination, value)
        self.failure_hook("deployment-request-published")
        return value

    @staticmethod
    def _work_projection(record: WorkRecord) -> Mapping[str, Any]:
        return {
            "work_id": record.work_id,
            "state": record.state.value,
            "attempt": record.payload.get("attempt"),
            "stage": record.payload.get("stage"),
            "role": record.payload.get("role"),
        }

    def _status_value(
        self,
        *,
        registry_status: Optional[Mapping[str, Any]] = None,
        state: Optional[Any] = None,
        context: Optional[RequestContext] = None,
        error: Optional[BaseException] = None,
    ) -> Dict[str, Any]:
        observed_at = self._format_time(self._now())
        if registry_status is None:
            registry_status = self.registry.status()
        if state is None:
            state = self.registry.reconstruct()
        if context is None:
            context = self._request_context(registry_status, state)

        phase = "idle"
        next_action = "wait-for-cadence"
        candidate_suite_id = None
        works: Dict[str, Any] = {}
        outputs: Dict[str, Any] = {}
        request_id = None if context is None else context.request_id
        if context is not None:
            phase = "materialization-pending"
            next_action = "materialize-curation-specifications"
            supplement = self._result_path("supplement_spec", context.request_id)
            pipeline = self._result_path("pipeline_spec", context.request_id)
            receipt_path = self._materialization_receipt_path(context.request_id)
            materialized = (
                supplement.is_file() and pipeline.is_file() and receipt_path.is_file()
            )
            outputs["materialization"] = {
                "complete": materialized,
                "supplement_spec": str(supplement),
                "pipeline_spec": str(pipeline),
                "receipt": str(receipt_path),
            }
            curation_records = self._stage_records(
                request_id=context.request_id, stage="curation", role=None
            )
            if curation_records:
                works["curation"] = self._work_projection(curation_records[-1])
            registrations = [
                suite_id
                for suite_id, registration in state.registrations.items()
                if registration["request_id"] == context.request_id
            ]
            if len(registrations) == 1:
                candidate_suite_id = registrations[0]
            if materialized:
                phase = "curation-pending"
                next_action = "enqueue-or-resume-curation"
            if candidate_suite_id is not None:
                phase = "continuity-pending"
                next_action = "enqueue-or-resume-continuity"
                for role in ("current_champion", "previous_champion"):
                    records = self._stage_records(
                        request_id=context.request_id,
                        stage="continuity",
                        role=role,
                    )
                    if records:
                        works[f"continuity:{role}"] = self._work_projection(records[-1])
                if candidate_suite_id in state.continuity:
                    phase = "deployment-blocked" if state.pins else "deployment-pending"
                    next_action = (
                        "wait-for-zero-pins"
                        if state.pins
                        else "publish-privileged-deployment-request"
                    )
                    deployment = self._result_path(
                        "deployment_request", context.request_id
                    )
                    if deployment.is_file():
                        phase = "deployment-requested"
                        next_action = "await-privileged-clean-boundary-deployment"
            outputs["suite_manifest"] = {
                "path": str(self._result_path("suite_manifest", context.request_id)),
                "present": self._result_path(
                    "suite_manifest", context.request_id
                ).is_file(),
            }
            outputs["continuity_manifest"] = {
                "path": str(
                    self._result_path("continuity_manifest", context.request_id)
                ),
                "present": self._result_path(
                    "continuity_manifest", context.request_id
                ).is_file(),
            }
            outputs["deployment_request"] = {
                "path": str(
                    self._result_path("deployment_request", context.request_id)
                ),
                "present": self._result_path(
                    "deployment_request", context.request_id
                ).is_file(),
            }
        if error is not None:
            phase = "failed"
            next_action = "repair-or-retry"
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": STATUS_CONTRACT,
            "generated_at_utc": observed_at,
            "service_spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "actor": self.spec.actor,
            "state": phase,
            "next_action": next_action,
            "registry": {
                "spec_path": str(self.spec.registry_spec.path),
                "status_contract": registry_status.get("contract"),
                "status_sha256": registry_status.get("status_sha256"),
                "active_suite_id": state.active_suite_id,
                "current_champion_sha256": (
                    None
                    if state.current_champion is None
                    else state.current_champion.sha256
                ),
                "current_request_id": request_id,
                "pin_count": len(state.pins),
            },
            "candidate_suite_id": candidate_suite_id,
            "work": works,
            "outputs": outputs,
            "activation_performed": False,
            "active_pointer_mutated": False,
            "error": (
                None
                if error is None
                else {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ),
        }
        value["status_sha256"] = canonical_sha256(value)
        return value

    def _persist_status(self, status: Mapping[str, Any]) -> None:
        if status.get("contract") != STATUS_CONTRACT:
            raise ServiceStateError("refusing to publish an invalid status projection")
        _atomic_replace_json(self.spec.status_path, status)

    def status(self) -> Mapping[str, Any]:
        """Return a validated read-only projection without advancing work."""

        self._assert_frozen()
        return self._status_value()

    def _once_locked(self) -> Mapping[str, Any]:
        self._assert_frozen()
        registry_status = self.registry.once()
        state = self.registry.reconstruct()
        context = self._request_context(registry_status, state)
        if context is None:
            status = self._status_value(
                registry_status=registry_status,
                state=state,
                context=None,
            )
            self._persist_status(status)
            return status

        self._assert_request_current(context)
        materialization = self._ensure_materialization(context)
        self._assert_request_current(context)
        candidate_suite_id, _ = self._ensure_candidate(context, materialization)
        if candidate_suite_id is None:
            state = self.registry.reconstruct()
            status = self._status_value(
                registry_status=self.registry.status(),
                state=state,
                context=context,
            )
            self._persist_status(status)
            return status

        continuity_ready, _ = self._ensure_continuity(context, candidate_suite_id)
        if not continuity_ready:
            state = self.registry.reconstruct()
            status = self._status_value(
                registry_status=self.registry.status(),
                state=state,
                context=context,
            )
            self._persist_status(status)
            return status

        self._ensure_deployment_request(context, candidate_suite_id)
        state = self.registry.reconstruct()
        status = self._status_value(
            registry_status=self.registry.status(),
            state=state,
            context=context,
        )
        self._persist_status(status)
        return status

    def once(self) -> Mapping[str, Any]:
        """Run one bounded reconciliation pass."""

        with self._exclusive_lock():
            try:
                return self._once_locked()
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    failed = self._status_value(error=exc)
                    self._persist_status(failed)
                raise

    reconcile_once = once

    def watch(self) -> None:
        """Reconcile forever, sleeping only after each bounded pass."""

        while True:
            self.once()
            self.sleeper(self.spec.poll_interval_seconds)


RotationExecutionService = SuiteRotationService
SuiteRotationExecutionService = SuiteRotationService


def publish_continuity_evidence(
    path: Path,
    *,
    service: SuiteRotationService,
    request_id: str,
    candidate_suite_id: str,
    role: str,
    completed_at_utc: Optional[str] = None,
) -> Mapping[str, Any]:
    """Publish the strict PASS envelope expected from a continuity worker."""

    if role not in {"current_champion", "previous_champion"}:
        raise ValueError("continuity role is unsupported")
    state = service.registry.reconstruct()
    request = state.requests.get(request_id)
    if request is None:
        raise ValueError("unknown rotation request")
    registry_status = service.registry.status()
    context = service._request_context(registry_status, state)
    if context is None or context.request_id != request_id:
        raise ServiceStaleError("continuity request is not current")
    version = state.versions[candidate_suite_id]
    model = service._continuity_model(state, context, role)
    if completed_at_utc is None:
        completed = service._format_time(service._now())
    else:
        try:
            parsed = datetime.fromisoformat(completed_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("completed_at_utc is invalid") from exc
        if parsed.tzinfo is None:
            raise ValueError("completed_at_utc must be timezone-aware")
        completed = service._format_time(parsed)
    value = service._expected_continuity_evidence(
        context,
        candidate_suite_id=candidate_suite_id,
        candidate_version=version,
        role=role,
        model=model,
        completed_at_utc=completed,
    )
    _atomic_immutable_json(Path(path), value)
    return value


def status(
    spec_path: Path,
    **service_kwargs: Any,
) -> Mapping[str, Any]:
    return SuiteRotationService(spec_path, **service_kwargs).status()


def once(
    spec_path: Path,
    **service_kwargs: Any,
) -> Mapping[str, Any]:
    return SuiteRotationService(spec_path, **service_kwargs).once()


def watch(
    spec_path: Path,
    **service_kwargs: Any,
) -> None:
    SuiteRotationService(spec_path, **service_kwargs).watch()


load_spec = load_service_spec
load_suite_rotation_service_spec = load_service_spec
publish_suite_rotation_service_spec = publish_service_spec


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "once", "watch"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        service = SuiteRotationService(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
        )
        if args.mode == "status":
            result = service.status()
        elif args.mode == "once":
            result = service.once()
        else:
            service.watch()
            return 0
    except KeyboardInterrupt:
        return 0
    except (
        OSError,
        SchedulerError,
        SuiteRotationError,
        SuiteRotationServiceError,
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


__all__ = [
    "CONTINUITY_EVIDENCE_CONTRACT",
    "DEPLOYMENT_REQUEST_CONTRACT",
    "MATERIALIZATION_RECEIPT_CONTRACT",
    "MaterializerError",
    "ResultBinding",
    "RotationExecutionService",
    "SCHEMA_VERSION",
    "SERVICE_STATUS_CONTRACT",
    "SERVICE_SPEC_CONTRACT",
    "SPEC_CONTRACT",
    "STATUS_CONTRACT",
    "ServiceBusyError",
    "ServiceConflictError",
    "ServiceSpec",
    "ServiceSpecError",
    "ServiceStaleError",
    "ServiceStateError",
    "SuiteRotationService",
    "SuiteRotationExecutionService",
    "SuiteRotationServiceError",
    "WORK_PRODUCER_CONTRACT",
    "load_service_spec",
    "load_spec",
    "load_suite_rotation_service_spec",
    "main",
    "once",
    "parse_args",
    "publish_continuity_evidence",
    "publish_service_spec",
    "publish_suite_rotation_service_spec",
    "status",
    "watch",
]


if __name__ == "__main__":
    raise SystemExit(main())
