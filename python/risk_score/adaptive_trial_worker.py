"""Execute one bounded, hash-bound adaptive-training trial round.

The adaptive-training controller decides *which* recipe and round may run. This
module is only the execution boundary. It validates the controller's immutable
trial manifest, translates the frozen recipe allowlist to explicit argv flags,
executes a fixed sequence of receipt-producing commands, and publishes the
``adaptive_training.TRIAL_RESULT_CONTRACT``.

No command is passed through a shell. A command that reaches its deadline is
sent SIGINT and allowed to drain; this worker never escalates to SIGKILL. If a
child lifetime or checkpoint cannot be proved complete, the worker fails closed
without publishing a result that would falsely tell the controller the GPU is
safe to reuse.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import stat
import string
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are Unix.
    fcntl = None  # type: ignore[assignment]

from katago.train.training_controls import (
    VALIDATION_MANIFEST_CONTRACT,
    validate_validation_manifest,
)

from risk_score.adaptive_training import (
    POLICY_HASH,
    RECIPE_CONTRACT,
    TRIAL_CONTRACT,
    AdaptiveTrainingError,
    atomic_create_json,
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
    load_policy,
    load_trial_result,
    publish_trial_result,
    validate_evidence,
    validate_recipe,
)

SCHEMA_VERSION = 1
WORKER_SPEC_CONTRACT = "risk-score-adaptive-trial-worker-spec-v1"
COMMAND_INTENT_CONTRACT = "risk-score-adaptive-trial-command-intent-v1"
COMMAND_RECEIPT_CONTRACT = "risk-score-adaptive-trial-command-receipt-v1"
RUN_REQUEST_CONTRACT = "risk-score-adaptive-trial-run-request-v1"
CURRICULUM_MANIFEST_CONTRACT = "risk-score-adaptive-curriculum-manifest-v1"
ERROR_CONTRACT = "risk-score-adaptive-trial-worker-error-v1"
SPEC_CONTRACT = WORKER_SPEC_CONTRACT
RECEIPT_CONTRACT = COMMAND_RECEIPT_CONTRACT

GPU_ID = "7"
MAX_JSON_BYTES = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_STAGES = (
    "curriculum",
    "trainer",
    "export",
    "model_probe",
    "fixed_validation",
    "discovery",
)
_EVIDENCE_STAGES = frozenset({"fixed_validation", "discovery"})
_FORBIDDEN_COMMAND_TERMS = frozenset({"audit", "confirmation", "holdout", "promotion"})

_SPEC_FIELDS = {
    "autonomy_policy",
    "checkpoint_stable_seconds",
    "checkpoint_timeout_seconds",
    "contract",
    "curriculum_argv_template",
    "deployment",
    "discovery_argv_template",
    "drain_timeout_seconds",
    "export_argv_template",
    "fixed_validation_argv_template",
    "fixed_validation_manifest",
    "gpu_id",
    "katago_binary",
    "model_probe_argv_template",
    "poll_interval_seconds",
    "python_executable",
    "schema_version",
    "spec_sha256",
    "trainer_argv_template",
}
_FILE_BINDING_FIELDS = {"path", "sha256"}
_IDENTITY_BINDING_FIELDS = {"identity", "path", "sha256"}
_DEPLOYMENT_FIELDS = {
    "repository_path",
    "source_revision",
    "source_sha256",
}
_TRIAL_FIELDS = {
    "admitted_data",
    "champion_checkpoint",
    "contract",
    "epoch_id",
    "isolation_root",
    "manifest_sha256",
    "parent_champion_model_sha256",
    "policy_hash",
    "recipe_path",
    "recipe_sha256",
    "schema_version",
    "trial_id",
}
_RECIPE_FIELDS = {
    "contract",
    "policy_hash",
    "recipe",
    "recipe_sha256",
    "schema_version",
}
_COMMAND_RECEIPT_FIELDS = {
    "argv_sha256",
    "contract",
    "inputs_sha256",
    "outputs",
    "receipt_sha256",
    "returncode",
    "round_index",
    "schema_version",
    "stage",
    "status",
    "trial_id",
    "trial_manifest_path",
    "trial_manifest_sha256",
    "work_id",
    "worker_spec_sha256",
}
_COMMAND_INTENT_FIELDS = {
    "argv",
    "argv_sha256",
    "contract",
    "inputs_sha256",
    "intent_sha256",
    "receipt_path",
    "round_index",
    "run_request_sha256",
    "schema_version",
    "stage",
    "trial_id",
    "trial_manifest_path",
    "trial_manifest_sha256",
    "work_id",
    "worker_spec_sha256",
}
_RUN_REQUEST_FIELDS = {
    "contract",
    "deadline_unix",
    "request_sha256",
    "reservation_gpu_seconds",
    "result_path",
    "round_index",
    "schema_version",
    "started_at_unix",
    "trial_id",
    "trial_manifest_path",
    "trial_manifest_sha256",
    "work_id",
    "worker_spec_path",
    "worker_spec_sha256",
}
_CURRICULUM_FIELDS = {
    "admitted_data_manifest",
    "contract",
    "curriculum_directory",
    "files",
    "manifest_sha256",
    "recipe_sha256",
    "round_index",
    "schema_version",
    "shuffle_argv",
    "trial_id",
    "trial_manifest_sha256",
    "worker_spec_sha256",
}
_CURRICULUM_FILE_FIELDS = {"path", "sha256", "size"}

_COMMON_TEMPLATE_FIELDS = frozenset(
    {
        "admitted_data_manifest_path",
        "admitted_data_manifest_sha256",
        "autonomy_policy_identity",
        "autonomy_policy_path",
        "autonomy_policy_sha256",
        "candidate_checkpoint_path",
        "candidate_model_path",
        "champion_checkpoint_path",
        "champion_checkpoint_sha256",
        "checkpoint_path",
        "curriculum_data_path",
        "curriculum_manifest_path",
        "deadline_unix",
        "discovery_evidence_path",
        "epoch_id",
        "fixed_validation_directory",
        "fixed_validation_evidence_path",
        "fixed_validation_manifest_identity",
        "fixed_validation_manifest_path",
        "fixed_validation_manifest_sha256",
        "gpu_id",
        "initial_checkpoint_path",
        "initial_checkpoint_sha256",
        "inputs_sha256",
        "isolation_root",
        "katago_binary",
        "katago_binary_sha256",
        "model_probe_path",
        "python_executable",
        "python_executable_sha256",
        "receipt_path",
        "recipe_path",
        "recipe_sha256",
        "repository_path",
        "reservation_gpu_seconds",
        "round_index",
        "round_root",
        "run_request_sha256",
        "source_revision",
        "stage",
        "started_at_unix",
        "trial_id",
        "trial_manifest_path",
        "trial_manifest_sha256",
        "work_id",
        "worker_spec_path",
        "worker_spec_sha256",
    }
)
_COMMON_REQUIRED_TEMPLATE_FIELDS = frozenset(
    {
        "inputs_sha256",
        "receipt_path",
        "round_index",
        "trial_manifest_sha256",
        "work_id",
        "worker_spec_sha256",
    }
)
_STAGE_REQUIRED_TEMPLATE_FIELDS = {
    "curriculum": frozenset(
        {
            "admitted_data_manifest_path",
            "admitted_data_manifest_sha256",
            "curriculum_data_path",
            "curriculum_manifest_path",
        }
    ),
    "trainer": frozenset(
        {
            "checkpoint_path",
            "curriculum_manifest_path",
            "deadline_unix",
            "initial_checkpoint_path",
            "initial_checkpoint_sha256",
            "reservation_gpu_seconds",
        }
    ),
    "export": frozenset(
        {
            "candidate_checkpoint_path",
            "candidate_model_path",
            "checkpoint_path",
        }
    ),
    "model_probe": frozenset(
        {
            "candidate_model_path",
            "katago_binary",
            "model_probe_path",
        }
    ),
    "fixed_validation": frozenset(
        {
            "candidate_model_path",
            "fixed_validation_evidence_path",
            "fixed_validation_manifest_path",
            "fixed_validation_manifest_sha256",
            "katago_binary",
        }
    ),
    "discovery": frozenset(
        {
            "candidate_model_path",
            "discovery_evidence_path",
            "katago_binary",
        }
    ),
}

_SHUFFLE_FLAGS = (
    "--recent-window-samples",
    "--recent-fraction",
    "--historical-window-samples",
    "--historical-fraction",
)
_TRAINER_FLAGS = (
    "--bucket-cap-samples",
    "--bucket-ratio",
    "--export-cadence-epochs",
    "--learning-rate-scale",
    "--learning-rate-schedule",
    "--swa-cadence-samples",
)
_RESERVED_RECIPE_FLAGS = frozenset(_SHUFFLE_FLAGS + _TRAINER_FLAGS)


class AdaptiveTrialWorkerError(RuntimeError):
    """Fail-closed worker error with a stable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
        state_unambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.state_unambiguous = state_unambiguous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract": ERROR_CONTRACT,
            "error": {
                "code": self.code,
                "details": self.details,
                "message": self.message,
            },
            "schema_version": SCHEMA_VERSION,
        }


class WorkerSpecError(AdaptiveTrialWorkerError, ValueError):
    """The immutable adaptive-trial worker specification is invalid."""


class TrialBindingError(AdaptiveTrialWorkerError, ValueError):
    """A controller-owned trial input is malformed or changed."""


class CommandReceiptError(AdaptiveTrialWorkerError, ValueError):
    """A command receipt is absent, malformed, or contradicts its launch."""


class CommandFailure(AdaptiveTrialWorkerError):
    """A child exited completely but did not complete its stage."""


class AmbiguousTrialState(AdaptiveTrialWorkerError):
    """A live process or checkpoint state cannot be proved safe."""


class FrozenInputChanged(AdaptiveTrialWorkerError):
    """A hash-bound deployment or trial input drifted during execution."""


class CommandSpawnError(AdaptiveTrialWorkerError):
    """A child could not be spawned and no child lifetime began."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str
    identity: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        result = {"path": str(self.path), "sha256": self.sha256}
        if self.identity is not None:
            result["identity"] = self.identity
        return result


@dataclass(frozen=True)
class DeploymentBinding:
    repository_path: Path
    source_revision: str
    source_sha256: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "repository_path": str(self.repository_path),
            "source_revision": self.source_revision,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class WorkerSpec:
    path: Path
    file_sha256: str
    spec_sha256: str
    deployment: DeploymentBinding
    autonomy_policy: FileBinding
    python_executable: FileBinding
    katago_binary: FileBinding
    fixed_validation_manifest: FileBinding
    fixed_validation_directory: Path
    gpu_id: str
    curriculum_argv_template: Tuple[str, ...]
    trainer_argv_template: Tuple[str, ...]
    export_argv_template: Tuple[str, ...]
    model_probe_argv_template: Tuple[str, ...]
    fixed_validation_argv_template: Tuple[str, ...]
    discovery_argv_template: Tuple[str, ...]
    poll_interval_seconds: float
    drain_timeout_seconds: float
    checkpoint_timeout_seconds: float
    checkpoint_stable_seconds: float
    raw: Mapping[str, Any]

    @property
    def identity(self) -> str:
        return self.spec_sha256

    @property
    def source_path(self) -> Path:
        return self.path

    @property
    def repository_path(self) -> Path:
        return self.deployment.repository_path

    def template(self, stage: str) -> Tuple[str, ...]:
        if stage not in _STAGES:
            raise WorkerSpecError(
                "invalid_stage", f"unsupported command stage: {stage}"
            )
        return getattr(self, f"{stage}_argv_template")


@dataclass(frozen=True)
class RecipeArguments:
    shuffle_argv: Tuple[str, ...]
    trainer_argv: Tuple[str, ...]
    recipe: Mapping[str, Any]
    recipe_sha256: str


@dataclass(frozen=True)
class TrialContext:
    path: Path
    manifest_sha256: str
    trial_id: str
    epoch_id: str
    isolation_root: Path
    champion_checkpoint: FileBinding
    admitted_data_manifest: FileBinding
    recipe_path: Path
    recipe_sha256: str
    recipe: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TrialPaths:
    round_root: Path
    request: Path
    lock: Path
    intents: Path
    receipts: Path
    logs: Path
    curriculum_directory: Path
    curriculum_manifest: Path
    checkpoint: Path
    candidate_model: Path
    candidate_checkpoint: Path
    model_probe: Path
    fixed_validation_evidence: Path
    discovery_evidence: Path
    result: Path

    def intent(self, stage: str) -> Path:
        return self.intents / f"{stage}.json"

    def receipt(self, stage: str) -> Path:
        return self.receipts / f"{stage}.json"

    def log(self, stage: str) -> Path:
        return self.logs / f"{stage}.log"


class RunningCommand(Protocol):
    def poll(self) -> Optional[int]: ...

    def send_signal(self, sig: int) -> None: ...

    def process_group_alive(self) -> bool: ...


class CommandRunner(Protocol):
    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        log_path: Path,
    ) -> RunningCommand: ...


class _SubprocessCommand:
    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self.process = process

    def poll(self) -> Optional[int]:
        value = self.process.poll()
        return None if value is None else int(value)

    def send_signal(self, sig: int) -> None:
        if sig == signal.SIGKILL:
            raise AmbiguousTrialState(
                "sigkill_forbidden",
                "adaptive trial workers must never send SIGKILL",
            )
        if self.process.poll() is not None:
            return
        os.killpg(self.process.pid, sig)

    def process_group_alive(self) -> bool:
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


class SubprocessCommandRunner:
    """Production argv-only runner; every launch uses ``shell=False``."""

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        log_path: Path,
    ) -> RunningCommand:
        command = _validate_expanded_argv(argv, "child command")
        _ensure_directory(log_path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(os.fspath(log_path), flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("command log is not a regular file")
            with os.fdopen(descriptor, "ab", buffering=0) as output:
                descriptor = -1
                process = subprocess.Popen(
                    list(command),
                    cwd=os.fspath(cwd),
                    env=dict(environment),
                    stdout=output,
                    stderr=output,
                    start_new_session=True,
                    shell=False,
                )
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            raise CommandSpawnError(
                "command_spawn_failed",
                f"could not spawn child command: {exc}",
                state_unambiguous=True,
            ) from exc
        return _SubprocessCommand(process)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(
    value: Any,
    role: str,
    *,
    error_type: type[AdaptiveTrialWorkerError] = WorkerSpecError,
) -> str:
    if not _is_sha256(value):
        raise error_type("invalid_hash", f"{role} must be a lowercase SHA-256")
    return str(value)


def _require_id(
    value: Any,
    role: str,
    *,
    error_type: type[AdaptiveTrialWorkerError] = TrialBindingError,
) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise error_type(
            "unsafe_identifier",
            f"{role} must be a safe nonempty path component",
        )
    return value


def _require_number(
    value: Any,
    role: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    error_type: type[AdaptiveTrialWorkerError] = WorkerSpecError,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise error_type("invalid_number", f"{role} must be finite")
    result = float(value)
    if positive and result <= 0:
        raise error_type("invalid_number", f"{role} must be positive")
    if nonnegative and result < 0:
        raise error_type("invalid_number", f"{role} must be nonnegative")
    return result


def _require_nonnegative_integer(
    value: Any,
    role: str,
    *,
    error_type: type[AdaptiveTrialWorkerError] = TrialBindingError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_type(
            "invalid_integer",
            f"{role} must be a nonnegative integer",
        )
    return value


def _reject_symlink_ancestors(
    path: Path,
    role: str,
    *,
    error_type: type[AdaptiveTrialWorkerError],
) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise error_type(
                "unsafe_path",
                f"{role} has a symlinked path component",
                details={"component": str(current)},
            )
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(
    value: Any,
    role: str,
    *,
    error_type: type[AdaptiveTrialWorkerError] = WorkerSpecError,
) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise error_type("unsafe_path", f"{role} must be an absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if (
        not path.is_absolute()
        or path != normalized
        or "\x00" in os.fspath(path)
        or "\n" in os.fspath(path)
        or "\r" in os.fspath(path)
    ):
        raise error_type(
            "unsafe_path",
            f"{role} must be absolute and lexically normalized",
        )
    _reject_symlink_ancestors(path, role, error_type=error_type)
    return path


def _required_file(
    value: Any,
    role: str,
    *,
    expected_sha256: Optional[str] = None,
    error_type: type[AdaptiveTrialWorkerError] = WorkerSpecError,
) -> Path:
    path = _absolute_path(value, role, error_type=error_type)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise error_type("missing_file", f"{role} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise error_type(
            "unsafe_path",
            f"{role} must be a regular non-symlink file",
        )
    if expected_sha256 is not None and _stable_file_sha256(path) != expected_sha256:
        raise error_type(
            "hash_changed",
            f"{role} does not match its SHA-256 binding",
        )
    return path


def _required_directory(
    value: Any,
    role: str,
    *,
    error_type: type[AdaptiveTrialWorkerError] = WorkerSpecError,
) -> Path:
    path = _absolute_path(value, role, error_type=error_type)
    if path.is_symlink() or not path.is_dir():
        raise error_type(
            "unsafe_path",
            f"{role} must be an existing non-symlink directory",
        )
    return path


def _stable_file_sha256(path: Path) -> str:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular non-symlink file")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        after = path.lstat()
    except OSError as exc:
        raise FrozenInputChanged(
            "file_unverifiable",
            f"cannot stably hash {path}: {exc}",
        ) from exc
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_key != after_key:
        raise FrozenInputChanged(
            "file_changed_while_hashing",
            f"file changed while it was hashed: {path}",
        )
    return digest.hexdigest()


def _stable_file_identity(path: Path) -> Optional[Dict[str, Any]]:
    if not os.path.lexists(os.fspath(path)):
        return None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        after = path.lstat()
    except OSError:
        return None
    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_key != after_key:
        return None
    return {
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "size": after.st_size,
    }


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
    missing = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise AmbiguousTrialState(
                "unsafe_output_path",
                f"output directory has a symlinked component: {current}",
            )
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise AmbiguousTrialState(
            "unsafe_output_path",
            f"output directory ancestor is unsafe: {current}",
        )
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)
    if path.is_symlink() or not path.is_dir():
        raise AmbiguousTrialState(
            "unsafe_output_path",
            f"output directory is unsafe: {path}",
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _strictly_within(path: Path, root: Path) -> bool:
    if path == root:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise WorkerSpecError(
            "source_revision_unavailable",
            "cannot resolve the deployed repository revision",
        )
    return completed.stdout.strip()


def _git_status(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise WorkerSpecError(
            "source_status_unavailable",
            "cannot inspect the deployed repository status",
        )
    return completed.stdout


def _binding(
    value: Any,
    role: str,
    *,
    identity: bool,
) -> FileBinding:
    expected_fields = _IDENTITY_BINDING_FIELDS if identity else _FILE_BINDING_FIELDS
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise WorkerSpecError(
            "invalid_worker_spec",
            f"{role} binding fields differ from the worker schema",
        )
    expected = _require_sha256(value["sha256"], f"{role} file hash")
    path = _required_file(value["path"], role, expected_sha256=expected)
    binding_identity = (
        _require_sha256(value["identity"], f"{role} identity") if identity else None
    )
    return FileBinding(path, expected, binding_identity)


def _template_fields(template: Sequence[str]) -> frozenset[str]:
    formatter = string.Formatter()
    fields = set()
    for part in template:
        try:
            parsed = formatter.parse(part)
        except ValueError as exc:
            raise WorkerSpecError(
                "invalid_command_template",
                f"command template has invalid formatting: {exc}",
            ) from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                field_name not in _COMMON_TEMPLATE_FIELDS
                or format_spec
                or conversion is not None
                or "." in field_name
                or "[" in field_name
            ):
                raise WorkerSpecError(
                    "forbidden_placeholder",
                    f"unsupported command placeholder: {field_name!r}",
                )
            fields.add(field_name)
    return frozenset(fields)


def _flag_name(part: str) -> str:
    return part.split("=", 1)[0].lower().replace("_", "-")


def _validate_template(value: Any, stage: str) -> Tuple[str, ...]:
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
        raise WorkerSpecError(
            "invalid_command_template",
            f"{stage} command must be a nonempty argv string array",
        )
    template = tuple(value)
    fields = _template_fields(template)
    required = _COMMON_REQUIRED_TEMPLATE_FIELDS | _STAGE_REQUIRED_TEMPLATE_FIELDS[stage]
    missing = sorted(required - fields)
    if missing:
        raise WorkerSpecError(
            "unbound_command_receipt",
            f"{stage} command omits required hash/receipt placeholders",
            details={"missing": missing},
        )
    if template[0] not in {"{python_executable}", "{katago_binary}"}:
        raise WorkerSpecError(
            "unbound_executable",
            f"{stage} command must launch a hash-bound executable",
        )
    for part in template:
        lowered = part.lower().replace("-", "_")
        tokens = set(re.split(r"[^a-z0-9_]+", lowered))
        if any(
            term in tokens or f"_{term}" in lowered or f"{term}_" in lowered
            for term in _FORBIDDEN_COMMAND_TERMS
        ):
            raise WorkerSpecError(
                "protected_surface_forbidden",
                f"{stage} command references a forbidden protected surface",
                details={"argument": part},
            )
        if _flag_name(part) in _RESERVED_RECIPE_FLAGS:
            raise WorkerSpecError(
                "recipe_flag_in_template",
                f"{stage} template may not override worker-translated recipe flags",
                details={"argument": part},
            )
    return template


def _load_validation_binding(binding: FileBinding) -> Tuple[Mapping[str, Any], Path]:
    try:
        raw = load_canonical_json(binding.path, "fixed validation manifest")
        if (
            raw.get("contract") != VALIDATION_MANIFEST_CONTRACT
            or raw.get("schema_version") != SCHEMA_VERSION
            or raw.get("manifest_sha256") != binding.identity
        ):
            raise ValueError("fixed-validation manifest identity is invalid")
        directory = _required_directory(
            raw.get("directory"),
            "fixed validation directory",
        )
        validated = validate_validation_manifest(directory, binding.path)
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise WorkerSpecError(
            "invalid_fixed_validation_manifest",
            f"fixed-validation manifest is invalid: {exc}",
        ) from exc
    if dict(validated) != raw:
        raise WorkerSpecError(
            "invalid_fixed_validation_manifest",
            "fixed-validation validator returned contradictory content",
        )
    return raw, directory


def _assert_deployment(
    deployment: DeploymentBinding,
    *,
    revision_reader: Callable[[Path], str],
    repository_status_reader: Callable[[Path], str],
    error_type: type[AdaptiveTrialWorkerError] = WorkerSpecError,
) -> None:
    try:
        revision = revision_reader(deployment.repository_path)
        status = repository_status_reader(deployment.repository_path)
    except AdaptiveTrialWorkerError:
        raise
    except BaseException as exc:
        raise error_type(
            "source_verification_failed",
            f"cannot verify deployed source: {exc}",
        ) from exc
    if revision != deployment.source_revision:
        raise error_type(
            "source_revision_changed",
            "deployed repository revision differs from the worker specification",
        )
    if status:
        raise error_type(
            "source_checkout_dirty",
            "deployed repository has uncommitted or untracked changes",
        )


def load_worker_spec(
    path: Union[str, os.PathLike],
    *,
    expected_spec_sha256: Optional[str] = None,
    revision_reader: Callable[[Path], str] = _git_revision,
    repository_status_reader: Callable[[Path], str] = _git_status,
) -> WorkerSpec:
    """Load and fully validate one canonical self-hashed worker spec."""

    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = requested.resolve()
    source = _required_file(requested, "adaptive trial worker specification")
    try:
        raw = load_canonical_json(source, "adaptive trial worker specification")
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise WorkerSpecError(
            "invalid_worker_spec",
            f"cannot load adaptive trial worker specification: {exc}",
        ) from exc
    if set(raw) != _SPEC_FIELDS:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "worker specification fields differ from the strict schema",
            details={
                "extra": sorted(set(raw) - _SPEC_FIELDS),
                "missing": sorted(_SPEC_FIELDS - set(raw)),
            },
        )
    body = dict(raw)
    supplied_hash = _require_sha256(
        body.pop("spec_sha256"),
        "worker specification identity",
    )
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["contract"] != WORKER_SPEC_CONTRACT
        or canonical_sha256(body) != supplied_hash
    ):
        raise WorkerSpecError(
            "invalid_worker_spec",
            "worker specification contract or self-hash is invalid",
        )
    if expected_spec_sha256 is not None and supplied_hash != _require_sha256(
        expected_spec_sha256,
        "expected worker specification identity",
    ):
        raise WorkerSpecError(
            "worker_spec_hash_mismatch",
            "worker specification identity is not the CLI-pinned identity",
        )

    deployment_value = raw["deployment"]
    if (
        not isinstance(deployment_value, Mapping)
        or set(deployment_value) != _DEPLOYMENT_FIELDS
    ):
        raise WorkerSpecError(
            "invalid_worker_spec",
            "deployment binding fields differ from the schema",
        )
    repository = _required_directory(
        deployment_value["repository_path"],
        "deployed repository",
    )
    revision = deployment_value["source_revision"]
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "deployment source revision must be a lowercase Git object ID",
        )
    source_hash = _require_sha256(
        deployment_value["source_sha256"],
        "deployment source hash",
    )
    if hashlib.sha256(revision.encode("utf-8")).hexdigest() != source_hash:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "deployment source hash does not bind the source revision",
        )
    deployment = DeploymentBinding(repository, revision, source_hash)
    _assert_deployment(
        deployment,
        revision_reader=revision_reader,
        repository_status_reader=repository_status_reader,
    )

    policy = _binding(raw["autonomy_policy"], "autonomy policy", identity=True)
    if policy.identity != POLICY_HASH:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "worker specification does not bind the frozen autonomy policy",
        )
    try:
        loaded_policy = load_policy(policy.path)
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise WorkerSpecError(
            "invalid_worker_spec",
            f"worker autonomy policy is invalid: {exc}",
        ) from exc
    if canonical_sha256(loaded_policy) != policy.identity:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "worker autonomy policy identity changed during validation",
        )

    python = _binding(raw["python_executable"], "Python executable", identity=False)
    katago = _binding(raw["katago_binary"], "KataGo binary", identity=False)
    if not os.access(python.path, os.X_OK) or not os.access(katago.path, os.X_OK):
        raise WorkerSpecError(
            "invalid_worker_spec",
            "bound Python and KataGo files must both be executable",
        )
    fixed_validation = _binding(
        raw["fixed_validation_manifest"],
        "fixed validation manifest",
        identity=True,
    )
    _, fixed_directory = _load_validation_binding(fixed_validation)

    gpu_id = _require_id(raw["gpu_id"], "GPU ID", error_type=WorkerSpecError)
    if gpu_id != GPU_ID:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "adaptive trial worker must remain pinned to GPU ID 7",
        )

    templates = {
        stage: _validate_template(raw[f"{stage}_argv_template"], stage)
        for stage in _STAGES
    }
    poll = _require_number(
        raw["poll_interval_seconds"],
        "poll interval",
        positive=True,
    )
    drain = _require_number(
        raw["drain_timeout_seconds"],
        "drain timeout",
        positive=True,
    )
    checkpoint_timeout = _require_number(
        raw["checkpoint_timeout_seconds"],
        "checkpoint timeout",
        positive=True,
    )
    checkpoint_stable = _require_number(
        raw["checkpoint_stable_seconds"],
        "checkpoint stability interval",
        nonnegative=True,
    )
    if checkpoint_stable > checkpoint_timeout:
        raise WorkerSpecError(
            "invalid_worker_spec",
            "checkpoint stability interval exceeds its timeout",
        )

    return WorkerSpec(
        path=source,
        file_sha256=_stable_file_sha256(source),
        spec_sha256=supplied_hash,
        deployment=deployment,
        autonomy_policy=policy,
        python_executable=python,
        katago_binary=katago,
        fixed_validation_manifest=fixed_validation,
        fixed_validation_directory=fixed_directory,
        gpu_id=gpu_id,
        curriculum_argv_template=templates["curriculum"],
        trainer_argv_template=templates["trainer"],
        export_argv_template=templates["export"],
        model_probe_argv_template=templates["model_probe"],
        fixed_validation_argv_template=templates["fixed_validation"],
        discovery_argv_template=templates["discovery"],
        poll_interval_seconds=poll,
        drain_timeout_seconds=drain,
        checkpoint_timeout_seconds=checkpoint_timeout,
        checkpoint_stable_seconds=checkpoint_stable,
        raw=MappingProxyType(raw),
    )


def publish_worker_spec(
    path: Union[str, os.PathLike],
    *,
    repository_path: Union[str, os.PathLike],
    source_revision: str,
    autonomy_policy_path: Union[str, os.PathLike],
    python_executable: Union[str, os.PathLike],
    katago_binary: Union[str, os.PathLike],
    fixed_validation_manifest_path: Union[str, os.PathLike],
    curriculum_argv_template: Sequence[str],
    trainer_argv_template: Sequence[str],
    export_argv_template: Sequence[str],
    model_probe_argv_template: Sequence[str],
    fixed_validation_argv_template: Sequence[str],
    discovery_argv_template: Sequence[str],
    poll_interval_seconds: float = 0.25,
    drain_timeout_seconds: float = 120.0,
    checkpoint_timeout_seconds: float = 120.0,
    checkpoint_stable_seconds: float = 2.0,
    gpu_id: str = GPU_ID,
    revision_reader: Callable[[Path], str] = _git_revision,
    repository_status_reader: Callable[[Path], str] = _git_status,
) -> WorkerSpec:
    """Publish one immutable canonical worker specification."""

    destination = Path(path).resolve()
    repository = _required_directory(
        Path(repository_path).resolve(),
        "deployed repository",
    )
    if (
        not isinstance(source_revision, str)
        or _REVISION_RE.fullmatch(source_revision) is None
    ):
        raise WorkerSpecError(
            "invalid_worker_spec",
            "source revision must be a lowercase Git object ID",
        )
    deployment = DeploymentBinding(
        repository,
        source_revision,
        hashlib.sha256(source_revision.encode("utf-8")).hexdigest(),
    )
    _assert_deployment(
        deployment,
        revision_reader=revision_reader,
        repository_status_reader=repository_status_reader,
    )
    policy_path = _required_file(
        Path(autonomy_policy_path).resolve(),
        "autonomy policy",
    )
    policy = load_policy(policy_path)
    python_path = _required_file(
        Path(python_executable).resolve(),
        "Python executable",
    )
    katago_path = _required_file(Path(katago_binary).resolve(), "KataGo binary")
    validation_path = _required_file(
        Path(fixed_validation_manifest_path).resolve(),
        "fixed validation manifest",
    )
    validation = load_canonical_json(validation_path, "fixed validation manifest")
    identity = _require_sha256(
        validation.get("manifest_sha256"),
        "fixed validation manifest identity",
    )
    value: Dict[str, Any] = {
        "autonomy_policy": {
            "identity": canonical_sha256(policy),
            "path": str(policy_path),
            "sha256": _stable_file_sha256(policy_path),
        },
        "checkpoint_stable_seconds": checkpoint_stable_seconds,
        "checkpoint_timeout_seconds": checkpoint_timeout_seconds,
        "contract": WORKER_SPEC_CONTRACT,
        "curriculum_argv_template": list(curriculum_argv_template),
        "deployment": deployment.to_dict(),
        "discovery_argv_template": list(discovery_argv_template),
        "drain_timeout_seconds": drain_timeout_seconds,
        "export_argv_template": list(export_argv_template),
        "fixed_validation_argv_template": list(fixed_validation_argv_template),
        "fixed_validation_manifest": {
            "identity": identity,
            "path": str(validation_path),
            "sha256": _stable_file_sha256(validation_path),
        },
        "gpu_id": gpu_id,
        "katago_binary": {
            "path": str(katago_path),
            "sha256": _stable_file_sha256(katago_path),
        },
        "model_probe_argv_template": list(model_probe_argv_template),
        "poll_interval_seconds": poll_interval_seconds,
        "python_executable": {
            "path": str(python_path),
            "sha256": _stable_file_sha256(python_path),
        },
        "schema_version": SCHEMA_VERSION,
        "trainer_argv_template": list(trainer_argv_template),
    }
    value["spec_sha256"] = canonical_sha256(value)
    atomic_create_json(destination, value)
    return load_worker_spec(
        destination,
        expected_spec_sha256=value["spec_sha256"],
        revision_reader=revision_reader,
        repository_status_reader=repository_status_reader,
    )


def _number_text(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


def translate_recipe(
    recipe: Mapping[str, Any],
    *,
    policy: Optional[Mapping[str, Any]] = None,
) -> RecipeArguments:
    """Translate only the seven frozen recipe knobs to explicit argv flags."""

    active_policy = load_policy() if policy is None else policy
    normalized = validate_recipe(recipe, active_policy)
    mixture = normalized["data_recency_window_mixture"]
    if not isinstance(mixture, Mapping) or set(mixture) != {
        "historical_fraction",
        "historical_window_samples",
        "recent_fraction",
        "recent_window_samples",
    }:
        raise TrialBindingError(
            "recipe_surface_forbidden",
            "data recency mixture fields differ from the frozen allowlist",
        )
    shuffle_argv = (
        "--recent-window-samples",
        _number_text(mixture["recent_window_samples"]),
        "--recent-fraction",
        _number_text(mixture["recent_fraction"]),
        "--historical-window-samples",
        _number_text(mixture["historical_window_samples"]),
        "--historical-fraction",
        _number_text(mixture["historical_fraction"]),
    )
    trainer_argv = (
        "--bucket-cap-samples",
        _number_text(normalized["bucket_cap_samples"]),
        "--bucket-ratio",
        _number_text(normalized["bucket_ratio"]),
        "--export-cadence-epochs",
        _number_text(normalized["export_cadence_epochs"]),
        "--learning-rate-scale",
        _number_text(normalized["learning_rate_scale"]),
        "--learning-rate-schedule",
        str(normalized["learning_rate_schedule"]),
        "--swa-cadence-samples",
        _number_text(normalized["swa_cadence_samples"]),
    )
    return RecipeArguments(
        shuffle_argv=shuffle_argv,
        trainer_argv=trainer_argv,
        recipe=MappingProxyType(normalized),
        recipe_sha256=canonical_sha256(normalized),
    )


recipe_to_argv = translate_recipe


def _trial_file_binding(value: Any, role: str) -> FileBinding:
    if not isinstance(value, Mapping) or set(value) != _FILE_BINDING_FIELDS:
        raise TrialBindingError(
            "invalid_trial_binding",
            f"{role} binding fields differ from the trial contract",
        )
    expected = _require_sha256(
        value["sha256"],
        f"{role} hash",
        error_type=TrialBindingError,
    )
    path = _required_file(
        value["path"],
        role,
        expected_sha256=expected,
        error_type=TrialBindingError,
    )
    return FileBinding(path, expected)


def load_trial_manifest(
    path: Union[str, os.PathLike],
    *,
    expected_manifest_sha256: str,
    policy_path: Union[str, os.PathLike],
) -> TrialContext:
    """Validate the adaptive-training trial, recipe, and immutable inputs."""

    source = _required_file(
        path,
        "adaptive trial manifest",
        error_type=TrialBindingError,
    )
    try:
        raw = load_canonical_json(source, "adaptive trial manifest")
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise TrialBindingError(
            "invalid_trial_manifest",
            f"cannot load adaptive trial manifest: {exc}",
        ) from exc
    if set(raw) != _TRIAL_FIELDS:
        raise TrialBindingError(
            "invalid_trial_manifest",
            "adaptive trial manifest fields differ from the contract",
        )
    body = dict(raw)
    supplied_hash = _require_sha256(
        body.pop("manifest_sha256"),
        "adaptive trial manifest identity",
        error_type=TrialBindingError,
    )
    if (
        supplied_hash
        != _require_sha256(
            expected_manifest_sha256,
            "expected adaptive trial manifest identity",
            error_type=TrialBindingError,
        )
        or canonical_sha256(body) != supplied_hash
        or raw["schema_version"] != SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["contract"] != TRIAL_CONTRACT
        or raw["policy_hash"] != POLICY_HASH
    ):
        raise TrialBindingError(
            "trial_manifest_hash_mismatch",
            "adaptive trial manifest contract or hash is invalid",
        )
    trial_id = _require_id(raw["trial_id"], "trial ID")
    epoch_id = _require_id(raw["epoch_id"], "epoch ID")
    _require_sha256(
        raw["parent_champion_model_sha256"],
        "parent champion model hash",
        error_type=TrialBindingError,
    )
    isolation_root = _required_directory(
        raw["isolation_root"],
        "trial isolation root",
        error_type=TrialBindingError,
    )
    if source != isolation_root / "trial.json":
        raise TrialBindingError(
            "trial_isolation_mismatch",
            "adaptive trial manifest is not canonical under its isolation root",
        )
    champion = _trial_file_binding(
        raw["champion_checkpoint"],
        "champion checkpoint",
    )
    admitted = _trial_file_binding(
        raw["admitted_data"],
        "admitted-data manifest",
    )
    recipe_path = _required_file(
        raw["recipe_path"],
        "adaptive recipe",
        error_type=TrialBindingError,
    )
    recipe_sha = _require_sha256(
        raw["recipe_sha256"],
        "adaptive recipe identity",
        error_type=TrialBindingError,
    )
    try:
        policy = load_policy(policy_path)
        recipe_raw = load_canonical_json(recipe_path, "adaptive recipe")
        if set(recipe_raw) != _RECIPE_FIELDS:
            raise ValueError("adaptive recipe fields differ from the contract")
        normalized = validate_recipe(recipe_raw["recipe"], policy)
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise TrialBindingError(
            "invalid_trial_recipe",
            f"adaptive trial recipe is invalid: {exc}",
        ) from exc
    if (
        recipe_raw["contract"] != RECIPE_CONTRACT
        or recipe_raw["schema_version"] != SCHEMA_VERSION
        or isinstance(recipe_raw["schema_version"], bool)
        or recipe_raw["policy_hash"] != POLICY_HASH
        or recipe_raw["recipe_sha256"] != recipe_sha
        or canonical_sha256(normalized) != recipe_sha
    ):
        raise TrialBindingError(
            "trial_recipe_hash_mismatch",
            "adaptive recipe no longer matches the manifest binding",
        )
    return TrialContext(
        path=source,
        manifest_sha256=supplied_hash,
        trial_id=trial_id,
        epoch_id=epoch_id,
        isolation_root=isolation_root,
        champion_checkpoint=champion,
        admitted_data_manifest=admitted,
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha,
        recipe=MappingProxyType(normalized),
        raw=MappingProxyType(raw),
    )


def trial_paths(
    context: TrialContext,
    round_index: int,
    *,
    result_path: Optional[Union[str, os.PathLike]] = None,
) -> TrialPaths:
    round_value = _require_nonnegative_integer(round_index, "round index")
    root = context.isolation_root / "adaptive-trial-worker" / f"round-{round_value:02d}"
    canonical_result = (
        context.isolation_root / "results" / f"round-{round_value:02d}.json"
    )
    if result_path is not None:
        supplied = _absolute_path(
            result_path,
            "trial result path",
            error_type=TrialBindingError,
        )
        if supplied != canonical_result:
            raise TrialBindingError(
                "result_path_mismatch",
                "trial result path differs from the adaptive service contract",
            )
    return TrialPaths(
        round_root=root,
        request=root / "request.json",
        lock=root / ".worker.lock",
        intents=root / "intents",
        receipts=root / "receipts",
        logs=root / "logs",
        curriculum_directory=root / "curriculum" / "data",
        curriculum_manifest=root / "curriculum" / "manifest.json",
        checkpoint=root / "training" / "model.ckpt",
        candidate_model=root / "candidate" / "model.bin.gz",
        candidate_checkpoint=root / "candidate" / "model.ckpt",
        model_probe=root / "probe" / "model-probe.json",
        fixed_validation_evidence=root / "evidence" / "fixed-validation.json",
        discovery_evidence=root / "evidence" / "discovery.json",
        result=canonical_result,
    )


def _directory_inventory(root: Path) -> Tuple[Dict[str, Any], ...]:
    if root.is_symlink() or not root.is_dir():
        raise CommandReceiptError(
            "invalid_curriculum",
            "curriculum directory must be a non-symlink directory",
        )
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CommandReceiptError(
                "invalid_curriculum",
                f"curriculum contains a symlink: {path}",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise CommandReceiptError(
                "invalid_curriculum",
                f"curriculum contains a non-regular entry: {path}",
            )
        relative = path.relative_to(root).as_posix()
        metadata = path.stat()
        files.append(
            {
                "path": relative,
                "sha256": _stable_file_sha256(path),
                "size": metadata.st_size,
            }
        )
    if not files:
        raise CommandReceiptError(
            "invalid_curriculum",
            "curriculum materialization produced no files",
        )
    return tuple(files)


def publish_curriculum_manifest(
    path: Union[str, os.PathLike],
    *,
    directory: Union[str, os.PathLike],
    worker_spec_sha256: str,
    trial_manifest_sha256: str,
    trial_id: str,
    round_index: int,
    admitted_data_manifest: Mapping[str, str],
    recipe_sha256: str,
    shuffle_argv: Sequence[str],
) -> Mapping[str, Any]:
    """Publish the canonical materialized-curriculum inventory."""

    destination = Path(path)
    root = Path(directory)
    value: Dict[str, Any] = {
        "admitted_data_manifest": dict(admitted_data_manifest),
        "contract": CURRICULUM_MANIFEST_CONTRACT,
        "curriculum_directory": str(root),
        "files": list(_directory_inventory(root)),
        "recipe_sha256": recipe_sha256,
        "round_index": round_index,
        "schema_version": SCHEMA_VERSION,
        "shuffle_argv": list(shuffle_argv),
        "trial_id": trial_id,
        "trial_manifest_sha256": trial_manifest_sha256,
        "worker_spec_sha256": worker_spec_sha256,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    atomic_create_json(destination, value)
    return value


def publish_command_receipt(
    path: Union[str, os.PathLike],
    *,
    stage: str,
    worker_spec_sha256: str,
    trial_manifest_path: Union[str, os.PathLike],
    trial_manifest_sha256: str,
    trial_id: str,
    work_id: str,
    round_index: int,
    argv: Sequence[str],
    inputs_sha256: str,
    returncode: int,
    status: str,
    outputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Publish one canonical command receipt for a configured stage adapter."""

    command = _validate_expanded_argv(argv, "receipt argv")
    value: Dict[str, Any] = {
        "argv_sha256": canonical_sha256(list(command)),
        "contract": COMMAND_RECEIPT_CONTRACT,
        "inputs_sha256": inputs_sha256,
        "outputs": dict(outputs),
        "returncode": returncode,
        "round_index": round_index,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "trial_id": trial_id,
        "trial_manifest_path": str(trial_manifest_path),
        "trial_manifest_sha256": trial_manifest_sha256,
        "work_id": work_id,
        "worker_spec_sha256": worker_spec_sha256,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    atomic_create_json(path, value)
    return value


def _validate_expanded_argv(argv: Sequence[str], role: str) -> Tuple[str, ...]:
    if (
        not isinstance(argv, (tuple, list))
        or not argv
        or any(
            not isinstance(part, str)
            or not part
            or "\x00" in part
            or "\n" in part
            or "\r" in part
            for part in argv
        )
    ):
        raise AdaptiveTrialWorkerError(
            "invalid_command",
            f"{role} must be a nonempty argv string array",
        )
    return tuple(argv)


def _load_command_receipt(path: Path) -> Dict[str, Any]:
    try:
        raw = load_canonical_json(path, "adaptive trial command receipt")
    except (AdaptiveTrainingError, OSError, ValueError) as exc:
        raise CommandReceiptError(
            "invalid_command_receipt",
            f"cannot load command receipt: {exc}",
        ) from exc
    if set(raw) != _COMMAND_RECEIPT_FIELDS:
        raise CommandReceiptError(
            "invalid_command_receipt",
            "command receipt fields differ from the strict schema",
        )
    body = dict(raw)
    supplied = _require_sha256(
        body.pop("receipt_sha256"),
        "command receipt identity",
        error_type=CommandReceiptError,
    )
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["contract"] != COMMAND_RECEIPT_CONTRACT
        or type(raw["round_index"]) is not int
        or raw["round_index"] < 0
        or canonical_sha256(body) != supplied
    ):
        raise CommandReceiptError(
            "invalid_command_receipt",
            "command receipt contract or self-hash is invalid",
        )
    return raw


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


class _ExecutionLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock = _process_lock(path)
        self._descriptor = -1

    def __enter__(self) -> "_ExecutionLock":
        if fcntl is None:
            raise AmbiguousTrialState(
                "locking_unavailable",
                "POSIX advisory locking is required for trial execution",
            )
        if not self._thread_lock.acquire(blocking=False):
            raise AmbiguousTrialState(
                "worker_busy",
                "another worker is executing this exact trial round",
            )
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            self._descriptor = os.open(os.fspath(self.path), flags, 0o600)
            if not stat.S_ISREG(os.fstat(self._descriptor).st_mode):
                raise AmbiguousTrialState(
                    "unsafe_lock",
                    "trial worker lock is not a regular file",
                )
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise AmbiguousTrialState(
                    "worker_busy",
                    "another process owns this exact trial round",
                ) from exc
            return self
        except BaseException:
            if self._descriptor >= 0:
                os.close(self._descriptor)
                self._descriptor = -1
            self._thread_lock.release()
            raise

    def __exit__(self, *_args: Any) -> None:
        if self._descriptor >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = -1
        self._thread_lock.release()


class AdaptiveTrialWorker:
    """Execute one exact adaptive trial round and publish one exact result."""

    def __init__(
        self,
        spec: Union[WorkerSpec, str, os.PathLike],
        *,
        expected_spec_sha256: Optional[str] = None,
        trial_manifest_path: Union[str, os.PathLike],
        expected_trial_manifest_sha256: Optional[str] = None,
        trial_manifest_sha256: Optional[str] = None,
        work_id: str,
        round_index: int,
        reservation_gpu_seconds: Optional[float] = None,
        gpu_time_reservation_seconds: Optional[float] = None,
        deadline_unix: float,
        result_path: Union[str, os.PathLike],
        command_runner: Optional[CommandRunner] = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        revision_reader: Callable[[Path], str] = _git_revision,
        repository_status_reader: Callable[[Path], str] = _git_status,
        install_signal_handlers: bool = True,
    ) -> None:
        raw_invocation_time = clock()
        if isinstance(raw_invocation_time, bool):
            raise TrialBindingError(
                "invalid_clock",
                "worker clock returned a non-finite invocation time",
            )
        try:
            invocation_started_at = float(raw_invocation_time)
        except (TypeError, ValueError) as exc:
            raise TrialBindingError(
                "invalid_clock",
                "worker clock returned a nonnumeric invocation time",
            ) from exc
        if not math.isfinite(invocation_started_at):
            raise TrialBindingError(
                "invalid_clock",
                "worker clock returned a non-finite invocation time",
            )
        self._revision_reader = revision_reader
        self._repository_status_reader = repository_status_reader
        self.spec = (
            spec
            if isinstance(spec, WorkerSpec)
            else load_worker_spec(
                spec,
                expected_spec_sha256=expected_spec_sha256,
                revision_reader=revision_reader,
                repository_status_reader=repository_status_reader,
            )
        )
        if isinstance(spec, WorkerSpec) and expected_spec_sha256 is not None:
            if self.spec.spec_sha256 != _require_sha256(
                expected_spec_sha256,
                "expected worker specification identity",
            ):
                raise WorkerSpecError(
                    "worker_spec_hash_mismatch",
                    "injected worker specification identity is not expected",
                )
        supplied_manifest_hashes = [
            value
            for value in (
                expected_trial_manifest_sha256,
                trial_manifest_sha256,
            )
            if value is not None
        ]
        if len(set(supplied_manifest_hashes)) != 1:
            raise TrialBindingError(
                "trial_manifest_hash_mismatch",
                "exactly one consistent trial manifest hash is required",
            )
        self.context = load_trial_manifest(
            trial_manifest_path,
            expected_manifest_sha256=supplied_manifest_hashes[0],
            policy_path=self.spec.autonomy_policy.path,
        )
        self.work_id = _require_id(work_id, "work ID")
        self.round_index = _require_nonnegative_integer(round_index, "round index")
        reservations = [
            value
            for value in (
                reservation_gpu_seconds,
                gpu_time_reservation_seconds,
            )
            if value is not None
        ]
        if len(reservations) != 1:
            raise TrialBindingError(
                "invalid_reservation",
                "exactly one GPU-time reservation is required",
            )
        self.reservation_gpu_seconds = _require_number(
            reservations[0],
            "GPU-time reservation",
            positive=True,
            error_type=TrialBindingError,
        )
        self.deadline_unix = _require_number(
            deadline_unix,
            "GPU-time deadline",
            nonnegative=True,
            error_type=TrialBindingError,
        )
        policy = load_policy(self.spec.autonomy_policy.path)
        round_reservations = policy["successive_halving"]["round_gpu_seconds"]
        if self.round_index >= len(round_reservations):
            raise TrialBindingError(
                "invalid_trial_round",
                "trial round exceeds the frozen successive-halving schedule",
            )
        if self.reservation_gpu_seconds != float(round_reservations[self.round_index]):
            raise TrialBindingError(
                "invalid_reservation",
                "GPU-time reservation differs from the frozen round budget",
            )
        self.paths = trial_paths(
            self.context,
            self.round_index,
            result_path=result_path,
        )
        protected_paths = (
            self.spec.repository_path,
            self.spec.path,
            self.spec.autonomy_policy.path,
            self.spec.python_executable.path,
            self.spec.katago_binary.path,
            self.spec.fixed_validation_manifest.path,
            self.spec.fixed_validation_directory,
            self.context.champion_checkpoint.path,
            self.context.admitted_data_manifest.path,
            self.context.recipe_path,
        )
        if any(
            _paths_overlap(self.context.isolation_root, path)
            for path in protected_paths
        ):
            raise TrialBindingError(
                "trial_isolation_overlap",
                "trial isolation root overlaps a frozen control-plane input",
            )
        self.recipe_arguments = translate_recipe(
            self.context.recipe,
            policy=policy,
        )
        if self.recipe_arguments.recipe_sha256 != self.context.recipe_sha256:
            raise TrialBindingError(
                "trial_recipe_hash_mismatch",
                "translated recipe does not match the trial binding",
            )
        self.runner = command_runner or SubprocessCommandRunner()
        self.clock = clock
        self.sleeper = sleeper
        self._invocation_started_at = invocation_started_at
        self.install_signal_handlers = install_signal_handlers
        self._active_process: Optional[RunningCommand] = None
        self._active_stage: Optional[str] = None
        self._interrupted = False
        self._sigint_sent_to_active = False
        self._signal_error: Optional[BaseException] = None
        self._run_started_at: Optional[float] = None
        self._gpu_started_at: Optional[float] = None
        self._gpu_ended_at: Optional[float] = None
        self._trainer_started = False
        self._trainer_checkpoint_proof: Optional[Mapping[str, Any]] = None
        self._validated_receipts: Dict[str, Mapping[str, Any]] = {}
        self._initial_checkpoint = self._resolve_initial_checkpoint()

    def _now(self) -> float:
        raw = self.clock()
        if isinstance(raw, bool):
            raise AdaptiveTrialWorkerError(
                "invalid_clock",
                "worker clock returned a boolean value",
            )
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise AdaptiveTrialWorkerError(
                "invalid_clock",
                "worker clock returned a nonnumeric value",
            ) from exc
        if not math.isfinite(value):
            raise AdaptiveTrialWorkerError(
                "invalid_clock",
                "worker clock returned a non-finite value",
            )
        return value

    def _resolve_initial_checkpoint(self) -> FileBinding:
        if self.round_index == 0:
            return self.context.champion_checkpoint
        previous_path = (
            self.context.isolation_root
            / "results"
            / f"round-{self.round_index - 1:02d}.json"
        )
        try:
            previous = load_trial_result(
                previous_path,
                expected_trial_id=self.context.trial_id,
                expected_epoch_id=self.context.epoch_id,
                expected_round_index=self.round_index - 1,
                expected_manifest_path=self.context.path,
                expected_manifest_sha256=self.context.manifest_sha256,
                require_candidate_bindings=True,
            )
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise TrialBindingError(
                "missing_resume_checkpoint",
                f"previous adaptive round has no valid resumable checkpoint: {exc}",
            ) from exc
        checkpoint = previous["candidate_checkpoint"]
        if checkpoint is None or checkpoint.get("resumable") is not True:
            raise TrialBindingError(
                "missing_resume_checkpoint",
                "previous adaptive round checkpoint is not resumable",
            )
        expected_checkpoint = (
            self.context.isolation_root
            / "adaptive-trial-worker"
            / f"round-{self.round_index - 1:02d}"
            / "candidate"
            / "model.ckpt"
        )
        if Path(checkpoint["path"]) != expected_checkpoint:
            raise TrialBindingError(
                "resume_checkpoint_path_mismatch",
                "previous round checkpoint is outside its canonical worker path",
            )
        return FileBinding(
            Path(checkpoint["path"]),
            checkpoint["sha256"],
        )

    def _request_value(self) -> Dict[str, Any]:
        if self._run_started_at is None:
            raise AdaptiveTrialWorkerError(
                "invalid_run_request",
                "trial run request has no measured start time",
            )
        value: Dict[str, Any] = {
            "contract": RUN_REQUEST_CONTRACT,
            "deadline_unix": self.deadline_unix,
            "reservation_gpu_seconds": self.reservation_gpu_seconds,
            "result_path": str(self.paths.result),
            "round_index": self.round_index,
            "schema_version": SCHEMA_VERSION,
            "started_at_unix": self._run_started_at,
            "trial_id": self.context.trial_id,
            "trial_manifest_path": str(self.context.path),
            "trial_manifest_sha256": self.context.manifest_sha256,
            "work_id": self.work_id,
            "worker_spec_path": str(self.spec.path),
            "worker_spec_sha256": self.spec.spec_sha256,
        }
        value["request_sha256"] = canonical_sha256(value)
        return value

    def _load_request(self) -> Dict[str, Any]:
        try:
            value = load_canonical_json(self.paths.request, "trial run request")
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise AmbiguousTrialState(
                "invalid_run_request",
                f"cannot validate prior trial run request: {exc}",
            ) from exc
        if set(value) != _RUN_REQUEST_FIELDS:
            raise AmbiguousTrialState(
                "invalid_run_request",
                "prior trial run request fields differ from the schema",
            )
        if (
            value.get("contract") != RUN_REQUEST_CONTRACT
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != SCHEMA_VERSION
            or type(value.get("round_index")) is not int
            or value["round_index"] < 0
        ):
            raise AmbiguousTrialState(
                "invalid_run_request",
                "prior trial run request contract is invalid",
            )
        body = dict(value)
        supplied = body.pop("request_sha256", None)
        started_at = _require_number(
            value.get("started_at_unix"),
            "prior run start",
            nonnegative=True,
            error_type=AmbiguousTrialState,
        )
        self._run_started_at = started_at
        self._gpu_started_at = started_at
        if (
            not _is_sha256(supplied)
            or canonical_sha256(body) != supplied
            or value != self._request_value()
        ):
            raise AmbiguousTrialState(
                "run_request_conflict",
                "prior trial run request contradicts this invocation",
            )
        observed_now = self._now()
        if started_at > observed_now + 1e-6:
            raise AmbiguousTrialState(
                "run_request_from_future",
                "prior trial run request starts in the future",
            )
        self._gpu_ended_at = max(started_at, observed_now)
        return value

    def _assert_frozen(self, *, validate_fixed_inventory: bool = False) -> None:
        if (
            self.spec.path.is_symlink()
            or not self.spec.path.is_file()
            or _stable_file_sha256(self.spec.path) != self.spec.file_sha256
        ):
            raise FrozenInputChanged(
                "worker_spec_changed",
                "worker specification changed during trial execution",
            )
        _assert_deployment(
            self.spec.deployment,
            revision_reader=self._revision_reader,
            repository_status_reader=self._repository_status_reader,
            error_type=FrozenInputChanged,
        )
        for binding, role in (
            (self.spec.autonomy_policy, "autonomy policy"),
            (self.spec.python_executable, "Python executable"),
            (self.spec.katago_binary, "KataGo binary"),
            (self.spec.fixed_validation_manifest, "fixed validation manifest"),
        ):
            if (
                binding.path.is_symlink()
                or not binding.path.is_file()
                or _stable_file_sha256(binding.path) != binding.sha256
            ):
                raise FrozenInputChanged(
                    "frozen_input_changed",
                    f"{role} changed during trial execution",
                )
        if not os.access(self.spec.python_executable.path, os.X_OK) or not os.access(
            self.spec.katago_binary.path, os.X_OK
        ):
            raise FrozenInputChanged(
                "frozen_input_changed",
                "bound Python or KataGo executable lost execute permission",
            )
        try:
            policy = load_policy(self.spec.autonomy_policy.path)
            if canonical_sha256(policy) != self.spec.autonomy_policy.identity:
                raise ValueError("policy identity changed")
            if validate_fixed_inventory:
                _load_validation_binding(self.spec.fixed_validation_manifest)
            current = load_trial_manifest(
                self.context.path,
                expected_manifest_sha256=self.context.manifest_sha256,
                policy_path=self.spec.autonomy_policy.path,
            )
        except (
            AdaptiveTrialWorkerError,
            AdaptiveTrainingError,
            OSError,
            ValueError,
        ) as exc:
            if isinstance(exc, FrozenInputChanged):
                raise
            raise FrozenInputChanged(
                "frozen_input_changed",
                f"frozen worker or trial input changed: {exc}",
            ) from exc
        if dict(current.raw) != dict(self.context.raw):
            raise FrozenInputChanged(
                "trial_manifest_changed",
                "adaptive trial manifest changed during execution",
            )

    def _command_values(
        self,
        stage: str,
        *,
        receipt_path: Path,
        inputs_sha256: str,
    ) -> Dict[str, str]:
        return {
            "admitted_data_manifest_path": str(
                self.context.admitted_data_manifest.path
            ),
            "admitted_data_manifest_sha256": (
                self.context.admitted_data_manifest.sha256
            ),
            "autonomy_policy_identity": str(self.spec.autonomy_policy.identity),
            "autonomy_policy_path": str(self.spec.autonomy_policy.path),
            "autonomy_policy_sha256": self.spec.autonomy_policy.sha256,
            "candidate_checkpoint_path": str(self.paths.candidate_checkpoint),
            "candidate_model_path": str(self.paths.candidate_model),
            "champion_checkpoint_path": str(self.context.champion_checkpoint.path),
            "champion_checkpoint_sha256": (self.context.champion_checkpoint.sha256),
            "checkpoint_path": str(self.paths.checkpoint),
            "curriculum_data_path": str(self.paths.curriculum_directory),
            "curriculum_manifest_path": str(self.paths.curriculum_manifest),
            "deadline_unix": _number_text(self.deadline_unix),
            "discovery_evidence_path": str(self.paths.discovery_evidence),
            "epoch_id": self.context.epoch_id,
            "fixed_validation_directory": str(self.spec.fixed_validation_directory),
            "fixed_validation_evidence_path": str(self.paths.fixed_validation_evidence),
            "fixed_validation_manifest_identity": str(
                self.spec.fixed_validation_manifest.identity
            ),
            "fixed_validation_manifest_path": str(
                self.spec.fixed_validation_manifest.path
            ),
            "fixed_validation_manifest_sha256": (
                self.spec.fixed_validation_manifest.sha256
            ),
            "gpu_id": self.spec.gpu_id,
            "initial_checkpoint_path": str(self._initial_checkpoint.path),
            "initial_checkpoint_sha256": self._initial_checkpoint.sha256,
            "inputs_sha256": inputs_sha256,
            "isolation_root": str(self.context.isolation_root),
            "katago_binary": str(self.spec.katago_binary.path),
            "katago_binary_sha256": self.spec.katago_binary.sha256,
            "model_probe_path": str(self.paths.model_probe),
            "python_executable": str(self.spec.python_executable.path),
            "python_executable_sha256": self.spec.python_executable.sha256,
            "receipt_path": str(receipt_path),
            "recipe_path": str(self.context.recipe_path),
            "recipe_sha256": self.context.recipe_sha256,
            "repository_path": str(self.spec.repository_path),
            "reservation_gpu_seconds": _number_text(self.reservation_gpu_seconds),
            "round_index": str(self.round_index),
            "round_root": str(self.paths.round_root),
            "run_request_sha256": self._request_value()["request_sha256"],
            "source_revision": self.spec.deployment.source_revision,
            "stage": stage,
            "started_at_unix": _number_text(self._run_started_at),
            "trial_id": self.context.trial_id,
            "trial_manifest_path": str(self.context.path),
            "trial_manifest_sha256": self.context.manifest_sha256,
            "work_id": self.work_id,
            "worker_spec_path": str(self.spec.path),
            "worker_spec_sha256": self.spec.spec_sha256,
        }

    def _stage_inputs(self, stage: str) -> Dict[str, Any]:
        common: Dict[str, Any] = {
            "stage": stage,
            "worker_spec_sha256": self.spec.spec_sha256,
            "run_request_sha256": self._request_value()["request_sha256"],
            "started_at_unix": self._run_started_at,
            "trial_manifest_sha256": self.context.manifest_sha256,
            "trial_id": self.context.trial_id,
            "work_id": self.work_id,
            "round_index": self.round_index,
            "recipe_sha256": self.context.recipe_sha256,
            "deadline_unix": self.deadline_unix,
            "reservation_gpu_seconds": self.reservation_gpu_seconds,
        }
        if stage == "curriculum":
            common.update(
                {
                    "admitted_data_manifest": (
                        self.context.admitted_data_manifest.to_dict()
                    ),
                    "curriculum_directory": str(self.paths.curriculum_directory),
                    "curriculum_manifest": str(self.paths.curriculum_manifest),
                    "shuffle_argv": list(self.recipe_arguments.shuffle_argv),
                }
            )
        elif stage == "trainer":
            common.update(
                {
                    "curriculum_manifest": {
                        "path": str(self.paths.curriculum_manifest),
                        "sha256": _stable_file_sha256(self.paths.curriculum_manifest),
                    },
                    "initial_checkpoint": self._initial_checkpoint.to_dict(),
                    "checkpoint_path": str(self.paths.checkpoint),
                    "trainer_argv": list(self.recipe_arguments.trainer_argv),
                }
            )
        elif stage == "export":
            common.update(
                {
                    "checkpoint": {
                        "path": str(self.paths.checkpoint),
                        "sha256": _stable_file_sha256(self.paths.checkpoint),
                    },
                    "candidate_model_path": str(self.paths.candidate_model),
                    "candidate_checkpoint_path": str(self.paths.candidate_checkpoint),
                }
            )
        elif stage == "model_probe":
            common.update(
                {
                    "candidate_model": {
                        "path": str(self.paths.candidate_model),
                        "sha256": _stable_file_sha256(self.paths.candidate_model),
                    },
                    "katago_binary": self.spec.katago_binary.to_dict(),
                    "probe_path": str(self.paths.model_probe),
                }
            )
        elif stage == "fixed_validation":
            common.update(
                {
                    "candidate_model": {
                        "path": str(self.paths.candidate_model),
                        "sha256": _stable_file_sha256(self.paths.candidate_model),
                    },
                    "fixed_validation_manifest": (
                        self.spec.fixed_validation_manifest.to_dict()
                    ),
                    "evidence_path": str(self.paths.fixed_validation_evidence),
                }
            )
        elif stage == "discovery":
            common.update(
                {
                    "candidate_model": {
                        "path": str(self.paths.candidate_model),
                        "sha256": _stable_file_sha256(self.paths.candidate_model),
                    },
                    "evidence_path": str(self.paths.discovery_evidence),
                }
            )
        else:  # pragma: no cover - callers use the frozen stage sequence.
            raise AdaptiveTrialWorkerError(
                "invalid_stage",
                f"unsupported stage: {stage}",
            )
        return common

    def _render_command(
        self,
        stage: str,
        *,
        receipt_path: Path,
        inputs_sha256: str,
    ) -> Tuple[str, ...]:
        values = self._command_values(
            stage,
            receipt_path=receipt_path,
            inputs_sha256=inputs_sha256,
        )
        try:
            rendered = tuple(
                part.format_map(values) for part in self.spec.template(stage)
            )
        except (KeyError, ValueError) as exc:
            raise WorkerSpecError(
                "invalid_command_template",
                f"cannot expand {stage} command: {exc}",
            ) from exc
        if stage == "curriculum":
            rendered += self.recipe_arguments.shuffle_argv
        elif stage == "trainer":
            rendered += self.recipe_arguments.trainer_argv
        command = _validate_expanded_argv(rendered, f"{stage} command")
        expected_executables = {
            str(self.spec.python_executable.path),
            str(self.spec.katago_binary.path),
        }
        if command[0] not in expected_executables:
            raise WorkerSpecError(
                "executable_binding_lost",
                f"{stage} command lost its hash-bound executable",
            )
        return command

    def _intent_value(
        self,
        stage: str,
        command: Sequence[str],
        inputs_sha256: str,
        receipt_path: Path,
    ) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "argv": list(command),
            "argv_sha256": canonical_sha256(list(command)),
            "contract": COMMAND_INTENT_CONTRACT,
            "inputs_sha256": inputs_sha256,
            "receipt_path": str(receipt_path),
            "round_index": self.round_index,
            "run_request_sha256": self._request_value()["request_sha256"],
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "trial_id": self.context.trial_id,
            "trial_manifest_path": str(self.context.path),
            "trial_manifest_sha256": self.context.manifest_sha256,
            "work_id": self.work_id,
            "worker_spec_sha256": self.spec.spec_sha256,
        }
        value["intent_sha256"] = canonical_sha256(value)
        return value

    def _load_intent(
        self,
        path: Path,
        expected: Mapping[str, Any],
    ) -> Dict[str, Any]:
        try:
            raw = load_canonical_json(path, "adaptive trial command intent")
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise AmbiguousTrialState(
                "invalid_command_intent",
                f"cannot load prior command intent: {exc}",
            ) from exc
        if set(raw) != _COMMAND_INTENT_FIELDS:
            raise AmbiguousTrialState(
                "invalid_command_intent",
                "prior command intent fields differ from the schema",
            )
        if (
            raw.get("contract") != COMMAND_INTENT_CONTRACT
            or type(raw.get("schema_version")) is not int
            or raw["schema_version"] != SCHEMA_VERSION
            or type(raw.get("round_index")) is not int
            or raw["round_index"] < 0
        ):
            raise AmbiguousTrialState(
                "invalid_command_intent",
                "prior command intent contract is invalid",
            )
        body = dict(raw)
        supplied = body.pop("intent_sha256", None)
        if (
            not _is_sha256(supplied)
            or canonical_sha256(body) != supplied
            or raw != dict(expected)
        ):
            raise AmbiguousTrialState(
                "command_intent_conflict",
                "prior command intent contradicts this exact stage launch",
            )
        return raw

    def _receipt_binding(
        self,
        value: Any,
        role: str,
        expected_path: Path,
        *,
        resumable: bool = False,
    ) -> FileBinding:
        fields = {"path", "sha256", "resumable"} if resumable else {"path", "sha256"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CommandReceiptError(
                "invalid_command_output",
                f"{role} binding fields differ from the schema",
            )
        if resumable and value["resumable"] is not True:
            raise CommandReceiptError(
                "invalid_command_output",
                f"{role} must be marked resumable",
            )
        expected_hash = _require_sha256(
            value["sha256"],
            f"{role} hash",
            error_type=CommandReceiptError,
        )
        path = _required_file(
            value["path"],
            role,
            expected_sha256=expected_hash,
            error_type=CommandReceiptError,
        )
        if path != expected_path or not _strictly_within(
            path, self.context.isolation_root
        ):
            raise CommandReceiptError(
                "command_output_escaped",
                f"{role} escaped the isolated trial path",
            )
        return FileBinding(path, expected_hash)

    def _validate_curriculum_manifest(
        self,
        binding: FileBinding,
    ) -> Mapping[str, Any]:
        try:
            raw = load_canonical_json(binding.path, "adaptive curriculum manifest")
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise CommandReceiptError(
                "invalid_curriculum",
                f"cannot load curriculum manifest: {exc}",
            ) from exc
        if set(raw) != _CURRICULUM_FIELDS:
            raise CommandReceiptError(
                "invalid_curriculum",
                "curriculum manifest fields differ from the schema",
            )
        body = dict(raw)
        supplied = body.pop("manifest_sha256", None)
        expected_common = {
            "contract": CURRICULUM_MANIFEST_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "worker_spec_sha256": self.spec.spec_sha256,
            "trial_manifest_sha256": self.context.manifest_sha256,
            "trial_id": self.context.trial_id,
            "round_index": self.round_index,
            "admitted_data_manifest": (self.context.admitted_data_manifest.to_dict()),
            "recipe_sha256": self.context.recipe_sha256,
            "curriculum_directory": str(self.paths.curriculum_directory),
            "shuffle_argv": list(self.recipe_arguments.shuffle_argv),
        }
        if (
            not _is_sha256(supplied)
            or canonical_sha256(body) != supplied
            or type(raw.get("schema_version")) is not int
            or type(raw.get("round_index")) is not int
            or any(raw.get(key) != value for key, value in expected_common.items())
        ):
            raise CommandReceiptError(
                "invalid_curriculum",
                "curriculum manifest contradicts frozen trial inputs",
            )
        files = raw["files"]
        if (
            not isinstance(files, list)
            or not files
            or any(
                not isinstance(item, Mapping)
                or set(item) != _CURRICULUM_FILE_FIELDS
                or type(item.get("size")) is not int
                or item["size"] < 0
                or not _is_sha256(item.get("sha256"))
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or Path(item["path"]).is_absolute()
                or ".." in Path(item["path"]).parts
                for item in files
            )
        ):
            raise CommandReceiptError(
                "invalid_curriculum",
                "curriculum file inventory is malformed",
            )
        observed = list(_directory_inventory(self.paths.curriculum_directory))
        if files != observed:
            raise CommandReceiptError(
                "invalid_curriculum",
                "curriculum directory changed after materialization",
            )
        return raw

    def _validate_probe(self, binding: FileBinding) -> Mapping[str, Any]:
        try:
            probe = load_canonical_json(binding.path, "adaptive model probe")
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise CommandReceiptError(
                "invalid_model_probe",
                f"cannot load model probe: {exc}",
            ) from exc
        expected_fields = {
            "config_sha256",
            "contract",
            "finite",
            "gpu_uuid",
            "katago_sha256",
            "model_sha256",
            "schema_version",
        }
        if (
            set(probe) != expected_fields
            or type(probe.get("schema_version")) is not int
            or probe["schema_version"] != SCHEMA_VERSION
            or probe.get("contract") != "risk-score-model-probe-v1"
            or probe.get("finite") is not True
            or probe.get("model_sha256")
            != _stable_file_sha256(self.paths.candidate_model)
            or probe.get("katago_sha256") != self.spec.katago_binary.sha256
            or not _is_sha256(probe.get("config_sha256"))
            or not isinstance(probe.get("gpu_uuid"), str)
            or not probe["gpu_uuid"]
        ):
            raise CommandReceiptError(
                "invalid_model_probe",
                "model probe does not prove a finite load of the bound candidate",
            )
        return probe

    def _validate_evidence_binding(
        self,
        binding: FileBinding,
        source: str,
    ) -> Mapping[str, Any]:
        try:
            raw = load_canonical_json(binding.path, f"{source} adaptive evidence")
            if (
                type(raw.get("schema_version")) is not int
                or type(raw.get("round_index")) is not int
                or type(raw.get("sample_count")) is not int
            ):
                raise ValueError("adaptive evidence integer fields are malformed")
            evidence = validate_evidence(
                raw,
                expected_trial_id=self.context.trial_id,
                expected_round_index=self.round_index,
                policy=load_policy(self.spec.autonomy_policy.path),
            )
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise CommandReceiptError(
                "invalid_tuning_evidence",
                f"{source} evidence is invalid: {exc}",
            ) from exc
        if evidence != raw or evidence["source"] != source:
            raise CommandReceiptError(
                "invalid_tuning_evidence",
                f"{source} command published a different evidence source",
            )
        return evidence

    def _validate_stage_outputs(
        self,
        stage: str,
        outputs: Any,
    ) -> Mapping[str, Any]:
        if not isinstance(outputs, Mapping):
            raise CommandReceiptError(
                "invalid_command_output",
                f"{stage} command outputs must be an object",
            )
        if stage == "curriculum":
            if set(outputs) != {"curriculum_manifest"}:
                raise CommandReceiptError(
                    "invalid_command_output",
                    "curriculum command output fields differ from the schema",
                )
            binding = self._receipt_binding(
                outputs["curriculum_manifest"],
                "curriculum manifest",
                self.paths.curriculum_manifest,
            )
            self._validate_curriculum_manifest(binding)
        elif stage == "trainer":
            if set(outputs) != {"checkpoint"}:
                raise CommandReceiptError(
                    "invalid_command_output",
                    "trainer command output fields differ from the schema",
                )
            self._receipt_binding(
                outputs["checkpoint"],
                "trial checkpoint",
                self.paths.checkpoint,
                resumable=True,
            )
        elif stage == "export":
            if set(outputs) != {"candidate_checkpoint", "candidate_model"}:
                raise CommandReceiptError(
                    "invalid_command_output",
                    "export command output fields differ from the schema",
                )
            self._receipt_binding(
                outputs["candidate_model"],
                "candidate model",
                self.paths.candidate_model,
            )
            self._receipt_binding(
                outputs["candidate_checkpoint"],
                "candidate checkpoint",
                self.paths.candidate_checkpoint,
                resumable=True,
            )
        elif stage == "model_probe":
            if set(outputs) != {"probe"}:
                raise CommandReceiptError(
                    "invalid_command_output",
                    "model-probe output fields differ from the schema",
                )
            binding = self._receipt_binding(
                outputs["probe"],
                "model probe",
                self.paths.model_probe,
            )
            self._validate_probe(binding)
        elif stage in _EVIDENCE_STAGES:
            if set(outputs) != {"evidence"}:
                raise CommandReceiptError(
                    "invalid_command_output",
                    f"{stage} output fields differ from the schema",
                )
            source = "fixed_validation" if stage == "fixed_validation" else "discovery"
            expected_path = (
                self.paths.fixed_validation_evidence
                if stage == "fixed_validation"
                else self.paths.discovery_evidence
            )
            binding = self._receipt_binding(
                outputs["evidence"],
                f"{source} evidence",
                expected_path,
            )
            self._validate_evidence_binding(binding, source)
        else:  # pragma: no cover
            raise CommandReceiptError(
                "invalid_stage",
                f"unsupported command output stage: {stage}",
            )
        return outputs

    def _validate_receipt(
        self,
        stage: str,
        receipt_path: Path,
        command: Sequence[str],
        inputs_sha256: str,
        *,
        expected_returncode: Optional[int] = None,
    ) -> Mapping[str, Any]:
        raw = _load_command_receipt(receipt_path)
        expected = {
            "argv_sha256": canonical_sha256(list(command)),
            "inputs_sha256": inputs_sha256,
            "round_index": self.round_index,
            "stage": stage,
            "trial_id": self.context.trial_id,
            "trial_manifest_path": str(self.context.path),
            "trial_manifest_sha256": self.context.manifest_sha256,
            "work_id": self.work_id,
            "worker_spec_sha256": self.spec.spec_sha256,
        }
        contradictions = [
            key for key, value in expected.items() if raw.get(key) != value
        ]
        if contradictions:
            raise CommandReceiptError(
                "command_receipt_binding_mismatch",
                f"{stage} command receipt contradicts its launch",
                details={"fields": contradictions},
            )
        returncode = raw.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise CommandReceiptError(
                "invalid_command_receipt",
                f"{stage} command receipt return code is malformed",
            )
        if expected_returncode is not None and returncode != expected_returncode:
            raise CommandReceiptError(
                "command_receipt_binding_mismatch",
                f"{stage} receipt return code differs from the child",
            )
        status = raw.get("status")
        if status not in {"completed", "drained", "failed"}:
            raise CommandReceiptError(
                "invalid_command_receipt",
                f"{stage} command receipt status is unsupported",
            )
        if status == "completed":
            if returncode != 0:
                raise CommandReceiptError(
                    "invalid_command_receipt",
                    f"{stage} completion receipt has a nonzero return code",
                )
            self._validate_stage_outputs(stage, raw["outputs"])
        elif raw["outputs"] != {}:
            raise CommandReceiptError(
                "invalid_command_receipt",
                f"{stage} non-completion receipt must not publish outputs",
            )
        return raw

    def _assert_prior_outputs(self) -> None:
        for stage in _STAGES:
            receipt = self._validated_receipts.get(stage)
            if receipt is None:
                continue
            if receipt.get("status") != "completed":
                raise CommandReceiptError(
                    "invalid_command_receipt",
                    f"remembered {stage} receipt is not complete",
                )
            self._validate_stage_outputs(stage, receipt.get("outputs"))

    def _environment(
        self,
        stage: str,
        command: Sequence[str],
        inputs_sha256: str,
        receipt_path: Path,
    ) -> Dict[str, str]:
        return {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": self.spec.gpu_id,
            "RISK_SCORE_ADAPTIVE_ARGV_SHA256": canonical_sha256(list(command)),
            "RISK_SCORE_ADAPTIVE_INPUTS_SHA256": inputs_sha256,
            "RISK_SCORE_ADAPTIVE_RECEIPT_PATH": str(receipt_path),
            "RISK_SCORE_ADAPTIVE_ROUND_INDEX": str(self.round_index),
            "RISK_SCORE_ADAPTIVE_RUN_REQUEST_SHA256": self._request_value()[
                "request_sha256"
            ],
            "RISK_SCORE_ADAPTIVE_STAGE": stage,
            "RISK_SCORE_ADAPTIVE_STARTED_AT_UNIX": _number_text(self._run_started_at),
            "RISK_SCORE_ADAPTIVE_TRIAL_ID": self.context.trial_id,
            "RISK_SCORE_ADAPTIVE_TRIAL_MANIFEST_PATH": str(self.context.path),
            "RISK_SCORE_ADAPTIVE_TRIAL_MANIFEST_SHA256": (self.context.manifest_sha256),
            "RISK_SCORE_ADAPTIVE_WORK_ID": self.work_id,
            "RISK_SCORE_ADAPTIVE_WORKER_SPEC_SHA256": self.spec.spec_sha256,
        }

    def _wait_for_process(
        self,
        process: RunningCommand,
        stage: str,
    ) -> Tuple[int, bool]:
        interrupted = False
        while True:
            returncode = process.poll()
            if returncode is not None:
                completed_at = self._now()
                self._gpu_ended_at = completed_at
                if process.process_group_alive():
                    raise AmbiguousTrialState(
                        "child_process_group_not_drained",
                        f"{stage} parent exited while descendants remain alive",
                    )
                return int(returncode), interrupted
            now = self._now()
            if self._interrupted or now >= self.deadline_unix:
                interrupted = True
                if self._signal_error is not None:
                    raise AmbiguousTrialState(
                        "graceful_interrupt_failed",
                        f"could not send SIGINT to {stage} child: {self._signal_error}",
                    ) from self._signal_error
                if not self._sigint_sent_to_active:
                    try:
                        process.send_signal(signal.SIGINT)
                        self._sigint_sent_to_active = True
                    except BaseException as exc:
                        raise AmbiguousTrialState(
                            "graceful_interrupt_failed",
                            f"could not send SIGINT to {stage} child: {exc}",
                        ) from exc
                drain_deadline = now + self.spec.drain_timeout_seconds
                while True:
                    returncode = process.poll()
                    if returncode is not None:
                        completed_at = self._now()
                        self._gpu_ended_at = completed_at
                        if process.process_group_alive():
                            raise AmbiguousTrialState(
                                "child_process_group_not_drained",
                                f"{stage} descendants survived graceful drain",
                            )
                        return int(returncode), True
                    current = self._now()
                    if current >= drain_deadline:
                        raise AmbiguousTrialState(
                            "graceful_drain_timeout",
                            f"{stage} child did not drain after SIGINT; "
                            "SIGKILL is forbidden",
                        )
                    self.sleeper(
                        min(
                            self.spec.poll_interval_seconds,
                            max(0.0, drain_deadline - current),
                        )
                    )
            self.sleeper(
                min(
                    self.spec.poll_interval_seconds,
                    max(0.0, self.deadline_unix - now),
                )
            )

    def _execute_stage(self, stage: str) -> Mapping[str, Any]:
        self._assert_frozen(validate_fixed_inventory=stage in _EVIDENCE_STAGES)
        self._assert_prior_outputs()
        inputs = self._stage_inputs(stage)
        inputs_sha256 = canonical_sha256(inputs)
        receipt_path = self.paths.receipt(stage)
        command = self._render_command(
            stage,
            receipt_path=receipt_path,
            inputs_sha256=inputs_sha256,
        )
        intent_path = self.paths.intent(stage)
        intent = self._intent_value(
            stage,
            command,
            inputs_sha256,
            receipt_path,
        )
        receipt_exists = os.path.lexists(os.fspath(receipt_path))
        intent_exists = os.path.lexists(os.fspath(intent_path))
        if receipt_exists:
            if not intent_exists:
                raise AmbiguousTrialState(
                    "receipt_without_intent",
                    f"{stage} receipt exists without its durable launch intent",
                )
            self._load_intent(intent_path, intent)
            try:
                receipt = self._validate_receipt(
                    stage,
                    receipt_path,
                    command,
                    inputs_sha256,
                )
            except CommandReceiptError as exc:
                raise AmbiguousTrialState(
                    "replayed_receipt_invalid",
                    f"prior {stage} receipt is invalid; duplicate launch refused",
                    details={"cause": exc.code},
                ) from exc
            if receipt["status"] != "completed":
                raise AmbiguousTrialState(
                    "replayed_command_not_complete",
                    f"prior {stage} command did not complete successfully",
                )
            self._validated_receipts[stage] = receipt
            return receipt
        if intent_exists:
            self._load_intent(intent_path, intent)
            raise AmbiguousTrialState(
                "incomplete_command_lifetime",
                f"{stage} launch intent has no completion receipt; duplicate "
                "launch refused",
            )
        now = self._now()
        if self._interrupted:
            raise CommandFailure(
                "worker_interrupted",
                f"worker was interrupted before {stage} could start",
                state_unambiguous=True,
            )
        if now >= self.deadline_unix:
            raise CommandFailure(
                "gpu_deadline_exhausted",
                f"GPU deadline expired before {stage} could start",
                state_unambiguous=True,
            )
        if self._gpu_started_at is None:
            self._gpu_started_at = now
        atomic_create_json(intent_path, intent)
        if stage == "trainer":
            self._trainer_started = True
        try:
            process = self.runner.spawn(
                command,
                cwd=self.paths.round_root,
                environment=self._environment(
                    stage,
                    command,
                    inputs_sha256,
                    receipt_path,
                ),
                log_path=self.paths.log(stage),
            )
        except CommandSpawnError:
            raise
        except BaseException as exc:
            raise AmbiguousTrialState(
                "command_spawn_ambiguous",
                f"{stage} runner failed without proving whether a child exists: {exc}",
            ) from exc
        self._active_process = process
        self._active_stage = stage
        self._sigint_sent_to_active = False
        try:
            returncode, interrupted = self._wait_for_process(process, stage)
        finally:
            self._active_process = None
            self._active_stage = None
            self._sigint_sent_to_active = False
        if self._signal_error is not None:
            raise AmbiguousTrialState(
                "graceful_interrupt_failed",
                f"could not forward SIGINT: {self._signal_error}",
            ) from self._signal_error
        if interrupted:
            if not receipt_path.is_file() or receipt_path.is_symlink():
                raise CommandReceiptError(
                    "missing_command_receipt",
                    f"{stage} child drained without a canonical receipt",
                    state_unambiguous=True,
                )
            self._validate_receipt(
                stage,
                receipt_path,
                command,
                inputs_sha256,
                expected_returncode=returncode,
            )
            raise CommandFailure(
                "gpu_deadline_exhausted",
                f"{stage} reached the GPU deadline and drained via SIGINT",
                state_unambiguous=True,
            )
        if returncode != 0:
            if not receipt_path.is_file() or receipt_path.is_symlink():
                raise CommandReceiptError(
                    "missing_command_receipt",
                    f"{stage} child failed without a canonical receipt",
                    state_unambiguous=True,
                )
            self._validate_receipt(
                stage,
                receipt_path,
                command,
                inputs_sha256,
                expected_returncode=returncode,
            )
            raise CommandFailure(
                "child_command_failed",
                f"{stage} child exited with status {returncode}",
                details={"returncode": returncode, "stage": stage},
                state_unambiguous=True,
            )
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise CommandReceiptError(
                "missing_command_receipt",
                f"{stage} child succeeded without a canonical receipt",
                state_unambiguous=True,
            )
        receipt = self._validate_receipt(
            stage,
            receipt_path,
            command,
            inputs_sha256,
            expected_returncode=returncode,
        )
        if receipt["status"] != "completed":
            raise CommandFailure(
                "child_command_failed",
                f"{stage} child did not publish completed status",
                details={"stage": stage, "status": receipt["status"]},
                state_unambiguous=True,
            )
        self._validated_receipts[stage] = receipt
        return receipt

    def _prove_checkpoint(
        self,
        *,
        require_file: bool,
    ) -> Mapping[str, Any]:
        proof_deadline = self._now() + self.spec.checkpoint_timeout_seconds
        previous: Any = object()
        stable_since: Optional[float] = None
        while True:
            now = self._now()
            current = _stable_file_identity(self.paths.checkpoint)
            marker: Any = current if current is not None else "absent"
            if marker != previous:
                previous = marker
                stable_since = now
            if stable_since is not None and (
                now - stable_since >= self.spec.checkpoint_stable_seconds
            ):
                if current is not None:
                    proof = {
                        "checkpoint": current,
                        "checkpoint_path": str(self.paths.checkpoint),
                        "stable_seconds": self.spec.checkpoint_stable_seconds,
                        "state": "stable-file",
                    }
                    self._trainer_checkpoint_proof = proof
                    return proof
                if not require_file and not os.path.lexists(
                    os.fspath(self.paths.checkpoint)
                ):
                    proof = {
                        "checkpoint": None,
                        "checkpoint_path": str(self.paths.checkpoint),
                        "stable_seconds": self.spec.checkpoint_stable_seconds,
                        "state": "stably-absent",
                    }
                    self._trainer_checkpoint_proof = proof
                    return proof
            if now >= proof_deadline:
                raise AmbiguousTrialState(
                    "checkpoint_stability_unproven",
                    "trainer exited without a stable checkpoint boundary",
                )
            self.sleeper(
                min(
                    self.spec.poll_interval_seconds,
                    max(0.0, proof_deadline - now),
                )
            )

    def _evidence(self) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        fixed = self._validate_evidence_binding(
            FileBinding(
                self.paths.fixed_validation_evidence,
                _stable_file_sha256(self.paths.fixed_validation_evidence),
            ),
            "fixed_validation",
        )
        discovery = self._validate_evidence_binding(
            FileBinding(
                self.paths.discovery_evidence,
                _stable_file_sha256(self.paths.discovery_evidence),
            ),
            "discovery",
        )
        return fixed, discovery

    def _gpu_interval(self) -> Tuple[float, float]:
        started = (
            self._run_started_at
            if self._gpu_started_at is None
            else self._gpu_started_at
        )
        ended = started if self._gpu_ended_at is None else self._gpu_ended_at
        if started is None:
            raise AdaptiveTrialWorkerError(
                "invalid_gpu_interval",
                "worker has no measured start time",
            )
        if (
            ended < started
            or ended - started > self.reservation_gpu_seconds + 1e-6
            or ended > self.deadline_unix + 1e-6
        ):
            raise AmbiguousTrialState(
                "gpu_budget_exceeded",
                "measured GPU lifetime exceeded its reservation or deadline",
            )
        return started, ended

    def _publish_completed_result(self) -> Mapping[str, Any]:
        self._assert_frozen()
        self._assert_prior_outputs()
        if self._trainer_checkpoint_proof is None:
            raise AmbiguousTrialState(
                "checkpoint_stability_unproven",
                "completed trial has no checkpoint stability proof",
            )
        fixed, discovery = self._evidence()
        started, ended = self._gpu_interval()
        self._assert_frozen(validate_fixed_inventory=True)
        self._assert_prior_outputs()
        result = publish_trial_result(
            self.paths.result,
            trial_manifest_path=self.context.path,
            work_id=self.work_id,
            round_index=self.round_index,
            gpu_id=self.spec.gpu_id,
            started_at_unix=started,
            ended_at_unix=ended,
            status="completed",
            evidence=(fixed, discovery),
            candidate_model_path=self.paths.candidate_model,
            candidate_checkpoint_path=self.paths.candidate_checkpoint,
        )
        return result

    def _can_publish_failure(self) -> bool:
        if self._active_process is not None:
            return False
        if self._trainer_started and self._trainer_checkpoint_proof is None:
            try:
                self._prove_checkpoint(require_file=False)
            except AmbiguousTrialState:
                return False
        try:
            self._gpu_interval()
        except AmbiguousTrialState:
            return False
        return True

    def _publish_failed_result(
        self,
        error: AdaptiveTrialWorkerError,
    ) -> Mapping[str, Any]:
        self._assert_frozen()
        if not self._can_publish_failure():
            raise AmbiguousTrialState(
                "failure_state_ambiguous",
                "refusing to publish a failed result without a closed child "
                "lifetime and stable checkpoint state",
                details={"cause": error.code},
            ) from error
        started, ended = self._gpu_interval()
        self._assert_frozen()
        reason = error.code
        return publish_trial_result(
            self.paths.result,
            trial_manifest_path=self.context.path,
            work_id=self.work_id,
            round_index=self.round_index,
            gpu_id=self.spec.gpu_id,
            started_at_unix=started,
            ended_at_unix=ended,
            status="failed",
            failure_reason=reason,
        )

    def _validate_existing_result(self) -> Mapping[str, Any]:
        request = self._load_request()
        try:
            result = load_trial_result(
                self.paths.result,
                expected_trial_id=self.context.trial_id,
                expected_epoch_id=self.context.epoch_id,
                expected_round_index=self.round_index,
                expected_work_id=self.work_id,
                expected_gpu_id=self.spec.gpu_id,
                expected_manifest_path=self.context.path,
                expected_manifest_sha256=self.context.manifest_sha256,
            )
        except (AdaptiveTrainingError, OSError, ValueError) as exc:
            raise AmbiguousTrialState(
                "result_replay_conflict",
                f"existing adaptive trial result is invalid: {exc}",
            ) from exc
        usage = result["gpu_usage"]
        if (
            usage["started_at_unix"] != request["started_at_unix"]
            or usage["ended_at_unix"] > request["deadline_unix"] + 1e-6
            or usage["ended_at_unix"] - usage["started_at_unix"]
            > request["reservation_gpu_seconds"] + 1e-6
        ):
            raise AmbiguousTrialState(
                "result_replay_conflict",
                "existing result GPU interval contradicts its run request",
            )
        return result

    @contextlib.contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        if not self.install_signal_handlers:
            yield
            return
        try:
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
        except (OSError, ValueError) as exc:
            raise AmbiguousTrialState(
                "signal_handler_unavailable",
                "adaptive trial worker must install a SIGINT handler",
            ) from exc
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)

    def _handle_sigint(
        self,
        _signum: int,
        _frame: Optional[FrameType],
    ) -> None:
        first = not self._interrupted
        self._interrupted = True
        if first and self._active_process is not None:
            try:
                self._active_process.send_signal(signal.SIGINT)
                self._sigint_sent_to_active = True
            except BaseException as exc:
                self._signal_error = exc

    def run(self) -> Mapping[str, Any]:
        """Execute or read-only replay one exact adaptive trial result."""

        self._assert_frozen(validate_fixed_inventory=True)
        if os.path.lexists(os.fspath(self.paths.result)):
            return self._validate_existing_result()
        observed_now = self._now()
        if os.path.lexists(os.fspath(self.paths.request)):
            self._load_request()
        else:
            self._run_started_at = self._invocation_started_at
            self._gpu_started_at = self._invocation_started_at
            self._gpu_ended_at = observed_now
        if self._run_started_at is None:
            raise AmbiguousTrialState(
                "invalid_run_request",
                "trial execution has no durable start time",
            )
        if (
            not os.path.lexists(os.fspath(self.paths.request))
            and self.deadline_unix <= self._run_started_at
        ):
            raise TrialBindingError(
                "expired_gpu_deadline",
                "GPU-time deadline is not in the future",
            )
        if (
            self.deadline_unix - self._run_started_at
            > self.reservation_gpu_seconds + 1e-6
        ):
            raise TrialBindingError(
                "invalid_gpu_deadline",
                "GPU deadline exceeds the supplied round reservation",
            )
        _ensure_directory(self.paths.round_root)
        for directory in (
            self.paths.intents,
            self.paths.receipts,
            self.paths.logs,
        ):
            _ensure_directory(directory)
        with _ExecutionLock(self.paths.lock), self._signal_handlers():
            if os.path.lexists(os.fspath(self.paths.result)):
                return self._validate_existing_result()
            request = self._request_value()
            if os.path.lexists(os.fspath(self.paths.request)):
                self._load_request()
            else:
                atomic_create_json(self.paths.request, request)
            try:
                self._execute_stage("curriculum")
                self._execute_stage("trainer")
                self._prove_checkpoint(require_file=True)
                self._execute_stage("export")
                self._execute_stage("model_probe")
                self._execute_stage("fixed_validation")
                self._execute_stage("discovery")
                return self._publish_completed_result()
            except (FrozenInputChanged, AmbiguousTrialState):
                raise
            except AdaptiveTrialWorkerError as exc:
                if not exc.state_unambiguous and not self._can_publish_failure():
                    raise AmbiguousTrialState(
                        "failure_state_ambiguous",
                        "worker failure did not prove a closed trainer state",
                        details={"cause": exc.code},
                    ) from exc
                return self._publish_failed_result(exc)
            except (AdaptiveTrainingError, OSError, TypeError, ValueError) as exc:
                wrapped = AdaptiveTrialWorkerError(
                    "trial_execution_failed",
                    str(exc),
                )
                if not self._can_publish_failure():
                    raise AmbiguousTrialState(
                        "failure_state_ambiguous",
                        "unexpected worker failure left trainer state ambiguous",
                    ) from exc
                return self._publish_failed_result(wrapped)


BoundedAdaptiveTrialWorker = AdaptiveTrialWorker
AdaptiveTrainingTrialWorker = AdaptiveTrialWorker
AdaptiveTrialWorkerSpec = WorkerSpec
AdaptiveTrialWorkerSpecError = WorkerSpecError


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        "--worker-spec",
        dest="spec",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-spec-sha256",
        "--worker-spec-sha256",
        dest="expected_spec_sha256",
        required=True,
    )
    parser.add_argument(
        "--trial-manifest",
        "--trial-manifest-path",
        dest="trial_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-trial-manifest-sha256",
        "--trial-manifest-sha256",
        dest="trial_manifest_sha256",
        required=True,
    )
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--round-index", required=True, type=int)
    parser.add_argument(
        "--reservation-gpu-seconds",
        "--gpu-time-reservation-seconds",
        "--gpu-reservation-seconds",
        dest="reservation_gpu_seconds",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--deadline-unix",
        "--gpu-time-deadline-unix",
        "--gpu-deadline-unix",
        dest="deadline_unix",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--result",
        "--result-path",
        dest="result",
        required=True,
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        worker = AdaptiveTrialWorker(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
            trial_manifest_path=args.trial_manifest,
            expected_trial_manifest_sha256=args.trial_manifest_sha256,
            work_id=args.work_id,
            round_index=args.round_index,
            reservation_gpu_seconds=args.reservation_gpu_seconds,
            deadline_unix=args.deadline_unix,
            result_path=args.result,
        )
        result = worker.run()
    except KeyboardInterrupt:
        error = AmbiguousTrialState(
            "worker_interrupted",
            "adaptive trial worker was interrupted outside a drainable child lifetime",
        )
        sys.stderr.buffer.write(canonical_json_bytes(error.to_dict()) + b"\n")
        return 2
    except (AdaptiveTrialWorkerError, AdaptiveTrainingError, OSError) as exc:
        error = (
            exc
            if isinstance(exc, AdaptiveTrialWorkerError)
            else AdaptiveTrialWorkerError(
                "adaptive_training_error",
                str(exc),
                details={"cause": getattr(exc, "code", type(exc).__name__)},
            )
        )
        sys.stderr.buffer.write(canonical_json_bytes(error.to_dict()) + b"\n")
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


load_adaptive_trial_worker_spec = load_worker_spec
publish_adaptive_trial_worker_spec = publish_worker_spec


__all__ = [
    "COMMAND_INTENT_CONTRACT",
    "COMMAND_RECEIPT_CONTRACT",
    "CURRICULUM_MANIFEST_CONTRACT",
    "ERROR_CONTRACT",
    "GPU_ID",
    "RECEIPT_CONTRACT",
    "RUN_REQUEST_CONTRACT",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "WORKER_SPEC_CONTRACT",
    "AdaptiveTrainingTrialWorker",
    "AdaptiveTrialWorker",
    "AdaptiveTrialWorkerError",
    "AdaptiveTrialWorkerSpec",
    "AdaptiveTrialWorkerSpecError",
    "AmbiguousTrialState",
    "BoundedAdaptiveTrialWorker",
    "CommandFailure",
    "CommandReceiptError",
    "CommandRunner",
    "CommandSpawnError",
    "DeploymentBinding",
    "FileBinding",
    "FrozenInputChanged",
    "RecipeArguments",
    "RunningCommand",
    "SubprocessCommandRunner",
    "TrialBindingError",
    "TrialContext",
    "TrialPaths",
    "WorkerSpec",
    "WorkerSpecError",
    "load_adaptive_trial_worker_spec",
    "load_trial_manifest",
    "load_worker_spec",
    "main",
    "parse_args",
    "publish_adaptive_trial_worker_spec",
    "publish_command_receipt",
    "publish_curriculum_manifest",
    "publish_worker_spec",
    "recipe_to_argv",
    "translate_recipe",
    "trial_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
