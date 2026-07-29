#!/usr/bin/env python3
"""Exclusive trainer/evaluator GPU handoff with crash reconciliation.

The module deliberately uses only the Python standard library. Process
execution, time, sleeping, and GPU observations are injectable so the safety
state machine can be tested without signals, CUDA, or real subprocesses.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - the production target is Unix.
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 2


class GpuLeaseError(RuntimeError):
    """An operational error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class ProcessIdentity:
    """Identity fields that distinguish PID reuse.

    A PID by itself is never considered a verifiable process identity.
    Linux exposes all fields through procfs. Fields may be absent on other
    systems, in which case matching remains conservative.
    """

    pid: int
    start_time_ticks: Optional[int]
    process_group_id: Optional[int]
    boot_id: Optional[str]
    command_sha256: Optional[str]
    cgroup: Optional[str]

    @classmethod
    def capture(cls, pid: int) -> "ProcessIdentity":
        if pid <= 0:
            raise GpuLeaseError(
                "invalid_pid", "Process PID must be positive", details={"pid": pid}
            )

        proc_dir = Path("/proc") / str(pid)
        boot_id = _read_optional_text(Path("/proc/sys/kernel/random/boot_id"))
        start_time_ticks: Optional[int] = None
        stat_text = _read_optional_text(proc_dir / "stat")
        if stat_text is not None:
            close_paren = stat_text.rfind(")")
            if close_paren >= 0:
                fields_after_comm = stat_text[close_paren + 1 :].split()
                # The first item is field 3 (state); field 22 is index 19.
                if len(fields_after_comm) > 19:
                    try:
                        start_time_ticks = int(fields_after_comm[19])
                    except ValueError:
                        start_time_ticks = None

        try:
            process_group_id: Optional[int] = os.getpgid(pid)
        except (OSError, AttributeError):
            process_group_id = None

        command_sha256: Optional[str] = None
        try:
            command_bytes = (proc_dir / "cmdline").read_bytes()
            if command_bytes:
                command_sha256 = hashlib.sha256(command_bytes).hexdigest()
        except OSError:
            pass

        cgroup = _read_optional_text(proc_dir / "cgroup")
        identity = cls(
            pid=pid,
            start_time_ticks=start_time_ticks,
            process_group_id=process_group_id,
            boot_id=boot_id,
            command_sha256=command_sha256,
            cgroup=cgroup,
        )
        if not identity.is_verifiable:
            raise GpuLeaseError(
                "unverifiable_process_identity",
                "Could not obtain enough process metadata to distinguish PID reuse",
                details={"pid": pid},
            )
        return identity

    @property
    def is_verifiable(self) -> bool:
        if self.start_time_ticks is not None:
            return True
        return self.command_sha256 is not None and self.cgroup is not None

    def same_process_as(self, other: "ProcessIdentity") -> bool:
        if self.pid != other.pid:
            return False

        matched_start_time = False
        matched_command = False
        matched_cgroup = False
        for left, right, field_name in (
            (self.boot_id, other.boot_id, "boot"),
            (self.start_time_ticks, other.start_time_ticks, "start"),
            (self.process_group_id, other.process_group_id, "process_group"),
            (self.command_sha256, other.command_sha256, "command"),
            (self.cgroup, other.cgroup, "cgroup"),
        ):
            if left is None and right is None:
                continue
            if left is None or right is None or left != right:
                return False
            matched_start_time = matched_start_time or field_name == "start"
            matched_command = matched_command or field_name == "command"
            matched_cgroup = matched_cgroup or field_name == "cgroup"
        return matched_start_time or (matched_command and matched_cgroup)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "startTimeTicks": self.start_time_ticks,
            "processGroupId": self.process_group_id,
            "bootId": self.boot_id,
            "commandSha256": self.command_sha256,
            "cgroup": self.cgroup,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessIdentity":
        return cls(
            pid=int(value["pid"]),
            start_time_ticks=_optional_int(value.get("startTimeTicks")),
            process_group_id=_optional_int(value.get("processGroupId")),
            boot_id=_optional_str(value.get("bootId")),
            command_sha256=_optional_str(value.get("commandSha256")),
            cgroup=_optional_str(value.get("cgroup")),
        )


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str = ""
    used_memory_mib: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "processName": self.process_name,
            "usedMemoryMiB": self.used_memory_mib,
        }


@dataclass(frozen=True)
class GpuObservation:
    gpu_uuid: str
    processes: Tuple[GpuProcess, ...] = ()


@dataclass(frozen=True)
class CheckpointIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtimeNs": self.mtime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointIdentity":
        sha256 = str(value["sha256"])
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise GpuLeaseError(
                "invalid_lease_state", "Checkpoint SHA-256 is malformed"
            )
        identity = cls(
            device=int(value["device"]),
            inode=int(value["inode"]),
            size=int(value["size"]),
            mtime_ns=int(value["mtimeNs"]),
            sha256=sha256,
        )
        if identity.device < 0 or identity.inode < 0 or identity.size < 0:
            raise GpuLeaseError(
                "invalid_lease_state", "Checkpoint identity fields are invalid"
            )
        return identity

    def content_changed_from(self, before: "CheckpointIdentity") -> bool:
        return self.sha256 != before.sha256


class ProcessRunner(Protocol):
    def run(
        self, argv: Sequence[str], *, timeout: Optional[float] = None
    ) -> CommandResult: ...

    def spawn(
        self, argv: Sequence[str], *, new_process_group: bool = True
    ) -> ProcessIdentity: ...

    def current_identity(self, pid: int) -> Optional[ProcessIdentity]: ...

    def is_running(self, identity: ProcessIdentity) -> bool: ...

    def process_group_alive(self, identity: ProcessIdentity) -> bool: ...

    def signal_process_group(self, identity: ProcessIdentity, sig: int) -> None: ...


class SubprocessRunner:
    """Production process runner. No command is interpreted by a shell."""

    def __init__(self) -> None:
        self._children: Dict[int, subprocess.Popen[Any]] = {}

    def run(
        self, argv: Sequence[str], *, timeout: Optional[float] = None
    ) -> CommandResult:
        _validate_argv(argv, "command")
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def spawn(
        self, argv: Sequence[str], *, new_process_group: bool = True
    ) -> ProcessIdentity:
        _validate_argv(argv, "command")
        process = subprocess.Popen(
            list(argv),
            start_new_session=new_process_group,
            shell=False,
        )
        self._children[process.pid] = process
        try:
            return ProcessIdentity.capture(process.pid)
        except BaseException:
            with contextlib.suppress(OSError):
                process.terminate()
            self._children.pop(process.pid, None)
            raise

    def current_identity(self, pid: int) -> Optional[ProcessIdentity]:
        try:
            return ProcessIdentity.capture(pid)
        except GpuLeaseError:
            return None

    def is_running(self, identity: ProcessIdentity) -> bool:
        child = self._children.get(identity.pid)
        if child is not None and child.poll() is not None:
            self._children.pop(identity.pid, None)
            return False
        current = self.current_identity(identity.pid)
        return current is not None and identity.same_process_as(current)

    def process_group_alive(self, identity: ProcessIdentity) -> bool:
        pgid = identity.process_group_id
        if pgid is None:
            return self.is_running(identity)
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def signal_process_group(self, identity: ProcessIdentity, sig: int) -> None:
        if sig == signal.SIGSTOP:
            raise GpuLeaseError("sigstop_prohibited", "SIGSTOP is prohibited")
        current = self.current_identity(identity.pid)
        if current is None or not identity.same_process_as(current):
            raise GpuLeaseError(
                "process_identity_changed",
                "Refusing to signal a PID whose identity changed",
                details={"expected": identity.to_dict()},
            )
        if identity.process_group_id is None:
            os.kill(identity.pid, sig)
        else:
            os.killpg(identity.process_group_id, sig)


class NvidiaSmiGpuProbe:
    """Observe one UUID and its compute processes using nvidia-smi."""

    def __init__(
        self,
        expected_gpu_uuid: str,
        runner: ProcessRunner,
        inventory_command: Sequence[str] = (
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ),
        process_query_command: Sequence[str] = (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
    ) -> None:
        self._expected_gpu_uuid = expected_gpu_uuid
        self._runner = runner
        _validate_argv(inventory_command, "GPU inventory command")
        _validate_argv(process_query_command, "GPU process query command")
        self._inventory_command = tuple(inventory_command)
        self._process_query_command = tuple(process_query_command)

    def __call__(self) -> GpuObservation:
        inventory = self._runner.run(self._inventory_command)
        if inventory.returncode != 0:
            raise GpuLeaseError(
                "gpu_probe_failed",
                "GPU inventory command failed",
                details={
                    "stderr": inventory.stderr,
                    "returncode": inventory.returncode,
                },
            )
        uuids = {line.strip() for line in inventory.stdout.splitlines() if line.strip()}
        if self._expected_gpu_uuid not in uuids:
            raise GpuLeaseError(
                "gpu_uuid_not_observed",
                "Expected GPU UUID was not reported by nvidia-smi",
                details={
                    "expectedGpuUuid": self._expected_gpu_uuid,
                    "observedGpuUuids": sorted(uuids),
                },
            )

        query = self._runner.run(self._process_query_command)
        if query.returncode != 0:
            raise GpuLeaseError(
                "gpu_probe_failed",
                "GPU process query failed",
                details={"stderr": query.stderr, "returncode": query.returncode},
            )
        processes: List[GpuProcess] = []
        for line in query.stdout.splitlines():
            columns = [column.strip() for column in line.split(",", 3)]
            if len(columns) != 4 or columns[0] != self._expected_gpu_uuid:
                continue
            try:
                processes.append(
                    GpuProcess(
                        pid=int(columns[1]),
                        process_name=columns[2],
                        used_memory_mib=int(columns[3]),
                    )
                )
            except ValueError as exc:
                raise GpuLeaseError(
                    "gpu_probe_malformed",
                    "Could not parse nvidia-smi process output",
                    details={"line": line},
                ) from exc
        return GpuObservation(self._expected_gpu_uuid, tuple(processes))


@dataclass(frozen=True)
class RuntimeConfig:
    mutation_enabled: bool
    run_root: Path
    promotion_root: Path
    lease_state_path: Path
    event_log_path: Path
    expected_gpu_uuid: str
    gpu_index: int
    gpu_inventory_command: Tuple[str, ...]
    gpu_process_query_command: Tuple[str, ...]
    clean_observations: int
    clean_observation_interval_seconds: float
    poll_interval_seconds: float
    trainer_launch_command: Tuple[str, ...]
    trainer_graceful_command: Tuple[str, ...]
    trainer_checkpoint_path: Path
    trainer_drain_timeout_seconds: float
    trainer_checkpoint_timeout_seconds: float
    trainer_checkpoint_stable_seconds: float
    require_checkpoint_change: bool
    trainer_start_timeout_seconds: float
    evaluator_launch_command: Tuple[str, ...]
    evaluator_drain_command: Tuple[str, ...]
    evaluator_process_count: int
    evaluator_drain_timeout_seconds: float
    owner_id: str = field(default_factory=lambda: f"controller-{os.getpid()}")

    @property
    def lock_path(self) -> Path:
        return Path(str(self.lease_state_path) + ".lock")

    def __post_init__(self) -> None:
        _validate_runtime_paths(self)
        if not self.expected_gpu_uuid:
            raise GpuLeaseError(
                "invalid_runtime_config", "expected_gpu_uuid must not be empty"
            )
        if self.clean_observations < 2:
            raise GpuLeaseError(
                "invalid_runtime_config",
                "At least two clean GPU observations are required",
            )
        if self.gpu_index < 0 or self.evaluator_process_count < 1:
            raise GpuLeaseError(
                "invalid_runtime_config",
                "GPU index and evaluator process count are out of range",
            )
        if self.poll_interval_seconds <= 0:
            raise GpuLeaseError(
                "invalid_runtime_config",
                "poll_interval_seconds must be positive",
            )
        for name, duration in (
            (
                "clean_observation_interval_seconds",
                self.clean_observation_interval_seconds,
            ),
            ("trainer_drain_timeout_seconds", self.trainer_drain_timeout_seconds),
            (
                "trainer_checkpoint_timeout_seconds",
                self.trainer_checkpoint_timeout_seconds,
            ),
            (
                "trainer_checkpoint_stable_seconds",
                self.trainer_checkpoint_stable_seconds,
            ),
            ("trainer_start_timeout_seconds", self.trainer_start_timeout_seconds),
            (
                "evaluator_drain_timeout_seconds",
                self.evaluator_drain_timeout_seconds,
            ),
        ):
            if duration < 0:
                raise GpuLeaseError(
                    "invalid_runtime_config",
                    f"{name} must not be negative",
                )
        for name, command, allow_empty in (
            ("trainer_launch_command", self.trainer_launch_command, False),
            ("trainer_graceful_command", self.trainer_graceful_command, True),
            ("evaluator_launch_command", self.evaluator_launch_command, False),
            ("evaluator_drain_command", self.evaluator_drain_command, True),
            ("gpu_inventory_command", self.gpu_inventory_command, False),
            ("gpu_process_query_command", self.gpu_process_query_command, False),
        ):
            if command or not allow_empty:
                _validate_argv(command, name)
            _reject_sigstop(command, name)

    @classmethod
    def from_json_file(cls, path: Path) -> "RuntimeConfig":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GpuLeaseError(
                "invalid_runtime_config",
                f"Could not read runtime config: {exc}",
                details={"path": str(path)},
            ) from exc
        if not isinstance(value, Mapping):
            raise GpuLeaseError(
                "invalid_runtime_config", "Runtime config root must be an object"
            )
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        paths = _mapping(value, "paths")
        gpu = _mapping(value, "gpu")
        trainer = _mapping(value, "trainer")
        evaluator = _mapping(value, "evaluator")

        config = cls(
            mutation_enabled=_bool(value, "mutationEnabled"),
            run_root=Path(_str(paths, "runRoot")),
            promotion_root=Path(_str(paths, "promotionRoot")),
            lease_state_path=Path(_str(paths, "leaseState")),
            event_log_path=Path(_str(paths, "eventLog")),
            expected_gpu_uuid=_str(gpu, "expectedUuid"),
            gpu_index=_int(gpu, "index", minimum=0),
            gpu_inventory_command=_command(gpu, "inventoryCommand"),
            gpu_process_query_command=_command(gpu, "processQueryCommand"),
            clean_observations=_int(gpu, "cleanObservations", minimum=2),
            clean_observation_interval_seconds=_number(
                gpu, "cleanObservationIntervalSeconds", minimum=0.0
            ),
            poll_interval_seconds=_number(value, "pollIntervalSeconds", minimum=0.001),
            trainer_launch_command=_command(trainer, "launchCommand"),
            trainer_graceful_command=_command(
                trainer, "gracefulCommand", allow_empty=True
            ),
            trainer_checkpoint_path=Path(_str(trainer, "checkpointPath")),
            trainer_drain_timeout_seconds=_number(
                trainer, "drainTimeoutSeconds", minimum=0.0
            ),
            trainer_checkpoint_timeout_seconds=_number(
                trainer, "checkpointTimeoutSeconds", minimum=0.0
            ),
            trainer_checkpoint_stable_seconds=_number(
                trainer, "checkpointStableSeconds", minimum=0.0
            ),
            require_checkpoint_change=_bool(trainer, "requireCheckpointChange"),
            trainer_start_timeout_seconds=_number(
                trainer, "startTimeoutSeconds", minimum=0.0
            ),
            evaluator_launch_command=_command(evaluator, "launchCommand"),
            evaluator_drain_command=_command(
                evaluator, "drainCommand", allow_empty=True
            ),
            evaluator_process_count=_int(evaluator, "processCount", minimum=1),
            evaluator_drain_timeout_seconds=_number(
                evaluator, "drainTimeoutSeconds", minimum=0.0
            ),
            owner_id=str(value.get("ownerId") or f"controller-{os.getpid()}"),
        )
        for name, command in (
            ("trainer.gracefulCommand", config.trainer_graceful_command),
            ("trainer.launchCommand", config.trainer_launch_command),
            ("evaluator.launchCommand", config.evaluator_launch_command),
            ("evaluator.drainCommand", config.evaluator_drain_command),
        ):
            _reject_sigstop(command, name)
        return config


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    owner_id: str
    phase: str
    expected_gpu_uuid: str
    trainer: Optional[ProcessIdentity]
    evaluators: Tuple[ProcessIdentity, ...]
    checkpoint_sha256: Optional[str]
    checkpoint_size: Optional[int]
    safety_halt: bool
    safety_reason: Optional[str]
    created_at: float
    updated_at: float
    pre_drain_checkpoint: Optional[CheckpointIdentity] = None
    handoff_checkpoint: Optional[CheckpointIdentity] = None
    lease_clean_observation_times: Tuple[float, ...] = ()
    release_clean_observation_times: Tuple[float, ...] = ()
    restoration_status: str = "not_started"
    restored_trainer: Optional[ProcessIdentity] = None

    @property
    def lease_clean_observation_count(self) -> int:
        return len(self.lease_clean_observation_times)

    @property
    def release_clean_observation_count(self) -> int:
        return len(self.release_clean_observation_times)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "leaseId": self.lease_id,
            "ownerId": self.owner_id,
            "phase": self.phase,
            "expectedGpuUuid": self.expected_gpu_uuid,
            "trainer": None if self.trainer is None else self.trainer.to_dict(),
            "evaluators": [identity.to_dict() for identity in self.evaluators],
            "checkpointSha256": self.checkpoint_sha256,
            "checkpointSize": self.checkpoint_size,
            "preDrainCheckpoint": (
                None
                if self.pre_drain_checkpoint is None
                else self.pre_drain_checkpoint.to_dict()
            ),
            "handoffCheckpoint": (
                None
                if self.handoff_checkpoint is None
                else self.handoff_checkpoint.to_dict()
            ),
            "leaseCleanObservationTimes": list(self.lease_clean_observation_times),
            "leaseCleanObservationCount": (self.lease_clean_observation_count),
            "releaseCleanObservationTimes": list(self.release_clean_observation_times),
            "releaseCleanObservationCount": (self.release_clean_observation_count),
            "restorationStatus": self.restoration_status,
            "restoredTrainer": (
                None
                if self.restored_trainer is None
                else self.restored_trainer.to_dict()
            ),
            "safetyHalt": self.safety_halt,
            "safetyReason": self.safety_reason,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LeaseRecord":
        if value.get("schemaVersion") != SCHEMA_VERSION:
            raise GpuLeaseError(
                "unsupported_lease_schema",
                "Lease state has an unsupported schema version",
                details={"schemaVersion": value.get("schemaVersion")},
            )
        trainer_value = value.get("trainer")
        restored_trainer_value = value.get("restoredTrainer")
        evaluator_values = value.get("evaluators", [])
        if trainer_value is not None and not isinstance(trainer_value, Mapping):
            raise GpuLeaseError("invalid_lease_state", "trainer must be an object")
        if restored_trainer_value is not None and not isinstance(
            restored_trainer_value, Mapping
        ):
            raise GpuLeaseError(
                "invalid_lease_state", "restoredTrainer must be an object"
            )
        if not isinstance(evaluator_values, list):
            raise GpuLeaseError("invalid_lease_state", "evaluators must be an array")
        if any(not isinstance(item, Mapping) for item in evaluator_values):
            raise GpuLeaseError(
                "invalid_lease_state",
                "Every evaluator identity must be an object",
            )
        record = cls(
            lease_id=str(value["leaseId"]),
            owner_id=str(value["ownerId"]),
            phase=str(value["phase"]),
            expected_gpu_uuid=str(value["expectedGpuUuid"]),
            trainer=(
                None
                if trainer_value is None
                else ProcessIdentity.from_dict(trainer_value)
            ),
            evaluators=tuple(
                ProcessIdentity.from_dict(item)
                for item in evaluator_values
                if isinstance(item, Mapping)
            ),
            checkpoint_sha256=_optional_str(value.get("checkpointSha256")),
            checkpoint_size=_optional_int(value.get("checkpointSize")),
            safety_halt=bool(value.get("safetyHalt", False)),
            safety_reason=_optional_str(value.get("safetyReason")),
            created_at=float(value["createdAt"]),
            updated_at=float(value["updatedAt"]),
            pre_drain_checkpoint=_optional_checkpoint_identity(
                value.get("preDrainCheckpoint")
            ),
            handoff_checkpoint=_optional_checkpoint_identity(
                value.get("handoffCheckpoint")
            ),
            lease_clean_observation_times=_float_tuple(
                value.get("leaseCleanObservationTimes", [])
            ),
            release_clean_observation_times=_float_tuple(
                value.get("releaseCleanObservationTimes", [])
            ),
            restoration_status=str(value.get("restorationStatus", "not_started")),
            restored_trainer=(
                None
                if restored_trainer_value is None
                else ProcessIdentity.from_dict(restored_trainer_value)
            ),
        )
        identities = tuple(
            identity
            for identity in (
                record.trainer,
                record.restored_trainer,
            )
            + record.evaluators
            if identity is not None
        )
        if any(not identity.is_verifiable for identity in identities):
            raise GpuLeaseError(
                "invalid_lease_state",
                "Lease state contains an unverifiable process identity",
            )
        valid_restoration_statuses = {
            "not_started",
            "not_needed",
            "pending",
            "started",
            "restored",
            "safety_halt",
        }
        if record.restoration_status not in valid_restoration_statuses:
            raise GpuLeaseError("invalid_lease_state", "Unknown restoration status")
        for count_key, actual_count in (
            (
                "leaseCleanObservationCount",
                record.lease_clean_observation_count,
            ),
            (
                "releaseCleanObservationCount",
                record.release_clean_observation_count,
            ),
        ):
            persisted_count = value.get(count_key, actual_count)
            if (
                isinstance(persisted_count, bool)
                or not isinstance(persisted_count, int)
                or persisted_count != actual_count
            ):
                raise GpuLeaseError(
                    "invalid_lease_state",
                    f"{count_key} does not match observation times",
                )
        return record


@dataclass(frozen=True)
class ReconcileReport:
    previous_phase: Optional[str]
    current_phase: Optional[str]
    actions: Tuple[str, ...]
    mutation_performed: bool
    safety_halt: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previousPhase": self.previous_phase,
            "currentPhase": self.current_phase,
            "actions": list(self.actions),
            "mutationPerformed": self.mutation_performed,
            "safetyHalt": self.safety_halt,
        }


class GpuLeaseManager:
    """Supervise one exclusive GPU lease between trainer and evaluators."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        process_runner: Optional[ProcessRunner] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        gpu_probe: Optional[Callable[[], GpuObservation]] = None,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        lease_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self.config = config
        self.runner = process_runner or SubprocessRunner()
        self.clock = clock
        self.sleep = sleep
        self.gpu_probe = gpu_probe or NvidiaSmiGpuProbe(
            config.expected_gpu_uuid,
            self.runner,
            config.gpu_inventory_command,
            config.gpu_process_query_command,
        )
        self._event_sink = event_sink
        self._lease_id_factory = lease_id_factory
        self._active_record: Optional[LeaseRecord] = None

    @contextlib.contextmanager
    def exclusive_handoff(
        self, trainer: Optional[ProcessIdentity] = None
    ) -> Iterator[LeaseRecord]:
        """Yield an exclusive clean-GPU lease without launching evaluators."""

        self._require_mutation()
        with self._exclusive_lock():
            existing = self._load_record()
            if existing is not None and existing.safety_halt:
                raise GpuLeaseError(
                    "safety_halt_active",
                    "Lease state is halted and requires operator reconciliation",
                    details={"reason": existing.safety_reason},
                )
            if existing is not None:
                _, existing = self._reconcile_locked(existing)
                if existing.safety_halt:
                    raise GpuLeaseError(
                        "safety_halt_active",
                        "Lease reconciliation entered a safety halt",
                        details={"reason": existing.safety_reason},
                    )
            if trainer is None and existing is not None:
                trainer = existing.trainer
            if trainer is None:
                raise GpuLeaseError(
                    "trainer_identity_required",
                    "A verified trainer identity is required for a new handoff",
                )
            self._require_identity_alive(trainer, "trainer")

            pre_drain = self._wait_for_checkpoint_identity(
                before=None, require_change=False
            )
            now = self.clock()
            record = LeaseRecord(
                lease_id=self._lease_id_factory(),
                owner_id=self.config.owner_id,
                phase="draining_trainer",
                expected_gpu_uuid=self.config.expected_gpu_uuid,
                trainer=trainer,
                evaluators=(),
                checkpoint_sha256=None,
                checkpoint_size=None,
                safety_halt=False,
                safety_reason=None,
                created_at=now,
                updated_at=now,
                pre_drain_checkpoint=pre_drain,
                restoration_status="pending",
            )
            self._active_record = record
            self._save_record(record)
            self._emit(
                "trainer_drain_started",
                record,
                preDrainCheckpoint=pre_drain.to_dict(),
            )

            try:
                record = self._complete_trainer_drain(record, issue_command=True)
                clean_times = self._require_clean_gpu()
                record = self._replace_record(
                    record,
                    phase="leased",
                    lease_clean_observation_times=clean_times,
                )
                self._emit("exclusive_handoff_granted", record)
                yield record
            except BaseException as exc:
                self._emit_exception("exclusive_handoff_failed", record, exc)
                raise
            finally:
                active = self._active_record or record
                cleanup_error: Optional[BaseException] = None
                original_trainer_alive = self.runner.is_running(trainer)
                if (
                    original_trainer_alive
                    and active.handoff_checkpoint is None
                    and not active.evaluators
                ):
                    active = self._replace_record(
                        active,
                        phase="trainer_running",
                        trainer=trainer,
                        restoration_status="not_needed",
                    )
                    self._emit("trainer_remained_running", active)
                else:
                    if active.handoff_checkpoint is None:
                        try:
                            active = self._complete_trainer_drain(
                                active, issue_command=False
                            )
                        except BaseException as exc:
                            cleanup_error = exc
                            active = self._set_safety_halt(
                                active,
                                f"checkpoint handoff validation failed: {exc}",
                            )
                    if not active.safety_halt:
                        try:
                            active = self._replace_record(active, phase="releasing")
                            release_times = self._require_clean_gpu()
                            active = self._replace_record(
                                active,
                                phase="release_gpu_verified",
                                release_clean_observation_times=release_times,
                            )
                            self._require_recorded_handoff_checkpoint(active)
                            active = self._restore_trainer(active)
                        except BaseException as exc:
                            cleanup_error = cleanup_error or exc
                            active = self._set_safety_halt(
                                active, f"trainer restoration failed: {exc}"
                            )
                self._active_record = active
                if cleanup_error is not None and sys.exc_info()[0] is None:
                    if isinstance(cleanup_error, GpuLeaseError):
                        raise cleanup_error
                    raise GpuLeaseError(
                        "handoff_cleanup_failed", str(cleanup_error)
                    ) from cleanup_error

    handoff = exclusive_handoff

    @contextlib.contextmanager
    def evaluator_lease(
        self, trainer: Optional[ProcessIdentity] = None
    ) -> Iterator[LeaseRecord]:
        """Build the legacy evaluator launcher on the exclusive handoff."""

        with self.exclusive_handoff(trainer) as handoff_record:
            evaluators: List[ProcessIdentity] = []
            record = handoff_record
            try:
                record = self._replace_record(record, phase="evaluator_starting")
                for worker_index in range(self.config.evaluator_process_count):
                    argv = self._expand(
                        self.config.evaluator_launch_command,
                        record=record,
                        identity=None,
                        worker_index=worker_index,
                    )
                    identity = self.runner.spawn(argv, new_process_group=True)
                    if not identity.is_verifiable:
                        raise GpuLeaseError(
                            "unverifiable_evaluator_identity",
                            "Evaluator launch did not return a verifiable identity",
                            details={"workerIndex": worker_index},
                        )
                    self._require_identity_alive(identity, "evaluator")
                    evaluators.append(identity)
                    record = self._replace_record(record, evaluators=tuple(evaluators))
                    self._emit(
                        "evaluator_started",
                        record,
                        workerIndex=worker_index,
                        process=identity.to_dict(),
                    )
                record = self._replace_record(record, phase="evaluating")
                self._emit("evaluation_started", record)
                yield record
            except BaseException as exc:
                self._emit_exception("evaluation_lease_failed", record, exc)
                raise
            finally:
                body_exception_active = sys.exc_info()[0] is not None
                if evaluators or record.evaluators:
                    try:
                        record = self._replace_record(
                            self._active_record or record,
                            phase="draining_evaluators",
                            evaluators=tuple(evaluators) or record.evaluators,
                        )
                        self._drain_evaluators(record)
                    except BaseException as exc:
                        self._set_safety_halt(
                            self._active_record or record,
                            f"evaluator drain failed: {exc}",
                        )
                        if not body_exception_active:
                            raise

    lease = evaluator_lease

    def request_safety_halt(self, reason: str) -> None:
        if not reason.strip():
            raise GpuLeaseError(
                "invalid_safety_halt", "Safety halt reason must not be empty"
            )
        self._require_mutation()
        if self._active_record is None:
            raise GpuLeaseError("no_active_lease", "No active lease can be halted")
        self._active_record = self._set_safety_halt(self._active_record, reason.strip())

    def read_record(self) -> Optional[LeaseRecord]:
        return self._load_record()

    def reconcile(self, *, mutate: Optional[bool] = None) -> ReconcileReport:
        should_mutate = self.config.mutation_enabled if mutate is None else mutate
        record = self._load_record()
        if record is None:
            return ReconcileReport(None, None, (), False, False)

        if not should_mutate:
            actions = self._planned_reconcile_actions(record)
            return ReconcileReport(
                record.phase,
                record.phase,
                tuple(actions),
                False,
                record.safety_halt,
            )

        self._require_mutation()
        with self._exclusive_lock():
            record = self._load_record()
            if record is None:
                return ReconcileReport(None, None, (), False, False)
            previous_phase = record.phase
            actions, current = self._reconcile_locked(record)
            return ReconcileReport(
                previous_phase,
                current.phase,
                tuple(actions),
                bool(actions),
                current.safety_halt,
            )

    def _reconcile_locked(self, record: LeaseRecord) -> Tuple[List[str], LeaseRecord]:
        actions: List[str] = []
        self._active_record = record
        if record.expected_gpu_uuid != self.config.expected_gpu_uuid:
            halted = self._set_safety_halt(
                record,
                "lease GPU UUID differs from runtime configuration",
            )
            return ["safety_halt_gpu_uuid_mismatch"], halted
        if record.safety_halt:
            return actions, record

        trainer_alive = record.trainer is not None and self.runner.is_running(
            record.trainer
        )
        restored_trainer_alive = (
            record.restored_trainer is not None
            and self.runner.is_running(record.restored_trainer)
        )
        live_evaluators = tuple(
            identity
            for identity in record.evaluators
            if self.runner.is_running(identity)
        )
        if (trainer_alive or restored_trainer_alive) and live_evaluators:
            halted = self._set_safety_halt(
                record, "trainer and evaluator identities are both live"
            )
            return ["safety_halt_process_overlap"], halted

        if record.phase == "restoring_trainer" and restored_trainer_alive:
            if record.handoff_checkpoint is None:
                halted = self._set_safety_halt(
                    record,
                    "restored trainer is live without checkpoint handoff proof",
                )
                return ["safety_halt_missing_checkpoint_handoff"], halted
            restored = record.restored_trainer
            record = self._replace_record(
                record,
                phase="trainer_running",
                trainer=restored,
                restored_trainer=restored,
                restoration_status="restored",
                evaluators=(),
                owner_id=self.config.owner_id,
            )
            actions.append("completed_trainer_restoration")
            return actions, record

        if record.phase == "trainer_running" and trainer_alive:
            if record.evaluators:
                record = self._replace_record(
                    record,
                    evaluators=(),
                    owner_id=self.config.owner_id,
                )
                actions.append("cleared_stale_evaluator_identities")
            return actions, record

        if record.phase == "draining_trainer":
            try:
                record = self._complete_trainer_drain(
                    record, issue_command=trainer_alive
                )
                actions.append("validated_trainer_checkpoint_handoff")
            except BaseException as exc:
                halted = self._set_safety_halt(
                    record,
                    f"reconcile checkpoint handoff validation failed: {exc}",
                )
                actions.append("safety_halt_checkpoint_handoff")
                return actions, halted
            trainer_alive = False
        elif record.handoff_checkpoint is None:
            halted = self._set_safety_halt(
                record,
                "reconcile refused restore without checkpoint handoff proof",
            )
            actions.append("safety_halt_missing_checkpoint_handoff")
            return actions, halted

        if record.evaluators:
            record = self._replace_record(
                record,
                phase="draining_evaluators",
                evaluators=record.evaluators,
            )
            try:
                record = self._drain_evaluators(record)
                actions.append("drained_evaluators")
            except BaseException as exc:
                halted = self._set_safety_halt(
                    record, f"reconcile evaluator drain failed: {exc}"
                )
                actions.append("safety_halt_evaluator_drain")
                return actions, halted

        if trainer_alive or restored_trainer_alive:
            halted = self._set_safety_halt(
                record,
                "unexpected live trainer during incomplete handoff",
            )
            actions.append("safety_halt_unexpected_trainer")
            return actions, halted

        try:
            release_times = self._require_clean_gpu()
            record = self._replace_record(
                record,
                phase="release_gpu_verified",
                release_clean_observation_times=release_times,
            )
            self._require_recorded_handoff_checkpoint(record)
            record = self._restore_trainer(record)
            actions.append("restored_trainer")
            return actions, record
        except BaseException as exc:
            halted = self._set_safety_halt(
                record, f"reconcile trainer restoration failed: {exc}"
            )
            actions.append("safety_halt_trainer_restore")
            return actions, halted

    def _planned_reconcile_actions(self, record: LeaseRecord) -> List[str]:
        if record.safety_halt:
            return ["operator_clear_safety_halt"]
        trainer_alive = record.trainer is not None and self.runner.is_running(
            record.trainer
        )
        restored_alive = record.restored_trainer is not None and self.runner.is_running(
            record.restored_trainer
        )
        evaluator_alive = any(
            self.runner.is_running(identity) for identity in record.evaluators
        )
        if (trainer_alive or restored_alive) and evaluator_alive:
            return ["safety_halt_process_overlap"]
        if record.phase == "restoring_trainer" and restored_alive:
            return ["complete_trainer_restoration"]
        if record.phase == "draining_trainer":
            if record.pre_drain_checkpoint is None:
                return ["safety_halt_missing_pre_drain_checkpoint"]
            return [
                "complete_trainer_drain"
                if trainer_alive
                else "validate_checkpoint_handoff",
                "restore_trainer",
            ]
        if record.phase == "trainer_running" and trainer_alive:
            return ["clear_stale_evaluator_identities"] if record.evaluators else []
        if record.handoff_checkpoint is None:
            return ["safety_halt_missing_checkpoint_handoff"]
        actions: List[str] = []
        if evaluator_alive or record.evaluators:
            actions.append("drain_evaluators")
        if not trainer_alive and not restored_alive:
            actions.append("verify_release_gpu")
            actions.append("restore_trainer")
        return actions

    def _complete_trainer_drain(
        self, record: LeaseRecord, *, issue_command: bool
    ) -> LeaseRecord:
        identity = record.trainer
        before = record.pre_drain_checkpoint
        if identity is None or before is None:
            raise GpuLeaseError(
                "missing_pre_drain_checkpoint",
                "Drain state lacks trainer or pre-drain checkpoint proof",
            )
        trainer_alive = self.runner.is_running(identity)
        current_identity = self.runner.current_identity(identity.pid)
        identity_reused = current_identity is not None and not identity.same_process_as(
            current_identity
        )
        if issue_command and trainer_alive:
            self._request_trainer_drain(identity)
        elif issue_command and not trainer_alive:
            if not identity_reused and self.runner.process_group_alive(identity):
                raise GpuLeaseError(
                    "unverifiable_trainer_process_group",
                    "Trainer PID is gone but its process group is still live",
                    details={"trainer": identity.to_dict()},
                )
        elif not issue_command and trainer_alive:
            raise GpuLeaseError(
                "trainer_still_running",
                "Cannot validate a completed handoff while trainer is live",
            )

        current_identity = self.runner.current_identity(identity.pid)
        identity_reused = current_identity is not None and not identity.same_process_as(
            current_identity
        )
        if self.runner.is_running(identity) or (
            not identity_reused and self.runner.process_group_alive(identity)
        ):
            self._wait_for_exit(
                identity,
                self.config.trainer_drain_timeout_seconds,
                code="trainer_drain_timeout",
            )
        handoff = self._wait_for_checkpoint_identity(
            before=before,
            require_change=self.config.require_checkpoint_change,
        )
        record = self._replace_record(
            record,
            phase="trainer_drained",
            checkpoint_sha256=handoff.sha256,
            checkpoint_size=handoff.size,
            handoff_checkpoint=handoff,
            restoration_status="pending",
        )
        self._emit(
            "trainer_drained",
            record,
            handoffCheckpoint=handoff.to_dict(),
        )
        return record

    def _request_trainer_drain(self, identity: ProcessIdentity) -> None:
        self._require_identity_alive(identity, "trainer")
        if self.config.trainer_graceful_command:
            argv = self._expand(
                self.config.trainer_graceful_command,
                record=self._active_record,
                identity=identity,
            )
            result = self.runner.run(
                argv, timeout=self.config.trainer_drain_timeout_seconds
            )
            if result.returncode != 0:
                raise GpuLeaseError(
                    "trainer_graceful_command_failed",
                    "Trainer graceful-drain command failed",
                    details={
                        "returncode": result.returncode,
                        "stderr": result.stderr,
                    },
                )
        else:
            self.runner.signal_process_group(identity, signal.SIGTERM)

    def _wait_for_checkpoint_identity(
        self,
        *,
        before: Optional[CheckpointIdentity],
        require_change: bool,
    ) -> CheckpointIdentity:
        deadline = self.clock() + self.config.trainer_checkpoint_timeout_seconds
        stable_since: Optional[float] = None
        previous: Optional[Tuple[int, int]] = None
        while True:
            current = _checkpoint_stat(self.config.trainer_checkpoint_path)
            if current is not None:
                if current != previous:
                    previous = current
                    stable_since = self.clock()
                if (
                    stable_since is not None
                    and self.clock() - stable_since
                    >= self.config.trainer_checkpoint_stable_seconds
                ):
                    path = self.config.trainer_checkpoint_path
                    identity = _stable_checkpoint_identity(path)
                    if identity is not None:
                        changed = (
                            before is None
                            or not require_change
                            or identity.content_changed_from(before)
                        )
                        if changed:
                            return identity
            if self.clock() >= deadline:
                raise GpuLeaseError(
                    "trainer_checkpoint_timeout",
                    "Trainer exited without a complete checkpoint handoff",
                    details={
                        "checkpointPath": str(self.config.trainer_checkpoint_path),
                        "requireCheckpointChange": require_change,
                        "preDrainCheckpoint": (
                            None if before is None else before.to_dict()
                        ),
                    },
                )
            self.sleep(self.config.poll_interval_seconds)

    def _require_recorded_handoff_checkpoint(
        self, record: LeaseRecord
    ) -> CheckpointIdentity:
        expected = record.handoff_checkpoint
        if expected is None:
            raise GpuLeaseError(
                "missing_handoff_checkpoint",
                "Refusing trainer restore without handoff checkpoint proof",
            )
        observed = _stable_checkpoint_identity(self.config.trainer_checkpoint_path)
        if observed is None or observed != expected:
            raise GpuLeaseError(
                "handoff_checkpoint_changed",
                "Trainer checkpoint no longer matches the validated handoff",
                details={
                    "expected": expected.to_dict(),
                    "observed": (None if observed is None else observed.to_dict()),
                },
            )
        return observed

    def _drain_evaluators(self, record: LeaseRecord) -> LeaseRecord:
        live_evaluators = [
            (worker_index, identity)
            for worker_index, identity in enumerate(record.evaluators)
            if self.runner.is_running(identity)
        ]
        for worker_index, identity in live_evaluators:
            if self.config.evaluator_drain_command:
                argv = self._expand(
                    self.config.evaluator_drain_command,
                    record=record,
                    identity=identity,
                    worker_index=worker_index,
                )
                result = self.runner.run(
                    argv, timeout=self.config.evaluator_drain_timeout_seconds
                )
                if result.returncode != 0:
                    raise GpuLeaseError(
                        "evaluator_drain_command_failed",
                        "Evaluator drain command failed",
                        details={
                            "workerIndex": worker_index,
                            "returncode": result.returncode,
                            "stderr": result.stderr,
                        },
                    )
            else:
                self.runner.signal_process_group(identity, signal.SIGTERM)

        for _, identity in live_evaluators:
            if self.runner.is_running(identity) or self.runner.process_group_alive(
                identity
            ):
                self._wait_for_exit(
                    identity,
                    self.config.evaluator_drain_timeout_seconds,
                    code="evaluator_drain_timeout",
                )
        record = self._replace_record(
            record,
            phase="safety_halt" if record.safety_halt else "evaluator_drained",
            evaluators=(),
        )
        self._emit("evaluators_drained", record)
        return record

    def _restore_trainer(self, record: LeaseRecord) -> LeaseRecord:
        self._require_recorded_handoff_checkpoint(record)
        record = self._replace_record(
            record,
            phase="restoring_trainer",
            evaluators=(),
            owner_id=self.config.owner_id,
            restoration_status="pending",
        )
        argv = self._expand(
            self.config.trainer_launch_command,
            record=record,
            identity=record.trainer,
        )
        identity = self.runner.spawn(argv, new_process_group=True)
        record = self._replace_record(
            record,
            restored_trainer=identity,
            restoration_status="started",
        )
        deadline = self.clock() + self.config.trainer_start_timeout_seconds
        while not self.runner.is_running(identity):
            if self.clock() >= deadline:
                raise GpuLeaseError(
                    "trainer_restart_failed",
                    "Restarted trainer did not remain alive",
                    details={"process": identity.to_dict()},
                )
            self.sleep(self.config.poll_interval_seconds)
        record = self._replace_record(
            record,
            phase="trainer_running",
            trainer=identity,
            evaluators=(),
            safety_halt=False,
            safety_reason=None,
            restoration_status="restored",
            restored_trainer=identity,
        )
        self._emit("trainer_restored", record, process=identity.to_dict())
        return record

    def _wait_for_exit(
        self, identity: ProcessIdentity, timeout: float, *, code: str
    ) -> None:
        deadline = self.clock() + timeout
        while True:
            current = self.runner.current_identity(identity.pid)
            if current is not None and not identity.same_process_as(current):
                return
            if not self.runner.is_running(
                identity
            ) and not self.runner.process_group_alive(identity):
                return
            if self.clock() >= deadline:
                raise GpuLeaseError(
                    code,
                    "Process group did not exit before its deadline",
                    details={"process": identity.to_dict(), "timeoutSeconds": timeout},
                )
            self.sleep(self.config.poll_interval_seconds)

    def _require_clean_gpu(self) -> Tuple[float, ...]:
        observation_times: List[float] = []
        for observation_index in range(self.config.clean_observations):
            try:
                observation = self.gpu_probe()
            except GpuLeaseError:
                raise
            except BaseException as exc:
                raise GpuLeaseError(
                    "gpu_probe_failed", f"GPU probe failed: {exc}"
                ) from exc
            if observation.gpu_uuid != self.config.expected_gpu_uuid:
                raise GpuLeaseError(
                    "gpu_uuid_mismatch",
                    "GPU observation did not identify the configured UUID",
                    details={
                        "expectedGpuUuid": self.config.expected_gpu_uuid,
                        "observedGpuUuid": observation.gpu_uuid,
                    },
                )
            if observation.processes:
                raise GpuLeaseError(
                    "foreign_gpu_process",
                    "GPU is not clean; refusing exclusive lease handoff",
                    details={
                        "gpuUuid": observation.gpu_uuid,
                        "processes": [
                            process.to_dict() for process in observation.processes
                        ],
                    },
                )
            observed_at = self.clock()
            observation_times.append(observed_at)
            self._emit(
                "gpu_clean_observation",
                self._active_record,
                observationIndex=observation_index + 1,
                requiredObservations=self.config.clean_observations,
                observedAt=observed_at,
            )
            if observation_index + 1 < self.config.clean_observations:
                self.sleep(self.config.clean_observation_interval_seconds)
        return tuple(observation_times)

    def _require_identity_alive(
        self, identity: ProcessIdentity, process_role: str
    ) -> None:
        if not identity.is_verifiable:
            raise GpuLeaseError(
                "unverifiable_process_identity",
                f"{process_role} identity cannot distinguish PID reuse",
                details={"process": identity.to_dict()},
            )
        current = self.runner.current_identity(identity.pid)
        if current is None or not identity.same_process_as(current):
            raise GpuLeaseError(
                "process_identity_changed",
                f"{process_role} PID is absent or has been reused",
                details={
                    "expected": identity.to_dict(),
                    "observed": None if current is None else current.to_dict(),
                },
            )

    def _expand(
        self,
        template: Sequence[str],
        *,
        record: Optional[LeaseRecord],
        identity: Optional[ProcessIdentity],
        worker_index: int = 0,
    ) -> List[str]:
        values = {
            "run_root": str(self.config.run_root),
            "promotion_root": str(self.config.promotion_root),
            "lease_state": str(self.config.lease_state_path),
            "event_log": str(self.config.event_log_path),
            "checkpoint_path": str(self.config.trainer_checkpoint_path),
            "gpu_uuid": self.config.expected_gpu_uuid,
            "gpu_index": str(self.config.gpu_index),
            "worker_index": str(worker_index),
            "lease_id": "" if record is None else record.lease_id,
            "owner_id": self.config.owner_id,
            "pid": "" if identity is None else str(identity.pid),
            "process_group_id": (
                ""
                if identity is None or identity.process_group_id is None
                else str(identity.process_group_id)
            ),
            "checkpoint_sha256": (
                ""
                if record is None or record.checkpoint_sha256 is None
                else record.checkpoint_sha256
            ),
        }
        try:
            argv = [part.format_map(values) for part in template]
        except KeyError as exc:
            raise GpuLeaseError(
                "unknown_command_placeholder",
                f"Unknown command placeholder: {exc}",
            ) from exc
        _validate_argv(argv, "expanded command")
        _reject_sigstop(argv, "expanded command")
        return argv

    def _replace_record(self, record: LeaseRecord, **changes: Any) -> LeaseRecord:
        changes["updated_at"] = self.clock()
        updated = dataclasses.replace(record, **changes)
        self._active_record = updated
        self._save_record(updated)
        return updated

    def _set_safety_halt(self, record: LeaseRecord, reason: str) -> LeaseRecord:
        updated = self._replace_record(
            record,
            phase="safety_halt",
            safety_halt=True,
            safety_reason=reason,
            restoration_status="safety_halt",
        )
        self._emit("safety_halt", updated, reason=reason)
        return updated

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if fcntl is None:
            raise GpuLeaseError(
                "locking_unavailable",
                "POSIX advisory locking is unavailable on this platform",
            )
        _validate_runtime_paths(self.config)
        lock_path = self.config.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise GpuLeaseError(
                        "lease_already_held",
                        "Another controller holds the GPU lease lock",
                        details={"lockPath": str(lock_path)},
                    ) from exc
                raise
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def _save_record(self, record: LeaseRecord) -> None:
        self._require_mutation()
        _validate_runtime_paths(self.config)
        path = self.config.lease_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, record.to_dict())

    def _load_record(self) -> Optional[LeaseRecord]:
        _validate_runtime_paths(self.config)
        path = self.config.lease_state_path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GpuLeaseError(
                "lease_state_read_failed",
                f"Could not read lease state: {exc}",
                details={"path": str(path)},
            ) from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GpuLeaseError(
                "invalid_lease_state",
                f"Lease state is not valid JSON: {exc}",
                details={"path": str(path)},
            ) from exc
        if not isinstance(value, Mapping):
            raise GpuLeaseError(
                "invalid_lease_state", "Lease state root must be an object"
            )
        return LeaseRecord.from_dict(value)

    def _emit(
        self,
        event_type: str,
        record: Optional[LeaseRecord],
        **details: Any,
    ) -> None:
        event: Dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "event": event_type,
            "time": self.clock(),
            "ownerId": self.config.owner_id,
            "leaseId": None if record is None else record.lease_id,
            "phase": None if record is None else record.phase,
            "details": details,
        }
        if self._event_sink is not None:
            self._event_sink(event)
            return
        if not self.config.mutation_enabled:
            return
        _validate_runtime_paths(self.config)
        path = self.config.event_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_json(event)
        with path.open("ab") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())

    def _emit_exception(
        self,
        event_type: str,
        record: Optional[LeaseRecord],
        exc: BaseException,
    ) -> None:
        if isinstance(exc, GpuLeaseError):
            error = exc.to_dict()["error"]
        else:
            error = {
                "code": "unexpected_error",
                "message": str(exc),
                "details": {"type": type(exc).__name__},
            }
        self._emit(event_type, record, error=error)

    def _require_mutation(self) -> None:
        if not self.config.mutation_enabled:
            raise GpuLeaseError(
                "mutation_disabled",
                "External process and filesystem mutation is disabled by runtime config",
            )


def _read_optional_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _optional_checkpoint_identity(
    value: Any,
) -> Optional[CheckpointIdentity]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GpuLeaseError(
            "invalid_lease_state", "Checkpoint identity must be an object"
        )
    return CheckpointIdentity.from_dict(value)


def _float_tuple(value: Any) -> Tuple[float, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise GpuLeaseError(
            "invalid_lease_state", "Observation times must be numeric arrays"
        )
    result = tuple(float(item) for item in value)
    if any(item < 0 for item in result):
        raise GpuLeaseError(
            "invalid_lease_state", "Observation times must not be negative"
        )
    return result


def _validate_runtime_paths(config: RuntimeConfig) -> None:
    paths = {
        "runRoot": config.run_root,
        "promotionRoot": config.promotion_root,
        "leaseState": config.lease_state_path,
        "eventLog": config.event_log_path,
        "checkpointPath": config.trainer_checkpoint_path,
        "lockPath": config.lock_path,
    }
    normalized: Dict[str, Path] = {}
    for name, path in paths.items():
        if not path.is_absolute():
            raise GpuLeaseError(
                "invalid_runtime_path",
                f"{name} must be absolute",
                details={"path": str(path)},
            )
        canonical = Path(os.path.abspath(os.fspath(path)))
        if canonical != path:
            raise GpuLeaseError(
                "invalid_runtime_path",
                f"{name} must be lexically normalized",
                details={"path": str(path), "normalized": str(canonical)},
            )
        normalized[name] = canonical
        _reject_existing_symlink_ancestors(canonical, name)

    run_root = normalized["runRoot"]
    promotion_root = normalized["promotionRoot"]
    if not _is_strictly_within(promotion_root, run_root):
        raise GpuLeaseError(
            "invalid_runtime_path",
            "promotionRoot must be strictly contained by runRoot",
        )
    for name in ("leaseState", "eventLog", "lockPath"):
        if not _is_strictly_within(normalized[name], promotion_root):
            raise GpuLeaseError(
                "invalid_runtime_path",
                f"{name} must be strictly contained by promotionRoot",
            )
    checkpoint = normalized["checkpointPath"]
    if not _is_strictly_within(checkpoint, run_root) or _is_within(
        checkpoint, promotion_root
    ):
        raise GpuLeaseError(
            "invalid_runtime_path",
            "checkpointPath must be inside runRoot and outside promotionRoot",
        )

    sensitive_names = ("leaseState", "eventLog", "checkpointPath", "lockPath")
    sensitive_paths = [normalized[name] for name in sensitive_names]
    if len(set(sensitive_paths)) != len(sensitive_paths):
        raise GpuLeaseError(
            "runtime_path_alias",
            "State, event, checkpoint, and lock paths must be distinct",
        )
    for index, left in enumerate(sensitive_paths):
        for right in sensitive_paths[index + 1 :]:
            if left.exists() and right.exists():
                try:
                    if os.path.samefile(left, right):
                        raise GpuLeaseError(
                            "runtime_path_alias",
                            "Runtime files must not alias through hard links",
                            details={"left": str(left), "right": str(right)},
                        )
                except FileNotFoundError:
                    pass

    for name in ("runRoot", "promotionRoot"):
        path = normalized[name]
        if path.exists() and not path.is_dir():
            raise GpuLeaseError("invalid_runtime_path", f"{name} must be a directory")
    for name in ("leaseState", "eventLog", "lockPath"):
        path = normalized[name]
        if path.exists() and not path.is_file():
            raise GpuLeaseError(
                "invalid_runtime_path", f"{name} must be a regular file"
            )
    if checkpoint.exists() and (checkpoint.is_symlink() or not checkpoint.is_file()):
        raise GpuLeaseError(
            "invalid_runtime_path",
            "checkpointPath must be a regular file when it exists",
        )


def _reject_existing_symlink_ancestors(path: Path, name: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise GpuLeaseError(
                "runtime_path_symlink",
                f"{name} has a symlinked path component",
                details={"component": str(current)},
            )
        if current.parent == current:
            break
        current = current.parent


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_strictly_within(path: Path, parent: Path) -> bool:
    return path != parent and _is_within(path, parent)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise GpuLeaseError("invalid_runtime_config", f"{key} must be an object")
    return child


def _str(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise GpuLeaseError(
            "invalid_runtime_config", f"{key} must be a non-empty string"
        )
    return result


def _bool(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise GpuLeaseError("invalid_runtime_config", f"{key} must be a boolean")
    return result


def _int(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise GpuLeaseError(
            "invalid_runtime_config",
            f"{key} must be an integer >= {minimum}",
        )
    return result


def _number(value: Mapping[str, Any], key: str, *, minimum: float) -> float:
    result = value.get(key)
    if (
        isinstance(result, bool)
        or not isinstance(result, (int, float))
        or float(result) < minimum
    ):
        raise GpuLeaseError(
            "invalid_runtime_config",
            f"{key} must be a number >= {minimum}",
        )
    return float(result)


def _command(
    value: Mapping[str, Any], key: str, *, allow_empty: bool = False
) -> Tuple[str, ...]:
    result = value.get(key)
    if not isinstance(result, list):
        raise GpuLeaseError(
            "invalid_runtime_config", f"{key} must be a JSON argv array"
        )
    if not allow_empty and not result:
        raise GpuLeaseError("invalid_runtime_config", f"{key} must not be empty")
    if not all(isinstance(part, str) and part for part in result):
        raise GpuLeaseError(
            "invalid_runtime_config",
            f"{key} entries must be non-empty strings",
        )
    return tuple(result)


def _validate_argv(argv: Sequence[str], name: str) -> None:
    if not argv or not all(isinstance(part, str) and part for part in argv):
        raise GpuLeaseError(
            "invalid_command", f"{name} must be a non-empty argv sequence"
        )


def _reject_sigstop(argv: Sequence[str], name: str) -> None:
    prohibited = {"sigstop", "stop", "-stop", "-sigstop", "-19"}
    if any(part.strip().lower() in prohibited for part in argv):
        raise GpuLeaseError("sigstop_prohibited", f"{name} must never use SIGSTOP")


def _checkpoint_stat(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink() or not stat.S_ISREG(stat_result.st_mode):
        return None
    return (stat_result.st_size, stat_result.st_mtime_ns)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_checkpoint_identity(
    path: Path,
) -> Optional[CheckpointIdentity]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            return None
        digest = _sha256_file(path)
        after = path.lstat()
    except (FileNotFoundError, OSError):
        return None
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
        return None
    return CheckpointIdentity(
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(str(temporary_path), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary_path.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _status_payload(
    manager: GpuLeaseManager, record: Optional[LeaseRecord]
) -> Dict[str, Any]:
    if record is None:
        return {"lease": None}
    trainer_alive = record.trainer is not None and manager.runner.is_running(
        record.trainer
    )
    evaluators_alive = [
        identity.to_dict()
        for identity in record.evaluators
        if manager.runner.is_running(identity)
    ]
    return {
        "lease": record.to_dict(),
        "observed": {
            "trainerAlive": trainer_alive,
            "evaluatorsAlive": evaluators_alive,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("status")
    reconcile_parser = subparsers.add_parser("reconcile")
    reconcile_mode = reconcile_parser.add_mutually_exclusive_group()
    reconcile_mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply reconciliation (also requires mutationEnabled=true)",
    )
    reconcile_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without mutation (the default)",
    )
    args = parser.parse_args(argv)

    try:
        config = RuntimeConfig.from_json_file(args.config)
        manager = GpuLeaseManager(config)
        if args.action == "status":
            payload: Mapping[str, Any] = _status_payload(manager, manager.read_record())
        else:
            payload = manager.reconcile(mutate=args.apply).to_dict()
        sys.stdout.buffer.write(_canonical_json(payload))
        return 0
    except GpuLeaseError as exc:
        sys.stderr.buffer.write(_canonical_json(exc.to_dict()))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
