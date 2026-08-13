#!/usr/bin/env python3
"""Materialize hash-pinned live promotion runtime JSON from checked-in examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from risk_score.paired_stats import canonical_sha256 as policy_sha256
from risk_score.paired_stats import load_policy
from risk_score.position_samples import canonical_json, file_sha256
from risk_score.promotion_host import HostCommandError, atomic_write_json
from risk_score.promotion_state import atomic_write_bytes
from risk_score.service_activation import (
    AUTONOMY_SERVICE_SPEC_CONTRACT,
    AUTONOMY_SERVICE_UNIT_NAMES,
)


def _replace(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required_file(path: Path, role: str) -> Path:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise HostCommandError(f"{role} must be an absolute regular file")
    return source


def _canonical_json_file(path: Path, role: str) -> Mapping[str, Any]:
    source = _required_file(path, role)
    data = source.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostCommandError(f"{role} is invalid JSON: {exc}") from exc
    if (
        not isinstance(value, dict)
        or data != (canonical_json(value) + "\n").encode("utf-8")
    ):
        raise HostCommandError(f"{role} must be canonical JSON")
    return value


def _self_hashed_spec(
    path: Path,
    *,
    role: str,
    contract: str,
) -> Mapping[str, Any]:
    value = _canonical_json_file(path, role)
    body = dict(value)
    supplied = body.pop("spec_sha256", None)
    if value.get("contract") != contract or supplied != policy_sha256(body):
        raise HostCommandError(f"{role} contract or self-hash is invalid")
    return value


def _service_argv(value: Optional[Sequence[str]], role: str) -> list[str]:
    if value is None:
        return []
    result = list(value)
    if not result or any(
        not isinstance(argument, str)
        or not argument
        or "\n" in argument
        or "\r" in argument
        for argument in result
    ):
        raise HostCommandError(f"{role} service command must be a nonempty argv")
    if not Path(result[0]).is_absolute():
        raise HostCommandError(f"{role} service executable must be absolute")
    return result


def _reconcile_argv(value: Sequence[str], role: str) -> list[str]:
    result = list(value)
    indexes = [index for index, argument in enumerate(result) if argument == "watch"]
    if len(indexes) != 1:
        raise HostCommandError(
            f"{role} full-autonomy command must contain one watch mode"
        )
    result[indexes[0]] = "once"
    return result


def _systemd_quote(value: str) -> str:
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def _systemd_service(
    *,
    description: str,
    argv: Sequence[str],
    service_user: str,
    working_directory: Path,
    run_root: Path,
    environment: Optional[Mapping[str, str]] = None,
    after: Sequence[str] = (),
    before: Sequence[str] = (),
    requires: Sequence[str] = (),
    restart: str = "on-failure",
    service_type: str = "simple",
    additional_argv: Sequence[Sequence[str]] = (),
    remain_after_exit: bool = False,
) -> str:
    if restart not in {
        "no",
        "on-success",
        "on-failure",
        "on-abnormal",
        "on-watchdog",
        "on-abort",
        "always",
    }:
        raise HostCommandError("systemd restart policy is invalid")
    if service_type not in {"simple", "oneshot"}:
        raise HostCommandError("systemd service type is invalid")
    if additional_argv and service_type != "oneshot":
        raise HostCommandError("multiple ExecStart commands require a oneshot service")
    if remain_after_exit and service_type != "oneshot":
        raise HostCommandError("RemainAfterExit requires a oneshot service")
    commands = (tuple(argv), *(tuple(command) for command in additional_argv))
    if any(
        not command
        or any(
            not isinstance(argument, str)
            or not argument
            or "\n" in argument
            or "\r" in argument
            for argument in command
        )
        for command in commands
    ):
        raise HostCommandError("systemd service command is invalid")
    unit_after = ["network-online.target", *after]
    lines = [
        "[Unit]",
        f"Description={description}",
        "Wants=network-online.target",
        "PartOf=katago-risk-training.target",
        "After=" + " ".join(unit_after),
        "RequiresMountsFor=" + _systemd_quote(str(run_root)),
        "StartLimitIntervalSec=300",
        "StartLimitBurst=3",
    ]
    if requires:
        lines.append("Requires=" + " ".join(requires))
    if before:
        lines.append("Before=" + " ".join(before))
    lines.extend(
        [
            "",
            "[Service]",
            f"Type={service_type}",
            f"User={service_user}",
            "WorkingDirectory=" + _systemd_quote(str(working_directory)),
            "Environment=" + _systemd_quote(f"PYTHONPATH={working_directory}"),
        ]
    )
    for key, value in sorted((environment or {}).items()):
        if (
            re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) is None
            or not isinstance(value, str)
            or "\n" in value
            or "\r" in value
        ):
            raise HostCommandError("systemd service environment is invalid")
        lines.append("Environment=" + _systemd_quote(f"{key}={value}"))
    for command in commands:
        lines.append(
            "ExecStart=" + " ".join(_systemd_quote(value) for value in command)
        )
    if remain_after_exit:
        lines.append("RemainAfterExit=yes")
    lines.extend(
        [
            f"Restart={restart}",
            "RestartSec=5",
            "KillSignal=SIGINT",
            "KillMode=control-group",
            "TimeoutStopSec=300",
            "",
            "[Install]",
            "WantedBy=katago-risk-training.target",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_file(path: Path, data: str) -> None:
    encoded = data.encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise HostCommandError(f"generated service artifact conflicts: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, encoded)


def _parse_command_json(value: Optional[str], role: str) -> Optional[Sequence[str]]:
    if value is None:
        return None
    try:
        command = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HostCommandError(f"{role} command JSON is invalid: {exc}") from exc
    if not isinstance(command, list):
        raise HostCommandError(f"{role} command JSON must be an argv array")
    return command


def build_live_runtime(
    *,
    repo: Path,
    run_root: Path,
    suite_dir: Path,
    katago_binary: Path,
    python_executable: Path,
    trainer_spec: Path,
    consumer_spec: Path,
    original_model: Path,
    trainer_checkpoint: Path,
    gpu_uuid: str,
    actor: str,
    source_revision: str,
    output_dir: Path,
    mutation_enabled: bool = False,
    require_clean_source: bool = True,
    service_user: Optional[str] = None,
    shuffler_command: Optional[Sequence[str]] = None,
    exporter_command: Optional[Sequence[str]] = None,
    evaluator_command: Optional[Sequence[str]] = None,
    full_autonomy: bool = False,
    cluster_executor_command: Optional[Sequence[str]] = None,
    adaptive_training_command: Optional[Sequence[str]] = None,
    suite_rotation_command: Optional[Sequence[str]] = None,
    autonomy_policy: Optional[Path] = None,
    cluster_executor_spec: Optional[Path] = None,
    adaptive_training_spec: Optional[Path] = None,
    suite_registry_spec: Optional[Path] = None,
    evaluator_process_count: int = 8,
) -> Mapping[str, Any]:
    repository = Path(repo).resolve()
    root = Path(run_root).resolve()
    suites = Path(suite_dir).resolve()
    output = Path(output_dir).resolve()
    if repository.is_symlink() or not repository.is_dir():
        raise HostCommandError("repository must be a non-symlink directory")
    if root.is_symlink() or not root.is_dir():
        raise HostCommandError("run root must be a non-symlink directory")
    if suites.is_symlink() or not suites.is_dir():
        raise HostCommandError("suite directory must be a non-symlink directory")
    if not actor or not gpu_uuid.startswith("GPU-"):
        raise HostCommandError("actor and verified GPU UUID are required")
    if (
        type(evaluator_process_count) is not int
        or evaluator_process_count not in {4, 8, 16}
    ):
        raise HostCommandError(
            "evaluator process count must be benchmarked at 4, 8, or 16"
        )
    shuffler_argv = _service_argv(shuffler_command, "shuffler")
    exporter_argv = _service_argv(exporter_command, "exporter")
    evaluator_argv = _service_argv(evaluator_command, "evaluator")
    executor_argv = _service_argv(
        cluster_executor_command, "cluster executor"
    )
    adaptive_argv = _service_argv(
        adaptive_training_command, "adaptive training"
    )
    suite_rotation_argv = _service_argv(
        suite_rotation_command, "suite rotation"
    )
    autonomy_commands = (executor_argv, adaptive_argv, suite_rotation_argv)
    runtime_autonomy: Optional[Mapping[str, Any]] = None
    if type(full_autonomy) is not bool:
        raise HostCommandError("full autonomy flag must be boolean")
    if full_autonomy and (
        not mutation_enabled
        or not all(autonomy_commands)
    ):
        raise HostCommandError(
            "full autonomy requires mutation mode and every autonomy service command"
        )
    if not full_autonomy and any(autonomy_commands):
        raise HostCommandError(
            "autonomy service commands require full autonomy mode"
        )
    autonomy_input_paths: Dict[str, Path] = {}
    if full_autonomy:
        raw_inputs = {
            "autonomy_policy": autonomy_policy,
            "executor_spec": cluster_executor_spec,
            "adaptive_spec": adaptive_training_spec,
            "suite_registry_spec": suite_registry_spec,
        }
        if any(path is None for path in raw_inputs.values()):
            raise HostCommandError(
                "full autonomy requires every hash-bound service input"
            )
        autonomy_input_paths = {
            name: _required_file(Path(path), name.replace("_", " "))
            for name, path in raw_inputs.items()
        }
        for role, command, spec_name in (
            ("cluster executor", executor_argv, "executor_spec"),
            ("adaptive training", adaptive_argv, "adaptive_spec"),
            ("suite rotation", suite_rotation_argv, "suite_registry_spec"),
        ):
            if str(autonomy_input_paths[spec_name]) not in command:
                raise HostCommandError(
                    f"{role} command does not bind its frozen specification"
                )
        from risk_score.adaptive_training import (
            POLICY_HASH as AUTONOMY_POLICY_HASH,
        )

        policy_value = _canonical_json_file(
            autonomy_input_paths["autonomy_policy"],
            "autonomy policy",
        )
        if policy_sha256(policy_value) != AUTONOMY_POLICY_HASH:
            raise HostCommandError("autonomy policy identity is not frozen")
        executor_spec_value = _self_hashed_spec(
            autonomy_input_paths["executor_spec"],
            role="cluster executor specification",
            contract="risk-score-cluster-executor-spec-v1",
        )
        adaptive_spec_value = _self_hashed_spec(
            autonomy_input_paths["adaptive_spec"],
            role="adaptive training specification",
            contract="risk-score-adaptive-training-service-spec-v1",
        )
        suite_service_value = _self_hashed_spec(
            autonomy_input_paths["suite_registry_spec"],
            role="suite rotation service specification",
            contract="risk-score-suite-rotation-service-spec-v1",
        )
        scheduler_paths = {
            executor_spec_value.get("scheduler_directory"),
            adaptive_spec_value.get("scheduler_directory"),
            suite_service_value.get("scheduler_directory"),
        }
        guardian_prefixes = {
            canonical_json(executor_spec_value.get("gpu7_guardian_prefix")),
            canonical_json(
                adaptive_spec_value.get(
                    "gpu_lease_guardian_argv_prefix"
                )
            ),
            canonical_json(suite_service_value.get("guardian_argv_prefix")),
        }
        if (
            len(scheduler_paths) != 1
            or None in scheduler_paths
            or {
                executor_spec_value.get("gpu7_id"),
                adaptive_spec_value.get("gpu7_id"),
                suite_service_value.get("gpu7_id"),
            }
            != {"7"}
            or len(guardian_prefixes) != 1
            or "null" in guardian_prefixes
        ):
            raise HostCommandError(
                "autonomy services disagree on scheduler, GPU7, or guardian"
            )
        if (
            adaptive_spec_value.get("autonomy_policy_path")
            != str(autonomy_input_paths["autonomy_policy"])
            or adaptive_spec_value.get("autonomy_policy_sha256")
            != AUTONOMY_POLICY_HASH
        ):
            raise HostCommandError(
                "adaptive service does not bind the frozen autonomy policy"
            )
        registry_binding = suite_service_value.get("registry_spec")
        if not isinstance(registry_binding, Mapping):
            raise HostCommandError(
                "suite rotation service has no registry specification"
            )
        registry_path = Path(str(registry_binding.get("path", "")))
        if (
            not registry_path.is_absolute()
            or registry_path.is_symlink()
            or not registry_path.is_file()
            or file_sha256(registry_path) != registry_binding.get("sha256")
        ):
            raise HostCommandError(
                "suite registry specification changed or is unsafe"
            )
        registry_value = _self_hashed_spec(
            registry_path,
            role="suite registry specification",
            contract="risk-score-evaluation-suite-registry-spec-v1",
        )
        active_suite_path = (
            Path(str(registry_value.get("registry_root", "")))
            / "active-suite.json"
        )
        active_suite = _canonical_json_file(
            active_suite_path, "active suite pointer"
        )
        active_body = dict(active_suite)
        active_record_hash = active_body.pop("record_sha256", None)
        manifest_path = suites / "manifest.json"
        if (
            active_suite.get("contract")
            != "risk-score-active-evaluation-suite-v1"
            or active_record_hash != policy_sha256(active_body)
            or active_suite.get("spec_sha256")
            != registry_value.get("spec_sha256")
            or active_suite.get("manifest_path") != str(manifest_path)
            or active_suite.get("manifest_sha256")
            != file_sha256(manifest_path)
            or suites.name != active_suite.get("suite_id")
        ):
            raise HostCommandError(
                "active suite pointer does not match the frozen runtime suite"
            )
        runtime_autonomy = {
            "suiteRegistrySpec": {
                "path": str(registry_path),
                "sha256": file_sha256(registry_path),
                "identity": registry_value["spec_sha256"],
            },
            "activeSuitePointer": {
                "path": str(active_suite_path),
                "sha256": file_sha256(active_suite_path),
                "recordSha256": active_suite["record_sha256"],
                "suiteId": active_suite["suite_id"],
            },
            "adaptive": {
                "policySha256": AUTONOMY_POLICY_HASH,
                "root": str(root / "promotion" / "adaptive"),
            },
        }
    if mutation_enabled and (
        service_user is None
        or not shuffler_argv
        or not exporter_argv
    ):
        raise HostCommandError(
            "automatic runtime requires service user, shuffler, and exporter commands"
        )
    if evaluator_argv and any(
        "evaluator-unsupported" in argument for argument in evaluator_argv
    ):
        raise HostCommandError(
            "automatic runtime evaluator command must not be evaluator-unsupported"
        )
    original = _required_file(original_model, "original model")
    checkpoint = _required_file(trainer_checkpoint, "trainer checkpoint")
    katago = _required_file(katago_binary, "KataGo binary")
    python = _required_file(Path(python_executable).resolve(), "Python executable")
    trainer_spec_path = _required_file(trainer_spec, "trainer launch spec")
    consumer_spec_path = _required_file(consumer_spec, "consumer stop spec")
    try:
        trainer_spec_value = json.loads(
            trainer_spec_path.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostCommandError(f"trainer launch spec is invalid: {exc}") from exc
    if not isinstance(trainer_spec_value, dict):
        raise HostCommandError("trainer launch spec must be an object")
    if mutation_enabled:
        trainer_argv = trainer_spec_value.get("argv")
        provenance_root = str(
            root / "promotion" / "provenance" / "trainer"
        )
        provenance_index = (
            trainer_argv.index("-generation-provenance-dir")
            if isinstance(trainer_argv, list)
            and "-generation-provenance-dir" in trainer_argv
            else None
        )
        if (
            not isinstance(trainer_argv, list)
            or provenance_index is None
            or provenance_index + 1 >= len(trainer_argv)
            or trainer_argv[provenance_index + 1] != provenance_root
            or "-require-shuffle-provenance" not in trainer_argv
        ):
            raise HostCommandError(
                "automatic trainer spec requires strict generation provenance"
            )
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    actual_revision = completed.stdout.strip()
    if (
        completed.returncode != 0
        or status.returncode != 0
        or (require_clean_source and status.stdout.strip())
        or actual_revision != source_revision
    ):
        raise HostCommandError(
            "runtime source revision must match a clean deployed checkout"
        )
    promotion_example = (
        repository / "python" / "risk_score" / "promotion_runtime.example.json"
    )
    gpu_example = (
        repository / "python" / "risk_score" / "gpu_lease_runtime.example.json"
    )
    promotion = json.loads(promotion_example.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_example.read_text(encoding="utf-8"))
    replacements = {
        "/replace/me/katago-run": str(root),
        "/replace/me/kata-go": str(repository),
    }
    promotion = _replace(promotion, replacements)
    gpu = _replace(gpu, replacements)
    for command in promotion["commands"].values():
        if command and command[0].endswith("/python/.venv/bin/python"):
            command[0] = str(python)
    for key in ("launchCommand", "gracefulCommand"):
        gpu["trainer"][key][0] = str(python)
    promotion_root = root / "promotion"

    gpu["ownerId"] = actor
    gpu["mutationEnabled"] = mutation_enabled
    gpu["paths"] = {
        "runRoot": str(root),
        "promotionRoot": str(promotion_root),
        "leaseState": str(promotion_root / "gpu-lease.json"),
        "eventLog": str(promotion_root / "gpu-lease-events.jsonl"),
    }
    gpu["gpu"]["expectedUuid"] = gpu_uuid
    gpu["evaluator"]["processCount"] = evaluator_process_count
    gpu["trainer"]["checkpointPath"] = str(checkpoint)
    gpu["trainer"]["launchCommand"][
        gpu["trainer"]["launchCommand"].index("--spec") + 1
    ] = str(trainer_spec_path)
    # The controller uses GpuLeaseManager.exclusive_handoff and runs its
    # manifest-bound evaluator adapter inside the yielded lease. This command
    # belongs only to gpu_lease's legacy evaluator_lease launcher, so keep it
    # fail-closed unless an audited external worker command is explicitly set.
    gpu["evaluator"]["launchCommand"] = evaluator_argv or [
        str(python),
        "-m",
        "risk_score.promotion_host",
        "evaluator-unsupported",
    ]
    gpu_path = output / "gpu-lease-runtime.json"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(gpu_path, gpu)

    policy_path = repository / "python" / "risk_score" / "promotion_policy_v3.json"
    powered = repository / "cpp" / "configs" / "risk_score" / "promotion_powered_match.cfg"
    standard = repository / "cpp" / "configs" / "risk_score" / "promotion_standard_match.cfg"
    analysis_config = (
        repository
        / "cpp"
        / "configs"
        / "risk_score"
        / "promotion_curation_analysis.cfg"
    )
    selfplay = (
        repository
        / "cpp"
        / "configs"
        / "risk_score"
        / "promotion_selfplay_worker_19x19.cfg"
    )
    manifest = suites / "manifest.json"
    standard_confirmation = output / "standard-confirmation.jsonl"
    confirmation_source = suites / "schedules" / "confirmation.jsonl"
    confirmation_bytes = confirmation_source.read_bytes()
    if standard_confirmation.exists():
        if standard_confirmation.read_bytes() != confirmation_bytes:
            raise HostCommandError("standard confirmation copy conflicts")
    else:
        standard_confirmation.write_bytes(confirmation_bytes)
    schedules = {
        "discoveryOrdinarySchedule": suites / "schedules" / "discovery.jsonl",
        "confirmationOrdinarySchedule": suites
        / "schedules"
        / "confirmation.jsonl",
        "auditSchedule": suites / "schedules" / "audit.jsonl",
        "lead40Schedule": suites
        / "schedules"
        / "lead-40-confirmation.jsonl",
        "lead80Schedule": suites
        / "schedules"
        / "lead-80-confirmation.jsonl",
        "standardConfirmationSchedule": standard_confirmation,
    }
    for path, role in (
        (policy_path, "promotion policy"),
        (powered, "powered config"),
        (standard, "standard config"),
        (analysis_config, "analysis config"),
        (selfplay, "self-play config"),
        (manifest, "suite manifest"),
        *((path, name) for name, path in schedules.items()),
    ):
        _required_file(path, role)
    promotion["mutationEnabled"] = mutation_enabled
    if full_autonomy:
        if runtime_autonomy is None:
            raise HostCommandError(
                "full-autonomy runtime has no control-plane binding"
            )
        promotion["schemaVersion"] = 2
        promotion["autonomy"] = dict(runtime_autonomy)
    promotion["actor"] = actor
    promotion["commands"]["trainer"][
        promotion["commands"]["trainer"].index("--spec") + 1
    ] = str(trainer_spec_path)
    promotion["commands"]["rollback"][
        promotion["commands"]["rollback"].index("--spec") + 1
    ] = str(promotion_root / "supervisor" / "consumers.json")
    for command_name in ("stage0Probe", "evaluator", "selfplay"):
        command = promotion["commands"][command_name]
        if "--katago" not in command:
            raise HostCommandError(f"{command_name} command has no --katago flag")
        command[command.index("--katago") + 1] = str(katago)
    promotion["paths"].update(
        {
            "promotionRoot": str(promotion_root),
            "controllerLock": str(promotion_root / "controller.lock"),
            "champion": str(promotion_root / "champion.json"),
            "candidateInbox": str(root / "modelstobetested"),
            "candidateQuarantine": str(
                promotion_root / "candidates" / "quarantined"
            ),
            "candidateSuperseded": str(
                promotion_root / "candidates" / "superseded"
            ),
            "candidateRejected": str(promotion_root / "candidates" / "rejected"),
            "candidateDeduplicated": str(
                promotion_root / "candidates" / "deduplicated"
            ),
            "acceptedModels": str(promotion_root / "accepted"),
            "admittedSelfplay": str(root / "selfplay"),
            "rolloutQuarantine": str(promotion_root / "rollouts"),
            "rollbackQuarantine": str(promotion_root / "rollback"),
            "trainerCheckpoint": str(checkpoint),
            "evaluations": str(promotion_root / "evaluations"),
            "reports": str(promotion_root / "reports"),
            "suites": str(suites),
            "policy": str(policy_path),
            "poweredConfig": str(powered),
            "standardConfig": str(standard),
            "selfplayConfig": str(selfplay),
            "gpuLeaseConfig": str(gpu_path),
            "dataWatermark": str(promotion_root / "watermarks" / "data.json"),
            "shuffleWatermark": str(
                promotion_root / "watermarks" / "shuffle.json"
            ),
            "workerAckInbox": str(promotion_root / "ipc" / "worker-acks"),
            "rolloutReportInbox": str(promotion_root / "ipc" / "rollout-reports"),
            "originalModel": str(original),
            **{name: str(path) for name, path in schedules.items()},
        }
    )
    promotion["hashes"] = {
        "controller": file_sha256(
            repository / "python" / "risk_score" / "promotion_controller.py"
        ),
        "source": _sha256_text(source_revision),
        "original": file_sha256(original),
        "policy": policy_sha256(load_policy(policy_path)),
        "poweredConfig": file_sha256(powered),
        "standardConfig": file_sha256(standard),
        **{name: file_sha256(path) for name, path in schedules.items()},
        "selfplayConfig": file_sha256(selfplay),
        "gpuLeaseConfig": file_sha256(gpu_path),
        "suiteManifest": file_sha256(manifest),
    }
    promotion_path = output / "promotion-runtime.json"
    atomic_write_json(promotion_path, promotion)
    boot_ready_path = promotion_root / "supervisor" / "boot-ready.json"
    supervisor_argv = [
        str(python),
        "-m",
        "risk_score.promotion_host",
        "supervise",
        "--runtime-config",
        str(promotion_path),
        "--state-root",
        str(promotion_root / "supervisor"),
        "--boot-ready",
        str(boot_ready_path),
        "--katago",
        str(katago),
        "--config",
        str(selfplay),
        "--trainer-spec",
        str(trainer_spec_path),
        "--trainer-checkpoint",
        str(checkpoint),
        "--consumer-policy",
        str(consumer_spec_path),
        "--consumer-state",
        str(promotion_root / "supervisor" / "consumers.json"),
        "--interval",
        "5",
    ]
    controller_argv = [
        str(python),
        "-m",
        "risk_score.promotion_controller",
        "--runtime-config",
        str(promotion_path),
        "--mode",
        "watch",
        "--automatic" if mutation_enabled else "--recommend-only",
    ]
    if mutation_enabled:
        controller_argv.extend(["--status-output", str(promotion_root / "status.json")])
    auditor_argv = [
        str(python),
        "-m",
        "risk_score.promotion_auditor",
        "--runtime-config",
        str(promotion_path),
        "--katago",
        str(katago),
        "--mode",
        "watch",
        "--interval",
        "5",
    ]
    feedback_common_argv = [
        str(python),
        "-m",
        "risk_score.promotion_feedback",
        "--runtime-config",
        str(promotion_path),
        "--run-root",
        str(root),
    ]
    feedback_argv = [
        *feedback_common_argv,
        "--mode",
        "watch",
        "--interval",
        "15",
    ]
    if mutation_enabled:
        feedback_argv.append("--strict")
    feedback_once_argv = [
        *feedback_common_argv,
        "--mode",
        "once",
        "--strict",
    ]
    gpu_reconcile_argv = [
        str(python),
        "-m",
        "risk_score.gpu_lease",
        "--config",
        str(gpu_path),
        "reconcile",
        "--apply",
    ]
    controller_reconcile_argv = [
        str(python),
        "-m",
        "risk_score.promotion_controller",
        "--runtime-config",
        str(promotion_path),
        "--mode",
        "reconcile",
        "--automatic",
        "--status-output",
        str(promotion_root / "status.json"),
    ]
    boot_ready_argv = [
        str(python),
        "-m",
        "risk_score.promotion_host",
        "boot-ready",
        "--runtime-config",
        str(promotion_path),
        "--output",
        str(boot_ready_path),
    ]
    model_probe_argv = [
        str(python),
        "-m",
        "risk_score.model_probe",
        "--katago",
        str(katago),
        "--config",
        str(analysis_config),
        "--model",
        "{model_file}",
        "--gpu-index",
        "7",
    ]
    shuffler_environment = {
        "KATAGO_DATA_WATERMARK": str(
            promotion_root / "watermarks" / "data.json"
        ),
        "KATAGO_STRICT_SHUFFLE_PROVENANCE": (
            "1" if mutation_enabled else "0"
        ),
    }
    exporter_environment = {
        "KATAGO_HARDENED_EXPORTER": str(
            repository / "python" / "risk_score" / "hardened_exporter.py"
        ),
        "KATAGO_MODEL_PROBE_COMMAND_JSON": json.dumps(
            model_probe_argv,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "KATAGO_PROMOTION_BACKPRESSURE_FILE": str(
            promotion_root / "operations" / "backpressure.json"
        ),
        "KATAGO_PROMOTION_POLICY_HASH": promotion["hashes"]["policy"],
        "KATAGO_PROMOTION_BACKPRESSURE_MAX_AGE_SECONDS": "120",
    }
    services: Dict[str, Mapping[str, Any]] = {
        "supervisor": {
            "argv": supervisor_argv,
            "description": "KataGo risk-training host supervisor",
            "environment": {},
            "restart": "always",
        },
        "controller": {
            "argv": controller_argv,
            "description": "KataGo risk-training promotion controller",
            "environment": {},
            "restart": "always",
        },
        "feedback": {
            "argv": feedback_argv,
            "description": "KataGo risk-training provenance feedback watcher",
            "environment": {},
            "restart": "always",
        },
    }
    if mutation_enabled:
        reconcile_commands = [
            feedback_once_argv,
            gpu_reconcile_argv,
        ]
        if full_autonomy:
            reconcile_commands.extend(
                [
                    _reconcile_argv(
                        executor_argv, "cluster executor"
                    ),
                    _reconcile_argv(
                        adaptive_argv, "adaptive training"
                    ),
                    _reconcile_argv(
                        suite_rotation_argv, "suite rotation"
                    ),
                ]
            )
        reconcile_commands.extend(
            [
                controller_reconcile_argv,
                boot_ready_argv,
            ]
        )
        services["auditor"] = {
            "argv": auditor_argv,
            "description": "KataGo risk-training rollout and deep-audit worker",
            "environment": {},
            "restart": "always",
        }
        services["reconcile"] = {
            "commands": reconcile_commands,
            "description": "KataGo risk-training boot reconciliation",
            "environment": {},
            "restart": "no",
            "type": "oneshot",
        }
    if full_autonomy:
        services.update(
            {
                "executor": {
                    "argv": executor_argv,
                    "description": "KataGo durable GPU cluster executor",
                    "environment": {},
                    "restart": "always",
                },
                "adaptive": {
                    "argv": adaptive_argv,
                    "description": "KataGo bounded adaptive training controller",
                    "environment": {},
                    "restart": "always",
                },
                "suite_rotation": {
                    "argv": suite_rotation_argv,
                    "description": "KataGo evaluation-suite rotation controller",
                    "environment": {},
                    "restart": "always",
                },
            }
        )
    for name, command, description, environment in (
        (
            "shuffler",
            shuffler_argv,
            "KataGo risk-training gated shuffler",
            shuffler_environment,
        ),
        (
            "exporter",
            exporter_argv,
            "KataGo risk-training hardened exporter",
            exporter_environment,
        ),
    ):
        if command:
            services[name] = {
                "argv": command,
                "description": description,
                "environment": environment,
                "restart": "always",
            }

    systemd_units: Dict[str, Mapping[str, str]] = {}
    if service_user is not None:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", service_user) is None:
            raise HostCommandError("systemd service user is invalid")
        units_dir = output / "systemd"
        unit_names = {
            "supervisor": "katago-risk-promotion-host.service",
            "controller": "katago-risk-promotion-controller.service",
            "auditor": "katago-risk-promotion-auditor.service",
            "feedback": "katago-risk-promotion-feedback.service",
            "shuffler": "katago-risk-shuffler.service",
            "exporter": "katago-risk-exporter.service",
            "reconcile": "katago-risk-boot-reconcile.service",
            **(
                AUTONOMY_SERVICE_UNIT_NAMES
                if full_autonomy
                else {}
            ),
        }
        for name, spec in services.items():
            dependencies = []
            if name in {
                "controller",
                "auditor",
                "feedback",
                "reconcile",
                "executor",
                "adaptive",
                "suite_rotation",
            }:
                dependencies.append("katago-risk-promotion-host.service")
            if mutation_enabled and name not in {"supervisor", "reconcile"}:
                dependencies.append("katago-risk-boot-reconcile.service")
            if name in {"adaptive", "suite_rotation"}:
                dependencies.append(
                    AUTONOMY_SERVICE_UNIT_NAMES["executor"]
                )
                dependencies.append(
                    "katago-risk-promotion-controller.service"
                )
            if name == "suite_rotation":
                dependencies.append(
                    "katago-risk-promotion-auditor.service"
                )
            before = (
                tuple(
                    unit_names[service_name]
                    for service_name in services
                    if service_name not in {"supervisor", "reconcile"}
                )
                if name == "reconcile"
                else ()
            )
            unit_path = units_dir / unit_names[name]
            commands = spec.get("commands")
            if name == "reconcile":
                if not isinstance(commands, list) or not commands:
                    raise HostCommandError(
                        "boot reconciliation service commands are invalid"
                    )
                primary_argv = commands[0]
                additional_argv = commands[1:]
            else:
                primary_argv = spec["argv"]
                additional_argv = ()
            _write_text_file(
                unit_path,
                _systemd_service(
                    description=str(spec["description"]),
                    argv=primary_argv,
                    service_user=service_user,
                    working_directory=repository / "python",
                    run_root=root,
                    environment=spec.get("environment", {}),
                    after=dependencies,
                    before=before,
                    requires=dependencies,
                    restart=str(spec.get("restart", "on-failure")),
                    service_type=str(spec.get("type", "simple")),
                    additional_argv=additional_argv,
                    remain_after_exit=name == "reconcile",
                ),
            )
            systemd_units[name] = {
                "path": str(unit_path),
                "sha256": file_sha256(unit_path),
            }
        generated_unit_names = tuple(unit_names[name] for name in services)
        target_path = units_dir / "katago-risk-training.target"
        _write_text_file(
            target_path,
            "\n".join(
                [
                    "[Unit]",
                    "Description=KataGo risk-training closed-loop services",
                    "Conflicts=katago-risk-supplement-v3.service "
                    "katago-risk-curation-pipeline-v3.service",
                    "Wants=" + " ".join(generated_unit_names),
                    "After=" + " ".join(generated_unit_names),
                    "",
                    "[Install]",
                    "WantedBy=multi-user.target",
                    "",
                ]
            ),
        )
        systemd_units["target"] = {
            "path": str(target_path),
            "sha256": file_sha256(target_path),
        }
    service_spec = {
        "schema_version": 3 if full_autonomy else 2,
        "contract": (
            AUTONOMY_SERVICE_SPEC_CONTRACT
            if full_autonomy
            else "risk-score-host-services-v2"
        ),
        "mutation_enabled": mutation_enabled,
        "full_autonomy": full_autonomy,
        "evaluator_process_count": evaluator_process_count,
        **(
            {
                "service_inputs": {
                    name: {
                        "path": str(path),
                        "sha256": file_sha256(path),
                    }
                    for name, path in sorted(
                        autonomy_input_paths.items()
                    )
                }
            }
            if full_autonomy
            else {}
        ),
        "service_user": service_user,
        "services": services,
        "systemd_units": systemd_units,
        "supervisor_argv": supervisor_argv,
        "controller_argv": controller_argv,
    }
    service_path = output / "promotion-services.json"
    atomic_write_json(service_path, service_spec)
    from risk_score.promotion_controller import RuntimeConfig

    RuntimeConfig.load(promotion_path)
    deployment_files = {
        "promotion_controller": repository
        / "python"
        / "risk_score"
        / "promotion_controller.py",
        "promotion_host": repository / "python" / "risk_score" / "promotion_host.py",
        "stage0_probe": repository / "python" / "risk_score" / "stage0_probe.py",
        "model_probe": repository / "python" / "risk_score" / "model_probe.py",
        "gpu_lease": repository / "python" / "risk_score" / "gpu_lease.py",
        "promotion_evaluator": repository
        / "python"
        / "risk_score"
        / "promotion_evaluator.py",
        "evaluation_runner": repository
        / "python"
        / "risk_score"
        / "evaluation_runner.py",
        "promotion_evidence": repository
        / "python"
        / "risk_score"
        / "promotion_evidence.py",
        "promotion_gate": repository
        / "python"
        / "risk_score"
        / "promotion_gate.py",
        "hardened_exporter": repository
        / "python"
        / "risk_score"
        / "hardened_exporter.py",
        "export_loop": repository
        / "python"
        / "selfplay"
        / "export_model_for_selfplay.sh",
        "policy": policy_path,
        "powered_config": powered,
        "standard_config": standard,
        "selfplay_config": selfplay,
        "suite_manifest": manifest,
        "analysis_config": repository
        / "cpp"
        / "configs"
        / "risk_score"
        / "promotion_curation_analysis.cfg",
        "katago": katago,
        "python": python,
        "trainer_spec": trainer_spec_path,
        "consumer_spec": consumer_spec_path,
        "promotion_runtime": promotion_path,
        "gpu_lease_runtime": gpu_path,
        "service_spec": service_path,
        **{
            f"autonomy:{name}": path
            for name, path in autonomy_input_paths.items()
        },
    }
    command_bindings = {
        "trainer": tuple(trainer_spec_value.get("argv", ())),
        "shuffler": shuffler_argv,
        "exporter": exporter_argv,
        "legacy_evaluator": evaluator_argv,
        "cluster_executor": executor_argv,
        "adaptive_training": adaptive_argv,
        "suite_rotation": suite_rotation_argv,
    }
    for command_name, command in command_bindings.items():
        for index, argument in enumerate(command):
            candidate = Path(argument)
            if not candidate.is_absolute() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                resolved.relative_to(repository)
            except ValueError:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
            deployment_files[f"command:{command_name}:{index}"] = resolved
    for name, unit in systemd_units.items():
        deployment_files[f"systemd:{name}"] = Path(unit["path"])
    for module_path in sorted(
        (repository / "python" / "risk_score").glob("*.py")
    ):
        deployment_files[f"module:{module_path.name}"] = module_path
    deployment = {
        "schema_version": 1,
        "contract": "risk-score-live-runtime-deployment-v1",
        "source_revision": actual_revision,
        "source_sha256": _sha256_text(actual_revision),
        "files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in sorted(deployment_files.items())
        },
    }
    deployment["manifest_sha256"] = policy_sha256(deployment)
    deployment_path = output / "deployment-manifest.json"
    atomic_write_json(deployment_path, deployment)
    return {
        "promotion_runtime": str(promotion_path),
        "promotion_runtime_sha256": file_sha256(promotion_path),
        "gpu_lease_runtime": str(gpu_path),
        "gpu_lease_runtime_sha256": file_sha256(gpu_path),
        "deployment_manifest": str(deployment_path),
        "deployment_manifest_sha256": file_sha256(deployment_path),
        "service_spec": str(service_path),
        "service_spec_sha256": file_sha256(service_path),
        "systemd_units": systemd_units,
        "mutation_enabled": mutation_enabled,
        "full_autonomy": full_autonomy,
        "evaluator_process_count": evaluator_process_count,
    }


def verify_deployment_manifest(path: Path) -> Mapping[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HostCommandError("deployment manifest is missing")
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("contract") != "risk-score-live-runtime-deployment-v1"
    ):
        raise HostCommandError("deployment manifest contract is unsupported")
    payload = dict(value)
    expected = payload.pop("manifest_sha256", None)
    if expected != policy_sha256(payload):
        raise HostCommandError("deployment manifest self-hash is invalid")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise HostCommandError("deployment manifest has no files")
    for name, artifact in files.items():
        if not isinstance(artifact, dict):
            raise HostCommandError(f"deployment artifact {name} is malformed")
        artifact_path = Path(artifact.get("path", ""))
        if (
            not artifact_path.is_absolute()
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise HostCommandError(f"deployment artifact {name} changed")
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--katago-binary", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--trainer-spec", required=True, type=Path)
    parser.add_argument("--consumer-spec", required=True, type=Path)
    parser.add_argument("--original-model", required=True, type=Path)
    parser.add_argument("--trainer-checkpoint", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mutation-enabled", action="store_true")
    parser.add_argument("--service-user")
    parser.add_argument("--shuffler-command-json")
    parser.add_argument("--exporter-command-json")
    parser.add_argument("--evaluator-command-json")
    parser.add_argument("--full-autonomy", action="store_true")
    parser.add_argument("--cluster-executor-command-json")
    parser.add_argument("--adaptive-training-command-json")
    parser.add_argument("--suite-rotation-command-json")
    parser.add_argument("--autonomy-policy", type=Path)
    parser.add_argument("--cluster-executor-spec", type=Path)
    parser.add_argument("--adaptive-training-spec", type=Path)
    parser.add_argument("--suite-registry-spec", type=Path)
    parser.add_argument(
        "--evaluator-process-count",
        type=int,
        choices=(4, 8, 16),
        default=8,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        shuffler_command = _parse_command_json(args.shuffler_command_json, "shuffler")
        exporter_command = _parse_command_json(args.exporter_command_json, "exporter")
        evaluator_command = _parse_command_json(
            args.evaluator_command_json, "evaluator"
        )
        cluster_executor_command = _parse_command_json(
            args.cluster_executor_command_json, "cluster executor"
        )
        adaptive_training_command = _parse_command_json(
            args.adaptive_training_command_json, "adaptive training"
        )
        suite_rotation_command = _parse_command_json(
            args.suite_rotation_command_json, "suite rotation"
        )
        result = build_live_runtime(
            repo=args.repo,
            run_root=args.run_root,
            suite_dir=args.suite_dir,
            katago_binary=args.katago_binary,
            python_executable=args.python_executable,
            trainer_spec=args.trainer_spec,
            consumer_spec=args.consumer_spec,
            original_model=args.original_model,
            trainer_checkpoint=args.trainer_checkpoint,
            gpu_uuid=args.gpu_uuid,
            actor=args.actor,
            source_revision=args.source_revision,
            output_dir=args.output_dir,
            mutation_enabled=args.mutation_enabled,
            service_user=args.service_user,
            shuffler_command=shuffler_command,
            exporter_command=exporter_command,
            evaluator_command=evaluator_command,
            full_autonomy=args.full_autonomy,
            cluster_executor_command=cluster_executor_command,
            adaptive_training_command=adaptive_training_command,
            suite_rotation_command=suite_rotation_command,
            autonomy_policy=args.autonomy_policy,
            cluster_executor_spec=args.cluster_executor_spec,
            adaptive_training_spec=args.adaptive_training_spec,
            suite_registry_spec=args.suite_registry_spec,
            evaluator_process_count=args.evaluator_process_count,
        )
    except (HostCommandError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
