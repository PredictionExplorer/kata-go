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
    shuffler_argv = _service_argv(shuffler_command, "shuffler")
    exporter_argv = _service_argv(exporter_command, "exporter")
    evaluator_argv = _service_argv(evaluator_command, "evaluator")
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
        services["auditor"] = {
            "argv": auditor_argv,
            "description": "KataGo risk-training rollout and deep-audit worker",
            "environment": {},
            "restart": "always",
        }
        services["reconcile"] = {
            "commands": [
                feedback_once_argv,
                gpu_reconcile_argv,
                controller_reconcile_argv,
                boot_ready_argv,
            ],
            "description": "KataGo risk-training boot reconciliation",
            "environment": {},
            "restart": "no",
            "type": "oneshot",
        }
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
        }
        for name, spec in services.items():
            dependencies = []
            if name in {"controller", "auditor", "feedback", "reconcile"}:
                dependencies.append("katago-risk-promotion-host.service")
            if mutation_enabled and name not in {"supervisor", "reconcile"}:
                dependencies.append("katago-risk-boot-reconcile.service")
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
        "schema_version": 2,
        "contract": "risk-score-host-services-v2",
        "mutation_enabled": mutation_enabled,
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
    }
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        shuffler_command = _parse_command_json(args.shuffler_command_json, "shuffler")
        exporter_command = _parse_command_json(args.exporter_command_json, "exporter")
        evaluator_command = _parse_command_json(
            args.evaluator_command_json, "evaluator"
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
        )
    except (HostCommandError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
