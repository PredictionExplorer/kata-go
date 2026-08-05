#!/usr/bin/env python3
"""Materialize hash-pinned live promotion runtime JSON from checked-in examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from risk_score.paired_stats import canonical_sha256 as policy_sha256
from risk_score.paired_stats import load_policy
from risk_score.position_samples import canonical_json, file_sha256
from risk_score.promotion_host import HostCommandError, atomic_write_json


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
    original = _required_file(original_model, "original model")
    checkpoint = _required_file(trainer_checkpoint, "trainer checkpoint")
    katago = _required_file(katago_binary, "KataGo binary")
    python = _required_file(Path(python_executable).resolve(), "Python executable")
    trainer_spec_path = _required_file(trainer_spec, "trainer launch spec")
    consumer_spec_path = _required_file(consumer_spec, "consumer stop spec")
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
    gpu["evaluator"]["launchCommand"] = [
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
    service_spec = {
        "schema_version": 1,
        "contract": "risk-score-host-services-v1",
        "supervisor_argv": [
            str(python),
            "-m",
            "risk_score.promotion_host",
            "supervise",
            "--runtime-config",
            str(promotion_path),
            "--state-root",
            str(promotion_root / "supervisor"),
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
        ],
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
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
        )
    except (HostCommandError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
