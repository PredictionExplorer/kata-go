import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from risk_score import (
    autonomy_bootstrap,
    autonomy_bootstrap_spec,
    suite_rotation_service,
)
from risk_score import autonomy_provisioner as provisioner
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256

DETERMINISTIC_CONFIG = """\
forDeterministicTesting = true
numAnalysisThreads = 1
numSearchThreadsPerAnalysisThread = 1
nnRandomize = false
rootNoiseEnabled = false
rootNumSymmetriesToSample = 1
useUncertainty = false
cpuctUtilityStdevScale = 0
reportAnalysisWinratesAs = sidetomove
"""


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_self_hashed(path, contract, **fields):
    value = {"schema_version": 1, "contract": contract, **fields}
    value["spec_sha256"] = canonical_sha256(value)
    data = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data
    else:
        path.write_bytes(data)
    return value


def binding(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def git(repo, *arguments):
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def initialize_repository(repo):
    repo.mkdir()
    git(repo, "init", "-q")
    package = repo / "python" / "risk_score"
    package.mkdir(parents=True)
    source_root = Path(provisioner.__file__).parents[2]
    source_package = Path(provisioner.__file__).parent
    for source in source_package.glob("*.py"):
        (package / source.name).write_bytes(source.read_bytes())
    runtime_modules = {
        "risk_score.shuffler",
        "risk_score.exporter",
        "risk_score.evaluator",
        "risk_score.autonomy_provisioner",
        "risk_score.cluster_executor",
        "risk_score.adaptive_training",
        "risk_score.suite_rotation_service",
    }
    for module in sorted(autonomy_bootstrap_spec._CONTROL_MODULES | runtime_modules):
        relative = module.removeprefix("risk_score.").replace(".", "/") + ".py"
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'MODULE = "{module}"\n', encoding="utf-8")
    (package / "promotion_policy_v3.json").write_bytes(
        Path(provisioner.__file__).with_name("promotion_policy_v3.json").read_bytes()
    )
    (package / "promotion_controller.py").write_bytes(
        Path(provisioner.__file__).with_name("promotion_controller.py").read_bytes()
    )
    for name in (
        "promotion_runtime.example.json",
        "gpu_lease_runtime.example.json",
    ):
        (package / name).write_bytes(
            Path(provisioner.__file__).with_name(name).read_bytes()
        )
    exporter = source_root / "python" / "selfplay" / "export_model_for_selfplay.sh"
    exported = repo / "python" / "selfplay" / exporter.name
    exported.parent.mkdir(parents=True, exist_ok=True)
    exported.write_bytes(exporter.read_bytes())
    for name in (
        "promotion_powered_match.cfg",
        "promotion_standard_match.cfg",
        "promotion_curation_analysis.cfg",
        "promotion_selfplay_worker_19x19.cfg",
    ):
        source = source_root / "cpp" / "configs" / "risk_score" / name
        destination = repo / "cpp" / "configs" / "risk_score" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    git(repo, "add", ".")
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Provisioner Test",
            "-c",
            "user.email=provisioner@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    return git(repo, "rev-parse", "HEAD")


def publish_readiness(fixture, *, marker="initial"):
    suite = {
        "schemaVersion": 3,
        "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
        "machineReviewOnly": True,
        "marker": marker,
        "banks": [],
        "cells": [],
    }
    suite["manifestPayloadSha256"] = canonical_sha256(suite)
    write_canonical(fixture["suite_manifest"], suite)
    schedules = fixture["suite_manifest"].parent / "schedules"
    schedules.mkdir(parents=True, exist_ok=True)
    for name in (
        "discovery.jsonl",
        "confirmation.jsonl",
        "audit.jsonl",
        "lead-40-confirmation.jsonl",
        "lead-80-confirmation.jsonl",
    ):
        (schedules / name).write_text(
            canonical_json({"schedule": name}) + "\n", encoding="utf-8"
        )
    status = {
        "schema_version": 1,
        "contract": "risk-score-curation-pipeline-status-v1",
        "state": "complete",
        "error": None,
        "artifacts": {
            "reviewed_bank": {"complete": True},
            "suite": {
                "complete": True,
                "path": str(fixture["suite_manifest"].parent),
            },
        },
    }
    status["status_sha256"] = canonical_sha256(status)
    write_canonical(fixture["curation_status"], status)


class FakePublishers:
    def __init__(self, fixture, *, mutate_candidates=False):
        self.fixture = fixture
        self.mutate_candidates = mutate_candidates
        self.mutated = False
        self.calls = []

    def candidate_inventory(self, *, inbox, output):
        names = sorted(path.name for path in inbox.iterdir())
        value = {
            "schema_version": 1,
            "contract": "risk-score-live-candidate-inventory-v1",
            "inbox": str(inbox.resolve()),
            "candidate_count": len(names),
            "ignored": [],
            "candidates": [{"name": name} for name in names],
        }
        value["inventory_sha256"] = canonical_sha256(value)
        write_canonical(output, value)
        return value

    def publish_model_probe_config(self, *, source, destination, **_kwargs):
        self.calls.append("model-probe")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes()
        else:
            destination.write_bytes(source.read_bytes())
        return binding(destination)

    def publish_cluster_executor_spec(self, *, destination, **_kwargs):
        self.calls.append("cluster")
        return write_self_hashed(
            destination,
            "risk-score-cluster-executor-spec-v1",
            name="cluster",
        )

    def publish_registry_spec(self, *, destination, **_kwargs):
        self.calls.append("registry-spec")
        return write_self_hashed(
            destination,
            "risk-score-evaluation-suite-registry-spec-v1",
            name="registry",
        )

    def bootstrap_registry(self, *, spec, **_kwargs):
        self.calls.append("registry-bootstrap")
        root = Path(spec.publisher_config["suite_rotation"]["registry_root"])
        suite_id = file_sha256(spec.suite_manifest.path)
        manifest = root / "suites" / suite_id / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        if manifest.exists():
            assert manifest.read_bytes() == spec.suite_manifest.path.read_bytes()
        else:
            manifest.write_bytes(spec.suite_manifest.path.read_bytes())
        active = {
            "schema_version": 1,
            "contract": "risk-score-active-evaluation-suite-v1",
            "suite_id": suite_id,
            "manifest_path": str(manifest),
        }
        write_canonical(root / "active-suite.json", active)
        return {
            "event_sha256": "a" * 64,
            "active_suite": binding(root / "active-suite.json"),
            "manifest": binding(manifest),
            "suite_id": active["suite_id"],
            "champion_sha256": spec.immutable_inputs["suite_champion_model"].sha256,
            "generation_id": spec.initial_generation_id,
        }

    def verify_target_inactive(self, **_kwargs):
        self.calls.append("target-inactive")
        return {
            "unit": provisioner.PRODUCTION_TARGET_UNIT_NAME,
            "state": "inactive",
            "inactive": True,
        }

    def publish_adaptive_spec(self, *, destination, **_kwargs):
        self.calls.append("adaptive")
        return write_self_hashed(
            destination,
            "risk-score-adaptive-training-service-spec-v1",
            name="adaptive",
        )

    def publish_suite_service_spec(self, *, destination, **_kwargs):
        self.calls.append("suite-service")
        return write_self_hashed(
            destination,
            "risk-score-suite-rotation-service-spec-v1",
            name="suite-service",
        )

    def build_shadow_runtime(self, *, output_dir, **_kwargs):
        self.calls.append("shadow-runtime")
        promotion = output_dir / "promotion-runtime.json"
        gpu = output_dir / "gpu-lease-runtime.json"
        deployment = output_dir / "deployment-manifest.json"
        service = output_dir / "promotion-services.json"
        write_canonical(promotion, {"mutationEnabled": False})
        write_canonical(gpu, {"mutationEnabled": False})
        write_canonical(deployment, {"shadow": True})
        write_canonical(service, {"shadow": True})
        return {
            "promotion_runtime": str(promotion),
            "promotion_runtime_sha256": file_sha256(promotion),
            "gpu_lease_runtime": str(gpu),
            "gpu_lease_runtime_sha256": file_sha256(gpu),
            "deployment_manifest": str(deployment),
            "deployment_manifest_sha256": file_sha256(deployment),
            "service_spec": str(service),
            "service_spec_sha256": file_sha256(service),
            "systemd_units": {},
            "mutation_enabled": False,
            "full_autonomy": False,
            "evaluator_process_count": 8,
        }

    def initialize_champion(self, *, spec, receipt_path, **_kwargs):
        self.calls.append("champion")
        champion = spec.run_root / "promotion" / "champion.json"
        value = {
            "champion_hash": spec.immutable_inputs["suite_champion_model"].sha256,
            "generation_id": spec.initial_generation_id,
            "bootstrap": True,
        }
        write_canonical(champion, value)
        receipt = {
            "schema_version": 1,
            "contract": "risk-score-autonomy-original-champion-projection-v1",
            "champion": {
                **binding(champion),
                "path": str(champion),
                "champion_hash": value["champion_hash"],
                "generation_id": value["generation_id"],
            },
            "candidate_admission": False,
            "training_started": False,
            "export_started": False,
            "mutation_activated": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        write_canonical(receipt_path, receipt)
        return receipt

    def publish_promotion_drill_spec(self, *, spec, destination, **_kwargs):
        self.calls.append("promotion-drill")
        return write_self_hashed(
            destination,
            "risk-score-autonomy-promotion-drill-spec-v1",
            evidence_root=str(spec.bootstrap_state_root / "gate-evidence"),
        )

    def publish_topology_specs(
        self, *, spec, workload_spec_path, benchmark_spec_path, **_kwargs
    ):
        self.calls.append("topology")
        write_self_hashed(
            workload_spec_path,
            "risk-score-evaluator-topology-benchmark-workload-spec-v1",
            name="workload",
        )
        return write_self_hashed(
            benchmark_spec_path,
            "risk-score-evaluator-topology-benchmark-spec-v1",
            evidence_output=str(
                spec.bootstrap_state_root
                / "gate-evidence"
                / "evaluator-topology-benchmark.json"
            ),
        )

    def publish_lease_specs(self, *, spec, probe_spec_path, drill_spec_path, **_kwargs):
        self.calls.append("lease")
        write_self_hashed(
            probe_spec_path,
            "risk-score-autonomy-lease-probe-spec-v1",
            name="probe",
        )
        return write_self_hashed(
            drill_spec_path,
            "risk-score-autonomy-lease-drill-spec-v1",
            trainer_source={"kind": "launch"},
            evidence_output=str(
                spec.bootstrap_state_root
                / "gate-evidence"
                / "trainer-evaluator-lease-drill.json"
            ),
        )

    def build_lease_runtime(self, *, destination, **_kwargs):
        self.calls.append("lease-runtime")
        write_canonical(
            destination,
            {
                "schemaVersion": 1,
                "mutationEnabled": True,
                "evaluator": {"processCount": 1},
            },
        )
        return {
            **binding(destination),
            "mutation_enabled": True,
            "evaluator_process_count": 1,
            "installed": False,
        }

    def publish_backpressure(self, *, destination, **_kwargs):
        self.calls.append("backpressure")
        write_canonical(
            destination,
            {
                "schema_version": 1,
                "policy_hash": "b" * 64,
                "controller_hash": "0" * 64,
                "allowExport": False,
                "allowEvaluation": False,
                "exportPaused": True,
                "evaluationPaused": True,
                "exportBacklogDepth": 0,
                "evaluationBacklogDepth": 0,
                "maximumActiveEvaluatorEntries": 3,
                "importantQueueWarningDepth": 4,
                "reasons": ["bootstrap-pre-controller"],
            },
        )
        if self.mutate_candidates and not self.mutated:
            (self.fixture["candidate_inbox"] / "drifted").write_text(
                "candidate\n", encoding="utf-8"
            )
            self.mutated = True
        return {"allowExport": False}

    @staticmethod
    def materialize_bootstrap(
        publisher_spec_path,
        *,
        expected_spec_sha256,
        command_runner,
        **_kwargs,
    ):
        return autonomy_bootstrap_spec.materialize_bootstrap(
            publisher_spec_path,
            expected_spec_sha256=expected_spec_sha256,
            command_runner=command_runner,
        )

    def adapters(self):
        return provisioner.ProvisionerAdapters(
            candidate_inventory=self.candidate_inventory,
            publish_model_probe_config=self.publish_model_probe_config,
            publish_cluster_executor_spec=self.publish_cluster_executor_spec,
            publish_registry_spec=self.publish_registry_spec,
            verify_target_inactive=self.verify_target_inactive,
            bootstrap_registry=self.bootstrap_registry,
            publish_adaptive_spec=self.publish_adaptive_spec,
            publish_suite_service_spec=self.publish_suite_service_spec,
            build_shadow_runtime=self.build_shadow_runtime,
            initialize_champion=self.initialize_champion,
            publish_promotion_drill_spec=self.publish_promotion_drill_spec,
            publish_topology_specs=self.publish_topology_specs,
            build_lease_runtime=self.build_lease_runtime,
            publish_lease_specs=self.publish_lease_specs,
            publish_backpressure=self.publish_backpressure,
            materialize_bootstrap=self.materialize_bootstrap,
        )


class FakeSystemctl:
    def __init__(self, fail=None):
        self.fail = tuple(fail) if fail is not None else None
        self.commands = []
        self.enabled = {provisioner.LEGACY_PATH_UNIT_NAME}
        self.active = {provisioner.LEGACY_PATH_UNIT_NAME}

    def __call__(self, argv, **kwargs):
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "shell": False,
        }
        command = tuple(argv)
        self.commands.append(command)
        if command[1] == "is-enabled":
            enabled = command[-1] in self.enabled
            return subprocess.CompletedProcess(
                argv,
                0 if enabled else 1,
                stdout="enabled\n" if enabled else "disabled\n",
                stderr="",
            )
        if command[1] == "is-active":
            active = command[-1] in self.active
            return subprocess.CompletedProcess(
                argv,
                0 if active else 3,
                stdout="active\n" if active else "inactive\n",
                stderr="",
            )
        if command != self.fail:
            unit = command[-1]
            if command[1] == "disable":
                self.enabled.discard(unit)
                if "--now" in command:
                    self.active.discard(unit)
            elif command[1] == "enable":
                self.enabled.add(unit)
                if "--now" in command:
                    self.active.add(unit)
            elif command[1] == "stop":
                self.active.discard(unit)
            elif command[1] == "start":
                self.active.add(unit)
        return subprocess.CompletedProcess(
            argv,
            1 if command == self.fail else 0,
            stdout="",
            stderr="injected failure" if command == self.fail else "",
        )


def make_fixture(tmp_path, *, candidate_count=2):
    repo = tmp_path / "deployed"
    revision = initialize_repository(repo)
    run_root = tmp_path / "run"
    candidate_inbox = run_root / "modelstobetested"
    suite_dir = run_root / "evaluation" / "promotion-suites-v3"
    curation_dir = run_root / "evaluation" / "curation"
    scheduler = run_root / "promotion" / "scheduler"
    activation = tmp_path / "systemd"
    for directory in (
        run_root,
        candidate_inbox,
        suite_dir,
        curation_dir,
        scheduler,
        activation,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(candidate_count):
        candidate = candidate_inbox / f"candidate-s{index + 1}-d{index + 1}"
        candidate.mkdir()
        (candidate / "model.bin.gz").write_bytes(f"model-{index}\n".encode())
        (candidate / "model.ckpt").write_bytes(f"checkpoint-{index}\n".encode())

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    executable = Path(sys.executable).resolve()
    files = {}
    for name, data in {
        "katago_binary": b"katago\n",
        "trainer_spec": b'{"trainer":true}\n',
        "consumer_spec": b'{"consumer":true}\n',
        "original_model": b"original-model\n",
        "suite_champion_model": b"suite-champion\n",
        "trainer_checkpoint": b"checkpoint\n",
        "promotion_policy": (
            Path(provisioner.__file__)
            .with_name("promotion_policy_v3.json")
            .read_bytes()
        ),
        "autonomy_policy": b'{"policy":"autonomy"}\n',
        "model_probe_config_source": DETERMINISTIC_CONFIG.encode("utf-8"),
    }.items():
        path = inputs / name
        path.write_bytes(data)
        files[name] = path
    policy = json.loads(files["promotion_policy"].read_text(encoding="utf-8"))
    stages = policy["evaluation_stages"]
    stages["stage_1_cheap_paired_screen"]["ordinary_color_pairs"] = 1
    stages["stage_2_finalist_selection"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    for look in stages["stage_3_promotion_confirmation"]["looks"]:
        look.update(
            {
                "powered_ordinary_color_pairs_per_matchup": 1,
                "standard_ordinary_color_pairs": 1,
                "lead_40_color_pairs": 1,
                "lead_80_color_pairs": 1,
                "minimum_independent_position_clusters": {
                    "powered_candidate_vs_champion": 1,
                    "powered_candidate_vs_original": 1,
                    "standard_candidate_vs_original": 1,
                    "lead_40": 1,
                    "lead_80": 1,
                },
            }
        )
    stages["deep_audit"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    write_canonical(files["promotion_policy"], policy)
    files["katago_binary"].chmod(0o755)
    topology_queries = inputs / "topology-queries.jsonl"
    topology_queries.write_text(
        "".join(
            canonical_json(
                {
                    "id": f"query-{index:02d}",
                    "moves": [],
                    "initialStones": [],
                    "initialPlayer": "B",
                    "rules": "tromp-taylor",
                    "komi": 7.5,
                    "boardXSize": 19,
                    "boardYSize": 19,
                    "includePolicy": True,
                    "maxVisits": 4,
                    "overrideSettings": {
                        "useScoreMaximizingUtility": False,
                        "scorePower": 1,
                        "scoreScale": 1,
                        "winWeight": 1,
                        "rootNoiseEnabled": False,
                        "rootNumSymmetriesToSample": 1,
                    },
                }
            )
            + "\n"
            for index in range(16)
        ),
        encoding="utf-8",
    )
    lease_query = inputs / "lease-query.jsonl"
    lease_query.write_text(
        canonical_json(
            {
                "id": "lease-probe",
                "maxVisits": 4,
                "boardXSize": 19,
                "boardYSize": 19,
                "initialStones": [],
                "initialPlayer": "B",
                "moves": [],
                "rules": "tromp-taylor",
                "komi": 7.5,
                "overrideSettings": {
                    "rootNoiseEnabled": False,
                    "rootNumSymmetriesToSample": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    nvidia_smi = inputs / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)

    deployment = {
        "schema_version": 1,
        "contract": "risk-score-live-runtime-deployment-v1",
        "source_revision": revision,
        "source_sha256": hashlib.sha256(revision.encode("utf-8")).hexdigest(),
        "files": {
            "module": binding(repo / "python" / "risk_score" / "autonomy_bootstrap.py")
        },
    }
    deployment["manifest_sha256"] = canonical_sha256(deployment)
    deployment_path = inputs / "deployment-manifest.json"
    write_canonical(deployment_path, deployment)
    files["deployment_manifest"] = deployment_path

    immutable_inputs = {
        "python_executable": executable,
        **files,
    }
    models = [files["original_model"], files["suite_champion_model"]]
    extra_inputs = [topology_queries, lease_query, nvidia_smi]
    python = str(executable)
    guardian = [
        python,
        "-m",
        "risk_score.gpu_lease_worker",
        "--expected-spec-sha256",
        "1" * 64,
        "--claim-id",
        "{claim_id}",
        "--work-id",
        "{work_id}",
        "--receipt",
        "{guardian_receipt}",
        "--",
    ]
    suite_root = run_root / "promotion" / "suite-service"
    suite_results = {
        name: {
            "path": str(
                suite_root
                / (
                    "status.json"
                    if name == "status"
                    else (
                        "{request_id}-{role}.json"
                        if name == "continuity_evidence"
                        else f"{{request_id}}-{name}.json"
                    )
                )
            ),
            "contract": suite_rotation_service._RESULT_CONTRACTS[name],
        }
        for name in suite_rotation_service._RESULT_NAMES
    }
    publisher_config = {
        "scheduler_directory": str(scheduler),
        "cluster_executor": {
            "state_directory": str(run_root / "promotion" / "executor-state"),
            "owner_id": "production-executor",
            "gpu_ids": [str(index) for index in range(8)],
            "poll_interval_seconds": 1,
            "heartbeat_interval_seconds": 2,
            "stale_after_seconds": 10,
            "retry_budget": 2,
            "backoff_initial_seconds": 1,
            "backoff_max_seconds": 30,
            "lease_proof_command": None,
            "lease_proof_timeout_seconds": 10,
            "guardian_argv_prefix": guardian,
        },
        "adaptive_training": {
            "root": str(run_root / "promotion" / "adaptive"),
            "observation_path": str(
                run_root / "promotion" / "adaptive-observation.json"
            ),
            "trial_command_argv_template": [
                python,
                "{trial_manifest}",
                "{trial_result}",
                "{work_id}",
            ],
            "poll_interval_seconds": 5,
        },
        "suite_rotation": {
            "registry_root": str(run_root / "promotion" / "suite-registry"),
            "root": str(suite_root),
            "materializer_argv_template": [
                python,
                "{request_id}",
                "{rotation_request}",
                "{supplement_request}",
                "{pipeline_request}",
                "{supplement_spec}",
                "{pipeline_spec}",
            ],
            "curation_argv_template": [
                python,
                "{request_id}",
                "{supplement_spec}",
                "{pipeline_spec}",
                "{suite_manifest}",
            ],
            "continuity_argv_template": [
                python,
                "{request_id}",
                "{role}",
                "{model_path}",
                "{model_sha256}",
                "{candidate_suite_id}",
                "{continuity_evidence}",
            ],
            "results": suite_results,
            "poll_interval_seconds": 5,
        },
        "topology_benchmark": {
            "publisher_options": {
                "queries": str(topology_queries),
                "nvidia_smi": str(nvidia_smi),
                "row_count": 16,
                "max_visits": 4,
            },
            "timeout_seconds": 60,
        },
        "lease_drill": {
            "publisher_options": {
                "query": str(lease_query),
                "nvidia_smi": str(nvidia_smi),
            },
            "probe_timeout_seconds": 30,
            "probe_minimum_completed_work": 1,
        },
        "promotion_drill": {
            "disposable_root": str(run_root / "promotion" / "bootstrap-disposable"),
            "command_timeout_seconds": 15,
            "max_evaluator_rows": 100,
            "max_worker_games": 100,
            "max_replay_attempts": 16,
        },
        "shadow_runtime": {"evaluator_process_count": 8},
    }
    state_root = run_root / "promotion" / "autonomy-provisioner"
    bootstrap_state = run_root / "promotion" / "autonomy-bootstrap"
    spec_path = tmp_path / "provisioner-spec.json"
    loaded = provisioner.publish_provisioner_spec(
        spec_path,
        repository=repo,
        source_revision=revision,
        run_root=run_root,
        state_root=state_root,
        bootstrap_state_root=bootstrap_state,
        curation_status=curation_dir / "status.json",
        suite_manifest=suite_dir / "manifest.json",
        candidate_inbox=candidate_inbox,
        activation_destination=activation,
        immutable_inputs=immutable_inputs,
        models=models,
        extra_inputs=extra_inputs,
        runtime_commands={
            "shuffler": [python, "-m", "risk_score.shuffler"],
            "exporter": [python, "-m", "risk_score.exporter"],
            "evaluator": [python, "-m", "risk_score.evaluator"],
        },
        publisher_config=publisher_config,
        gpu_index=7,
        gpu_uuid="GPU-production-test-7",
        actor="autonomy-provisioner",
        service_user="katago",
        initial_generation_id="bootstrap-original",
        created_at_utc="2026-08-13T00:00:00Z",
    )
    return {
        "repo": repo,
        "revision": revision,
        "run_root": run_root,
        "candidate_inbox": candidate_inbox,
        "suite_manifest": suite_dir / "manifest.json",
        "curation_status": curation_dir / "status.json",
        "activation": activation,
        "state_root": state_root,
        "bootstrap_state": bootstrap_state,
        "spec_path": spec_path,
        "spec": loaded,
    }


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_missing_suite_publishes_wait_without_downstream_artifacts(tmp_path):
    fixture = make_fixture(tmp_path)
    fake = FakePublishers(fixture)

    result = provisioner.materialize_provisioner(
        fixture["spec_path"], adapters=fake.adapters()
    )

    assert result["state"] == "WAIT"
    assert not fixture["spec"].outputs.artifacts_root.exists()
    assert not fixture["spec"].outputs.bootstrap_spec.exists()
    service = fixture["spec"].outputs.prepare_service_unit.read_text(encoding="utf-8")
    path = fixture["spec"].outputs.prepare_path_unit.read_text(encoding="utf-8")
    assert "--expected-spec-sha256" in service
    assert "--apply" in service
    assert "PartOf=" not in service
    assert f"Unit={provisioner.PREPARE_SERVICE_UNIT_NAME}" in path
    assert "katago-risk-training.target" not in path
    assert fake.calls == []


def test_cli_plan_materialize_wait_and_status(tmp_path, capsys):
    fixture = make_fixture(tmp_path)
    expected = file_sha256(fixture["spec_path"])

    assert (
        provisioner.main(
            [
                "plan",
                "--spec",
                str(fixture["spec_path"]),
                "--expected-spec-sha256",
                expected,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["contract"] == provisioner.PLAN_CONTRACT

    assert (
        provisioner.main(
            [
                "materialize",
                "--spec",
                str(fixture["spec_path"]),
                "--expected-spec-sha256",
                expected,
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["state"] == "WAIT"

    assert provisioner.main(["status", "--spec", str(fixture["spec_path"])]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "WAIT"


def test_success_builds_exact_bootstrap_and_replays_idempotently(tmp_path):
    fixture = make_fixture(tmp_path, candidate_count=2)
    publish_readiness(fixture)
    fake = FakePublishers(fixture)

    first = provisioner.materialize_provisioner(
        fixture["spec_path"], adapters=fake.adapters()
    )
    loaded = autonomy_bootstrap.BootstrapSpec.load(
        fixture["spec"].outputs.bootstrap_spec
    )

    assert first["state"] == "MATERIALIZED"
    assert [gate.gate_id for gate in loaded.gates] == list(
        autonomy_bootstrap.GATE_ORDER
    )
    gates = {gate.gate_id: gate for gate in loaded.gates}
    assert gates["evaluator-topology-benchmark"].requirements["choices"] == [
        4,
        8,
        16,
    ]
    backlog = gates["backlog-bound"]
    assert backlog.requirements == {
        "maximum_candidates": 2,
        "maximum_active_queue": 3,
    }
    assert flag(backlog.argv, "--maximum-candidates") == "2"
    assert flag(backlog.argv, "--maximum-active-queue") == "3"
    assert [tuple(gate.argv[1:3]) for gate in loaded.gates] == [
        ("-m", autonomy_bootstrap_spec._EXPECTED_GATE_MODULES[gate_id])
        for gate_id in autonomy_bootstrap.GATE_ORDER
    ]
    gate_one = gates["curation-suite-readiness"]
    registry_path = (
        fixture["state_root"]
        / "artifacts"
        / fixture["spec"].identity
        / "suite-registry-spec.json"
    )
    assert flag(gate_one.argv, "--suite-registry-spec") == str(registry_path)
    assert (
        json.loads(registry_path.read_text(encoding="utf-8"))["contract"]
        == "risk-score-evaluation-suite-registry-spec-v1"
    )
    rotation_path = registry_path.with_name("suite-rotation-service-spec.json")
    assert flag(loaded.runtime.argv, "--suite-rotation-spec") == str(rotation_path)
    assert "--suite-registry-spec" not in loaded.runtime.argv
    suite_runtime_command = json.loads(
        flag(loaded.runtime.argv, "--suite-rotation-command-json")
    )
    assert str(rotation_path) in suite_runtime_command
    assert str(registry_path) not in suite_runtime_command
    assert (
        json.loads(rotation_path.read_text(encoding="utf-8"))["contract"]
        == "risk-score-suite-rotation-service-spec-v1"
    )
    publisher_value = json.loads(
        (
            fixture["state_root"]
            / "artifacts"
            / fixture["spec"].identity
            / "bootstrap-publisher-spec.json"
        ).read_text(encoding="utf-8")
    )
    assert publisher_value["schema_version"] == 2
    assert (
        publisher_value["contract"] == "risk-score-autonomy-bootstrap-publisher-spec-v2"
    )
    lease = json.loads(
        (
            fixture["state_root"]
            / "artifacts"
            / fixture["spec"].identity
            / "autonomy-lease-drill-spec.json"
        ).read_text(encoding="utf-8")
    )
    assert lease["trainer_source"] == {"kind": "launch"}
    assert "--mutation-enabled" in loaded.runtime.argv
    assert "--full-autonomy" in loaded.runtime.argv
    reported = provisioner.provisioner_status(
        fixture["spec_path"], adapters=FakePublishers(fixture).adapters()
    )
    assert reported["state"] == "MATERIALIZED"

    immutable = [
        fixture["spec"].outputs.bootstrap_spec,
        fixture["spec"].outputs.bootstrap_service_unit,
        fixture["spec"].outputs.bootstrap_path_unit,
    ]
    before = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in immutable}
    second_fake = FakePublishers(fixture)
    second = provisioner.materialize_provisioner(
        fixture["spec_path"], adapters=second_fake.adapters()
    )
    assert second == first
    assert second_fake.calls == []
    assert {
        path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in immutable
    } == before


def test_default_workload_and_lease_adapters_publish_real_specs(tmp_path):
    from risk_score.autonomy_lease_drill import load_drill_spec
    from risk_score.autonomy_lease_probe import (
        DEFAULT_CLEANUP_MARGIN_SECONDS,
        DEFAULT_POLL_INTERVAL_SECONDS,
        OUTER_DRILL_TIMEOUT_RESERVE_SECONDS,
        load_probe_spec,
    )
    from risk_score.evaluator_benchmark_workload import load_workload_spec
    from risk_score.evaluator_topology_benchmark import load_benchmark_spec

    fixture = make_fixture(tmp_path)
    spec = fixture["spec"]
    paths = provisioner._generation_paths(spec)
    provisioner._ensure_directory(paths.root)
    provisioner._default_publish_model_probe_config(
        source=spec.immutable_inputs["model_probe_config_source"].path,
        destination=paths.model_probe_config,
    )
    topology = provisioner._default_publish_topology_specs(
        spec=spec,
        workload_spec_path=paths.topology_workload_spec,
        benchmark_spec_path=paths.topology_benchmark_spec,
        model_probe_config=paths.model_probe_config,
    )
    gpu_runtime = paths.shadow_runtime_root / "gpu-lease-runtime.json"
    control = tmp_path / "shadow-control"
    write_canonical(
        gpu_runtime,
        {
            "mutationEnabled": False,
            "paths": {
                "runRoot": str(tmp_path / "shadow"),
                "promotionRoot": str(control),
                "leaseState": str(control / "gpu-lease.json"),
                "eventLog": str(control / "gpu-events.jsonl"),
            },
            "gpu": {
                "expectedUuid": spec.gpu_uuid,
                "index": spec.gpu_index,
                "inventoryCommand": [str(tmp_path / "nvidia-smi"), "inventory"],
                "processQueryCommand": [str(tmp_path / "nvidia-smi"), "processes"],
                "cleanObservations": 3,
                "cleanObservationIntervalSeconds": 0.25,
            },
            "pollIntervalSeconds": 0.1,
            "trainer": {
                "launchCommand": ["trainer", "{checkpoint_path}"],
                "gracefulCommand": ["graceful-trainer", "{pid}"],
                "checkpointPath": str(spec.immutable_inputs["trainer_checkpoint"].path),
                "drainTimeoutSeconds": 0.3,
                "checkpointTimeoutSeconds": 0.3,
                "checkpointStableSeconds": 0.0,
                "requireCheckpointChange": True,
                "startTimeoutSeconds": 0.3,
            },
            "evaluator": {
                "launchCommand": ["disabled"],
                "drainCommand": [],
                "processCount": 8,
                "drainTimeoutSeconds": 0.3,
            },
            "ownerId": "shadow",
        },
    )
    runtime = provisioner._default_build_lease_runtime(
        spec=spec,
        source_gpu_lease_config=gpu_runtime,
        destination=paths.lease_runtime,
        model_probe_config=paths.model_probe_config,
    )
    lease = provisioner._default_publish_lease_specs(
        spec=spec,
        probe_spec_path=paths.lease_probe_spec,
        drill_spec_path=paths.lease_drill_spec,
        gpu_lease_config=paths.lease_runtime,
        model_probe_config=paths.model_probe_config,
    )

    loaded_workload = load_workload_spec(paths.topology_workload_spec)
    assert loaded_workload.identity == topology["workload_identity"]
    assert (
        load_benchmark_spec(paths.topology_benchmark_spec).identity
        == topology["benchmark_identity"]
    )
    loaded_probe = load_probe_spec(paths.lease_probe_spec)
    assert loaded_probe.identity == lease["probe_identity"]
    loaded_drill = load_drill_spec(paths.lease_drill_spec)
    assert loaded_drill.identity == lease["drill_identity"]
    assert loaded_drill.trainer_source_kind == "launch"
    assert (
        loaded_workload.timeout_seconds
        == spec.publisher_config["topology_benchmark"]["timeout_seconds"] - 5
    )
    assert (
        loaded_probe.timeout_seconds
        == spec.publisher_config["lease_drill"]["probe_timeout_seconds"]
        - OUTER_DRILL_TIMEOUT_RESERVE_SECONDS
        - DEFAULT_CLEANUP_MARGIN_SECONDS
        - DEFAULT_POLL_INTERVAL_SECONDS
    )
    assert loaded_probe.cleanup_margin_seconds == DEFAULT_CLEANUP_MARGIN_SECONDS
    assert loaded_probe.poll_interval_seconds == DEFAULT_POLL_INTERVAL_SECONDS
    assert (
        loaded_drill.evaluator_probe_timeout_seconds
        == spec.publisher_config["lease_drill"]["probe_timeout_seconds"]
    )
    from risk_score.gpu_lease import RuntimeConfig

    loaded_runtime = RuntimeConfig.from_json_file(paths.lease_runtime)
    assert runtime["mutation_enabled"] is True
    assert runtime["installed"] is False
    assert loaded_runtime.mutation_enabled is True
    assert loaded_runtime.evaluator_process_count == 1
    assert "autonomy_lease_worker.py" in loaded_runtime.evaluator_launch_command[1]
    assert load_probe_spec(paths.lease_probe_spec).expected_evaluator_process_count == 1


def test_champion_control_planes_share_one_locked_inactive_transaction(
    tmp_path, monkeypatch
):
    from risk_score import promotion_state

    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    real_lock = promotion_state.ControllerLock
    observations = {"active": False, "entries": 0, "exits": 0}

    class TrackingLock:
        def __init__(self, path, **kwargs):
            self.path = Path(path)
            self.delegate = real_lock(path, **kwargs)

        def __enter__(self):
            result = self.delegate.__enter__()
            if self.path.name == "controller.lock":
                assert observations["active"] is False
                observations["active"] = True
                observations["entries"] += 1
            return result

        def __exit__(self, *args):
            try:
                return self.delegate.__exit__(*args)
            finally:
                if self.path.name == "controller.lock":
                    observations["active"] = False
                    observations["exits"] += 1

    class LockAwarePublishers(FakePublishers):
        def verify_target_inactive(self, **kwargs):
            assert observations["active"] is True
            return super().verify_target_inactive(**kwargs)

        def bootstrap_registry(self, **kwargs):
            assert observations["active"] is True
            assert kwargs["controller_transaction"]["held"] is True
            return super().bootstrap_registry(**kwargs)

        def initialize_champion(self, **kwargs):
            assert observations["active"] is True
            assert kwargs["controller_transaction"]["held"] is True
            return super().initialize_champion(**kwargs)

    monkeypatch.setattr(promotion_state, "ControllerLock", TrackingLock)
    fake = LockAwarePublishers(fixture)
    provisioner.materialize_provisioner(fixture["spec_path"], adapters=fake.adapters())

    assert observations == {"active": False, "entries": 1, "exits": 1}
    assert fake.calls.count("target-inactive") == 2
    assert fake.calls.index("registry-bootstrap") < fake.calls.index("champion")


def test_default_registry_service_and_champion_adapters_share_contracts(tmp_path):
    from test_suite_rotation import make_suite

    from risk_score.promotion_state import EventRegistry, load_champion
    from risk_score.suite_rotation import SuiteRotationRegistry, load_registry_spec
    from risk_score.suite_rotation_service import load_service_spec

    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    spec = fixture["spec"]
    paths = provisioner._generation_paths(spec)
    provisioner._ensure_directory(paths.root)
    provisioner._default_publish_registry_spec(
        spec=spec,
        destination=paths.registry_spec,
    )
    registry_spec = load_registry_spec(paths.registry_spec)
    shutil.rmtree(spec.suite_manifest.path.parent)
    make_suite(
        spec.suite_manifest.path.parent,
        registry_spec,
        spec.immutable_inputs["suite_champion_model"].sha256,
        nonce=1,
    )
    provisioner._default_publish_suite_service_spec(
        spec=spec,
        destination=paths.suite_rotation_spec,
        registry_spec_path=paths.registry_spec,
    )
    shadow = provisioner._default_build_shadow_runtime(
        spec=spec,
        output_dir=paths.shadow_runtime_root,
        suite_manifest_path=spec.suite_manifest.path,
    )
    inactive = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        ["systemctl"], 3, stdout="inactive\n", stderr=""
    )

    with provisioner._default_controller_transaction(spec=spec) as transaction:
        before = provisioner._default_verify_target_inactive(command_runner=inactive)
        registry_result = provisioner._default_bootstrap_registry(
            spec=spec,
            registry_spec_path=paths.registry_spec,
            suite_manifest=spec.suite_manifest.path,
            controller_transaction=transaction,
        )
        champion_result = provisioner._default_initialize_champion(
            spec=spec,
            promotion_runtime_path=Path(shadow["promotion_runtime"]),
            receipt_path=paths.champion_receipt,
            target_observation=before,
            controller_transaction=transaction,
        )
        after = provisioner._default_verify_target_inactive(command_runner=inactive)

    service_spec = load_service_spec(paths.suite_rotation_spec)
    suite_state = SuiteRotationRegistry(paths.registry_spec).reconstruct()
    promotion_state = EventRegistry(spec.run_root / "promotion").reconstruct()
    champion = load_champion(spec.run_root / "promotion" / "champion.json")
    expected_hash = spec.immutable_inputs["suite_champion_model"].sha256

    assert registry_spec.initial_champion.sha256 == expected_hash
    assert service_spec.registry_spec.path == paths.registry_spec
    assert registry_result["champion_sha256"] == expected_hash
    assert champion_result["champion"]["champion_hash"] == expected_hash
    assert suite_state.current_champion.sha256 == expected_hash
    assert promotion_state.current_champion_hash == expected_hash
    assert champion.generation_id == spec.initial_generation_id
    assert before["inactive"] is after["inactive"] is True


def test_plan_and_apply_cutover_never_enable_bootstrap_service(tmp_path):
    fixture = make_fixture(tmp_path)
    fake = FakePublishers(fixture)
    systemctl = FakeSystemctl()

    plan = provisioner.plan_provisioning(fixture["spec_path"])
    assert plan["legacy_to_prepare"]["actions"][-1] == [
        "systemctl",
        "enable",
        "--now",
        provisioner.PREPARE_PATH_UNIT_NAME,
    ]
    assert {
        action[-1]
        for action in plan["legacy_to_prepare"]["actions"]
        if "disable" in action
    } == {
        provisioner.LEGACY_PATH_UNIT_NAME,
        provisioner.PREPARE_PATH_UNIT_NAME,
        provisioner.PREPARE_SERVICE_UNIT_NAME,
        provisioner.BOOTSTRAP_PATH_UNIT_NAME,
        provisioner.BOOTSTRAP_SERVICE_UNIT_NAME,
    }
    result = provisioner.materialize_provisioner(
        fixture["spec_path"],
        apply=True,
        systemctl_runner=systemctl,
        adapters=fake.adapters(),
    )
    assert result["state"] == "WAIT"
    mutations = [
        command
        for command in systemctl.commands
        if command[1] not in {"is-enabled", "is-active"}
    ]
    assert mutations == [
        (
            "systemctl",
            "disable",
            "--now",
            provisioner.LEGACY_PATH_UNIT_NAME,
        ),
        ("systemctl", "daemon-reload"),
        (
            "systemctl",
            "enable",
            "--now",
            provisioner.PREPARE_PATH_UNIT_NAME,
        ),
    ]
    assert all(
        not (
            "enable" in command
            and command[-1] == provisioner.BOOTSTRAP_SERVICE_UNIT_NAME
        )
        for command in systemctl.commands
    )


def test_apply_ready_arms_only_bootstrap_path(tmp_path):
    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    fake = FakePublishers(fixture)
    systemctl = FakeSystemctl()

    result = provisioner.materialize_provisioner(
        fixture["spec_path"],
        apply=True,
        systemctl_runner=systemctl,
        adapters=fake.adapters(),
    )

    assert result["state"] == "APPLIED"
    assert (
        "systemctl",
        "enable",
        "--now",
        provisioner.BOOTSTRAP_PATH_UNIT_NAME,
    ) in systemctl.commands
    assert not any(
        "enable" in command and command[-1] == provisioner.BOOTSTRAP_SERVICE_UNIT_NAME
        for command in systemctl.commands
    )
    mutation_count = len(
        [
            command
            for command in systemctl.commands
            if command[1] not in {"is-enabled", "is-active"}
        ]
    )
    replay = provisioner.materialize_provisioner(
        fixture["spec_path"],
        apply=True,
        systemctl_runner=systemctl,
        adapters=FakePublishers(fixture).adapters(),
    )
    assert replay["state"] == "APPLIED"
    assert (
        len(
            [
                command
                for command in systemctl.commands
                if command[1] not in {"is-enabled", "is-active"}
            ]
        )
        == mutation_count
    )


def test_apply_reconciles_inactive_bootstrap_path_with_repair_receipt(tmp_path):
    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    systemctl = FakeSystemctl()
    provisioner.materialize_provisioner(
        fixture["spec_path"],
        apply=True,
        systemctl_runner=systemctl,
        adapters=FakePublishers(fixture).adapters(),
    )
    systemctl.enabled.discard(provisioner.BOOTSTRAP_PATH_UNIT_NAME)
    systemctl.active.discard(provisioner.BOOTSTRAP_PATH_UNIT_NAME)

    replay = provisioner.materialize_provisioner(
        fixture["spec_path"],
        apply=True,
        systemctl_runner=systemctl,
        adapters=FakePublishers(fixture).adapters(),
    )

    assert replay["state"] == "APPLIED"
    assert provisioner.BOOTSTRAP_PATH_UNIT_NAME in systemctl.enabled
    assert provisioner.BOOTSTRAP_PATH_UNIT_NAME in systemctl.active
    repairs = sorted(
        (fixture["spec"].outputs.receipts_root / "installation").glob(
            "arm-bootstrap-path.repair-*.json"
        )
    )
    assert len(repairs) == 1


def test_service_ownership_handoff_targets_control_state_not_immutable_inputs(
    tmp_path, monkeypatch
):
    fixture = make_fixture(tmp_path)
    spec = fixture["spec"]
    promotion = spec.run_root / "promotion"
    events = promotion / "events"
    events.mkdir(parents=True, exist_ok=True)
    lock = promotion / "controller.lock"
    lock.write_text("owner\n", encoding="utf-8")
    mutable_directories = []
    mutable_files = []
    readable_roots = []

    monkeypatch.setattr(
        provisioner.pwd,
        "getpwnam",
        lambda _user: type("Account", (), {"pw_uid": 123, "pw_gid": 456})(),
    )
    monkeypatch.setattr(
        provisioner,
        "_secure_mutable_directory",
        lambda path, **_kwargs: mutable_directories.append(Path(path)),
    )
    monkeypatch.setattr(
        provisioner,
        "_secure_mutable_file",
        lambda path, **_kwargs: mutable_files.append(Path(path)),
    )
    monkeypatch.setattr(
        provisioner,
        "_make_root_owned_tree_readable",
        lambda path: readable_roots.append(Path(path)),
    )
    monkeypatch.setattr(provisioner.os, "geteuid", lambda: 0)

    provisioner._prepare_service_ownership(spec)

    assert promotion not in mutable_directories
    assert events in mutable_directories
    assert Path(spec.publisher_config["scheduler_directory"]) in mutable_directories
    assert lock in mutable_files
    assert spec.state_root not in mutable_directories
    assert spec.bootstrap_state_root not in mutable_directories
    assert set(readable_roots) == {
        spec.state_root,
        spec.bootstrap_state_root,
        spec.outputs.artifacts_root,
    }
    assert all(
        binding.path not in {*mutable_directories, *mutable_files}
        for binding in spec.immutable_inputs.values()
    )


def test_real_apply_requires_root_and_canonical_systemd_destination(
    tmp_path, monkeypatch
):
    fixture = make_fixture(tmp_path)
    monkeypatch.setattr(provisioner.os, "geteuid", lambda: 501)
    with pytest.raises(provisioner.ProvisionerApplyError, match="requires root"):
        provisioner._validate_real_apply(fixture["spec"], subprocess.run)

    monkeypatch.setattr(provisioner.os, "geteuid", lambda: 0)
    with pytest.raises(
        provisioner.ProvisionerApplyError,
        match="activation_destination=/etc/systemd/system",
    ):
        provisioner._validate_real_apply(fixture["spec"], subprocess.run)


def test_lease_outer_timeout_requires_full_exported_reserve(tmp_path):
    from risk_score.autonomy_lease_probe import (
        DEFAULT_CLEANUP_MARGIN_SECONDS,
        DEFAULT_POLL_INTERVAL_SECONDS,
        OUTER_DRILL_TIMEOUT_RESERVE_SECONDS,
    )

    fixture = make_fixture(tmp_path)
    value = json.loads(fixture["spec_path"].read_text(encoding="utf-8"))
    value["publisher_config"]["lease_drill"]["probe_timeout_seconds"] = (
        OUTER_DRILL_TIMEOUT_RESERVE_SECONDS
        + DEFAULT_CLEANUP_MARGIN_SECONDS
        + DEFAULT_POLL_INTERVAL_SECONDS
    )
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    os.chmod(fixture["spec_path"], 0o644)
    write_canonical(fixture["spec_path"], value)

    with pytest.raises(provisioner.ProvisionerSpecError, match="poll interval"):
        provisioner.ProvisionerSpec.load(fixture["spec_path"])


def test_bootstrap_enable_verification_failure_cleans_path_and_service(tmp_path):
    class StartsServiceThenFailsVerification(FakeSystemctl):
        def __init__(self):
            super().__init__()
            self.enabled = {provisioner.PREPARE_PATH_UNIT_NAME}
            self.active = {provisioner.PREPARE_PATH_UNIT_NAME}
            self.fail_bootstrap_verification = False

        def __call__(self, argv, **kwargs):
            command = tuple(argv)
            if command == (
                "systemctl",
                "enable",
                "--now",
                provisioner.BOOTSTRAP_PATH_UNIT_NAME,
            ):
                result = super().__call__(argv, **kwargs)
                self.active.add(provisioner.BOOTSTRAP_SERVICE_UNIT_NAME)
                self.fail_bootstrap_verification = True
                return result
            if self.fail_bootstrap_verification and command == (
                "systemctl",
                "is-enabled",
                provisioner.BOOTSTRAP_PATH_UNIT_NAME,
            ):
                self.commands.append(command)
                self.fail_bootstrap_verification = False
                return subprocess.CompletedProcess(
                    argv, 1, stdout="disabled\n", stderr=""
                )
            return super().__call__(argv, **kwargs)

    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    provisioner.materialize_provisioner(
        fixture["spec_path"], adapters=FakePublishers(fixture).adapters()
    )
    systemctl = StartsServiceThenFailsVerification()

    with pytest.raises(provisioner.ProvisionerApplyError, match="rollback"):
        provisioner.apply_bootstrap_arm(
            fixture["spec"],
            command_runner=systemctl,
        )

    mutations = [
        command
        for command in systemctl.commands
        if command[1] not in {"is-enabled", "is-active"}
    ]
    assert (
        "systemctl",
        "disable",
        provisioner.BOOTSTRAP_PATH_UNIT_NAME,
    ) in mutations
    assert (
        "systemctl",
        "stop",
        provisioner.BOOTSTRAP_SERVICE_UNIT_NAME,
    ) in mutations
    assert provisioner.BOOTSTRAP_SERVICE_UNIT_NAME not in systemctl.enabled
    assert provisioner.BOOTSTRAP_SERVICE_UNIT_NAME not in systemctl.active


def test_partial_enable_failure_restores_every_initial_unit_state(tmp_path):
    class PartialEnableFailure(FakeSystemctl):
        def __init__(self):
            super().__init__()
            self.enabled = {provisioner.PREPARE_PATH_UNIT_NAME}
            self.active = {provisioner.PREPARE_PATH_UNIT_NAME}

        def __call__(self, argv, **kwargs):
            command = tuple(argv)
            if command == (
                "systemctl",
                "enable",
                "--now",
                provisioner.BOOTSTRAP_PATH_UNIT_NAME,
            ):
                self.commands.append(command)
                self.enabled.add(provisioner.BOOTSTRAP_PATH_UNIT_NAME)
                self.active.update(
                    {
                        provisioner.BOOTSTRAP_PATH_UNIT_NAME,
                        provisioner.BOOTSTRAP_SERVICE_UNIT_NAME,
                    }
                )
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout="",
                    stderr="enable failed after starting units",
                )
            return super().__call__(argv, **kwargs)

    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    provisioner.materialize_provisioner(
        fixture["spec_path"], adapters=FakePublishers(fixture).adapters()
    )
    systemctl = PartialEnableFailure()
    initial_enabled = set(systemctl.enabled)
    initial_active = set(systemctl.active)

    with pytest.raises(
        provisioner.ProvisionerApplyError, match="rollback_errors=\\[\\]"
    ):
        provisioner.apply_bootstrap_arm(
            fixture["spec"],
            command_runner=systemctl,
        )

    assert systemctl.enabled == initial_enabled
    assert systemctl.active == initial_active
    assert (
        "systemctl",
        "stop",
        provisioner.BOOTSTRAP_SERVICE_UNIT_NAME,
    ) in systemctl.commands
    assert (
        "systemctl",
        "disable",
        provisioner.BOOTSTRAP_PATH_UNIT_NAME,
    ) in systemctl.commands


def test_dirty_checkout_is_rejected_before_any_publication(tmp_path):
    fixture = make_fixture(tmp_path)
    (fixture["repo"] / "dirty").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(
        provisioner.ProvisionerDriftError, match="clean deployed checkout"
    ):
        provisioner.materialize_provisioner(fixture["spec_path"])

    assert not fixture["spec"].outputs.prepare_service_unit.exists()
    assert not fixture["state_root"].exists()


def test_suite_hash_and_candidate_inventory_drift_fail_closed(tmp_path):
    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    fake = FakePublishers(fixture)
    provisioner.materialize_provisioner(fixture["spec_path"], adapters=fake.adapters())

    publish_readiness(fixture, marker="changed")
    with pytest.raises(provisioner.ProvisionerDriftError, match="suite/hash"):
        provisioner.materialize_provisioner(
            fixture["spec_path"], adapters=FakePublishers(fixture).adapters()
        )

    publish_readiness(fixture)
    (fixture["candidate_inbox"] / "late-candidate").write_text(
        "late\n", encoding="utf-8"
    )
    with pytest.raises(provisioner.ProvisionerDriftError, match="candidate inventory"):
        provisioner.materialize_provisioner(
            fixture["spec_path"], adapters=FakePublishers(fixture).adapters()
        )


def test_candidate_change_during_generation_never_arms_bootstrap(tmp_path):
    fixture = make_fixture(tmp_path)
    publish_readiness(fixture)
    fake = FakePublishers(fixture, mutate_candidates=True)

    with pytest.raises(
        provisioner.ProvisionerDriftError, match="candidate inventory changed"
    ):
        provisioner.materialize_provisioner(
            fixture["spec_path"], adapters=fake.adapters()
        )

    assert not fixture["spec"].outputs.bootstrap_spec.exists()
    assert not fixture["spec"].outputs.bootstrap_path_unit.exists()


def test_conflicting_prepare_unit_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    unit = fixture["spec"].outputs.prepare_service_unit
    unit.write_text("conflict\n", encoding="utf-8")

    with pytest.raises(provisioner.ProvisionerConflictError, match="conflicts"):
        provisioner.materialize_provisioner(fixture["spec_path"])

    assert not fixture["spec"].outputs.prepare_path_unit.exists()


def test_symlinked_and_duplicate_unit_destinations_are_rejected(tmp_path):
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    linked = make_fixture(linked_root)
    target = tmp_path / "unsafe-unit"
    target.write_text("unsafe\n", encoding="utf-8")
    linked["spec"].outputs.prepare_service_unit.symlink_to(target)

    with pytest.raises(provisioner.ProvisionerSpecError, match="symlink"):
        provisioner.materialize_provisioner(linked["spec_path"])

    duplicated_root = tmp_path / "duplicated"
    duplicated_root.mkdir()
    duplicated = make_fixture(duplicated_root)
    value = json.loads(duplicated["spec_path"].read_text(encoding="utf-8"))
    value["outputs"]["bootstrap_path_unit"] = value["outputs"]["prepare_path_unit"]
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    os.chmod(duplicated["spec_path"], 0o644)
    write_canonical(duplicated["spec_path"], value)

    with pytest.raises(
        provisioner.ProvisionerSpecError, match="unit output|duplicates"
    ):
        provisioner.load_provisioner_spec(duplicated["spec_path"])


def test_failed_cutover_rolls_back_legacy_path(tmp_path):
    fixture = make_fixture(tmp_path)
    fake = FakePublishers(fixture)
    failure = (
        "systemctl",
        "enable",
        "--now",
        provisioner.PREPARE_PATH_UNIT_NAME,
    )
    systemctl = FakeSystemctl(fail=failure)

    with pytest.raises(provisioner.ProvisionerApplyError, match="rollback"):
        provisioner.materialize_provisioner(
            fixture["spec_path"],
            apply=True,
            systemctl_runner=systemctl,
            adapters=fake.adapters(),
        )

    mutations = [
        command
        for command in systemctl.commands
        if command[1] not in {"is-enabled", "is-active"}
    ]
    assert mutations[:3] == [
        (
            "systemctl",
            "disable",
            "--now",
            provisioner.LEGACY_PATH_UNIT_NAME,
        ),
        ("systemctl", "daemon-reload"),
        failure,
    ]
    assert (
        "systemctl",
        "enable",
        provisioner.LEGACY_PATH_UNIT_NAME,
    ) in mutations
    assert ("systemctl", "start", provisioner.LEGACY_PATH_UNIT_NAME) in mutations
    assert mutations[-1] == ("systemctl", "daemon-reload")
    assert systemctl.enabled == {provisioner.LEGACY_PATH_UNIT_NAME}
    assert systemctl.active == {provisioner.LEGACY_PATH_UNIT_NAME}


def test_410_candidate_inventory_is_the_exact_backlog_bound(tmp_path):
    fixture = make_fixture(tmp_path, candidate_count=410)
    publish_readiness(fixture)
    adapters = replace(
        FakePublishers(fixture).adapters(),
        candidate_inventory=provisioner._default_candidate_inventory,
    )
    result = provisioner.materialize_provisioner(
        fixture["spec_path"],
        adapters=adapters,
    )
    loaded = autonomy_bootstrap.BootstrapSpec.load(
        fixture["spec"].outputs.bootstrap_spec
    )
    backlog = next(gate for gate in loaded.gates if gate.gate_id == "backlog-bound")
    assert result["state"] == "MATERIALIZED"
    assert backlog.requirements == {
        "maximum_candidates": 410,
        "maximum_active_queue": 3,
    }
    assert flag(backlog.argv, "--maximum-candidates") == "410"
    inventory = json.loads(
        (
            fixture["state_root"]
            / "artifacts"
            / fixture["spec"].identity
            / "candidate-inventory-before.json"
        ).read_text(encoding="utf-8")
    )
    assert inventory["candidate_count"] == 410
    assert inventory["ignored"] == []
