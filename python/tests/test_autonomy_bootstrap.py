import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from risk_score import autonomy_bootstrap as bootstrap
from risk_score.gpu_lease import SCHEMA_VERSION as GPU_LEASE_SCHEMA_VERSION
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_controller import PROMOTION_FAILURE_STEPS
from risk_score.promotion_host import atomic_replace_json
from risk_score.service_activation import (
    FULL_SERVICE_UNIT_NAMES,
    TARGET_UNIT,
    apply_service_activation,
)


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def binding(path, sha256=None):
    return {
        "path": str(path),
        "sha256": file_sha256(path) if sha256 is None else sha256,
    }


def flag(argv, name):
    return argv[argv.index(name) + 1]


class RecordingRunner:
    def __init__(self, fixture, *, fail_gate=None, selected_process_count=8):
        self.fixture = fixture
        self.fail_gate = fail_gate
        self.selected_process_count = selected_process_count
        self.commands = []

    def __call__(self, argv, **kwargs):
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "shell": False,
        }
        argv = list(argv)
        self.commands.append(argv)
        if argv[1:3] == ["-m", "risk_score.build_live_runtime"]:
            result = self._write_runtime(argv)
            return subprocess.CompletedProcess(
                argv, 0, stdout=canonical_json(result) + "\n", stderr=""
            )
        if argv[1:3] == ["-m", "risk_score.service_activation"]:
            result = self._apply_activation(argv)
            return subprocess.CompletedProcess(
                argv, 0, stdout=canonical_json(result) + "\n", stderr=""
            )

        gate_id = argv[1]
        gate = self.fixture["gates"][gate_id]
        if gate_id == "filesystem-rename-fsync":
            value = {
                "schema_version": 1,
                "contract": "risk-score-live-filesystem-test-v1",
                "root": gate["requirements"]["root"],
                "device": Path(gate["requirements"]["root"]).stat().st_dev,
                "atomic_rename_preserved_inode": True,
                "directory_fsync_succeeded": True,
                "payload_sha256": "f" * 64,
            }
            atomic_replace_json(Path(gate["evidence"]), value)
        elif gate_id == "candidate-inventory":
            value = {
                "schema_version": 1,
                "contract": "risk-score-live-candidate-inventory-v1",
                "inbox": gate["requirements"]["inbox"],
                "candidate_count": 0,
                "ignored": [],
                "candidates": [],
            }
            value["inventory_sha256"] = canonical_sha256(value)
            atomic_replace_json(Path(gate["evidence"]), value)
        else:
            decision = "FAIL" if gate_id == self.fail_gate else "PASS"
            bootstrap.publish_gate_evidence(
                Path(gate["evidence"]),
                gate_id,
                self._checks(gate_id, gate),
                decision=decision,
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def _checks(self, gate_id, gate):
        common_hash = "9" * 64
        if gate_id == "curation-suite-readiness":
            return {
                "curation_complete": True,
                "suite_complete": True,
            }
        if gate_id == "deployment-hash-validation":
            return {"deployment_valid": True}
        if gate_id == "cuda-model-probes":
            return {
                "cuda_available": True,
                "gpu_exclusive": True,
                "gpu_uuid": gate["requirements"]["expected_gpu_uuid"],
                "models": [
                    {
                        "model_sha256": digest,
                        "finite": True,
                        "deterministic": True,
                    }
                    for digest in gate["requirements"]["model_sha256s"]
                ],
            }
        if gate_id == "evaluator-topology-benchmark":
            throughput = {4: 4.0, 8: 4.0, 16: 4.0}
            throughput[self.selected_process_count] = 9.0
            return {
                "benchmarks": [
                    {
                        "process_count": count,
                        "throughput_per_second": throughput[count],
                        "output_sha256": common_hash,
                        "repeat_output_sha256": common_hash,
                    }
                    for count in (4, 8, 16)
                ],
                "selected_process_count": self.selected_process_count,
            }
        if gate_id == "trainer-evaluator-lease-drill":
            return {
                "gpu_uuid": gate["requirements"]["expected_gpu_uuid"],
                "lease_schema_version": GPU_LEASE_SCHEMA_VERSION,
                "trainer_drained": True,
                "checkpoint_handoff_verified": True,
                "evaluator_exclusive": True,
                "process_overlap_observed": False,
                "trainer_restored": True,
                "lease_clean_observations": 3,
                "release_clean_observations": 3,
                "safety_halt": False,
            }
        if gate_id == "disposable-canary-drill":
            return {
                "canary_passed": True,
                "fresh_audit_passed": True,
                "disposable_root_removed": True,
                "production_unchanged": True,
            }
        if gate_id == "crash-replay-drill":
            return {
                "boundaries": [
                    {
                        "step": step,
                        "crash_injected": True,
                        "replay_converged": True,
                    }
                    for step in PROMOTION_FAILURE_STEPS
                ],
                "production_unchanged": True,
            }
        if gate_id == "rollback-before-admission-drill":
            return {
                "rollback_requested": True,
                "refused_without_forensic_flow": True,
                "staged_data_preserved": True,
                "champion_unchanged": True,
                "production_unchanged": True,
            }
        if gate_id == "rollback-after-admission-drill":
            return {
                "rollback_complete": True,
                "champion_restored": True,
                "checkpoint_restored": True,
                "admitted_data_quarantined": True,
                "derived_data_removed": True,
                "watermarks_restored": True,
                "production_unchanged": True,
            }
        if gate_id == "shadow-controller-replay":
            return {
                "mutation_enabled": False,
                "first_replay_sha256": common_hash,
                "second_replay_sha256": common_hash,
                "event_log_sha256": "8" * 64,
                "production_unchanged": True,
            }
        if gate_id == "backlog-bound":
            return {
                "candidate_count": 3,
                "maximum_candidates": gate["requirements"]["maximum_candidates"],
                "active_queue_depth": 2,
                "maximum_active_queue": gate["requirements"]["maximum_active_queue"],
                "backpressure_enforced": True,
                "evidence_preserved": True,
            }
        raise AssertionError(f"missing test evidence for {gate_id}")

    def _write_runtime(self, argv):
        output = Path(flag(argv, "--output-dir"))
        selected_process_count = int(flag(argv, "--evaluator-process-count"))
        output.mkdir(parents=True, exist_ok=True)
        gpu_path = output / "gpu-lease-runtime.json"
        promotion_path = output / "promotion-runtime.json"
        service_path = output / "promotion-services.json"
        deployment_path = output / "deployment-manifest.json"
        gpu = {
            "schemaVersion": 1,
            "mutationEnabled": True,
            "evaluator": {"processCount": selected_process_count},
        }
        write_canonical(gpu_path, gpu)
        promotion = {
            "schemaVersion": 1,
            "mutationEnabled": True,
            "commands": {
                "evaluator": [
                    str(Path(sys.executable).resolve()),
                    "--shards",
                    "8",
                ]
            },
            "hashes": {"gpuLeaseConfig": file_sha256(gpu_path)},
            "paths": {"gpuLeaseConfig": str(gpu_path)},
        }
        write_canonical(promotion_path, promotion)

        unit_names = {**FULL_SERVICE_UNIT_NAMES, "target": TARGET_UNIT}
        unit_records = {}
        units_dir = output / "systemd"
        units_dir.mkdir()
        for key, name in sorted(unit_names.items()):
            unit_path = units_dir / name
            unit_path.write_text(f"[Unit]\nDescription={name}\n", encoding="utf-8")
            unit_records[key] = {
                "path": str(unit_path),
                "sha256": file_sha256(unit_path),
            }
        service_inputs = {
            "autonomy_policy": binding(Path(flag(argv, "--autonomy-policy"))),
            "executor_spec": binding(Path(flag(argv, "--cluster-executor-spec"))),
            "adaptive_spec": binding(Path(flag(argv, "--adaptive-training-spec"))),
            "suite_registry_spec": binding(Path(flag(argv, "--suite-registry-spec"))),
        }
        service = {
            "schema_version": 3,
            "contract": "risk-score-host-services-v3",
            "mutation_enabled": True,
            "full_autonomy": True,
            "evaluator_process_count": selected_process_count,
            "service_user": "test",
            "service_inputs": service_inputs,
            "services": {
                key: {"argv": ["/bin/true"]} for key in FULL_SERVICE_UNIT_NAMES
            },
            "systemd_units": unit_records,
        }
        write_canonical(service_path, service)

        files = {
            "module:autonomy_bootstrap.py": {
                "path": str(Path(bootstrap.__file__)),
                "sha256": file_sha256(Path(bootstrap.__file__)),
            },
            "promotion_runtime": binding(promotion_path),
            "gpu_lease_runtime": binding(gpu_path),
            "service_spec": binding(service_path),
        }
        files.update(
            {f"systemd:{key}": dict(record) for key, record in unit_records.items()}
        )
        deployment = {
            "schema_version": 1,
            "contract": "risk-score-live-runtime-deployment-v1",
            "source_revision": "a" * 40,
            "source_sha256": hashlib.sha256(("a" * 40).encode()).hexdigest(),
            "files": files,
        }
        deployment["manifest_sha256"] = canonical_sha256(deployment)
        write_canonical(deployment_path, deployment)
        return {
            "promotion_runtime": str(promotion_path),
            "promotion_runtime_sha256": file_sha256(promotion_path),
            "gpu_lease_runtime": str(gpu_path),
            "gpu_lease_runtime_sha256": file_sha256(gpu_path),
            "deployment_manifest": str(deployment_path),
            "deployment_manifest_sha256": file_sha256(deployment_path),
            "service_spec": str(service_path),
            "service_spec_sha256": file_sha256(service_path),
            "systemd_units": unit_records,
            "mutation_enabled": True,
            "full_autonomy": True,
            "evaluator_process_count": selected_process_count,
        }

    def _apply_activation(self, argv):
        def systemctl(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="active\n", stderr="")

        return apply_service_activation(
            spec_path=Path(flag(argv, "--spec")),
            destination=Path(flag(argv, "--destination")),
            receipt_path=Path(flag(argv, "--receipt")),
            command_runner=systemctl,
        )


def make_fixture(tmp_path):
    run_root = tmp_path / "run"
    state_root = run_root / "promotion" / "autonomy-bootstrap"
    candidate_inbox = run_root / "modelstobetested"
    systemd = tmp_path / "systemd"
    repo = tmp_path / "repo"
    suites = run_root / "evaluation" / "promotion-suites-v3"
    for directory in (
        run_root,
        state_root.parent,
        candidate_inbox,
        systemd,
        repo,
        suites,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    suite_manifest = suites / "manifest.json"
    suite = {
        "schemaVersion": 3,
        "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
        "machineReviewOnly": True,
        "banks": [],
        "cells": [],
    }
    suite["manifestPayloadSha256"] = canonical_sha256(suite)
    write_canonical(suite_manifest, suite)
    curation_status = run_root / "evaluation" / "curation" / "status.json"
    status = {
        "schema_version": 1,
        "contract": "risk-score-curation-pipeline-status-v1",
        "state": "complete",
        "error": None,
        "artifacts": {
            "reviewed_bank": {"complete": True},
            "suite": {"complete": True, "path": str(suites)},
        },
    }
    status["status_sha256"] = canonical_sha256(status)
    write_canonical(curation_status, status)

    deployed = repo / "deployed.py"
    deployed.write_text("VALUE = 1\n", encoding="utf-8")
    deployment_manifest = repo / "deployment-manifest.json"
    deployment = {
        "schema_version": 1,
        "contract": "risk-score-live-runtime-deployment-v1",
        "source_revision": "b" * 40,
        "source_sha256": hashlib.sha256(("b" * 40).encode()).hexdigest(),
        "files": {"deployed": binding(deployed)},
    }
    deployment["manifest_sha256"] = canonical_sha256(deployment)
    write_canonical(deployment_manifest, deployment)

    model_paths = []
    for name, contents in (
        ("original.bin.gz", b"original"),
        ("champion.bin.gz", b"champion"),
    ):
        path = run_root / name
        path.write_bytes(contents)
        model_paths.append(path)
    model_hashes = sorted(file_sha256(path) for path in model_paths)

    evidence_root = state_root / "gate-work"
    requirements = {
        "curation-suite-readiness": {
            "curation_status": str(curation_status),
            "suite_manifest": str(suite_manifest),
        },
        "filesystem-rename-fsync": {"root": str(run_root)},
        "deployment-hash-validation": {
            "manifest": str(deployment_manifest),
        },
        "candidate-inventory": {"inbox": str(candidate_inbox)},
        "cuda-model-probes": {
            "expected_gpu_uuid": "GPU-test-uuid",
            "model_sha256s": model_hashes,
        },
        "evaluator-topology-benchmark": {"choices": [4, 8, 16]},
        "trainer-evaluator-lease-drill": {
            "expected_gpu_uuid": "GPU-test-uuid",
            "minimum_clean_observations": 3,
        },
        "disposable-canary-drill": {},
        "crash-replay-drill": {},
        "rollback-before-admission-drill": {},
        "rollback-after-admission-drill": {},
        "shadow-controller-replay": {},
        "backlog-bound": {
            "maximum_candidates": 10,
            "maximum_active_queue": 3,
        },
    }
    gate_inputs = {
        "curation-suite-readiness": [
            binding(curation_status, None),
            binding(suite_manifest, None),
        ],
        "deployment-hash-validation": [binding(deployment_manifest)],
        "cuda-model-probes": [binding(path) for path in model_paths],
    }
    gates = []
    gate_map = {}
    for gate_id in bootstrap.GATE_ORDER:
        evidence = evidence_root / f"{gate_id}.json"
        gate = {
            "id": gate_id,
            "argv": [sys.executable, gate_id],
            "evidence": str(evidence),
            "inputs": gate_inputs.get(gate_id, []),
            "outputs": [str(evidence)],
            "requirements": requirements[gate_id],
        }
        gates.append(gate)
        gate_map[gate_id] = gate

    executable = str(Path(sys.executable).resolve())
    placeholder = repo / "placeholder"
    placeholder.write_text("placeholder\n", encoding="utf-8")
    autonomy_inputs = {}
    autonomy_input_dir = run_root / "autonomy-inputs"
    autonomy_input_dir.mkdir()
    for name in (
        "autonomy-policy",
        "cluster-executor-spec",
        "adaptive-training-spec",
        "suite-registry-spec",
    ):
        path = autonomy_input_dir / f"{name}.json"
        write_canonical(path, {"name": name})
        autonomy_inputs[name] = path
    runtime_dir = state_root / "generated-runtime"
    runtime_argv = [
        executable,
        "-m",
        "risk_score.build_live_runtime",
        "--repo",
        str(repo),
        "--run-root",
        str(run_root),
        "--suite-dir",
        str(suites),
        "--katago-binary",
        str(placeholder),
        "--python-executable",
        executable,
        "--trainer-spec",
        str(placeholder),
        "--consumer-spec",
        str(placeholder),
        "--original-model",
        str(model_paths[0]),
        "--trainer-checkpoint",
        str(placeholder),
        "--gpu-uuid",
        "GPU-test-uuid",
        "--actor",
        "test-bootstrap",
        "--source-revision",
        "a" * 40,
        "--evaluator-process-count",
        "{selected_evaluator_processes}",
        "--output-dir",
        str(runtime_dir),
        "--mutation-enabled",
        "--full-autonomy",
        "--service-user",
        "test",
        "--shuffler-command-json",
        canonical_json([executable, "shuffler"]),
        "--exporter-command-json",
        canonical_json([executable, "exporter"]),
        "--cluster-executor-command-json",
        canonical_json([executable, "cluster-executor"]),
        "--adaptive-training-command-json",
        canonical_json([executable, "adaptive-training"]),
        "--suite-rotation-command-json",
        canonical_json([executable, "suite-rotation"]),
        "--autonomy-policy",
        str(autonomy_inputs["autonomy-policy"]),
        "--cluster-executor-spec",
        str(autonomy_inputs["cluster-executor-spec"]),
        "--adaptive-training-spec",
        str(autonomy_inputs["adaptive-training-spec"]),
        "--suite-registry-spec",
        str(autonomy_inputs["suite-registry-spec"]),
    ]
    activation_receipt = state_root / "activation.json"
    activation_argv = [
        executable,
        "-m",
        "risk_score.service_activation",
        "--spec",
        str(runtime_dir / "promotion-services.json"),
        "--destination",
        str(systemd),
        "--receipt",
        str(activation_receipt),
        "--apply",
    ]
    value = {
        "schema_version": 1,
        "contract": bootstrap.SPEC_CONTRACT,
        "state_root": str(state_root),
        "poll_interval_seconds": 0.01,
        "gates": gates,
        "runtime": {
            "argv": runtime_argv,
            "output_dir": str(runtime_dir),
        },
        "activation": {
            "argv": activation_argv,
            "destination": str(systemd),
            "receipt": str(activation_receipt),
            "required_units": list(bootstrap.REQUIRED_ACTIVATION_UNITS),
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    spec_path = tmp_path / "bootstrap.json"
    write_canonical(spec_path, value)
    return {
        "spec_path": spec_path,
        "spec_value": value,
        "state_root": state_root,
        "gates": gate_map,
        "systemd": systemd,
    }


def command_roles(commands):
    roles = []
    for argv in commands:
        if argv[1:3] == ["-m", "risk_score.build_live_runtime"]:
            roles.append("runtime")
        elif argv[1:3] == ["-m", "risk_score.service_activation"]:
            roles.append("activation")
        else:
            roles.append(argv[1])
    return roles


def test_spec_is_strict_canonical_and_hash_bound(tmp_path):
    fixture = make_fixture(tmp_path)
    loaded = bootstrap.load_bootstrap_spec(fixture["spec_path"])
    assert loaded.identity == fixture["spec_value"]["spec_sha256"]
    assert [gate.gate_id for gate in loaded.gates] == list(bootstrap.GATE_ORDER)

    fixture["spec_path"].write_text(
        json.dumps(fixture["spec_value"], indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(bootstrap.BootstrapError, match="canonical"):
        bootstrap.load_bootstrap_spec(fixture["spec_path"])

    changed = dict(fixture["spec_value"])
    changed["poll_interval_seconds"] = 1.0
    write_canonical(fixture["spec_path"], changed)
    with pytest.raises(bootstrap.BootstrapError, match="self-hash"):
        bootstrap.load_bootstrap_spec(fixture["spec_path"])


def test_gates_run_in_fixed_order_before_runtime_and_activation(tmp_path):
    fixture = make_fixture(tmp_path)
    runner = RecordingRunner(fixture)
    result = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=runner
    ).run_once()

    assert result["state"] == "active"
    assert command_roles(runner.commands) == [
        *bootstrap.GATE_ORDER,
        "runtime",
        "activation",
    ]


@pytest.mark.parametrize("damage", ["missing-receipt", "changed-evidence"])
def test_missing_or_changed_receipt_enters_terminal_halt(tmp_path, damage):
    fixture = make_fixture(tmp_path)
    commands = RecordingRunner(fixture)
    orchestrator = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=commands
    )
    assert orchestrator.run_once()["state"] == "active"
    activation_count = command_roles(commands.commands).count("activation")

    if damage == "missing-receipt":
        orchestrator.gate_receipt_path(2).unlink()
    else:
        canary = fixture["gates"]["disposable-canary-drill"]
        checks = commands._checks("disposable-canary-drill", canary)
        checks["canary_passed"] = False
        bootstrap.publish_gate_evidence(
            Path(canary["evidence"]),
            "disposable-canary-drill",
            checks,
        )

    with pytest.raises(bootstrap.BootstrapSafetyHalt):
        orchestrator.run_once()
    status = orchestrator.status()
    assert status["state"] == "safety-halt"
    assert command_roles(commands.commands).count("activation") == activation_count


def test_failed_gate_writes_halt_and_never_invokes_activation(tmp_path):
    fixture = make_fixture(tmp_path)
    commands = RecordingRunner(fixture, fail_gate="disposable-canary-drill")
    orchestrator = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=commands
    )

    with pytest.raises(bootstrap.BootstrapSafetyHalt):
        orchestrator.run_once()

    assert "runtime" not in command_roles(commands.commands)
    assert "activation" not in command_roles(commands.commands)
    halt = json.loads(orchestrator.safety_halt_path.read_text(encoding="utf-8"))
    payload = dict(halt)
    assert payload.pop("halt_sha256") == canonical_sha256(payload)
    assert halt["activation_invoked"] is False
    assert orchestrator.status()["state"] == "safety-halt"


def test_receipt_is_adopted_after_crash_before_journal_update(tmp_path):
    fixture = make_fixture(tmp_path)
    commands = RecordingRunner(fixture)
    fired = []

    def crash(stage):
        if stage == "after-gate-receipt:disposable-canary-drill" and not fired:
            fired.append(stage)
            raise bootstrap.BootstrapInterrupted("simulated process death")

    crashing = bootstrap.AutonomyBootstrap(
        fixture["spec_path"],
        command_runner=commands,
        failure_hook=crash,
    )
    with pytest.raises(bootstrap.BootstrapInterrupted):
        crashing.run_once()
    assert not crashing.safety_halt_path.exists()

    recovered = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=commands
    )
    assert recovered.run_once()["state"] == "active"
    roles = command_roles(commands.commands)
    assert roles.count("disposable-canary-drill") == 1
    assert roles.count("activation") == 1


def test_topology_selection_requires_deterministic_outputs():
    output = "1" * 64
    benchmarks = [
        {
            "process_count": count,
            "throughput_per_second": throughput,
            "output_sha256": output,
            "repeat_output_sha256": output,
        }
        for count, throughput in ((4, 5.0), (8, 8.0), (16, 12.0))
    ]
    assert bootstrap.select_evaluator_topology(benchmarks) == 16

    benchmarks[2] = {
        **benchmarks[2],
        "repeat_output_sha256": "2" * 64,
    }
    with pytest.raises(bootstrap.BootstrapError, match="not deterministic"):
        bootstrap.select_evaluator_topology(benchmarks)


def test_selected_topology_is_bound_into_generated_runtime_command(tmp_path):
    fixture = make_fixture(tmp_path)
    commands = RecordingRunner(fixture, selected_process_count=16)
    result = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=commands
    ).run_once()

    runtime_command = next(
        command
        for command in commands.commands
        if command[1:3] == ["-m", "risk_score.build_live_runtime"]
    )
    assert result["selected_evaluator_processes"] == 16
    assert flag(runtime_command, "--evaluator-process-count") == "16"


def test_all_pass_activation_is_exactly_once_and_topology_is_applied(tmp_path):
    fixture = make_fixture(tmp_path)
    commands = RecordingRunner(fixture)
    orchestrator = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=commands
    )

    first = orchestrator.run_once()
    second = orchestrator.run_once()

    assert first["state"] == second["state"] == "active"
    assert first["selected_evaluator_processes"] == 8
    status_path = fixture["state_root"] / "status.json"
    status_value = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_value == first
    assert status_value["status_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in status_value.items()
            if key != "status_sha256"
        }
    )
    assert command_roles(commands.commands).count("activation") == 1
    runtime_command = next(
        command
        for command in commands.commands
        if command[1:3] == ["-m", "risk_score.build_live_runtime"]
    )
    assert flag(runtime_command, "--evaluator-process-count") == "8"
    gpu = json.loads(
        (
            fixture["state_root"] / "generated-runtime" / "gpu-lease-runtime.json"
        ).read_text(encoding="utf-8")
    )
    assert gpu["evaluator"]["processCount"] == 8
    assert set(
        json.loads(
            (fixture["state_root"] / "activation.json").read_text(encoding="utf-8")
        )["active"]
    ) == set(bootstrap.REQUIRED_ACTIVATION_UNITS)


def test_gate_receipt_binds_spec_inputs_outputs_and_receipt_hash(tmp_path):
    fixture = make_fixture(tmp_path)
    commands = RecordingRunner(fixture)
    orchestrator = bootstrap.AutonomyBootstrap(
        fixture["spec_path"], command_runner=commands
    )
    orchestrator.run_once()

    receipt = json.loads(orchestrator.gate_receipt_path(0).read_text(encoding="utf-8"))
    payload = dict(receipt)
    assert payload.pop("receipt_sha256") == canonical_sha256(payload)
    assert (
        receipt["spec"]["identity"]
        == bootstrap.load_bootstrap_spec(fixture["spec_path"]).identity
    )
    assert receipt["input_set_sha256"] == canonical_sha256(receipt["inputs"])
    assert receipt["output_set_sha256"] == canonical_sha256(receipt["outputs"])
    assert receipt["evidence"] in receipt["outputs"]
    assert all(
        file_sha256(Path(record["path"])) == record["sha256"]
        for record in [*receipt["inputs"], *receipt["outputs"]]
    )


def test_bootstrap_spec_file_hash_is_pinned(tmp_path):
    fixture = make_fixture(tmp_path)

    with pytest.raises(bootstrap.BootstrapError, match="installed hash"):
        bootstrap.AutonomyBootstrap(
            fixture["spec_path"],
            expected_spec_sha256="0" * 64,
        )


def test_root_bootstrap_unit_is_separate_and_hash_pinned(tmp_path):
    fixture = make_fixture(tmp_path)
    destination = (
        tmp_path / "generated" / bootstrap.BOOTSTRAP_UNIT_NAME
    ).resolve()

    record = bootstrap.publish_bootstrap_systemd_unit(
        destination,
        python_executable=Path(sys.executable).resolve(),
        working_directory=Path(__file__).resolve().parents[1],
        spec_path=fixture["spec_path"],
        run_root=fixture["state_root"].parents[1],
    )

    unit = destination.read_text(encoding="utf-8")
    assert f"Before={TARGET_UNIT}" not in unit
    assert "synchronously restarts that target" in unit
    assert "PartOf=" not in unit
    assert "User=" not in unit
    assert "--expected-spec-sha256" in unit
    assert file_sha256(fixture["spec_path"]) in unit
    assert record == {
        "path": str(destination),
        "sha256": file_sha256(destination),
    }
