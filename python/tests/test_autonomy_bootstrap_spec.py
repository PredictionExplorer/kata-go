import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from risk_score import autonomy_bootstrap
from risk_score import autonomy_bootstrap_spec as publisher
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def binding(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def flag(argv, name):
    return argv[argv.index(name) + 1]


def git(repo, *arguments):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def initialize_repository(repo):
    repo.mkdir()
    git(repo, "init", "-q")


def commit_repository(repo):
    git(repo, "add", ".")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Bootstrap Test",
            "-c",
            "user.email=bootstrap@example.invalid",
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
    assert result.returncode == 0, result.stderr
    return git(repo, "rev-parse", "HEAD")


def gate_templates(extra_specs, models):
    return {
        "curation-suite-readiness": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "curation-suite-readiness",
            "--curation-status",
            "{curation_status}",
            "--suite-manifest",
            "{suite_manifest}",
            "--suite-registry-spec",
            str(extra_specs["suite_registry"]),
            "--output",
            "{evidence}",
        ],
        "filesystem-rename-fsync": [
            "{python}",
            "-m",
            "risk_score.promotion_preflight",
            "filesystem-test",
            "--root",
            "{run_root}",
            "--output",
            "{evidence}",
        ],
        "deployment-hash-validation": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "deployment-hash-validation",
            "--manifest",
            "{deployment_manifest}",
            "--output",
            "{evidence}",
        ],
        "candidate-inventory": [
            "{python}",
            "-m",
            "risk_score.promotion_preflight",
            "candidate-inventory",
            "--inbox",
            "{candidate_inbox}",
            "--output",
            "{evidence}",
        ],
        "cuda-model-probes": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "cuda-model-probes",
            "--katago",
            "{katago_binary}",
            "--config",
            "{model_probe_config}",
            "--gpu-index",
            "7",
            "--expected-gpu-uuid",
            "GPU-production-7",
            *[
                argument
                for model in models
                for argument in (
                    "--model",
                    f"{model['path']}={model['sha256']}",
                )
            ],
            "--output",
            "{evidence}",
        ],
        "evaluator-topology-benchmark": [
            "{python}",
            "-m",
            "risk_score.evaluator_topology_benchmark",
            "--spec",
            str(extra_specs["topology"]),
            "--expected-spec-sha256",
            file_sha256(extra_specs["topology"]),
        ],
        "trainer-evaluator-lease-drill": [
            "{python}",
            "-m",
            "risk_score.autonomy_lease_drill",
            "--spec",
            str(extra_specs["lease"]),
            "--expected-spec-sha256",
            file_sha256(extra_specs["lease"]),
        ],
        "disposable-canary-drill": [
            "{python}",
            "-m",
            "risk_score.autonomy_promotion_drills",
            "disposable-canary-drill",
            "--spec",
            str(extra_specs["promotion"]),
        ],
        "crash-replay-drill": [
            "{python}",
            "-m",
            "risk_score.autonomy_promotion_drills",
            "crash-replay-drill",
            "--spec",
            str(extra_specs["promotion"]),
        ],
        "rollback-before-admission-drill": [
            "{python}",
            "-m",
            "risk_score.autonomy_promotion_drills",
            "rollback-before-admission-drill",
            "--spec",
            str(extra_specs["promotion"]),
        ],
        "rollback-after-admission-drill": [
            "{python}",
            "-m",
            "risk_score.autonomy_promotion_drills",
            "rollback-after-admission-drill",
            "--spec",
            str(extra_specs["promotion"]),
        ],
        "shadow-controller-replay": [
            "{python}",
            "-m",
            "risk_score.autonomy_promotion_drills",
            "shadow-controller-replay",
            "--spec",
            str(extra_specs["promotion"]),
        ],
        "backlog-bound": [
            "{python}",
            "-m",
            "risk_score.autonomy_gate_runners",
            "backlog-bound",
            "--candidate-inbox",
            "{candidate_inbox}",
            "--controller-status",
            str(extra_specs["controller_status"]),
            "--backpressure",
            str(extra_specs["backpressure"]),
            "--maximum-candidates",
            "20",
            "--maximum-active-queue",
            "4",
            "--output",
            "{evidence}",
        ],
    }


def make_fixture(tmp_path, *, suite_present=False):
    repo = tmp_path / "deployed"
    initialize_repository(repo)
    package = repo / "python" / "risk_score"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    fixture_runtime_modules = {
        "risk_score.shuffler",
        "risk_score.exporter",
        "risk_score.evaluator",
        "risk_score.cluster_executor",
        "risk_score.adaptive_training",
        "risk_score.suite_rotation",
    }
    for module in sorted(publisher._CONTROL_MODULES | fixture_runtime_modules):
        relative = module.removeprefix("risk_score.").replace(".", "/") + ".py"
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'MODULE = "{module}"\n', encoding="utf-8")

    executable = Path(sys.executable).resolve()
    file_paths = {}
    for name in sorted(publisher._REQUIRED_FILES.difference({"python_executable"})):
        suffix = ".bin" if name in {"katago_binary", "original_model"} else ".json"
        path = repo / "inputs" / f"{name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".json":
            write_canonical(path, {"name": name})
        else:
            path.write_bytes((name + "\n").encode("utf-8"))
        file_paths[name] = path
    os.chmod(file_paths["katago_binary"], 0o755)

    run_root = tmp_path / "run"
    candidate_inbox = run_root / "modelstobetested"
    suite_dir = run_root / "evaluation" / "promotion-suites-v3"
    curation_dir = run_root / "evaluation" / "curation"
    activation = tmp_path / "systemd"
    for directory in (
        run_root,
        candidate_inbox,
        suite_dir,
        curation_dir,
        activation,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    state_root = run_root / "promotion" / "autonomy-bootstrap"
    state_root.parent.mkdir(parents=True, exist_ok=True)
    gate_evidence = {
        gate_id: state_root / "gate-evidence" / f"{gate_id}.json"
        for gate_id in autonomy_bootstrap.GATE_ORDER
    }
    extra_specs = {}
    for name, gate_id in (
        ("topology", "evaluator-topology-benchmark"),
        ("lease", "trainer-evaluator-lease-drill"),
        ("promotion", "disposable-canary-drill"),
    ):
        path = repo / "gate-specs" / f"{name}.json"
        value = {"name": name}
        if name in {"topology", "lease"}:
            value.update(
                {
                    "schema_version": 1,
                    "contract": (
                        "risk-score-evaluator-topology-benchmark-spec-v1"
                        if name == "topology"
                        else "risk-score-autonomy-lease-drill-spec-v1"
                    ),
                    "evidence_output": str(gate_evidence[gate_id]),
                }
            )
            value["spec_sha256"] = canonical_sha256(value)
        else:
            value.update(
                {
                    "schema_version": 1,
                    "contract": ("risk-score-autonomy-promotion-drill-spec-v1"),
                    "evidence_root": str(gate_evidence[gate_id].parent),
                }
            )
            value["spec_sha256"] = canonical_sha256(value)
        write_canonical(path, value)
        extra_specs[name] = path
    extra_specs["suite_registry"] = file_paths["suite_registry_spec"]
    for name in ("controller_status", "backpressure"):
        path = repo / "gate-inputs" / f"{name}.json"
        write_canonical(path, {"name": name})
        extra_specs[name] = path
    curation_status = curation_dir / "status.json"
    suite_manifest = suite_dir / "manifest.json"
    if suite_present:
        suite = {
            "schemaVersion": 3,
            "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
            "machineReviewOnly": True,
            "banks": [],
            "cells": [],
        }
        suite["manifestPayloadSha256"] = canonical_sha256(suite)
        write_canonical(suite_manifest, suite)
        status = {
            "schema_version": 1,
            "contract": "risk-score-curation-pipeline-status-v1",
            "state": "complete",
            "error": None,
            "artifacts": {
                "reviewed_bank": {"complete": True},
                "suite": {"complete": True, "path": str(suite_dir)},
            },
        }
        status["status_sha256"] = canonical_sha256(status)
        write_canonical(curation_status, status)

    source_revision = commit_repository(repo)
    model_paths = sorted([file_paths["original_model"]], key=str)
    model_bindings = [binding(path) for path in model_paths]
    files = {
        "python_executable": binding(executable),
        **{name: binding(path) for name, path in file_paths.items()},
    }
    runtime_commands = {
        "shuffler": [str(executable), "-m", "risk_score.shuffler"],
        "exporter": [str(executable), "-m", "risk_score.exporter"],
        "evaluator": [str(executable), "-m", "risk_score.evaluator"],
        "cluster_executor": [
            str(executable),
            "-m",
            "risk_score.cluster_executor",
            "--spec",
            str(file_paths["cluster_executor_spec"]),
        ],
        "adaptive_training": [
            str(executable),
            "-m",
            "risk_score.adaptive_training",
            "--spec",
            str(file_paths["adaptive_training_spec"]),
        ],
        "suite_rotation": [
            str(executable),
            "-m",
            "risk_score.suite_rotation",
            "--spec",
            str(file_paths["suite_registry_spec"]),
        ],
    }
    value = {
        "schema_version": 1,
        "contract": publisher.PUBLISHER_SPEC_CONTRACT,
        "repository": str(repo),
        "source_revision": source_revision,
        "run_root": str(run_root),
        "state_root": str(state_root),
        "curation_status": {
            "path": str(curation_status),
            "sha256": (
                file_sha256(curation_status) if curation_status.is_file() else None
            ),
        },
        "suite_manifest": {
            "path": str(suite_manifest),
            "sha256": (
                file_sha256(suite_manifest) if suite_manifest.is_file() else None
            ),
        },
        "candidate_inbox": str(candidate_inbox),
        "activation_destination": str(activation),
        "files": files,
        "models": model_bindings,
        "extra_inputs": [
            binding(path) for path in sorted(extra_specs.values(), key=str)
        ],
        "gate_argv_templates": gate_templates(extra_specs, model_bindings),
        "runtime_commands": runtime_commands,
        "gpu": {"index": 7, "uuid": "GPU-production-7"},
        "actor": "autonomy-bootstrap",
        "service_user": "katago",
        "poll_interval_seconds": 5,
        "minimum_clean_observations": 3,
        "maximum_candidates": 20,
        "maximum_active_queue": 4,
        "outputs": {
            "bootstrap_spec": str(state_root / "bootstrap-spec.json"),
            "bootstrap_service_unit": str(
                activation / autonomy_bootstrap.BOOTSTRAP_UNIT_NAME
            ),
            "bootstrap_path_unit": str(activation / publisher.BOOTSTRAP_PATH_UNIT_NAME),
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    input_spec = tmp_path / "publisher-spec.json"
    write_canonical(input_spec, value)
    return {
        "repo": repo,
        "source_revision": source_revision,
        "run_root": run_root,
        "state_root": state_root,
        "suite_manifest": suite_manifest,
        "curation_status": curation_status,
        "activation": activation,
        "input_spec": input_spec,
        "input_value": value,
        "bootstrap_spec": Path(value["outputs"]["bootstrap_spec"]),
        "service_unit": Path(value["outputs"]["bootstrap_service_unit"]),
        "path_unit": Path(value["outputs"]["bootstrap_path_unit"]),
    }


def rewrite_input(fixture, mutate):
    value = json.loads(fixture["input_spec"].read_text(encoding="utf-8"))
    value.pop("spec_sha256")
    mutate(value)
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(fixture["input_spec"], value)
    fixture["input_value"] = value


def test_missing_suite_materializes_wait_safe_spec_and_independent_path_unit(tmp_path):
    fixture = make_fixture(tmp_path)

    first = publisher.materialize_bootstrap(fixture["input_spec"])
    spec = autonomy_bootstrap.BootstrapSpec.load(fixture["bootstrap_spec"])

    assert first["suite_manifest_present"] is False
    assert first["enable_unit"] == publisher.BOOTSTRAP_PATH_UNIT_NAME
    assert [gate.gate_id for gate in spec.gates] == list(autonomy_bootstrap.GATE_ORDER)
    readiness = spec.gates[0]
    readiness_by_path = {item.path: item.expected_sha256 for item in readiness.inputs}
    assert readiness_by_path[fixture["curation_status"]] is None
    assert readiness_by_path[fixture["suite_manifest"]] is None
    assert all(
        binding.expected_sha256 is not None
        for gate in spec.gates[1:]
        for binding in gate.inputs
    )
    trainer_checkpoint = Path(
        fixture["input_value"]["files"]["trainer_checkpoint"]["path"]
    )
    assert all(
        trainer_checkpoint not in {binding.path for binding in gate.inputs}
        for gate in spec.gates
    )

    evidence_paths = [gate.evidence for gate in spec.gates]
    assert len(set(evidence_paths)) == len(autonomy_bootstrap.GATE_ORDER)
    assert all(path.is_relative_to(fixture["state_root"]) for path in evidence_paths)
    assert (fixture["state_root"] / "gate-evidence").is_dir()
    output_paths = [path for gate in spec.gates for path in gate.outputs]
    assert len(output_paths) == len(set(output_paths))
    assert all(path.is_relative_to(fixture["state_root"]) for path in output_paths)
    assert all(gate.evidence in gate.outputs for gate in spec.gates)

    modules = [tuple(gate.argv[1:3]) for gate in spec.gates]
    assert modules == [
        ("-m", publisher._EXPECTED_GATE_MODULES[gate_id])
        for gate_id in autonomy_bootstrap.GATE_ORDER
    ]
    runtime = list(spec.runtime.argv)
    assert "--mutation-enabled" in runtime
    assert "--full-autonomy" in runtime
    assert flag(runtime, "--evaluator-process-count") == (
        "{selected_evaluator_processes}"
    )
    for command_flag in (
        "--shuffler-command-json",
        "--exporter-command-json",
        "--evaluator-command-json",
        "--cluster-executor-command-json",
        "--adaptive-training-command-json",
        "--suite-rotation-command-json",
    ):
        command = json.loads(flag(runtime, command_flag))
        assert Path(command[0]).is_absolute()
    for input_flag in (
        "--autonomy-policy",
        "--cluster-executor-spec",
        "--adaptive-training-spec",
        "--suite-registry-spec",
    ):
        assert Path(flag(runtime, input_flag)).is_file()

    assert spec.activation.required_units == (
        autonomy_bootstrap.REQUIRED_ACTIVATION_UNITS
    )
    assert spec.activation.argv[1:3] == ("-m", "risk_score.service_activation")
    assert "--apply" in spec.activation.argv

    service = fixture["service_unit"].read_text(encoding="utf-8")
    assert service == autonomy_bootstrap.render_bootstrap_systemd_unit(
        python_executable=Path(sys.executable).resolve(),
        working_directory=fixture["repo"] / "python",
        spec_path=fixture["bootstrap_spec"],
        run_root=fixture["run_root"],
    )
    assert "--expected-spec-sha256" in service
    assert file_sha256(fixture["bootstrap_spec"]) in service
    assert "PartOf=" not in service

    path_unit = fixture["path_unit"].read_text(encoding="utf-8")
    assert f'PathExists="{fixture["suite_manifest"]}"' in path_unit
    assert f"Unit={autonomy_bootstrap.BOOTSTRAP_UNIT_NAME}" in path_unit
    assert "katago-risk-training.target" not in path_unit
    assert "WantedBy=multi-user.target" in path_unit

    bootstrap_data = fixture["bootstrap_spec"].read_bytes()
    assert bootstrap_data == (canonical_json(json.loads(bootstrap_data)) + "\n").encode(
        "utf-8"
    )
    bootstrap_raw = json.loads(bootstrap_data)
    bootstrap_payload = dict(bootstrap_raw)
    assert bootstrap_payload.pop("spec_sha256") == canonical_sha256(bootstrap_payload)
    publication_payload = dict(first)
    assert publication_payload.pop("publication_sha256") == canonical_sha256(
        publication_payload
    )

    inodes = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (
            fixture["bootstrap_spec"],
            fixture["service_unit"],
            fixture["path_unit"],
        )
    }
    assert publisher.materialize_bootstrap(fixture["input_spec"]) == first
    assert {
        path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in inodes
    } == inodes


def test_present_authoritative_suite_is_hash_bound(tmp_path):
    fixture = make_fixture(tmp_path, suite_present=True)

    result = publisher.materialize_bootstrap(fixture["input_spec"])
    loaded = autonomy_bootstrap.BootstrapSpec.load(fixture["bootstrap_spec"])
    readiness = {item.path: item.expected_sha256 for item in loaded.gates[0].inputs}

    assert result["suite_manifest_present"] is True
    assert readiness[fixture["curation_status"]] == file_sha256(
        fixture["curation_status"]
    )
    assert readiness[fixture["suite_manifest"]] == file_sha256(
        fixture["suite_manifest"]
    )


def test_unbound_wait_spec_replays_after_suite_publication(tmp_path):
    fixture = make_fixture(tmp_path)
    first = publisher.materialize_bootstrap(fixture["input_spec"])
    artifact_hashes = {
        key: first[key]["sha256"]
        for key in (
            "bootstrap_spec",
            "bootstrap_service_unit",
            "bootstrap_path_unit",
        )
    }
    suite = {
        "schemaVersion": 3,
        "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
        "machineReviewOnly": True,
        "banks": [],
        "cells": [],
    }
    suite["manifestPayloadSha256"] = canonical_sha256(suite)
    write_canonical(fixture["suite_manifest"], suite)
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

    replay = publisher.materialize_bootstrap(fixture["input_spec"])
    loaded = autonomy_bootstrap.BootstrapSpec.load(fixture["bootstrap_spec"])

    assert replay["suite_manifest_present"] is True
    assert {key: replay[key]["sha256"] for key in artifact_hashes} == artifact_hashes
    readiness = {item.path: item.expected_sha256 for item in loaded.gates[0].inputs}
    assert readiness[fixture["curation_status"]] is None
    assert readiness[fixture["suite_manifest"]] is None


@pytest.mark.parametrize("artifact", ["bootstrap_spec", "service_unit", "path_unit"])
def test_conflicting_immutable_artifact_is_rejected_before_publication(
    tmp_path, artifact
):
    fixture = make_fixture(tmp_path)
    conflict = fixture[artifact]
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_text("conflict\n", encoding="utf-8")

    with pytest.raises(publisher.BootstrapSpecPublicationError, match="conflicts"):
        publisher.materialize_bootstrap(fixture["input_spec"])

    assert all(
        not fixture[name].exists()
        for name in ("bootstrap_spec", "service_unit", "path_unit")
        if name != artifact
    )


def test_symlinked_output_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path)
    target = tmp_path / "unsafe-target"
    target.write_text("unsafe\n", encoding="utf-8")
    fixture["service_unit"].symlink_to(target)

    with pytest.raises(publisher.BootstrapSpecPublicationError, match="symlink"):
        publisher.materialize_bootstrap(fixture["input_spec"])
    assert not fixture["bootstrap_spec"].exists()


def test_dirty_deployed_checkout_is_rejected_without_outputs(tmp_path):
    fixture = make_fixture(tmp_path)
    source = fixture["repo"] / "python" / "risk_score" / "autonomy_bootstrap.py"
    source.write_text("DIRTY = True\n", encoding="utf-8")

    with pytest.raises(
        publisher.BootstrapSpecPublicationError, match="clean deployed checkout"
    ):
        publisher.materialize_bootstrap(fixture["input_spec"])

    assert not fixture["bootstrap_spec"].exists()
    assert not fixture["service_unit"].exists()
    assert not fixture["path_unit"].exists()


@pytest.mark.parametrize("damage", ["noncanonical", "stale-self-hash"])
def test_publisher_input_is_canonical_and_self_hashed(tmp_path, damage):
    fixture = make_fixture(tmp_path)
    value = json.loads(fixture["input_spec"].read_text(encoding="utf-8"))
    if damage == "noncanonical":
        fixture["input_spec"].write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )
        message = "canonical"
    else:
        value["actor"] = "changed-without-rehash"
        write_canonical(fixture["input_spec"], value)
        message = "self-hash"

    with pytest.raises(publisher.BootstrapSpecPublicationError, match=message):
        publisher.materialize_bootstrap(fixture["input_spec"])
    assert not fixture["bootstrap_spec"].exists()


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("{unknown}", "unsupported placeholders"),
        ("/missing-evidence", r"invalid \{evidence\} placeholder count"),
    ],
)
def test_gate_command_placeholders_fail_closed(tmp_path, replacement, message):
    fixture = make_fixture(tmp_path)

    def mutate(value):
        template = value["gate_argv_templates"]["backlog-bound"]
        template[template.index("{evidence}")] = replacement

    rewrite_input(fixture, mutate)

    with pytest.raises(publisher.BootstrapSpecPublicationError, match=message):
        publisher.materialize_bootstrap(fixture["input_spec"])
    assert not fixture["bootstrap_spec"].exists()


def test_cli_materializes_and_honors_input_file_hash(tmp_path, capsys):
    fixture = make_fixture(tmp_path)

    assert (
        publisher.main(
            [
                "materialize",
                "--spec",
                str(fixture["input_spec"]),
                "--expected-spec-sha256",
                file_sha256(fixture["input_spec"]),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["contract"] == publisher.PUBLICATION_CONTRACT
    assert autonomy_bootstrap.BootstrapSpec.load(fixture["bootstrap_spec"])

    assert (
        publisher.main(
            [
                "--spec",
                str(fixture["input_spec"]),
                "--expected-spec-sha256",
                "0" * 64,
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert "installed hash changed" in error["error"]["message"]
