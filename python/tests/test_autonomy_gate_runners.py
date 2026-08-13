import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score import autonomy_gate_runners as runners
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def self_hashed(value, field):
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def assert_canonical_evidence(path, value):
    assert path.read_bytes() == (canonical_json(value) + "\n").encode("utf-8")
    payload = dict(value)
    assert payload.pop("evidence_sha256") == canonical_sha256(payload)


def make_readiness_artifacts(tmp_path, *, status_state="complete"):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    manifest_path = suite_dir / "manifest.json"
    manifest = self_hashed(
        {
            "schemaVersion": 3,
            "manifestContract":
                "risk-score-authoritative-evaluation-manifest-v3",
            "machineReviewOnly": True,
            "banks": [],
            "cells": [],
        },
        "manifestPayloadSha256",
    )
    write_canonical(manifest_path, manifest)
    complete = status_state == "complete"
    status_path = tmp_path / "curation-status.json"
    status = self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-curation-pipeline-status-v1",
            "state": status_state,
            "error": None,
            "artifacts": {
                "reviewed_bank": {"complete": complete},
                "suite": {
                    "complete": complete,
                    "path": str(suite_dir),
                },
            },
        },
        "status_sha256",
    )
    write_canonical(status_path, status)
    return status_path, manifest_path


def test_curation_readiness_publishes_verified_pass(tmp_path):
    status, manifest = make_readiness_artifacts(tmp_path)
    output = tmp_path / "gate-work" / "readiness.json"

    evidence = runners.curation_suite_readiness(
        curation_status=status,
        suite_manifest=manifest,
        output=output,
    )

    assert evidence["decision"] == "PASS"
    assert evidence["checks"] == {
        "curation_complete": True,
        "suite_complete": True,
    }
    assert_canonical_evidence(output, evidence)


def test_curation_readiness_runs_transitive_suite_validator(tmp_path):
    status, manifest = make_readiness_artifacts(tmp_path)
    registry = tmp_path / "suite-registry.json"
    write_canonical(registry, {"contract": "test-registry"})
    observed = []

    def validate(manifest_path, registry_path):
        observed.append((manifest_path, registry_path))
        return {"validated": True}

    evidence = runners.curation_suite_readiness(
        curation_status=status,
        suite_manifest=manifest,
        suite_registry_spec=registry,
        output=tmp_path / "gate-work" / "readiness.json",
        suite_validator=validate,
    )

    assert evidence["decision"] == "PASS"
    assert observed == [(manifest, registry)]


@pytest.mark.parametrize(
    ("missing", "expected"),
    (
        ("status", {"curation_complete": False, "suite_complete": True}),
        ("suite", {"curation_complete": False, "suite_complete": False}),
    ),
)
def test_curation_readiness_honestly_waits_for_missing_artifacts(
    tmp_path, missing, expected
):
    status, manifest = make_readiness_artifacts(
        tmp_path,
        status_state="build_suite",
    )
    if missing == "status":
        status.unlink()
    else:
        manifest.unlink()
    output = tmp_path / "readiness.json"

    evidence = runners.curation_suite_readiness(
        curation_status=status,
        suite_manifest=manifest,
        output=output,
    )

    assert evidence["decision"] == "WAIT"
    assert evidence["checks"] == expected
    assert_canonical_evidence(output, evidence)


def test_curation_readiness_rejects_forged_or_noncanonical_artifacts(tmp_path):
    status, manifest = make_readiness_artifacts(tmp_path)
    suite = json.loads(manifest.read_text(encoding="utf-8"))
    suite["machineReviewOnly"] = False
    write_canonical(manifest, suite)

    with pytest.raises(runners.GateRunnerError, match="authoritative"):
        runners.curation_suite_readiness(
            curation_status=status,
            suite_manifest=manifest,
            output=tmp_path / "readiness.json",
        )

    write_canonical(manifest, self_hashed(
        {
            "schemaVersion": 3,
            "manifestContract":
                "risk-score-authoritative-evaluation-manifest-v3",
            "machineReviewOnly": True,
        },
        "manifestPayloadSha256",
    ))
    manifest.write_text(
        json.dumps(json.loads(manifest.read_text()), indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runners.GateRunnerError, match="canonical"):
        runners.curation_suite_readiness(
            curation_status=status,
            suite_manifest=manifest,
            output=tmp_path / "readiness.json",
        )


def make_deployment(tmp_path):
    artifact = tmp_path / "deployed.py"
    artifact.write_text("VALUE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "deployment.json"
    manifest = self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-live-runtime-deployment-v1",
            "source_revision": "a" * 40,
            "source_sha256": "b" * 64,
            "files": {
                "deployed": {
                    "path": str(artifact),
                    "sha256": file_sha256(artifact),
                }
            },
        },
        "manifest_sha256",
    )
    write_canonical(manifest_path, manifest)
    return artifact, manifest_path


def test_deployment_validation_uses_live_file_hashes(tmp_path):
    _, manifest = make_deployment(tmp_path)
    output = tmp_path / "deployment-evidence.json"

    evidence = runners.deployment_hash_validation(
        manifest=manifest,
        output=output,
    )

    assert evidence["decision"] == "PASS"
    assert evidence["checks"] == {"deployment_valid": True}
    assert_canonical_evidence(output, evidence)


def test_deployment_validation_rejects_changed_artifact_without_evidence(tmp_path):
    artifact, manifest = make_deployment(tmp_path)
    artifact.write_text("VALUE = 2\n", encoding="utf-8")
    output = tmp_path / "deployment-evidence.json"

    with pytest.raises(Exception, match="changed"):
        runners.deployment_hash_validation(
            manifest=manifest,
            output=output,
        )
    assert not output.exists()


def deterministic_config(path):
    path.write_text(
        "\n".join(
            (
                "forDeterministicTesting = true",
                "numAnalysisThreads = 1",
                "nnRandomize = false",
                "rootNoiseEnabled = false",
                "rootNumSymmetriesToSample = 1",
                "useUncertainty = false",
                "cpuctUtilityStdevScale = 0",
                "reportAnalysisWinratesAs = SIDETOMOVE",
            )
        )
        + "\n",
        encoding="utf-8",
    )


class FakeCudaRunner:
    def __init__(self, *, busy=False, nondeterministic=False):
        self.busy = busy
        self.nondeterministic = nondeterministic
        self.analysis_calls = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        if argv[0] == "nvidia-smi":
            assert kwargs == {
                "check": False,
                "capture_output": True,
                "text": True,
                "shell": False,
            }
            if argv[1] == "--query-gpu=uuid":
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    "GPU-other\nGPU-target\n",
                    "",
                )
            assert argv[1].startswith("--query-compute-apps=")
            stdout = "GPU-target, 42, python\n" if self.busy else ""
            return subprocess.CompletedProcess(argv, 0, stdout, "")

        assert argv[1] == "analysis"
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        queries = [
            json.loads(line)
            for line in kwargs["stdin"].read().decode("utf-8").splitlines()
        ]
        assert len(queries) == 1
        assert queries[0]["maxVisits"] == 4
        model = argv[argv.index("-model") + 1]
        model_call = sum(
            prior_model == model for prior_model, _ in self.analysis_calls
        )
        score = (
            float(model_call)
            if self.nondeterministic
            else 0.0
        )
        response = {
            "id": queries[0]["id"],
            "rootInfo": {
                "winrate": 0.5,
                "scoreLead": score,
                "utility": 0.0,
                "visits": 4,
            },
            "moveInfos": [{"move": "D4", "visits": 4}],
        }
        kwargs["stdout"].write(
            (canonical_json(response) + "\n").encode("utf-8")
        )
        self.analysis_calls.append((model, queries[0]))
        return SimpleNamespace(returncode=0, stderr=b"")


def make_probe_inputs(tmp_path):
    katago = tmp_path / "katago"
    katago.write_bytes(b"fake-katago")
    config = tmp_path / "analysis.cfg"
    deterministic_config(config)
    models = []
    for name, contents in (
        ("model=one.bin.gz", b"model-one"),
        ("model-two.bin.gz", b"model-two"),
    ):
        model = tmp_path / name
        model.write_bytes(contents)
        models.append(model)
    return katago, config, models


def test_cuda_gate_probes_each_model_twice_without_real_cuda(tmp_path):
    katago, config, models = make_probe_inputs(tmp_path)
    fake = FakeCudaRunner()
    output = tmp_path / "cuda-evidence.json"
    bindings = [f"{model}={file_sha256(model)}" for model in reversed(models)]

    evidence = runners.cuda_model_probes(
        katago=katago,
        config=config,
        gpu_index=1,
        expected_gpu_uuid="GPU-target",
        model_bindings=bindings,
        output=output,
        subprocess_runner=fake,
    )

    assert len(fake.analysis_calls) == 4
    assert evidence["decision"] == "PASS"
    assert evidence["checks"]["cuda_available"] is True
    assert evidence["checks"]["gpu_exclusive"] is True
    assert evidence["checks"]["gpu_uuid"] == "GPU-target"
    assert evidence["checks"]["models"] == [
        {
            "model_sha256": digest,
            "finite": True,
            "deterministic": True,
        }
        for digest in sorted(file_sha256(model) for model in models)
    ]
    assert_canonical_evidence(output, evidence)


def test_cuda_gate_rejects_busy_gpu_before_analysis(tmp_path):
    katago, config, models = make_probe_inputs(tmp_path)
    fake = FakeCudaRunner(busy=True)
    output = tmp_path / "cuda-evidence.json"

    with pytest.raises(runners.GateRunnerError, match="exclusively idle"):
        runners.cuda_model_probes(
            katago=katago,
            config=config,
            gpu_index=1,
            expected_gpu_uuid="GPU-target",
            model_bindings=[f"{models[0]}={file_sha256(models[0])}"],
            output=output,
            subprocess_runner=fake,
        )

    assert fake.analysis_calls == []
    assert not output.exists()


def test_cuda_gate_rejects_nondeterministic_canonical_output(tmp_path):
    katago, config, models = make_probe_inputs(tmp_path)
    fake = FakeCudaRunner(nondeterministic=True)
    output = tmp_path / "cuda-evidence.json"

    with pytest.raises(runners.GateRunnerError, match="not deterministic"):
        runners.cuda_model_probes(
            katago=katago,
            config=config,
            gpu_index=1,
            expected_gpu_uuid="GPU-target",
            model_bindings=[f"{models[0]}={file_sha256(models[0])}"],
            output=output,
            subprocess_runner=fake,
        )

    assert len(fake.analysis_calls) == 2
    assert not output.exists()


def test_cuda_gate_rejects_unbound_or_duplicate_models(tmp_path):
    katago, config, models = make_probe_inputs(tmp_path)
    with pytest.raises(runners.GateRunnerError, match="hash mismatch"):
        runners.cuda_model_probes(
            katago=katago,
            config=config,
            gpu_index=1,
            expected_gpu_uuid="GPU-target",
            model_bindings=[f"{models[0]}={'0' * 64}"],
            output=tmp_path / "cuda-evidence.json",
            subprocess_runner=FakeCudaRunner(),
        )
    binding = f"{models[0]}={file_sha256(models[0])}"
    with pytest.raises(runners.GateRunnerError, match="paths must be unique"):
        runners.cuda_model_probes(
            katago=katago,
            config=config,
            gpu_index=1,
            expected_gpu_uuid="GPU-target",
            model_bindings=[binding, binding],
            output=tmp_path / "cuda-evidence.json",
            subprocess_runner=FakeCudaRunner(),
        )


def make_backlog_artifacts(tmp_path, *, fail_closed=True):
    inbox = tmp_path / "inbox"
    candidate = inbox / "net-s500000-d1000000"
    candidate.mkdir(parents=True)
    (candidate / "model.bin.gz").write_bytes(b"model")
    (candidate / "model.ckpt").write_bytes(b"checkpoint")
    (inbox / ".uploading").mkdir()

    policy_hash = "9" * 64
    controller_hash = "7" * 64
    claimed_hash = "a" * 64
    evaluating_hash = "b" * 64
    result = {
        "mode": "automatic",
        "candidates": [
            {"hash": claimed_hash, "state": "claimed", "present": True},
            {
                "hash": evaluating_hash,
                "state": "evaluating_screen",
                "present": True,
            },
        ],
        "activeEvaluations": [
            {
                "candidateHash": evaluating_hash,
                "stage": "screen",
            }
        ],
        "queue": {
            "depth": 3,
            "pendingDepth": 2,
            "readyDepth": 1,
            "ignoredDepth": 1,
            "activeDepth": 1,
        },
        "backpressure": {
            "exportBacklogDepth": 2,
            "evaluationBacklogDepth": 2,
            "maximumActiveEvaluatorEntries": 3,
            "importantQueueWarningDepth": 4,
            "allowExport": not fail_closed,
            "allowEvaluation": not fail_closed,
            "exportPaused": fail_closed,
            "evaluationPaused": fail_closed,
            "reasons": ["bootstrap-pre-controller"],
        },
    }
    status = {
        "schema_version": 1,
        "contract": "risk-score-controller-status-v1",
        "observed_at_utc": "2026-08-12T00:00:00Z",
        "controller_actor": "controller",
        "controller_hash": controller_hash,
        "source_revision_hash": "8" * 64,
        "policy_hash": policy_hash,
        "result": result,
    }
    status_path = tmp_path / "controller-status.json"
    write_canonical(status_path, status)

    backpressure = {
        "schema_version": 1,
        "updated_at_utc": "2026-08-12T00:00:00Z",
        "controller_hash": controller_hash,
        "policy_hash": policy_hash,
        "allowExport": not fail_closed,
        "allowEvaluation": not fail_closed,
        "exportPaused": fail_closed,
        "evaluationPaused": fail_closed,
        "exportBacklogDepth": 2,
        "evaluationBacklogDepth": 2,
        "maximumActiveEvaluatorEntries": 3,
        "importantQueueWarningDepth": 4,
        "reasons": ["bootstrap-pre-controller"],
    }
    backpressure_path = tmp_path / "backpressure.json"
    write_canonical(backpressure_path, backpressure)
    return inbox, status_path, backpressure_path


def test_backlog_gate_publishes_actual_stable_counts(tmp_path):
    inbox, status, backpressure = make_backlog_artifacts(tmp_path)
    output = tmp_path / "backlog-evidence.json"

    evidence = runners.backlog_bound(
        inbox=inbox,
        controller_status=status,
        backpressure=backpressure,
        maximum_candidates=2,
        maximum_active_queue=3,
        output=output,
    )

    assert evidence["decision"] == "PASS"
    assert evidence["checks"] == {
        "candidate_count": 2,
        "maximum_candidates": 2,
        "active_queue_depth": 2,
        "maximum_active_queue": 3,
        "backpressure_enforced": True,
        "evidence_preserved": True,
    }
    assert_canonical_evidence(output, evidence)


def test_backlog_gate_publishes_fail_for_real_bound_violation(tmp_path):
    inbox, status, backpressure = make_backlog_artifacts(tmp_path)
    output = tmp_path / "backlog-evidence.json"

    evidence = runners.backlog_bound(
        inbox=inbox,
        controller_status=status,
        backpressure=backpressure,
        maximum_candidates=1,
        maximum_active_queue=3,
        output=output,
    )

    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["candidate_count"] == 2
    assert evidence["checks"]["maximum_candidates"] == 1
    assert_canonical_evidence(output, evidence)


def test_backlog_gate_does_not_claim_allowing_backpressure_is_enforced(tmp_path):
    inbox, status, backpressure = make_backlog_artifacts(
        tmp_path,
        fail_closed=False,
    )
    evidence = runners.backlog_bound(
        inbox=inbox,
        controller_status=status,
        backpressure=backpressure,
        maximum_candidates=2,
        maximum_active_queue=3,
        output=tmp_path / "backlog-evidence.json",
    )

    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["backpressure_enforced"] is False


def test_backlog_gate_rejects_controller_and_artifact_disagreement(tmp_path):
    inbox, status, backpressure = make_backlog_artifacts(tmp_path)
    value = json.loads(status.read_text(encoding="utf-8"))
    value["result"]["backpressure"]["allowExport"] = True
    write_canonical(status, value)

    evidence = runners.backlog_bound(
        inbox=inbox,
        controller_status=status,
        backpressure=backpressure,
        maximum_candidates=2,
        maximum_active_queue=3,
        output=tmp_path / "backlog-evidence.json",
    )

    assert evidence["decision"] == "FAIL"
    assert evidence["checks"]["backpressure_enforced"] is False


def test_backlog_gate_supports_fail_closed_precontroller_state(tmp_path):
    inbox, _status, backpressure = make_backlog_artifacts(tmp_path)

    def systemctl(argv, **kwargs):
        assert argv == [
            "systemctl",
            "is-active",
            "katago-risk-training.target",
        ]
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, 3, stdout="inactive\n", stderr="")

    evidence = runners.backlog_bound(
        inbox=inbox,
        controller_status=None,
        backpressure=backpressure,
        maximum_candidates=2,
        maximum_active_queue=3,
        output=tmp_path / "backlog-evidence.json",
        subprocess_runner=systemctl,
    )

    assert evidence["decision"] == "PASS"
    assert evidence["checks"]["active_queue_depth"] == 0


def test_backlog_gate_rejects_stale_controller_queue_counts(tmp_path):
    inbox, status, backpressure = make_backlog_artifacts(tmp_path)
    value = json.loads(status.read_text(encoding="utf-8"))
    value["result"]["queue"]["pendingDepth"] = 1
    write_canonical(status, value)

    with pytest.raises(runners.GateRunnerError, match="does not match"):
        runners.backlog_bound(
            inbox=inbox,
            controller_status=status,
            backpressure=backpressure,
            maximum_candidates=2,
            maximum_active_queue=3,
            output=tmp_path / "backlog-evidence.json",
        )


def test_all_paths_are_absolute_and_symlink_free(tmp_path):
    status, manifest = make_readiness_artifacts(tmp_path)
    with pytest.raises(runners.GateRunnerError, match="absolute"):
        runners.curation_suite_readiness(
            curation_status=Path("relative-status.json"),
            suite_manifest=manifest,
            output=tmp_path / "readiness.json",
        )

    linked = tmp_path / "linked-status.json"
    linked.symlink_to(status)
    with pytest.raises(runners.GateRunnerError, match="symlinked"):
        runners.curation_suite_readiness(
            curation_status=linked,
            suite_manifest=manifest,
            output=tmp_path / "readiness.json",
        )


def test_main_returns_two_for_argument_and_verification_errors(tmp_path, capsys):
    assert runners.main(["deployment-hash-validation"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["type"] == "GateRunnerError"

    _, manifest = make_deployment(tmp_path)
    assert runners.main(
        [
            "deployment-hash-validation",
            "--manifest",
            str(manifest),
            "--output",
            "relative-output.json",
        ]
    ) == 2
    assert not (Path.cwd() / "relative-output.json").exists()


def test_cuda_cli_accepts_repeated_model_bindings(tmp_path, capsys):
    katago, config, models = make_probe_inputs(tmp_path)
    output = tmp_path / "cuda-evidence.json"
    argv = [
        "cuda-model-probes",
        "--katago",
        str(katago),
        "--config",
        str(config),
        "--gpu-index",
        "1",
        "--expected-gpu-uuid",
        "GPU-target",
        "--output",
        str(output),
    ]
    for model in models:
        argv.extend(["--model", f"{model}={file_sha256(model)}"])

    assert runners.main(argv, subprocess_runner=FakeCudaRunner()) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text(encoding="utf-8"))
