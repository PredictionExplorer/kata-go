import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from risk_score.build_evaluation_suites import MACHINE_MANIFEST_CONTRACT
from risk_score.curation_pipeline import PIPELINE_SPEC_CONTRACT
from risk_score.curation_supplement import SUPPLEMENT_SPEC_CONTRACT
from risk_score.suite_rotation import (
    CONTINUITY_CONTRACT,
    PIPELINE_REQUEST_CONTRACT,
    POLICY_VERSION,
    ROTATION_REQUEST_CONTRACT,
    SUPPLEMENT_REQUEST_CONTRACT,
    canonical_json,
    canonical_sha256,
    file_sha256,
    publish_registry_spec,
)
from risk_score.suite_rotation_service import CONTINUITY_EVIDENCE_CONTRACT
from risk_score.suite_rotation_worker import (
    COMMAND_RECEIPT_CONTRACT,
    CURATION_RECEIPT_CONTRACT,
    WORKER_SPEC_CONTRACT,
    SuiteRotationWorker,
    WorkerCommandError,
    WorkerSpecError,
    WorkerStateError,
    load_worker_spec,
    publish_shadow_replay_evidence,
)


DEPLOYED_MODULES = (
    "board_symmetry.py",
    "build_evaluation_suites.py",
    "build_live_runtime.py",
    "consensus_prefilter.py",
    "curate_position_bank.py",
    "curation_orchestrator.py",
    "curation_pipeline.py",
    "curation_supplement.py",
    "gpu_lease.py",
    "position_samples.py",
    "promotion_evaluator.py",
    "suite_rotation.py",
    "suite_rotation_service.py",
    "suite_rotation_worker.py",
)
REPO = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def self_hashed(value, field):
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def binding(path, identity=None):
    result = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if identity is not None:
        result["identity"] = identity
    return result


def directory_binding(path):
    rows = [
        {
            "path": child.name,
            "size": child.stat().st_size,
            "sha256": file_sha256(child),
        }
        for child in sorted(path.iterdir(), key=lambda item: item.name)
    ]
    return {"path": str(path.resolve()), "sha256": canonical_sha256(rows)}


def fake_loader(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        raw=value,
        identity=value["spec_sha256"],
        file_sha256=file_sha256(path),
    )


def make_fixture(tmp_path, *, output_base=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = tmp_path / "policy-v3.json"
    policy.write_bytes(
        (
            Path(__file__).parents[1]
            / "risk_score"
            / "promotion_policy_v3.json"
        ).read_bytes()
    )
    models = tmp_path / "selfplay-models"
    models.mkdir()
    original = models / "model.bin.gz"
    champion = tmp_path / "champion-current.bin.gz"
    previous = tmp_path / "champion-previous.bin.gz"
    original.write_bytes(b"immutable-original")
    champion.write_bytes(b"current-champion")
    previous.write_bytes(b"previous-champion")
    registry_path = tmp_path / "registry-spec.json"
    registry = publish_registry_spec(
        registry_path,
        registry_root=tmp_path / "registry",
        policy_path=policy,
        original_model_path=original,
        initial_champion_path=champion,
        initial_generation_id="generation-5",
        created_at_utc="2026-08-11T00:00:00.000000Z",
    )

    repository = tmp_path / "deployment"
    deployed_modules = repository / "python" / "risk_score"
    deployed_modules.mkdir(parents=True)
    source_modules = Path(__file__).parents[1] / "risk_score"
    deployment_files = {}
    for module_name in DEPLOYED_MODULES:
        deployed = deployed_modules / module_name
        shutil.copyfile(source_modules / module_name, deployed)
        deployment_files[f"module:{module_name}"] = binding(deployed)
    revision = "a" * 40
    deployment_manifest = tmp_path / "deployment-manifest.json"
    deployment_value = {
        "schema_version": 1,
        "contract": "risk-score-live-runtime-deployment-v1",
        "source_revision": revision,
        "source_sha256": digest(revision),
        "files": deployment_files,
    }
    deployment_value["manifest_sha256"] = canonical_sha256(deployment_value)
    write_canonical(deployment_manifest, deployment_value)

    static = tmp_path / "static"
    static.mkdir()
    katago = static / "katago"
    analysis = static / "analysis.cfg"
    selfplay = static / "selfplay.cfg"
    powered = static / "powered.cfg"
    standard = static / "standard.cfg"
    guardian = static / "guardian"
    katago.write_bytes(b"katago")
    shutil.copyfile(
        REPO
        / "cpp"
        / "configs"
        / "risk_score"
        / "promotion_curation_analysis.cfg",
        analysis,
    )
    shutil.copyfile(
        REPO
        / "cpp"
        / "configs"
        / "risk_score"
        / "promotion_curation_lead_selfplay_19x19.cfg",
        selfplay,
    )
    selfplay_lines = [
        line
        for line in selfplay.read_text(encoding="utf-8").splitlines()
        if not line.startswith("cudaDeviceToUseModel0Thread")
    ]
    selfplay_lines = [
        (
            "numNNServerThreadsPerModel = 1"
            if line.startswith("numNNServerThreadsPerModel")
            else line
        )
        for line in selfplay_lines
    ]
    selfplay_lines.append("cudaDeviceToUseModel0Thread0 = 7")
    selfplay.write_text(
        "\n".join(selfplay_lines) + "\n",
        encoding="utf-8",
    )
    powered.write_text("powered\n", encoding="utf-8")
    standard.write_text("standard\n", encoding="utf-8")
    guardian.write_bytes(b"guardian")

    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    selected = quarantine / "ordinary-primary.jsonl"
    prefilter = quarantine / "ordinary-primary.manifest.json"
    inventory = quarantine / "primary-inventory.json"
    write_canonical(selected, {"source": "ordinary"})
    write_canonical(prefilter, {"source": "ordinary-primary"})
    write_canonical(inventory, {"manifests": [binding(prefilter)]})
    training = tmp_path / "training-input"
    training.mkdir()

    guardian_argv = [
        str(guardian.resolve()),
        "--expected-spec-sha256",
        "b" * 64,
        "--claim-id",
        "{claim_id}",
        "--work-id",
        "{work_id}",
        "--receipt",
        "{guardian_receipt}",
        "--",
    ]
    replay_base = [
        "/fake-shadow-replay",
        "--request-id",
        "{request_id}",
        "--role",
        "{role}",
        "--model",
        "{model_path}",
        "--model-sha256",
        "{model_sha256}",
        "--suite-id",
        "{candidate_suite_id}",
        "--suite",
        "{candidate_suite_manifest}",
        "--suite-sha256",
        "{candidate_suite_manifest_sha256}",
        "--policy-identity",
        "{policy_identity}",
        "--phase",
        "{phase}",
        "--evidence",
        "{stage_evidence}",
    ]
    worker_value = {
        "schema_version": 1,
        "contract": WORKER_SPEC_CONTRACT,
        "deployment": {
            "repository_path": str(repository.resolve()),
            "source_revision": revision,
            "source_sha256": digest(revision),
        },
        "deployment_manifest": binding(deployment_manifest),
        "registry_spec": binding(registry_path, registry.identity),
        "policy": {
            "path": str(policy.resolve()),
            "sha256": file_sha256(policy),
            "identity": registry.policy_identity,
            "version": POLICY_VERSION,
        },
        "katago": binding(katago),
        "configs": {
            "analysis": binding(analysis),
            "selfplay": binding(selfplay),
            "powered": binding(powered),
            "standard": binding(standard),
        },
        "original_model": binding(original),
        "gpu_guardian": {
            "gpu_id": "7",
            "argv_prefix": guardian_argv,
            "argv_prefix_sha256": canonical_sha256(guardian_argv),
        },
        "curation_topology": {
            "supplement": {
                "shards_per_role": 1,
                "gpus": ["7"],
                "selfplay_gpus": ["7"],
                "per_gpu_parallelism": 1,
            },
            "pipeline": {
                "shards_per_role": 1,
                "gpus": ["7"],
                "per_gpu_parallelism": 1,
            },
        },
        "quarantined_source_root": str(quarantine.resolve()),
        "training_input_exclusion_roots": [str(training.resolve())],
        "curation_templates": {
            "supplement": {
                "training_input_root": str(training.resolve()),
                "selfplay_models_directory": directory_binding(models),
                "selfplay_override_args": [],
                "game_count": 8,
                "consensus_reserve_fraction": 1.0,
                "primary_prefilter_inventory": binding(inventory),
                "primary_prefilter_manifests": [binding(prefilter)],
                "round": 1,
                "prior_round_summaries": [],
                "downstream_accepted_counts": None,
            },
            "pipeline": {
                "sources": [
                    {
                        "name": "ordinary-primary",
                        "label": "ordinary",
                        "selected": binding(selected),
                        "prefilter_manifest": binding(prefilter),
                    }
                ]
            },
        },
        "continuity_templates": {
            "discovery_argv": replay_base,
            "confirmation_argv": replay_base,
        },
    }
    worker_value["spec_sha256"] = canonical_sha256(worker_value)
    worker_spec_path = tmp_path / "suite-rotation-worker.json"
    write_canonical(worker_spec_path, worker_value)
    worker_spec = load_worker_spec(worker_spec_path)

    request_id = "rotation-" + digest("rotation-request")
    base = (
        tmp_path / "rotation-results" / request_id
        if output_base is None
        else Path(output_base) / request_id
    )
    models_value = {
        "original": {
            "role": "immutable_original",
            **binding(original),
        },
        "champion": {
            "role": "frozen_champion",
            **binding(champion),
        },
    }
    policy_value = {
        "path": str(policy.resolve()),
        "sha256": file_sha256(policy),
        "identity": registry.policy_identity,
        "version": POLICY_VERSION,
    }
    request_directory = registry.root / "requests" / request_id
    request_directory.mkdir(parents=True)
    supplement_request = self_hashed(
        {
            "schema_version": 1,
            "contract": SUPPLEMENT_REQUEST_CONTRACT,
            "requested_spec_contract": SUPPLEMENT_SPEC_CONTRACT,
            "request_id": request_id,
            "models": models_value,
            "policy": policy_value,
            "target_counts": {
                "lead-40": registry.source_quotas["lead-40"],
                "lead-80": registry.source_quotas["lead-80"],
            },
            "quarantined_source_generation": True,
            "output_root": str((base / "supplement").resolve()),
        },
        "request_sha256",
    )
    supplement_request_path = request_directory / "curation-supplement.json"
    write_canonical(supplement_request_path, supplement_request)
    supplement_request_binding = binding(
        supplement_request_path, supplement_request["request_sha256"]
    )
    pipeline_request = self_hashed(
        {
            "schema_version": 1,
            "contract": PIPELINE_REQUEST_CONTRACT,
            "requested_spec_contract": PIPELINE_SPEC_CONTRACT,
            "request_id": request_id,
            "models": models_value,
            "policy": policy_value,
            "source_quotas": dict(registry.source_quotas),
            "holdout_quotas": {
                label: dict(registry.holdout_quotas[label])
                for label in registry.holdout_quotas
            },
            "supplement_request": supplement_request_binding,
            "suite_seed": "suite-rotation-" + request_id.removeprefix("rotation-"),
            "output_suite_contract": MACHINE_MANIFEST_CONTRACT,
            "output_root": str((base / "pipeline").resolve()),
        },
        "request_sha256",
    )
    pipeline_request_path = request_directory / "curation-pipeline.json"
    write_canonical(pipeline_request_path, pipeline_request)
    pipeline_request_binding = binding(
        pipeline_request_path, pipeline_request["request_sha256"]
    )
    rotation_request = self_hashed(
        {
            "schema_version": 1,
            "contract": ROTATION_REQUEST_CONTRACT,
            "request_id": request_id,
            "registry_spec": binding(registry_path, registry.identity),
            "base_active_suite": {
                "suite_id": digest("base-suite"),
                "version_sha256": digest("base-version"),
            },
            "models": models_value,
            "policy": policy_value,
            "trigger": {"eligible": True},
            "requests": {
                "curation_supplement": supplement_request_binding,
                "curation_pipeline": pipeline_request_binding,
            },
        },
        "request_sha256",
    )
    rotation_request_path = request_directory / "manifest.json"
    write_canonical(rotation_request_path, rotation_request)
    return {
        "tmp": tmp_path,
        "policy": policy,
        "registry": registry,
        "registry_path": registry_path,
        "worker_value": worker_value,
        "worker_spec": worker_spec,
        "worker_spec_path": worker_spec_path,
        "original": original,
        "champion": champion,
        "previous": previous,
        "training": training,
        "quarantine": quarantine,
        "request_id": request_id,
        "base": base.resolve(),
        "rotation_request": rotation_request,
        "rotation_request_path": rotation_request_path,
        "supplement_request": supplement_request,
        "supplement_request_path": supplement_request_path,
        "pipeline_request": pipeline_request,
        "pipeline_request_path": pipeline_request_path,
        "supplement_spec_path": (base / "supplement" / "spec.json").resolve(),
        "pipeline_spec_path": (base / "pipeline" / "spec.json").resolve(),
        "suite_manifest_path": (base / "pipeline" / "suite" / "manifest.json").resolve(),
    }


def make_worker(fixture, **kwargs):
    return SuiteRotationWorker(
        fixture["worker_spec"],
        registry=kwargs.pop("registry", SimpleNamespace()),
        supplement_loader=kwargs.pop("supplement_loader", fake_loader),
        pipeline_loader=kwargs.pop("pipeline_loader", fake_loader),
        **kwargs,
    )


def materialize(fixture, worker):
    return worker.materialize(
        request_id=fixture["request_id"],
        rotation_request=fixture["rotation_request_path"],
        supplement_request=fixture["supplement_request_path"],
        pipeline_request=fixture["pipeline_request_path"],
        supplement_spec=fixture["supplement_spec_path"],
        pipeline_spec=fixture["pipeline_spec_path"],
        rotation_request_sha256=file_sha256(fixture["rotation_request_path"]),
        rotation_request_identity=fixture["rotation_request"]["request_sha256"],
        supplement_request_sha256=file_sha256(fixture["supplement_request_path"]),
        supplement_request_identity=fixture["supplement_request"]["request_sha256"],
        pipeline_request_sha256=file_sha256(fixture["pipeline_request_path"]),
        pipeline_request_identity=fixture["pipeline_request"]["request_sha256"],
    )


def suite_value(fixture, *, review_mode="machine-consensus", overlap=False):
    ids = {
        "discovery": digest("discovery"),
        "confirmation": digest("discovery" if overlap else "confirmation"),
        "audit": digest("audit"),
    }
    models = {
        "original": {
            "role": "immutable_original",
            "sha256": file_sha256(fixture["original"]),
        },
        "champion": {
            "role": "frozen_champion",
            "sha256": file_sha256(fixture["champion"]),
        },
    }
    value = {
        "schemaVersion": 3,
        "manifestContract": MACHINE_MANIFEST_CONTRACT,
        "machineReviewOnly": True,
        "policy_hash": fixture["registry"].policy_identity,
        "policy_version": POLICY_VERSION,
        "curationSources": [
            {
                "source_name": "source-positions.jsonl",
                "review_mode": review_mode,
                "consensus_rules_version": 1,
                "policy_hash": fixture["registry"].policy_identity,
                "models": models,
            }
        ],
        "banks": [
            {
                "name": holdout,
                "holdout": holdout,
                "independentClusterIds": [identifier],
            }
            for holdout, identifier in ids.items()
        ],
    }
    value["manifestPayloadSha256"] = canonical_sha256(value)
    return value


def write_suite(fixture, **kwargs):
    value = suite_value(fixture, **kwargs)
    write_canonical(fixture["suite_manifest_path"], value)
    return value


def suite_validator(path, registry_spec, *, expected_champion_sha256):
    del registry_spec
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["curationSources"][0]["models"]["champion"]["sha256"] == (
        expected_champion_sha256
    )
    return SimpleNamespace(suite_id=file_sha256(path))


def test_materialize_emits_exact_request_bound_specs_and_replays(tmp_path):
    fixture = make_fixture(tmp_path)
    worker = make_worker(fixture)

    first = materialize(fixture, worker)
    supplement = json.loads(
        fixture["supplement_spec_path"].read_text(encoding="utf-8")
    )
    pipeline = json.loads(
        fixture["pipeline_spec_path"].read_text(encoding="utf-8")
    )
    first_bytes = (
        fixture["supplement_spec_path"].read_bytes(),
        fixture["pipeline_spec_path"].read_bytes(),
    )
    replay = materialize(fixture, worker)

    common = {
        "deployment": fixture["worker_value"]["deployment"],
        "deployment_manifest": fixture["worker_value"]["deployment_manifest"],
        "run_root": str(fixture["base"]),
        "policy": {
            "path": str(fixture["policy"].resolve()),
            "sha256": file_sha256(fixture["policy"]),
        },
        "katago": fixture["worker_value"]["katago"],
        "analysis_config": fixture["worker_value"]["configs"]["analysis"],
        "models": {
            role: {
                "path": model["path"],
                "sha256": model["sha256"],
            }
            for role, model in fixture["rotation_request"]["models"].items()
        },
    }
    expected_supplement = {
        "schema_version": 1,
        "contract": SUPPLEMENT_SPEC_CONTRACT,
        **common,
        "training_input_root": str(fixture["training"].resolve()),
        "work_root": str(fixture["base"] / "supplement" / "work"),
        "selfplay_config": fixture["worker_value"]["configs"]["selfplay"],
        "selfplay_models_directory": fixture["worker_value"]["curation_templates"][
            "supplement"
        ]["selfplay_models_directory"],
        "selfplay_override_args": [],
        "game_count": 8,
        "topology": fixture["worker_value"]["curation_topology"]["supplement"],
        "consensus_reserve_fraction": 1.0,
        "target_counts": {
            label: count * 2
            for label, count in fixture["supplement_request"][
                "target_counts"
            ].items()
        },
        "primary_prefilter_inventory": fixture["worker_value"][
            "curation_templates"
        ]["supplement"]["primary_prefilter_inventory"],
        "primary_prefilter_manifests": fixture["worker_value"][
            "curation_templates"
        ]["supplement"]["primary_prefilter_manifests"],
        "round": 1,
        "prior_round_summaries": [],
        "downstream_accepted_counts": None,
    }
    expected_supplement["spec_sha256"] = canonical_sha256(expected_supplement)
    expected_pipeline = {
        "schema_version": 1,
        "contract": PIPELINE_SPEC_CONTRACT,
        **common,
        "sources": fixture["worker_value"]["curation_templates"]["pipeline"][
            "sources"
        ],
        "work_root": str(fixture["base"] / "pipeline" / "work"),
        "outputs": {
            "reviewed_bank": str(
                fixture["base"] / "pipeline" / "outputs" / "source-positions.jsonl"
            ),
            "reviewed_manifest": str(
                fixture["base"]
                / "pipeline"
                / "outputs"
                / "source-positions.manifest.json"
            ),
            "suite_directory": str(fixture["base"] / "pipeline" / "suite"),
        },
        "quotas": dict(fixture["registry"].source_quotas),
        "topology": fixture["worker_value"]["curation_topology"]["pipeline"],
        "suite_seed": fixture["pipeline_request"]["suite_seed"],
    }
    expected_pipeline["spec_sha256"] = canonical_sha256(expected_pipeline)
    assert supplement == expected_supplement
    assert pipeline == expected_pipeline
    assert first == replay
    assert first_bytes == (
        fixture["supplement_spec_path"].read_bytes(),
        fixture["pipeline_spec_path"].read_bytes(),
    )
    assert first["source_quotas"] == dict(fixture["registry"].source_quotas)
    assert first["training_inputs_admitted"] is False
    assert first["activation_performed"] is False


def test_training_exclusions_reject_sources_and_outputs(tmp_path):
    fixture = make_fixture(tmp_path)
    unsafe = json.loads(fixture["worker_spec_path"].read_text(encoding="utf-8"))
    unsafe["quarantined_source_root"] = str(fixture["training"].resolve())
    unsafe.pop("spec_sha256")
    unsafe["spec_sha256"] = canonical_sha256(unsafe)
    unsafe_path = tmp_path / "unsafe-worker.json"
    write_canonical(unsafe_path, unsafe)
    with pytest.raises(WorkerSpecError, match="overlaps.*training-input"):
        load_worker_spec(unsafe_path)

    output_fixture = make_fixture(
        tmp_path / "unsafe-output",
        output_base=(tmp_path / "unsafe-output" / "training-input"),
    )
    with pytest.raises(WorkerStateError, match="training-input"):
        materialize(output_fixture, make_worker(output_fixture))
    assert not output_fixture["supplement_spec_path"].exists()
    assert not output_fixture["pipeline_spec_path"].exists()


@pytest.mark.parametrize(
    "suite_kwargs,match",
    [
        ({"review_mode": "human-review"}, "machine-consensus"),
        ({"overlap": True}, "holdout overlap"),
    ],
)
def test_curate_rejects_invalid_suite_provenance(
    tmp_path, suite_kwargs, match
):
    fixture = make_fixture(tmp_path)
    calls = []

    def supplement(path, *, poll_interval):
        calls.append(("supplement", path, poll_interval))
        return {"state": "complete"}

    def pipeline(path, *, poll_interval):
        calls.append(("pipeline", path, poll_interval))
        write_suite(fixture, **suite_kwargs)
        return {"state": "complete"}

    worker = make_worker(
        fixture,
        supplement_executor=supplement,
        pipeline_executor=pipeline,
        suite_validator=suite_validator,
    )
    materialize(fixture, worker)
    with pytest.raises(WorkerStateError, match=match):
        worker.curate(
            request_id=fixture["request_id"],
            supplement_spec=fixture["supplement_spec_path"],
            pipeline_spec=fixture["pipeline_spec_path"],
            suite_manifest=fixture["suite_manifest_path"],
            poll_interval=0,
        )
    assert [call[0] for call in calls] == ["supplement", "pipeline"]
    receipt = (
        fixture["pipeline_spec_path"].parent
        / ".suite-rotation-worker"
        / "curation-receipt.json"
    )
    assert not receipt.exists()


def test_curate_orders_apis_and_replays_valid_outputs(tmp_path):
    fixture = make_fixture(tmp_path)
    calls = []

    def supplement(path, *, poll_interval):
        calls.append(("supplement", path, poll_interval))
        return {"state": "complete"}

    def pipeline(path, *, poll_interval):
        calls.append(("pipeline", path, poll_interval))
        write_suite(fixture)
        return {"state": "complete"}

    worker = make_worker(
        fixture,
        supplement_executor=supplement,
        pipeline_executor=pipeline,
        suite_validator=suite_validator,
    )
    materialize(fixture, worker)
    first = worker.curate(
        request_id=fixture["request_id"],
        supplement_spec=fixture["supplement_spec_path"],
        pipeline_spec=fixture["pipeline_spec_path"],
        suite_manifest=fixture["suite_manifest_path"],
        poll_interval=0,
    )
    replay = worker.curate(
        request_id=fixture["request_id"],
        supplement_spec=fixture["supplement_spec_path"],
        pipeline_spec=fixture["pipeline_spec_path"],
        suite_manifest=fixture["suite_manifest_path"],
        poll_interval=0,
    )

    assert [call[0] for call in calls] == ["supplement", "pipeline"]
    assert first == replay
    assert first["suite_id"] == file_sha256(fixture["suite_manifest_path"])
    receipt = json.loads(
        Path(first["receipt"]["path"]).read_text(encoding="utf-8")
    )
    assert receipt["contract"] == CURATION_RECEIPT_CONTRACT
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert receipt["training_inputs_admitted"] is False
    assert receipt["activation_performed"] is False
    assert receipt["service_activation_invoked"] is False


class FakeRegistry:
    def __init__(self, fixture):
        self.fixture = fixture
        self.current = SimpleNamespace(
            path=fixture["champion"],
            sha256=file_sha256(fixture["champion"]),
            generation_id="generation-5",
        )
        self.previous = SimpleNamespace(
            path=fixture["previous"],
            sha256=file_sha256(fixture["previous"]),
            generation_id="generation-4",
        )
        self.suite_id = file_sha256(fixture["suite_manifest_path"])
        manifest = json.loads(
            fixture["suite_manifest_path"].read_text(encoding="utf-8")
        )
        self.version = SimpleNamespace(
            suite_id=self.suite_id,
            version_sha256=digest("candidate-version"),
            manifest_path=fixture["suite_manifest_path"],
            manifest_sha256=self.suite_id,
            manifest_identity=manifest["manifestPayloadSha256"],
        )

    def reconstruct(self):
        request_id = self.fixture["request_id"]
        return SimpleNamespace(
            requests=MappingProxyType(
                {
                    request_id: {
                        "request_id": request_id,
                        "base_suite_id": digest("base-suite"),
                        "champion_sha256": self.current.sha256,
                    }
                }
            ),
            registrations=MappingProxyType(
                {
                    self.suite_id: {
                        "request_id": request_id,
                        "suite_id": self.suite_id,
                    }
                }
            ),
            versions=MappingProxyType({self.suite_id: self.version}),
            current_champion=self.current,
            previous_champion_sha256=self.previous.sha256,
            champion_history=MappingProxyType(
                {
                    self.current.sha256: self.current,
                    self.previous.sha256: self.previous,
                }
            ),
        )


class ShadowRunner:
    def __init__(self, fixture, *, fail_phase=None, fail_decision=False):
        self.fixture = fixture
        self.fail_phase = fail_phase
        self.fail_decision = fail_decision
        self.calls = []

    @staticmethod
    def flag(argv, name):
        return argv[argv.index(name) + 1]

    def __call__(self, argv, **kwargs):
        assert kwargs == {
            "cwd": str(
                self.fixture["worker_spec"].deployment.repository_path
            ),
            "check": False,
            "capture_output": True,
            "text": True,
            "shell": False,
        }
        argv = list(argv)
        phase = self.flag(argv, "--phase")
        self.calls.append(phase)
        if phase == self.fail_phase:
            return subprocess.CompletedProcess(argv, 9, "", "injected failure")
        manifest = json.loads(
            Path(self.flag(argv, "--suite")).read_text(encoding="utf-8")
        )
        ids = sorted(
            identifier
            for bank in manifest["banks"]
            if bank["holdout"] == phase
            for identifier in bank["independentClusterIds"]
        )
        publish_shadow_replay_evidence(
            Path(self.flag(argv, "--evidence")),
            request_id=self.flag(argv, "--request-id"),
            role=self.flag(argv, "--role"),
            phase=phase,
            candidate_suite_id=self.flag(argv, "--suite-id"),
            candidate_suite_manifest_sha256=self.flag(argv, "--suite-sha256"),
            model_sha256=self.flag(argv, "--model-sha256"),
            policy_identity=self.flag(argv, "--policy-identity"),
            independent_cluster_ids=ids,
            command_argv=argv,
            decision=("FAIL" if self.fail_decision else "PASS"),
        )
        return subprocess.CompletedProcess(argv, 0, "", "")


def continuity_worker(fixture, runner):
    registry = FakeRegistry(fixture)
    worker = make_worker(
        fixture,
        registry=registry,
        command_runner=runner,
        suite_validator=suite_validator,
    )
    return worker, registry


def run_continuity(
    fixture,
    worker,
    registry,
    role,
    *,
    continuity_manifest=None,
):
    model = registry.current if role == "current_champion" else registry.previous
    evidence = fixture["base"] / "continuity" / f"{role}.json"
    return worker.continuity(
        request_id=fixture["request_id"],
        role=role,
        model_path=model.path,
        model_sha256=model.sha256,
        candidate_suite_id=registry.suite_id,
        candidate_suite_manifest=fixture["suite_manifest_path"],
        continuity_evidence=evidence,
        continuity_manifest=continuity_manifest,
    )


def test_continuity_failure_is_fail_closed_and_resumes_missing_stage(tmp_path):
    fixture = make_fixture(tmp_path)
    write_suite(fixture)
    failing = ShadowRunner(fixture, fail_phase="confirmation")
    worker, registry = continuity_worker(fixture, failing)

    with pytest.raises(WorkerCommandError, match="returned 9"):
        run_continuity(fixture, worker, registry, "current_champion")

    final = (
        fixture["base"] / "continuity" / "current_champion.json"
    )
    assert not final.exists()
    assert failing.calls == ["discovery", "confirmation"]
    discovery_receipts = list(
        (fixture["base"] / "continuity").glob(
            ".suite-rotation-worker/current_champion/*/"
            "discovery-command-receipt.json"
        )
    )
    assert len(discovery_receipts) == 1

    recovered = ShadowRunner(fixture)
    worker.command_runner = recovered
    result = run_continuity(fixture, worker, registry, "current_champion")
    assert recovered.calls == ["confirmation"]
    assert result["decision"] == "PASS"

    invalid_fixture = make_fixture(tmp_path / "invalid")
    write_suite(invalid_fixture)
    invalid = ShadowRunner(invalid_fixture, fail_decision=True)
    invalid_worker, invalid_registry = continuity_worker(invalid_fixture, invalid)
    with pytest.raises(WorkerCommandError, match="did not PASS"):
        run_continuity(
            invalid_fixture,
            invalid_worker,
            invalid_registry,
            "current_champion",
        )
    assert not (
        invalid_fixture["base"] / "continuity" / "current_champion.json"
    ).exists()


def test_continuity_orders_commands_replays_and_publishes_manifest(tmp_path):
    fixture = make_fixture(tmp_path)
    write_suite(fixture)
    runner = ShadowRunner(fixture)
    worker, registry = continuity_worker(fixture, runner)
    continuity_manifest = fixture["base"] / "continuity" / "manifest.json"

    current = run_continuity(
        fixture,
        worker,
        registry,
        "current_champion",
        continuity_manifest=continuity_manifest,
    )
    current_replay = run_continuity(
        fixture,
        worker,
        registry,
        "current_champion",
        continuity_manifest=continuity_manifest,
    )
    previous = run_continuity(
        fixture,
        worker,
        registry,
        "previous_champion",
        continuity_manifest=continuity_manifest,
    )

    assert runner.calls == [
        "discovery",
        "confirmation",
        "discovery",
        "confirmation",
    ]
    assert current == current_replay
    assert current["continuity_manifest"] is None
    assert previous["continuity_manifest"]["path"] == str(continuity_manifest)
    for result, role in (
        (current, "current_champion"),
        (previous, "previous_champion"),
    ):
        evidence = json.loads(
            Path(result["evidence"]["path"]).read_text(encoding="utf-8")
        )
        assert evidence["contract"] == CONTINUITY_EVIDENCE_CONTRACT
        assert evidence["role"] == role
        assert evidence["decision"] == "PASS"
        assert evidence["evidence_sha256"] == canonical_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_sha256"
            }
        )
        for phase in ("discovery", "confirmation"):
            receipt = json.loads(
                Path(result["shadow_replays"][phase]["receipt"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            assert receipt["contract"] == COMMAND_RECEIPT_CONTRACT
            assert receipt["shell"] is False
            assert receipt["decision"] == "PASS"
    manifest = json.loads(continuity_manifest.read_text(encoding="utf-8"))
    assert manifest["contract"] == CONTINUITY_CONTRACT
    assert manifest["request_id"] == fixture["request_id"]
    assert manifest["candidate_suite_id"] == registry.suite_id
    assert set(manifest["shadow_replays"]) == {
        "current_champion",
        "previous_champion",
    }
    assert all(
        replay["decision"] == "PASS"
        for replay in manifest["shadow_replays"].values()
    )
    assert current["training_inputs_admitted"] is False
    assert previous["activation_performed"] is False
