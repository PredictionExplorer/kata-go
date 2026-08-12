import hashlib
import json
import subprocess
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from risk_score.cluster_executor import WORK_SPEC_CONTRACT
from risk_score.cluster_executor import _work_execution_spec as parse_work_spec
from risk_score.cluster_scheduler import (
    ClusterScheduler,
    ReleaseOutcome,
    WorkKind,
    WorkState,
)
from risk_score.curation_pipeline import SPEC_CONTRACT as PIPELINE_SPEC_CONTRACT
from risk_score.curation_supplement import SPEC_CONTRACT as SUPPLEMENT_SPEC_CONTRACT
from risk_score.suite_rotation import (
    CONTINUITY_CONTRACT,
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from risk_score.suite_rotation_service import (
    CONTINUITY_EVIDENCE_CONTRACT,
    DEPLOYMENT_REQUEST_CONTRACT,
    SERVICE_SPEC_CONTRACT,
    STATUS_CONTRACT,
    MaterializerError,
    ServiceConflictError,
    ServiceSpecError,
    ServiceStaleError,
    ServiceStateError,
    SuiteRotationService,
    load_service_spec,
    parse_args,
    publish_continuity_evidence,
    publish_service_spec,
)

T0 = 1_786_467_600.0


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def self_hashed(value, field):
    value = dict(value)
    value[field] = canonical_sha256(value)
    return value


def binding(path, identity=None):
    value = {"path": str(path), "sha256": file_sha256(path)}
    if identity is not None:
        value["identity"] = identity
    return value


class Clock:
    def __init__(self, value=T0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds=1):
        self.value += seconds


class FakeRegistry:
    def __init__(self, fixture):
        self.fixture = fixture
        self.request_enabled = True
        self.request_id = fixture["request_id"]
        self.active_suite_id = fixture["base_suite_id"]
        self.activate_calls = 0
        self.register_calls = 0
        self.continuity_calls = 0
        self.sequence = 10
        self.spec = SimpleNamespace(
            path=fixture["registry_spec"],
            identity=digest("registry-identity"),
            root=fixture["registry_root"],
            policy_path=fixture["policy"],
            policy_file_sha256=file_sha256(fixture["policy"]),
            policy_identity=fixture["policy_identity"],
        )
        self.current = SimpleNamespace(
            path=fixture["current_model"],
            sha256=file_sha256(fixture["current_model"]),
            generation_id="generation-5",
        )
        self.previous = SimpleNamespace(
            path=fixture["previous_model"],
            sha256=file_sha256(fixture["previous_model"]),
            generation_id="generation-4",
        )
        self.original = SimpleNamespace(
            path=fixture["original_model"],
            sha256=file_sha256(fixture["original_model"]),
        )
        self.requests = {
            self.request_id: {
                "request_id": self.request_id,
                "request_manifest": fixture["rotation_binding"],
                "base_suite_id": self.active_suite_id,
                "champion_sha256": self.current.sha256,
                "generation_id": self.current.generation_id,
                "_sequence": 10,
                "_manifest": fixture["rotation_request"],
            }
        }
        self.registrations = {}
        self.continuity = {}
        self.pins = {}
        self.versions = {
            self.active_suite_id: SimpleNamespace(
                suite_id=self.active_suite_id,
                version_sha256=digest("base-version"),
                manifest_path=fixture["base_manifest"],
                manifest_sha256=file_sha256(fixture["base_manifest"]),
                manifest_identity=digest("base-manifest-identity"),
            )
        }
        self.champion_history = {
            self.current.sha256: self.current,
            self.previous.sha256: self.previous,
        }

    def reconstruct(self):
        return SimpleNamespace(
            active_suite_id=self.active_suite_id,
            current_champion=self.current,
            previous_champion_sha256=self.previous.sha256,
            champion_history=MappingProxyType(dict(self.champion_history)),
            requests=MappingProxyType(dict(self.requests)),
            registrations=MappingProxyType(dict(self.registrations)),
            continuity=MappingProxyType(dict(self.continuity)),
            pins=MappingProxyType(dict(self.pins)),
            versions=MappingProxyType(dict(self.versions)),
        )

    def status(self):
        value = {
            "schema_version": 1,
            "contract": "risk-score-evaluation-suite-rotation-status-v1",
            "active_suite": {"suite_id": self.active_suite_id},
            "current_champion": {
                "sha256": self.current.sha256,
                "generation_id": self.current.generation_id,
            },
            "current_request_id": self.request_id if self.request_enabled else None,
        }
        value["status_sha256"] = canonical_sha256(value)
        return value

    def once(self):
        return self.status()

    def register_suite(self, request_id, manifest_path):
        assert request_id == self.request_id
        self.register_calls += 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        suite_id = manifest["testSuiteId"]
        self.sequence += 1
        version = SimpleNamespace(
            suite_id=suite_id,
            version_sha256=digest("version:" + suite_id),
            manifest_path=manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            manifest_identity=manifest["manifestPayloadSha256"],
        )
        self.versions[suite_id] = version
        self.registrations[suite_id] = {
            "request_id": request_id,
            "suite_id": suite_id,
            "_sequence": self.sequence,
        }
        return SimpleNamespace(payload={"suite_id": suite_id})

    def record_continuity(self, request_id, suite_id, manifest_path):
        assert request_id == self.request_id
        self.continuity_calls += 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.sequence += 1
        self.continuity[suite_id] = {
            "request_id": request_id,
            "suite_id": suite_id,
            "manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "identity": manifest["manifest_sha256"],
            },
            "_sequence": self.sequence,
        }
        return SimpleNamespace(payload={"suite_id": suite_id})

    def activate_suite(self, *args, **kwargs):
        self.activate_calls += 1
        raise AssertionError("execution service must never activate a suite")


class Materializer:
    def __init__(self, fixture, *, fail_once=False):
        self.fixture = fixture
        self.fail_once = fail_once
        self.calls = []

    @staticmethod
    def flag(argv, name):
        return argv[argv.index(name) + 1]

    def __call__(self, argv, **kwargs):
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "shell": False,
        }
        argv = list(argv)
        self.calls.append(argv)
        if self.fail_once and len(self.calls) == 1:
            return subprocess.CompletedProcess(argv, 7, "", "transient")
        supplement = Path(self.flag(argv, "--supplement-spec"))
        pipeline = Path(self.flag(argv, "--pipeline-spec"))
        request_id = self.flag(argv, "--request-id")
        self.fixture["write_materialized_specs"](
            request_id=request_id,
            supplement_path=supplement,
            pipeline_path=pipeline,
        )
        return subprocess.CompletedProcess(argv, 0, "", "")


def make_fixture(tmp_path):
    scheduler = ClusterScheduler(tmp_path / "scheduler", ("0", "7"), clock=lambda: T0)
    service_root = tmp_path / "service"
    result_root = tmp_path / "rotation-results"
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    static = tmp_path / "static"
    static.mkdir()
    files = {}
    for name, contents in (
        ("registry-spec.json", b'{"fake":"registry"}\n'),
        ("policy.json", b'{"policy":"v3"}\n'),
        ("original.bin.gz", b"original"),
        ("current.bin.gz", b"champion-current"),
        ("previous.bin.gz", b"champion-previous"),
        ("katago", b"binary"),
        ("analysis.cfg", b"analysis"),
        ("selfplay.cfg", b"selfplay"),
        ("deployment.json", b'{"deployment":"frozen"}\n'),
        ("primary.jsonl", b'{"position":"one"}\n'),
        ("primary.manifest.json", b'{"manifest":"primary"}\n'),
        ("inventory.json", b'{"inventory":"primary"}\n'),
    ):
        path = static / name
        path.write_bytes(contents)
        files[name] = path
    models_directory = static / "models"
    models_directory.mkdir()
    (models_directory / "model.bin.gz").write_bytes(b"original")

    base_manifest = registry_root / "base-manifest.json"
    write_canonical(base_manifest, {"suite": "base"})
    base_suite_id = digest("base-suite")
    request_id = "rotation-" + digest("request")
    policy_identity = digest("policy-identity")
    models = {
        "original": {
            "role": "immutable_original",
            "path": str(files["original.bin.gz"]),
            "sha256": file_sha256(files["original.bin.gz"]),
        },
        "champion": {
            "role": "frozen_champion",
            "path": str(files["current.bin.gz"]),
            "sha256": file_sha256(files["current.bin.gz"]),
        },
    }
    policy = {
        "path": str(files["policy.json"]),
        "sha256": file_sha256(files["policy.json"]),
        "identity": policy_identity,
        "version": "risk-seeking-checkpoint-promotion-v3",
    }
    request_dir = registry_root / "requests" / request_id
    request_dir.mkdir(parents=True)
    supplement_request = self_hashed(
        {
            "schema_version": 1,
            "contract": ("risk-score-suite-rotation-curation-supplement-request-v1"),
            "requested_spec_contract": SUPPLEMENT_SPEC_CONTRACT,
            "request_id": request_id,
            "models": models,
            "policy": policy,
            "target_counts": {"lead-40": 2, "lead-80": 2},
            "quarantined_source_generation": True,
            "output_root": str(result_root / request_id / "supplement"),
        },
        "request_sha256",
    )
    supplement_request_path = request_dir / "curation-supplement.json"
    write_canonical(supplement_request_path, supplement_request)
    supplement_binding = binding(
        supplement_request_path, supplement_request["request_sha256"]
    )
    pipeline_request = self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-suite-rotation-curation-pipeline-request-v1",
            "requested_spec_contract": PIPELINE_SPEC_CONTRACT,
            "request_id": request_id,
            "models": models,
            "policy": policy,
            "source_quotas": {
                "ordinary": 1,
                "lead-40": 2,
                "lead-80": 2,
            },
            "holdout_quotas": {
                label: {
                    "discovery": 1,
                    "confirmation": 0,
                    "audit": 0,
                }
                for label in ("ordinary", "lead-40", "lead-80")
            },
            "supplement_request": supplement_binding,
            "suite_seed": "suite-rotation-test",
            "output_suite_contract": (
                "risk-score-authoritative-evaluation-manifest-v3"
            ),
            "output_root": str(result_root / request_id / "pipeline"),
        },
        "request_sha256",
    )
    pipeline_request_path = request_dir / "curation-pipeline.json"
    write_canonical(pipeline_request_path, pipeline_request)
    pipeline_binding = binding(
        pipeline_request_path, pipeline_request["request_sha256"]
    )
    rotation_request = self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-evaluation-suite-rotation-request-v1",
            "request_id": request_id,
            "registry_spec": {
                "path": str(files["registry-spec.json"]),
                "sha256": file_sha256(files["registry-spec.json"]),
                "identity": digest("registry-identity"),
            },
            "base_active_suite": {
                "suite_id": base_suite_id,
                "version_sha256": digest("base-version"),
            },
            "models": models,
            "policy": policy,
            "trigger": {"eligible": True},
            "requests": {
                "curation_supplement": supplement_binding,
                "curation_pipeline": pipeline_binding,
            },
        },
        "request_sha256",
    )
    rotation_path = request_dir / "manifest.json"
    write_canonical(rotation_path, rotation_request)

    result_templates = {
        "supplement_spec": {
            "path": str(result_root / "{request_id}" / "supplement" / "spec.json"),
            "contract": SUPPLEMENT_SPEC_CONTRACT,
        },
        "pipeline_spec": {
            "path": str(result_root / "{request_id}" / "pipeline" / "spec.json"),
            "contract": PIPELINE_SPEC_CONTRACT,
        },
        "suite_manifest": {
            "path": str(
                result_root / "{request_id}" / "pipeline" / "suite" / "manifest.json"
            ),
            "contract": "risk-score-authoritative-evaluation-manifest-v3",
        },
        "continuity_evidence": {
            "path": str(result_root / "{request_id}" / "continuity" / "{role}.json"),
            "contract": CONTINUITY_EVIDENCE_CONTRACT,
        },
        "continuity_manifest": {
            "path": str(result_root / "{request_id}" / "continuity" / "manifest.json"),
            "contract": CONTINUITY_CONTRACT,
        },
        "deployment_request": {
            "path": str(result_root / "{request_id}" / "deployment-request.json"),
            "contract": DEPLOYMENT_REQUEST_CONTRACT,
        },
        "status": {
            "path": str(service_root / "status.json"),
            "contract": STATUS_CONTRACT,
        },
    }
    service_spec_path = tmp_path / "suite-rotation-service.json"
    service_spec = publish_service_spec(
        service_spec_path,
        root=service_root,
        registry_spec_path=files["registry-spec.json"],
        scheduler_directory=scheduler.directory,
        gpu7_id="7",
        guardian_argv_prefix=[
            "/guardian",
            "--expected-spec-sha256",
            "a" * 64,
            "--claim-id",
            "{claim_id}",
            "--work-id",
            "{work_id}",
            "--receipt",
            "{guardian_receipt}",
            "--command-json",
        ],
        materializer_argv_template=[
            "/materializer",
            "--request",
            "{rotation_request}",
            "--supplement-request",
            "{supplement_request}",
            "--pipeline-request",
            "{pipeline_request}",
            "--supplement-spec",
            "{supplement_spec}",
            "--pipeline-spec",
            "{pipeline_spec}",
            "--request-id",
            "{request_id}",
        ],
        curation_argv_template=[
            "/curation",
            "--request-id",
            "{request_id}",
            "--supplement-spec",
            "{supplement_spec}",
            "--pipeline-spec",
            "{pipeline_spec}",
            "--suite-manifest",
            "{suite_manifest}",
        ],
        continuity_argv_template=[
            "/continuity",
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
            "--evidence",
            "{continuity_evidence}",
        ],
        results=result_templates,
        poll_interval_seconds=3.0,
        actor="suite-rotation-test",
    )
    fixture = {
        "tmp": tmp_path,
        "scheduler": scheduler,
        "service_root": service_root,
        "result_root": result_root,
        "registry_root": registry_root,
        "registry_spec": files["registry-spec.json"],
        "policy": files["policy.json"],
        "policy_identity": policy_identity,
        "original_model": files["original.bin.gz"],
        "current_model": files["current.bin.gz"],
        "previous_model": files["previous.bin.gz"],
        "base_manifest": base_manifest,
        "base_suite_id": base_suite_id,
        "request_id": request_id,
        "rotation_request": rotation_request,
        "rotation_binding": binding(rotation_path, rotation_request["request_sha256"]),
        "supplement_request": supplement_request,
        "pipeline_request": pipeline_request,
        "service_spec": service_spec,
        "service_spec_path": service_spec_path,
        "files": files,
        "models_directory": models_directory,
    }

    def write_materialized_specs(*, request_id, supplement_path, pipeline_path):
        assert request_id == fixture["request_id"]
        deployment = {
            "repository_path": str(tmp_path),
            "source_revision": "a" * 40,
            "source_sha256": digest("a" * 40),
        }
        common = {
            "deployment": deployment,
            "deployment_manifest": binding(files["deployment.json"]),
            "run_root": str(result_root),
            "policy": {
                "path": str(files["policy.json"]),
                "sha256": file_sha256(files["policy.json"]),
            },
            "katago": binding(files["katago"]),
            "analysis_config": binding(files["analysis.cfg"]),
            "models": {
                role: {
                    "path": model["path"],
                    "sha256": model["sha256"],
                }
                for role, model in models.items()
            },
        }
        supplement = self_hashed(
            {
                "schema_version": 1,
                "contract": SUPPLEMENT_SPEC_CONTRACT,
                **common,
                "training_input_root": str(tmp_path / "training"),
                "work_root": str(result_root / request_id / "supplement" / "work"),
                "selfplay_config": binding(files["selfplay.cfg"]),
                "selfplay_models_directory": {
                    "path": str(models_directory),
                    "sha256": digest("models-directory"),
                },
                "selfplay_override_args": [],
                "game_count": 1,
                "topology": {
                    "shards_per_role": 1,
                    "gpus": ["7"],
                    "selfplay_gpus": ["7"],
                    "per_gpu_parallelism": 1,
                },
                "consensus_reserve_fraction": 1.0,
                "target_counts": dict(supplement_request["target_counts"]),
                "primary_prefilter_inventory": binding(files["inventory.json"]),
                "primary_prefilter_manifests": [
                    binding(files["primary.manifest.json"])
                ],
                "round": 1,
                "prior_round_summaries": [],
                "downstream_accepted_counts": None,
            },
            "spec_sha256",
        )
        pipeline_root = result_root / request_id / "pipeline"
        pipeline = self_hashed(
            {
                "schema_version": 1,
                "contract": PIPELINE_SPEC_CONTRACT,
                **common,
                "sources": [
                    {
                        "name": "ordinary-primary",
                        "label": "ordinary",
                        "selected": binding(files["primary.jsonl"]),
                        "prefilter_manifest": binding(files["primary.manifest.json"]),
                    }
                ],
                "work_root": str(pipeline_root / "work"),
                "outputs": {
                    "reviewed_bank": str(
                        pipeline_root / "outputs" / "source-positions.jsonl"
                    ),
                    "reviewed_manifest": str(
                        pipeline_root / "outputs" / "source-positions.manifest.json"
                    ),
                    "suite_directory": str(pipeline_root / "suite"),
                },
                "quotas": dict(pipeline_request["source_quotas"]),
                "topology": {
                    "shards_per_role": 1,
                    "gpus": ["7"],
                    "per_gpu_parallelism": 1,
                },
                "suite_seed": pipeline_request["suite_seed"],
            },
            "spec_sha256",
        )
        write_canonical(supplement_path, supplement)
        write_canonical(pipeline_path, pipeline)

    fixture["write_materialized_specs"] = write_materialized_specs
    fixture["registry"] = FakeRegistry(fixture)
    return fixture


def fake_loader(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        raw=raw,
        identity=raw["spec_sha256"],
        file_sha256=file_sha256(path),
    )


def suite_validator(path, registry_spec, *, expected_champion_sha256):
    del registry_spec
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["curationChampionSha256"] == expected_champion_sha256
    return SimpleNamespace(suite_id=value["testSuiteId"])


def make_service(fixture, *, runner=None, failure_hook=None):
    return SuiteRotationService(
        fixture["service_spec"],
        registry=fixture["registry"],
        scheduler=fixture["scheduler"],
        clock=Clock(),
        runner=runner or Materializer(fixture),
        supplement_loader=fake_loader,
        pipeline_loader=fake_loader,
        suite_validator=suite_validator,
        failure_hook=failure_hook,
    )


def latest_work(fixture, stage, role=None):
    records = [
        record
        for record in fixture["scheduler"].reconstruct().work.values()
        if record.payload.get("stage") == stage and record.payload.get("role") == role
    ]
    return max(records, key=lambda record: record.enqueue_sequence)


def complete_work(fixture, record):
    scheduler = fixture["scheduler"]
    claim = scheduler.claim("7", "executor-test")
    assert claim is not None
    assert claim.work_id == record.work_id
    scheduler.release(claim, outcome=ReleaseOutcome.COMPLETED)


def write_suite_output(fixture, suite_id=None):
    suite_id = suite_id or digest("candidate-suite")
    path = (
        fixture["service_spec"]
        .results["suite_manifest"]
        .path(request_id=fixture["request_id"])
    )
    value = {
        "schemaVersion": 3,
        "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
        "machineReviewOnly": True,
        "curationChampionSha256": file_sha256(fixture["current_model"]),
        "testSuiteId": suite_id,
    }
    value["manifestPayloadSha256"] = canonical_sha256(value)
    write_canonical(path, value)
    return path, suite_id


def advance_to_continuity(fixture, service):
    service.once()
    curation = latest_work(fixture, "curation")
    complete_work(fixture, curation)
    _, suite_id = write_suite_output(fixture)
    status = service.once()
    assert status["state"] == "continuity-pending"
    assert fixture["registry"].register_calls == 1
    return suite_id


def complete_continuity(fixture, service, suite_id):
    for role in ("current_champion", "previous_champion"):
        path = (
            fixture["service_spec"]
            .results["continuity_evidence"]
            .path(request_id=fixture["request_id"], role=role)
        )
        publish_continuity_evidence(
            path,
            service=service,
            request_id=fixture["request_id"],
            candidate_suite_id=suite_id,
            role=role,
            completed_at_utc="2026-08-11T22:00:00.000000Z",
        )
        complete_work(fixture, latest_work(fixture, "continuity", role))


def test_service_spec_is_strict_canonical_self_hashed_and_cli_bounded(tmp_path):
    fixture = make_fixture(tmp_path)
    loaded = load_service_spec(fixture["service_spec_path"])
    assert loaded.identity == fixture["service_spec"].identity
    assert loaded.gpu7_id == "7"
    assert loaded.raw["contract"] == SERVICE_SPEC_CONTRACT
    assert (
        parse_args(["status", "--spec", str(fixture["service_spec_path"])]).mode
        == "status"
    )
    assert (
        parse_args(["once", "--spec", str(fixture["service_spec_path"])]).mode == "once"
    )
    assert (
        parse_args(["watch", "--spec", str(fixture["service_spec_path"])]).mode
        == "watch"
    )
    assert (
        parse_args(["--spec", str(fixture["service_spec_path"]), "watch"]).mode
        == "watch"
    )

    raw = json.loads(fixture["service_spec_path"].read_text(encoding="utf-8"))
    raw["actor"] = "changed"
    write_canonical(fixture["service_spec_path"], raw)
    with pytest.raises(ServiceSpecError, match="self-hash"):
        load_service_spec(fixture["service_spec_path"])

    raw["spec_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "spec_sha256"}
    )
    fixture["service_spec_path"].write_text(
        json.dumps(raw, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ServiceSpecError, match="canonical"):
        load_service_spec(fixture["service_spec_path"])


def test_cadence_noop_enqueues_nothing_and_writes_status(tmp_path):
    fixture = make_fixture(tmp_path)
    fixture["registry"].request_enabled = False
    materializer = Materializer(fixture)
    service = make_service(fixture, runner=materializer)

    status = service.once()

    assert status["state"] == "idle"
    assert materializer.calls == []
    assert fixture["scheduler"].reconstruct().work == {}
    persisted = json.loads(
        fixture["service_spec"].status_path.read_text(encoding="utf-8")
    )
    assert persisted["contract"] == STATUS_CONTRACT
    assert persisted["status_sha256"] == status["status_sha256"]


def test_watch_reconciles_before_sleep_without_noisy_runner_reentry(tmp_path):
    fixture = make_fixture(tmp_path)
    fixture["registry"].request_enabled = False
    materializer = Materializer(fixture)
    service = make_service(fixture, runner=materializer)
    sleeps = []

    def stop(interval):
        sleeps.append(interval)
        raise KeyboardInterrupt

    service.sleeper = stop
    with pytest.raises(KeyboardInterrupt):
        service.watch()

    assert sleeps == [3.0]
    assert materializer.calls == []
    assert fixture["service_spec"].status_path.is_file()


def test_request_materializes_exact_specs_and_enqueues_replay_safe_backfill(
    tmp_path,
):
    fixture = make_fixture(tmp_path)
    materializer = Materializer(fixture)
    service = make_service(fixture, runner=materializer)

    first = service.once()
    second = service.once()

    assert first["state"] == second["state"] == "curation-pending"
    assert len(materializer.calls) == 1
    work = latest_work(fixture, "curation")
    assert work.kind == WorkKind.BACKFILL
    assert work.state == WorkState.QUEUED
    assert work.preemptible is True
    assert work.eligible_gpus == ("7",)
    assert work.preferred_gpu == "7"
    executor = work.payload["executor_spec"]
    assert executor["contract"] == WORK_SPEC_CONTRACT
    assert executor["kind"] == WorkKind.BACKFILL.value
    assert executor["lease_role"] == "none"
    assert executor["argv"][:2] == ["/guardian", "--expected-spec-sha256"]
    assert "{claim_id}" in executor["argv"]
    assert "{work_id}" in executor["argv"]
    child_argv = json.loads(executor["argv"][-1])
    assert child_argv[0] == "/curation"
    assert child_argv[child_argv.index("--request-id") + 1] == fixture["request_id"]
    assert executor["spec_sha256"] == canonical_sha256(
        {key: value for key, value in executor.items() if key != "spec_sha256"}
    )
    parsed = parse_work_spec(
        work,
        ("0", "7"),
        "7",
        service.spec.guardian_argv_prefix,
    )
    assert parsed.argv == tuple(executor["argv"])
    assert len(fixture["scheduler"].reconstruct().work) == 1


def test_curation_continuity_and_privileged_deployment_never_activate_pointer(
    tmp_path,
):
    fixture = make_fixture(tmp_path)
    service = make_service(fixture)
    original_active = fixture["registry"].active_suite_id
    suite_id = advance_to_continuity(fixture, service)
    continuity_records = [
        record
        for record in fixture["scheduler"].reconstruct().work.values()
        if record.payload.get("stage") == "continuity"
    ]
    assert {record.payload["role"] for record in continuity_records} == {
        "current_champion",
        "previous_champion",
    }
    assert all(record.preemptible for record in continuity_records)

    complete_continuity(fixture, service, suite_id)
    status = service.once()

    assert status["state"] == "deployment-requested"
    assert fixture["registry"].continuity_calls == 1
    assert fixture["registry"].activate_calls == 0
    assert fixture["registry"].active_suite_id == original_active
    deployment_path = (
        fixture["service_spec"]
        .results["deployment_request"]
        .path(request_id=fixture["request_id"])
    )
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    assert deployment["contract"] == DEPLOYMENT_REQUEST_CONTRACT
    assert deployment["candidate_suite"]["suite_id"] == suite_id
    assert deployment["compare_and_swap"]["expected_active_suite_id"] == (
        original_active
    )
    assert deployment["compare_and_swap"]["expected_pin_count"] == 0
    assert deployment["compare_and_swap"]["require_clean_generation_boundary"] is True
    assert deployment["privilege_boundary"]["service_may_activate_suite"] is False
    assert (
        deployment["privilege_boundary"]["service_may_mutate_active_suite_pointer"]
        is False
    )
    assert set(deployment["proposed_frozen_commands"]) == {"runtime", "service"}

    replay = service.once()
    assert replay["state"] == "deployment-requested"
    assert fixture["registry"].register_calls == 1
    assert fixture["registry"].continuity_calls == 1


def test_stale_champion_fails_before_registration(tmp_path):
    fixture = make_fixture(tmp_path)
    service = make_service(fixture)
    service.once()
    complete_work(fixture, latest_work(fixture, "curation"))
    write_suite_output(fixture)
    replacement = fixture["tmp"] / "replacement.bin.gz"
    replacement.write_bytes(b"new champion")
    fixture["registry"].current = SimpleNamespace(
        path=replacement,
        sha256=file_sha256(replacement),
        generation_id="generation-6",
    )
    fixture["registry"].champion_history[fixture["registry"].current.sha256] = fixture[
        "registry"
    ].current

    with pytest.raises(ServiceStaleError, match="changed"):
        service.once()

    assert fixture["registry"].register_calls == 0
    assert fixture["registry"].active_suite_id == fixture["base_suite_id"]
    assert (
        json.loads(fixture["service_spec"].status_path.read_text(encoding="utf-8"))[
            "state"
        ]
        == "failed"
    )


def test_pins_block_deployment_request_until_zero(tmp_path):
    fixture = make_fixture(tmp_path)
    service = make_service(fixture)
    suite_id = advance_to_continuity(fixture, service)
    complete_continuity(fixture, service, suite_id)
    fixture["registry"].pins["evaluation-live"] = {
        "evaluation_id": "evaluation-live",
        "suite_id": fixture["base_suite_id"],
        "champion_sha256": file_sha256(fixture["current_model"]),
        "generation_id": "generation-5",
    }

    blocked = service.once()

    deployment_path = (
        fixture["service_spec"]
        .results["deployment_request"]
        .path(request_id=fixture["request_id"])
    )
    assert blocked["state"] == "deployment-blocked"
    assert not deployment_path.exists()
    assert fixture["registry"].activate_calls == 0

    fixture["registry"].pins.clear()
    ready = service.once()
    assert ready["state"] == "deployment-requested"
    assert deployment_path.is_file()


def test_materializer_failure_recovers_without_duplicate_scheduler_work(tmp_path):
    fixture = make_fixture(tmp_path)
    materializer = Materializer(fixture, fail_once=True)
    service = make_service(fixture, runner=materializer)

    with pytest.raises(MaterializerError, match="returned 7"):
        service.once()
    failed = json.loads(fixture["service_spec"].status_path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert fixture["scheduler"].reconstruct().work == {}

    recovered = service.once()
    assert recovered["state"] == "curation-pending"
    assert len(materializer.calls) == 2
    assert len(fixture["scheduler"].reconstruct().work) == 1


def test_partial_materialization_and_claimed_output_resume_without_overlap(
    tmp_path,
):
    fixture = make_fixture(tmp_path)
    supplement_path = (
        fixture["service_spec"]
        .results["supplement_spec"]
        .path(request_id=fixture["request_id"])
    )
    pipeline_path = (
        fixture["service_spec"]
        .results["pipeline_spec"]
        .path(request_id=fixture["request_id"])
    )
    fixture["write_materialized_specs"](
        request_id=fixture["request_id"],
        supplement_path=supplement_path,
        pipeline_path=pipeline_path,
    )
    pipeline_path.unlink()
    materializer = Materializer(fixture)
    service = make_service(fixture, runner=materializer)

    service.once()

    assert len(materializer.calls) == 1
    assert supplement_path.is_file() and pipeline_path.is_file()
    curation = latest_work(fixture, "curation")
    claim = fixture["scheduler"].claim("7", "executor-test")
    assert claim.work_id == curation.work_id
    _, suite_id = write_suite_output(fixture)

    waiting = service.once()

    assert waiting["state"] == "curation-pending"
    assert fixture["registry"].register_calls == 0
    fixture["scheduler"].release(claim, outcome=ReleaseOutcome.COMPLETED)
    service.once()
    assert suite_id in fixture["registry"].registrations


def test_failed_curation_work_is_reenqueued_for_immutable_resume(tmp_path):
    fixture = make_fixture(tmp_path)
    service = make_service(fixture)
    service.once()
    first = latest_work(fixture, "curation")
    claim = fixture["scheduler"].claim("7", "executor-test")
    fixture["scheduler"].release(claim, outcome=ReleaseOutcome.FAILED)

    service.once()

    second = latest_work(fixture, "curation")
    assert second.work_id != first.work_id
    assert fixture["scheduler"].get_work(first.work_id).state == WorkState.FAILED
    assert second.state == WorkState.QUEUED
    assert second.payload["attempt"] == 2
    assert len(fixture["scheduler"].reconstruct().work) == 2


def test_completed_work_without_output_and_changed_provenance_fail_closed(tmp_path):
    fixture = make_fixture(tmp_path)
    service = make_service(fixture)
    service.once()
    complete_work(fixture, latest_work(fixture, "curation"))

    with pytest.raises(ServiceStateError, match="published no bound output"):
        service.once()

    second_fixture = make_fixture(tmp_path / "second")
    second_service = make_service(second_fixture)
    second_service.once()
    pipeline_path = (
        second_fixture["service_spec"]
        .results["pipeline_spec"]
        .path(request_id=second_fixture["request_id"])
    )
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["topology"]["per_gpu_parallelism"] = 2
    pipeline.pop("spec_sha256")
    pipeline["spec_sha256"] = canonical_sha256(pipeline)
    write_canonical(pipeline_path, pipeline)

    with pytest.raises(ServiceConflictError, match="immutable artifact conflicts"):
        second_service.once()
