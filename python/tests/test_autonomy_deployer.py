import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from risk_score.autonomy_deployer import (
    DEPLOYER_UNIT_NAME,
    REQUIRED_ACTIVATION_UNITS,
    ActivationRetryableHalt,
    AutonomyDeployer,
    DeployerInterrupted,
    DeployerSafetyHalt,
    DeploymentRequestError,
    build_fence_proof,
    publish_deployer_spec,
    render_deployer_systemd_unit,
)
from risk_score.service_activation import (
    AUTONOMY_SERVICE_SPEC_CONTRACT,
    FULL_SERVICE_UNIT_NAMES,
    TARGET_UNIT,
    apply_service_activation,
)
from risk_score.suite_rotation import (
    CONTINUITY_CONTRACT,
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from risk_score.suite_rotation_service import (
    DEPLOYMENT_REQUEST_CONTRACT,
    SERVICE_SPEC_CONTRACT,
)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def self_hashed(value, field):
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def binding(path, identity=None):
    value = {"path": str(path), "sha256": file_sha256(path)}
    if identity is not None:
        value["identity"] = identity
    return value


class FakeRegistry:
    def __init__(self, fixture):
        self.fixture = fixture
        self.spec = SimpleNamespace(
            path=fixture.registry_spec,
            identity=fixture.registry_identity,
        )
        self.active_suite_id = fixture.base_suite_id
        self.current = SimpleNamespace(
            sha256=fixture.champion_sha256,
            generation_id=fixture.generation_id,
        )
        self.pins = {}
        self.boundaries = {
            "boundary-6": {
                "boundary_id": "boundary-6",
                "champion_sha256": fixture.champion_sha256,
                "generation_id": fixture.generation_id,
                "clean": True,
                "_sequence": 13,
            }
        }
        self.events = []
        self.activate_calls = 0
        self.crash_after_cas = False
        self.requests = {
            fixture.request_id: {
                "request_manifest": fixture.rotation_binding,
                "_manifest": fixture.rotation_request,
            }
        }
        self.registrations = {
            fixture.candidate_suite_id: {
                "request_id": fixture.request_id,
                "_sequence": 11,
            }
        }
        self.versions = {
            fixture.candidate_suite_id: SimpleNamespace(
                version_sha256=fixture.candidate_version,
                manifest_path=fixture.suite_manifest,
                manifest_sha256=file_sha256(fixture.suite_manifest),
                manifest_identity=fixture.suite_identity,
            )
        }
        self.continuity = {
            fixture.candidate_suite_id: {
                "request_id": fixture.request_id,
                "manifest": fixture.continuity_binding,
                "_sequence": 12,
            }
        }

    def reconstruct(self):
        return SimpleNamespace(
            active_suite_id=self.active_suite_id,
            current_champion=self.current,
            pins=MappingProxyType(dict(self.pins)),
            requests=MappingProxyType(dict(self.requests)),
            registrations=MappingProxyType(dict(self.registrations)),
            versions=MappingProxyType(dict(self.versions)),
            continuity=MappingProxyType(dict(self.continuity)),
            boundaries=MappingProxyType(dict(self.boundaries)),
            events=tuple(self.events),
            last_sequence=13 + len(self.events),
            last_event_sha256=(
                self.events[-1].event_sha256
                if self.events
                else digest("boundary-event")
            ),
        )

    def activate_suite(
        self,
        request_id,
        suite_id,
        *,
        expected_active_suite_id,
        expected_champion_sha256,
        boundary_id,
    ):
        self.activate_calls += 1
        if self.active_suite_id == suite_id:
            return self.events[-1]
        assert self.active_suite_id == expected_active_suite_id
        assert self.current.sha256 == expected_champion_sha256
        assert request_id == self.fixture.request_id
        assert suite_id == self.fixture.candidate_suite_id
        assert boundary_id == "boundary-6"
        payload = {
            "request_id": request_id,
            "suite_id": suite_id,
            "previous_suite_id": expected_active_suite_id,
            "expected_champion_sha256": expected_champion_sha256,
            "generation_id": self.current.generation_id,
            "boundary_id": boundary_id,
            "continuity_manifest_sha256": self.fixture.continuity_binding["sha256"],
        }
        event = SimpleNamespace(
            event_type="suite.activated",
            payload=payload,
            sequence=14,
            event_sha256=canonical_sha256(payload),
        )
        self.events.append(event)
        self.active_suite_id = suite_id
        if self.crash_after_cas:
            self.crash_after_cas = False
            raise RuntimeError("simulated registry crash after pointer CAS")
        return event


class RuntimeCommandRunner:
    def __init__(self, fixture):
        self.fixture = fixture
        self.calls = []
        self.fence_returncode = 0
        self.runtime_stdout_mutator = None

    def __call__(self, argv, **kwargs):
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "shell": False,
        }
        argv = list(argv)
        self.calls.append(argv)
        if argv == list(self.fixture.deployer_spec.controller_fence_argv):
            if self.fence_returncode:
                return subprocess.CompletedProcess(
                    argv, self.fence_returncode, "", "fence failed"
                )
            proof = build_fence_proof(self.fixture.deployment_request)
            return subprocess.CompletedProcess(
                argv, 0, canonical_json(proof) + "\n", ""
            )
        output = Path(argv[argv.index("--output-dir") + 1])
        result = self.fixture.write_runtime(output)
        if self.runtime_stdout_mutator is not None:
            result = self.runtime_stdout_mutator(dict(result))
        return subprocess.CompletedProcess(argv, 0, canonical_json(result) + "\n", "")


def make_fixture(tmp_path):
    inbox = tmp_path / "inbox"
    state_root = tmp_path / "deployer"
    systemd = tmp_path / "systemd"
    static = tmp_path / "static"
    suite_root = tmp_path / "candidate-suite"
    for directory in (inbox, systemd, static, suite_root / "schedules"):
        directory.mkdir(parents=True)

    registry_spec = static / "registry.json"
    registry_identity = digest("registry-identity")
    write_canonical(
        registry_spec,
        self_hashed(
            {
                "schema_version": 1,
                "contract": "test-registry-spec",
            },
            "spec_sha256",
        ),
    )
    service_spec = static / "suite-rotation-service.json"
    service_value = self_hashed(
        {
            "schema_version": 1,
            "contract": SERVICE_SPEC_CONTRACT,
            "actor": "suite-rotation-service",
        },
        "spec_sha256",
    )
    write_canonical(service_spec, service_value)

    rotation_request = self_hashed(
        {
            "schema_version": 1,
            "contract": "risk-score-evaluation-suite-rotation-request-v1",
            "request_id": "rotation-" + digest("rotation"),
        },
        "request_sha256",
    )
    request_id = rotation_request["request_id"]
    rotation_path = static / "rotation.json"
    write_canonical(rotation_path, rotation_request)
    rotation_binding = binding(rotation_path, rotation_request["request_sha256"])

    schedule_names = (
        "discovery",
        "confirmation",
        "audit",
        "lead-40-confirmation",
        "lead-80-confirmation",
    )
    banks = []
    for name in schedule_names:
        schedule = suite_root / "schedules" / f"{name}.jsonl"
        schedule.write_text(canonical_json({"schedule": name}) + "\n", encoding="utf-8")
        banks.append(
            {
                "qualifiedName": name,
                "schedule": {
                    "path": schedule.relative_to(suite_root).as_posix(),
                    "sha256": file_sha256(schedule),
                },
            }
        )
    suite_value = {
        "schemaVersion": 3,
        "manifestContract": "risk-score-authoritative-evaluation-manifest-v3",
        "banks": banks,
    }
    suite_value["manifestPayloadSha256"] = canonical_sha256(suite_value)
    suite_manifest = suite_root / "manifest.json"
    write_canonical(suite_manifest, suite_value)
    suite_identity = suite_value["manifestPayloadSha256"]
    candidate_suite_id = file_sha256(suite_manifest)
    candidate_version = digest("candidate-version")

    continuity_value = self_hashed(
        {
            "schema_version": 1,
            "contract": CONTINUITY_CONTRACT,
            "request_id": request_id,
            "candidate_suite_id": candidate_suite_id,
        },
        "manifest_sha256",
    )
    continuity = static / "continuity.json"
    write_canonical(continuity, continuity_value)
    continuity_binding = binding(continuity, continuity_value["manifest_sha256"])

    runtime_input_files = {}
    for name in (
        "autonomy-policy",
        "executor-spec",
        "adaptive-spec",
    ):
        path = static / f"{name}.json"
        write_canonical(path, {"name": name})
        runtime_input_files[name] = path
    for name in (
        "katago",
        "trainer-spec",
        "consumer-spec",
        "original-model",
        "checkpoint",
    ):
        path = static / name
        path.write_bytes(name.encode("utf-8"))
        runtime_input_files[name] = path
    repo = tmp_path / "repo"
    run_root = tmp_path / "run"
    repo.mkdir()
    run_root.mkdir()

    python = str(Path(sys.executable).resolve())
    service_argv = [
        python,
        "-m",
        "risk_score.suite_rotation_service",
        "watch",
        "--spec",
        str(service_spec),
    ]
    runtime_argv = [
        python,
        "-m",
        "risk_score.build_live_runtime",
        "--repo",
        str(repo),
        "--run-root",
        str(run_root),
        "--suite-dir",
        str(suite_root),
        "--katago-binary",
        str(runtime_input_files["katago"]),
        "--python-executable",
        python,
        "--trainer-spec",
        str(runtime_input_files["trainer-spec"]),
        "--consumer-spec",
        str(runtime_input_files["consumer-spec"]),
        "--original-model",
        str(runtime_input_files["original-model"]),
        "--trainer-checkpoint",
        str(runtime_input_files["checkpoint"]),
        "--gpu-uuid",
        "GPU-test",
        "--actor",
        "runtime-builder",
        "--source-revision",
        "a" * 40,
        "--output-dir",
        "{output_dir}",
        "--mutation-enabled",
        "--service-user",
        "katago",
        "--shuffler-command-json",
        canonical_json([python, "-c", "shuffler"]),
        "--exporter-command-json",
        canonical_json([python, "-c", "exporter"]),
        "--full-autonomy",
        "--cluster-executor-command-json",
        canonical_json([python, "-c", "executor"]),
        "--adaptive-training-command-json",
        canonical_json([python, "-c", "adaptive"]),
        "--suite-rotation-command-json",
        canonical_json(service_argv),
        "--autonomy-policy",
        str(runtime_input_files["autonomy-policy"]),
        "--cluster-executor-spec",
        str(runtime_input_files["executor-spec"]),
        "--adaptive-training-spec",
        str(runtime_input_files["adaptive-spec"]),
        "--suite-registry-spec",
        str(service_spec),
        "--evaluator-process-count",
        "8",
    ]
    champion_sha256 = digest("champion")
    generation_id = "generation-6"
    base_suite_id = digest("base-suite")
    runtime_inputs = {
        "candidate_suite_id": candidate_suite_id,
        "candidate_version_sha256": candidate_version,
        "candidate_manifest_path": str(suite_manifest),
        "candidate_manifest_sha256": file_sha256(suite_manifest),
        "candidate_manifest_identity": suite_identity,
        "materialization_receipt_sha256": digest("materialization"),
    }
    service_inputs = {
        "service_spec_path": str(service_spec),
        "service_spec_sha256": file_sha256(service_spec),
        "service_spec_identity": service_value["spec_sha256"],
        "registry_spec_path": str(registry_spec),
        "registry_spec_sha256": file_sha256(registry_spec),
    }
    commands = {
        "runtime": {
            "argv": runtime_argv,
            "argv_sha256": canonical_sha256(runtime_argv),
            "frozen_inputs": runtime_inputs,
            "frozen_inputs_sha256": canonical_sha256(runtime_inputs),
        },
        "service": {
            "argv": service_argv,
            "argv_sha256": canonical_sha256(service_argv),
            "executable_sha256": file_sha256(Path(python)),
            "frozen_inputs": service_inputs,
            "frozen_inputs_sha256": canonical_sha256(service_inputs),
        },
    }
    deployment_request = self_hashed(
        {
            "schema_version": 1,
            "contract": DEPLOYMENT_REQUEST_CONTRACT,
            "request_id": request_id,
            "actor": "suite-rotation-service",
            "created_at_utc": "2026-08-11T22:00:00.000000Z",
            "service_spec": binding(service_spec, service_value["spec_sha256"]),
            "registry_spec": binding(registry_spec, registry_identity),
            "rotation_request": rotation_binding,
            "candidate_suite": {
                "suite_id": candidate_suite_id,
                "version_sha256": candidate_version,
                "manifest": binding(suite_manifest, suite_identity),
            },
            "continuity": continuity_binding,
            "compare_and_swap": {
                "expected_active_suite_id": base_suite_id,
                "expected_champion_sha256": champion_sha256,
                "expected_generation_id": generation_id,
                "expected_pin_count": 0,
                "require_clean_generation_boundary": True,
                "boundary_must_follow_continuity": True,
                "continuity_event_sequence": 12,
            },
            "proposed_frozen_commands": commands,
            "proposed_frozen_commands_sha256": canonical_sha256(commands),
            "privilege_boundary": {
                "privileged_deployer_required": True,
                "service_may_activate_suite": False,
                "service_may_mutate_active_suite_pointer": False,
                "activation_api": "SuiteRotationRegistry.activate_suite",
            },
        },
        "request_sha256",
    )
    deployment_request_path = inbox / "deployment-request.json"
    write_canonical(deployment_request_path, deployment_request)

    deployer_spec_path = tmp_path / "deployer-spec.json"
    deployer_spec = publish_deployer_spec(
        deployer_spec_path,
        root=state_root,
        request_inbox=inbox,
        registry_spec_path=registry_spec,
        current_deployment=state_root / "current.json",
        activation_destination=systemd,
        controller_fence_argv=[python, "-c", "fence-controller"],
        activation_argv_template=[
            python,
            "-m",
            "risk_score.service_activation",
            "--spec",
            "{service_spec}",
            "--destination",
            "{activation_destination}",
            "--receipt",
            "{activation_receipt}",
            "--apply",
        ],
        activation_receipt_root=state_root / "activation-receipts",
        actor="root-autonomy-deployer",
        poll_interval_seconds=2.0,
    )
    fixture = SimpleNamespace(
        tmp=tmp_path,
        inbox=inbox,
        state_root=state_root,
        systemd=systemd,
        static=static,
        suite_root=suite_root,
        suite_manifest=suite_manifest,
        suite_identity=suite_identity,
        candidate_suite_id=candidate_suite_id,
        candidate_version=candidate_version,
        registry_spec=registry_spec,
        registry_identity=registry_identity,
        service_spec=service_spec,
        service_value=service_value,
        rotation_request=rotation_request,
        rotation_binding=rotation_binding,
        continuity=continuity,
        continuity_binding=continuity_binding,
        request_id=request_id,
        base_suite_id=base_suite_id,
        champion_sha256=champion_sha256,
        generation_id=generation_id,
        runtime_input_files=runtime_input_files,
        deployment_request=deployment_request,
        deployment_request_path=deployment_request_path,
        deployer_spec=deployer_spec,
        deployer_spec_path=deployer_spec_path,
    )

    def write_runtime(output):
        output.mkdir(parents=True, exist_ok=True)
        standard = output / "standard-confirmation.jsonl"
        confirmation = suite_root / "schedules" / "confirmation.jsonl"
        standard.write_bytes(confirmation.read_bytes())
        schedule_fields = {
            "discoveryOrdinarySchedule": "discovery",
            "confirmationOrdinarySchedule": "confirmation",
            "auditSchedule": "audit",
            "lead40Schedule": "lead-40-confirmation",
            "lead80Schedule": "lead-80-confirmation",
        }
        promotion_paths = {
            "suites": str(suite_root),
            **{
                key: str(suite_root / "schedules" / f"{name}.jsonl")
                for key, name in schedule_fields.items()
            },
            "standardConfirmationSchedule": str(standard),
        }
        promotion_hashes = {
            "suiteManifest": file_sha256(suite_manifest),
            **{
                key: file_sha256(Path(path))
                for key, path in promotion_paths.items()
                if key != "suites"
            },
        }
        promotion = {
            "schemaVersion": 1,
            "mutationEnabled": True,
            "paths": promotion_paths,
            "hashes": promotion_hashes,
        }
        promotion_path = output / "promotion-runtime.json"
        write_canonical(promotion_path, promotion)
        gpu_path = output / "gpu-lease-runtime.json"
        write_canonical(gpu_path, {"schemaVersion": 1, "mutationEnabled": True})

        units_dir = output / "systemd"
        units_dir.mkdir(exist_ok=True)
        unit_records = {}
        for key, unit_name in sorted(
            {**FULL_SERVICE_UNIT_NAMES, "target": TARGET_UNIT}.items()
        ):
            unit = units_dir / unit_name
            unit.write_text(
                f"[Unit]\nDescription={unit_name}\nPartOf={TARGET_UNIT}\n",
                encoding="utf-8",
            )
            unit_records[key] = binding(unit)
        service_input_map = {
            "autonomy_policy": runtime_input_files["autonomy-policy"],
            "executor_spec": runtime_input_files["executor-spec"],
            "adaptive_spec": runtime_input_files["adaptive-spec"],
            "suite_registry_spec": registry_spec,
        }
        service = {
            "schema_version": 3,
            "contract": AUTONOMY_SERVICE_SPEC_CONTRACT,
            "mutation_enabled": True,
            "full_autonomy": True,
            "evaluator_process_count": 8,
            "service_inputs": {
                name: binding(path) for name, path in service_input_map.items()
            },
            "services": {
                name: {"argv": [str(Path(sys.executable).resolve()), "-c", name]}
                for name in FULL_SERVICE_UNIT_NAMES
            },
            "systemd_units": unit_records,
        }
        service_path = output / "promotion-services.json"
        write_canonical(service_path, service)
        deployment_files = {
            "promotion_runtime": binding(promotion_path),
            "gpu_lease_runtime": binding(gpu_path),
            "service_spec": binding(service_path),
            "suite_manifest": binding(suite_manifest),
            "module:autonomy_deployer.py": binding(
                Path(__file__).parents[1] / "risk_score" / "autonomy_deployer.py"
            ),
        }
        deployment = {
            "schema_version": 1,
            "contract": "risk-score-live-runtime-deployment-v1",
            "source_revision": "a" * 40,
            "source_sha256": digest("a" * 40),
            "files": deployment_files,
        }
        deployment["manifest_sha256"] = canonical_sha256(deployment)
        deployment_path = output / "deployment-manifest.json"
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
            "evaluator_process_count": 8,
        }

    fixture.write_runtime = write_runtime
    fixture.registry = FakeRegistry(fixture)
    fixture.runner = RuntimeCommandRunner(fixture)
    fixture.systemctl_calls = []

    def systemctl(argv, **kwargs):
        fixture.systemctl_calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "active\n", "")

    fixture.systemctl = systemctl
    return fixture


def fake_registry_spec_loader(fixture):
    return lambda path: SimpleNamespace(
        path=path,
        file_sha256=file_sha256(path),
        identity=fixture.registry_identity,
    )


def fake_service_spec_loader(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        path=path,
        file_sha256=file_sha256(path),
        identity=raw["spec_sha256"],
        raw=raw,
    )


def fake_suite_validator(path, _spec, *, expected_champion_sha256):
    assert expected_champion_sha256
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        suite_id=file_sha256(path),
        manifest_sha256=file_sha256(path),
        manifest_identity=raw["manifestPayloadSha256"],
    )


def make_deployer(
    fixture,
    *,
    failure_hook=None,
    activation_applier=apply_service_activation,
):
    return AutonomyDeployer(
        fixture.deployer_spec,
        registry=fixture.registry,
        registry_spec_loader=fake_registry_spec_loader(fixture),
        service_spec_loader=fake_service_spec_loader,
        suite_validator=fake_suite_validator,
        promotion_runtime_loader=lambda _path: SimpleNamespace(),
        command_runner=fixture.runner,
        systemctl_runner=fixture.systemctl,
        activation_applier=activation_applier,
        failure_hook=failure_hook,
    )


def rewrite_request(fixture, mutator, *, rehash=False):
    value = json.loads(fixture.deployment_request_path.read_text(encoding="utf-8"))
    mutator(value)
    if rehash:
        value.pop("request_sha256", None)
        value["request_sha256"] = canonical_sha256(value)
    write_canonical(fixture.deployment_request_path, value)


def test_tampered_request_is_rejected_before_fence(tmp_path):
    fixture = make_fixture(tmp_path)
    rewrite_request(fixture, lambda value: value.__setitem__("actor", "tampered"))
    deployer = make_deployer(fixture)

    with pytest.raises(DeploymentRequestError, match="self-hash"):
        deployer.once()

    assert fixture.runner.calls == []
    assert fixture.registry.activate_calls == 0


@pytest.mark.parametrize("failure", ("suite", "champion", "pins", "boundary"))
def test_stale_registry_inputs_and_missing_boundary_fail_before_fence(
    tmp_path, failure
):
    fixture = make_fixture(tmp_path)
    if failure == "suite":
        fixture.registry.active_suite_id = digest("other-suite")
    elif failure == "champion":
        fixture.registry.current = SimpleNamespace(
            sha256=digest("other-champion"),
            generation_id=fixture.generation_id,
        )
    elif failure == "pins":
        fixture.registry.pins["evaluation-live"] = {"suite_id": fixture.base_suite_id}
    else:
        fixture.registry.boundaries.clear()
    deployer = make_deployer(fixture)

    with pytest.raises(DeploymentRequestError):
        deployer.once()

    assert fixture.runner.calls == []
    assert fixture.registry.activate_calls == 0


def test_fence_failure_halts_without_runtime_or_pointer(tmp_path):
    fixture = make_fixture(tmp_path)
    fixture.runner.fence_returncode = 9
    deployer = make_deployer(fixture)

    with pytest.raises(DeployerSafetyHalt, match="fence"):
        deployer.once()

    assert len(fixture.runner.calls) == 1
    assert fixture.registry.activate_calls == 0
    journals = list((fixture.state_root / "requests").glob("*/journal.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["state"] == "safety-halt"


def test_runtime_report_mismatch_halts_before_pointer_cas(tmp_path):
    fixture = make_fixture(tmp_path)

    def mismatch(result):
        result["evaluator_process_count"] = 16
        return result

    fixture.runner.runtime_stdout_mutator = mismatch
    deployer = make_deployer(fixture)

    with pytest.raises(DeployerSafetyHalt, match="stdout"):
        deployer.once()

    assert len(fixture.runner.calls) == 2
    assert fixture.registry.activate_calls == 0
    assert fixture.registry.active_suite_id == fixture.base_suite_id


@pytest.mark.parametrize("crash_stage", ("before-pointer-cas", "after-pointer-cas"))
def test_crash_before_or_after_pointer_cas_replays_without_rebuilding(
    tmp_path, crash_stage
):
    fixture = make_fixture(tmp_path)
    fired = []

    def crash(stage):
        if stage == crash_stage == "before-pointer-cas" and not fired:
            fired.append(stage)
            raise RuntimeError("simulated deployer crash")

    if crash_stage == "after-pointer-cas":
        fixture.registry.crash_after_cas = True
    deployer = make_deployer(fixture, failure_hook=crash)
    with pytest.raises(DeployerInterrupted):
        deployer.once()
    first_commands = list(fixture.runner.calls)
    assert len(first_commands) == 2
    if crash_stage == "before-pointer-cas":
        assert fixture.registry.active_suite_id == fixture.base_suite_id
    else:
        assert fixture.registry.active_suite_id == fixture.candidate_suite_id

    recovered = make_deployer(fixture).once()

    assert recovered["state"] == "active"
    assert fixture.runner.calls == first_commands
    assert fixture.registry.active_suite_id == fixture.candidate_suite_id
    assert len(fixture.registry.events) == 1


def test_apply_failure_after_pointer_cas_allows_exact_runtime_replay(tmp_path):
    fixture = make_fixture(tmp_path)
    attempts = []

    def fail_once(**kwargs):
        attempts.append(kwargs["spec_path"])
        if len(attempts) == 1:
            raise RuntimeError("systemd temporarily unavailable")
        return apply_service_activation(**kwargs)

    deployer = make_deployer(fixture, activation_applier=fail_once)
    with pytest.raises(ActivationRetryableHalt, match="exact new runtime"):
        deployer.once()

    assert fixture.registry.active_suite_id == fixture.candidate_suite_id
    assert len(fixture.runner.calls) == 2
    journal = json.loads(
        next((fixture.state_root / "requests").glob("*/journal.json")).read_text()
    )
    assert journal["state"] == "activation-halt"
    halt = json.loads(
        next(
            (fixture.state_root / "requests").glob("*/retryable-halt.json")
        ).read_text()
    )
    assert halt["old_controller_remains_fenced"] is True
    assert halt["pointer_revert_permitted"] is False

    recovered = make_deployer(fixture, activation_applier=fail_once).once()

    assert recovered["state"] == "active"
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert len(fixture.runner.calls) == 2
    assert len(fixture.registry.events) == 1


def test_success_and_exact_idempotence_activate_all_full_v3_units(tmp_path):
    fixture = make_fixture(tmp_path)
    deployer = make_deployer(fixture)

    first = deployer.once()
    command_count = len(fixture.runner.calls)
    systemctl_count = len(fixture.systemctl_calls)
    activation_calls = fixture.registry.activate_calls
    second = deployer.once()

    assert first["state"] == second["state"] == "active"
    assert fixture.registry.active_suite_id == fixture.candidate_suite_id
    assert len(fixture.runner.calls) == command_count == 2
    assert len(fixture.systemctl_calls) == systemctl_count
    assert fixture.registry.activate_calls == activation_calls == 1
    current = json.loads(fixture.deployer_spec.current_deployment.read_text())
    assert current["candidate_suite_id"] == fixture.candidate_suite_id
    installed = {path.name for path in fixture.systemd.iterdir()}
    assert installed == set(REQUIRED_ACTIVATION_UNITS)
    checked = {
        argv[-1]
        for argv, _ in fixture.systemctl_calls
        if argv[:2] == ["systemctl", "is-active"]
    }
    assert checked == set(REQUIRED_ACTIVATION_UNITS)
    assert all(kwargs["shell"] is False for _, kwargs in fixture.systemctl_calls)


def test_root_unit_is_hash_pinned_and_never_partof_runtime_target(tmp_path):
    fixture = make_fixture(tmp_path)
    unit = render_deployer_systemd_unit(
        python_executable=Path(sys.executable),
        working_directory=Path(__file__).parents[1],
        spec_path=fixture.deployer_spec_path,
    )

    assert "Description=KataGo privileged suite/runtime deployer" in unit
    assert "User=root" in unit
    assert "WantedBy=multi-user.target" in unit
    assert f"Before={TARGET_UNIT}" in unit
    assert "PartOf=" not in unit
    assert "--expected-spec-sha256" in unit
    assert file_sha256(fixture.deployer_spec_path) in unit
    assert DEPLOYER_UNIT_NAME not in {
        *FULL_SERVICE_UNIT_NAMES.values(),
        TARGET_UNIT,
    }
