import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import risk_score.build_live_runtime as live_runtime_builder
from risk_score.adaptive_training import POLICY_HASH as AUTONOMY_POLICY_HASH
from risk_score.build_live_runtime import (
    build_live_runtime,
    verify_deployment_manifest,
)
from risk_score.position_samples import (
    canonical_json,
    canonical_sha256,
    file_sha256,
)
from risk_score.promotion_host import HostCommandError
from risk_score.service_activation import (
    AUTONOMY_SERVICE_SPEC_CONTRACT,
    AUTONOMY_SERVICE_UNIT_NAMES,
)
from risk_score.suite_rotation import publish_registry_spec

REPO = Path(__file__).resolve().parents[2]

SYSTEMD_SERVICE_UNITS = {
    "supervisor": "katago-risk-promotion-host.service",
    "controller": "katago-risk-promotion-controller.service",
    "auditor": "katago-risk-promotion-auditor.service",
    "feedback": "katago-risk-promotion-feedback.service",
    "shuffler": "katago-risk-shuffler.service",
    "exporter": "katago-risk-exporter.service",
    "reconcile": "katago-risk-boot-reconcile.service",
}
FULL_SYSTEMD_SERVICE_UNITS = {
    **SYSTEMD_SERVICE_UNITS,
    **AUTONOMY_SERVICE_UNIT_NAMES,
}


def _minimal_runtime_build_kwargs(tmp_path):
    run = tmp_path / "run"
    (run / "modelstobetested").mkdir(parents=True)
    (run / "selfplay").mkdir()
    (run / "promotion").mkdir()
    original = run / "original" / "model.bin.gz"
    original.parent.mkdir()
    original.write_bytes(b"original")
    checkpoint = run / "train" / "riskb40" / "checkpoint.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    katago = tmp_path / "katago"
    katago.write_bytes(b"binary")
    trainer_spec = run / "inputs" / "trainer-launch.json"
    trainer_spec.parent.mkdir()
    trainer_spec.write_text("{}\n", encoding="utf-8")
    consumer_spec = run / "inputs" / "consumer-stop.json"
    consumer_spec.write_text("{}\n", encoding="utf-8")
    suites = run / "evaluation" / "promotion-suites-v2"
    (suites / "schedules").mkdir(parents=True)
    for relative in (
        "manifest.json",
        "schedules/discovery.jsonl",
        "schedules/confirmation.jsonl",
        "schedules/audit.jsonl",
        "schedules/lead-40-confirmation.jsonl",
        "schedules/lead-80-confirmation.jsonl",
    ):
        path = suites / relative
        path.write_text(relative + "\n", encoding="utf-8")
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repo": REPO,
        "run_root": run,
        "suite_dir": suites,
        "katago_binary": katago,
        "python_executable": Path(sys.executable),
        "trainer_spec": trainer_spec,
        "consumer_spec": consumer_spec,
        "original_model": original,
        "trainer_checkpoint": checkpoint,
        "gpu_uuid": "GPU-test-production",
        "actor": "controller-test",
        "source_revision": revision,
        "require_clean_source": False,
    }


def _promotion_evaluator_command():
    example = json.loads(
        (REPO / "python" / "risk_score" / "promotion_runtime.example.json").read_text(
            encoding="utf-8"
        )
    )
    return example["commands"]["evaluator"]


def _lease_evaluator_command(tmp_path):
    return [
        str(tmp_path / "run-promotion-evaluator-shard"),
        "--lease-id",
        "{lease_id}",
        "--worker-index",
        "{worker_index}",
        "--gpu-index",
        "{gpu_index}",
        "--expected-gpu-uuid",
        "{gpu_uuid}",
        "--promotion-root",
        "{promotion_root}",
    ]


def _ignore_ownership(**_kwargs):
    return None


def _root_lstat(path):
    values = list(os.lstat(path))
    values[4] = 0
    values[5] = 0
    return os.stat_result(values)


def _ownership_lstat(control_root, uid=123, gid=456, writable_generated_ancestors=()):
    def owned_lstat(path):
        values = list(os.lstat(path))
        candidate = Path(path)
        if candidate == control_root or candidate.is_relative_to(control_root):
            values[4] = uid
            values[5] = gid
        else:
            values[4] = 0
            values[5] = 0
            if stat.S_ISDIR(values[0]):
                values[0] = stat.S_IFDIR | (
                    0o777 if candidate in writable_generated_ancestors else 0o755
                )
        return os.stat_result(values)

    return owned_lstat


@pytest.mark.parametrize(
    ("selected_processes", "stage1_processes"),
    ((4, 4), (8, 8), (16, 8)),
)
def test_bootstrap_deployer_topology_and_stage1_cap_are_compatible(
    tmp_path, selected_processes, stage1_processes
):
    kwargs = _minimal_runtime_build_kwargs(tmp_path)
    lease_command = _lease_evaluator_command(tmp_path)
    result = build_live_runtime(
        **kwargs,
        output_dir=kwargs["run_root"] / "configs" / f"p{selected_processes}",
        evaluator_command=lease_command,
        evaluator_process_count=selected_processes,
    )
    promotion = json.loads(
        Path(result["promotion_runtime"]).read_text(encoding="utf-8")
    )
    gpu = json.loads(Path(result["gpu_lease_runtime"]).read_text(encoding="utf-8"))
    services = json.loads(Path(result["service_spec"]).read_text(encoding="utf-8"))
    deployment = json.loads(
        Path(result["deployment_manifest"]).read_text(encoding="utf-8")
    )
    expected_binding = {
        "benchmark_selected_process_count": selected_processes,
        "policy_maximum_evaluator_processes": 8,
        "effective_process_count": stage1_processes,
    }

    evaluator = promotion["commands"]["evaluator"]
    assert evaluator[evaluator.index("--shards") + 1] == str(stage1_processes)
    assert evaluator[evaluator.index("--max-parallel") + 1] == str(stage1_processes)
    assert gpu["evaluator"]["processCount"] == selected_processes
    assert gpu["evaluator"]["launchCommand"] == lease_command
    assert promotion["hashes"]["gpuLeaseConfig"] == file_sha256(
        Path(result["gpu_lease_runtime"])
    )
    assert services["evaluator_process_count"] == selected_processes
    assert services["stage1_evaluator_parallelism"] == expected_binding
    assert deployment["evaluator_process_count"] == selected_processes
    assert deployment["stage1_evaluator_parallelism"] == expected_binding
    assert result["evaluator_process_count"] == selected_processes
    assert set(result) == {
        "promotion_runtime",
        "promotion_runtime_sha256",
        "gpu_lease_runtime",
        "gpu_lease_runtime_sha256",
        "deployment_manifest",
        "deployment_manifest_sha256",
        "service_spec",
        "service_spec_sha256",
        "systemd_units",
        "mutation_enabled",
        "full_autonomy",
        "evaluator_process_count",
    }
    assert deployment["files"]["promotion_runtime"]["sha256"] == file_sha256(
        Path(result["promotion_runtime"])
    )
    assert deployment["files"]["gpu_lease_runtime"]["sha256"] == file_sha256(
        Path(result["gpu_lease_runtime"])
    )
    assert deployment["files"]["service_spec"]["sha256"] == file_sha256(
        Path(result["service_spec"])
    )
    deployment_body = dict(deployment)
    manifest_sha256 = deployment_body.pop("manifest_sha256")
    assert manifest_sha256 == canonical_sha256(deployment_body)
    assert verify_deployment_manifest(Path(result["deployment_manifest"])) == deployment


def test_service_ownership_is_a_nonroot_noop(tmp_path):
    generated = tmp_path / "generated.json"
    generated.write_text("{}\n", encoding="utf-8")
    control_root = tmp_path / "promotion"
    calls = []

    live_runtime_builder._ensure_service_user_ownership(
        service_user="service-test",
        generated_root=tmp_path,
        generated_paths=(generated,),
        control_root=control_root,
        mutable_directories=(control_root / "supervisor",),
        mutable_files=(control_root / "controller.lock",),
        geteuid=lambda: 501,
        user_lookup=lambda _name: pytest.fail("nonroot lookup"),
    )

    assert calls == []
    assert not control_root.exists()


def test_builder_scopes_injected_service_ownership(tmp_path):
    kwargs = _minimal_runtime_build_kwargs(tmp_path)
    output = kwargs["run_root"] / "generated-runtime"
    captured = {}

    result = build_live_runtime(
        **kwargs,
        output_dir=output,
        service_user="service-test",
        ownership_helper=lambda **values: captured.update(values),
    )

    generated_paths = set(captured["generated_paths"])
    assert captured["service_user"] == "service-test"
    assert captured["generated_root"] == output
    assert captured["control_root"] == kwargs["run_root"] / "promotion"
    assert {
        output,
        Path(result["promotion_runtime"]),
        Path(result["gpu_lease_runtime"]),
        Path(result["service_spec"]),
        Path(result["deployment_manifest"]),
    } <= generated_paths
    assert {
        kwargs["trainer_spec"],
        kwargs["consumer_spec"],
        kwargs["original_model"],
        kwargs["trainer_checkpoint"],
        kwargs["suite_dir"],
    }.isdisjoint(generated_paths)
    assert all(
        directory.is_relative_to(captured["control_root"])
        for directory in captured["mutable_directories"]
    )
    assert captured["control_root"] / "events" in captured["mutable_directories"]
    assert captured["mutable_files"] == (captured["control_root"] / "controller.lock",)


def test_verification_accepts_service_owned_mutable_paths_without_mutation(
    tmp_path, monkeypatch
):
    immutable = tmp_path / "immutable"
    immutable.mkdir(mode=0o755)
    output = immutable / "generated"
    output.mkdir(mode=0o755)
    generated = output / "runtime.json"
    generated.write_text("{}\n", encoding="utf-8")
    generated.chmod(0o644)
    input_path = tmp_path / "model.bin.gz"
    input_path.write_bytes(b"model")
    control_root = tmp_path / "promotion"
    control_root.mkdir(mode=0o750)
    events = control_root / "events"
    events.mkdir(mode=0o750)
    controller_lock = control_root / "controller.lock"
    controller_lock.write_bytes(b"")
    controller_lock.chmod(0o600)
    account = type("Account", (), {"pw_uid": 123, "pw_gid": 456})()
    monkeypatch.setattr(
        live_runtime_builder.os,
        "chown",
        lambda *_args, **_kwargs: pytest.fail("unexpected chown"),
    )
    monkeypatch.setattr(
        live_runtime_builder.os,
        "chmod",
        lambda *_args, **_kwargs: pytest.fail("unexpected chmod"),
    )

    live_runtime_builder._ensure_service_user_ownership(
        service_user="service-test",
        generated_root=output,
        generated_paths=(output, generated),
        control_root=control_root,
        mutable_directories=(events,),
        mutable_files=(controller_lock,),
        geteuid=lambda: 0,
        user_lookup=lambda name: account if name == "service-test" else None,
        lstat=_ownership_lstat(control_root),
    )

    assert stat.S_IMODE(output.stat().st_mode) == 0o755
    assert stat.S_IMODE(generated.stat().st_mode) == 0o644
    assert stat.S_IMODE(events.stat().st_mode) == 0o750
    assert stat.S_IMODE(controller_lock.stat().st_mode) == 0o600
    assert input_path.read_bytes() == b"model"


def test_root_ownership_rejects_unknown_service_user(tmp_path):
    with pytest.raises(HostCommandError, match="does not exist"):
        live_runtime_builder._ensure_service_user_ownership(
            service_user="missing-user",
            generated_root=tmp_path,
            generated_paths=(),
            control_root=tmp_path / "promotion",
            mutable_directories=(),
            mutable_files=(),
            geteuid=lambda: 0,
            user_lookup=lambda _name: (_ for _ in ()).throw(KeyError()),
        )


def test_root_ownership_rejects_control_directory_escape(tmp_path):
    account = type("Account", (), {"pw_uid": 123, "pw_gid": 456})()
    immutable = tmp_path / "immutable"
    immutable.mkdir(mode=0o755)
    generated = immutable / "generated"
    generated.mkdir(mode=0o755)
    control_root = tmp_path / "promotion"
    control_root.mkdir(mode=0o750)
    with pytest.raises(HostCommandError, match="escapes"):
        live_runtime_builder._ensure_service_user_ownership(
            service_user="service-test",
            generated_root=generated,
            generated_paths=(generated,),
            control_root=control_root,
            mutable_directories=(control_root / ".." / "outside",),
            mutable_files=(),
            geteuid=lambda: 0,
            user_lookup=lambda _name: account,
            lstat=_ownership_lstat(control_root),
        )


def test_verification_rejects_writable_generated_parent(tmp_path):
    account = type("Account", (), {"pw_uid": 123, "pw_gid": 456})()
    immutable = tmp_path / "immutable"
    immutable.mkdir(mode=0o777)
    immutable.chmod(0o777)
    generated = immutable / "generated"
    generated.mkdir(mode=0o755)
    control_root = tmp_path / "promotion"
    control_root.mkdir(mode=0o750)
    with pytest.raises(HostCommandError, match="trusted ancestor"):
        live_runtime_builder._ensure_service_user_ownership(
            service_user="service-test",
            generated_root=generated,
            generated_paths=(generated,),
            control_root=control_root,
            mutable_directories=(),
            mutable_files=(),
            geteuid=lambda: 0,
            user_lookup=lambda _name: account,
            lstat=_ownership_lstat(
                control_root, writable_generated_ancestors=(immutable,)
            ),
        )


def test_root_generation_rejects_untrusted_ancestor_before_creation(
    tmp_path, monkeypatch
):
    target = tmp_path / "generated"
    monkeypatch.setattr(live_runtime_builder.os, "geteuid", lambda: 0)

    with pytest.raises(HostCommandError, match="ancestor"):
        live_runtime_builder._ensure_generated_directory(target)

    assert not target.exists()


def test_verification_requires_precreated_mutable_paths(tmp_path):
    immutable = tmp_path / "immutable"
    immutable.mkdir(mode=0o755)
    generated = immutable / "generated"
    generated.mkdir(mode=0o755)
    control_root = tmp_path / "promotion"
    control_root.mkdir(mode=0o750)
    account = type("Account", (), {"pw_uid": 123, "pw_gid": 456})()

    with pytest.raises(HostCommandError, match="pre-created"):
        live_runtime_builder._ensure_service_user_ownership(
            service_user="service-test",
            generated_root=generated,
            generated_paths=(generated,),
            control_root=control_root,
            mutable_directories=(control_root / "events",),
            mutable_files=(),
            geteuid=lambda: 0,
            user_lookup=lambda _name: account,
            lstat=_ownership_lstat(control_root),
        )


def test_root_ownership_rejects_mutable_symlink(tmp_path):
    immutable = tmp_path / "immutable"
    immutable.mkdir(mode=0o755)
    generated = immutable / "generated"
    generated.mkdir(mode=0o755)
    control_root = tmp_path / "promotion"
    control_root.mkdir(mode=0o750)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = control_root / "events"
    linked.symlink_to(outside, target_is_directory=True)
    account = type("Account", (), {"pw_uid": 123, "pw_gid": 456})()

    with pytest.raises(HostCommandError, match="unsafe"):
        live_runtime_builder._ensure_service_user_ownership(
            service_user="service-test",
            generated_root=generated,
            generated_paths=(generated,),
            control_root=control_root,
            mutable_directories=(linked,),
            mutable_files=(),
            geteuid=lambda: 0,
            user_lookup=lambda _name: account,
            lstat=_ownership_lstat(control_root),
        )


def test_verification_detects_generated_ancestor_swap(tmp_path):
    immutable = tmp_path / "immutable"
    immutable.mkdir(mode=0o755)
    generated_root = immutable / "generated"
    generated_root.mkdir(mode=0o755)
    generated = generated_root / "runtime.json"
    generated.write_text("{}\n", encoding="utf-8")
    generated.chmod(0o644)
    control_root = tmp_path / "promotion"
    control_root.mkdir(mode=0o750)
    account = type("Account", (), {"pw_uid": 123, "pw_gid": 456})()
    observations = 0
    owned_lstat = _ownership_lstat(control_root)

    def swapped_lstat(path):
        nonlocal observations
        status = owned_lstat(path)
        if Path(path) == immutable:
            observations += 1
            if observations > 1:
                values = list(status)
                values[1] += 1
                status = os.stat_result(values)
        return status

    with pytest.raises(HostCommandError, match="changed during verification"):
        live_runtime_builder._ensure_service_user_ownership(
            service_user="service-test",
            generated_root=generated_root,
            generated_paths=(generated_root, generated),
            control_root=control_root,
            mutable_directories=(),
            mutable_files=(),
            geteuid=lambda: 0,
            user_lookup=lambda _name: account,
            lstat=swapped_lstat,
        )


def test_stage1_policy_process_cap_rejects_missing_bool_and_nonpositive():
    valid = {
        "evaluation_stages": {
            "stage_1_cheap_paired_screen": {
                "maximum_evaluator_processes": 8,
            }
        }
    }
    assert live_runtime_builder._stage1_evaluator_process_cap(valid) == 8

    malformed = []
    for value in (True, 0, -1):
        candidate = json.loads(json.dumps(valid))
        candidate["evaluation_stages"]["stage_1_cheap_paired_screen"][
            "maximum_evaluator_processes"
        ] = value
        malformed.append(candidate)
    missing = json.loads(json.dumps(valid))
    del missing["evaluation_stages"]["stage_1_cheap_paired_screen"][
        "maximum_evaluator_processes"
    ]
    malformed.append(missing)

    for candidate in malformed:
        with pytest.raises(
            HostCommandError,
            match="maximum_evaluator_processes",
        ):
            live_runtime_builder._stage1_evaluator_process_cap(candidate)


def test_suite_rotation_spec_alias_rejects_contradictions(tmp_path):
    rotation = tmp_path / "suite-rotation.json"
    other = tmp_path / "other-rotation.json"
    assert live_runtime_builder._resolve_suite_rotation_spec(None, rotation) == rotation
    assert (
        live_runtime_builder._resolve_suite_rotation_spec(rotation, rotation)
        == rotation
    )
    with pytest.raises(HostCommandError, match="contradicts"):
        live_runtime_builder._resolve_suite_rotation_spec(rotation, other)


def test_stage1_policy_identity_must_match_frozen_runtime(tmp_path, monkeypatch):
    kwargs = _minimal_runtime_build_kwargs(tmp_path)
    policy = json.loads(
        (REPO / "python" / "risk_score" / "promotion_policy_v3.json").read_text(
            encoding="utf-8"
        )
    )
    policy["queue"]["important_queue_warning_depth"] += 1
    monkeypatch.setattr(live_runtime_builder, "load_policy", lambda _path: policy)

    with pytest.raises(HostCommandError, match="hash-bound frozen runtime"):
        build_live_runtime(
            **kwargs,
            output_dir=kwargs["run_root"] / "configs" / "policy-mismatch",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "exactly one --max-parallel"),
        ("malformed", "--max-parallel must be a positive integer"),
        ("excess", "exceeds the frozen Stage-1 policy cap"),
        ("incoherent", "--max-parallel may not exceed --shards"),
        ("opaque", "must invoke risk_score.promotion_evaluator"),
    ),
)
def test_generated_evaluator_command_contract_fails_closed(mutation, message):
    command = _promotion_evaluator_command()
    if mutation == "missing":
        index = command.index("--max-parallel")
        del command[index : index + 2]
    elif mutation == "malformed":
        command[command.index("--max-parallel") + 1] = "invalid"
    elif mutation == "excess":
        command[command.index("--shards") + 1] = "9"
        command[command.index("--max-parallel") + 1] = "9"
    elif mutation == "incoherent":
        command[command.index("--shards") + 1] = "1"
        command[command.index("--max-parallel") + 1] = "2"
    else:
        command[command.index("risk_score.promotion_evaluator")] = (
            "risk_score.unrelated"
        )

    with pytest.raises(HostCommandError, match=message):
        live_runtime_builder._configure_promotion_evaluator_argv(
            command,
            effective_processes=8,
            policy_maximum=8,
        )


def _assert_durable_systemd_runtime(result, services):
    expected_services = set(services["services"])
    expected_units = {FULL_SYSTEMD_SERVICE_UNITS[name] for name in expected_services}
    target_unit = Path(services["systemd_units"]["target"]["path"]).read_text(
        encoding="utf-8"
    )
    target_lines = target_unit.splitlines()
    wants = next(line for line in target_lines if line.startswith("Wants="))
    after = next(line for line in target_lines if line.startswith("After="))
    conflicts = next(line for line in target_lines if line.startswith("Conflicts="))
    assert set(wants.removeprefix("Wants=").split()) == expected_units
    assert set(after.removeprefix("After=").split()) == expected_units
    assert set(conflicts.removeprefix("Conflicts=").split()) == {
        "katago-risk-supplement-v3.service",
        "katago-risk-curation-pipeline-v3.service",
    }

    deployment = json.loads(
        Path(result["deployment_manifest"]).read_text(encoding="utf-8")
    )
    for service_name in expected_services:
        unit = services["systemd_units"][service_name]
        unit_path = Path(unit["path"])
        unit_lines = unit_path.read_text(encoding="utf-8").splitlines()
        assert "PartOf=katago-risk-training.target" in unit_lines
        if service_name == "reconcile":
            assert services["services"][service_name]["restart"] == "no"
            assert unit_lines.count("Restart=no") == 1
            assert "Type=oneshot" in unit_lines
            assert "RemainAfterExit=yes" in unit_lines
        else:
            assert services["services"][service_name]["restart"] == "always"
            assert unit_lines.count("Restart=always") == 1
            assert "Type=simple" in unit_lines
            assert "KillSignal=SIGINT" in unit_lines
            assert "KillMode=control-group" in unit_lines
            assert "TimeoutStopSec=300" in unit_lines
        assert file_sha256(unit_path) == unit["sha256"]
        assert deployment["files"][f"systemd:{service_name}"] == unit

    target = services["systemd_units"]["target"]
    assert file_sha256(Path(target["path"])) == target["sha256"]
    assert deployment["files"]["systemd:target"] == target


def test_live_runtime_builder_materializes_real_hashes_with_mutation_off(tmp_path):
    run = tmp_path / "run"
    (run / "modelstobetested").mkdir(parents=True)
    (run / "selfplay").mkdir()
    (run / "promotion").mkdir()
    original_dir = run / "original"
    original_dir.mkdir()
    original = original_dir / "model.bin.gz"
    original.write_bytes(b"original")
    train = run / "train" / "riskb40"
    train.mkdir(parents=True)
    checkpoint = train / "checkpoint.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    katago = tmp_path / "katago"
    katago.write_bytes(b"binary")
    trainer_spec = run / "configs" / "trainer-launch.json"
    trainer_spec.parent.mkdir(parents=True)
    trainer_spec.write_text(
        json.dumps(
            {
                "contract": "risk-score-host-trainer-spec-v1",
                "cwd": str((REPO / "python").resolve()),
                "argv": [
                    sys.executable,
                    "-c",
                    "pass",
                    "-stop-when-train-bucket-limited",
                    "-generation-provenance-dir",
                    str(run / "promotion" / "provenance" / "trainer"),
                    "-require-shuffle-provenance",
                    "{checkpoint_path}",
                ],
                "env": {"CUDA_VISIBLE_DEVICES": "7"},
                "logPath": str((run / "logs" / "trainer.log").resolve()),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    consumer_spec = run / "configs" / "consumer-stop.json"
    consumer_spec.write_text(
        json.dumps(
            {
                "contract": "risk-score-host-consumer-spec-v1",
                "identities": {
                    role: []
                    for role in (
                        "selfplay",
                        "shuffler",
                        "trainer",
                        "exporter",
                        "evaluator",
                    )
                },
                "runRoot": str(run.resolve()),
                "activeRoot": str((run / "selfplay" / "continuous").resolve()),
                "rollbackRoot": str((run / "promotion" / "rollback").resolve()),
                "supervisorStateRoot": str(
                    (run / "promotion" / "supervisor").resolve()
                ),
                "rolePatterns": {
                    "selfplay": ["katago selfplay"],
                    "shuffler": ["shuffle.py"],
                    "trainer": ["train.py"],
                    "exporter": ["export_model_for_selfplay"],
                    "evaluator": ["promotion_evaluator"],
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    suites = run / "evaluation" / "promotion-suites-v2"
    (suites / "schedules" / "prefixes").mkdir(parents=True)
    for relative in (
        "manifest.json",
        "schedules/discovery.jsonl",
        "schedules/confirmation.jsonl",
        "schedules/audit.jsonl",
        "schedules/lead-40-confirmation.jsonl",
        "schedules/lead-80-confirmation.jsonl",
        "schedules/prefixes/confirmation-pairs-128.jsonl",
    ):
        path = suites / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    output = run / "configs"
    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = build_live_runtime(
        repo=REPO,
        run_root=run,
        suite_dir=suites,
        katago_binary=katago,
        python_executable=Path(sys.executable),
        trainer_spec=trainer_spec,
        consumer_spec=consumer_spec,
        original_model=original,
        trainer_checkpoint=checkpoint,
        gpu_uuid="GPU-test-production",
        actor="controller-test",
        source_revision=revision,
        output_dir=output,
        require_clean_source=False,
        service_user="ubuntu",
        ownership_helper=_ignore_ownership,
        shuffler_command=[sys.executable, "-c", "print('shuffle')"],
        exporter_command=[sys.executable, "-c", "print('export')"],
    )
    promotion_path = Path(result["promotion_runtime"])
    gpu_path = Path(result["gpu_lease_runtime"])
    service_path = Path(result["service_spec"])
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    services = json.loads(service_path.read_text(encoding="utf-8"))
    assert promotion["mutationEnabled"] is False
    assert gpu["mutationEnabled"] is False
    assert gpu["evaluator"]["launchCommand"][-1] == "evaluator-unsupported"
    assert promotion["hashes"]["gpuLeaseConfig"] == file_sha256(gpu_path)
    assert promotion["paths"]["candidateInbox"] == str(run / "modelstobetested")
    assert "risk_score.stage0_probe" in promotion["commands"]["stage0Probe"]
    assert "risk_score.promotion_host" in promotion["commands"]["selfplay"]
    assert services["contract"] == "risk-score-host-services-v2"
    boot_ready_path = run / "promotion" / "supervisor" / "boot-ready.json"
    supervisor_argv = services["services"]["supervisor"]["argv"]
    assert supervisor_argv[supervisor_argv.index("--boot-ready") + 1] == str(
        boot_ready_path
    )
    shadow_controller = services["services"]["controller"]["argv"]
    assert shadow_controller[-1] == "--recommend-only"
    assert shadow_controller[shadow_controller.index("--mode") + 1] == "watch"
    assert "--strict" not in services["services"]["feedback"]["argv"]
    assert (
        services["services"]["shuffler"]["environment"][
            "KATAGO_STRICT_SHUFFLE_PROVENANCE"
        ]
        == "0"
    )
    probe = json.loads(
        services["services"]["exporter"]["environment"][
            "KATAGO_MODEL_PROBE_COMMAND_JSON"
        ]
    )
    assert probe[probe.index("--katago") + 1] == str(katago)
    assert set(services["services"]) == {
        "controller",
        "exporter",
        "feedback",
        "shuffler",
        "supervisor",
    }
    assert set(services["systemd_units"]) == {
        "controller",
        "exporter",
        "feedback",
        "shuffler",
        "supervisor",
        "target",
    }
    assert services["mutation_enabled"] is False
    assert "reconcile" not in services["services"]
    _assert_durable_systemd_runtime(result, services)
    controller_unit = Path(services["systemd_units"]["controller"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "User=ubuntu" in controller_unit
    assert "Requires=katago-risk-promotion-host.service" in controller_unit
    with pytest.raises(HostCommandError, match="automatic runtime requires"):
        build_live_runtime(
            repo=REPO,
            run_root=run,
            suite_dir=suites,
            katago_binary=katago,
            python_executable=Path(sys.executable),
            trainer_spec=trainer_spec,
            consumer_spec=consumer_spec,
            original_model=original,
            trainer_checkpoint=checkpoint,
            gpu_uuid="GPU-test-production",
            actor="controller-test",
            source_revision=revision,
            output_dir=run / "invalid-automatic-configs",
            mutation_enabled=True,
            require_clean_source=False,
        )
    automatic_without_legacy_evaluator = build_live_runtime(
        repo=REPO,
        run_root=run,
        suite_dir=suites,
        katago_binary=katago,
        python_executable=Path(sys.executable),
        trainer_spec=trainer_spec,
        consumer_spec=consumer_spec,
        original_model=original,
        trainer_checkpoint=checkpoint,
        gpu_uuid="GPU-test-production",
        actor="controller-test",
        source_revision=revision,
        output_dir=run / "missing-evaluator-configs",
        mutation_enabled=True,
        require_clean_source=False,
        service_user="ubuntu",
        ownership_helper=_ignore_ownership,
        shuffler_command=[sys.executable, "-c", "print('shuffle')"],
        exporter_command=[sys.executable, "-c", "print('export')"],
    )
    automatic_without_legacy_gpu = json.loads(
        Path(automatic_without_legacy_evaluator["gpu_lease_runtime"]).read_text(
            encoding="utf-8"
        )
    )
    assert (
        automatic_without_legacy_gpu["evaluator"]["launchCommand"][-1]
        == "evaluator-unsupported"
    )
    with pytest.raises(HostCommandError, match="must not be evaluator-unsupported"):
        build_live_runtime(
            repo=REPO,
            run_root=run,
            suite_dir=suites,
            katago_binary=katago,
            python_executable=Path(sys.executable),
            trainer_spec=trainer_spec,
            consumer_spec=consumer_spec,
            original_model=original,
            trainer_checkpoint=checkpoint,
            gpu_uuid="GPU-test-production",
            actor="controller-test",
            source_revision=revision,
            output_dir=run / "unsupported-evaluator-configs",
            mutation_enabled=True,
            require_clean_source=False,
            service_user="ubuntu",
            ownership_helper=_ignore_ownership,
            shuffler_command=[sys.executable, "-c", "print('shuffle')"],
            exporter_command=[sys.executable, "-c", "print('export')"],
            evaluator_command=[
                sys.executable,
                "-m",
                "risk_score.promotion_host",
                "evaluator-unsupported",
            ],
        )
    evaluator_command = _lease_evaluator_command(run)
    shuffler_service = run / "configs" / "shuffler-loop"
    exporter_service = run / "configs" / "exporter-loop"
    shuffler_service.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exporter_service.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shuffler_service.chmod(0o755)
    exporter_service.chmod(0o755)
    automatic = build_live_runtime(
        repo=REPO,
        run_root=run,
        suite_dir=suites,
        katago_binary=katago,
        python_executable=Path(sys.executable),
        trainer_spec=trainer_spec,
        consumer_spec=consumer_spec,
        original_model=original,
        trainer_checkpoint=checkpoint,
        gpu_uuid="GPU-test-production",
        actor="controller-test",
        source_revision=revision,
        output_dir=run / "automatic-configs",
        mutation_enabled=True,
        require_clean_source=False,
        service_user="ubuntu",
        ownership_helper=_ignore_ownership,
        shuffler_command=[str(shuffler_service)],
        exporter_command=[str(exporter_service)],
        evaluator_command=evaluator_command,
    )
    automatic_services = json.loads(
        Path(automatic["service_spec"]).read_text(encoding="utf-8")
    )
    automatic_gpu = json.loads(
        Path(automatic["gpu_lease_runtime"]).read_text(encoding="utf-8")
    )
    automatic_deployment = json.loads(
        Path(automatic["deployment_manifest"]).read_text(encoding="utf-8")
    )
    assert automatic_services["mutation_enabled"] is True
    assert set(automatic_services["services"]) == set(SYSTEMD_SERVICE_UNITS)
    assert automatic_gpu["evaluator"]["launchCommand"] == evaluator_command
    assert "evaluator-unsupported" not in automatic_gpu["evaluator"]["launchCommand"]
    assert automatic_deployment["files"]["command:shuffler:0"] == {
        "path": str(shuffler_service.resolve()),
        "sha256": file_sha256(shuffler_service),
    }
    assert automatic_deployment["files"]["command:exporter:0"] == {
        "path": str(exporter_service.resolve()),
        "sha256": file_sha256(exporter_service),
    }
    auditor_argv = automatic_services["services"]["auditor"]["argv"]
    assert auditor_argv[auditor_argv.index("--katago") + 1] == str(katago)
    controller_argv = automatic_services["services"]["controller"]["argv"]
    assert "--automatic" in controller_argv
    assert controller_argv[-2:] == [
        "--status-output",
        str(run / "promotion" / "status.json"),
    ]
    assert "--strict" in automatic_services["services"]["feedback"]["argv"]
    assert (
        automatic_services["services"]["shuffler"]["environment"][
            "KATAGO_STRICT_SHUFFLE_PROVENANCE"
        ]
        == "1"
    )
    reconcile = automatic_services["services"]["reconcile"]
    commands = reconcile["commands"]
    assert [command[2] for command in commands] == [
        "risk_score.promotion_feedback",
        "risk_score.gpu_lease",
        "risk_score.promotion_controller",
        "risk_score.promotion_host",
    ]
    assert commands[0][commands[0].index("--mode") + 1] == "once"
    assert "--strict" in commands[0]
    assert commands[1][-2:] == ["reconcile", "--apply"]
    assert commands[2][commands[2].index("--mode") + 1] == "reconcile"
    assert "--automatic" in commands[2]
    assert commands[2][-2:] == [
        "--status-output",
        str(run / "promotion" / "status.json"),
    ]
    assert commands[3][3] == "boot-ready"
    assert commands[3][-4:] == [
        "--runtime-config",
        str(automatic["promotion_runtime"]),
        "--output",
        str(boot_ready_path),
    ]
    reconcile_unit = Path(
        automatic_services["systemd_units"]["reconcile"]["path"]
    ).read_text(encoding="utf-8")
    reconcile_lines = reconcile_unit.splitlines()
    assert reconcile_lines.count("Type=oneshot") == 1
    assert reconcile_lines.count("RemainAfterExit=yes") == 1
    assert reconcile_lines.count("Restart=no") == 1
    assert len([line for line in reconcile_lines if line.startswith("ExecStart=")]) == 4
    assert "Requires=katago-risk-promotion-host.service" in reconcile_lines
    before = next(line for line in reconcile_lines if line.startswith("Before="))
    assert set(before.removeprefix("Before=").split()) == {
        unit_name
        for name, unit_name in SYSTEMD_SERVICE_UNITS.items()
        if name not in {"supervisor", "reconcile"}
    }
    for name in set(automatic_services["services"]) - {
        "supervisor",
        "reconcile",
    }:
        unit_lines = (
            Path(automatic_services["systemd_units"][name]["path"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
        requires = next(line for line in unit_lines if line.startswith("Requires="))
        assert "katago-risk-boot-reconcile.service" in (
            requires.removeprefix("Requires=").split()
        )
    supervisor_unit = Path(
        automatic_services["systemd_units"]["supervisor"]["path"]
    ).read_text(encoding="utf-8")
    assert "katago-risk-boot-reconcile.service" not in supervisor_unit
    _assert_durable_systemd_runtime(automatic, automatic_services)

    autonomy_inputs = run / "autonomy-inputs"
    autonomy_inputs.mkdir()
    autonomy_policy = (
        REPO / "python" / "risk_score" / "autonomy_policy_v1.json"
    ).resolve()
    executor_spec = autonomy_inputs / "cluster-executor.json"
    adaptive_spec = autonomy_inputs / "adaptive-training.json"
    suite_rotation_spec = autonomy_inputs / "suite-rotation.json"
    scheduler_directory = (autonomy_inputs / "scheduler").resolve()
    scheduler_directory.mkdir()
    guardian_prefix = [
        sys.executable,
        "-m",
        "risk_score.gpu_lease_worker",
        "--spec-sha256",
        "a" * 64,
        "--receipt",
        "{guardian_receipt}",
        "--claim-id",
        "{claim_id}",
        "--work-id",
        "{work_id}",
        "--",
    ]

    def write_self_hashed(path, value):
        value["spec_sha256"] = canonical_sha256(value)
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    write_self_hashed(
        executor_spec,
        {
            "schema_version": 1,
            "contract": "risk-score-cluster-executor-spec-v1",
            "scheduler_directory": str(scheduler_directory),
            "state_directory": str((autonomy_inputs / "executor-state").resolve()),
            "owner_id": "executor-test",
            "gpu_ids": [str(index) for index in range(8)],
            "gpu7_id": "7",
            "poll_interval_seconds": 1.0,
            "heartbeat_interval_seconds": 2.0,
            "stale_after_seconds": 10.0,
            "retry_budget": 2,
            "backoff_initial_seconds": 5.0,
            "backoff_max_seconds": 60.0,
            "lease_proof_command": None,
            "lease_proof_timeout_seconds": 10.0,
            "gpu7_guardian_prefix": guardian_prefix,
        },
    )
    write_self_hashed(
        adaptive_spec,
        {
            "schema_version": 1,
            "contract": "risk-score-adaptive-training-service-spec-v1",
            "root": str((autonomy_inputs / "adaptive-root").resolve()),
            "autonomy_policy_path": str(autonomy_policy),
            "autonomy_policy_sha256": AUTONOMY_POLICY_HASH,
            "scheduler_directory": str(scheduler_directory),
            "gpu7_id": "7",
            "observation_path": str(
                (autonomy_inputs / "adaptive-observation.json").resolve()
            ),
            "trial_command_argv_template": [
                "/bin/true",
                "{trial_manifest}",
                "{trial_result}",
                "{work_id}",
            ],
            "gpu_lease_guardian_argv_prefix": guardian_prefix,
            "poll_interval_seconds": 5.0,
            "actor": "adaptive-test",
        },
    )
    registry_root = (autonomy_inputs / "suite-registry").resolve()
    registry_root.mkdir()
    registry_spec = autonomy_inputs / "suite-registry.json"
    initial_suite_champion = autonomy_inputs / "initial-suite-champion.bin.gz"
    initial_suite_champion.write_bytes(b"initial-suite-champion")
    registry = publish_registry_spec(
        registry_spec,
        registry_root=registry_root,
        policy_path=(REPO / "python" / "risk_score" / "promotion_policy_v3.json"),
        original_model_path=original,
        initial_champion_path=initial_suite_champion,
        initial_generation_id="generation-initial",
    )
    suite_id = file_sha256(suites / "manifest.json")
    full_suites = registry_root / "suites" / suite_id
    shutil.copytree(suites, full_suites)
    active_suite = {
        "schema_version": 1,
        "contract": "risk-score-active-evaluation-suite-v1",
        "spec_sha256": registry.identity,
        "suite_id": suite_id,
        "version_sha256": "b" * 64,
        "manifest_path": str(full_suites / "manifest.json"),
        "manifest_sha256": suite_id,
        "manifest_identity": "c" * 64,
        "activated_at_utc": "2026-08-11T00:00:00.000000Z",
        "activation_champion_sha256": file_sha256(initial_suite_champion),
        "activation_generation_id": "generation-initial",
        "event_sequence": 1,
        "event_sha256": "d" * 64,
    }
    active_suite["record_sha256"] = canonical_sha256(active_suite)
    (registry_root / "active-suite.json").write_text(
        canonical_json(active_suite) + "\n",
        encoding="utf-8",
    )
    write_self_hashed(
        suite_rotation_spec,
        {
            "schema_version": 1,
            "contract": "risk-score-suite-rotation-service-spec-v1",
            "root": str((autonomy_inputs / "suite-service").resolve()),
            "registry_spec": {
                "path": str(registry_spec.resolve()),
                "sha256": file_sha256(registry_spec),
            },
            "scheduler_directory": str(scheduler_directory),
            "gpu7_id": "7",
            "guardian_argv_prefix": guardian_prefix,
        },
    )
    full = build_live_runtime(
        repo=REPO,
        run_root=run,
        suite_dir=full_suites,
        katago_binary=katago,
        python_executable=Path(sys.executable),
        trainer_spec=trainer_spec,
        consumer_spec=consumer_spec,
        original_model=original,
        trainer_checkpoint=checkpoint,
        gpu_uuid="GPU-test-production",
        actor="controller-test",
        source_revision=revision,
        output_dir=run / "full-autonomy-configs",
        mutation_enabled=True,
        require_clean_source=False,
        service_user="ubuntu",
        ownership_helper=_ignore_ownership,
        shuffler_command=[str(shuffler_service)],
        exporter_command=[str(exporter_service)],
        evaluator_command=evaluator_command,
        full_autonomy=True,
        cluster_executor_command=[
            sys.executable,
            "-m",
            "risk_score.cluster_executor",
            "--spec",
            str(executor_spec),
            "watch",
        ],
        adaptive_training_command=[
            sys.executable,
            "-m",
            "risk_score.adaptive_training",
            "--spec",
            str(adaptive_spec),
            "watch",
        ],
        suite_rotation_command=[
            sys.executable,
            "-m",
            "risk_score.suite_rotation_service",
            "--spec",
            str(suite_rotation_spec),
            "watch",
        ],
        autonomy_policy=autonomy_policy,
        cluster_executor_spec=executor_spec,
        adaptive_training_spec=adaptive_spec,
        suite_rotation_spec=suite_rotation_spec,
        evaluator_process_count=16,
    )
    full_services = json.loads(Path(full["service_spec"]).read_text(encoding="utf-8"))
    assert full["full_autonomy"] is True
    assert full["evaluator_process_count"] == 16
    assert "stage1_evaluator_parallelism" not in full
    assert full_services["schema_version"] == 3
    assert full_services["contract"] == AUTONOMY_SERVICE_SPEC_CONTRACT
    assert full_services["evaluator_process_count"] == 16
    assert full_services["stage1_evaluator_parallelism"] == {
        "benchmark_selected_process_count": 16,
        "policy_maximum_evaluator_processes": 8,
        "effective_process_count": 8,
    }
    assert set(full_services["services"]) == set(FULL_SYSTEMD_SERVICE_UNITS)
    assert set(full_services["systemd_units"]) == {
        *FULL_SYSTEMD_SERVICE_UNITS,
        "target",
    }
    assert set(full_services["service_inputs"]) == {
        "autonomy_policy",
        "executor_spec",
        "adaptive_spec",
        "suite_registry_spec",
    }
    assert full_services["service_inputs"]["suite_registry_spec"] == {
        "path": str(registry_spec),
        "sha256": file_sha256(registry_spec),
    }
    assert full_services["suite_rotation_spec"] == {
        "path": str(suite_rotation_spec),
        "sha256": file_sha256(suite_rotation_spec),
    }
    full_gpu = json.loads(Path(full["gpu_lease_runtime"]).read_text(encoding="utf-8"))
    assert full_gpu["evaluator"]["processCount"] == 16
    full_promotion = json.loads(
        Path(full["promotion_runtime"]).read_text(encoding="utf-8")
    )
    full_evaluator = full_promotion["commands"]["evaluator"]
    assert full_evaluator[full_evaluator.index("--shards") + 1] == "8"
    assert full_evaluator[full_evaluator.index("--max-parallel") + 1] == "8"
    assert full_promotion["schemaVersion"] == 2
    assert full_promotion["autonomy"]["activeSuitePointer"]["suiteId"] == suite_id
    assert full_promotion["paths"]["suites"] == str(full_suites)
    full_reconcile = full_services["services"]["reconcile"]["commands"]
    assert [command[command.index("-m") + 1] for command in full_reconcile] == [
        "risk_score.promotion_feedback",
        "risk_score.gpu_lease",
        "risk_score.cluster_executor",
        "risk_score.adaptive_training",
        "risk_score.suite_rotation_service",
        "risk_score.promotion_controller",
        "risk_score.promotion_host",
    ]
    assert all("watch" not in command for command in full_reconcile[2:5])
    assert all("once" in command for command in full_reconcile[2:5])
    for name in ("adaptive", "suite_rotation"):
        unit = Path(full_services["systemd_units"][name]["path"]).read_text(
            encoding="utf-8"
        )
        expected_dependencies = [
            "katago-risk-promotion-host.service",
            "katago-risk-boot-reconcile.service",
            "katago-risk-cluster-executor.service",
            "katago-risk-promotion-controller.service",
        ]
        if name == "suite_rotation":
            expected_dependencies.append("katago-risk-promotion-auditor.service")
        assert ("Requires=" + " ".join(expected_dependencies)) in unit
    _assert_durable_systemd_runtime(full, full_services)
    deployment = verify_deployment_manifest(Path(result["deployment_manifest"]))
    assert deployment["source_revision"] == revision
    katago.write_bytes(b"changed")
    with pytest.raises(HostCommandError, match="katago changed"):
        verify_deployment_manifest(Path(result["deployment_manifest"]))
