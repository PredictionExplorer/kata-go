import json
import subprocess
import sys
from pathlib import Path

import pytest

from risk_score.build_live_runtime import (
    build_live_runtime,
    verify_deployment_manifest,
)
from risk_score.promotion_host import HostCommandError
from risk_score.position_samples import file_sha256


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


def _assert_durable_systemd_runtime(result, services):
    expected_services = set(services["services"])
    expected_units = {
        SYSTEMD_SERVICE_UNITS[name] for name in expected_services
    }
    target_unit = Path(services["systemd_units"]["target"]["path"]).read_text(
        encoding="utf-8"
    )
    target_lines = target_unit.splitlines()
    wants = next(line for line in target_lines if line.startswith("Wants="))
    after = next(line for line in target_lines if line.startswith("After="))
    assert set(wants.removeprefix("Wants=").split()) == expected_units
    assert set(after.removeprefix("After=").split()) == expected_units

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
        shuffler_command=[sys.executable, "-c", "print('shuffle')"],
        exporter_command=[sys.executable, "-c", "print('export')"],
    )
    automatic_without_legacy_gpu = json.loads(
        Path(
            automatic_without_legacy_evaluator["gpu_lease_runtime"]
        ).read_text(encoding="utf-8")
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
            shuffler_command=[sys.executable, "-c", "print('shuffle')"],
            exporter_command=[sys.executable, "-c", "print('export')"],
            evaluator_command=[
                sys.executable,
                "-m",
                "risk_score.promotion_host",
                "evaluator-unsupported",
            ],
        )
    evaluator_command = [
        sys.executable,
        "-c",
        "print('evaluate')",
        "{lease_id}",
        "{worker_index}",
    ]
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
        shuffler_command=[sys.executable, "-c", "print('shuffle')"],
        exporter_command=[sys.executable, "-c", "print('export')"],
        evaluator_command=evaluator_command,
    )
    automatic_services = json.loads(
        Path(automatic["service_spec"]).read_text(encoding="utf-8")
    )
    automatic_gpu = json.loads(
        Path(automatic["gpu_lease_runtime"]).read_text(encoding="utf-8")
    )
    assert automatic_services["mutation_enabled"] is True
    assert set(automatic_services["services"]) == set(
        SYSTEMD_SERVICE_UNITS
    )
    assert automatic_gpu["evaluator"]["launchCommand"] == evaluator_command
    assert "evaluator-unsupported" not in automatic_gpu["evaluator"]["launchCommand"]
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
    assert len(
        [line for line in reconcile_lines if line.startswith("ExecStart=")]
    ) == 4
    assert "Requires=katago-risk-promotion-host.service" in reconcile_lines
    before = next(
        line for line in reconcile_lines if line.startswith("Before=")
    )
    assert set(before.removeprefix("Before=").split()) == {
        unit_name
        for name, unit_name in SYSTEMD_SERVICE_UNITS.items()
        if name not in {"supervisor", "reconcile"}
    }
    for name in set(automatic_services["services"]) - {
        "supervisor",
        "reconcile",
    }:
        unit_lines = Path(
            automatic_services["systemd_units"][name]["path"]
        ).read_text(encoding="utf-8").splitlines()
        requires = next(
            line for line in unit_lines if line.startswith("Requires=")
        )
        assert "katago-risk-boot-reconcile.service" in (
            requires.removeprefix("Requires=").split()
        )
    supervisor_unit = Path(
        automatic_services["systemd_units"]["supervisor"]["path"]
    ).read_text(encoding="utf-8")
    assert "katago-risk-boot-reconcile.service" not in supervisor_unit
    _assert_durable_systemd_runtime(automatic, automatic_services)
    deployment = verify_deployment_manifest(Path(result["deployment_manifest"]))
    assert deployment["source_revision"] == revision
    katago.write_bytes(b"changed")
    with pytest.raises(HostCommandError, match="katago changed"):
        verify_deployment_manifest(Path(result["deployment_manifest"]))
