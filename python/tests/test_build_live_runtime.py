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
    )
    promotion_path = Path(result["promotion_runtime"])
    gpu_path = Path(result["gpu_lease_runtime"])
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    assert promotion["mutationEnabled"] is False
    assert gpu["mutationEnabled"] is False
    assert promotion["hashes"]["gpuLeaseConfig"] == file_sha256(gpu_path)
    assert promotion["paths"]["candidateInbox"] == str(run / "modelstobetested")
    assert "risk_score.stage0_probe" in promotion["commands"]["stage0Probe"]
    assert "risk_score.promotion_host" in promotion["commands"]["selfplay"]
    deployment = verify_deployment_manifest(Path(result["deployment_manifest"]))
    assert deployment["source_revision"] == revision
    katago.write_bytes(b"changed")
    with pytest.raises(HostCommandError, match="katago changed"):
        verify_deployment_manifest(Path(result["deployment_manifest"]))
