import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from katago.train.extreme_score_policy import (
    load_extreme_score_training_policy,
)
from risk_score.extreme_score_controller import initialize_accepted_state
from risk_score.extreme_score_league import (
    DEFAULT_POLICY_PATH,
    REQUEST_CONTRACT,
    _publish_immutable_json,
    build_plan,
    execute_worker,
    file_sha256,
)
from risk_score.extreme_score_provenance import (
    ExtremeScoreProvenanceError,
    run_extreme_shuffle,
    validate_extreme_shuffle_manifest,
)
from risk_score.promotion_feedback import load_shuffle_manifest


def _file_binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _model_binding(root, snapshot_id, payload):
    root.mkdir(parents=True)
    model = root / "model.bin.gz"
    model.write_bytes(payload)
    return {
        "snapshot_id": snapshot_id,
        "directory": str(root.resolve()),
        "model_sha256": file_sha256(model),
    }


def _plan_request(tmp_path):
    binary = tmp_path / "katago"
    config = tmp_path / "extreme.cfg"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    config.write_text("useExpectedMaxScoreUtility = true\n", encoding="utf-8")
    focal = _model_binding(tmp_path / "focal", "focal-v1", b"focal")
    checkpoint = tmp_path / "initial.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    state_root = tmp_path / "controller-state"
    state_path = state_root / "accepted-current.json"
    accepted = initialize_accepted_state(
        model_id=focal["snapshot_id"],
        model_path=Path(focal["directory"]) / "model.bin.gz",
        checkpoint_path=checkpoint,
        training_policy_path=DEFAULT_POLICY_PATH,
        state_root=state_root,
        state_path=state_path,
        lock_path=state_root / "controller.lock",
    )
    opponents = [
        {
            "role": "latest_frozen",
            "weight": 0.5,
            "model": _model_binding(tmp_path / "latest", "latest-v1", b"latest"),
        },
        {
            "role": "recent_frozen",
            "weight": 0.3,
            "model": _model_binding(tmp_path / "recent", "recent-v1", b"recent"),
        },
        {
            "role": "score_minimizing_exploiter",
            "weight": 0.2,
            "model": _model_binding(
                tmp_path / "exploiter", "exploiter-v1", b"exploiter"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "contract": REQUEST_CONTRACT,
        "generation_id": "generation-1",
        "binary": _file_binding(binary),
        "config": _file_binding(config),
        "accepted_state": {
            "path": str(state_path.resolve()),
            "file_sha256": file_sha256(state_path),
            "state_sha256": accepted["state_sha256"],
        },
        "opponents": opponents,
        "gpu_indices": list(range(7)),
        "threads_per_worker": 50,
        "group_size": 1,
        "games_per_worker": 64,
        "output_root": str((tmp_path / "selfplay").resolve()),
    }


def _completed_fixture(tmp_path):
    plan = build_plan(_plan_request(tmp_path))
    plan_path = tmp_path / "league-plan.json"
    _publish_immutable_json(plan_path, plan)

    for worker in plan["workers"]:
        worker_id = worker["worker_id"]

        def executor(argv, worker_id=worker_id):
            output = Path(argv[argv.index("-output-dir") + 1])
            shard = output / "tdata" / f"{worker_id}.npz"
            shard.parent.mkdir()
            shard.write_bytes(worker_id.encode("utf-8"))
            return SimpleNamespace(returncode=0)

        execute_worker(plan, worker_id, executor=executor)

    shuffle_script = tmp_path / "shuffle.sh"
    shuffle_script.write_text("#!/bin/sh\n", encoding="utf-8")
    command_path = tmp_path / "shuffle-command.json"
    command_path.write_text(
        json.dumps(
            [
                str(shuffle_script.resolve()),
                str(tmp_path.resolve()),
                str((tmp_path / "scratch").resolve()),
                "4",
            ]
        ),
        encoding="utf-8",
    )
    return plan, plan_path, command_path


def _run_shuffle(tmp_path, plan_path, command_path, *, output_id="0001"):
    shuffled_root = tmp_path / "shuffleddata"
    shuffled_root.mkdir(exist_ok=True)

    def executor(_argv, *, env):
        assert env["KATAGO_SHUFFLE_GATE_BYPASS"] == "1"
        input_root = Path(env["KATAGO_SHUFFLE_INPUT_ROOT"])
        assert len(list(input_root.rglob("*.npz"))) == 14
        output = (
            Path(env["KATAGO_SHUFFLE_OUTPUT_ROOT"]) / env["KATAGO_SHUFFLE_OUTPUT_ID"]
        )
        (output / "train").mkdir(parents=True)
        (output / "train" / "batch.npz").write_bytes(b"shuffled")
        (output / "train.json").write_text(
            json.dumps({"range": [0, 1]}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    return run_extreme_shuffle(
        plan_paths=[plan_path],
        shuffle_command_json=command_path,
        shuffled_root=shuffled_root,
        lock_path=tmp_path / "shuffle.lock",
        output_id=output_id,
        claim_root=tmp_path / "shuffle-claims",
        executor=executor,
    )


def test_unplanned_source_npz_cannot_enter_claimed_shuffle(tmp_path):
    plan, plan_path, command_path = _completed_fixture(tmp_path)
    rogue = Path(plan["output_root"]) / "rogue" / "unplanned.npz"
    rogue.parent.mkdir()
    rogue.write_bytes(b"rogue")
    manifest = _run_shuffle(tmp_path, plan_path, command_path)
    bound = load_shuffle_manifest(manifest, verify_output=True, verify_sources=True)
    assert all("rogue" not in item["path"] for item in bound["source_inventory"])


def test_extreme_shuffle_binds_every_worker_policy_and_npz(tmp_path):
    plan, plan_path, command_path = _completed_fixture(tmp_path)
    manifest_path = _run_shuffle(tmp_path, plan_path, command_path)
    policy = load_extreme_score_training_policy(DEFAULT_POLICY_PATH, 1)
    extreme = validate_extreme_shuffle_manifest(
        manifest_path,
        expected_policy=policy,
        expected_cohort_size=1,
    )
    manifest = load_shuffle_manifest(
        manifest_path, verify_output=True, verify_sources=True
    )

    assert manifest["strict"] is True
    assert manifest["generation_ids"] == [plan["generation_id"]]
    assert len(manifest["source_inventory"]) == len(plan["workers"])
    assert len(extreme["worker_receipts"]) == len(plan["workers"])
    assert extreme["policy"] == plan["policy"]
    assert manifest_path.stat().st_mode & 0o222 == 0


def test_extreme_shuffle_rejects_missing_or_changed_worker_receipts(tmp_path):
    plan, plan_path, command_path = _completed_fixture(tmp_path)
    receipt = (
        Path(plan["workers"][0]["output_directory"]) / "worker-execution-receipt.json"
    )
    os.chmod(receipt.parent, 0o755)
    receipt.unlink()
    with pytest.raises((ExtremeScoreProvenanceError, ValueError), match="receipt"):
        _run_shuffle(tmp_path, plan_path, command_path)


def test_trainer_validation_detects_source_and_policy_drift(tmp_path):
    plan, plan_path, command_path = _completed_fixture(tmp_path)
    manifest_path = _run_shuffle(tmp_path, plan_path, command_path)
    policy = load_extreme_score_training_policy(DEFAULT_POLICY_PATH, 1)

    wrong_policy = dict(policy)
    wrong_policy["file_sha256"] = "0" * 64
    with pytest.raises(ExtremeScoreProvenanceError, match="policy differs"):
        validate_extreme_shuffle_manifest(
            manifest_path,
            expected_policy=wrong_policy,
            expected_cohort_size=1,
        )

    shard = Path(plan["workers"][0]["output_directory"]) / "tdata"
    os.chmod(shard.parent, 0o755)
    os.chmod(shard, 0o755)
    npz = next(shard.glob("*.npz"))
    os.chmod(npz, 0o644)
    npz.write_bytes(b"changed")
    with pytest.raises((ExtremeScoreProvenanceError, ValueError)):
        validate_extreme_shuffle_manifest(
            manifest_path,
            expected_policy=policy,
            expected_cohort_size=1,
        )


def test_bounded_shuffle_wrapper_identifies_and_binds_only_new_output(tmp_path):
    _, plan_path, command_path = _completed_fixture(tmp_path)
    manifest = _run_shuffle(tmp_path, plan_path, command_path, output_id="0002")
    assert manifest == tmp_path / "shuffleddata/0002/generation-provenance.json"

    resumed = run_extreme_shuffle(
        plan_paths=[plan_path],
        shuffle_command_json=command_path,
        shuffled_root=tmp_path / "shuffleddata",
        lock_path=tmp_path / "shuffle.lock",
        output_id="0002",
        claim_root=tmp_path / "shuffle-claims",
        executor=lambda *_args, **_kwargs: pytest.fail(
            "completed claimed shuffle must not execute again"
        ),
    )
    assert resumed == manifest


def test_claim_recovers_final_output_after_wrapper_crash(tmp_path):
    _, plan_path, command_path = _completed_fixture(tmp_path)
    shuffled_root = tmp_path / "shuffleddata"
    shuffled_root.mkdir()

    def crashing_executor(_argv, *, env):
        output = (
            Path(env["KATAGO_SHUFFLE_OUTPUT_ROOT"]) / env["KATAGO_SHUFFLE_OUTPUT_ID"]
        )
        (output / "train").mkdir(parents=True)
        (output / "train" / "batch.npz").write_bytes(b"batch")
        (output / "train.json").write_text(
            json.dumps({"range": [0, 1]}), encoding="utf-8"
        )
        raise OSError("lost wrapper after completed rename")

    kwargs = {
        "plan_paths": [plan_path],
        "shuffle_command_json": command_path,
        "shuffled_root": shuffled_root,
        "lock_path": tmp_path / "shuffle.lock",
        "output_id": "recoverable",
        "claim_root": tmp_path / "shuffle-claims",
    }
    with pytest.raises(OSError, match="lost wrapper"):
        run_extreme_shuffle(**kwargs, executor=crashing_executor)
    recovered = run_extreme_shuffle(
        **kwargs,
        executor=lambda *_args, **_kwargs: pytest.fail(
            "final claimed output should be recovered without rerun"
        ),
    )
    assert recovered == shuffled_root / "recoverable/generation-provenance.json"


def test_preexisting_unclaimed_shuffle_output_is_never_misattributed(tmp_path):
    _, plan_path, command_path = _completed_fixture(tmp_path)
    shuffled_root = tmp_path / "shuffleddata"
    unclaimed = shuffled_root / "collision"
    unclaimed.mkdir(parents=True)
    with pytest.raises(
        ExtremeScoreProvenanceError, match="without its pre-execution claim"
    ):
        run_extreme_shuffle(
            plan_paths=[plan_path],
            shuffle_command_json=command_path,
            shuffled_root=shuffled_root,
            lock_path=tmp_path / "shuffle.lock",
            output_id="collision",
            claim_root=tmp_path / "shuffle-claims",
            executor=lambda *_args, **_kwargs: pytest.fail(
                "unclaimed output must fail before execution"
            ),
        )


def test_stock_shuffle_script_supports_claimed_input_and_output_overrides():
    script = (
        Path(__file__).resolve().parents[1] / "selfplay" / "shuffle.sh"
    ).read_text(encoding="utf-8")
    for variable in (
        "KATAGO_SHUFFLE_GATE_BYPASS",
        "KATAGO_SHUFFLE_INPUT_ROOT",
        "KATAGO_SHUFFLE_OUTPUT_ROOT",
        "KATAGO_SHUFFLE_OUTPUT_ID",
    ):
        assert variable in script
