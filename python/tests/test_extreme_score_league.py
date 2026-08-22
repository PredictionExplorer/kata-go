import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from risk_score.extreme_score_controller import (
    initialize_accepted_state,
    load_accepted_state,
)
from risk_score.extreme_score_league import (
    DEFAULT_POLICY_PATH,
    REQUEST_CONTRACT,
    WORKER_RECEIPT_CONTRACT,
    ExtremeScoreLeagueError,
    _publish_immutable_json,
    _ensure_output_parent,
    build_plan,
    canonical_json,
    execute_worker,
    file_sha256,
    status,
    validate_plan,
)


def file_binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def model_binding(root, snapshot_id, payload):
    root.mkdir(parents=True)
    model = root / "model.bin.gz"
    model.write_bytes(payload)
    return {
        "snapshot_id": snapshot_id,
        "directory": str(root.resolve()),
        "model_sha256": file_sha256(model),
    }


def request(
    tmp_path,
    *,
    group_size=8,
    selected_training_samples=8_000_000,
    gpu_indices=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "katago"
    config = tmp_path / "extreme.cfg"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    config.write_text("useExpectedMaxScoreUtility = true\n", encoding="utf-8")
    focal = model_binding(tmp_path / "focal", "focal-v1", b"focal")
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
        selected_training_samples=selected_training_samples,
    )
    opponents = [
        {
            "role": "latest_frozen",
            "weight": 0.5,
            "model": model_binding(tmp_path / "latest", "latest-v1", b"latest"),
        },
        {
            "role": "recent_frozen",
            "weight": 0.3,
            "model": model_binding(tmp_path / "recent", "recent-v1", b"recent"),
        },
        {
            "role": "score_minimizing_exploiter",
            "weight": 0.2,
            "model": model_binding(
                tmp_path / "exploiter", "exploiter-v1", b"exploiter"
            ),
        },
    ]
    output = tmp_path / "selfplay"
    return {
        "schema_version": 1,
        "contract": REQUEST_CONTRACT,
        "generation_id": "extreme-generation-1",
        "binary": file_binding(binary),
        "config": file_binding(config),
        "accepted_state": {
            "path": str(state_path.resolve()),
            "file_sha256": file_sha256(state_path),
            "state_sha256": accepted["state_sha256"],
        },
        "opponents": opponents,
        "gpu_indices": list(range(7)) if gpu_indices is None else gpu_indices,
        "threads_per_worker": 50,
        "group_size": group_size,
        "games_per_worker": 800,
        "output_root": str(output.resolve()),
    }


def overrides(argv):
    value = argv[argv.index("-override-config") + 1]
    return dict(item.split("=", 1) for item in value.split(","))


def output_directory(argv):
    return Path(argv[argv.index("-output-dir") + 1])


def test_balanced_workers_and_weighted_frozen_league(tmp_path):
    value = request(tmp_path)
    accepted = load_accepted_state(Path(value["accepted_state"]["path"]))
    source_focal = Path(accepted["artifact"]["path"]).parent
    plan = build_plan(value)

    assert validate_plan(plan) == plan
    assert plan["policy"]["path"] == str(DEFAULT_POLICY_PATH.resolve())
    assert plan["policy"]["schema_version"] == 1
    assert plan["policy"]["canonical_sha256"]
    assert plan["curriculum_state"]["stage_index"] == 3
    assert plan["curriculum_state"]["cohort_size"] == 8
    assert plan["curriculum_state"]["minimum_selected_training_samples"] == 8_000_000
    assert plan["curriculum_state"]["selected_training_samples"] == 8_000_000
    assert plan["curriculum_state"]["production"] is True
    assert len(plan["curriculum_state"]["curriculum_sha256"]) == 64
    assert len(plan["workers"]) == 14
    assert [worker["focal_color"] for worker in plan["workers"]].count("B") == 7
    assert [worker["focal_color"] for worker in plan["workers"]].count("W") == 7
    for color in ("B", "W"):
        roles = [
            worker["opponent_role"]
            for worker in plan["workers"]
            if worker["focal_color"] == color
        ]
        assert roles.count("latest_frozen") == 4
        assert roles.count("recent_frozen") == 2
        assert roles.count("score_minimizing_exploiter") == 1

    assert [
        item["workers_per_focal_color"]
        for item in plan["realized_allocation"]["opponents"]
    ] == [4, 2, 1]
    assert plan["realized_allocation"]["total_workers"] == 14

    for worker in plan["workers"]:
        argv = worker["katago_argv"]
        values = overrides(argv)
        assert argv[1] == "selfplay"
        assert "-opponent-models-dir" in argv
        assert argv[argv.index("-max-games-total") + 1] == "800"
        assert values["extremeCohortSize"] == "8"
        assert values["extremeScoreGroupSize"] == "8"
        assert values["extremeCohortFocalColor"] == worker["focal_color"]
        assert values["expectedMaxFocalColor"] == worker["focal_color"]
        assert values["useExpectedMaxScoreUtility"] == "true"
        assert values["winWeight"] == "0"

    snapshot = Path(plan["focal_model"]["directory"])
    assert snapshot != source_focal
    assert [path.name for path in snapshot.iterdir()] == ["model.bin.gz"]
    assert snapshot.stat().st_mode & 0o222 == 0
    assert (snapshot / "model.bin.gz").stat().st_mode & 0o222 == 0


def test_plan_detects_model_config_and_identity_drift(tmp_path):
    value = request(tmp_path)
    plan = build_plan(value)

    changed = json.loads(canonical_json(plan))
    changed["workers"][0]["focal_color"] = "W"
    with pytest.raises(ExtremeScoreLeagueError, match="self-hash"):
        validate_plan(changed)

    accepted = load_accepted_state(Path(value["accepted_state"]["path"]))
    accepted_model = Path(accepted["artifact"]["path"])
    os.chmod(accepted_model, 0o644)
    accepted_model.write_bytes(b"changed")
    with pytest.raises(ExtremeScoreLeagueError, match="changed"):
        build_plan(value)


def test_policy_roles_weights_threads_and_n_fail_closed(tmp_path):
    bad_weights = request(tmp_path / "weights")
    bad_weights["opponents"][0]["weight"] = 0.4
    with pytest.raises(ExtremeScoreLeagueError, match="weight differs"):
        build_plan(bad_weights)

    bad_role = request(tmp_path / "role")
    bad_role["opponents"][0]["role"] = "arbitrary"
    with pytest.raises(ExtremeScoreLeagueError, match="roles differ"):
        build_plan(bad_role)

    bad_threads = request(tmp_path / "threads")
    bad_threads["threads_per_worker"] = 49
    with pytest.raises(ExtremeScoreLeagueError, match="training policy"):
        build_plan(bad_threads)

    bad_group = request(
        tmp_path / "group",
        group_size=9,
        selected_training_samples=9_000_000,
    )
    with pytest.raises(ExtremeScoreLeagueError, match="group_size"):
        build_plan(bad_group)

    premature_group = request(
        tmp_path / "curriculum",
        group_size=8,
        selected_training_samples=4_000_000,
    )
    with pytest.raises(ExtremeScoreLeagueError, match="curriculum state"):
        build_plan(premature_group)

    partial_cohort = request(tmp_path / "partial")
    partial_cohort["games_per_worker"] = 801
    with pytest.raises(ExtremeScoreLeagueError, match="multiple"):
        build_plan(partial_cohort)

    unbounded_worker = request(tmp_path / "unbounded")
    unbounded_worker["games_per_worker"] = 100_001
    with pytest.raises(ExtremeScoreLeagueError, match="positive integer"):
        build_plan(unbounded_worker)

    injected_focal = request(tmp_path / "injected-focal")
    injected_focal["focal_model"] = model_binding(
        tmp_path / "rejected-model", "rejected", b"rejected"
    )
    with pytest.raises(ExtremeScoreLeagueError, match="request keys"):
        build_plan(injected_focal)


def test_required_opponents_cannot_receive_zero_workers(tmp_path):
    value = request(tmp_path, gpu_indices=[0, 1])
    with pytest.raises(ExtremeScoreLeagueError, match="insufficient worker slots"):
        build_plan(value)


def test_status_observes_outputs_without_creating_them(tmp_path):
    plan = build_plan(request(tmp_path))
    first = status(plan)
    assert not any(item["output_exists"] for item in first["workers"])
    assert {item["state"] for item in first["workers"]} == {"PLANNED"}

    output = Path(plan["workers"][0]["output_directory"])
    output.mkdir(parents=True)
    second = status(plan)
    assert second["workers"][0]["output_exists"] is True
    assert second["workers"][0]["state"] == "RUNNING"
    assert sum(item["output_exists"] for item in second["workers"]) == 1


def test_worker_output_parent_creation_is_concurrency_safe(tmp_path):
    root = tmp_path / "selfplay"
    root.mkdir()
    outputs = [root / "generation" / f"worker-{index}" for index in range(32)]
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(lambda output: _ensure_output_parent(root, output), outputs))
    assert (root / "generation").is_dir()


def test_policy_tampering_is_rejected_before_and_after_planning(tmp_path):
    policy_path = tmp_path / "training-policy.json"
    policy_path.write_bytes(DEFAULT_POLICY_PATH.read_bytes())
    plan = build_plan(request(tmp_path / "run"), policy_path=policy_path)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["league"]["threads_per_worker"] = 49
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ExtremeScoreLeagueError, match="policy"):
        validate_plan(plan)
    with pytest.raises(ExtremeScoreLeagueError, match="policy"):
        build_plan(request(tmp_path / "tampered"), policy_path=policy_path)


def test_launcher_rejects_mutable_binary_and_snapshot_replacement(tmp_path):
    binary_request = request(tmp_path / "binary")
    binary_plan = build_plan(binary_request)
    Path(binary_request["binary"]["path"]).write_bytes(b"replaced")
    called = False

    def must_not_execute(_argv):
        nonlocal called
        called = True
        return 0

    with pytest.raises(ExtremeScoreLeagueError, match="changed|read-only"):
        execute_worker(
            binary_plan,
            binary_plan["workers"][0]["worker_id"],
            executor=must_not_execute,
        )
    assert called is False

    snapshot_plan = build_plan(request(tmp_path / "snapshot"))
    snapshot_directory = Path(snapshot_plan["focal_model"]["directory"])
    snapshot_model = snapshot_directory / "model.bin.gz"
    os.chmod(snapshot_directory, 0o755)
    os.chmod(snapshot_model, 0o644)
    snapshot_model.write_bytes(b"replaced")
    with pytest.raises(ExtremeScoreLeagueError, match="changed|read-only"):
        execute_worker(
            snapshot_plan,
            snapshot_plan["workers"][0]["worker_id"],
            executor=must_not_execute,
        )
    assert called is False


def test_focal_directory_drift_cannot_enter_worker_selection(tmp_path):
    value = request(tmp_path)
    accepted = load_accepted_state(Path(value["accepted_state"]["path"]))
    mutable_focal = Path(accepted["artifact"]["path"]).parent
    plan = build_plan(value)
    unplanned = mutable_focal / "new-checkpoint" / "model.bin.gz"
    unplanned.parent.mkdir()
    unplanned.write_bytes(b"not-in-plan")
    worker = plan["workers"][0]
    observed = {}

    def fake_executor(argv):
        focal_directory = Path(argv[argv.index("-models-dir") + 1])
        observed["focal_directory"] = focal_directory
        assert focal_directory == Path(plan["focal_model"]["directory"])
        assert focal_directory != mutable_focal
        assert list(focal_directory.rglob("model.bin.gz")) == [
            focal_directory / "model.bin.gz"
        ]
        shard = output_directory(argv) / "games" / "0.tdata"
        shard.parent.mkdir()
        shard.write_bytes(b"shard")
        return SimpleNamespace(returncode=0)

    receipt = execute_worker(plan, worker["worker_id"], executor=fake_executor)
    assert observed["focal_directory"] == Path(plan["focal_model"]["directory"])
    assert receipt["process_outcome"]["status"] == "succeeded"


def test_receipt_binds_plan_artifacts_shards_and_process_outcome(tmp_path):
    plan = build_plan(request(tmp_path))
    worker = plan["workers"][0]
    calls = 0

    def fake_executor(argv):
        nonlocal calls
        calls += 1
        shard = output_directory(argv) / "games" / "chunk-000.jsonl"
        shard.parent.mkdir()
        shard.write_bytes(b'{"game":"g0"}\n')
        return SimpleNamespace(returncode=7)

    receipt = execute_worker(plan, worker["worker_id"], executor=fake_executor)
    assert receipt["contract"] == WORKER_RECEIPT_CONTRACT
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert receipt["worker_sha256"]
    assert receipt["process_outcome"] == {
        "status": "failed",
        "returncode": 7,
        "error_type": None,
        "error_message": None,
    }
    assert receipt["artifact_verification"]["artifacts_unchanged"] is True
    assert set(receipt["artifact_bindings"]) == {
        "policy",
        "accepted_state",
        "launcher",
        "binary",
        "config",
        "focal_model",
        "opponents",
    }
    assert len(receipt["artifact_bindings"]["opponents"]) == 3
    assert receipt["output_shards"] == [
        {
            "relative_path": "games/chunk-000.jsonl",
            "sha256": file_sha256(
                Path(worker["output_directory"]) / "games" / "chunk-000.jsonl"
            ),
            "size_bytes": len(b'{"game":"g0"}\n'),
        }
    ]

    receipt_path = Path(worker["output_directory"]) / "worker-execution-receipt.json"
    shard_path = Path(worker["output_directory"]) / "games" / "chunk-000.jsonl"
    assert receipt_path.is_file()
    assert receipt_path.stat().st_mode & 0o222 == 0
    assert shard_path.is_file()
    assert shard_path.stat().st_mode & 0o222 == 0

    resumed = execute_worker(
        plan,
        worker["worker_id"],
        executor=lambda _argv: pytest.fail("completed worker must not execute twice"),
    )
    assert resumed == receipt
    assert calls == 1
    worker_status = next(
        item
        for item in status(plan)["workers"]
        if item["worker_id"] == worker["worker_id"]
    )
    assert worker_status["state"] == "FAILED"
    assert worker_status["process_status"] == "failed"


def test_immutable_publication_requires_matching_read_only_regular_file(tmp_path):
    target = tmp_path / "plan.json"
    value = {"schema_version": 1, "contract": "test"}
    _publish_immutable_json(target, value)
    assert target.is_file()
    assert target.stat().st_mode & 0o222 == 0

    _publish_immutable_json(target, value)
    os.chmod(target, 0o644)
    with pytest.raises(ExtremeScoreLeagueError, match="read-only"):
        _publish_immutable_json(target, value)
