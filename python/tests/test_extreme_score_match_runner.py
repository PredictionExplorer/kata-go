import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from risk_score.extreme_score_evaluator import (
    PLAN_REQUEST_CONTRACT,
    build_plan,
    canonical_json,
    evaluate_with_runner,
    file_sha256,
)
from risk_score.extreme_score_match_runner import (
    EMPTY_BOARD_POSITION,
    ExtremeScoreMatchRunner,
    ExtremeScoreMatchRunnerConflictError,
    ExtremeScoreMatchRunnerSpec,
    ExtremeScoreMatchRunnerValidationError,
    GpuIdentityObservation,
    build_empty_board_schedule,
    build_match_command,
    evaluator_rows_from_match_results,
    group_jobs_by_opponent_and_color,
)

PLAN_HASH = "1" * 64
GPU_UUID = "GPU-test-0007"
GPU_LEASE_PROVENANCE = "test-lease:extreme-score-gpu-7"


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def runner_files(tmp_path, *, focal_name="candidate.bin"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "binary": tmp_path / "katago",
        "candidate": tmp_path / focal_name,
        "reference": tmp_path / "reference.bin",
        "opponent_a": tmp_path / "opponent-a.bin",
        "opponent_b": tmp_path / "opponent-b.bin",
        "config": tmp_path / "extreme-score-match.cfg",
        "output": tmp_path / "runner-output",
    }
    paths["binary"].write_bytes(b"fake-katago")
    paths["candidate"].write_bytes(b"candidate")
    paths["reference"].write_bytes(b"reference")
    paths["opponent_a"].write_bytes(b"opponent-a")
    paths["opponent_b"].write_bytes(b"opponent-b")
    paths["config"].write_text("extremeScoreGroupSize = 1\n", encoding="utf-8")
    paths["hashes"] = {
        name: file_sha256(path)
        for name, path in paths.items()
        if name
        in {
            "binary",
            "candidate",
            "reference",
            "opponent_a",
            "opponent_b",
            "config",
        }
    }
    return paths


def make_spec(
    paths,
    *,
    process_count=2,
    topology="two-gpu-processes",
    gpu_index=7,
    expected_gpu_uuid=GPU_UUID,
):
    hashes = paths["hashes"]
    return ExtremeScoreMatchRunnerSpec(
        katago_binary=paths["binary"],
        focal_models={
            hashes["candidate"]: paths["candidate"],
            hashes["reference"]: paths["reference"],
        },
        opponent_models={
            hashes["opponent_a"]: paths["opponent_a"],
            hashes["opponent_b"]: paths["opponent_b"],
        },
        match_config=paths["config"],
        output_root=paths["output"],
        topology=topology,
        process_count=process_count,
        expected_gpu_uuid=expected_gpu_uuid,
        gpu_lease_provenance=GPU_LEASE_PROVENANCE,
        gpu_index=gpu_index,
    )


def make_jobs(paths, arm, *, cohort_size=2):
    hashes = paths["hashes"]
    model_hash = hashes["candidate"] if arm == "candidate" else hashes["reference"]
    cohorts = (
        ("cohort-a-black", "league-a", "snapshot-a", hashes["opponent_a"], "B"),
        ("cohort-a-white", "league-a", "snapshot-a", hashes["opponent_a"], "W"),
        ("cohort-b-black", "league-b", "snapshot-b", hashes["opponent_b"], "B"),
    )
    jobs = []
    for cohort_id, league, snapshot, opponent_hash, color in cohorts:
        for trial_index in range(cohort_size):
            jobs.append(
                {
                    "schema_version": 1,
                    "plan_sha256": PLAN_HASH,
                    "arm": arm,
                    "model_sha256": model_hash,
                    "cohort_id": cohort_id,
                    "cohort_sha256": digest_text("cohort:" + cohort_id),
                    "trial_index": trial_index,
                    "seed": f"seed:{cohort_id}:{trial_index}",
                    "config_sha256": hashes["config"],
                    "league_cell": league,
                    "opponent_snapshot_id": snapshot,
                    "opponent_model_sha256": opponent_hash,
                    "focal_color": color,
                }
            )
    return tuple(jobs)


def cpp_result(
    schedule_row,
    *,
    white_minus_black=3.5,
    no_result=False,
    hit_turn_limit=False,
):
    black_index = schedule_row["blackBot"]
    white_index = schedule_row["whiteBot"]
    names = {0: "focal", 1: "frozen-opponent"}
    if no_result or hit_turn_limit:
        score = None
        winner = None
        scored = False
        final_result = "turn_limit" if hit_turn_limit else "Void"
    else:
        score = float(white_minus_black)
        winner = "W" if score > 0 else "B" if score < 0 else "draw"
        scored = True
        final_result = f"{winner}+{abs(score)}" if winner in {"B", "W"} else "0"
    return {
        "schemaVersion": 1,
        "scheduleId": schedule_row["scheduleId"],
        "gameId": schedule_row["gameId"],
        "pairId": schedule_row["pairId"],
        "positionId": schedule_row["positionId"],
        "seed": schedule_row["seed"],
        "blackBot": names[black_index],
        "whiteBot": names[white_index],
        "blackBotIndex": black_index,
        "whiteBotIndex": white_index,
        "board": {"xSize": 19, "ySize": 19},
        "rules": {
            "ko": "POSITIONAL",
            "scoring": "AREA",
            "tax": "NONE",
            "suicide": True,
            "hasButton": False,
            "whiteHandicapBonus": "0",
            "friendlyPassOk": False,
            "komi": 7.5,
        },
        "komi": 7.5,
        "finalResult": final_result,
        "finalWhiteMinusBlackScore": score,
        "winner": winner,
        "moveCount": 0,
        "blackMoveCount": 0,
        "whiteMoveCount": 0,
        "startTurnNumber": 0,
        "hitTurnLimit": hit_turn_limit,
        "resignation": False,
        "noResult": no_result,
        "scored": scored,
        "gameHash": digest_text(schedule_row["gameId"]),
    }


class FakeResultProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, cell):
        self.calls.append(cell)
        return [cpp_result(row) for row in cell.schedule_rows]


def parse_overrides(argv):
    index = argv.index("-override-config")
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in argv[index + 1].split(",")
    }


class FakeSubprocess:
    def __init__(self):
        self.calls = []
        self.schedule_modes = []

    def __call__(self, argv, **kwargs):
        self.calls.append((tuple(argv), dict(kwargs)))
        overrides = parse_overrides(argv)
        schedule_path = Path(overrides["deterministicScheduleFile"])
        self.schedule_modes.append(stat.S_IMODE(schedule_path.stat().st_mode))
        result_path = Path(overrides["matchResultJsonlFile"])
        schedule = [
            json.loads(line)
            for line in schedule_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result_path.write_text(
            "".join(canonical_json(cpp_result(row)) + "\n" for row in schedule),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def stable_gpu_probe(gpu_index):
    return GpuIdentityObservation(
        gpu_index=gpu_index,
        gpu_uuid=GPU_UUID,
        provenance="deterministic-test-gpu-inventory",
    )


class SequencedGpuProbe:
    def __init__(self, *observations):
        self.observations = list(observations)
        self.calls = []

    def __call__(self, gpu_index):
        self.calls.append(gpu_index)
        if not self.observations:
            raise AssertionError("unexpected GPU identity probe")
        return self.observations.pop(0)


def test_schedule_maps_exact_job_seed_identity_and_focal_color(tmp_path):
    paths = runner_files(tmp_path)
    candidate_jobs = make_jobs(paths, "candidate")
    reference_jobs = make_jobs(paths, "reference")
    black_jobs = candidate_jobs[:2]
    black_schedule = build_empty_board_schedule(black_jobs)

    assert [row["seed"] for row in black_schedule] == [
        job["seed"] for job in black_jobs
    ]
    assert [row["job"] for row in black_schedule] == list(black_jobs)
    assert all(row["startPosition"] == EMPTY_BOARD_POSITION for row in black_schedule)
    assert all((row["blackBot"], row["whiteBot"]) == (0, 1) for row in black_schedule)

    reference_schedule = build_empty_board_schedule(reference_jobs[:2])
    assert [row["scheduleId"] for row in reference_schedule] == [
        row["scheduleId"] for row in black_schedule
    ]
    assert [row["gameId"] for row in reference_schedule] == [
        row["gameId"] for row in black_schedule
    ]
    assert [row["seed"] for row in reference_schedule] == [
        row["seed"] for row in black_schedule
    ]

    white_schedule = build_empty_board_schedule(candidate_jobs[2:4])
    assert all((row["blackBot"], row["whiteBot"]) == (1, 0) for row in white_schedule)


def test_terminal_score_is_converted_to_focal_perspective(tmp_path):
    paths = runner_files(tmp_path)
    jobs = make_jobs(paths, "candidate")
    black_schedule = build_empty_board_schedule(jobs[:2])
    white_schedule = build_empty_board_schedule(jobs[2:4])

    black_rows = evaluator_rows_from_match_results(
        black_schedule,
        [cpp_result(row, white_minus_black=12.5) for row in black_schedule],
    )
    white_rows = evaluator_rows_from_match_results(
        white_schedule,
        [cpp_result(row, white_minus_black=12.5) for row in white_schedule],
    )
    unresolved = evaluator_rows_from_match_results(
        black_schedule,
        [cpp_result(row, no_result=True) for row in black_schedule],
    )
    turn_limited = evaluator_rows_from_match_results(
        black_schedule,
        [cpp_result(row, hit_turn_limit=True) for row in black_schedule],
    )

    assert [row["score"] for row in black_rows] == [-12.5, -12.5]
    assert [row["score"] for row in white_rows] == [12.5, 12.5]
    assert all(row["score"] == 0.0 and row["no_result"] for row in unresolved)
    assert all(row["score"] == 0.0 and row["hit_turn_limit"] for row in turn_limited)
    assert all(set(job).issubset(row) for job, row in zip(jobs[:2], black_rows))


def test_result_conversion_rejects_focal_color_authority_mismatches(tmp_path):
    paths = runner_files(tmp_path)
    black_schedule = build_empty_board_schedule(make_jobs(paths, "candidate")[:2])
    results = [cpp_result(row, white_minus_black=12.5) for row in black_schedule]

    with pytest.raises(
        ExtremeScoreMatchRunnerValidationError,
        match="caller focal_color contradicts",
    ):
        evaluator_rows_from_match_results(
            black_schedule,
            results,
            focal_color="W",
        )

    contradictory_schedule = json.loads(canonical_json(black_schedule))
    contradictory_schedule[0]["job"]["focal_color"] = "W"
    with pytest.raises(
        ExtremeScoreMatchRunnerValidationError,
        match="bot assignment contradicts",
    ):
        evaluator_rows_from_match_results(contradictory_schedule, results)


def test_runner_groups_cells_and_pairs_candidate_reference_plans(tmp_path):
    paths = runner_files(tmp_path)
    provider = FakeResultProvider()
    runner = ExtremeScoreMatchRunner(make_spec(paths), fake_result_provider=provider)
    candidate_jobs = make_jobs(paths, "candidate")
    reference_jobs = make_jobs(paths, "reference")

    grouped = group_jobs_by_opponent_and_color(candidate_jobs)
    assert [(key[1], len(value)) for key, value in grouped.items()] == [
        ("B", 2),
        ("W", 2),
        ("B", 2),
    ]

    candidate_rows = runner("candidate", candidate_jobs)
    reference_rows = runner("reference", reference_jobs)

    assert len(provider.calls) == 6
    assert {
        (cell.arm, cell.opponent_model_sha256, cell.focal_color)
        for cell in provider.calls
    } == {
        (
            arm,
            paths["hashes"][opponent],
            color,
        )
        for arm in ("candidate", "reference")
        for opponent, color in (
            ("opponent_a", "B"),
            ("opponent_a", "W"),
            ("opponent_b", "B"),
        )
    }
    assert [row["seed"] for row in candidate_rows] == [
        row["seed"] for row in reference_rows
    ]
    assert set(runner.source_bindings) == {"candidate", "reference"}
    provenance = runner.execution_provenance
    assert provenance["runner_spec_sha256"] == runner.runner_spec_sha256
    assert set(provenance["arm_receipts"]) == {"candidate", "reference"}
    assert all(
        binding["cell_receipts"] for binding in provenance["arm_receipts"].values()
    )


def test_runner_is_directly_compatible_with_expected_max_evaluator(tmp_path):
    paths = runner_files(tmp_path)
    hashes = paths["hashes"]
    cohorts = []
    for index, color in enumerate(("B", "W")):
        cohorts.append(
            {
                "cohort_id": f"evaluator-cohort-{color}",
                "cluster_id": "evaluator-cluster",
                "league_cell": "league-a",
                "opponent_snapshot_id": "snapshot-a",
                "opponent_model_sha256": hashes["opponent_a"],
                "focal_color": color,
                "seeds": [f"evaluator-seed-{index}-{trial}" for trial in range(2)],
            }
        )
    plan = build_plan(
        {
            "schema_version": 1,
            "contract": PLAN_REQUEST_CONTRACT,
            "candidate_model": {
                "model_id": "candidate",
                "sha256": hashes["candidate"],
            },
            "reference_model": {
                "model_id": "reference",
                "sha256": hashes["reference"],
            },
            "config": {
                "config_id": "expected-max-match",
                "sha256": hashes["config"],
            },
            "cohort_size": 2,
            "legal_score_bounds": {"minimum": -400.0, "maximum": 400.0},
            "cohorts": cohorts,
            "rollback_recommendation": {
                "action": "retain_reference",
                "reference_model": {
                    "model_id": "reference",
                    "sha256": hashes["reference"],
                },
                "reference_model_artifact": {
                    "path": str(paths["reference"].resolve()),
                    "file_sha256": hashes["reference"],
                },
                "trainer_checkpoint_artifact": {
                    "path": str(paths["candidate"].resolve()),
                    "file_sha256": hashes["candidate"],
                },
                "quarantine_candidate_on_failure": True,
            },
        }
    )
    runner = ExtremeScoreMatchRunner(
        make_spec(paths), fake_result_provider=FakeResultProvider()
    )

    report = evaluate_with_runner(plan, runner=runner)

    assert report["integrity"]["valid"] is True
    assert report["result_bindings"]["candidate"]["source"] is not None
    assert report["result_bindings"]["reference"]["source"] is not None
    assert report["integrity"]["expected_rows_per_arm"] == 4


def test_runner_rejects_identity_hash_and_pairing_mismatches(tmp_path):
    paths = runner_files(tmp_path)
    provider = FakeResultProvider()
    runner = ExtremeScoreMatchRunner(make_spec(paths), fake_result_provider=provider)
    bad_config_jobs = [dict(job) for job in make_jobs(paths, "candidate")]
    for job in bad_config_jobs:
        job["config_sha256"] = "f" * 64
    with pytest.raises(
        ExtremeScoreMatchRunnerValidationError, match="config|match file"
    ):
        runner("candidate", tuple(bad_config_jobs))

    candidate_jobs = make_jobs(paths, "candidate")
    runner("candidate", candidate_jobs)
    changed_reference_jobs = [dict(job) for job in make_jobs(paths, "reference")]
    changed_reference_jobs[0]["seed"] = "different-paired-seed"
    with pytest.raises(
        ExtremeScoreMatchRunnerConflictError, match="identical plan and seeds"
    ):
        runner("reference", tuple(changed_reference_jobs))

    other_paths = runner_files(tmp_path / "mutated")
    other_runner = ExtremeScoreMatchRunner(
        make_spec(other_paths), fake_result_provider=FakeResultProvider()
    )
    other_paths["candidate"].write_bytes(b"changed-after-binding")
    with pytest.raises(ExtremeScoreMatchRunnerValidationError, match="changed after"):
        other_runner("candidate", make_jobs(other_paths, "candidate"))


def test_exact_cells_and_arm_results_resume_but_tampering_conflicts(tmp_path):
    paths = runner_files(tmp_path)
    provider = FakeResultProvider()
    spec = make_spec(paths)
    first_runner = ExtremeScoreMatchRunner(spec, fake_result_provider=provider)
    jobs = make_jobs(paths, "candidate")

    first_rows = first_runner("candidate", jobs)
    call_count = len(provider.calls)
    assert first_runner("candidate", jobs) == first_rows
    assert len(provider.calls) == call_count

    resumed_runner = ExtremeScoreMatchRunner(spec, fake_result_provider=provider)
    assert resumed_runner("candidate", jobs) == first_rows
    assert len(provider.calls) == call_count
    arm_result = (
        paths["output"] / "plans" / PLAN_HASH / "arms" / "candidate" / "results.jsonl"
    )
    arm_result.chmod(0o644)
    arm_result.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ExtremeScoreMatchRunnerConflictError,
        match="read-only|hash changed|contradicts",
    ):
        ExtremeScoreMatchRunner(spec, fake_result_provider=provider)("candidate", jobs)


def test_runner_spec_binds_topology_process_count_and_rejects_conflict(tmp_path):
    paths = runner_files(tmp_path)
    provider = FakeResultProvider()
    jobs = make_jobs(paths, "candidate")
    first = ExtremeScoreMatchRunner(
        make_spec(paths, process_count=2, topology="topology-a"),
        fake_result_provider=provider,
    )
    second = ExtremeScoreMatchRunner(
        make_spec(paths, process_count=1, topology="topology-b"),
        fake_result_provider=provider,
    )
    assert first.runner_spec_sha256 != second.runner_spec_sha256
    first("candidate", jobs)
    with pytest.raises(ExtremeScoreMatchRunnerConflictError, match="runner.json"):
        second("candidate", jobs)


def test_match_command_is_shell_free_and_expected_max_overrides_are_safe(tmp_path):
    paths = runner_files(tmp_path, focal_name="candidate;touch SHOULD_NOT_EXIST")
    fake_subprocess = FakeSubprocess()
    runner = ExtremeScoreMatchRunner(
        make_spec(paths, process_count=1),
        subprocess_runner=fake_subprocess,
        gpu_identity_probe=stable_gpu_probe,
    )
    rows = runner("candidate", make_jobs(paths, "candidate"))

    assert len(rows) == 6
    assert len(fake_subprocess.calls) == 3
    argv, kwargs = fake_subprocess.calls[0]
    overrides = parse_overrides(argv)
    assert argv[1:4] == ("match", "-config", str(paths["config"].resolve()))
    assert ";touch SHOULD_NOT_EXIST" in overrides["nnModelFile0"]
    assert overrides["useExpectedMaxScoreUtility"] == "true"
    assert overrides["extremeScoreGroupSize"] == "2"
    assert overrides["expectedMaxFocalColor"] in {"B", "W"}
    assert overrides["numNNServerThreadsPerModel"] == "1"
    assert kwargs["shell"] is False
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert fake_subprocess.schedule_modes == [0o444, 0o444, 0o444]
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()

    receipt_paths = sorted(
        (paths["output"] / "plans" / PLAN_HASH / "cells" / "candidate").glob(
            "*/receipt.json"
        )
    )
    assert len(receipt_paths) == 3
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        execution = receipt["execution"]
        assert execution["expected_gpu_uuid"] == GPU_UUID
        assert execution["gpu_lease_provenance"] == GPU_LEASE_PROVENANCE
        assert execution["observed_before"]["gpu_uuid"] == GPU_UUID
        assert execution["observed_after"]["gpu_uuid"] == GPU_UUID

    with pytest.raises(ValueError, match="comma"):
        build_match_command(
            paths["binary"],
            paths["config"],
            Path("/tmp/focal,unsafe"),
            paths["opponent_a"],
            tmp_path / "schedule.jsonl",
            tmp_path / "result.jsonl",
            focal_color="B",
            group_size=2,
            game_count=2,
        )


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            GpuIdentityObservation(
                gpu_index=6,
                gpu_uuid=GPU_UUID,
                provenance="remapped-test-inventory",
            ),
            "ordinal remapped",
        ),
        (
            GpuIdentityObservation(
                gpu_index=7,
                gpu_uuid="GPU-other-0008",
                provenance="wrong-uuid-test-inventory",
            ),
            "wrong physical UUID",
        ),
    ],
)
def test_runner_rejects_gpu_ordinal_or_uuid_mismatch(tmp_path, observation, message):
    paths = runner_files(tmp_path)
    fake_subprocess = FakeSubprocess()
    runner = ExtremeScoreMatchRunner(
        make_spec(paths, process_count=1),
        subprocess_runner=fake_subprocess,
        gpu_identity_probe=lambda _gpu_index: observation,
    )

    with pytest.raises(ExtremeScoreMatchRunnerValidationError, match=message):
        runner("candidate", make_jobs(paths, "candidate")[:2])

    assert fake_subprocess.calls == []


def test_runner_rejects_gpu_identity_change_after_subprocess(tmp_path):
    paths = runner_files(tmp_path)
    fake_subprocess = FakeSubprocess()
    probe = SequencedGpuProbe(
        GpuIdentityObservation(
            gpu_index=7,
            gpu_uuid=GPU_UUID,
            provenance="pre-execution-test-inventory",
        ),
        GpuIdentityObservation(
            gpu_index=7,
            gpu_uuid="GPU-replaced-0007",
            provenance="post-execution-test-inventory",
        ),
    )
    runner = ExtremeScoreMatchRunner(
        make_spec(paths, process_count=1),
        subprocess_runner=fake_subprocess,
        gpu_identity_probe=probe,
    )

    with pytest.raises(
        ExtremeScoreMatchRunnerValidationError,
        match="identity changed between pre- and post-execution",
    ):
        runner("candidate", make_jobs(paths, "candidate")[:2])

    assert probe.calls == [7, 7]
    assert len(fake_subprocess.calls) == 1


def test_immutable_runner_artifacts_publish_read_only_and_validate_mode(tmp_path):
    paths = runner_files(tmp_path)
    provider = FakeResultProvider()
    spec = make_spec(paths)
    jobs = make_jobs(paths, "candidate")
    ExtremeScoreMatchRunner(spec, fake_result_provider=provider)("candidate", jobs)

    artifacts = [path for path in paths["output"].rglob("*") if path.is_file()]
    assert artifacts
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in artifacts)

    schedule_path = next(paths["output"].glob("plans/*/cells/*/*/schedule.jsonl"))
    schedule_path.chmod(0o644)
    with pytest.raises(
        ExtremeScoreMatchRunnerConflictError,
        match="immutable read-only mode",
    ):
        ExtremeScoreMatchRunner(spec, fake_result_provider=provider)("candidate", jobs)


def test_checked_in_config_freezes_expected_max_stochastic_match_policy():
    config_path = (
        Path(__file__).parents[2]
        / "cpp"
        / "configs"
        / "risk_score"
        / "extreme_score_match_19x19.cfg"
    )
    text = config_path.read_text(encoding="utf-8")
    required = (
        "useExpectedMaxScoreUtility = true",
        "extremeScoreGroupSize = 1",
        "expectedMaxFocalColor = B",
        "allowResignation = false",
        "handicapProb = 0",
        "bSizes = 19",
        "komiMean = 7.5",
        "rootNoiseEnabled = true",
        "chosenMoveTemperatureEarly = 0.75",
        "numNNServerThreadsPerModel = 1",
    )
    assert all(setting in text for setting in required)
