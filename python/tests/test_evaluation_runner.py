import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.build_evaluation_suites import (
    DEFAULT_POLICY_PATH,
    ORDINARY_BANKS,
    SPECIALIZED_LABELS,
    build_evaluation_suites,
    semantic_position_sha256,
)
from risk_score.build_evaluation_suites import (
    canonical_json as suite_canonical_json,
)
from risk_score.build_evaluation_suites import (
    canonical_sha256 as suite_canonical_sha256,
)
from risk_score.evaluation_runner import (
    EvaluationConflictError,
    EvaluationError,
    EvaluationPlan,
    EvaluationResult,
    EvaluationRunner,
    EvaluationSpec,
    EvaluationValidationError,
    build_evaluation_matrix,
    build_match_command,
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_schedule,
    resolve_manifest_cell,
    shard_schedule,
    validate_move_jsonl,
    validate_result_jsonl,
)
from risk_score.generate_schedule import build_schedule, write_schedule

LEGACY_POLICY_PATH = (
    Path(__file__).parents[1] / "risk_score" / "promotion_policy_v1.json"
)


def position(index, label="ordinary"):
    return {
        "xSize": 19,
        "ySize": 19,
        "board": "/".join(["." * 19] * 19),
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": index,
        "hintLoc": "null",
        "metadata": label,
    }


def write_labeled_positions(path, rows):
    path.write_text(
        "".join(suite_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_small_v2_policy(path):
    policy = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    policy["policy_version"] = "risk-seeking-checkpoint-promotion-v2-test"
    stages = policy["evaluation_stages"]
    stages["stage_1_cheap_paired_screen"]["ordinary_color_pairs"] = 1
    stages["stage_2_finalist_selection"].update(
        {
            "ordinary_color_pairs": 2,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    stages["deep_audit"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    quotas = (
        {
            "powered_candidate_vs_champion": 2,
            "powered_candidate_vs_original": 2,
            "standard_candidate_vs_original": 1,
            "lead_40": 2,
            "lead_80": 2,
        },
        {
            "powered_candidate_vs_champion": 3,
            "powered_candidate_vs_original": 3,
            "standard_candidate_vs_original": 1,
            "lead_40": 3,
            "lead_80": 4,
        },
    )
    for look, values in zip(stages["stage_3_promotion_confirmation"]["looks"], quotas):
        look.update(
            {
                "powered_ordinary_color_pairs_per_matchup": values[
                    "powered_candidate_vs_champion"
                ],
                "standard_ordinary_color_pairs": values[
                    "standard_candidate_vs_original"
                ],
                "lead_40_color_pairs": values["lead_40"],
                "lead_80_color_pairs": values["lead_80"],
                "minimum_independent_position_clusters": dict(values),
            }
        )
    path.write_text(suite_canonical_json(policy) + "\n", encoding="utf-8")
    return policy


def small_v2_suite_sources():
    return (
        [position(index, "ordinary") for index in range(6)]
        + [position(100 + index, "lead-40") for index in range(5)]
        + [position(200 + index, "lead-80") for index in range(6)]
        + [position(300, "tactical"), position(301, "exploitability")]
    )


def result_for(row):
    bot_names = {0: "candidate", 1: "reference"}
    return {
        "schemaVersion": 1,
        "scheduleId": row["scheduleId"],
        "gameId": row["gameId"],
        "pairId": row["pairId"],
        "positionId": row["positionId"],
        "seed": row["seed"],
        "blackBot": bot_names[row["blackBot"]],
        "whiteBot": bot_names[row["whiteBot"]],
        "blackBotIndex": row["blackBot"],
        "whiteBotIndex": row["whiteBot"],
        "board": {
            "xSize": row["startPosition"]["xSize"],
            "ySize": row["startPosition"]["ySize"],
        },
        "rules": {"ko": "POSITIONAL", "scoring": "AREA"},
        "komi": 7.5,
        "finalResult": "B+1",
        "finalWhiteMinusBlackScore": -1.0,
        "winner": "B",
        "moveCount": 2,
        "blackMoveCount": 1,
        "whiteMoveCount": 1,
        "startTurnNumber": row["startPosition"]["initialTurnNumber"]
        + len(row["startPosition"]["moveLocs"]),
        "hitTurnLimit": False,
        "resignation": False,
        "noResult": False,
        "scored": True,
        "gameHash": "hash-" + row["gameId"],
    }


def moves_for(row, result):
    rows = []
    first_player = row["startPosition"]["nextPla"]
    for offset in range(result["moveCount"]):
        player = (
            first_player if offset % 2 == 0 else "W" if first_player == "B" else "B"
        )
        rows.append(
            {
                "schemaVersion": 1,
                "scheduleId": row["scheduleId"],
                "gameId": row["gameId"],
                "pairId": row["pairId"],
                "positionId": row["positionId"],
                "seed": row["seed"],
                "turnNumber": result["startTurnNumber"] + offset,
                "player": player,
                "bot": result["blackBot"] if player == "B" else result["whiteBot"],
                "move": "D4" if offset == 0 else "Q16",
                "scoreLead": 0.0,
                "winProbability": 0.5,
            }
        )
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_overrides(argv):
    index = argv.index("-override-config")
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in argv[index + 1].split(",")
    }


class FakeMatchRunner:
    def __init__(self, behaviors=(), delay=0.0):
        self.behaviors = list(behaviors)
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        with self.lock:
            call_index = len(self.calls)
            self.calls.append((tuple(argv), dict(kwargs)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            behavior = (
                self.behaviors[call_index]
                if call_index < len(self.behaviors)
                else "valid"
            )
            overrides = parse_overrides(argv)
            schedule_path = Path(overrides["deterministicScheduleFile"])
            result_path = Path(overrides["matchResultJsonlFile"])
            move_value = overrides.get("matchMoveJsonlFile", "")
            move_path = Path(move_value) if move_value else None
            rows = [
                json.loads(line)
                for line in schedule_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            results = [result_for(row) for row in rows]

            if behavior == "process-failure":
                return SimpleNamespace(returncode=17, stdout="", stderr="injected")
            if behavior == "malformed":
                result_path.write_text('{"schemaVersion":', encoding="utf-8")
            elif behavior == "truncated":
                write_jsonl(result_path, results[:-1])
            elif behavior == "duplicate":
                write_jsonl(result_path, results + [results[0]])
            elif behavior == "wrong-id":
                results[0] = dict(results[0], gameId="not-scheduled")
                write_jsonl(result_path, results)
            elif behavior == "wrong-seed":
                results[0] = dict(results[0], seed="wrong-seed")
                write_jsonl(result_path, results)
            elif behavior == "wrong-bot-index":
                results[0] = dict(results[0], blackBotIndex=9)
                write_jsonl(result_path, results)
            elif behavior == "same-bot-name":
                results[0] = dict(results[0], whiteBot=results[0]["blackBot"])
                write_jsonl(result_path, results)
            elif behavior == "missing-field":
                results[0].pop("rules")
                write_jsonl(result_path, results)
            elif behavior == "resignation":
                results[0] = dict(
                    results[0],
                    resignation=True,
                    scored=False,
                    finalResult="B+R",
                    finalWhiteMinusBlackScore=None,
                )
                write_jsonl(result_path, results)
            elif behavior == "turn-limit":
                results[0] = dict(
                    results[0],
                    hitTurnLimit=True,
                    scored=False,
                    winner=None,
                    finalResult="turn_limit",
                    finalWhiteMinusBlackScore=None,
                )
                write_jsonl(result_path, results)
            elif behavior == "bad-scored-flag":
                results[0] = dict(results[0], scored=False)
                write_jsonl(result_path, results)
            elif behavior == "no-result-conflict":
                results[0] = dict(results[0], noResult=True)
                write_jsonl(result_path, results)
            elif behavior == "winner-score-conflict":
                results[0] = dict(results[0], winner="W")
                write_jsonl(result_path, results)
            elif behavior == "final-margin-conflict":
                results[0] = dict(results[0], finalResult="B+2")
                write_jsonl(result_path, results)
            else:
                write_jsonl(result_path, results)

            if move_path is not None:
                moves = [
                    move
                    for index, row in enumerate(rows)
                    for move in moves_for(row, results[index])
                ]
                if behavior == "duplicate-move":
                    moves.append(dict(moves[0]))
                elif behavior == "truncated-move":
                    moves.pop()
                elif behavior == "wrong-player":
                    moves[0] = dict(moves[0], player="W")
                elif behavior == "wrong-move-bot":
                    moves[0] = dict(moves[0], bot="not-the-color-bot")
                elif behavior == "missing-diagnostics":
                    moves[0].pop("scoreLead")
                write_jsonl(move_path, moves)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        finally:
            with self.lock:
                self.active -= 1


def evaluation_files(tmp_path, pair_count=4, candidate_name="candidate.bin"):
    candidate = tmp_path / candidate_name
    reference = tmp_path / "reference.bin"
    original = tmp_path / "original.bin"
    config = tmp_path / "match.cfg"
    policy = tmp_path / "policy.json"
    binary = tmp_path / "katago"
    schedule_path = tmp_path / "schedule.jsonl"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    original.write_bytes(b"original")
    config.write_text("numBots=2\n", encoding="utf-8")
    policy.write_text('{"version":1}\n', encoding="utf-8")
    binary.write_bytes(b"fake-katago-binary")
    schedule = build_schedule(
        [position(index) for index in range(pair_count)],
        base_seed="runner-tests",
    )
    write_schedule(schedule, str(schedule_path))
    spec = EvaluationSpec(
        candidate_model_sha=file_sha256(candidate),
        reference_model_sha=file_sha256(reference),
        original_model_sha=file_sha256(original),
        config_sha=file_sha256(config),
        schedule_sha=file_sha256(schedule_path),
        policy_sha=canonical_sha256(json.loads(policy.read_text(encoding="utf-8"))),
        comparison="candidate-vs-champion-powered",
        suite="discovery",
        stage="stage-1",
        look="look-1",
        topology="2-processes",
        max_visits=400,
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "original": original,
        "config": config,
        "policy": policy,
        "binary": binary,
        "schedule": schedule_path,
        "rows": tuple(schedule),
        "spec": spec,
        "output": tmp_path / "evaluations",
    }


def make_runner(files, fake, **kwargs):
    options = {
        "katago_binary": files["binary"],
        "config_path": files["config"],
        "output_root": files["output"],
        "shard_count": 2,
        "max_parallel": 2,
        "max_attempts": 2,
        "subprocess_runner": fake,
        "env": {"CUDA_VISIBLE_DEVICES": "7"},
    }
    options.update(kwargs)
    return EvaluationRunner(**options)


def run_runner(runner, files, **kwargs):
    return runner.run(
        files["spec"],
        files["schedule"],
        files["candidate"],
        files["reference"],
        original_model_path=files["original"],
        policy_path=files["policy"],
        suite_manifest_path=files.get("suite_manifest"),
        **kwargs,
    )


def test_pair_safe_sharding_is_deterministic_and_balanced():
    rows = build_schedule([position(index) for index in range(7)], base_seed="split")
    first = shard_schedule(rows, 3)
    second = shard_schedule(rows, 3)

    assert first == second
    assert (
        max(len(shard.rows) for shard in first)
        - min(len(shard.rows) for shard in first)
        <= 2
    )
    pair_to_shard = {}
    for shard in first:
        for pair_id in shard.pair_ids:
            assert pair_id not in pair_to_shard
            pair_to_shard[pair_id] = shard.index
        assert all(pair_to_shard[row["pairId"]] == shard.index for row in shard.rows)
    assert set(pair_to_shard) == {row["pairId"] for row in rows}


def test_evaluation_key_is_deterministic_and_covers_coordinates(tmp_path):
    files = evaluation_files(tmp_path)
    spec = files["spec"]
    same = EvaluationSpec(**spec.to_dict())
    changed = dict(spec.to_dict(), look="look-2")
    assert spec.evaluation_key == same.evaluation_key
    assert spec.evaluation_key.startswith("eval-")
    assert len(spec.evaluation_key) == len("eval-") + 64
    assert spec.evaluation_key != EvaluationSpec(**changed).evaluation_key
    assert spec.evaluation_key != EvaluationSpec(
        **dict(spec.to_dict(), max_visits=800)
    ).evaluation_key


def test_runner_execution_key_binds_every_execution_coordinate(tmp_path):
    files = evaluation_files(tmp_path, pair_count=4)
    cwd_a = tmp_path / "cwd-a"
    cwd_b = tmp_path / "cwd-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    other_binary = tmp_path / "katago-other"
    other_binary.write_bytes(b"different-katago-binary")

    def planned(**overrides):
        fake = FakeMatchRunner()
        options = {"shard_count": 2, "max_parallel": 2, "cwd": cwd_a}
        options.update(overrides)
        runner = make_runner(files, fake, **options)
        return runner.plan(
            files["spec"],
            files["schedule"],
            files["candidate"],
            files["reference"],
            original_model_path=files["original"],
            policy_path=files["policy"],
        )

    base = planned()
    variants = [
        planned(katago_binary=other_binary),
        planned(include_move_traces=True),
        planned(extra_args=("--log-file", "runner.log")),
        planned(shard_count=1),
        planned(max_parallel=1),
        planned(cwd=cwd_b),
        planned(timeout=10.0),
        planned(
            replace_env=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "7"},
        ),
        planned(env={"CUDA_VISIBLE_DEVICES": "6"}),
        planned(max_attempts=3),
    ]
    assert base.execution.katago_binary_sha == file_sha256(files["binary"])
    assert base.evaluation_key == planned().evaluation_key
    assert variants[7].execution.environment_sha == base.execution.environment_sha
    assert variants[7].execution.replace_env is True
    assert len({base.evaluation_key, *(plan.evaluation_key for plan in variants)}) == (
        len(variants) + 1
    )
    assert all(plan.partial_dir != base.partial_dir for plan in variants)


def test_runner_requires_original_policy_and_hash_verification(tmp_path):
    files = evaluation_files(tmp_path)
    runner = make_runner(files, FakeMatchRunner())
    with pytest.raises(
        EvaluationValidationError, match="original_model_path is required"
    ):
        runner.plan(
            files["spec"],
            files["schedule"],
            files["candidate"],
            files["reference"],
            policy_path=files["policy"],
        )
    with pytest.raises(EvaluationValidationError, match="policy_path is required"):
        runner.plan(
            files["spec"],
            files["schedule"],
            files["candidate"],
            files["reference"],
            original_model_path=files["original"],
        )
    with pytest.raises(EvaluationValidationError, match="unsafe"):
        runner.plan(
            files["spec"],
            files["schedule"],
            files["candidate"],
            files["reference"],
            original_model_path=files["original"],
            policy_path=files["policy"],
            verify_hashes=False,
        )


def test_command_is_argv_only_and_shell_metacharacters_are_not_executed(tmp_path):
    files = evaluation_files(
        tmp_path, pair_count=1, candidate_name="candidate;touch SHOULD_NOT_EXIST"
    )
    fake = FakeMatchRunner()
    runner = make_runner(files, fake, shard_count=1, max_parallel=1)

    dry_plan = run_runner(runner, files, dry_run=True)
    assert isinstance(dry_plan, EvaluationPlan)
    assert not files["output"].exists()
    assert ";touch SHOULD_NOT_EXIST" in dry_plan.commands[0].argv[5]
    assert "maxVisits=400" in dry_plan.commands[0].argv[5]

    outcome = run_runner(runner, files)
    assert isinstance(outcome, EvaluationResult)
    assert fake.calls[0][1]["shell"] is False
    assert fake.calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    finalized = validate_result_jsonl(outcome.result_path, files["rows"])
    assert all(row["metadata"] == "ordinary" for row in finalized)

    with pytest.raises(ValueError, match="comma"):
        build_match_command(
            Path("/fake/katago"),
            files["config"],
            Path("/tmp/model,one"),
            files["reference"],
            files["schedule"],
            tmp_path / "results.jsonl",
            game_count=2,
            max_visits=400,
        )


def test_failed_subprocess_is_retried(tmp_path):
    files = evaluation_files(tmp_path, pair_count=2)
    fake = FakeMatchRunner(["process-failure", "valid"])
    runner = make_runner(files, fake, shard_count=1, max_parallel=1, max_attempts=2)
    outcome = run_runner(runner, files)
    assert isinstance(outcome, EvaluationResult)
    assert len(fake.calls) == 2
    assert outcome.shards[0].attempts == 2


def test_result_validation_accepts_consistent_true_no_result(tmp_path):
    files = evaluation_files(tmp_path, pair_count=1)
    results = [result_for(row) for row in files["rows"]]
    results[0] = dict(
        results[0],
        noResult=True,
        scored=False,
        winner=None,
        finalResult="Void",
        finalWhiteMinusBlackScore=None,
    )
    path = tmp_path / "no-result.jsonl"
    write_jsonl(path, results)
    validated = validate_result_jsonl(path, files["rows"])
    assert validated[0]["noResult"] is True
    assert validated[0]["metadata"] == "ordinary"


@pytest.mark.parametrize(
    "bad_output",
    [
        "malformed",
        "truncated",
        "duplicate",
        "wrong-id",
        "wrong-seed",
        "wrong-bot-index",
        "same-bot-name",
        "missing-field",
        "resignation",
        "turn-limit",
        "bad-scored-flag",
        "no-result-conflict",
        "winner-score-conflict",
        "final-margin-conflict",
    ],
)
def test_malformed_result_output_is_retried(tmp_path, bad_output):
    files = evaluation_files(tmp_path, pair_count=2)
    fake = FakeMatchRunner([bad_output, "valid"])
    runner = make_runner(files, fake, shard_count=1, max_parallel=1, max_attempts=2)
    outcome = run_runner(runner, files)
    assert isinstance(outcome, EvaluationResult)
    assert len(fake.calls) == 2
    assert len(validate_result_jsonl(outcome.result_path, files["rows"])) == 4


@pytest.mark.parametrize(
    "bad_trace",
    [
        "duplicate-move",
        "truncated-move",
        "wrong-player",
        "wrong-move-bot",
        "missing-diagnostics",
    ],
)
def test_invalid_move_trace_is_retried(tmp_path, bad_trace):
    files = evaluation_files(tmp_path, pair_count=2)
    fake = FakeMatchRunner([bad_trace, "valid"])
    runner = make_runner(
        files,
        fake,
        shard_count=1,
        max_parallel=1,
        max_attempts=2,
        include_move_traces=True,
    )
    outcome = run_runner(runner, files)
    assert isinstance(outcome, EvaluationResult)
    assert len(fake.calls) == 2
    assert outcome.move_path is not None
    results = validate_result_jsonl(outcome.result_path, files["rows"])
    assert len(validate_move_jsonl(outcome.move_path, files["rows"], results)) == 8


def test_evaluator_death_mid_pair_never_publishes_final_output(tmp_path):
    files = evaluation_files(tmp_path, pair_count=2)
    fake = FakeMatchRunner(["truncated", "truncated"])
    runner = make_runner(files, fake, shard_count=1, max_parallel=1, max_attempts=2)
    plan = runner.plan(
        files["spec"],
        files["schedule"],
        files["candidate"],
        files["reference"],
        original_model_path=files["original"],
        policy_path=files["policy"],
    )
    with pytest.raises(EvaluationError, match="did not finalize"):
        run_runner(runner, files)
    assert not plan.final_dir.exists()
    assert plan.partial_dir.is_dir()


def test_complete_partial_shard_is_reconciled_without_rerun(tmp_path):
    files = evaluation_files(tmp_path, pair_count=4)
    fake = FakeMatchRunner()
    runner = make_runner(files, fake, shard_count=2, max_parallel=2)
    plan = runner.plan(
        files["spec"],
        files["schedule"],
        files["candidate"],
        files["reference"],
        original_model_path=files["original"],
        policy_path=files["policy"],
    )
    plan.partial_dir.mkdir(parents=True)
    first_shard = plan.shards[0]
    write_jsonl(
        plan.partial_dir / "shard-000.results.jsonl",
        [result_for(row) for row in first_shard.rows],
    )

    outcome = run_runner(runner, files)
    assert isinstance(outcome, EvaluationResult)
    assert len(fake.calls) == 1
    by_index = {result.shard_index: result for result in outcome.shards}
    assert by_index[0].reused is True
    assert by_index[1].reused is False


def test_final_output_is_atomic_and_idempotently_reconciled(tmp_path):
    files = evaluation_files(tmp_path, pair_count=4)
    fake = FakeMatchRunner()
    runner = make_runner(files, fake, shard_count=2, max_parallel=2)
    first = run_runner(runner, files)
    calls_after_first = len(fake.calls)
    manifest_bytes = first.manifest_path.read_bytes()
    result_bytes = first.result_path.read_bytes()

    second = run_runner(runner, files)
    assert isinstance(second, EvaluationResult)
    assert second.reused is True
    assert len(fake.calls) == calls_after_first
    assert second.manifest_path.read_bytes() == manifest_bytes
    assert second.result_path.read_bytes() == result_bytes
    assert not list((files["output"] / "final").glob(".*.partial-*"))

    first.result_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluationConflictError, match="contradicts manifest"):
        run_runner(runner, files)


def test_parallelism_never_exceeds_configured_bound(tmp_path):
    files = evaluation_files(tmp_path, pair_count=8)
    fake = FakeMatchRunner(delay=0.02)
    runner = make_runner(files, fake, shard_count=4, max_parallel=2, max_attempts=1)
    run_runner(runner, files)
    assert fake.max_active == 2


def test_matrix_helper_builds_powered_and_standard_controls():
    hashes = {
        name: (character * 64)
        for name, character in (
            ("candidate", "a"),
            ("champion", "b"),
            ("original", "c"),
            ("powered_config", "d"),
            ("standard_config", "e"),
            ("powered_schedule", "f"),
            ("standard_schedule", "1"),
            ("policy", "2"),
            ("lead_40_schedule", "3"),
            ("lead_80_schedule", "4"),
        )
    }
    matrix = build_evaluation_matrix(
        candidate_model_sha=hashes["candidate"],
        champion_model_sha=hashes["champion"],
        original_model_sha=hashes["original"],
        powered_config_sha=hashes["powered_config"],
        standard_config_sha=hashes["standard_config"],
        powered_schedule_sha=hashes["powered_schedule"],
        standard_schedule_sha=hashes["standard_schedule"],
        policy_sha=hashes["policy"],
        suite="confirmation",
        stage="stage-3",
        look="look-1",
        topology="8-processes",
        powered_visits=2000,
        standard_visits=800,
    )
    assert [spec.comparison for spec in matrix] == [
        "candidate-vs-champion-powered",
        "candidate-vs-original-powered",
        "candidate-vs-original-standard",
    ]
    assert matrix[0].reference_model_sha == hashes["champion"]
    assert matrix[1].reference_model_sha == hashes["original"]
    assert matrix[2].config_sha == hashes["standard_config"]
    assert len({spec.evaluation_key for spec in matrix}) == 3

    confirmation = build_evaluation_matrix(
        candidate_model_sha=hashes["candidate"],
        champion_model_sha=hashes["champion"],
        original_model_sha=hashes["original"],
        powered_config_sha=hashes["powered_config"],
        standard_config_sha=hashes["standard_config"],
        powered_schedule_sha=hashes["powered_schedule"],
        standard_schedule_sha=hashes["standard_schedule"],
        lead_40_schedule_sha=hashes["lead_40_schedule"],
        lead_80_schedule_sha=hashes["lead_80_schedule"],
        policy_sha=hashes["policy"],
        suite="confirmation",
        stage="stage-3",
        look="look-1",
        topology="8-processes",
        powered_visits=2000,
        standard_visits=800,
    )
    assert len(confirmation) == 5
    assert [(spec.comparison, spec.suite) for spec in confirmation[-2:]] == [
        ("candidate-vs-champion-powered-lead-40", "lead-40"),
        ("candidate-vs-champion-powered-lead-80", "lead-80"),
    ]
    assert all(
        spec.reference_model_sha == hashes["champion"] for spec in confirmation[-2:]
    )

    full_matrix = build_evaluation_matrix(
        candidate_model_sha=hashes["candidate"],
        champion_model_sha=hashes["champion"],
        original_model_sha=hashes["original"],
        powered_config_sha=hashes["powered_config"],
        standard_config_sha=hashes["standard_config"],
        powered_schedule_sha=hashes["powered_schedule"],
        standard_schedule_sha=hashes["standard_schedule"],
        policy_sha=hashes["policy"],
        suite="audit",
        stage="deep-audit",
        look="final",
        topology="8-processes",
        powered_visits=2000,
        standard_visits=800,
        include_standard_champion=True,
    )
    assert len(full_matrix) == 4


def test_suite_splits_and_manifests_are_deterministic(tmp_path):
    source = tmp_path / "labeled.jsonl"
    ordinary = [position(index, "ordinary") for index in range(9)]
    specialized = [
        position(100 + index, label) for index, label in enumerate(SPECIALIZED_LABELS)
    ]
    all_positions = ordinary + specialized
    write_labeled_positions(source, all_positions)

    first = build_evaluation_suites(
        [source],
        tmp_path / "suites-a",
        seed="frozen-seed",
        pairs_per_position=2,
        policy_path=LEGACY_POLICY_PATH,
    )
    second = build_evaluation_suites(
        [source],
        tmp_path / "suites-b",
        seed="frozen-seed",
        pairs_per_position=2,
        policy_path=LEGACY_POLICY_PATH,
    )
    assert first.manifest == second.manifest
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest["sources"][0]["sha256"] == file_sha256(source)
    policy = json.loads(LEGACY_POLICY_PATH.read_text(encoding="utf-8"))
    assert first.manifest["policy_hash"] == suite_canonical_sha256(policy)
    assert first.manifest["source_revision"] == policy["frozen_plan"]["source_revision"]

    banks = {bank["name"]: bank for bank in first.manifest["banks"]}
    ordinary_hash_sets = [set(banks[name]["contentSha256s"]) for name in ORDINARY_BANKS]
    ordinary_semantic_sets = [
        set(banks[name]["semanticSha256s"]) for name in ORDINARY_BANKS
    ]
    assert all(ordinary_hash_sets)
    assert not ordinary_hash_sets[0].intersection(ordinary_hash_sets[1])
    assert not ordinary_hash_sets[0].intersection(ordinary_hash_sets[2])
    assert not ordinary_hash_sets[1].intersection(ordinary_hash_sets[2])
    assert set.union(*ordinary_hash_sets) == {
        suite_canonical_sha256(row) for row in ordinary
    }
    assert set.union(*ordinary_semantic_sets) == {
        semantic_position_sha256(row) for row in ordinary
    }
    assert not ordinary_semantic_sets[0].intersection(ordinary_semantic_sets[1])
    assert not ordinary_semantic_sets[0].intersection(ordinary_semantic_sets[2])
    assert not ordinary_semantic_sets[1].intersection(ordinary_semantic_sets[2])
    assert set(SPECIALIZED_LABELS).issubset(banks)
    assert len(
        {bank["schedule"]["baseSeed"] for bank in first.manifest["banks"]}
    ) == len(first.manifest["banks"])

    source_canonical = {suite_canonical_json(row) for row in all_positions}
    for bank in first.manifest["banks"]:
        positions_path = first.output_dir / bank["positions"]["path"]
        emitted = {
            suite_canonical_json(json.loads(line))
            for line in positions_path.read_text(encoding="utf-8").splitlines()
        }
        assert emitted.issubset(source_canonical)
        schedule_rows = load_schedule(first.output_dir / bank["schedule"]["path"])
        assert len(schedule_rows) == bank["schedule"]["rowCount"]
        assert bank["schedule"]["rowCount"] == 4 * bank["positions"]["rowCount"]
        assert {row["suite"] for row in schedule_rows} == {bank["name"]}
        assert {row["suiteBankSha256"] for row in schedule_rows} == {
            bank["positions"]["sha256"]
        }
        assert {row["positionSemanticSha256"] for row in schedule_rows} == set(
            bank["semanticSha256s"]
        )

    reused = build_evaluation_suites(
        [source],
        first.output_dir,
        seed="frozen-seed",
        pairs_per_position=2,
        policy_path=LEGACY_POLICY_PATH,
    )
    assert reused.reused is True
    with pytest.raises(FileExistsError, match="contradictory manifest"):
        build_evaluation_suites(
            [source],
            first.output_dir,
            seed="different-seed",
            pairs_per_position=2,
            policy_path=LEGACY_POLICY_PATH,
        )


def test_runner_binds_suite_manifest_bank_and_schedule_metadata(tmp_path):
    source = tmp_path / "labeled.jsonl"
    write_labeled_positions(source, [position(index) for index in range(6)])
    suites = build_evaluation_suites(
        [source],
        tmp_path / "suites",
        seed="confirmation-seed",
        policy_path=LEGACY_POLICY_PATH,
    )
    bank = next(
        bank for bank in suites.manifest["banks"] if bank["name"] == "confirmation"
    )

    evaluation_dir = tmp_path / "evaluation-inputs"
    evaluation_dir.mkdir()
    files = evaluation_files(evaluation_dir, pair_count=1)
    files["schedule"] = suites.output_dir / bank["schedule"]["path"]
    files["rows"] = load_schedule(files["schedule"])
    files["suite_manifest"] = suites.manifest_path
    files["spec"] = EvaluationSpec(
        candidate_model_sha=file_sha256(files["candidate"]),
        reference_model_sha=file_sha256(files["reference"]),
        original_model_sha=file_sha256(files["original"]),
        config_sha=file_sha256(files["config"]),
        schedule_sha=file_sha256(files["schedule"]),
        policy_sha=canonical_sha256(
            json.loads(files["policy"].read_text(encoding="utf-8"))
        ),
        comparison="candidate-vs-champion-powered",
        suite="confirmation",
        stage="stage-3",
        look="look-1",
        topology="2-processes",
        max_visits=2000,
        suite_manifest_sha=suites.manifest_sha256,
        suite_bank_sha=bank["positions"]["sha256"],
        schedule_id=bank["schedule"]["scheduleId"],
    )
    outcome = run_runner(make_runner(files, FakeMatchRunner()), files)
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schedule"]["suiteManifestSha256"] == suites.manifest_sha256
    assert manifest["schedule"]["suiteBankSha256"] == bank["positions"]["sha256"]
    assert manifest["cell"]["gameCount"] == bank["schedule"]["rowCount"]
    assert manifest["cell"]["colorPairCount"] == bank["schedule"]["pairCount"]
    finalized = validate_result_jsonl(outcome.result_path, files["rows"])
    assert {row["metadata"] for row in finalized} == {"ordinary"}
    assert {row["suite"] for row in finalized} == {"confirmation"}
    assert {row["suiteBankSha256"] for row in finalized} == {
        bank["positions"]["sha256"]
    }


def test_suite_builder_records_explicit_exclusions(tmp_path):
    source = tmp_path / "labeled.jsonl"
    rows = [position(index) for index in range(6)] + [position(100, "tactical")]
    write_labeled_positions(source, rows)
    excluded_hash = suite_canonical_sha256(rows[-1])
    result = build_evaluation_suites(
        [source],
        tmp_path / "suites",
        seed="seed",
        exclude_content_hashes=[excluded_hash],
        policy_path=LEGACY_POLICY_PATH,
    )
    assert result.manifest["includedRowCount"] == len(rows) - 1
    assert result.manifest["exclusions"] == [
        {
            "contentSha256": excluded_hash,
            "semanticSha256": semantic_position_sha256(rows[-1]),
            "labels": ["tactical"],
            "reason": "explicit-content-hash-exclusion",
            "source": {"name": source.name, "line": len(rows)},
        }
    ]
    assert "tactical" not in {bank["name"] for bank in result.manifest["banks"]}


def test_suite_builder_rejects_unknown_labels_and_duplicate_content(tmp_path):
    unknown = tmp_path / "unknown.jsonl"
    write_labeled_positions(
        unknown,
        [position(index) for index in range(3)] + [position(10, "not-a-suite")],
    )
    with pytest.raises(ValueError, match="unsupported evaluation label"):
        build_evaluation_suites([unknown], tmp_path / "unknown-out", seed="seed")

    duplicate = tmp_path / "duplicate.jsonl"
    row = position(0)
    write_labeled_positions(duplicate, [row, row, position(1), position(2)])
    with pytest.raises(ValueError, match="duplicate position content"):
        build_evaluation_suites([duplicate], tmp_path / "duplicate-out", seed="seed")


def test_suite_builder_rejects_semantic_holdout_leakage(tmp_path):
    source_a = tmp_path / "ordinary.jsonl"
    source_b = tmp_path / "lead.jsonl"
    ordinary = position(0, "ordinary")
    relabeled = dict(ordinary)
    relabeled["metadata"] = {
        "label": "lead-40",
        "provenance": {"curator": "different"},
    }
    relabeled["weight"] = 999.0
    relabeled["hintLoc"] = "D4"
    write_labeled_positions(source_a, [ordinary, position(1), position(2), position(3)])
    write_labeled_positions(source_b, [relabeled])

    assert suite_canonical_sha256(ordinary) != suite_canonical_sha256(relabeled)
    assert semantic_position_sha256(ordinary) == semantic_position_sha256(relabeled)
    with pytest.raises(ValueError, match="duplicate gameplay-semantic position"):
        build_evaluation_suites(
            [source_a, source_b],
            tmp_path / "leaky-suites",
            seed="seed",
        )


def test_v2_policy_builds_exact_disjoint_holdouts_and_cumulative_prefixes(tmp_path):
    policy_path = tmp_path / "policy-v2-test.json"
    policy = write_small_v2_policy(policy_path)
    source = tmp_path / "labeled.jsonl"
    write_labeled_positions(source, small_v2_suite_sources())

    first = build_evaluation_suites(
        [source],
        tmp_path / "suites-a",
        seed="exact-v2-seed",
        policy_path=policy_path,
    )
    second = build_evaluation_suites(
        [source],
        tmp_path / "suites-b",
        seed="exact-v2-seed",
        policy_path=policy_path,
    )
    assert first.manifest == second.manifest
    assert first.manifest["policy_hash"] == suite_canonical_sha256(policy)
    assert first.manifest["exactPolicyQuotas"] is True
    assert len(first.manifest["cells"]) == 15

    stage_1 = resolve_manifest_cell(
        first.manifest,
        stage="stage-1",
        look="automatic",
        comparison="candidate-vs-champion-powered",
        suite="discovery",
    )
    assert stage_1["color_pairs"] == 1
    assert stage_1["visits"] == 400
    stage_2_ordinary = resolve_manifest_cell(
        first.manifest,
        stage="stage-2",
        look="automatic",
        comparison="candidate-vs-champion-powered",
        suite="discovery",
    )
    assert stage_2_ordinary["color_pairs"] == 2
    assert stage_2_ordinary["visits"] == 800
    for suite in ("lead-40", "lead-80"):
        stage_2_lead = resolve_manifest_cell(
            first.manifest,
            stage="stage-2",
            look="automatic",
            comparison=f"candidate-vs-champion-powered-{suite}",
            suite=suite,
        )
        assert stage_2_lead["color_pairs"] == 1
        assert stage_2_lead["visits"] == 800

    banks = {bank["qualifiedName"]: bank for bank in first.manifest["banks"]}
    for label in ("ordinary", "lead-40", "lead-80"):
        names = (
            ORDINARY_BANKS
            if label == "ordinary"
            else tuple(f"{label}-{holdout}" for holdout in ORDINARY_BANKS)
        )
        semantic_sets = [set(banks[name]["semanticSha256s"]) for name in names]
        assert all(semantic_sets)
        assert not semantic_sets[0].intersection(semantic_sets[1])
        assert not semantic_sets[0].intersection(semantic_sets[2])
        assert not semantic_sets[1].intersection(semantic_sets[2])

    for cell_name in (
        "powered_candidate_vs_champion",
        "lead_40",
        "lead_80",
    ):
        comparison = next(
            cell["comparison"]
            for cell in first.manifest["cells"]
            if cell["cell_name"] == cell_name
        )
        suite = (
            "confirmation"
            if cell_name == "powered_candidate_vs_champion"
            else cell_name.replace("_", "-")
        )
        look_1 = resolve_manifest_cell(
            first.manifest,
            stage="stage-3",
            look="look-1",
            comparison=comparison,
            suite=suite,
        )
        look_2 = resolve_manifest_cell(
            first.manifest,
            stage="stage-3",
            look="look-2",
            comparison=comparison,
            suite=suite,
        )
        look_1_data = (first.output_dir / look_1["schedule_path"]).read_bytes()
        look_2_data = (first.output_dir / look_2["schedule_path"]).read_bytes()
        assert look_2_data.startswith(look_1_data)
        assert look_1["schedule_id"] == look_2["schedule_id"]
        assert (
            look_1["independent_cluster_ids"]
            == look_2["independent_cluster_ids"][: look_1["color_pairs"]]
        )
        assert look_1["color_pairs"] == len(look_1["independent_cluster_ids"])


def test_v2_policy_rejects_insufficient_independent_positions(tmp_path):
    policy_path = tmp_path / "policy-v2-test.json"
    write_small_v2_policy(policy_path)
    source = tmp_path / "labeled.jsonl"
    rows = small_v2_suite_sources()
    write_labeled_positions(
        source,
        [row for row in rows if row["metadata"] != "lead-80"][:],
    )
    with pytest.raises(ValueError, match="insufficient independent lead-80"):
        build_evaluation_suites(
            [source],
            tmp_path / "suites",
            seed="too-small",
            policy_path=policy_path,
        )


def test_manifest_cell_resolution_and_cluster_hash_tamper_detection(tmp_path):
    policy_path = tmp_path / "policy-v2-test.json"
    write_small_v2_policy(policy_path)
    source = tmp_path / "labeled.jsonl"
    write_labeled_positions(source, small_v2_suite_sources())
    suites = build_evaluation_suites(
        [source], tmp_path / "suites", seed="resolve", policy_path=policy_path
    )
    cell = resolve_manifest_cell(
        suites.manifest,
        stage="stage-3",
        look="look-1",
        comparison="candidate-vs-champion-powered",
        suite="confirmation",
    )
    assert cell["color_pairs"] == 2
    assert (
        file_sha256(suites.output_dir / cell["schedule_path"]) == cell["schedule_hash"]
    )

    tampered = json.loads(suite_canonical_json(suites.manifest))
    target = next(
        item for item in tampered["cells"] if item["cell_id"] == cell["cell_id"]
    )
    target["independent_cluster_ids_hash"] = "0" * 64
    payload = dict(tampered)
    payload.pop("manifestPayloadSha256")
    tampered["manifestPayloadSha256"] = suite_canonical_sha256(payload)
    with pytest.raises(EvaluationValidationError, match="cell_id|cluster_ids_hash"):
        resolve_manifest_cell(
            tampered,
            stage="stage-3",
            look="look-1",
            comparison="candidate-vs-champion-powered",
            suite="confirmation",
        )


def test_runner_resolves_exact_manifest_cell_and_records_provenance(tmp_path):
    policy_path = tmp_path / "policy-v2-test.json"
    policy = write_small_v2_policy(policy_path)
    source = tmp_path / "labeled.jsonl"
    write_labeled_positions(source, small_v2_suite_sources())
    suites = build_evaluation_suites(
        [source], tmp_path / "suites", seed="runner-cell", policy_path=policy_path
    )
    cell = resolve_manifest_cell(
        suites.manifest,
        stage="stage-3",
        look="look-1",
        comparison="candidate-vs-champion-powered",
        suite="confirmation",
    )

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = evaluation_files(inputs, pair_count=1)
    files["policy"] = policy_path
    files["schedule"] = suites.output_dir / cell["schedule_path"]
    files["rows"] = load_schedule(files["schedule"])
    files["suite_manifest"] = suites.manifest_path
    files["spec"] = EvaluationSpec(
        candidate_model_sha=file_sha256(files["candidate"]),
        reference_model_sha=file_sha256(files["reference"]),
        original_model_sha=file_sha256(files["original"]),
        config_sha=file_sha256(files["config"]),
        schedule_sha=cell["schedule_hash"],
        policy_sha=suite_canonical_sha256(policy),
        comparison=cell["comparison"],
        suite=cell["suite"],
        stage=cell["stage"],
        look=cell["look"],
        topology="2-processes",
        max_visits=cell["visits"],
        suite_manifest_sha=suites.manifest_sha256,
        suite_bank_sha=cell["bank_hash"],
        schedule_id=cell["schedule_id"],
    )
    outcome = run_runner(
        make_runner(
            files,
            FakeMatchRunner(),
            include_move_traces=True,
            shard_count=1,
            max_parallel=1,
        ),
        files,
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schedule"]["manifestCell"] == cell
    assert manifest["schedule"]["manifestCellSha256"] == canonical_sha256(cell)
    results = validate_result_jsonl(outcome.result_path, files["rows"])
    assert [results[index]["independentClusterId"] for index in range(0, 4, 2)] == cell[
        "independent_cluster_ids"
    ]
