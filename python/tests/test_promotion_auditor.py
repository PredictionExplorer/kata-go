import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.build_evaluation_suites import semantic_position_sha256
from risk_score.evaluation_runner import load_schedule
from risk_score.generate_schedule import build_schedule
from risk_score.promotion_auditor import (
    AuditDecisionError,
    AuditorRuntime,
    EvaluationArtifacts,
    PromotionAuditor,
    PromotionAuditorError,
    parse_args,
    publish_canonical_json,
    tree_manifest,
)
from risk_score.promotion_controller import PromotionController
from risk_score.promotion_state import (
    ChampionRecord,
    EventProvenance,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def position(index, label):
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


def build_bank(suites, name, label, index):
    bank_dir = suites / "banks"
    schedule_dir = suites / "schedules"
    positions_path = bank_dir / f"{name}.jsonl"
    positions = [position(index, label)]
    write_jsonl(positions_path, positions)
    bank_hash = sha256_file(positions_path)
    schedule_id = f"audit-{name}"
    rows = build_schedule(
        positions,
        base_seed=f"seed-{name}",
        schedule_id=schedule_id,
    )
    for row in rows:
        semantic_hash = semantic_position_sha256(row["startPosition"])
        row.update(
            {
                "suite": name,
                "suiteBank": name,
                "suiteBankSha256": bank_hash,
                "suiteQualifiedName": name,
                "suiteHoldout": "audit",
                "positionContentSha256": canonical_sha256(row["startPosition"]),
                "positionSemanticSha256": semantic_hash,
                "independentClusterId": semantic_hash,
            }
        )
    schedule_path = schedule_dir / f"{name}.jsonl"
    write_jsonl(schedule_path, rows)
    return {
        "name": name,
        "qualifiedName": name,
        "positions": {
            "path": positions_path.relative_to(suites).as_posix(),
            "sha256": bank_hash,
            "rowCount": 1,
        },
        "schedule": {
            "path": schedule_path.relative_to(suites).as_posix(),
            "sha256": sha256_file(schedule_path),
            "scheduleId": schedule_id,
            "rowCount": 2,
            "pairCount": 1,
        },
    }


@pytest.fixture
def audit_fixture(tmp_path):
    promotion = tmp_path / "promotion"
    rollout = promotion / "rollouts"
    admitted = tmp_path / "selfplay"
    inbox = promotion / "ipc" / "rollout-reports"
    accepted = promotion / "accepted"
    suites = tmp_path / "suites"
    for path in (
        promotion,
        rollout,
        admitted,
        inbox,
        accepted,
        suites,
        promotion / "transactions",
        promotion / "audits" / "queue",
    ):
        path.mkdir(parents=True, exist_ok=True)

    candidate_path = accepted / "candidate-source"
    candidate_path.mkdir()
    candidate_model = candidate_path / "model.bin.gz"
    candidate_model.write_bytes(b"candidate-model")
    candidate_hash = sha256_file(candidate_model)

    champion_model_bytes = b"champion-model"
    champion_hash = hashlib.sha256(champion_model_bytes).hexdigest()
    original_path = tmp_path / "original" / "model.bin.gz"
    original_path.parent.mkdir()
    original_path.write_bytes(b"original-model")
    original_hash = sha256_file(original_path)

    b28_path = promotion / "controls" / "b28" / "model.bin.gz"
    b28_path.parent.mkdir(parents=True)
    b28_path.write_bytes(b"b28-model")
    b28_hash = sha256_file(b28_path)

    config_path = tmp_path / "powered.cfg"
    config_path.write_text("maxVisits = 1\n", encoding="utf-8")
    config_hash = sha256_file(config_path)
    selfplay_config_path = tmp_path / "selfplay.cfg"
    selfplay_config_path.write_text("switchNetsMidGame = false\n", encoding="utf-8")
    selfplay_config_hash = sha256_file(selfplay_config_path)

    policy = {
        "schema_version": 3,
        "policy_version": "risk-seeking-checkpoint-promotion-v3",
        "evaluation_stages": {
            "deep_audit": {
                "promotion_interval": 5,
                "ordinary_color_pairs": 1,
                "lead_40_color_pairs": 1,
                "lead_80_color_pairs": 1,
                "visits": [2000, 8000],
                "controls": [
                    "candidate",
                    "champion",
                    "original",
                    "b28",
                ],
                "asynchronous": True,
                "can_trigger_rollback": True,
            }
        },
        "promotion_thresholds": {
            "powered_win_rate_vs_champion_lower_bound_strictly_above": 0.47,
            "true_no_result_rate_strictly_below": 0.001,
        },
        "rollout": {
            "worker_count": 7,
            "canary_workers": 1,
            "canary_games": 2,
            "canary_fresh_audit_color_pairs": 1,
            "intermediate_workers": 3,
            "full_workers": 7,
            "games_per_worker_initial_threads": 100,
            "switch_networks_mid_game": False,
        },
    }
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, policy)
    policy_hash = canonical_sha256(policy)

    banks = [
        build_bank(suites, "audit", "ordinary", 1),
        build_bank(suites, "lead-40-audit", "lead-40", 2),
        build_bank(suites, "lead-80-audit", "lead-80", 3),
    ]
    suite_payload = {
        "schemaVersion": 3,
        "contract": "risk-score-authoritative-evaluation-manifest-v3",
        "policy_hash": policy_hash,
        "policy_version": policy["policy_version"],
        "source_revision": digest("source-revision"),
        "banks": banks,
        "cells": [],
    }
    suite_manifest = {
        **suite_payload,
        "manifestPayloadSha256": canonical_sha256(suite_payload),
    }
    suite_manifest_path = suites / "manifest.json"
    write_json(suite_manifest_path, suite_manifest)
    audit_schedule = suites / banks[0]["schedule"]["path"]

    gpu_config_path = tmp_path / "gpu-lease.json"
    write_json(gpu_config_path, {"unused": True})
    gpu_config_hash = sha256_file(gpu_config_path)

    runtime = AuditorRuntime(
        promotion_root=promotion,
        rollout_quarantine=rollout,
        admitted_selfplay=admitted,
        rollout_report_inbox=inbox,
        accepted_models=accepted,
        original_model_path=original_path,
        policy_path=policy_path,
        powered_config_path=config_path,
        selfplay_config_path=selfplay_config_path,
        suite_manifest_path=suite_manifest_path,
        audit_schedule_path=audit_schedule,
        gpu_lease_config_path=gpu_config_path,
        policy=policy,
        original_hash=original_hash,
        policy_hash=policy_hash,
        powered_config_hash=config_hash,
        suite_manifest_hash=sha256_file(suite_manifest_path),
        audit_schedule_hash=sha256_file(audit_schedule),
        selfplay_config_hash=selfplay_config_hash,
        gpu_lease_config_hash=gpu_config_hash,
        worker_count=7,
        canary_worker_count=1,
        intermediate_worker_count=3,
        worker_threads=100,
    )

    generation_id = "generation-auditor"
    previous_generation = "generation-previous"
    candidate_leaf = (
        accepted / "generations" / candidate_hash / generation_id / "model.bin.gz"
    )
    candidate_leaf.parent.mkdir(parents=True)
    candidate_leaf.write_bytes(candidate_model.read_bytes())
    champion_leaf = (
        accepted / "generations" / champion_hash / previous_generation / "model.bin.gz"
    )
    champion_leaf.parent.mkdir(parents=True)
    champion_leaf.write_bytes(champion_model_bytes)

    transaction = promotion / "transactions" / generation_id
    transaction.mkdir()
    pass_report = promotion / "reports" / "pass.json"
    write_json(pass_report, {"schema_version": 1, "finalized": True})
    intent = {
        "schema_version": 1,
        "generation_id": generation_id,
        "candidate_hash": candidate_hash,
        "tested_champion_hash": champion_hash,
        "config_hash": config_hash,
        "policy_hash": policy_hash,
        "selfplay_config_hash": runtime.selfplay_config_hash,
        "topology": "7-workers-100-threads",
        "pass_report_path": str(pass_report),
        "pass_report_hash": sha256_file(pass_report),
    }
    write_json(transaction / "intent.json", intent)
    provenance = EventProvenance(
        controller_hash=digest("controller"),
        source_hash=digest("source"),
        original_hash=original_hash,
        config_hash=config_hash,
        schedule_hash=sha256_file(audit_schedule),
        policy_hash=policy_hash,
    )
    previous = ChampionRecord.build(
        champion_hash=champion_hash,
        generation_id=previous_generation,
        previous_champion_hash=None,
        provenance=provenance,
        activated_at_utc="2026-08-08T00:00:00.000000Z",
        evaluation_key=None,
        pass_report_path=None,
        pass_report_hash=None,
        actor="test",
        bootstrap=True,
    )
    write_json(transaction / "previous-champion.json", previous.to_dict())

    fixture = {
        "runtime": runtime,
        "promotion": promotion,
        "rollout": rollout,
        "inbox": inbox,
        "accepted": accepted,
        "suites": suites,
        "banks": banks,
        "candidate_hash": candidate_hash,
        "champion_hash": champion_hash,
        "original_hash": original_hash,
        "b28_path": b28_path,
        "b28_hash": b28_hash,
        "generation_id": generation_id,
        "transaction": transaction,
        "intent": intent,
    }
    add_worker(fixture, 0)
    return fixture


def worker_phase(worker_id):
    return "canary" if worker_id == 0 else "intermediate"


def add_worker(fixture, worker_id, *, partial=False, game_count=None):
    runtime = fixture["runtime"]
    generation_id = fixture["generation_id"]
    rollout = fixture["rollout"] / generation_id
    worker_root = rollout / f"worker-{worker_id:03d}"
    data_root = rollout / "data" / f"worker-{worker_id:03d}"
    ack_root = rollout / "acknowledgements"
    worker_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    ack_root.mkdir(parents=True, exist_ok=True)
    if game_count is None:
        game_count = runtime.canary_games
    games = "".join(
        f"(;GM[1]FF[4]SZ[19]RE[B+{index + 1}.5]C[worker-{worker_id}-game-{index}])\n"
        for index in range(game_count)
    )
    (data_root / "games.sgfs").write_text(games, encoding="utf-8")
    if partial:
        (data_root / "chunk.partial").write_bytes(b"incomplete")

    identity = {
        "pid": 1000 + worker_id,
        "start_time_ticks": 2000 + worker_id,
        "command_sha256": digest(f"worker-command-{worker_id}"),
    }
    phase = worker_phase(worker_id)
    write_json(
        worker_root / "intent.json",
        {
            "schema_version": 1,
            "worker_id": worker_id,
            "generation_id": generation_id,
            "model_hash": fixture["candidate_hash"],
            "selfplay_config_hash": runtime.selfplay_config_hash,
            "policy": str(runtime.policy_path),
            "policy_hash": runtime.policy_hash,
            "threads": runtime.worker_threads,
        },
    )
    write_json(
        worker_root / f"launch-{phase}.json",
        {
            "schema_version": 1,
            "generation_id": generation_id,
            "model_hash": fixture["candidate_hash"],
            "worker_id": worker_id,
            "phase": phase,
            "selfplay_config_hash": runtime.selfplay_config_hash,
            "policy_hash": runtime.policy_hash,
            "supervisor_key": f"{generation_id}:worker-{worker_id:03d}",
            "process_identity": identity,
            "process_identity_verified": True,
        },
    )
    manifest_hash, _, _ = tree_manifest(data_root)
    source_report = {
        "schema_version": 1,
        "finalized": True,
        "generation_id": generation_id,
        "worker_id": worker_id,
        "model_hash": fixture["candidate_hash"],
        "selfplay_config_hash": runtime.selfplay_config_hash,
        "policy_hash": runtime.policy_hash,
        "threads": runtime.worker_threads,
        "output_manifest_hash": manifest_hash,
        "closed_files": True,
        "process_identity": identity,
    }
    source_hash = hashlib.sha256(
        canonical_json_bytes(source_report) + b"\n"
    ).hexdigest()
    write_json(
        ack_root / f"worker-{worker_id:03d}.json",
        {**source_report, "report_hash": source_hash},
    )


def result_for(row, *, candidate_wins):
    bot_names = {0: "candidate", 1: "reference"}
    candidate_color = "B" if row["blackBot"] == 0 else "W"
    winner = (
        candidate_color if candidate_wins else ("W" if candidate_color == "B" else "B")
    )
    score = -1.0 if winner == "B" else 1.0
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
        "finalResult": f"{winner}+1",
        "finalWhiteMinusBlackScore": score,
        "winner": winner,
        "moveCount": 2,
        "blackMoveCount": 1,
        "whiteMoveCount": 1,
        "startTurnNumber": row["startPosition"]["initialTurnNumber"],
        "hitTurnLimit": False,
        "resignation": False,
        "noResult": False,
        "scored": True,
        "gameHash": f"hash-{row['gameId']}",
    }


def moves_for(row, result):
    return [
        {
            "schemaVersion": 1,
            "scheduleId": row["scheduleId"],
            "gameId": row["gameId"],
            "pairId": row["pairId"],
            "positionId": row["positionId"],
            "seed": row["seed"],
            "turnNumber": result["startTurnNumber"] + offset,
            "player": "B" if offset == 0 else "W",
            "bot": result["blackBot"] if offset == 0 else result["whiteBot"],
            "move": "D4" if offset == 0 else "Q16",
            "scoreLead": 0.0,
            "winProbability": 0.5,
        }
        for offset in range(2)
    ]


class FakeEvaluationExecutor:
    def __init__(self, *, fail_job=None, fail_generation=None):
        self.calls = []
        self.fail_job = fail_job
        self.fail_generation = fail_generation

    def __call__(self, job):
        self.calls.append(job.job_id)
        schedule = load_schedule(job.schedule_path)
        candidate_wins = (
            job.job_id != self.fail_job
            and job.generation_id != self.fail_generation
        )
        results = [result_for(row, candidate_wins=candidate_wins) for row in schedule]
        moves = [
            move
            for row, result in zip(schedule, results)
            for move in moves_for(row, result)
        ]
        output = job.output_root / "injected"
        results_path = output / "results.jsonl"
        moves_path = output / "moves.jsonl"
        write_jsonl(results_path, results)
        write_jsonl(moves_path, moves)
        return EvaluationArtifacts(
            results_path=results_path.resolve(),
            moves_path=moves_path.resolve(),
        )


def make_auditor(fixture, executor, **kwargs):
    return PromotionAuditor(
        fixture["runtime"],
        evaluation_executor=executor,
        lease_factory=lambda: contextlib.nullcontext(
            {"lease_id": "test-lease", "trainer_restored": True}
        ),
        **kwargs,
    )


def install_canary_pass(fixture, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    write_json(
        fixture["transaction"] / "canary-pass.json",
        {**report, "report_hash": sha256_file(report_path)},
    )


def add_generation(fixture, generation_id):
    generated = dict(fixture)
    transaction = fixture["runtime"].transactions / generation_id
    transaction.mkdir()
    intent = {**fixture["intent"], "generation_id": generation_id}
    write_json(transaction / "intent.json", intent)
    previous = json.loads(
        (fixture["transaction"] / "previous-champion.json").read_text(
            encoding="utf-8"
        )
    )
    write_json(transaction / "previous-champion.json", previous)
    source = (
        fixture["accepted"]
        / "generations"
        / fixture["candidate_hash"]
        / fixture["generation_id"]
        / "model.bin.gz"
    )
    candidate = (
        fixture["accepted"]
        / "generations"
        / fixture["candidate_hash"]
        / generation_id
        / "model.bin.gz"
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(source.read_bytes())
    generated.update(
        {
            "generation_id": generation_id,
            "transaction": transaction,
            "intent": intent,
        }
    )
    add_worker(generated, 0)
    return generated


def controller_contract_validator(fixture):
    runtime = fixture["runtime"]
    controller = object.__new__(PromotionController)
    controller.automatic = True
    controller.runtime = SimpleNamespace(
        promotion_root=runtime.promotion_root,
        frozen_policy=runtime.policy,
        audit_schedule_path=runtime.audit_schedule_path,
        controller=SimpleNamespace(
            policy_hash=runtime.policy_hash,
            selfplay_config_hash=runtime.selfplay_config_hash,
            audit_schedule_hash=runtime.audit_schedule_hash,
            suite_manifest_hash=runtime.suite_manifest_hash,
            canary_worker_count=runtime.canary_worker_count,
            intermediate_worker_count=runtime.intermediate_worker_count,
        ),
    )
    return controller


def make_deep_request(fixture, *, tamper_cell=False, noncanonical=False):
    runtime = fixture["runtime"]
    policy_contract = runtime.policy["evaluation_stages"]["deep_audit"]
    labels = ("ordinary", "lead-40", "lead-80")
    audit_banks = []
    for label, bank in zip(labels, fixture["banks"]):
        schedule_path = fixture["suites"] / bank["schedule"]["path"]
        audit_banks.append(
            {
                "label": label,
                "qualified_name": bank["qualifiedName"],
                "schedule_path": str(schedule_path.resolve()),
                "schedule_hash": bank["schedule"]["sha256"],
                "schedule_id": bank["schedule"]["scheduleId"],
                "bank_hash": bank["positions"]["sha256"],
                "color_pairs": 1,
            }
        )
    control_hashes = {
        "candidate": fixture["candidate_hash"],
        "champion": fixture["champion_hash"],
        "original": fixture["original_hash"],
        "b28": fixture["b28_hash"],
    }
    cells = []
    for bank in audit_banks:
        for visit_count in policy_contract["visits"]:
            for control in policy_contract["controls"]:
                payload = {
                    "label": bank["label"],
                    "visit_count": visit_count,
                    "control": control,
                    "control_model_hash": control_hashes[control],
                    "schedule_hash": bank["schedule_hash"],
                    "bank_hash": bank["bank_hash"],
                    "color_pairs": bank["color_pairs"],
                }
                cells.append(
                    {
                        "cell_id": "deep-audit-cell-" + canonical_sha256(payload),
                        **payload,
                    }
                )
    if tamper_cell:
        cells[0] = {**cells[0], "cell_id": "deep-audit-cell-forged"}
    request = {
        "schema_version": 2,
        "contract": "risk-score-deep-audit-request-v2",
        "generation_id": fixture["generation_id"],
        "candidate_hash": fixture["candidate_hash"],
        "previous_champion_hash": fixture["champion_hash"],
        "policy_path": str(runtime.policy_path),
        "policy_hash": runtime.policy_hash,
        "policy_version": runtime.policy["policy_version"],
        "suite_manifest_path": str(runtime.suite_manifest_path),
        "suite_manifest_hash": runtime.suite_manifest_hash,
        "audit_schedule_path": str(runtime.audit_schedule_path),
        "audit_schedule_hash": runtime.audit_schedule_hash,
        "audit_schedule_id": audit_banks[0]["schedule_id"],
        "audit_bank_hash": audit_banks[0]["bank_hash"],
        "activation_event_hash": digest("activation-event"),
        "scheduled_at_utc": "2026-08-08T00:00:00.000000Z",
        "reasons": ["near-safety-boundary"],
        "audit_contract": policy_contract,
        "audit_banks": audit_banks,
        "visit_tiers": list(policy_contract["visits"]),
        "controls": list(policy_contract["controls"]),
        "control_model_hashes": control_hashes,
        "b28_model_path": str(fixture["b28_path"].resolve()),
        "audit_cells": cells,
    }
    path = (
        runtime.promotion_root / "audits" / "queue" / f"{fixture['generation_id']}.json"
    )
    if noncanonical:
        path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    else:
        write_json(path, request)
    return path, request


def test_canary_and_intermediate_reports_derive_closed_worker_evidence(
    audit_fixture,
):
    executor = FakeEvaluationExecutor()
    auditor = make_auditor(audit_fixture, executor)

    canary = auditor.produce_rollout_report(audit_fixture["generation_id"], "canary")
    retry = auditor.produce_rollout_report(audit_fixture["generation_id"], "canary")
    assert canary.decision == "PASS"
    assert retry.reused is True
    assert executor.calls == ["fresh-audit"]
    report = json.loads(canary.output_path.read_text(encoding="utf-8"))
    assert report["decision"] == "PASS"
    assert report["game_count"] == 2
    assert report["worker_evidence"][0]["game_count"] == 2
    assert report["fresh_audit_pairs"] == 1
    assert canary.output_path.read_bytes() == canonical_json_bytes(report) + b"\n"
    validated = controller_contract_validator(
        audit_fixture
    )._validate_rollout_health_report(
        audit_fixture["generation_id"],
        audit_fixture["candidate_hash"],
        "canary",
        canary.output_path,
        canary.output_sha256,
    )
    assert validated["decision"] == "PASS"
    statistics = json.loads(canary.artifact_paths[1].read_text(encoding="utf-8"))
    assert statistics["contract"] == "risk-score-canary-fresh-audit-statistics-v1"
    assert statistics["candidate_win_rate"] == 1.0

    install_canary_pass(audit_fixture, canary.output_path)
    add_worker(audit_fixture, 1)
    add_worker(audit_fixture, 2)
    intermediate = auditor.produce_rollout_report(
        audit_fixture["generation_id"], "intermediate"
    )
    intermediate_report = json.loads(
        intermediate.output_path.read_text(encoding="utf-8")
    )
    assert intermediate_report["decision"] == "PASS"
    assert intermediate_report["worker_count"] == 3
    assert intermediate_report["game_count"] == 6
    assert [row["worker_id"] for row in intermediate_report["worker_evidence"]] == [
        0,
        1,
        2,
    ]


def test_forged_or_malformed_acknowledgements_fail_closed(audit_fixture):
    executor = FakeEvaluationExecutor()
    auditor = make_auditor(audit_fixture, executor)
    ack_path = (
        audit_fixture["rollout"]
        / audit_fixture["generation_id"]
        / "acknowledgements"
        / "worker-000.json"
    )
    forged = json.loads(ack_path.read_text(encoding="utf-8"))
    forged["output_manifest_hash"] = digest("forged-output")
    payload = dict(forged)
    payload.pop("report_hash")
    forged["report_hash"] = hashlib.sha256(
        canonical_json_bytes(payload) + b"\n"
    ).hexdigest()
    write_json(ack_path, forged)
    with pytest.raises(PromotionAuditorError, match="output changed"):
        auditor.derive_rollout_health(audit_fixture["generation_id"], "canary")
    assert executor.calls == []

    ack_path.write_text(
        '{"schema_version":1,"schema_version":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(PromotionAuditorError, match="duplicate JSON key"):
        auditor.derive_rollout_health(audit_fixture["generation_id"], "canary")


def test_partial_worker_files_are_rejected_but_stale_publish_temps_do_not_block(
    audit_fixture,
):
    output = (
        audit_fixture["rollout"]
        / audit_fixture["generation_id"]
        / "data"
        / "worker-000"
    )
    (output / "late.partial").write_bytes(b"partial")
    with pytest.raises(PromotionAuditorError, match="partial worker output"):
        make_auditor(audit_fixture, FakeEvaluationExecutor()).derive_rollout_health(
            audit_fixture["generation_id"], "canary"
        )

    (output / "late.partial").unlink()
    add_worker(audit_fixture, 0)
    stale = (
        audit_fixture["inbox"]
        / f".{audit_fixture['generation_id']}.canary.json.partial-dead"
    )
    stale.write_bytes(b'{"truncated":')
    result = make_auditor(
        audit_fixture, FakeEvaluationExecutor()
    ).produce_rollout_report(audit_fixture["generation_id"], "canary")
    assert result.output_path.is_file()
    assert stale.is_file()


def test_canary_recovers_after_crash_without_reexecuting_games(audit_fixture):
    executor = FakeEvaluationExecutor()
    fired = []

    def crash(step):
        if step == "canary-runner-published" and not fired:
            fired.append(step)
            raise RuntimeError("simulated crash")

    crashing = make_auditor(audit_fixture, executor, failure_hook=crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.produce_rollout_report(audit_fixture["generation_id"], "canary")
    assert (audit_fixture["transaction"] / "fresh-audit-runner.json").is_file()
    assert not (audit_fixture["transaction"] / "fresh-audit-statistics.json").exists()

    recovered = make_auditor(audit_fixture, executor)
    result = recovered.produce_rollout_report(audit_fixture["generation_id"], "canary")
    assert result.decision == "PASS"
    assert executor.calls == ["fresh-audit"]


def test_canary_fail_is_published_canonically_and_is_recoverable(audit_fixture):
    executor = FakeEvaluationExecutor(fail_job="fresh-audit")
    crashed = []

    def crash_after_audit(step):
        if step == "canary-audit-published" and not crashed:
            crashed.append(step)
            raise RuntimeError("simulated pre-report crash")

    with pytest.raises(RuntimeError, match="simulated pre-report crash"):
        make_auditor(
            audit_fixture,
            executor,
            failure_hook=crash_after_audit,
        ).produce_rollout_report(audit_fixture["generation_id"], "canary")

    auditor = make_auditor(audit_fixture, executor)
    result = auditor.produce_rollout_report(
        audit_fixture["generation_id"], "canary"
    )
    report_path = (
        audit_fixture["inbox"] / f"{audit_fixture['generation_id']}.canary.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.decision == "FAIL"
    assert result.output_path == report_path
    assert report["decision"] == "FAIL"
    assert report["finalized"] is True
    assert report["behavior_pass"] is False
    assert report["fresh_audit_manifest_path"] == str(
        (audit_fixture["transaction"] / "fresh-audit.json").resolve()
    )
    assert report_path.read_bytes() == canonical_json_bytes(report) + b"\n"

    retry = auditor.produce_rollout_report(
        audit_fixture["generation_id"], "canary"
    )
    assert retry.decision == "FAIL"
    assert retry.reused is True
    assert retry.output_sha256 == result.output_sha256
    assert executor.calls == ["fresh-audit"]

    with pytest.raises(AuditDecisionError) as strict_failure:
        auditor.produce_rollout_report(
            audit_fixture["generation_id"],
            "canary",
            raise_on_failure=True,
        )
    assert strict_failure.value.evidence["decision"] == "FAIL"
    assert strict_failure.value.evidence["output_sha256"] == result.output_sha256
    assert report_path.is_file()

    statistics = json.loads(
        (audit_fixture["transaction"] / "fresh-audit-statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert statistics["decision"] == "FAIL"
    assert statistics["safety_failures"] > 0


def test_intermediate_worker_shortfall_publishes_finalized_fail(audit_fixture):
    executor = FakeEvaluationExecutor()
    auditor = make_auditor(audit_fixture, executor)
    canary = auditor.produce_rollout_report(
        audit_fixture["generation_id"], "canary"
    )
    install_canary_pass(audit_fixture, canary.output_path)
    add_worker(audit_fixture, 1, game_count=1)
    add_worker(audit_fixture, 2, game_count=1)

    result = auditor.produce_rollout_report(
        audit_fixture["generation_id"], "intermediate"
    )
    report = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.decision == "FAIL"
    assert report["decision"] == "FAIL"
    assert report["finalized"] is True
    assert report["throughput_pass"] is False
    assert report["game_count"] == 4
    assert result.output_path.read_bytes() == canonical_json_bytes(report) + b"\n"


def test_run_once_publishes_fail_and_continues_scanning(audit_fixture):
    second = add_generation(audit_fixture, "generation-second")
    executor = FakeEvaluationExecutor(
        fail_generation=audit_fixture["generation_id"]
    )
    auditor = make_auditor(audit_fixture, executor)

    scan = auditor.run_once()
    reports = {
        Path(item["output"]).name: item for item in scan["produced"]
    }
    first_name = f"{audit_fixture['generation_id']}.canary.json"
    second_name = f"{second['generation_id']}.canary.json"
    assert reports[first_name]["decision"] == "FAIL"
    assert reports[second_name]["decision"] == "PASS"
    assert scan["pending"] == []
    assert len(executor.calls) == 2

    retry = auditor.run_once()
    assert {item["decision"] for item in retry["produced"]} == {"PASS", "FAIL"}
    assert all(item["reused"] is True for item in retry["produced"])
    assert len(executor.calls) == 2


def test_executor_cannot_supply_an_arbitrary_pass_value(audit_fixture):
    def forged_executor(_job):
        return {"decision": "PASS"}

    auditor = make_auditor(audit_fixture, forged_executor)
    with pytest.raises(PromotionAuditorError, match="never a decision"):
        auditor.produce_rollout_report(audit_fixture["generation_id"], "canary")


def test_deep_audit_v2_is_output_derived_and_idempotent(audit_fixture):
    request_path, request = make_deep_request(audit_fixture)
    executor = FakeEvaluationExecutor()
    auditor = make_auditor(audit_fixture, executor)
    first = auditor.produce_deep_audit_report(request_path)
    retry = auditor.produce_deep_audit_report(request_path)
    assert first.decision == "PASS"
    assert retry.reused is True
    assert len(executor.calls) == len(request["audit_cells"]) == 24
    report = json.loads(first.output_path.read_text(encoding="utf-8"))
    assert report["contract"] == "risk-score-deep-audit-report-v2"
    assert report["decision"] == "PASS"
    assert report["rollback_required"] is False
    assert len(report["cells"]) == 24
    assert first.output_path.read_bytes() == canonical_json_bytes(report) + b"\n"
    reports = audit_fixture["runtime"].promotion_root / "audits" / "reports"
    reports.mkdir(parents=True)
    stored = controller_contract_validator(audit_fixture).record_deep_audit_report(
        audit_fixture["generation_id"],
        report_path=first.output_path,
        report_hash=first.output_sha256,
    )
    assert stored.is_file()


def test_deep_audit_failure_requires_rollback(audit_fixture):
    request_path, request = make_deep_request(audit_fixture)
    failing_cell = request["audit_cells"][0]["cell_id"]
    result = make_auditor(
        audit_fixture,
        FakeEvaluationExecutor(fail_job=failing_cell),
    ).produce_deep_audit_report(request_path)
    report = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.decision == "FAIL"
    assert report["decision"] == "FAIL"
    assert report["rollback_required"] is True
    assert report["cells"][0]["decision"] == "FAIL"


@pytest.mark.parametrize("case", ("forged-cell", "noncanonical"))
def test_deep_audit_rejects_forged_or_partial_requests(audit_fixture, case):
    request_path, _ = make_deep_request(
        audit_fixture,
        tamper_cell=case == "forged-cell",
        noncanonical=case == "noncanonical",
    )
    executor = FakeEvaluationExecutor()
    auditor = make_auditor(audit_fixture, executor)
    with pytest.raises(PromotionAuditorError):
        auditor.produce_deep_audit_report(request_path)
    assert executor.calls == []


def test_immutable_publication_rejects_conflicts(tmp_path):
    output = tmp_path / "report.json"
    assert publish_canonical_json(output, {"decision": "PASS"}) is False
    assert publish_canonical_json(output, {"decision": "PASS"}) is True
    with pytest.raises(PromotionAuditorError, match="conflicts"):
        publish_canonical_json(output, {"decision": "FAIL"})


def test_cli_exposes_strict_watch_without_decision_inputs(tmp_path):
    args = parse_args(
        [
            "--runtime-config",
            str(tmp_path / "runtime.json"),
            "--katago",
            str(tmp_path / "katago"),
            "--mode",
            "watch",
        ]
    )
    assert args.mode == "watch"
    assert not hasattr(args, "decision")
    assert not hasattr(args, "pass_value")
