import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.curate_position_bank import (
    generate_consensus_query_bundle,
    publish_normalized,
)
from risk_score.curation_orchestrator import (
    STATUS_CONTRACT,
    CurationContradiction,
    CurationOrchestrator,
)
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256

DETERMINISTIC_CONFIG = """\
forDeterministicTesting = true
numAnalysisThreads = 1
nnRandomize = false
rootNoiseEnabled = false
rootNumSymmetriesToSample = 1
useUncertainty = false
cpuctUtilityStdevScale = 0
reportAnalysisWinratesAs = SIDETOMOVE
"""


def write_jsonl(path, rows):
    path.write_bytes(
        "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    )


def position(turn):
    return {
        "xSize": 5,
        "ySize": 5,
        "board": "X..../..O../...../...X./.....",
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": turn,
        "hintLoc": "null",
    }


def consensus_bundle(root, *, position_count=4):
    root.mkdir()
    katago = root / "katago"
    config = root / "analysis.cfg"
    original = root / "original.bin.gz"
    champion = root / "champion.bin.gz"
    policy = root / "policy.json"
    source = root / "source.jsonl"
    normalized = root / "normalized.jsonl"
    katago.write_bytes(b"fake-katago")
    config.write_text(DETERMINISTIC_CONFIG, encoding="utf-8")
    original.write_bytes(b"immutable-original")
    champion.write_bytes(b"frozen-champion")
    policy_value = json.loads(
        (
            Path(__file__).parents[1] / "risk_score" / "promotion_policy_v3.json"
        ).read_text(encoding="utf-8")
    )
    policy.write_text(canonical_json(policy_value) + "\n", encoding="utf-8")
    write_jsonl(source, [position(index) for index in range(position_count)])
    publish_normalized([source], normalized, root / "normalized-manifest.json")
    query_dir = root / "queries"
    manifest = generate_consensus_query_bundle(
        normalized,
        query_dir,
        katago,
        config,
        original,
        champion,
        policy,
    )
    return {
        "manifest": query_dir / "manifest.json",
        "manifest_value": manifest,
        "katago": katago,
        "config": config,
        "original": original,
        "champion": champion,
    }


class FakeKataGo:
    def __init__(self, *, fail=None, delay=0.0, before_response=None):
        self.fail = fail or (lambda call: False)
        self.delay = delay
        self.before_response = before_response
        self.calls = []
        self.active = Counter()
        self.maximum_active = Counter()
        self.lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        call = {
            "argv": tuple(argv),
            "gpu": kwargs["env"]["CUDA_VISIBLE_DEVICES"],
            "query_path": str(Path(kwargs["stdin"].name).resolve()),
            "shell": kwargs["shell"],
        }
        call["model_path"] = call["argv"][call["argv"].index("-model") + 1]
        with self.lock:
            self.calls.append(call)
            self.active[call["gpu"]] += 1
            self.maximum_active[call["gpu"]] = max(
                self.maximum_active[call["gpu"]],
                self.active[call["gpu"]],
            )
        try:
            if self.before_response is not None:
                self.before_response(call)
            if self.delay:
                time.sleep(self.delay)
            if self.fail(call):
                return SimpleNamespace(returncode=19, stderr=b"injected failure")
            queries = [
                json.loads(line)
                for line in kwargs["stdin"].read().decode("utf-8").splitlines()
            ]
            kwargs["stdout"].write(
                "".join(
                    canonical_json({"id": row["id"], "complete": True}) + "\n"
                    for row in queries
                ).encode("utf-8")
            )
            return SimpleNamespace(returncode=0, stderr=b"")
        finally:
            with self.lock:
                self.active[call["gpu"]] -= 1


def orchestrator(fixture, work, runner, **overrides):
    arguments = {
        "query_manifest_path": fixture["manifest"],
        "work_dir": work,
        "shard_count": 2,
        "gpus": ["2", "5"],
        "per_gpu_parallelism": 1,
        "subprocess_runner": runner,
    }
    arguments.update(overrides)
    return CurationOrchestrator(**arguments)


def analysis_outputs(work):
    return sorted((work / "roles").glob("*/*/analysis-shards/shard-*.jsonl"))


def test_plan_materializes_canonical_restart_status(tmp_path):
    fixture = consensus_bundle(tmp_path / "bundle")
    events = []

    def should_not_run(*args, **kwargs):
        raise AssertionError("plan mode launched analysis")

    engine = orchestrator(
        fixture,
        tmp_path / "work",
        should_not_run,
        progress_callback=events.append,
    )
    status = engine.plan()
    status_path = tmp_path / "work" / "status.json"
    stored = json.loads(status_path.read_text(encoding="utf-8"))
    payload = dict(stored)
    status_hash = payload.pop("status_sha256")

    assert status["contract"] == STATUS_CONTRACT
    assert status["state"] == "planned"
    assert len(status["roles"]) == 8
    assert status["progress"] == {
        "completed_shards": 0,
        "total_shards": 16,
        "running_shards": 0,
        "failed_shards": 0,
        "remaining_shards": 16,
        "completed_rows": 0,
        "total_rows": status["progress"]["total_rows"],
        "completion_fraction": 0.0,
        "elapsed_seconds": status["progress"]["elapsed_seconds"],
        "rows_per_second": None,
        "eta_seconds": None,
    }
    assert status_path.read_bytes() == (canonical_json(stored) + "\n").encode("utf-8")
    assert status_hash == canonical_sha256(payload)
    assert events[-1]["event"] == "planned"
    assert not analysis_outputs(tmp_path / "work")
    assert all(
        Path(role["split_manifest_path"]).is_file() for role in status["roles"].values()
    )


def test_schedules_models_on_explicit_bounded_gpus_and_merges(tmp_path):
    fixture = consensus_bundle(tmp_path / "bundle")
    runner = FakeKataGo(delay=0.01)
    events = []
    status = orchestrator(
        fixture,
        tmp_path / "work",
        runner,
        progress_callback=events.append,
    ).once()

    assert status["state"] == "complete"
    assert status["ready_for_labeling"] is True
    assert status["progress"]["completed_shards"] == 16
    assert len(status["analysis_outputs"]) == 8
    assert len(runner.calls) == 16
    gpu_counts = Counter(call["gpu"] for call in runner.calls)
    assert sum(gpu_counts.values()) == 16
    assert set(gpu_counts) == {"2", "5"}
    assert runner.maximum_active == {"2": 1, "5": 1}
    assert all(call["shell"] is False for call in runner.calls)
    assert all(call["argv"][1] == "analysis" for call in runner.calls)
    model_counts = Counter(call["model_path"] for call in runner.calls)
    assert model_counts == {
        str(fixture["original"].resolve()): 8,
        str(fixture["champion"].resolve()): 8,
    }
    for role, role_status in status["roles"].items():
        expected_model = fixture[role.split("/")[0]]
        assert role_status["model_path"] == str(expected_model.resolve())
        assert role_status["merged"]["row_count"] > 0
        assert Path(role_status["merged"]["manifest_path"]).is_file()
    assert events[-1]["event"] == "complete"
    assert {
        "completed_rows",
        "total_rows",
        "rows_per_second",
        "eta_seconds",
    } <= events[-1].keys()


def test_fast_gpu_steals_from_global_pending_queue(tmp_path):
    fixture = consensus_bundle(tmp_path / "bundle")
    work = tmp_path / "work"
    release_slow_gpu = threading.Event()
    coordination_lock = threading.Lock()
    initial_status = {}
    fast_started = 0
    total_shards = 16

    def coordinate(call):
        nonlocal fast_started
        with coordination_lock:
            if not initial_status:
                initial_status.update(
                    json.loads((work / "status.json").read_text(encoding="utf-8"))
                )
            initially_assigned = sum(
                shard["gpu"] is not None
                for role in initial_status["roles"].values()
                for shard in role["shards"]
            )
            dynamic_scheduler = initially_assigned == 2
            if call["gpu"] == "2":
                fast_started += 1
                if fast_started == total_shards - 1:
                    release_slow_gpu.set()
        if call["gpu"] == "5" and dynamic_scheduler:
            assert release_slow_gpu.wait(timeout=10), "fast GPU stopped claiming work"

    runner = FakeKataGo(before_response=coordinate)
    status = orchestrator(fixture, work, runner).once()

    assert status["state"] == "complete"
    assert Counter(call["gpu"] for call in runner.calls) == {"2": 15, "5": 1}
    query_executions = Counter(call["query_path"] for call in runner.calls)
    assert len(query_executions) == total_shards
    assert set(query_executions.values()) == {1}
    assert runner.maximum_active == {"2": 1, "5": 1}

    initial_shards = [
        shard
        for role in initial_status["roles"].values()
        for shard in role["shards"]
    ]
    assert Counter(shard["state"] for shard in initial_shards) == {
        "running": 2,
        "pending": 14,
    }
    initial_assignments = {
        shard["query_path"]: shard["gpu"]
        for shard in initial_shards
        if shard["gpu"] is not None
    }
    assert Counter(initial_assignments.values()) == {"2": 1, "5": 1}

    actual_assignments = {
        call["query_path"]: call["gpu"] for call in runner.calls
    }
    assert set(initial_assignments.items()) <= set(actual_assignments.items())
    assert all(
        shard["gpu"] == actual_assignments[shard["query_path"]]
        for role in status["roles"].values()
        for shard in role["shards"]
    )


def test_failure_then_resume_runs_only_missing_shards(tmp_path):
    fixture = consensus_bundle(tmp_path / "bundle")
    first_runner = FakeKataGo(
        fail=lambda call: Path(call["query_path"]).name == "shard-001.jsonl"
    )
    work = tmp_path / "work"
    first = orchestrator(fixture, work, first_runner).once()

    assert first["state"] == "failed"
    assert first["progress"]["completed_shards"] == 8
    assert first["progress"]["failed_shards"] == 8
    assert len(first_runner.calls) == 16
    assert set(
        Counter(call["query_path"] for call in first_runner.calls).values()
    ) == {1}
    completed_before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in analysis_outputs(work)
    }
    assert len(completed_before) == 8

    resume_barrier = threading.Barrier(2)
    resumed_runner = FakeKataGo(
        before_response=lambda call: resume_barrier.wait(timeout=5)
    )
    resumed = orchestrator(
        fixture,
        work,
        resumed_runner,
        gpus=["7"],
        per_gpu_parallelism=2,
    ).resume(poll_interval=0, max_passes=1)

    assert resumed["state"] == "complete"
    assert resumed["mode"] == "resume"
    assert len(resumed_runner.calls) == 8
    assert {call["gpu"] for call in resumed_runner.calls} == {"7"}
    assert set(
        Counter(call["query_path"] for call in resumed_runner.calls).values()
    ) == {1}
    assert resumed_runner.maximum_active == {"7": 2}
    for path, (data, modified) in completed_before.items():
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == modified


def test_rejects_contradictory_sibling_manifest_without_running(tmp_path):
    fixture = consensus_bundle(tmp_path / "bundle")
    work = tmp_path / "work"
    assert (
        orchestrator(fixture, work, FakeKataGo(), shard_count=1).once()["state"]
        == "complete"
    )
    output = analysis_outputs(work)[0]
    manifest_path = Path(str(output) + ".manifest.json")
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["query_path"] = str((tmp_path / "other-query.jsonl").resolve())
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    runner = FakeKataGo()

    with pytest.raises(CurationContradiction, match="artifact contradiction"):
        orchestrator(fixture, work, runner, shard_count=1).once()
    assert runner.calls == []


def test_invalid_orphan_is_quarantined_and_failure_is_persisted(tmp_path):
    fixture = consensus_bundle(tmp_path / "bundle")
    work = tmp_path / "work"
    engine = orchestrator(fixture, work, FakeKataGo(), shard_count=1)
    plan = engine.plan()
    first_role = sorted(plan["roles"])[0]
    output = Path(plan["roles"][first_role]["shards"][0]["output_path"])
    output.write_bytes(b"not-jsonl")
    failing = FakeKataGo(fail=lambda call: True)

    failed = orchestrator(fixture, work, failing, shard_count=1, gpus=["3"]).once()

    assert failed["state"] == "failed"
    assert failed["progress"]["failed_shards"] == 8
    assert failed["progress"]["completed_shards"] == 0
    orphan_files = list((work / "orphaned" / Path(first_role)).iterdir())
    assert len(orphan_files) == 1
    assert orphan_files[0].read_bytes() == b"not-jsonl"
    assert orphan_files[0].name == (
        f"{output.name}.{file_sha256(orphan_files[0])}.orphan"
    )
    stored = json.loads((work / "status.json").read_text(encoding="utf-8"))
    assert (work / "status.json").read_text(encoding="utf-8") == (
        canonical_json(stored) + "\n"
    )
    assert all(call["gpu"] == "3" for call in failing.calls)
