import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.consensus_prefilter import (
    PREFILTER_ROLES,
    prefilter_consensus_sources,
)
from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    _normalized_positions,
    policy_pool_minima,
    publish_normalized,
)
from risk_score.curation_pipeline import (
    FileBinding as PipelineFileBinding,
    GpuComputeProcess,
    GpuOccupancy,
    OwnershipProbes,
    PipelineContradiction,
    _verify_supplement_summary,
)
from risk_score.curation_supplement import (
    PRIMARY_INVENTORY_CONTRACT,
    SPEC_CONTRACT,
    STATUS_CONTRACT,
    CurationSupplement,
    SupplementContradiction,
    SupplementError,
    SupplementSpecError,
    load_supplement_spec,
    main,
    plan_analysis_commands,
    plan_analysis_jobs,
    plan_selfplay_command,
)
from risk_score.gpu_lease import ProcessIdentity
from risk_score.position_samples import (
    build_analysis_query,
    canonical_json,
    canonical_sha256,
    file_sha256,
)

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
SELFPLAY_CONFIG = """\
numGameThreads = 2
numNNServerThreadsPerModel = 2
cudaDeviceToUseModel0Thread0 = 0
cudaDeviceToUseModel0Thread1 = 1
"""
EXECUTED_MODULES = (
    "board_symmetry.py",
    "build_live_runtime.py",
    "consensus_prefilter.py",
    "curate_position_bank.py",
    "curation_orchestrator.py",
    "curation_pipeline.py",
    "curation_supplement.py",
    "gpu_lease.py",
    "position_samples.py",
)


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def directory_binding(path):
    rows = [
        {
            "path": child.name,
            "size": child.stat().st_size,
            "sha256": file_sha256(child),
        }
        for child in sorted(path.iterdir())
    ]
    return {"path": str(path.resolve()), "sha256": canonical_sha256(rows)}


def position(turn, stone_count):
    cells = ["."] * 25
    for index in range(stone_count):
        cells[index] = "X"
    cells[-1] = "O"
    board = "/".join("".join(cells[offset : offset + 5]) for offset in range(0, 25, 5))
    return {
        "xSize": 5,
        "ySize": 5,
        "board": board,
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": turn,
        "hintLoc": "null",
        "metadata": "supplement-test",
    }


def analysis_record(record_id, score):
    return {
        "id": record_id,
        "rootInfo": {"scoreLead": score, "winrate": 0.75, "visits": 2000},
        "moveInfos": [
            {
                "move": "D4",
                "order": 0,
                "visits": 2000,
                "prior": 0.2,
                "scoreLead": score,
                "scoreSelfplay": score,
                "scoreStdev": 8.0,
            }
        ],
    }


def proc_hash(argv):
    data = b"\0".join(item.encode() for item in argv) + b"\0"
    return hashlib.sha256(data).hexdigest()


def make_tiny_policy(source, output):
    value = json.loads(source.read_text())
    stages = value["evaluation_stages"]
    for key in ("lead_40_color_pairs", "lead_80_color_pairs"):
        stages["stage_2_finalist_selection"][key] = 1
        stages["deep_audit"][key] = 0
        for look in stages["stage_3_promotion_confirmation"]["looks"]:
            look[key] = 0
    write_canonical(output, value)
    assert policy_pool_minima(value)["lead-40"] == 1
    assert policy_pool_minima(value)["lead-80"] == 1


def frozen_assets(tmp_path):
    run_root = tmp_path / "run"
    training = tmp_path / "training-selfplay"
    models = tmp_path / "models"
    repository = tmp_path / "deployment"
    modules = repository / "python" / "risk_score"
    for path in (run_root, training, models, modules):
        path.mkdir(parents=True)
    source_modules = Path(__file__).parents[1] / "risk_score"
    files = {}
    for module_name in EXECUTED_MODULES:
        target = modules / module_name
        target.write_bytes((source_modules / module_name).read_bytes())
        files[f"module:{module_name}"] = binding(target)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "frozen deployment",
        ],
        check=True,
        env=environment,
    )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    external_deployment_artifact = tmp_path / "external-deployment-artifact.bin"
    external_deployment_artifact.write_bytes(b"bound-external-deployment-artifact")
    files["external:deployment-artifact"] = binding(
        external_deployment_artifact
    )
    deployment = tmp_path / "deployment-manifest.json"
    deployment_value = {
        "schema_version": 1,
        "contract": "risk-score-live-runtime-deployment-v1",
        "source_revision": revision,
        "source_sha256": hashlib.sha256(revision.encode()).hexdigest(),
        "files": files,
    }
    deployment_value["manifest_sha256"] = canonical_sha256(deployment_value)
    write_canonical(deployment, deployment_value)

    katago = tmp_path / "katago"
    analysis_config = tmp_path / "analysis.cfg"
    selfplay_config = tmp_path / "selfplay.cfg"
    original = models / "model.bin.gz"
    champion = tmp_path / "champion.bin.gz"
    policy = tmp_path / "policy.json"
    katago.write_bytes(b"fake-katago")
    analysis_config.write_text(DETERMINISTIC_CONFIG, encoding="utf-8")
    selfplay_config.write_text(SELFPLAY_CONFIG, encoding="utf-8")
    original.write_bytes(b"immutable-original")
    champion.write_bytes(b"frozen-champion")
    make_tiny_policy(
        Path(__file__).parents[1] / "risk_score" / "promotion_policy_v3.json",
        policy,
    )
    return {
        "run_root": run_root.resolve(),
        "training": training.resolve(),
        "repository": repository.resolve(),
        "revision": revision,
        "deployment": deployment.resolve(),
        "katago": katago.resolve(),
        "analysis_config": analysis_config.resolve(),
        "selfplay_config": selfplay_config.resolve(),
        "models": models.resolve(),
        "original": original.resolve(),
        "champion": champion.resolve(),
        "policy": policy.resolve(),
        "external_deployment_artifact": external_deployment_artifact.resolve(),
    }


def primary_prefilter(root, assets, *, label, turn, stone_count):
    root.mkdir()
    raw = root / "raw.jsonl"
    normalized = root / "normalized.jsonl"
    write_jsonl(raw, [position(turn, stone_count)])
    publish_normalized([raw], normalized, root / "normalized.manifest.json")
    normalized_rows = _normalized_positions(normalized)
    record_id = normalized_rows[0]["semanticSha256"]
    score = 60.0 if label == "lead-40" else 95.0
    analyses = {}
    model_hashes = {
        "original": file_sha256(assets["original"]),
        "champion": file_sha256(assets["champion"]),
    }
    for role in PREFILTER_ROLES:
        model = role.split("/", 1)[0]
        safe = role.replace("/", "-")
        query = root / f"{safe}.query.jsonl"
        output = root / f"{safe}.analysis.jsonl"
        write_jsonl(
            query,
            [
                build_analysis_query(
                    normalized_rows[0],
                    query_id=record_id,
                    max_visits=2000,
                    powered=role.endswith("/powered-2000"),
                )
            ],
        )
        write_jsonl(output, [analysis_record(record_id, score)])
        argv = [
            str(assets["katago"]),
            "analysis",
            "-config",
            str(assets["analysis_config"]),
            "-model",
            str(assets[model]),
        ]
        execution = {
            "schema_version": 1,
            "contract": ANALYSIS_RUN_CONTRACT,
            "argv": argv,
            "katago_sha256": file_sha256(assets["katago"]),
            "config_sha256": file_sha256(assets["analysis_config"]),
            "model_sha256": model_hashes[model],
            "cuda_visible_devices": "0",
            "query_path": str(query.resolve()),
            "query_sha256": file_sha256(query),
            "output_path": str(output.resolve()),
            "output_sha256": file_sha256(output),
            "row_count": 1,
        }
        execution["manifest_sha256"] = canonical_sha256(execution)
        write_canonical(Path(str(output) + ".manifest.json"), execution)
        analyses[role] = output
    selected = root / "selected.jsonl"
    manifest = root / "prefilter.manifest.json"
    prefilter_consensus_sources(
        normalized_path=normalized,
        analysis_paths=analyses,
        label=label,
        output_path=selected,
        manifest_path=manifest,
        limit=None,
    )
    return binding(manifest)


def publish_spec(
    tmp_path,
    *,
    assets=None,
    primary=None,
    primary_per_label=1,
    targets=None,
    shards=1,
    parallelism=1,
    game_count=3,
    round_number=1,
    prior_summaries=(),
    downstream_counts=None,
    spec_name="supplement-spec.json",
    work_name="lead-supplement",
    override_args=(),
    harvest=None,
):
    assets = frozen_assets(tmp_path) if assets is None else assets
    if primary is None:
        primary = []
        for label_index, label in enumerate(("lead-40", "lead-80")):
            for index in range(primary_per_label):
                primary.append(
                    primary_prefilter(
                        tmp_path / f"primary-{label_index}-{index}",
                        assets,
                        label=label,
                        turn=10 + label_index * 100 + index,
                        stone_count=1 + label_index * 5 + index,
                    )
                )
        primary.sort(key=lambda item: item["path"])
    inventory_path = tmp_path / f"{spec_name}.primary-inventory.json"
    inventory = {
        "schema_version": 1,
        "contract": PRIMARY_INVENTORY_CONTRACT,
        "manifests": primary,
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    write_canonical(inventory_path, inventory)
    revision = assets["revision"]
    value = {
        "schema_version": 1,
        "contract": SPEC_CONTRACT,
        "deployment": {
            "repository_path": str(assets["repository"]),
            "source_revision": revision,
            "source_sha256": hashlib.sha256(revision.encode()).hexdigest(),
        },
        "deployment_manifest": binding(assets["deployment"]),
        "run_root": str(assets["run_root"]),
        "training_input_root": str(assets["training"]),
        "work_root": str(
            (assets["run_root"] / "evaluation" / "curation" / work_name).resolve()
        ),
        "katago": binding(assets["katago"]),
        "analysis_config": binding(assets["analysis_config"]),
        "selfplay_config": binding(assets["selfplay_config"]),
        "selfplay_models_directory": directory_binding(assets["models"]),
        "selfplay_override_args": [list(pair) for pair in override_args],
        "policy": binding(assets["policy"]),
        "models": {
            "original": binding(assets["original"]),
            "champion": binding(assets["champion"]),
        },
        "game_count": game_count,
        "topology": {
            "shards_per_role": shards,
            "gpus": ["2", "5"],
            "selfplay_gpus": ["0", "1"],
            "per_gpu_parallelism": parallelism,
        },
        "consensus_reserve_fraction": 1.0,
        "target_counts": targets or {"lead-40": 2, "lead-80": 2},
        "primary_prefilter_inventory": binding(inventory_path),
        "primary_prefilter_manifests": primary,
        "round": round_number,
        "prior_round_summaries": list(prior_summaries),
        "downstream_accepted_counts": downstream_counts,
    }
    if harvest is not None:
        value["harvest"] = dict(harvest)
    value["spec_sha256"] = canonical_sha256(value)
    spec_path = tmp_path / spec_name
    write_canonical(spec_path, value)
    return spec_path, assets, primary


class FakeProcesses:
    def __init__(
        self,
        *,
        fail_one_analysis=False,
        zero_candidates=False,
        malformed_sgf=False,
        duplicate_primary=False,
        interrupt_selfplay=False,
    ):
        self.fail_one_analysis = fail_one_analysis
        self.zero_candidates = zero_candidates
        self.malformed_sgf = malformed_sgf
        self.duplicate_primary = duplicate_primary
        self.interrupt_selfplay = interrupt_selfplay
        self.calls = []
        self.lock = threading.Lock()
        self.next_pid = 900_000

    def launcher(self, argv, **kwargs):
        argv = tuple(argv)
        self.calls.append({"command": argv[1], "argv": argv, "gpu": None})
        self.next_pid += 1
        return FakeHandle(self, argv, self.next_pid)

    def __call__(self, argv, **kwargs):
        argv = tuple(argv)
        command = argv[1]
        with self.lock:
            self.calls.append(
                {
                    "command": command,
                    "argv": argv,
                    "gpu": kwargs.get("env", {}).get("CUDA_VISIBLE_DEVICES"),
                }
            )
            should_fail = command == "analysis" and self.fail_one_analysis
            if should_fail:
                self.fail_one_analysis = False
        if command == "samplesgfs":
            output = Path(argv[argv.index("-outdir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            rows = [
                position(100, 3),
                position(101, 4),
                position(200, 7),
                position(201, 8),
            ]
            if self.duplicate_primary:
                rows[0] = position(10, 1)
            write_jsonl(output / "positions.startposes.txt", rows)
            return SimpleNamespace(returncode=0)
        if command == "analysis":
            if should_fail:
                return SimpleNamespace(returncode=17, stderr=b"injected")
            query_path = Path(kwargs["stdin"].name).resolve()
            queries = [
                json.loads(line)
                for line in kwargs["stdin"].read().decode("utf-8").splitlines()
            ]
            normalized = _normalized_positions(
                query_path.parents[2] / "normalized.jsonl"
            )
            turns = {
                row["semanticSha256"]: row["initialTurnNumber"] for row in normalized
            }
            responses = [
                analysis_record(
                    row["id"],
                    (
                        0.0
                        if self.zero_candidates
                        else (60.0 if turns[row["id"]] < 200 else 95.0)
                    ),
                )
                for row in queries
            ]
            kwargs["stdout"].write(
                "".join(canonical_json(row) + "\n" for row in responses).encode()
            )
            return SimpleNamespace(returncode=0, stderr=b"")
        raise AssertionError(f"unexpected fake command: {argv}")

    def finish_selfplay(self, argv):
        output = Path(argv[argv.index("-output-dir") + 1])
        games = int(argv[argv.index("-max-games-total") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if self.malformed_sgf:
            games_text = "(;GM[1]\n"
        else:
            games_text = "".join(
                f"(;GM[1]FF[4]SZ[19]RE[B+R]C[game-{index}])\n" for index in range(games)
            )
        (output / "games.sgfs").write_text(games_text, encoding="utf-8")
        if self.interrupt_selfplay:
            self.interrupt_selfplay = False
            raise KeyboardInterrupt
        return 0

    def count(self, command):
        return sum(call["command"] == command for call in self.calls)


class FakeHandle:
    def __init__(self, processes, argv, pid):
        self.processes = processes
        self.argv = argv
        self.pid = pid

    def wait(self):
        return self.processes.finish_selfplay(self.argv)


class ProbeFactory:
    def __init__(
        self,
        tmp_path,
        processes,
        *,
        conflict=False,
        safe=True,
        lease_safe=None,
        target_inactive=None,
    ):
        self.processes = processes
        self.owner = ProcessIdentity(
            os.getpid(),
            100,
            os.getpgrp(),
            "boot-a",
            "a" * 64,
            "/test",
        )
        self.conflict = conflict
        self.lease_safe = safe if lease_safe is None else lease_safe
        self.target_inactive = safe if target_inactive is None else target_inactive
        self.lock_path = tmp_path / "global-gpu.lock"

    def process_identity(self, pid):
        for call in reversed(self.processes.calls):
            if call["command"] == "selfplay":
                return ProcessIdentity(
                    pid,
                    pid,
                    self.owner.process_group_id,
                    "boot-a",
                    proc_hash(call["argv"]),
                    "/test",
                )
        return ProcessIdentity(pid, pid, 999, "boot-a", "f" * 64, "/foreign")

    def probes(self):
        occupancy = GpuOccupancy(
            {
                "0": "GPU-0",
                "1": "GPU-1",
                "2": "GPU-2",
                "5": "GPU-5",
                "7": "GPU-7",
            },
            (
                (GpuComputeProcess("GPU-0", 777_777, "foreign"),)
                if self.conflict
                else ()
            ),
        )
        return OwnershipProbes(
            gpu_occupancy=lambda: occupancy,
            current_process=lambda: self.owner,
            process_identity=self.process_identity,
            gpu7_lease_safe=lambda _: self.lease_safe,
            production_target_inactive=lambda: self.target_inactive,
            sleep=lambda _: None,
            global_lock_path=self.lock_path,
        )


def coordinator(spec_path, processes, probes=None):
    probe_factory = (
        ProbeFactory(spec_path.parent, processes) if probes is None else probes
    )
    return CurationSupplement(
        spec_path,
        process_runner=processes,
        process_launcher=processes.launcher,
        ownership_probes=probe_factory.probes(),
    )


def test_production_command_and_gpu_topology_are_exact(tmp_path):
    spec_path, assets, _ = publish_spec(
        tmp_path,
        shards=2,
        parallelism=24,
        game_count=7,
        override_args=(("-override-config", "maxMovesPerGame=1000"),),
    )
    spec = load_supplement_spec(spec_path)
    assert spec.topology.per_gpu_parallelism == 24
    command = plan_selfplay_command(spec)
    assert command["argv"] == [
        str(assets["katago"]),
        "selfplay",
        "-max-games-total",
        "7",
        "-output-dir",
        str(spec.work_root / "selfplay-attempt" / "working"),
        "-models-dir",
        str(assets["models"]),
        "-config",
        str(assets["selfplay_config"]),
        "-override-config",
        "maxMovesPerGame=1000",
    ]
    assert command["selfplay_gpus"] == ["0", "1"]
    jobs = plan_analysis_jobs(spec)
    assert [job.gpu for job in jobs] == ["2", "5"] * 4
    assert plan_analysis_commands(spec)[0]["environment"] == {
        "CUDA_VISIBLE_DEVICES": "2"
    }


def test_targeted_handicap_harvest_is_spec_bound(tmp_path):
    spec_path, _, _ = publish_spec(
        tmp_path,
        harvest={
            "max_handicap": 10,
            "min_turn_number_board_area_prop": 0.05,
            "max_turn_number_board_area_prop": 0.75,
        },
    )
    engine = coordinator(spec_path, FakeProcesses())

    assert engine.spec.harvest.max_handicap == 10
    assert engine.spec.harvest.max_turn_number_board_area_prop == 0.75
    assert engine.once()["next_stage"] == "create_harvest_plan"
    assert engine.once()["next_stage"] == "execute_harvest"

    plan = json.loads(engine.layout.harvest_plan.read_text())
    argv = plan["argv"]
    assert argv[argv.index("-max-handicap") + 1] == "10"
    assert argv[argv.index("-min-turn-number-board-area-prop") + 1] == "0.05"
    assert argv[argv.index("-max-turn-number-board-area-prop") + 1] == "0.75"


def test_primary_prefilter_validation_is_cached_and_invalidated(
    tmp_path, monkeypatch
):
    from risk_score import curation_supplement as supplement_module

    spec_path, _, primary = publish_spec(tmp_path)
    engine = coordinator(spec_path, FakeProcesses())
    original = supplement_module.validate_prefilter_artifact
    original_hash = supplement_module.file_sha256
    calls = []
    hash_calls = []

    def validate(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    def hash_file(path):
        hash_calls.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(supplement_module, "validate_prefilter_artifact", validate)
    monkeypatch.setattr(supplement_module, "file_sha256", hash_file)

    engine._assert_frozen_inputs()
    first_count = len(calls)
    first_hash_count = len(hash_calls)
    engine._assert_frozen_inputs()
    assert first_count == len(primary)
    assert len(calls) == first_count
    assert first_hash_count > 0
    assert len(hash_calls) == first_hash_count

    manifest = json.loads(Path(primary[0]["path"]).read_text(encoding="utf-8"))
    analysis = Path(manifest["analyses"][PREFILTER_ROLES[0]]["path"])
    analysis.write_bytes(analysis.read_bytes())
    engine._assert_frozen_inputs()
    assert len(calls) == first_count * 2
    assert len(hash_calls) > first_hash_count

    analysis.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        SupplementContradiction,
        match="frozen input changed|primary prefilter recomputation failed",
    ):
        engine._assert_frozen_inputs()


def test_frozen_input_cache_tracks_transitive_deployment_files(tmp_path):
    spec_path, assets, _ = publish_spec(tmp_path)
    engine = coordinator(spec_path, FakeProcesses())
    engine._assert_frozen_inputs()
    engine._assert_frozen_inputs()

    artifact = assets["external_deployment_artifact"]
    artifact.write_bytes(b"x" * artifact.stat().st_size)

    with pytest.raises(SupplementContradiction):
        engine._assert_frozen_inputs()


def test_noop_is_read_only_until_canonical_summary(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path, primary_per_label=2)
    processes = FakeProcesses()
    engine = coordinator(spec_path, processes)
    assert engine.status()["deficits"] == {}
    assert not engine.spec.work_root.exists()
    completed = engine.once()
    replayed = engine.once()
    assert completed["state"] == replayed["state"] == "noop"
    assert processes.calls == []
    assert json.loads(engine.layout.summary.read_text())["selfplay"] is None


def test_attrition_reserve_limits_and_downstream_summary_validate(tmp_path):
    spec_path, assets, _ = publish_spec(tmp_path)
    processes = FakeProcesses()
    engine = coordinator(spec_path, processes)
    result = engine.watch(poll_interval=0)
    assert result["state"] == "complete"
    assert result["deficits"] == {"lead-40": 1, "lead-80": 1}
    assert processes.count("selfplay") == 1
    for label in ("lead-40", "lead-80"):
        manifest = json.loads(
            (engine.layout.selected / f"{label}.manifest.json").read_text()
        )
        assert manifest["limit"] == 2
        assert manifest["selected"]["row_count"] == 2
    summary = json.loads(engine.layout.summary.read_text())
    assert summary["generation_limits"] == {"lead-40": 2, "lead-80": 2}
    assert not engine._gpu_claim_path.exists()
    assert json.loads(engine.layout.gpu_ownership_archive.read_text())["state"] == (
        "released"
    )
    source = SimpleNamespace(
        name="supplement",
        label="lead-40",
        selected=PipelineFileBinding(
            engine.layout.selected / "lead-40.jsonl",
            file_sha256(engine.layout.selected / "lead-40.jsonl"),
        ),
        prefilter_manifest=PipelineFileBinding(
            engine.layout.selected / "lead-40.manifest.json",
            file_sha256(engine.layout.selected / "lead-40.manifest.json"),
        ),
        supplement_summary=PipelineFileBinding(
            engine.layout.summary, file_sha256(engine.layout.summary)
        ),
    )
    pipeline_spec = SimpleNamespace(
        run_root=assets["run_root"],
        katago=SimpleNamespace(
            path=assets["katago"], sha256=file_sha256(assets["katago"])
        ),
        analysis_config=SimpleNamespace(
            path=assets["analysis_config"],
            sha256=file_sha256(assets["analysis_config"]),
        ),
        policy=SimpleNamespace(
            path=assets["policy"], sha256=file_sha256(assets["policy"])
        ),
        models={
            role: SimpleNamespace(path=assets[role], sha256=file_sha256(assets[role]))
            for role in ("original", "champion")
        },
    )
    _verify_supplement_summary(pipeline_spec, source)


def test_hard_kill_attempt_is_quarantined_and_resumed(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    interrupted_processes = FakeProcesses(interrupt_selfplay=True)
    first = coordinator(spec_path, interrupted_processes)
    with pytest.raises(KeyboardInterrupt):
        first.once()
    assert first.layout.selfplay_attempt_output.exists()
    resumed_processes = FakeProcesses()
    resumed = coordinator(spec_path, resumed_processes)
    completed = resumed.once()
    assert completed["next_stage"] == "create_harvest_plan"
    assert resumed_processes.count("selfplay") == 1
    assert (
        resumed.layout.selfplay_orphans / "generation-000001" / "games.sgfs"
    ).is_file()


def test_analysis_failure_resumes_only_missing_shard(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    first_processes = FakeProcesses(fail_one_analysis=True)
    first = coordinator(spec_path, first_processes)
    with pytest.raises(SupplementError, match="analysis shards failed"):
        first.watch(poll_interval=0)
    complete_before = sorted(first.layout.analysis_shards.rglob("shard-*.jsonl"))
    assert len(complete_before) == 3
    preserved = {path: path.read_bytes() for path in complete_before}
    resumed_processes = FakeProcesses()
    resumed = coordinator(spec_path, resumed_processes)
    assert resumed.watch(poll_interval=0)["state"] == "complete"
    assert resumed_processes.count("analysis") == 1
    assert all(path.read_bytes() == data for path, data in preserved.items())


def test_gpu_conflict_and_host_safety_fail_closed(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    processes = FakeProcesses()
    conflict = ProbeFactory(tmp_path, processes, conflict=True)
    with pytest.raises(PipelineContradiction, match="foreign compute process"):
        coordinator(spec_path, processes, conflict).once()
    unsafe = ProbeFactory(tmp_path, processes, safe=False)
    with pytest.raises(PipelineContradiction, match="lease state"):
        coordinator(spec_path, processes, unsafe).once()
    target_active = ProbeFactory(
        tmp_path, processes, lease_safe=True, target_inactive=False
    )
    with pytest.raises(PipelineContradiction, match="systemd target"):
        coordinator(spec_path, processes, target_active).once()


def test_old_boot_gpu_claim_is_recovered(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    processes = FakeProcesses()
    engine = coordinator(spec_path, processes)
    engine.once()
    claim = json.loads(engine._gpu_claim_path.read_text())
    claim["owner"]["bootId"] = "old-boot"
    claim.pop("claim_sha256")
    claim["claim_sha256"] = canonical_sha256(claim)
    write_canonical(engine._gpu_claim_path, claim)
    resumed = coordinator(spec_path, FakeProcesses())
    with resumed.gpu_ownership.global_lock():
        recovered = resumed.gpu_ownership.acquire(poll_interval=0)
    assert recovered["owner"]["bootId"] == "boot-a"
    assert recovered["recovered_from_claim_sha256"] == claim["claim_sha256"]


def test_dead_owned_harvest_and_analysis_temporaries_are_reclaimed(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    processes = FakeProcesses()
    engine = coordinator(spec_path, processes)
    engine.once()
    engine.once()
    dead = ProcessIdentity(888_888, 1, 888_888, "boot-a", "d" * 64, "/dead")
    harvest_temp = engine.layout.harvest_directory.parent / ".harvested.harvest-dead"
    harvest_temp.mkdir()
    write_canonical(
        engine._stage_attempt_path("harvest"),
        engine._stage_attempt_value(
            key="harvest", state="running", generation=1, owner=dead
        ),
    )
    engine.once()
    assert not harvest_temp.exists()
    while engine.status()["next_stage"] != "run_analyses":
        engine.once()
    job = plan_analysis_jobs(engine.spec)[0]
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_temp = job.output_path.parent / f".{job.output_path.name}.analysis-dead"
    analysis_temp.write_bytes(b"partial")
    key = f"analysis-{job.model}-{job.mode}-{job.shard_index:03d}"
    write_canonical(
        engine._stage_attempt_path(key),
        engine._stage_attempt_value(key=key, state="running", generation=1, owner=dead),
    )
    engine.once()
    assert not analysis_temp.exists()


def test_malformed_sgf_never_advances(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    engine = coordinator(spec_path, FakeProcesses(malformed_sgf=True))
    with pytest.raises(SupplementContradiction, match="malformed self-play SGF"):
        engine.once()
    assert not engine.layout.selfplay_directory.exists()


def test_primary_duplicates_are_filtered_with_provenance(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    processes = FakeProcesses(duplicate_primary=True)
    engine = coordinator(spec_path, processes)
    result = engine.watch(poll_interval=0)
    assert result["state"] in {"complete", "insufficient_candidates"}
    rejected = [
        json.loads(line)
        for line in engine.layout.rejected_duplicates.read_text().splitlines()
    ]
    assert rejected[0]["reason"] == "primary_semantic_duplicate"
    assert rejected[0]["primaryManifest"]["sha256"]


def test_zero_result_prefilter_publishes_insufficiency(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    engine = coordinator(spec_path, FakeProcesses(zero_candidates=True))
    result = engine.watch(poll_interval=0)
    assert result["state"] == "insufficient_candidates"
    for label in ("lead-40", "lead-80"):
        assert (engine.layout.selected / f"{label}.jsonl").read_bytes() == b""
    assert json.loads(engine.layout.summary.read_text())["state"] == (
        "insufficient_candidates"
    )
    assert main(["watch", "--spec", str(spec_path)]) == 0


def test_clean_deployment_and_manifest_are_reverified(tmp_path):
    spec_path, assets, _ = publish_spec(tmp_path, primary_per_label=2)
    processes = FakeProcesses()
    engine = coordinator(spec_path, processes)
    (assets["repository"] / "dirty").write_text("dirty")
    with pytest.raises(SupplementContradiction, match="became dirty"):
        engine.status()
    (assets["repository"] / "dirty").unlink()
    assets["deployment"].write_bytes(assets["deployment"].read_bytes() + b"\n")
    with pytest.raises(SupplementContradiction, match="deployment manifest"):
        engine.status()


def test_round_two_reopens_after_downstream_attrition(tmp_path):
    first_path, assets, primary = publish_spec(tmp_path)
    first = coordinator(first_path, FakeProcesses())
    assert first.watch(poll_interval=0)["state"] == "complete"
    second_path, _, _ = publish_spec(
        tmp_path,
        assets=assets,
        primary=primary,
        round_number=2,
        prior_summaries=(binding(first.layout.summary),),
        downstream_counts={"lead-40": 0, "lead-80": 0},
        spec_name="supplement-spec-round-2.json",
        work_name="lead-supplement-round-2",
    )
    second = load_supplement_spec(second_path)
    assert second.round == 2
    observed = coordinator(second_path, FakeProcesses()).status()
    assert observed["deficits"] == {"lead-40": 2, "lead-80": 2}


def test_spec_rejects_critical_override_and_incomplete_inventory(tmp_path):
    spec_path, _, _ = publish_spec(tmp_path)
    value = json.loads(spec_path.read_text())
    value["selfplay_override_args"] = [["-output-dir", "/tmp/escape"]]
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(spec_path, value)
    with pytest.raises(SupplementSpecError, match="critical flag"):
        load_supplement_spec(spec_path)

    value["selfplay_override_args"] = []
    inventory_path = Path(value["primary_prefilter_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["manifests"].pop()
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    write_canonical(inventory_path, inventory)
    value["primary_prefilter_inventory"] = binding(inventory_path)
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(spec_path, value)
    with pytest.raises(SupplementSpecError, match="incomplete"):
        load_supplement_spec(spec_path)


def test_cli_status_contract_is_canonical(tmp_path, capsys):
    spec_path, _, _ = publish_spec(tmp_path, primary_per_label=2)
    status = coordinator(spec_path, FakeProcesses()).once()
    payload = dict(status)
    supplied = payload.pop("status_sha256")
    assert status["contract"] == STATUS_CONTRACT
    assert supplied == canonical_sha256(payload)
    assert main(["status", "--spec", str(spec_path)]) == 0
    cli_status = json.loads(capsys.readouterr().out)
    assert cli_status["contract"] == STATUS_CONTRACT
