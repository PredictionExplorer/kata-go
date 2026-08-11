import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from risk_score import consensus_prefilter
from risk_score.consensus_prefilter import PREFILTER_CONTRACT, PREFILTER_ROLES
from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    _normalized_positions,
    publish_normalized,
)
from risk_score.curation_pipeline import (
    GPU_OWNERSHIP_CONTRACT,
    POLICY_MINIMA,
    SPEC_CONTRACT,
    STATUS_CONTRACT,
    CurationPipeline,
    GpuComputeProcess,
    GpuOccupancy,
    GpuOwnershipManager,
    OwnershipProbes,
    PipelineBusy,
    PipelineContradiction,
    PipelineRunners,
    PipelineSnapshot,
    PipelineSpecError,
    SourceInventory,
    SourceProgress,
    infer_pipeline_snapshot,
    load_pipeline_spec,
    plan_next_stage,
)
from risk_score.curation_supplement import SPEC_CONTRACT as SUPPLEMENT_SPEC_CONTRACT
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

REVISION = "a" * 40
EXECUTED_MODULES = (
    "board_symmetry.py",
    "build_evaluation_suites.py",
    "curate_position_bank.py",
    "curation_orchestrator.py",
    "curation_pipeline.py",
    "curation_supplement.py",
    "position_samples.py",
)


def write_canonical(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def process_identity(
    pid=100,
    *,
    boot_id="boot-test",
    process_group_id=77,
    command_sha256="c" * 64,
):
    return ProcessIdentity(
        pid=pid,
        start_time_ticks=pid + 1000,
        process_group_id=process_group_id,
        boot_id=boot_id,
        command_sha256=command_sha256,
        cgroup="/test.slice",
    )


def safe_ownership_probes(
    *,
    occupancy=None,
    owner=None,
    observed_process=None,
    sleep=lambda _: None,
    lease_safe=lambda _: True,
    target_inactive=lambda: True,
    global_lock_path=None,
):
    return OwnershipProbes(
        gpu_occupancy=occupancy or (lambda: GpuOccupancy({"2": "GPU-2", "5": "GPU-5"})),
        current_process=owner or (lambda: process_identity()),
        process_identity=observed_process or (lambda pid: process_identity(pid)),
        gpu7_lease_safe=lease_safe,
        production_target_inactive=target_inactive,
        sleep=sleep,
        global_lock_path=global_lock_path or Path("/tmp/curation-pipeline-test.lock"),
    )


def binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


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
        "metadata": "pipeline-fixture",
    }


def frozen_assets(tmp_path):
    repo = tmp_path / "deploy"
    run_root = tmp_path / "run"
    repo.mkdir()
    run_root.mkdir()
    deployed_modules = repo / "python" / "risk_score"
    deployed_modules.mkdir(parents=True)
    source_modules = Path(__file__).parents[1] / "risk_score"
    files = {}
    for module_name in EXECUTED_MODULES:
        deployed = deployed_modules / module_name
        deployed.write_bytes((source_modules / module_name).read_bytes())
        files[f"module:{module_name}"] = binding(deployed)
    deployment_manifest = tmp_path / "deployment-manifest.json"
    deployment_value = {
        "schema_version": 1,
        "contract": "risk-score-live-runtime-deployment-v1",
        "source_revision": REVISION,
        "source_sha256": hashlib.sha256(REVISION.encode("utf-8")).hexdigest(),
        "files": files,
    }
    deployment_value["manifest_sha256"] = canonical_sha256(deployment_value)
    write_canonical(deployment_manifest, deployment_value)
    katago = tmp_path / "katago"
    config = tmp_path / "analysis.cfg"
    original = tmp_path / "original.bin.gz"
    champion = tmp_path / "champion.bin.gz"
    policy = tmp_path / "policy.json"
    katago.write_bytes(b"katago")
    config.write_text(DETERMINISTIC_CONFIG, encoding="utf-8")
    original.write_bytes(b"immutable-original")
    champion.write_bytes(b"frozen-champion")
    policy.write_bytes(
        (
            Path(__file__).parents[1] / "risk_score" / "promotion_policy_v3.json"
        ).read_bytes()
    )
    return {
        "repo": repo.resolve(),
        "run_root": run_root.resolve(),
        "katago": katago.resolve(),
        "config": config.resolve(),
        "original": original.resolve(),
        "champion": champion.resolve(),
        "policy": policy.resolve(),
        "deployment_manifest": deployment_manifest.resolve(),
    }


def placeholder_source(tmp_path, name, label):
    selected = tmp_path / f"{name}.jsonl"
    manifest = tmp_path / f"{name}.prefilter.json"
    write_canonical(selected, {})
    write_canonical(manifest, {})
    return {
        "name": name,
        "label": label,
        "selected": binding(selected),
        "prefilter_manifest": binding(manifest),
    }


def spec_value(assets, sources):
    revision_hash = hashlib.sha256(REVISION.encode("utf-8")).hexdigest()
    run_root = assets["run_root"]
    value = {
        "schema_version": 1,
        "contract": SPEC_CONTRACT,
        "deployment": {
            "repository_path": str(assets["repo"]),
            "source_revision": REVISION,
            "source_sha256": revision_hash,
        },
        "deployment_manifest": binding(assets["deployment_manifest"]),
        "run_root": str(run_root),
        "policy": binding(assets["policy"]),
        "katago": binding(assets["katago"]),
        "analysis_config": binding(assets["config"]),
        "models": {
            "original": binding(assets["original"]),
            "champion": binding(assets["champion"]),
        },
        "sources": sources,
        "work_root": str((run_root / "evaluation" / "curation" / "pipeline").resolve()),
        "outputs": {
            "reviewed_bank": str(
                (run_root / "evaluation" / "source-positions.jsonl").resolve()
            ),
            "reviewed_manifest": str(
                (run_root / "evaluation" / "source-positions.manifest.json").resolve()
            ),
            "suite_directory": str(
                (run_root / "evaluation" / "promotion-suites-v3").resolve()
            ),
        },
        "quotas": dict(POLICY_MINIMA),
        "topology": {
            "shards_per_role": 2,
            "gpus": ["2", "5"],
            "per_gpu_parallelism": 1,
        },
        "suite_seed": "pipeline-test",
    }
    value["spec_sha256"] = canonical_sha256(value)
    return value


def publish_spec(tmp_path, assets, sources):
    path = tmp_path / "pipeline-spec.json"
    write_canonical(path, spec_value(assets, sources))
    return path


def load_fixture_spec(tmp_path, *, two_sources=False):
    assets = frozen_assets(tmp_path)
    sources = [placeholder_source(tmp_path, "ordinary-primary", "ordinary")]
    if two_sources:
        sources = [
            placeholder_source(tmp_path, "lead-supplement", "lead-40"),
            sources[0],
        ]
    path = publish_spec(tmp_path, assets, sources)
    return (
        load_pipeline_spec(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        ),
        assets,
        path,
    )


def prefiltered_source(root, assets, *, name, label, turn, prefilter_katago_hash=None):
    root.mkdir()
    raw = root / "raw.jsonl"
    write_canonical(raw, position(turn))
    normalized = root / "normalized.jsonl"
    publish_normalized([raw], normalized, root / "normalized.manifest.json")
    selected = root / "selected.jsonl"
    selected.write_bytes(
        "".join(
            canonical_json(row) + "\n" for row in _normalized_positions(normalized)
        ).encode("utf-8")
    )
    selected_rows = _normalized_positions(selected)
    semantic_ids = [row["semanticSha256"] for row in selected_rows]
    analyses = {}
    model_hashes = {
        "original": file_sha256(assets["original"]),
        "champion": file_sha256(assets["champion"]),
    }
    execution_katago_hash = (
        file_sha256(assets["katago"])
        if prefilter_katago_hash is None
        else prefilter_katago_hash
    )
    scores = {"ordinary": 0.0, "lead-40": 55.0, "lead-80": 95.0}
    for role in PREFILTER_ROLES:
        model = role.split("/", 1)[0]
        safe = role.replace("/", "-")
        query = root / f"{safe}.query.jsonl"
        output = root / f"{safe}.analysis.jsonl"
        write_canonical(
            query,
            build_analysis_query(
                selected_rows[0],
                query_id=semantic_ids[0],
                max_visits=2000,
                powered=role.endswith("/powered-2000"),
            ),
        )
        score = scores[label]
        write_canonical(
            output,
            {
                "id": semantic_ids[0],
                "rootInfo": {
                    "scoreLead": score,
                    "winrate": 0.75,
                    "visits": 2000,
                },
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
            },
        )
        execution = {
            "contract": ANALYSIS_RUN_CONTRACT,
            "output_path": str(output.resolve()),
            "output_sha256": file_sha256(output),
            "query_path": str(query.resolve()),
            "query_sha256": file_sha256(query),
            "model_sha256": model_hashes[model],
            "katago_sha256": execution_katago_hash,
            "config_sha256": file_sha256(assets["config"]),
        }
        execution_path = Path(str(output) + ".manifest.json")
        write_canonical(execution_path, execution)
        analyses[role] = {
            "path": str(output.resolve()),
            "sha256": file_sha256(output),
            "manifest_path": str(execution_path.resolve()),
            "manifest_sha256": file_sha256(execution_path),
            "query_path": str(query.resolve()),
            "query_sha256": file_sha256(query),
            "model_sha256": model_hashes[model],
            "katago_sha256": execution_katago_hash,
            "config_sha256": file_sha256(assets["config"]),
        }
    manifest = {
        "schema_version": 1,
        "contract": PREFILTER_CONTRACT,
        "advisory_only": True,
        "requires_full_machine_consensus": True,
        "label": label,
        "maximum_score_spread": 3.0,
        "threshold_buffer": 5.0,
        "limit": None,
        "normalized": {
            "path": str(normalized.resolve()),
            "sha256": file_sha256(normalized),
            "row_count": 1,
            "semantic_ids_sha256": canonical_sha256(semantic_ids),
        },
        "analyses": analyses,
        "model_hashes": model_hashes,
        "selected": {
            "path": str(selected.resolve()),
            "sha256": file_sha256(selected),
            "row_count": 1,
            "symmetry_orbit_count": 1,
        },
        "rejection_counts": {},
    }
    manifest_path = root / "prefilter.manifest.json"
    write_canonical(manifest_path, manifest)
    return {
        "name": name,
        "label": label,
        "selected": binding(selected),
        "prefilter_manifest": binding(manifest_path),
    }


def supplement_source_summary(tmp_path, assets, source):
    training = tmp_path / "supplement-training"
    training.mkdir()
    selfplay_config = tmp_path / "supplement-selfplay.cfg"
    selfplay_config.write_text("numGameThreads = 1\n", encoding="utf-8")
    supplement_spec = {
        "schema_version": 1,
        "contract": SUPPLEMENT_SPEC_CONTRACT,
        "run_root": str(assets["run_root"]),
        "training_input_root": str(training.resolve()),
        "work_root": str(
            (
                assets["run_root"] / "evaluation" / "curation" / "supplement-fixture"
            ).resolve()
        ),
        "katago": binding(assets["katago"]),
        "analysis_config": binding(assets["config"]),
        "selfplay_config": binding(selfplay_config),
        "policy": binding(assets["policy"]),
        "models": {
            "original": binding(assets["original"]),
            "champion": binding(assets["champion"]),
        },
        "game_count": 1,
        "topology": {
            "shards_per_role": 1,
            "gpus": ["2", "5"],
            "per_gpu_parallelism": 1,
        },
        "target_counts": {"lead-40": 2, "lead-80": 2},
        "primary_prefilter_manifests": [source["prefilter_manifest"]],
        "selfplay_argv_template": [
            "{katago}",
            "selfplay",
            "--model",
            "{model}",
            "--config",
            "{selfplay_config}",
            "--output",
            "{output_dir}",
            "--games",
            "{game_count}",
        ],
    }
    supplement_spec["spec_sha256"] = canonical_sha256(supplement_spec)
    spec_path = tmp_path / "supplement-spec.json"
    write_canonical(spec_path, supplement_spec)

    artifacts = tmp_path / "supplement-summary-artifacts"
    artifacts.mkdir()

    def artifact(name):
        path = artifacts / name
        write_canonical(path, {"artifact": name})
        return binding(path)

    analyses = {
        role: {
            "output": artifact(f"{role.replace('/', '-')}.jsonl"),
            "manifest": artifact(f"{role.replace('/', '-')}.manifest.json"),
        }
        for role in PREFILTER_ROLES
    }
    summary = {
        "schema_version": 1,
        "contract": "risk-score-curation-supplement-summary-v1",
        "spec": {
            "path": str(spec_path.resolve()),
            "sha256": file_sha256(spec_path),
            "identity": supplement_spec["spec_sha256"],
        },
        "state": "complete",
        "primary_counts": {"lead-40": 1, "lead-80": 0, "ordinary": 0},
        "target_counts": supplement_spec["target_counts"],
        "generation_limits": {"lead-40": 1, "lead-80": 2},
        "supplemental_counts": {"lead-40": 1, "lead-80": 0},
        "final_counts": {"lead-40": 2, "lead-80": 0},
        "primary_prefilter_manifests": [
            {
                "label": source["label"],
                "row_count": 1,
                "path": source["prefilter_manifest"]["path"],
                "sha256": source["prefilter_manifest"]["sha256"],
            }
        ],
        "selfplay": {"game_count": 1, "receipt": artifact("selfplay.json")},
        "harvest": {
            "plan": artifact("harvest-plan.json"),
            "receipt": artifact("harvest-receipt.json"),
        },
        "normalized": {
            "positions": artifact("normalized.jsonl"),
            "manifest": artifact("normalized.manifest.json"),
        },
        "query_bundle": artifact("query-manifest.json"),
        "analyses": analyses,
        "selected": {
            source["label"]: {
                "limit": 1,
                "row_count": 1,
                "output": dict(source["selected"]),
                "manifest": dict(source["prefilter_manifest"]),
            }
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "supplement-summary.json"
    write_canonical(summary_path, summary)
    source["supplement_summary"] = binding(summary_path)
    return summary_path, summary


def progress(
    name,
    label,
    *,
    queries=False,
    consensus=False,
    labeling=False,
):
    return SourceProgress(
        name=name,
        label=label,
        selected_count=POLICY_MINIMA[label],
        queries_complete=queries,
        consensus_complete=consensus,
        labeling_complete=labeling,
        accepted_count=POLICY_MINIMA[label] if labeling else None,
        rejected_count=0 if labeling else None,
    )


def snapshot(
    first,
    second,
    *,
    combined=False,
    reviewed=False,
    suite=False,
):
    return PipelineSnapshot(
        selected_counts=dict(POLICY_MINIMA),
        deficits={},
        sources=(first, second),
        combined_required=True,
        combined_complete=combined,
        reviewed_complete=reviewed,
        suite_complete=suite,
        accepted_counts=(dict(POLICY_MINIMA) if reviewed else None),
    )


def test_spec_contract_is_canonical_strict_and_hash_bound(tmp_path):
    spec, assets, path = load_fixture_spec(tmp_path)

    assert spec.identity == json.loads(path.read_text())["spec_sha256"]
    assert spec.deployment.repository_path == assets["repo"]
    assert spec.topology.gpus == ("2", "5")

    value = json.loads(path.read_text())
    value["unexpected"] = True
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(path, value)
    with pytest.raises(PipelineSpecError, match="keys differ"):
        load_pipeline_spec(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        )

    value.pop("unexpected")
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(PipelineSpecError, match="canonical"):
        load_pipeline_spec(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        )

    value["katago"]["sha256"] = "0" * 64
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(path, value)
    with pytest.raises(PipelineSpecError, match="does not match its hash"):
        load_pipeline_spec(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        )


def test_dirty_deployment_checkout_is_rejected(tmp_path):
    assets = frozen_assets(tmp_path)
    path = publish_spec(
        tmp_path,
        assets,
        [placeholder_source(tmp_path, "ordinary", "ordinary")],
    )

    with pytest.raises(PipelineSpecError, match="uncommitted changes"):
        load_pipeline_spec(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: " M python/risk_score/module.py\n",
        )


def test_deployment_manifest_mutation_fails_closed_after_load(tmp_path):
    spec, assets, path = load_fixture_spec(tmp_path)
    coordinator = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
    )
    assets["deployment_manifest"].write_bytes(
        assets["deployment_manifest"].read_bytes() + b"\n"
    )

    with pytest.raises(PipelineContradiction, match="deployment manifest"):
        coordinator.status()

    assert spec.deployment_manifest.path == assets["deployment_manifest"]


def test_insufficient_prefilter_gate_is_read_only_before_once(tmp_path):
    assets = frozen_assets(tmp_path)
    sources = [
        prefiltered_source(
            tmp_path / "ordinary", assets, name="ordinary", label="ordinary", turn=1
        ),
        prefiltered_source(
            tmp_path / "lead40", assets, name="lead-40", label="lead-40", turn=2
        ),
        prefiltered_source(
            tmp_path / "lead80", assets, name="lead-80", label="lead-80", turn=3
        ),
    ]
    sources.sort(key=lambda item: item["name"])
    path = publish_spec(tmp_path, assets, sources)
    calls = []
    runners = PipelineRunners(
        queries=lambda **kwargs: calls.append("queries"),
        consensus=lambda **kwargs: calls.append("consensus"),
        label=lambda **kwargs: calls.append("label"),
        merge=lambda **kwargs: calls.append("merge"),
        finalize=lambda **kwargs: calls.append("finalize"),
        suites=lambda *args, **kwargs: calls.append("suites"),
    )
    coordinator = CurationPipeline(
        path,
        runners=runners,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            global_lock_path=tmp_path / "global-gpu.lock"
        ),
    )

    observed = coordinator.status()

    assert observed["state"] == "blocked_insufficient_sources"
    assert observed["selected_counts"] == {
        "lead-40": 1,
        "lead-80": 1,
        "ordinary": 1,
    }
    assert observed["deficits"] == {
        "lead-40": 2079,
        "lead-80": 4127,
        "ordinary": 3199,
    }
    assert not coordinator.spec.work_root.exists()
    assert calls == []

    persisted = coordinator.once()
    status_path = coordinator.spec.work_root / "status.json"
    stored = json.loads(status_path.read_text(encoding="utf-8"))
    payload = dict(stored)
    status_hash = payload.pop("status_sha256")
    assert persisted["state"] == "blocked_insufficient_sources"
    assert status_path.read_bytes() == (canonical_json(stored) + "\n").encode("utf-8")
    assert status_hash == canonical_sha256(payload)
    assert stored["contract"] == STATUS_CONTRACT
    assert calls == []


def test_advisory_prefilter_may_bind_a_historical_katago_binary(tmp_path):
    assets = frozen_assets(tmp_path)
    source = prefiltered_source(
        tmp_path / "source",
        assets,
        name="ordinary-historical",
        label="ordinary",
        turn=1,
        prefilter_katago_hash="f" * 64,
    )
    path = publish_spec(tmp_path, assets, [source])

    status = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
    ).status()

    assert status["state"] == "blocked_insufficient_sources"
    assert status["selected_counts"]["ordinary"] == 1


def test_prefilter_central_validator_integration_point_is_used(tmp_path, monkeypatch):
    assets = frozen_assets(tmp_path)
    source = prefiltered_source(
        tmp_path / "source",
        assets,
        name="central-validator",
        label="ordinary",
        turn=1,
    )
    calls = []

    def central_validator(manifest_path, **kwargs):
        calls.append({"manifest_path": manifest_path, **kwargs})

    monkeypatch.setattr(
        consensus_prefilter,
        "validate_prefilter_artifact",
        central_validator,
        raising=False,
    )
    path = publish_spec(tmp_path, assets, [source])

    CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
    ).status()

    assert calls == [
        {
            "manifest_path": Path(source["prefilter_manifest"]["path"]),
            "expected_label": "ordinary",
            "expected_model_hashes": {
                "original": file_sha256(assets["original"]),
                "champion": file_sha256(assets["champion"]),
            },
        }
    ]


def test_supplement_summary_binds_transitive_provenance(tmp_path):
    assets = frozen_assets(tmp_path)
    source = prefiltered_source(
        tmp_path / "source",
        assets,
        name="lead-supplement",
        label="lead-40",
        turn=1,
    )
    summary_path, summary = supplement_source_summary(tmp_path, assets, source)
    path = publish_spec(tmp_path, assets, [source])

    status = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
    ).status()

    assert status["state"] == "blocked_insufficient_sources"
    assert status["selected_counts"]["lead-40"] == 1

    transitive = Path(summary["harvest"]["receipt"]["path"])
    transitive.write_bytes(transitive.read_bytes() + b"changed")
    with pytest.raises(PipelineContradiction, match="artifact"):
        CurationPipeline(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        ).status()
    assert summary_path.is_file()


def test_supplement_summary_selected_binding_contradiction_fails(tmp_path):
    assets = frozen_assets(tmp_path)
    source = prefiltered_source(
        tmp_path / "source",
        assets,
        name="lead-supplement",
        label="lead-40",
        turn=1,
    )
    summary_path, summary = supplement_source_summary(tmp_path, assets, source)
    summary["selected"]["lead-40"]["output"]["sha256"] = "0" * 64
    summary.pop("summary_sha256")
    summary["summary_sha256"] = canonical_sha256(summary)
    write_canonical(summary_path, summary)
    source["supplement_summary"] = binding(summary_path)
    path = publish_spec(tmp_path, assets, [source])

    with pytest.raises(PipelineContradiction, match="artifact|selected output"):
        CurationPipeline(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        ).status()


def test_supplement_summary_policy_ancestry_contradiction_fails(tmp_path):
    assets = frozen_assets(tmp_path)
    source = prefiltered_source(
        tmp_path / "source",
        assets,
        name="lead-supplement",
        label="lead-40",
        turn=1,
    )
    summary_path, summary = supplement_source_summary(tmp_path, assets, source)
    supplement_spec_path = Path(summary["spec"]["path"])
    supplement_spec = json.loads(supplement_spec_path.read_text(encoding="utf-8"))
    alternate_policy = tmp_path / "alternate-policy.json"
    alternate_policy.write_bytes(assets["policy"].read_bytes())
    supplement_spec["policy"] = binding(alternate_policy)
    supplement_spec.pop("spec_sha256")
    supplement_spec["spec_sha256"] = canonical_sha256(supplement_spec)
    write_canonical(supplement_spec_path, supplement_spec)
    summary["spec"] = {
        "path": str(supplement_spec_path),
        "sha256": file_sha256(supplement_spec_path),
        "identity": supplement_spec["spec_sha256"],
    }
    summary.pop("summary_sha256")
    summary["summary_sha256"] = canonical_sha256(summary)
    write_canonical(summary_path, summary)
    source["supplement_summary"] = binding(summary_path)
    path = publish_spec(tmp_path, assets, [source])

    with pytest.raises(PipelineContradiction, match="policy ancestry"):
        CurationPipeline(
            path,
            revision_reader=lambda _: REVISION,
            repository_status_reader=lambda _: "",
        ).status()


def test_prefilter_label_contradiction_fails_before_any_stage(tmp_path):
    assets = frozen_assets(tmp_path)
    source = prefiltered_source(
        tmp_path / "source",
        assets,
        name="mislabeled-source",
        label="ordinary",
        turn=7,
    )
    source["label"] = "lead-40"
    path = publish_spec(tmp_path, assets, [source])
    coordinator = CurationPipeline(
        path,
        runners=PipelineRunners(
            queries=lambda **kwargs: pytest.fail("query stage must not run"),
            consensus=lambda **kwargs: pytest.fail("consensus stage must not run"),
        ),
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
    )

    with pytest.raises(PipelineContradiction, match="label.*expect"):
        coordinator.status()

    assert not coordinator.spec.work_root.exists()


def test_pure_planner_sequences_each_source_then_merge_finalize_and_suite():
    first_pending = progress("first", "ordinary")
    second_pending = progress("second", "lead-40")
    states = [
        snapshot(first_pending, second_pending),
        snapshot(
            replace(first_pending, queries_complete=True),
            second_pending,
        ),
        snapshot(
            replace(
                first_pending,
                queries_complete=True,
                consensus_complete=True,
            ),
            second_pending,
        ),
        snapshot(
            progress(
                "first",
                "ordinary",
                queries=True,
                consensus=True,
                labeling=True,
            ),
            second_pending,
        ),
        snapshot(
            progress(
                "first",
                "ordinary",
                queries=True,
                consensus=True,
                labeling=True,
            ),
            replace(second_pending, queries_complete=True),
        ),
        snapshot(
            progress(
                "first",
                "ordinary",
                queries=True,
                consensus=True,
                labeling=True,
            ),
            replace(
                second_pending,
                queries_complete=True,
                consensus_complete=True,
            ),
        ),
        snapshot(
            progress(
                "first",
                "ordinary",
                queries=True,
                consensus=True,
                labeling=True,
            ),
            progress(
                "second",
                "lead-40",
                queries=True,
                consensus=True,
                labeling=True,
            ),
        ),
    ]
    states.extend(
        [
            replace(states[-1], combined_complete=True),
            replace(states[-1], combined_complete=True, reviewed_complete=True),
            replace(
                states[-1],
                combined_complete=True,
                reviewed_complete=True,
                suite_complete=True,
            ),
        ]
    )

    assert [
        (action.kind, action.source) for action in map(plan_next_stage, states)
    ] == [
        ("create_queries_consensus", "first"),
        ("run_consensus", "first"),
        ("label_consensus", "first"),
        ("create_queries_consensus", "second"),
        ("run_consensus", "second"),
        ("label_consensus", "second"),
        ("merge_labeling_consensus", None),
        ("finalize_consensus", None),
        ("build_evaluation_suites", None),
        ("complete", None),
    ]


def test_watch_uses_injected_stages_in_order_and_replay_is_idempotent(
    tmp_path, monkeypatch
):
    spec, _, path = load_fixture_spec(tmp_path, two_sources=True)
    first_name = spec.sources[0].name
    second_name = spec.sources[1].name
    first_label = spec.sources[0].label
    second_label = spec.sources[1].label
    first_pending = progress(first_name, first_label)
    second_pending = progress(second_name, second_label)
    complete_first = progress(
        first_name,
        first_label,
        queries=True,
        consensus=True,
        labeling=True,
    )
    complete_second = progress(
        second_name,
        second_label,
        queries=True,
        consensus=True,
        labeling=True,
    )
    snapshots = [
        snapshot(first_pending, second_pending),
        snapshot(replace(first_pending, queries_complete=True), second_pending),
        snapshot(
            replace(
                first_pending,
                queries_complete=True,
                consensus_complete=True,
            ),
            second_pending,
        ),
        snapshot(complete_first, second_pending),
        snapshot(
            complete_first,
            replace(second_pending, queries_complete=True),
        ),
        snapshot(
            complete_first,
            replace(
                second_pending,
                queries_complete=True,
                consensus_complete=True,
            ),
        ),
        snapshot(complete_first, complete_second),
        snapshot(complete_first, complete_second, combined=True),
        snapshot(
            complete_first,
            complete_second,
            combined=True,
            reviewed=True,
        ),
        snapshot(
            complete_first,
            complete_second,
            combined=True,
            reviewed=True,
            suite=True,
        ),
    ]
    state = {"index": 0}
    events = []

    def advance(event):
        events.append(event)
        state["index"] += 1
        return {}

    def queries(**kwargs):
        output = kwargs["output"]
        write_canonical(output / "manifest.json", {"queries": {}})
        return advance(("queries", output.parent.name))

    def consensus(**kwargs):
        return advance(("consensus", kwargs["work_dir"].parent.name))

    def label(**kwargs):
        return advance(("label", kwargs["output_dir"].parent.name))

    runners = PipelineRunners(
        queries=queries,
        consensus=consensus,
        label=label,
        merge=lambda **kwargs: advance(("merge", None)),
        finalize=lambda **kwargs: advance(("finalize", None)),
        suites=lambda *args, **kwargs: advance(("suites", None)),
    )
    coordinator = CurationPipeline(
        path,
        runners=runners,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            global_lock_path=tmp_path / "watch-global-gpu.lock"
        ),
    )
    monkeypatch.setattr(coordinator, "_snapshot", lambda: snapshots[state["index"]])

    result = coordinator.watch(poll_interval=0)

    assert result["state"] == "complete"
    assert [event[0] for event in events] == [
        "queries",
        "consensus",
        "label",
        "queries",
        "consensus",
        "label",
        "merge",
        "finalize",
        "suites",
    ]
    assert state["index"] == len(snapshots) - 1

    before_replay = list(events)
    first_replay = coordinator.once()
    status_bytes = (spec.work_root / "status.json").read_bytes()
    second_replay = coordinator.once()

    assert first_replay["state"] == second_replay["state"] == "complete"
    assert events == before_replay
    assert (spec.work_root / "status.json").read_bytes() == status_bytes


def test_once_stops_before_long_consensus_stage(tmp_path, monkeypatch):
    spec, _, path = load_fixture_spec(tmp_path)
    pending = SourceProgress(
        name=spec.sources[0].name,
        label=spec.sources[0].label,
        selected_count=POLICY_MINIMA["ordinary"],
        queries_complete=False,
        consensus_complete=False,
        labeling_complete=False,
    )
    after_queries = replace(pending, queries_complete=True)
    snapshots = [
        PipelineSnapshot(
            selected_counts=dict(POLICY_MINIMA),
            deficits={},
            sources=(pending,),
            combined_required=False,
            combined_complete=False,
            reviewed_complete=False,
            suite_complete=False,
        ),
        PipelineSnapshot(
            selected_counts=dict(POLICY_MINIMA),
            deficits={},
            sources=(after_queries,),
            combined_required=False,
            combined_complete=False,
            reviewed_complete=False,
            suite_complete=False,
        ),
    ]
    state = {"index": 0}
    calls = []

    def queries(**kwargs):
        calls.append("queries")
        state["index"] = 1
        return {}

    coordinator = CurationPipeline(
        path,
        runners=PipelineRunners(
            queries=queries,
            consensus=lambda **kwargs: calls.append("consensus"),
        ),
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
    )
    monkeypatch.setattr(coordinator, "_snapshot", lambda: snapshots[state["index"]])

    result = coordinator.once()

    assert result["state"] == "awaiting_watch"
    assert result["next_stage"] == {
        "kind": "run_consensus",
        "source": spec.sources[0].name,
    }
    assert calls == ["queries"]


def test_partial_downstream_artifact_is_a_contradiction(tmp_path):
    spec, _, _ = load_fixture_spec(tmp_path)
    spec.outputs.reviewed_bank.parent.mkdir(parents=True)
    write_canonical(spec.outputs.reviewed_bank, {"unexpected": True})
    inventories = (
        SourceInventory(
            name=spec.sources[0].name,
            label=spec.sources[0].label,
            row_count=POLICY_MINIMA["ordinary"],
            semantic_ids=("semantic",),
            symmetry_orbits=("orbit",),
        ),
    )
    counts = dict(POLICY_MINIMA)
    inventories = (
        inventories[0],
        SourceInventory(
            name="synthetic-lead-40",
            label="lead-40",
            row_count=counts["lead-40"],
            semantic_ids=(),
            symmetry_orbits=(),
        ),
        SourceInventory(
            name="synthetic-lead-80",
            label="lead-80",
            row_count=counts["lead-80"],
            semantic_ids=(),
            symmetry_orbits=(),
        ),
    )

    with pytest.raises(PipelineContradiction, match="reviewed bank exists before"):
        infer_pipeline_snapshot(spec, inventories=inventories)


def test_foreign_gpu_process_blocks_ownership_before_consensus(tmp_path):
    spec, _, path = load_fixture_spec(tmp_path)
    foreign = GpuComputeProcess("GPU-2", 991, "foreign")
    calls = []
    coordinator = CurationPipeline(
        path,
        runners=PipelineRunners(consensus=lambda **kwargs: calls.append("consensus")),
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            occupancy=lambda: GpuOccupancy({"2": "GPU-2", "5": "GPU-5"}, (foreign,))
        ),
    )

    with pytest.raises(PipelineContradiction, match="foreign compute process"):
        coordinator._acquire_gpu_ownership(poll_interval=0)

    assert calls == []
    assert not coordinator.layout.gpu_ownership.exists()
    assert spec.work_root.exists() is False


@pytest.mark.parametrize("prior_state", ["claimed", "recovering", "released"])
def test_old_boot_empty_gpu_claim_is_recovered(tmp_path, prior_state):
    _, _, path = load_fixture_spec(tmp_path)
    old_owner = process_identity(100, boot_id="boot-old")
    first = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(owner=lambda: old_owner),
    )
    claim = first._acquire_gpu_ownership(poll_interval=0)
    if prior_state == "released":
        claim = first._release_gpu_ownership()
    elif prior_state == "recovering":
        claim = first.gpu_ownership._payload(
            state="recovering",
            gpus=claim["gpus"],
            owner=old_owner,
            generation=claim["generation"],
            recovered_from=claim["claim_sha256"],
        )
        write_canonical(first.layout.gpu_ownership, claim)
    restarted = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            owner=lambda: process_identity(200, boot_id="boot-new")
        ),
    )

    recovered = restarted._acquire_gpu_ownership(poll_interval=0)

    assert recovered["state"] == "claimed"
    assert recovered["generation"] == claim["generation"] + 1
    assert recovered["recovered_from_claim_sha256"] == claim["claim_sha256"]
    assert recovered["owner"]["bootId"] == "boot-new"


def test_old_boot_claim_with_occupied_gpu_is_ambiguous(tmp_path):
    _, _, path = load_fixture_spec(tmp_path)
    first = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            owner=lambda: process_identity(100, boot_id="boot-old")
        ),
    )
    first._acquire_gpu_ownership(poll_interval=0)
    restarted = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            owner=lambda: process_identity(200, boot_id="boot-new"),
            occupancy=lambda: GpuOccupancy(
                {"2": "GPU-2", "5": "GPU-5"},
                (GpuComputeProcess("GPU-2", 333, "unknown"),),
            ),
        ),
    )

    with pytest.raises(PipelineContradiction, match="ambiguous occupied"):
        restarted._acquire_gpu_ownership(poll_interval=0)


def test_restart_waits_for_matching_owned_process_then_reclaims(tmp_path):
    _, _, path = load_fixture_spec(tmp_path)
    old_owner = process_identity(100, process_group_id=70)
    first = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(owner=lambda: old_owner),
    )
    original_claim = first._acquire_gpu_ownership(poll_interval=0)
    command_hash = first._expected_analysis_command_hashes()[0]
    old_child = process_identity(
        333,
        process_group_id=old_owner.process_group_id,
        command_sha256=command_hash,
    )
    occupancies = [
        GpuOccupancy(
            {"2": "GPU-2", "5": "GPU-5"},
            (GpuComputeProcess("GPU-2", old_child.pid, "katago"),),
        ),
        GpuOccupancy({"2": "GPU-2", "5": "GPU-5"}),
    ]
    sleeps = []
    new_owner = process_identity(200, process_group_id=80)
    restarted = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(
            occupancy=lambda: occupancies.pop(0),
            owner=lambda: new_owner,
            observed_process=lambda _: old_child,
            sleep=sleeps.append,
        ),
    )

    recovered = restarted._acquire_gpu_ownership(poll_interval=0.25)

    assert recovered["state"] == "claimed"
    assert recovered["generation"] == 2
    assert recovered["owner"] == new_owner.to_dict()
    assert recovered["recovered_from_claim_sha256"] == original_claim["claim_sha256"]
    assert sleeps == [0.25]


def test_gpu_ownership_release_is_canonical_and_restartable(tmp_path):
    _, _, path = load_fixture_spec(tmp_path)
    coordinator = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(),
    )
    coordinator._acquire_gpu_ownership(poll_interval=0)

    released = coordinator._release_gpu_ownership()
    stored = json.loads(coordinator.layout.gpu_ownership.read_text(encoding="utf-8"))

    assert released["state"] == "released"
    assert released["contract"] == GPU_OWNERSHIP_CONTRACT
    assert stored == released
    payload = dict(stored)
    assert payload.pop("claim_sha256") == canonical_sha256(payload)
    assert coordinator.layout.gpu_ownership.read_bytes() == (
        canonical_json(stored) + "\n"
    ).encode("utf-8")

    reacquired = coordinator._acquire_gpu_ownership(poll_interval=0)
    assert reacquired["state"] == "claimed"
    assert reacquired["generation"] == released["generation"] + 1


def test_release_refuses_residual_gpu_process_and_keeps_claim(tmp_path):
    _, _, path = load_fixture_spec(tmp_path)
    occupancies = [
        GpuOccupancy({"2": "GPU-2", "5": "GPU-5"}),
        GpuOccupancy(
            {"2": "GPU-2", "5": "GPU-5"},
            (GpuComputeProcess("GPU-5", 444, "katago"),),
        ),
    ]
    coordinator = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(occupancy=lambda: occupancies.pop(0)),
    )
    coordinator._acquire_gpu_ownership(poll_interval=0)

    with pytest.raises(PipelineContradiction, match="remains occupied"):
        coordinator._release_gpu_ownership()

    claim = json.loads(coordinator.layout.gpu_ownership.read_text(encoding="utf-8"))
    assert claim["state"] == "claimed"


def test_gpu_ownership_requires_safe_lease_and_inactive_target(tmp_path):
    _, _, path = load_fixture_spec(tmp_path)
    unsafe_lease = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(lease_safe=lambda _: False),
    )
    with pytest.raises(PipelineContradiction, match="lease state"):
        unsafe_lease._acquire_gpu_ownership(poll_interval=0)

    active_target = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=safe_ownership_probes(target_inactive=lambda: False),
    )
    with pytest.raises(PipelineContradiction, match="systemd target"):
        active_target._acquire_gpu_ownership(poll_interval=0)


def test_global_gpu_lock_excludes_other_pipeline_instances(tmp_path):
    _, _, path = load_fixture_spec(tmp_path)
    lock_path = tmp_path / "host-global-gpu.lock"
    probes = safe_ownership_probes(global_lock_path=lock_path)
    first = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=probes,
    )
    second = CurationPipeline(
        path,
        revision_reader=lambda _: REVISION,
        repository_status_reader=lambda _: "",
        ownership_probes=probes,
    )

    with first._global_gpu_lock():
        with pytest.raises(PipelineBusy, match="global GPU ownership"):
            with second._global_gpu_lock():
                pytest.fail("second coordinator must not acquire the lock")


def test_public_gpu_ownership_manager_is_reusable_without_pipeline(tmp_path):
    spec, _, _ = load_fixture_spec(tmp_path)
    probes = safe_ownership_probes(global_lock_path=tmp_path / "standalone.lock")
    manager = GpuOwnershipManager(
        claim_path=tmp_path / "standalone-claim.json",
        spec_path=spec.path,
        spec_sha256=spec.file_sha256,
        spec_identity=spec.identity,
        configured_gpu_ids=("2", "5"),
        topology_binding={
            "gpus": ["2", "5"],
            "per_gpu_parallelism": 1,
            "shards_per_role": 2,
        },
        expected_command_sha256s=("d" * 64,),
        run_root=spec.run_root,
        probes=probes,
    )

    with manager.global_lock():
        claimed = manager.acquire(poll_interval=0)
        released = manager.release()

    assert claimed["state"] == "claimed"
    assert released["state"] == "released"
