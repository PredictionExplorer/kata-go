import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.build_evaluation_suites import (
    DEFAULT_POLICY_PATH,
    build_evaluation_suites,
)
from risk_score.curate_position_bank import (
    analysis_features,
    build_harvest_argv,
    execute_harvest_plan,
    finalize_reviewed_bank,
    generate_query_bundle,
    label_positions,
    merge_analysis,
    policy_pool_minima,
    publish_harvest_plan,
    publish_normalized,
    run_analysis,
    split_queries,
)
from risk_score.position_samples import (
    build_analysis_query,
    canonical_json,
    canonical_sha256,
    file_sha256,
    normalize_position_sample,
    semantic_position_sha256,
)


def position(index, *, board=None):
    return {
        "xSize": 19,
        "ySize": 19,
        "board": board or "/".join(["." * 19] * 19),
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": index,
        "hintLoc": "null",
        "metadata": "source",
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


DETERMINISTIC_ANALYSIS_CONFIG = """
forDeterministicTesting = true
numAnalysisThreads = 1
nnRandomize = false
rootNoiseEnabled = false
rootNumSymmetriesToSample = 1
useUncertainty = false
cpuctUtilityStdevScale = 0
reportAnalysisWinratesAs = SIDETOMOVE
""".strip() + "\n"


def write_tiny_policy(path):
    policy = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    policy["policy_version"] = "risk-seeking-checkpoint-promotion-v2-curation-test"
    stages = policy["evaluation_stages"]
    stages["stage_0_integrity_and_fixed_probes"].update(
        {
            "fixed_analysis_positions": 1,
            "exploitability_sentinel_positions": 1,
        }
    )
    stages["stage_1_cheap_paired_screen"]["ordinary_color_pairs"] = 1
    stages["stage_2_finalist_selection"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    for look in stages["stage_3_promotion_confirmation"]["looks"]:
        look.update(
            {
                "powered_ordinary_color_pairs_per_matchup": 1,
                "standard_ordinary_color_pairs": 1,
                "lead_40_color_pairs": 1,
                "lead_80_color_pairs": 1,
                "minimum_independent_position_clusters": {
                    "powered_candidate_vs_champion": 1,
                    "powered_candidate_vs_original": 1,
                    "standard_candidate_vs_original": 1,
                    "lead_40": 1,
                    "lead_80": 1,
                },
            }
        )
    stages["deep_audit"].update(
        {
            "ordinary_color_pairs": 1,
            "lead_40_color_pairs": 0,
            "lead_80_color_pairs": 0,
            "exploitability_positions": 1,
        }
    )
    path.write_text(canonical_json(policy) + "\n", encoding="utf-8")
    return policy


def analysis_record(
    record_id,
    score,
    *,
    move="D4",
    prior=0.2,
    stdev=10.0,
    visits=800,
):
    return {
        "id": record_id,
        "rootInfo": {
            "scoreLead": score,
            "winrate": 0.6,
            "visits": visits,
        },
        "moveInfos": [
            {
                "move": move,
                "order": 0,
                "visits": visits,
                "prior": prior,
                "scoreLead": score,
                "scoreSelfplay": score + 1.0,
                "scoreStdev": stdev,
                "utility": 0.2,
            }
        ],
    }


def write_analysis_manifest(path, *, query, katago, config, model, row_count):
    manifest = {
        "schema_version": 1,
        "contract": "risk-score-position-analysis-run-v1",
        "argv": ["/fake/katago", "analysis"],
        "katago_sha256": file_sha256(katago),
        "config_sha256": file_sha256(config),
        "model_sha256": file_sha256(model),
        "query_path": str(query.resolve()),
        "query_sha256": file_sha256(query),
        "output_path": str(path.resolve()),
        "output_sha256": file_sha256(path),
        "row_count": row_count,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    Path(str(path) + ".manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )


def create_sharded_analysis_fixture(
    root, *, powered=False, visits=200, shard_count=2
):
    root.mkdir()
    queries = root / "queries.jsonl"
    query_rows = [
        build_analysis_query(
            position(index),
            query_id=f"id-{index}",
            max_visits=visits,
            powered=powered,
        )
        for index in range(8)
    ]
    write_jsonl(queries, query_rows)
    bundle = root / "shards"
    split_manifest = split_queries(queries, bundle, shard_count=shard_count)
    katago = root / "katago"
    config = root / "analysis.cfg"
    model = root / "model.bin.gz"
    katago.write_bytes(b"katago")
    config.write_text(DETERMINISTIC_ANALYSIS_CONFIG, encoding="utf-8")
    model.write_bytes(b"model")
    outputs = []
    for shard in split_manifest["shards"]:
        shard_query = bundle / shard["path"]
        rows = [
            analysis_record(row["id"], 0.0, visits=visits)
            for row in (
                json.loads(line)
                for line in shard_query.read_text(encoding="utf-8").splitlines()
            )
        ]
        output = root / f"result-{shard['index']}.jsonl"
        write_jsonl(output, rows)
        write_analysis_manifest(
            output,
            query=shard_query,
            katago=katago,
            config=config,
            model=model,
            row_count=len(rows),
        )
        outputs.append(output)
    return {
        "queries": queries,
        "bundle": bundle,
        "split_manifest": split_manifest,
        "katago": katago,
        "config": config,
        "model": model,
        "outputs": outputs,
        "query_rows": query_rows,
    }


def test_normalization_rejects_semantic_duplicates_and_builds_analysis_query(tmp_path):
    rows = ["X" + "." * 18] + ["." * 19] * 18
    source = tmp_path / "positions.jsonl"
    write_jsonl(source, [position(3, board="/".join(rows))])
    output = tmp_path / "normalized.jsonl"
    manifest = tmp_path / "normalized-manifest.json"
    first = publish_normalized([source], output, manifest)
    second = publish_normalized([source], output, manifest)
    assert first == second

    normalized = json.loads(output.read_text(encoding="utf-8"))
    checked = normalize_position_sample(normalized, "test")
    assert normalized["semanticSha256"] == semantic_position_sha256(checked)
    query = build_analysis_query(
        checked,
        query_id=normalized["semanticSha256"],
        max_visits=800,
        powered=True,
    )
    assert query["initialStones"] == [["B", "A19"]]
    assert query["overrideSettings"]["winWeight"] == 4.0
    assert query["overrideSettings"]["useScoreMaximizingUtility"] is True
    equivalent = {
        **checked,
        "board": checked["board"].lower() + "/",
        "nextPla": "b",
        "hintLoc": "",
    }
    assert semantic_position_sha256(equivalent) == semantic_position_sha256(
        checked
    )

    duplicate = tmp_path / "duplicate.jsonl"
    changed_annotation = {**position(3, board="/".join(rows)), "metadata": "other"}
    write_jsonl(duplicate, [position(3, board="/".join(rows)), changed_annotation])
    with pytest.raises(ValueError, match="duplicate semantic position"):
        publish_normalized(
            [duplicate],
            tmp_path / "duplicate-normalized.jsonl",
            tmp_path / "duplicate-manifest.json",
        )


def test_harvest_plan_is_shell_free_and_content_bound(tmp_path):
    katago = tmp_path / "katago"
    katago.write_bytes(b"binary")
    sgfs = tmp_path / "sgfs"
    sgfs.mkdir()
    (sgfs / "games.sgfs").write_bytes(b"(;GM[1])\n")
    training_input = tmp_path / "training-selfplay"
    training_input.mkdir()
    output = tmp_path / "harvested"
    argv = build_harvest_argv(
        katago=katago,
        sgfs_dirs=[sgfs],
        sgf_dirs=[],
        training_input_roots=[training_input],
        output_dir=output,
        threads=1,
    )
    assert argv[:2] == (str(katago.resolve()), "samplesgfs")
    assert argv[argv.index("-sample-prob") + 1] == "1"
    assert "-for-testing" in argv

    manifest_path = tmp_path / "harvest.json"
    manifest = publish_harvest_plan(
        katago=katago,
        sgfs_dirs=[sgfs],
        sgf_dirs=[],
        training_input_roots=[training_input],
        output_dir=output,
        manifest_path=manifest_path,
        threads=1,
    )
    assert manifest["katago_sha256"] == file_sha256(katago)
    assert manifest["inputs"][0]["files"][0]["sha256"] == file_sha256(
        sgfs / "games.sgfs"
    )

    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        target = Path(argv[argv.index("-outdir") + 1])
        target.mkdir(exist_ok=True)
        (target / "0.startposes.txt").write_text(
            canonical_json(position(0)) + "\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    receipt = execute_harvest_plan(manifest_path, subprocess_runner=fake)
    assert receipt["reused"] is False
    assert calls[0][1]["shell"] is False
    assert (output / "receipt.json").is_file()
    reused = execute_harvest_plan(manifest_path, subprocess_runner=fake)
    assert reused["reused"] is True
    assert len(calls) == 1

    mutable = tmp_path / "mutable-sgfs"
    mutable.mkdir()
    mutable_file = mutable / "growing.sgfs"
    mutable_file.write_bytes(b"(;GM[1])\n")
    mutable_plan = tmp_path / "mutable-plan.json"
    publish_harvest_plan(
        katago=katago,
        sgfs_dirs=[mutable],
        sgf_dirs=[],
        training_input_roots=[training_input],
        output_dir=tmp_path / "mutable-output",
        manifest_path=mutable_plan,
        threads=1,
    )
    mutable_file.write_bytes(b"(;GM[1];B[pd])\n")
    with pytest.raises(ValueError, match="inventory changed"):
        execute_harvest_plan(mutable_plan, subprocess_runner=fake)

    contaminated = training_input / "games"
    contaminated.mkdir()
    with pytest.raises(ValueError, match="training/shuffler"):
        build_harvest_argv(
            katago=katago,
            sgfs_dirs=[contaminated],
            sgf_dirs=[],
            training_input_roots=[training_input],
            output_dir=tmp_path / "bad-output",
            threads=1,
        )

    mixed_root = tmp_path / "mixed-root"
    mixed_root.mkdir()
    (mixed_root / "quarantine.sgfs").write_bytes(b"(;GM[1])\n")
    nested_training = mixed_root / "training"
    nested_training.mkdir()
    with pytest.raises(ValueError, match="training/shuffler"):
        build_harvest_argv(
            katago=katago,
            sgfs_dirs=[mixed_root],
            sgf_dirs=[],
            training_input_roots=[nested_training],
            output_dir=tmp_path / "mixed-output",
            threads=1,
        )


def test_query_generation_and_conservative_auto_labeling(tmp_path):
    policy_path = tmp_path / "policy.json"
    write_tiny_policy(policy_path)
    source = tmp_path / "source.jsonl"
    raw_positions = [position(index) for index in range(4)]
    write_jsonl(source, raw_positions)
    normalized = tmp_path / "normalized.jsonl"
    publish_normalized(
        [source], normalized, tmp_path / "normalized-manifest.json"
    )
    model = tmp_path / "original.bin.gz"
    model.write_bytes(b"original")
    katago = tmp_path / "katago"
    config = tmp_path / "analysis.cfg"
    katago.write_bytes(b"katago")
    config.write_text(DETERMINISTIC_ANALYSIS_CONFIG, encoding="utf-8")
    query_dir = tmp_path / "queries"
    query_manifest = generate_query_bundle(
        normalized,
        query_dir,
        katago_binary=katago,
        analysis_config=config,
        reference_model=model,
        policy_path=policy_path,
    )
    assert {
        "standard-200",
        "standard-800",
        "standard-2000",
        "powered-800",
        "powered-2000",
    } == set(query_manifest["queries"])

    normalized_rows = [
        json.loads(line)
        for line in normalized.read_text(encoding="utf-8").splitlines()
    ]
    desired_scores = {
        normalized_rows[0]["semanticSha256"]: 0.0,
        normalized_rows[1]["semanticSha256"]: 50.0,
        normalized_rows[2]["semanticSha256"]: 90.0,
        normalized_rows[3]["semanticSha256"]: 10.0,
    }
    analyses = {}
    for role in query_manifest["queries"]:
        path = tmp_path / f"{role}.jsonl"
        rows = []
        for record_id, score in desired_scores.items():
            powered_review = (
                role == "powered-2000"
                and record_id == normalized_rows[3]["semanticSha256"]
            )
            rows.append(
                analysis_record(
                    record_id,
                    score,
                    move="Q16" if powered_review else "D4",
                    visits=int(role.rsplit("-", 1)[1]),
                )
            )
        write_jsonl(path, rows)
        query_artifact = query_manifest["queries"][role]
        write_analysis_manifest(
            path,
            query=query_dir / query_artifact["path"],
            katago=katago,
            config=config,
            model=model,
            row_count=len(rows),
        )
        analyses[role] = path
    labeling_dir = tmp_path / "labeling"
    manifest = label_positions(
        normalized_path=normalized,
        query_manifest_path=query_dir / "manifest.json",
        analysis_paths=analyses,
        output_dir=labeling_dir,
    )
    auto_rows = [
        json.loads(line)
        for line in (labeling_dir / "auto-labeled.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    review_rows = [
        json.loads(line)
        for line in (labeling_dir / "review-queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["automatic_count"] == 3
    assert {row["labels"][0] for row in auto_rows} == {
        "ordinary",
        "lead-40",
        "lead-80",
    }
    assert manifest["review_count"] == 1
    assert review_rows[0]["suggested_specialized_labels"] == ["adversarial"]


def test_analysis_runner_uses_argv_and_rejects_incomplete_ids(tmp_path):
    with pytest.raises(ValueError, match="below"):
        analysis_features(
            analysis_record("short", 0.0, visits=799),
            "standard-800",
            expected_visits=800,
        )
    katago = tmp_path / "katago"
    config = tmp_path / "analysis.cfg"
    model = tmp_path / "model.bin.gz"
    katago.write_bytes(b"katago")
    config.write_text(DETERMINISTIC_ANALYSIS_CONFIG, encoding="utf-8")
    model.write_bytes(b"model")
    queries = tmp_path / "queries.jsonl"
    write_jsonl(
        queries,
        [
            build_analysis_query(
                position(index),
                query_id=f"id-{index}",
                max_visits=200,
                powered=False,
            )
            for index in range(2)
        ],
    )
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        query_rows = [
            json.loads(line)
            for line in kwargs["stdin"].read().decode().splitlines()
        ]
        kwargs["stdout"].write(
            "".join(
                canonical_json(analysis_record(row["id"], 0.0)) + "\n"
                for row in query_rows
            ).encode()
        )
        return SimpleNamespace(returncode=0, stderr=b"")

    output = tmp_path / "analysis.jsonl"
    result = run_analysis(
        katago=katago,
        config=config,
        model=model,
        queries=queries,
        output=output,
        subprocess_runner=fake,
    )
    assert result["row_count"] == 2
    assert Path(result["manifest_path"]).is_file()
    assert calls[0][1]["shell"] is False
    assert calls[0][0][1] == "analysis"

    def incomplete(argv, **kwargs):
        kwargs["stdout"].write(
            (canonical_json(analysis_record("id-0", 0.0)) + "\n").encode()
        )
        return SimpleNamespace(returncode=0, stderr=b"")

    with pytest.raises(ValueError, match="do not match"):
        run_analysis(
            katago=katago,
            config=config,
            model=model,
            queries=queries,
            output=tmp_path / "incomplete.jsonl",
            subprocess_runner=incomplete,
        )


def test_query_shards_merge_to_one_provenance_bound_result(tmp_path):
    fixture = create_sharded_analysis_fixture(tmp_path / "standard")
    merged = tmp_path / "merged.jsonl"
    receipt = merge_analysis(
        query_path=fixture["queries"],
        split_manifest_path=fixture["bundle"] / "manifest.json",
        shard_outputs=fixture["outputs"],
        output=merged,
    )
    assert receipt["row_count"] == len(fixture["query_rows"])
    assert receipt["query_sha256"] == file_sha256(fixture["queries"])
    assert receipt["split_manifest_sha256"] == file_sha256(
        fixture["bundle"] / "manifest.json"
    )
    assert {
        json.loads(line)["id"]
        for line in merged.read_text(encoding="utf-8").splitlines()
    } == {row["id"] for row in fixture["query_rows"]}


def test_merge_analysis_rejects_cross_role_and_incomplete_shards(tmp_path):
    standard = create_sharded_analysis_fixture(tmp_path / "standard")
    powered = create_sharded_analysis_fixture(
        tmp_path / "powered", powered=True
    )
    with pytest.raises(ValueError, match="not in split manifest"):
        merge_analysis(
            query_path=standard["queries"],
            split_manifest_path=standard["bundle"] / "manifest.json",
            shard_outputs=[standard["outputs"][0], powered["outputs"][1]],
            output=tmp_path / "cross-role.jsonl",
        )
    with pytest.raises(ValueError, match="output count"):
        merge_analysis(
            query_path=standard["queries"],
            split_manifest_path=standard["bundle"] / "manifest.json",
            shard_outputs=standard["outputs"][:1],
            output=tmp_path / "missing.jsonl",
        )
    with pytest.raises(ValueError, match="more than once"):
        merge_analysis(
            query_path=standard["queries"],
            split_manifest_path=standard["bundle"] / "manifest.json",
            shard_outputs=[standard["outputs"][0], standard["outputs"][0]],
            output=tmp_path / "duplicate.jsonl",
        )


def test_merge_analysis_rejects_misbound_shard_ids_and_source(tmp_path):
    fixture = create_sharded_analysis_fixture(tmp_path / "standard")
    first = fixture["outputs"][0]
    first_manifest_path = Path(str(first) + ".manifest.json")
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    second_query = fixture["bundle"] / fixture["split_manifest"]["shards"][1]["path"]
    first_manifest["query_path"] = str(second_query.resolve())
    first_manifest["query_sha256"] = file_sha256(second_query)
    first_manifest.pop("manifest_sha256")
    first_manifest["manifest_sha256"] = canonical_sha256(first_manifest)
    first_manifest_path.write_text(
        canonical_json(first_manifest) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="output IDs are misbound"):
        merge_analysis(
            query_path=fixture["queries"],
            split_manifest_path=fixture["bundle"] / "manifest.json",
            shard_outputs=fixture["outputs"],
            output=tmp_path / "misbound.jsonl",
        )

    other = create_sharded_analysis_fixture(
        tmp_path / "other", visits=800
    )
    with pytest.raises(ValueError, match="another source query file"):
        merge_analysis(
            query_path=fixture["queries"],
            split_manifest_path=other["bundle"] / "manifest.json",
            shard_outputs=fixture["outputs"],
            output=tmp_path / "wrong-source.jsonl",
        )


def test_analysis_runner_rejects_multi_thread_deterministic_config(tmp_path):
    fixture = create_sharded_analysis_fixture(tmp_path / "fixture")
    config = tmp_path / "multi-thread.cfg"
    config.write_text(
        DETERMINISTIC_ANALYSIS_CONFIG.replace(
            "numAnalysisThreads = 1", "numAnalysisThreads = 8"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="numAnalysisThreads"):
        run_analysis(
            katago=fixture["katago"],
            config=config,
            model=fixture["model"],
            queries=fixture["queries"],
            output=tmp_path / "not-run.jsonl",
        )


def test_finalize_requires_review_and_policy_minima_then_feeds_suite_builder(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy = write_tiny_policy(policy_path)
    minima = policy_pool_minima(policy)
    assert minima["ordinary"] == 3
    assert minima["lead-40"] == 2
    assert minima["lead-80"] == 2

    auto_rows = []
    index = 0
    for label, count in (
        ("ordinary", 3),
        ("lead-40", 2),
        ("lead-80", 2),
    ):
        for _ in range(count):
            item = position(index)
            index += 1
            auto_rows.append(
                {
                    **item,
                    "labels": [label],
                    "curation": {
                        "classification": "automatic",
                        "semanticSha256": semantic_position_sha256(item),
                    },
                }
            )
    reviewed = position(index)
    reviewed_hash = semantic_position_sha256(reviewed)
    review_rows = [
        {
            "semantic_sha256": reviewed_hash,
            "position": reviewed,
            "recommended_auto_label": None,
            "suggested_specialized_labels": ["baits"],
        }
    ]
    decisions = [
        {
            "semantic_sha256": reviewed_hash,
            "approved": True,
            "hint_loc": "D4",
            "labels": [
                "tactical",
                "exploitability",
                "baits",
                "tails",
                "sacrifice",
                "small-gain",
                "adversarial",
            ],
        }
    ]
    auto_path = tmp_path / "auto.jsonl"
    review_path = tmp_path / "review.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    write_jsonl(auto_path, auto_rows)
    write_jsonl(review_path, review_rows)
    write_jsonl(decisions_path, decisions)
    labeling_manifest_path = tmp_path / "labeling-manifest.json"
    labeling_manifest = {
        "contract": "risk-score-position-bank-labeling-v1",
        "automatic_count": len(auto_rows),
        "review_count": len(review_rows),
        "automatic_sha256": file_sha256(auto_path),
        "review_queue_sha256": file_sha256(review_path),
        "policy_hash": canonical_sha256(policy),
    }
    labeling_manifest["manifest_sha256"] = canonical_sha256(labeling_manifest)
    labeling_manifest_path.write_text(
        canonical_json(labeling_manifest) + "\n",
        encoding="utf-8",
    )
    wrong_policy_manifest = dict(labeling_manifest)
    wrong_policy_manifest["policy_hash"] = "0" * 64
    wrong_policy_manifest.pop("manifest_sha256")
    wrong_policy_manifest["manifest_sha256"] = canonical_sha256(
        wrong_policy_manifest
    )
    wrong_policy_path = tmp_path / "wrong-policy-labeling.json"
    wrong_policy_path.write_text(
        canonical_json(wrong_policy_manifest) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="another promotion policy"):
        finalize_reviewed_bank(
            auto_path=auto_path,
            review_queue_path=review_path,
            decisions_path=decisions_path,
            labeling_manifest_path=wrong_policy_path,
            policy_path=policy_path,
            output_path=tmp_path / "wrong-policy.jsonl",
            manifest_path=tmp_path / "wrong-policy-manifest.json",
        )
    output = tmp_path / "source-positions.jsonl"
    manifest_path = tmp_path / "curation-manifest.json"
    manifest = finalize_reviewed_bank(
        auto_path=auto_path,
        review_queue_path=review_path,
        decisions_path=decisions_path,
        labeling_manifest_path=labeling_manifest_path,
        policy_path=policy_path,
        output_path=output,
        manifest_path=manifest_path,
    )
    assert manifest["row_count"] == 8
    assert manifest["label_counts"]["exploitability"] == 1

    suites = build_evaluation_suites(
        [output],
        tmp_path / "suites",
        seed="curation-test",
        policy_path=policy_path,
    )
    assert suites.manifest["exactPolicyQuotas"] is True
    assert len(suites.manifest["cells"]) == 15

    write_jsonl(
        decisions_path,
        [
            {
                "semantic_sha256": reviewed_hash,
                "approved": True,
                "labels": ["lead-40", "baits"],
            }
        ],
    )
    with pytest.raises(ValueError, match="may not bridge"):
        finalize_reviewed_bank(
            auto_path=auto_path,
            review_queue_path=review_path,
            decisions_path=decisions_path,
            labeling_manifest_path=labeling_manifest_path,
            policy_path=policy_path,
            output_path=tmp_path / "bridged.jsonl",
            manifest_path=tmp_path / "bridged-manifest.json",
        )

    write_jsonl(auto_path, auto_rows[1:])
    with pytest.raises(ValueError, match="below policy minima|manifest inputs changed"):
        finalize_reviewed_bank(
            auto_path=auto_path,
            review_queue_path=review_path,
            decisions_path=decisions_path,
            labeling_manifest_path=labeling_manifest_path,
            policy_path=policy_path,
            output_path=tmp_path / "insufficient.jsonl",
            manifest_path=tmp_path / "insufficient-manifest.json",
        )
