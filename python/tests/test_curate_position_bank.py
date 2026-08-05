import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risk_score.build_evaluation_suites import (
    build_evaluation_suites,
)
from risk_score.curate_position_bank import (
    COMBINED_LABELING_CONTRACT,
    SCORE_PREFILTER_CONTRACT,
    SGFS_FILTER_CONTRACT,
    analysis_features,
    build_harvest_argv,
    execute_harvest_plan,
    filter_sgfs_by_result_margin,
    finalize_reviewed_bank,
    generate_query_bundle,
    label_positions,
    merge_analysis,
    merge_labeling_bundles,
    policy_pool_minima,
    publish_harvest_plan,
    publish_normalized,
    run_analysis,
    score_prefilter_positions,
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

LEGACY_POLICY_PATH = (
    Path(__file__).parents[1] / "risk_score" / "promotion_policy_v2.json"
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
    policy = json.loads(LEGACY_POLICY_PATH.read_text(encoding="utf-8"))
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


def create_sharded_analysis_fixture(root, *, powered=False, visits=200, shard_count=2):
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


def create_score_prefilter_fixture(root, scores):
    root.mkdir()
    policy_path = root / "policy.json"
    write_tiny_policy(policy_path)
    source = root / "source.jsonl"
    write_jsonl(source, [position(index) for index in range(len(scores))])
    normalized = root / "normalized.jsonl"
    publish_normalized([source], normalized, root / "normalized-manifest.json")
    katago = root / "katago"
    config = root / "analysis.cfg"
    model = root / "model.bin.gz"
    katago.write_bytes(b"katago")
    config.write_text(DETERMINISTIC_ANALYSIS_CONFIG, encoding="utf-8")
    model.write_bytes(b"model")
    query_dir = root / "queries"
    query_manifest = generate_query_bundle(
        normalized,
        query_dir,
        katago_binary=katago,
        analysis_config=config,
        reference_model=model,
        policy_path=policy_path,
    )
    normalized_rows = [
        json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()
    ]
    score_by_id = {
        row["semanticSha256"]: score for row, score in zip(normalized_rows, scores)
    }
    query_path = query_dir / query_manifest["queries"]["standard-200"]["path"]
    query_rows = [
        json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()
    ]
    analysis = root / "standard-200.jsonl"
    write_jsonl(
        analysis,
        [
            analysis_record(row["id"], score_by_id[row["id"]], visits=200)
            for row in query_rows
        ],
    )
    write_analysis_manifest(
        analysis,
        query=query_path,
        katago=katago,
        config=config,
        model=model,
        row_count=len(query_rows),
    )
    return {
        "normalized": normalized,
        "query_manifest": query_dir / "manifest.json",
        "query_path": query_path,
        "analysis": analysis,
        "model": model,
        "score_by_id": score_by_id,
    }


def automatic_row(index, label):
    item = position(index)
    return {
        **item,
        "labels": [label],
        "curation": {
            "classification": "automatic",
            "semanticSha256": semantic_position_sha256(item),
        },
    }


def review_row(index):
    item = position(index)
    return {
        "semantic_sha256": semantic_position_sha256(item),
        "position": item,
        "recommended_auto_label": None,
        "suggested_specialized_labels": [],
    }


def write_labeling_bundle(
    root,
    auto_rows,
    review_rows,
    *,
    policy_hash,
    reference_model_hash,
    stability_margin=5.0,
):
    root.mkdir()
    auto_path = root / "auto-labeled.jsonl"
    review_path = root / "review-queue.jsonl"
    write_jsonl(auto_path, auto_rows)
    write_jsonl(review_path, review_rows)
    manifest = {
        "schema_version": 1,
        "contract": "risk-score-position-bank-labeling-v1",
        "reference_model_sha256": reference_model_hash,
        "policy_hash": policy_hash,
        "stability_margin": stability_margin,
        "automatic_count": len(auto_rows),
        "review_count": len(review_rows),
        "automatic_sha256": file_sha256(auto_path),
        "review_queue_sha256": file_sha256(review_path),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    return manifest


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
    assert semantic_position_sha256(equivalent) == semantic_position_sha256(checked)

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


def test_sgfs_margin_filter_is_deterministic_and_content_bound(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "games.sgfs"
    second = second_dir / "games.sgfs"
    first.write_text(
        "(;GM[1]FF[4]RE[B+20.5])\n"
        "(;GM[1]FF[4]RE[W+80.5])\n"
        "(;GM[1]FF[4]RE[B+R])\n",
        encoding="utf-8",
    )
    second.write_text("(;GM[1]FF[4]RE[B+50.5])\n", encoding="utf-8")
    output = tmp_path / "filtered" / "games.sgfs"
    manifest_path = tmp_path / "filtered" / "manifest.json"
    initial = filter_sgfs_by_result_margin(
        [second, first],
        output_path=output,
        manifest_path=manifest_path,
        minimum_margin=40.0,
    )
    repeated = filter_sgfs_by_result_margin(
        [first, second],
        output_path=output,
        manifest_path=manifest_path,
        minimum_margin=40.0,
    )
    assert initial == repeated
    assert initial["contract"] == SGFS_FILTER_CONTRACT
    assert initial["source_count"] == 4
    assert initial["numeric_result_count"] == 3
    assert initial["non_numeric_result_count"] == 1
    assert initial["selected_count"] == 2
    assert initial["selected_margin_minimum"] == 50.5
    assert initial["selected_margin_maximum"] == 80.5
    assert set(output.read_text(encoding="utf-8").splitlines()) == {
        "(;GM[1]FF[4]RE[W+80.5])",
        "(;GM[1]FF[4]RE[B+50.5])",
    }

    second.write_text(
        "(;GM[1]FF[4]RE[B+50.5])\n(;GM[1]FF[4]RE[W+60.5])\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="immutable output conflicts"):
        filter_sgfs_by_result_margin(
            [first, second],
            output_path=output,
            manifest_path=manifest_path,
            minimum_margin=40.0,
        )


def test_sgfs_margin_filter_rejects_duplicates_and_source_mutation(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "games.sgfs"
    game = "(;GM[1]FF[4]RE[B+80.5])"
    source.write_text(game + "\n" + game + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate SGF game"):
        filter_sgfs_by_result_margin(
            [source],
            output_path=tmp_path / "output" / "games.sgfs",
            manifest_path=tmp_path / "output" / "manifest.json",
            minimum_margin=40.0,
        )

    source.write_text(game + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="may not modify a source directory"):
        filter_sgfs_by_result_margin(
            [source],
            output_path=source_dir / "filtered.sgfs",
            manifest_path=tmp_path / "manifest.json",
            minimum_margin=40.0,
        )


def test_query_generation_and_conservative_auto_labeling(tmp_path):
    policy_path = tmp_path / "policy.json"
    write_tiny_policy(policy_path)
    source = tmp_path / "source.jsonl"
    raw_positions = [position(index) for index in range(4)]
    write_jsonl(source, raw_positions)
    normalized = tmp_path / "normalized.jsonl"
    publish_normalized([source], normalized, tmp_path / "normalized-manifest.json")
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
        json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()
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


def test_score_prefilter_is_deterministic_and_applies_thresholds(tmp_path):
    fixture = create_score_prefilter_fixture(
        tmp_path / "fixture", [29.0, 30.0, 50.0, 80.0, 81.0]
    )
    output = tmp_path / "prefiltered.jsonl"
    manifest_path = tmp_path / "prefiltered-manifest.json"
    first = score_prefilter_positions(
        normalized_path=fixture["normalized"],
        query_manifest_path=fixture["query_manifest"],
        analysis_path=fixture["analysis"],
        output_path=output,
        manifest_path=manifest_path,
    )
    second = score_prefilter_positions(
        normalized_path=fixture["normalized"],
        query_manifest_path=fixture["query_manifest"],
        analysis_path=fixture["analysis"],
        output_path=output,
        manifest_path=manifest_path,
    )
    assert first == second
    assert first["contract"] == SCORE_PREFILTER_CONTRACT
    assert manifest_path.read_text(encoding="utf-8") == canonical_json(first) + "\n"
    selected = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    selected_ids = [row["semanticSha256"] for row in selected]
    expected_ids = sorted(
        record_id
        for record_id, score in fixture["score_by_id"].items()
        if score >= 30.0
    )
    assert selected_ids == expected_ids
    assert first["selected_count"] == len(expected_ids)
    assert first["selected_ids_sha256"] == canonical_sha256(expected_ids)
    assert all("labels" not in row for row in selected)

    bounded_output = tmp_path / "bounded.jsonl"
    bounded = score_prefilter_positions(
        normalized_path=fixture["normalized"],
        query_manifest_path=fixture["query_manifest"],
        analysis_path=fixture["analysis"],
        output_path=bounded_output,
        manifest_path=tmp_path / "bounded-manifest.json",
        minimum_score=50.0,
        maximum_score=80.0,
    )
    bounded_ids = [
        json.loads(line)["semanticSha256"]
        for line in bounded_output.read_text(encoding="utf-8").splitlines()
    ]
    assert bounded_ids == sorted(
        record_id
        for record_id, score in fixture["score_by_id"].items()
        if 50.0 <= score <= 80.0
    )
    assert bounded["minimum_score"] == 50.0
    assert bounded["maximum_score"] == 80.0


def test_score_prefilter_rejects_invalid_bounds_and_empty_selection(tmp_path):
    fixture = create_score_prefilter_fixture(tmp_path / "fixture", [40.0])
    arguments = {
        "normalized_path": fixture["normalized"],
        "query_manifest_path": fixture["query_manifest"],
        "analysis_path": fixture["analysis"],
        "output_path": tmp_path / "output.jsonl",
        "manifest_path": tmp_path / "manifest.json",
    }
    with pytest.raises(ValueError, match="finite"):
        score_prefilter_positions(**arguments, minimum_score=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        score_prefilter_positions(**arguments, maximum_score=float("inf"))
    with pytest.raises(ValueError, match="at least"):
        score_prefilter_positions(**arguments, minimum_score=50.0, maximum_score=49.0)
    with pytest.raises(ValueError, match="selected no positions"):
        score_prefilter_positions(**arguments, minimum_score=100.0)


def test_score_prefilter_rejects_output_inside_immutable_query_bundle(tmp_path):
    fixture = create_score_prefilter_fixture(tmp_path / "fixture", [40.0])
    with pytest.raises(ValueError, match="immutable query bundle"):
        score_prefilter_positions(
            normalized_path=fixture["normalized"],
            query_manifest_path=fixture["query_manifest"],
            analysis_path=fixture["analysis"],
            output_path=fixture["query_manifest"].parent / "selected.jsonl",
            manifest_path=tmp_path / "selected-manifest.json",
        )


def test_score_prefilter_accepts_merge_analysis_receipt(tmp_path):
    fixture = create_score_prefilter_fixture(
        tmp_path / "fixture", [20.0, 40.0, 80.0]
    )
    query_path = fixture["query_path"]
    shard_dir = tmp_path / "shards"
    split_manifest = split_queries(query_path, shard_dir, shard_count=1)
    shard_query = shard_dir / split_manifest["shards"][0]["path"]
    shard_rows = [
        json.loads(line)
        for line in shard_query.read_text(encoding="utf-8").splitlines()
    ]
    shard_output = tmp_path / "standard-200-shard.jsonl"
    write_jsonl(
        shard_output,
        [
            analysis_record(
                row["id"], fixture["score_by_id"][row["id"]], visits=200
            )
            for row in shard_rows
        ],
    )
    write_analysis_manifest(
        shard_output,
        query=shard_query,
        katago=tmp_path / "fixture" / "katago",
        config=tmp_path / "fixture" / "analysis.cfg",
        model=fixture["model"],
        row_count=len(shard_rows),
    )
    merged = tmp_path / "standard-200-merged.jsonl"
    merge_analysis(
        query_path=query_path,
        split_manifest_path=shard_dir / "manifest.json",
        shard_outputs=[shard_output],
        output=merged,
    )

    manifest = score_prefilter_positions(
        normalized_path=fixture["normalized"],
        query_manifest_path=fixture["query_manifest"],
        analysis_path=merged,
        output_path=tmp_path / "selected.jsonl",
        manifest_path=tmp_path / "selected-manifest.json",
    )
    assert manifest["selected_count"] == 2


@pytest.mark.parametrize("changed", ["query", "result", "model"])
def test_score_prefilter_rejects_changed_provenance(tmp_path, changed):
    fixture = create_score_prefilter_fixture(tmp_path / changed, [40.0, 50.0])
    if changed == "query":
        fixture["query_path"].chmod(0o644)
        fixture["query_path"].write_bytes(fixture["query_path"].read_bytes() + b"\n")
    elif changed == "result":
        fixture["analysis"].write_bytes(fixture["analysis"].read_bytes() + b"\n")
    else:
        fixture["model"].write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="changed|provenance|artifact"):
        score_prefilter_positions(
            normalized_path=fixture["normalized"],
            query_manifest_path=fixture["query_manifest"],
            analysis_path=fixture["analysis"],
            output_path=tmp_path / f"{changed}-output.jsonl",
            manifest_path=tmp_path / f"{changed}-manifest.json",
        )


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
            json.loads(line) for line in kwargs["stdin"].read().decode().splitlines()
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
    powered = create_sharded_analysis_fixture(tmp_path / "powered", powered=True)
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

    other = create_sharded_analysis_fixture(tmp_path / "other", visits=800)
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


def test_labeling_bundle_merge_is_deterministic_and_sorted(tmp_path):
    policy_hash = "1" * 64
    reference_hash = "2" * 64
    first_bundle = tmp_path / "first"
    second_bundle = tmp_path / "second"
    write_labeling_bundle(
        first_bundle,
        [automatic_row(3, "lead-40")],
        [review_row(1)],
        policy_hash=policy_hash,
        reference_model_hash=reference_hash,
    )
    write_labeling_bundle(
        second_bundle,
        [automatic_row(2, "ordinary")],
        [review_row(0)],
        policy_hash=policy_hash,
        reference_model_hash=reference_hash,
    )
    first_output = tmp_path / "combined-first"
    second_output = tmp_path / "combined-second"
    first = merge_labeling_bundles([second_bundle, first_bundle], first_output)
    second = merge_labeling_bundles([first_bundle, second_bundle], second_output)
    assert first == second
    assert first["contract"] == COMBINED_LABELING_CONTRACT
    assert first["source_bundle_count"] == 2
    assert [source["path"] for source in first["source_bundles"]] == sorted(
        [str(first_bundle.resolve()), str(second_bundle.resolve())]
    )
    for name in ("auto-labeled.jsonl", "review-queue.jsonl", "manifest.json"):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()
    auto_ids = [
        row["curation"]["semanticSha256"]
        for row in (
            json.loads(line)
            for line in (first_output / "auto-labeled.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ]
    review_ids = [
        row["semantic_sha256"]
        for row in (
            json.loads(line)
            for line in (first_output / "review-queue.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ]
    assert auto_ids == sorted(auto_ids)
    assert review_ids == sorted(review_ids)


def test_labeling_bundle_merge_rejects_cross_bundle_semantic_duplicates(tmp_path):
    policy_hash = "1" * 64
    reference_hash = "2" * 64
    write_labeling_bundle(
        tmp_path / "first",
        [automatic_row(0, "ordinary")],
        [],
        policy_hash=policy_hash,
        reference_model_hash=reference_hash,
    )
    write_labeling_bundle(
        tmp_path / "second",
        [],
        [review_row(0)],
        policy_hash=policy_hash,
        reference_model_hash=reference_hash,
    )
    with pytest.raises(ValueError, match="semantic duplicate"):
        merge_labeling_bundles(
            [tmp_path / "first", tmp_path / "second"],
            tmp_path / "combined",
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("policy_hash", "policy hash"),
        ("reference_model_hash", "reference model hash"),
        ("stability_margin", "stability margin"),
    ],
)
def test_labeling_bundle_merge_rejects_mismatched_coordinates(
    tmp_path, field, expected
):
    first_values = {
        "policy_hash": "1" * 64,
        "reference_model_hash": "2" * 64,
        "stability_margin": 5.0,
    }
    second_values = dict(first_values)
    second_values[field] = {
        "policy_hash": "3" * 64,
        "reference_model_hash": "4" * 64,
        "stability_margin": 6.0,
    }[field]
    write_labeling_bundle(
        tmp_path / "first",
        [automatic_row(0, "ordinary")],
        [],
        **first_values,
    )
    write_labeling_bundle(
        tmp_path / "second",
        [automatic_row(1, "ordinary")],
        [],
        **second_values,
    )
    with pytest.raises(ValueError, match=expected):
        merge_labeling_bundles(
            [tmp_path / "first", tmp_path / "second"],
            tmp_path / "combined",
        )


def test_finalize_accepts_a_merged_labeling_bundle(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy = write_tiny_policy(policy_path)
    auto_rows = []
    index = 0
    for label, count in (
        ("ordinary", 3),
        ("lead-40", 2),
        ("lead-80", 2),
    ):
        for _ in range(count):
            auto_rows.append(automatic_row(index, label))
            index += 1
    review = review_row(index)
    policy_hash = canonical_sha256(policy)
    reference_hash = "2" * 64
    write_labeling_bundle(
        tmp_path / "first",
        auto_rows[:4],
        [],
        policy_hash=policy_hash,
        reference_model_hash=reference_hash,
    )
    write_labeling_bundle(
        tmp_path / "second",
        auto_rows[4:],
        [review],
        policy_hash=policy_hash,
        reference_model_hash=reference_hash,
    )
    combined = tmp_path / "combined"
    merge_labeling_bundles([tmp_path / "second", tmp_path / "first"], combined)
    decisions = tmp_path / "decisions.jsonl"
    write_jsonl(
        decisions,
        [
            {
                "semantic_sha256": review["semantic_sha256"],
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
        ],
    )
    manifest = finalize_reviewed_bank(
        auto_path=combined / "auto-labeled.jsonl",
        review_queue_path=combined / "review-queue.jsonl",
        decisions_path=decisions,
        labeling_manifest_path=combined / "manifest.json",
        policy_path=policy_path,
        output_path=tmp_path / "reviewed.jsonl",
        manifest_path=tmp_path / "reviewed-manifest.json",
    )
    assert manifest["row_count"] == 8
    assert manifest["label_counts"]["exploitability"] == 1


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
    wrong_policy_manifest["manifest_sha256"] = canonical_sha256(wrong_policy_manifest)
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
