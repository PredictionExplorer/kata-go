import json
import shutil
from pathlib import Path

import pytest

from risk_score.board_symmetry import apply_symmetry, transform_gtp_location
from risk_score.build_evaluation_suites import build_evaluation_suites
from risk_score.curate_position_bank import (
    CONSENSUS_COMBINED_LABELING_CONTRACT,
    CONSENSUS_FINAL_MANIFEST_CONTRACT,
    CONSENSUS_LABELING_CONTRACT,
    CONSENSUS_QUERY_BUNDLE_CONTRACT,
    finalize_consensus_reviewed_bank,
    generate_consensus_query_bundle,
    label_positions_consensus,
    merge_consensus_labeling_bundles,
    parse_args,
    policy_pool_minima,
    publish_normalized,
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


def position(turn, *, x_size=5, y_size=5, board=None):
    if board is None:
        board = "X..../..O../...../...X./....."
    return {
        "xSize": x_size,
        "ySize": y_size,
        "board": board,
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": turn,
        "hintLoc": "null",
        "metadata": "fixture-source",
    }


def tiny_v3_policy(path):
    source = Path(__file__).parents[1] / "risk_score" / "promotion_policy_v3.json"
    policy = json.loads(source.read_text(encoding="utf-8"))
    stages = policy["evaluation_stages"]
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
            "lead_40_color_pairs": 1,
            "lead_80_color_pairs": 1,
        }
    )
    path.write_text(canonical_json(policy) + "\n", encoding="utf-8")
    return policy


def assets(root):
    root.mkdir()
    katago = root / "katago"
    config = root / "analysis.cfg"
    original = root / "original.bin.gz"
    champion = root / "champion.bin.gz"
    policy = root / "policy.json"
    katago.write_bytes(b"katago")
    config.write_text(DETERMINISTIC_CONFIG, encoding="utf-8")
    original.write_bytes(b"immutable-original")
    champion.write_bytes(b"frozen-champion")
    tiny_v3_policy(policy)
    return {
        "katago": katago,
        "config": config,
        "original": original,
        "champion": champion,
        "policy": policy,
    }


def analysis_record(record_id, score, move, visits, *, prior=0.2, selfplay=None):
    if selfplay is None:
        selfplay = score + 1.0
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
                "scoreSelfplay": selfplay,
                "scoreStdev": 10.0,
            }
        ],
    }


def write_analysis_manifest(path, *, query, assets, model, row_count):
    manifest = {
        "schema_version": 1,
        "contract": "risk-score-position-analysis-run-v1",
        "argv": ["katago", "analysis"],
        "katago_sha256": file_sha256(assets["katago"]),
        "config_sha256": file_sha256(assets["config"]),
        "model_sha256": file_sha256(assets[model]),
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


def prepare_analysis_bundle(root, shared_assets, cases):
    root.mkdir()
    source = root / "source.jsonl"
    write_jsonl(source, [position(turn) for turn, _, _ in cases])
    normalized = root / "normalized.jsonl"
    publish_normalized([source], normalized, root / "normalized-manifest.json")
    query_dir = root / "queries"
    query_manifest = generate_consensus_query_bundle(
        normalized,
        query_dir,
        shared_assets["katago"],
        shared_assets["config"],
        shared_assets["original"],
        shared_assets["champion"],
        shared_assets["policy"],
    )
    normalized_rows = [
        json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines()
    ]
    case_by_turn = {turn: (score, kind) for turn, score, kind in cases}
    case_by_hash = {
        row["semanticSha256"]: (
            row["initialTurnNumber"],
            *case_by_turn[row["initialTurnNumber"]],
        )
        for row in normalized_rows
    }
    entry_by_id = {
        entry["query_id"]: (
            orbit["canonical_semantic_sha256"],
            entry["symmetry"],
            orbit["x_size"],
            orbit["y_size"],
        )
        for orbit in query_manifest["orbits"]
        for entry in orbit["entries"]
    }
    analysis_paths = {}
    for role, artifact in query_manifest["queries"].items():
        model = artifact["model"]
        mode = artifact["mode"]
        visits = artifact["visits"]
        rows = []
        for query_id in sorted(entry_by_id):
            semantic_hash, symmetry, x_size, y_size = entry_by_id[query_id]
            turn, base_score, kind = case_by_hash[semantic_hash]
            score = base_score
            canonical_move = "B4"
            prior = 0.2
            selfplay = score + 1.0
            if kind == "visit" and visits == 8000:
                score += 10.0
                selfplay = score + 1.0
            elif kind == "model" and model == "champion":
                score += 10.0
                selfplay = score + 1.0
            elif kind == "symmetry" and symmetry == 0:
                score += 10.0
                selfplay = score + 1.0
            elif kind == "top-move" and model == "champion":
                canonical_move = "C3"
            elif kind == "specialized" and mode == "powered":
                prior = 0.01
                selfplay = score + 20.0
            move = transform_gtp_location(canonical_move, x_size, y_size, symmetry)
            rows.append(
                analysis_record(
                    query_id,
                    score,
                    move,
                    visits,
                    prior=prior,
                    selfplay=selfplay,
                )
            )
            assert turn >= 0
        path = root / f"{role.replace('/', '-')}.jsonl"
        write_jsonl(path, rows)
        query_path = query_dir / artifact["path"]
        write_analysis_manifest(
            path,
            query=query_path,
            assets=shared_assets,
            model=model,
            row_count=len(rows),
        )
        analysis_paths[role] = path
    return {
        "normalized": normalized,
        "query_dir": query_dir,
        "query_manifest": query_manifest,
        "analysis_paths": analysis_paths,
    }


def label_prepared(root, prepared):
    labeling = root / "labeling"
    manifest = label_positions_consensus(
        normalized_path=prepared["normalized"],
        query_manifest_path=prepared["query_dir"] / "manifest.json",
        analysis_paths=prepared["analysis_paths"],
        output_dir=labeling,
    )
    return labeling, manifest


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_consensus_queries_expand_shape_preserving_distinct_orbits(tmp_path):
    shared = assets(tmp_path / "assets")
    source = tmp_path / "source.jsonl"
    write_jsonl(
        source,
        [
            position(1),
            position(
                2,
                x_size=5,
                y_size=3,
                board="X..../.O.../.....",
            ),
            position(3, board="...../...../...../...../....."),
        ],
    )
    normalized = tmp_path / "normalized.jsonl"
    publish_normalized([source], normalized, tmp_path / "normalized-manifest.json")
    output = tmp_path / "queries"
    manifest = generate_consensus_query_bundle(
        normalized,
        output,
        shared["katago"],
        shared["config"],
        shared["original"],
        shared["champion"],
        shared["policy"],
    )

    assert manifest["contract"] == CONSENSUS_QUERY_BUNDLE_CONTRACT
    assert len(manifest["queries"]) == 8
    assert {
        (item["model"], item["mode"], item["visits"])
        for item in manifest["queries"].values()
    } == {
        (model, mode, visits)
        for model in ("original", "champion")
        for mode in ("standard", "powered")
        for visits in (2000, 8000)
    }
    assert sorted(orbit["distinct_symmetry_count"] for orbit in manifest["orbits"]) == [
        1,
        4,
        8,
    ]
    assert manifest["expanded_position_count"] == 13
    for artifact in manifest["queries"].values():
        rows = load_jsonl(output / artifact["path"])
        assert len(rows) == 13
        assert all(row["maxVisits"] == artifact["visits"] for row in rows)
        assert all(
            row["overrideSettings"]["useScoreMaximizingUtility"] is artifact["powered"]
            for row in rows
        )

    shared["champion"].write_bytes(shared["original"].read_bytes())
    with pytest.raises(ValueError, match="distinct hashes"):
        generate_consensus_query_bundle(
            normalized,
            tmp_path / "same-model-queries",
            shared["katago"],
            shared["config"],
            shared["original"],
            shared["champion"],
            shared["policy"],
        )


def test_consensus_queries_reject_symmetry_duplicate_sources(tmp_path):
    shared = assets(tmp_path / "assets")
    first = position(1)
    second = apply_symmetry(first, 2)
    second["initialTurnNumber"] = first["initialTurnNumber"]
    source = tmp_path / "source.jsonl"
    write_jsonl(source, [first, second])
    normalized = tmp_path / "normalized.jsonl"
    publish_normalized([source], normalized, tmp_path / "normalized-manifest.json")

    with pytest.raises(ValueError, match="share a symmetry orbit"):
        generate_consensus_query_bundle(
            normalized,
            tmp_path / "queries",
            shared["katago"],
            shared["config"],
            shared["original"],
            shared["champion"],
            shared["policy"],
        )


def test_consensus_labeling_accepts_only_unanimous_and_rejects_each_reason(
    tmp_path,
):
    shared = assets(tmp_path / "assets")
    cases = [
        (0, 0.0, "accepted"),
        (1, 0.0, "visit"),
        (2, 0.0, "model"),
        (3, 0.0, "symmetry"),
        (4, 0.0, "top-move"),
        (5, 40.0, "threshold"),
        (6, -50.0, "unclassifiable"),
        (7, 0.0, "specialized"),
        (8, 50.0, "accepted"),
        (9, 90.0, "accepted"),
    ]
    prepared = prepare_analysis_bundle(tmp_path / "fixture", shared, cases)
    labeling, manifest = label_prepared(tmp_path / "fixture", prepared)
    accepted = load_jsonl(labeling / "machine-labeled.jsonl")
    rejected = load_jsonl(labeling / "rejected.jsonl")

    assert manifest["contract"] == CONSENSUS_LABELING_CONTRACT
    assert {row["labels"][0] for row in accepted} == {
        "ordinary",
        "lead-40",
        "lead-80",
    }
    assert {row["curation"]["classification"] for row in accepted} == {
        "machine-reviewed"
    }
    by_turn = {row["position"]["initialTurnNumber"]: row["reasons"] for row in rejected}
    assert "visit_unstable" in by_turn[1]
    assert "model_disagreement" in by_turn[2]
    assert "symmetry_disagreement" in by_turn[3]
    assert "top_move_disagreement" in by_turn[4]
    assert "threshold_boundary" in by_turn[5]
    assert "label_unclassifiable" in by_turn[6]
    assert "specialized_signal" in by_turn[7]
    assert manifest["machine_labeled_count"] == 3
    assert manifest["rejected_count"] == 7


def test_consensus_labeling_aborts_on_execution_provenance_change(tmp_path):
    shared = assets(tmp_path / "assets")
    prepared = prepare_analysis_bundle(
        tmp_path / "fixture", shared, [(0, 0.0, "accepted")]
    )
    role = sorted(prepared["analysis_paths"])[0]
    execution_path = Path(str(prepared["analysis_paths"][role]) + ".manifest.json")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["config_sha256"] = "0" * 64
    execution.pop("manifest_sha256")
    execution["manifest_sha256"] = canonical_sha256(execution)
    execution_path.write_text(canonical_json(execution) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        label_positions_consensus(
            normalized_path=prepared["normalized"],
            query_manifest_path=prepared["query_dir"] / "manifest.json",
            analysis_paths=prepared["analysis_paths"],
            output_dir=tmp_path / "not-published",
        )
    assert not (tmp_path / "not-published").exists()


def test_consensus_labeling_rejects_policy_margin_override(tmp_path):
    shared = assets(tmp_path / "assets")
    prepared = prepare_analysis_bundle(
        tmp_path / "fixture", shared, [(0, 0.0, "accepted")]
    )

    with pytest.raises(ValueError, match="frozen v3 policy"):
        label_positions_consensus(
            normalized_path=prepared["normalized"],
            query_manifest_path=prepared["query_dir"] / "manifest.json",
            analysis_paths=prepared["analysis_paths"],
            output_dir=tmp_path / "not-published",
            stability_margin=6.0,
        )
    assert not (tmp_path / "not-published").exists()


def test_finalization_recomputes_labels_from_bound_analysis(tmp_path):
    shared = assets(tmp_path / "assets")
    prepared = prepare_analysis_bundle(
        tmp_path / "fixture", shared, [(0, 0.0, "accepted")]
    )
    labeling, _ = label_prepared(tmp_path / "fixture", prepared)
    machine_path = labeling / "machine-labeled.jsonl"
    machine_path.chmod(0o644)
    rows = load_jsonl(machine_path)
    rows[0]["labels"] = ["lead-40"]
    write_jsonl(machine_path, rows)
    manifest_path = labeling / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["machine_labeled_sha256"] = file_sha256(machine_path)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="bound analysis results"):
        finalize_consensus_reviewed_bank(
            machine_labeled_path=machine_path,
            rejected_path=labeling / "rejected.jsonl",
            labeling_manifest_path=manifest_path,
            policy_path=shared["policy"],
            output_path=tmp_path / "forged.jsonl",
            manifest_path=tmp_path / "forged.manifest.json",
        )


def test_consensus_merge_and_finalize_need_no_decisions(tmp_path):
    shared = assets(tmp_path / "assets")
    first_prepared = prepare_analysis_bundle(
        tmp_path / "first",
        shared,
        [
            (100, 0.0, "accepted"),
            (101, 50.0, "accepted"),
            (102, 90.0, "accepted"),
            (103, 0.0, "accepted"),
            (104, 50.0, "accepted"),
            (105, 90.0, "accepted"),
            (106, 0.0, "accepted"),
            (107, 50.0, "accepted"),
            (108, 90.0, "accepted"),
        ],
    )
    first, _ = label_prepared(tmp_path / "first", first_prepared)
    second_prepared = prepare_analysis_bundle(
        tmp_path / "second", shared, [(200, 40.0, "threshold")]
    )
    second, _ = label_prepared(tmp_path / "second", second_prepared)
    combined = tmp_path / "combined"
    combined_manifest = merge_consensus_labeling_bundles([second, first], combined)

    assert combined_manifest["contract"] == CONSENSUS_COMBINED_LABELING_CONTRACT
    assert combined_manifest["machine_labeled_count"] == 9
    assert combined_manifest["rejected_count"] == 1
    output = tmp_path / "reviewed.jsonl"
    final_manifest_path = tmp_path / "reviewed.manifest.json"
    final_manifest = finalize_consensus_reviewed_bank(
        machine_labeled_path=combined / "machine-labeled.jsonl",
        rejected_path=combined / "rejected.jsonl",
        labeling_manifest_path=combined / "manifest.json",
        policy_path=shared["policy"],
        output_path=output,
        manifest_path=final_manifest_path,
    )

    assert final_manifest["schema_version"] == 2
    assert final_manifest["contract"] == CONSENSUS_FINAL_MANIFEST_CONTRACT
    assert final_manifest["review_mode"] == "machine-consensus"
    assert final_manifest["rejected_count"] == 1
    assert final_manifest["required_minima"] == {
        "ordinary": 3,
        "lead-40": 3,
        "lead-80": 3,
    }
    assert {row["curation"]["classification"] for row in load_jsonl(output)} == {
        "machine-reviewed"
    }
    suites = build_evaluation_suites(
        [output],
        tmp_path / "suites",
        seed="consensus-finalization",
        policy_path=shared["policy"],
        curation_manifest_paths=[final_manifest_path],
    )
    assert suites.manifest["machineReviewOnly"] is True
    with pytest.raises(ValueError, match="one curation manifest per source"):
        build_evaluation_suites(
            [output],
            tmp_path / "suites-without-provenance",
            seed="missing-consensus-finalization",
            policy_path=shared["policy"],
        )
    args = parse_args(
        [
            "finalize-consensus",
            "--machine-labeled",
            str(combined / "machine-labeled.jsonl"),
            "--rejected",
            str(combined / "rejected.jsonl"),
            "--labeling-manifest",
            str(combined / "manifest.json"),
            "--policy",
            str(shared["policy"]),
            "--output",
            str(output),
            "--manifest",
            str(final_manifest_path),
        ]
    )
    assert not hasattr(args, "decisions")

    collision = tmp_path / "colliding-final-output"
    with pytest.raises(ValueError, match="must be distinct"):
        finalize_consensus_reviewed_bank(
            machine_labeled_path=combined / "machine-labeled.jsonl",
            rejected_path=combined / "rejected.jsonl",
            labeling_manifest_path=combined / "manifest.json",
            policy_path=shared["policy"],
            output_path=collision,
            manifest_path=collision,
        )
    assert not collision.exists()

    forged_bundle = tmp_path / "forged-combined-bundle"
    shutil.copytree(combined, forged_bundle)
    forged_machine = forged_bundle / "machine-labeled.jsonl"
    forged_machine.chmod(0o644)
    forged_rows = load_jsonl(forged_machine)
    forged_rows[0]["labels"] = [
        "ordinary"
        if forged_rows[0]["labels"] != ["ordinary"]
        else "lead-80"
    ]
    write_jsonl(forged_machine, forged_rows)
    forged_manifest_path = forged_bundle / "manifest.json"
    forged_manifest_path.chmod(0o644)
    forged_manifest = json.loads(
        forged_manifest_path.read_text(encoding="utf-8")
    )
    forged_manifest["machine_labeled_sha256"] = file_sha256(
        forged_machine
    )
    forged_manifest.pop("manifest_sha256")
    forged_manifest["manifest_sha256"] = canonical_sha256(
        forged_manifest
    )
    forged_manifest_path.write_text(
        canonical_json(forged_manifest) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source-bundle union"):
        finalize_consensus_reviewed_bank(
            machine_labeled_path=forged_machine,
            rejected_path=forged_bundle / "rejected.jsonl",
            labeling_manifest_path=forged_manifest_path,
            policy_path=shared["policy"],
            output_path=tmp_path / "forged-combined.jsonl",
            manifest_path=tmp_path / "forged-combined.manifest.json",
        )

    stripped_bundle = tmp_path / "stripped-bundle"
    shutil.copytree(combined, stripped_bundle)
    combined_path = stripped_bundle / "manifest.json"
    combined_path.chmod(0o644)
    stripped = json.loads(combined_path.read_text(encoding="utf-8"))
    stripped.pop("source_bundles")
    stripped.pop("manifest_sha256")
    stripped["manifest_sha256"] = canonical_sha256(stripped)
    combined_path.write_text(
        canonical_json(stripped) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ancestry"):
        finalize_consensus_reviewed_bank(
            machine_labeled_path=stripped_bundle / "machine-labeled.jsonl",
            rejected_path=stripped_bundle / "rejected.jsonl",
            labeling_manifest_path=combined_path,
            policy_path=shared["policy"],
            output_path=tmp_path / "stripped.jsonl",
            manifest_path=tmp_path / "stripped.manifest.json",
        )

    legacy_policy = json.loads(
        (
            Path(__file__).parents[1] / "risk_score" / "promotion_policy_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert set(policy_pool_minima(legacy_policy)) > {
        "ordinary",
        "lead-40",
        "lead-80",
    }
