import json
from pathlib import Path

import pytest

from risk_score.consensus_prefilter import (
    PREFILTER_CONTRACT,
    PREFILTER_QUERY_BUNDLE_CONTRACT,
    PREFILTER_ROLES,
    generate_prefilter_query_bundle,
    main,
    prefilter_consensus_sources,
    validate_prefilter_artifact,
)
from risk_score.curate_position_bank import _normalized_positions, publish_normalized
from risk_score.position_samples import (
    build_analysis_query,
    canonical_json,
    canonical_sha256,
    file_sha256,
)


REPO = Path(__file__).resolve().parents[2]


def position(turn, board):
    return {
        "xSize": 5,
        "ySize": 5,
        "board": board,
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": turn,
        "hintLoc": "null",
        "metadata": "prefilter-fixture",
    }


def analysis_record(record_id, score, move="D4"):
    return {
        "id": record_id,
        "rootInfo": {
            "scoreLead": score,
            "winrate": 0.75,
            "visits": 2000,
        },
        "moveInfos": [
            {
                "move": move,
                "order": 0,
                "visits": 2000,
                "prior": 0.2,
                "scoreLead": score,
                "scoreSelfplay": score,
                "scoreStdev": 8.0,
            }
        ],
    }


def write_jsonl(path, rows):
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_json(path, value):
    if path.exists():
        path.chmod(0o644)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def analysis_files(tmp_path, normalized, values, *, mismatched_model=False):
    positions = _normalized_positions(normalized)
    position_ids = [row["semanticSha256"] for row in positions]
    result = {}
    for role in PREFILTER_ROLES:
        safe = role.replace("/", "-")
        query = tmp_path / f"{safe}.queries.jsonl"
        write_jsonl(
            query,
            [
                build_analysis_query(
                    row,
                    query_id=row["semanticSha256"],
                    max_visits=2000,
                    powered=role.endswith("/powered-2000"),
                )
                for row in positions
            ],
        )
        output = tmp_path / f"{safe}.results.jsonl"
        write_jsonl(
            output,
            [
                analysis_record(
                    record_id,
                    values[record_id][role][0],
                    values[record_id][role][1],
                )
                for record_id in position_ids
            ],
        )
        model, mode = role.split("/", 1)
        model_hash = ("a" if model == "original" else "b") * 64
        if mismatched_model and role == "champion/powered-2000":
            model_hash = "c" * 64
        manifest = {
            "schema_version": 1,
            "contract": "risk-score-position-analysis-run-v1",
            "output_path": str(output.resolve()),
            "output_sha256": file_sha256(output),
            "query_path": str(query.resolve()),
            "query_sha256": file_sha256(query),
            "model_sha256": model_hash,
            "katago_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "mode": mode,
            "row_count": len(position_ids),
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        Path(str(output) + ".manifest.json").write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        result[role] = output
    return result


def fixture(tmp_path):
    normalized = tmp_path / "normalized.jsonl"
    normalized_manifest = tmp_path / "normalized.manifest.json"
    write_jsonl(
        tmp_path / "source.jsonl",
        [
            position(10, "X..../..O../...../...X./....."),
            position(20, "XX.../..O../...../...X./....."),
            position(30, "XXX../..O../...../...X./....."),
        ],
    )
    publish_normalized([tmp_path / "source.jsonl"], normalized, normalized_manifest)
    ids = [row["semanticSha256"] for row in _normalized_positions(normalized)]
    good = {role: (95.0, "D4") for role in PREFILTER_ROLES}
    wrong_move = {role: (96.0, "D4") for role in PREFILTER_ROLES}
    wrong_move["champion/powered-2000"] = (96.0, "C3")
    wrong_label = {role: (70.0, "D4") for role in PREFILTER_ROLES}
    values = {
        ids[0]: good,
        ids[1]: wrong_move,
        ids[2]: wrong_label,
    }
    return normalized, ids, values


def publish_fixture(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(tmp_path, normalized, values)
    output = tmp_path / "selected.jsonl"
    manifest_path = tmp_path / "selected.manifest.json"
    manifest = prefilter_consensus_sources(
        normalized_path=normalized,
        analysis_paths=analyses,
        label="lead-80",
        output_path=output,
        manifest_path=manifest_path,
    )
    return normalized, ids, analyses, output, manifest_path, manifest


def refresh_analysis_bindings(manifest_path, role, output):
    execution_path = Path(str(output) + ".manifest.json")
    execution = json.loads(execution_path.read_text())
    execution.pop("manifest_sha256")
    execution["output_sha256"] = file_sha256(output)
    execution["query_sha256"] = file_sha256(Path(execution["query_path"]))
    execution["manifest_sha256"] = canonical_sha256(execution)
    write_json(execution_path, execution)

    manifest = json.loads(manifest_path.read_text())
    manifest.pop("manifest_sha256")
    manifest["analyses"][role]["sha256"] = file_sha256(output)
    manifest["analyses"][role]["query_sha256"] = execution["query_sha256"]
    manifest["analyses"][role]["manifest_sha256"] = file_sha256(execution_path)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(manifest_path, manifest)


def test_prefilter_selects_only_high_agreement_sources(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(tmp_path, normalized, values)
    output = tmp_path / "selected.jsonl"
    manifest_path = tmp_path / "selected.manifest.json"

    manifest = prefilter_consensus_sources(
        normalized_path=normalized,
        analysis_paths=analyses,
        label="lead-80",
        output_path=output,
        manifest_path=manifest_path,
    )

    selected = _normalized_positions(output)
    assert [row["semanticSha256"] for row in selected] == [ids[0]]
    assert manifest["contract"] == PREFILTER_CONTRACT
    assert manifest["advisory_only"] is True
    assert manifest["requires_full_machine_consensus"] is True
    assert manifest["selected"]["row_count"] == 1
    assert manifest["rejection_counts"]["top_move_disagreement"] == 1
    assert manifest["rejection_counts"]["label_disagreement"] == 1
    assert manifest["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert (
        validate_prefilter_artifact(
            manifest_path,
            expected_label="lead-80",
            expected_model_hashes={
                "original": "a" * 64,
                "champion": "b" * 64,
            },
        )
        == manifest
    )


def test_prefilter_publication_and_validation_are_deterministic(tmp_path):
    normalized, _, analyses, output, manifest_path, manifest = publish_fixture(tmp_path)
    original_bytes = manifest_path.read_bytes()

    repeated = prefilter_consensus_sources(
        normalized_path=normalized,
        analysis_paths=dict(reversed(list(analyses.items()))),
        label="lead-80",
        output_path=output,
        manifest_path=manifest_path,
    )

    assert repeated == manifest
    assert manifest_path.read_bytes() == original_bytes
    assert validate_prefilter_artifact(manifest_path) == manifest


def test_validate_prefilter_accepts_exact_legacy_manifest_only(tmp_path):
    _, _, _, _, manifest_path, manifest = publish_fixture(tmp_path)
    legacy = dict(manifest)
    legacy.pop("manifest_sha256")
    write_json(manifest_path, legacy)

    assert validate_prefilter_artifact(manifest_path) == legacy

    legacy["rejection_counts"] = {}
    write_json(manifest_path, legacy)
    with pytest.raises(ValueError, match="legacy .*fields"):
        validate_prefilter_artifact(manifest_path)


def test_validate_prefilter_requires_canonical_manifest_bytes(tmp_path):
    _, _, _, _, manifest_path, manifest = publish_fixture(tmp_path)
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical newline-terminated JSON"):
        validate_prefilter_artifact(manifest_path)


def test_validate_prefilter_rejects_bad_analysis_self_hash(tmp_path):
    _, _, analyses, _, manifest_path, _ = publish_fixture(tmp_path)
    role = "original/standard-2000"
    execution_path = Path(str(analyses[role]) + ".manifest.json")
    execution = json.loads(execution_path.read_text())
    execution["config_sha256"] = "f" * 64
    write_json(execution_path, execution)

    with pytest.raises(ValueError, match="analysis manifest self-hash is invalid"):
        validate_prefilter_artifact(manifest_path)


def test_validate_prefilter_rejects_fabricated_analysis_ids(tmp_path):
    _, _, analyses, _, manifest_path, _ = publish_fixture(tmp_path)
    role = "original/standard-2000"
    output = analyses[role]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    rows[0]["id"] = "fabricated"
    write_jsonl(output, rows)
    refresh_analysis_bindings(manifest_path, role, output)

    with pytest.raises(ValueError, match="IDs do not match normalized positions"):
        validate_prefilter_artifact(manifest_path)


def test_validate_prefilter_rejects_fabricated_query_ids(tmp_path):
    _, _, analyses, _, manifest_path, _ = publish_fixture(tmp_path)
    role = "original/standard-2000"
    output = analyses[role]
    execution = json.loads(Path(str(output) + ".manifest.json").read_text())
    query = Path(execution["query_path"])
    rows = [json.loads(line) for line in query.read_text().splitlines()]
    rows[0]["id"] = "fabricated"
    write_jsonl(query, rows)
    refresh_analysis_bindings(manifest_path, role, output)

    with pytest.raises(ValueError, match="exact IDs are invalid"):
        validate_prefilter_artifact(manifest_path)


def test_validate_prefilter_recomputes_changed_decision(tmp_path):
    _, ids, analyses, _, manifest_path, _ = publish_fixture(tmp_path)
    role = "original/standard-2000"
    output = analyses[role]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    changed = next(row for row in rows if row["id"] == ids[0])
    changed["rootInfo"]["scoreLead"] = 70.0
    changed["moveInfos"][0]["scoreLead"] = 70.0
    changed["moveInfos"][0]["scoreSelfplay"] = 70.0
    write_jsonl(output, rows)
    refresh_analysis_bindings(manifest_path, role, output)

    with pytest.raises(ValueError, match="selected JSONL bytes"):
        validate_prefilter_artifact(manifest_path)


def test_prefilter_can_publish_and_validate_empty_result(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    values[ids[0]] = {role: (70.0, "D4") for role in PREFILTER_ROLES}
    analyses = analysis_files(tmp_path, normalized, values)
    output = tmp_path / "selected.jsonl"
    manifest_path = tmp_path / "selected.manifest.json"

    with pytest.raises(ValueError, match="selected no positions"):
        prefilter_consensus_sources(
            normalized_path=normalized,
            analysis_paths=analyses,
            label="lead-80",
            output_path=output,
            manifest_path=manifest_path,
        )
    assert not output.exists()
    assert not manifest_path.exists()

    manifest = prefilter_consensus_sources(
        normalized_path=normalized,
        analysis_paths=analyses,
        label="lead-80",
        output_path=output,
        manifest_path=manifest_path,
        allow_empty=True,
    )

    assert output.read_bytes() == b""
    assert manifest["selected"] == {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "row_count": 0,
        "symmetry_orbit_count": 0,
    }
    assert validate_prefilter_artifact(manifest_path) == manifest


def test_prefilter_cli_allow_empty_preserves_select_syntax(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    values[ids[0]] = {role: (70.0, "D4") for role in PREFILTER_ROLES}
    analyses = analysis_files(tmp_path, normalized, values)
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "selected.manifest.json"
    argv = [str(normalized), "--label", "lead-80", "--allow-empty"]
    for role, path in analyses.items():
        argv.extend(["--analysis", f"{role}={path}"])
    argv.extend(["--output", str(output), "--manifest", str(manifest)])

    assert main(argv) == 0
    assert output.read_bytes() == b""
    assert json.loads(manifest.read_text())["selected"]["row_count"] == 0


def test_prefilter_rejects_changed_model_identity(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(tmp_path, normalized, values, mismatched_model=True)

    with pytest.raises(ValueError, match="one model each"):
        prefilter_consensus_sources(
            normalized_path=normalized,
            analysis_paths=analyses,
            label="lead-80",
            output_path=tmp_path / "selected.jsonl",
            manifest_path=tmp_path / "selected.manifest.json",
        )


def test_prefilter_requires_exact_roles(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(tmp_path, normalized, values)
    analyses.pop("champion/powered-2000")

    with pytest.raises(ValueError, match="exactly four"):
        prefilter_consensus_sources(
            normalized_path=normalized,
            analysis_paths=analyses,
            label="lead-80",
            output_path=tmp_path / "selected.jsonl",
            manifest_path=tmp_path / "selected.manifest.json",
        )


def test_prefilter_cli_publishes_manifest(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(tmp_path, normalized, values)
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "selected.manifest.json"
    argv = [str(normalized), "--label", "lead-80"]
    for role, path in analyses.items():
        argv.extend(["--analysis", f"{role}={path}"])
    argv.extend(["--output", str(output), "--manifest", str(manifest)])

    assert main(argv) == 0
    assert output.is_file()
    assert json.loads(manifest.read_text())["contract"] == PREFILTER_CONTRACT


def test_generate_prefilter_queries_is_canonical_and_model_bound(tmp_path):
    normalized, ids, _ = fixture(tmp_path)
    katago = tmp_path / "katago"
    model = tmp_path / "model.bin.gz"
    katago.write_bytes(b"katago")
    model.write_bytes(b"model")
    output = tmp_path / "query-bundle"
    config = REPO / "cpp/configs/risk_score/promotion_curation_analysis.cfg"
    policy = REPO / "python/risk_score/promotion_policy_v3.json"

    manifest = generate_prefilter_query_bundle(
        normalized_path=normalized,
        output_dir=output,
        katago_path=katago,
        config_path=config,
        model_path=model,
        policy_path=policy,
    )

    assert manifest["contract"] == PREFILTER_QUERY_BUNDLE_CONTRACT
    assert manifest["model_sha256"] == file_sha256(model)
    assert manifest["semantic_ids_sha256"]
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for role in ("standard-2000", "powered-2000"):
        rows = [
            json.loads(line)
            for line in (output / "queries" / f"{role}.jsonl").read_text().splitlines()
        ]
        assert [row["id"] for row in rows] == ids
        assert all(row["maxVisits"] == 2000 for row in rows)
        assert manifest["queries"][role]["row_count"] == len(ids)


def test_generate_prefilter_queries_cli(tmp_path):
    normalized, _, _ = fixture(tmp_path)
    katago = tmp_path / "katago"
    model = tmp_path / "model.bin.gz"
    katago.write_bytes(b"katago")
    model.write_bytes(b"model")
    output = tmp_path / "query-bundle"

    assert (
        main(
            [
                "generate-queries",
                str(normalized),
                "--output-dir",
                str(output),
                "--katago",
                str(katago),
                "--config",
                str(REPO / "cpp/configs/risk_score/promotion_curation_analysis.cfg"),
                "--model",
                str(model),
                "--policy",
                str(REPO / "python/risk_score/promotion_policy_v3.json"),
            ]
        )
        == 0
    )
    assert (output / "manifest.json").is_file()
