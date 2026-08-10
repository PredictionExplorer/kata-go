import json
from pathlib import Path

import pytest

from risk_score.consensus_prefilter import (
    PREFILTER_CONTRACT,
    PREFILTER_ROLES,
    main,
    prefilter_consensus_sources,
)
from risk_score.curate_position_bank import _normalized_positions, publish_normalized
from risk_score.position_samples import canonical_json, file_sha256


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


def analysis_files(tmp_path, position_ids, values, *, mismatched_model=False):
    result = {}
    for role in PREFILTER_ROLES:
        safe = role.replace("/", "-")
        query = tmp_path / f"{safe}.queries.jsonl"
        query.write_text(
            "".join(
                canonical_json({"id": record_id}) + "\n"
                for record_id in position_ids
            ),
            encoding="utf-8",
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
            "contract": "risk-score-position-analysis-run-v1",
            "output_path": str(output.resolve()),
            "output_sha256": file_sha256(output),
            "query_path": str(query.resolve()),
            "query_sha256": file_sha256(query),
            "model_sha256": model_hash,
            "katago_sha256": "d" * 64,
            "config_sha256": "e" * 64,
            "mode": mode,
        }
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
    publish_normalized(
        [tmp_path / "source.jsonl"], normalized, normalized_manifest
    )
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


def test_prefilter_selects_only_high_agreement_sources(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(tmp_path, ids, values)
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


def test_prefilter_rejects_changed_model_identity(tmp_path):
    normalized, ids, values = fixture(tmp_path)
    analyses = analysis_files(
        tmp_path, ids, values, mismatched_model=True
    )

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
    analyses = analysis_files(tmp_path, ids, values)
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
    analyses = analysis_files(tmp_path, ids, values)
    output = tmp_path / "selected.jsonl"
    manifest = tmp_path / "selected.manifest.json"
    argv = [str(normalized), "--label", "lead-80"]
    for role, path in analyses.items():
        argv.extend(["--analysis", f"{role}={path}"])
    argv.extend(["--output", str(output), "--manifest", str(manifest)])

    assert main(argv) == 0
    assert output.is_file()
    assert json.loads(manifest.read_text())["contract"] == PREFILTER_CONTRACT
