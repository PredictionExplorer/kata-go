#!/usr/bin/env python3
"""Assemble hash-bound runner outputs into controller promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from risk_score.evaluation_runner import (
    RUNNER_CONTRACT,
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_suite_manifest,
    resolve_manifest_cell,
)
from risk_score.paired_stats import (
    DEFAULT_POLICY_PATH,
    compute_paired_statistics,
    finalize_statistics_artifact,
    load_policy,
)

SCHEMA_VERSION = 1
EVIDENCE_CONTRACT = "risk-score-promotion-evidence-adapter-v1"
STAGE0_PROBE_CONTRACT = "risk-score-stage-0-probe-output-v1"
DISCOVERY_CONTRACT = "risk-score-derived-discovery-evidence-v1"
STAGE_EVIDENCE_CONTRACT = "risk-score-derived-stage-evidence-v1"
CELL_ORDER = (
    "powered_candidate_vs_champion",
    "powered_candidate_vs_original",
    "standard_candidate_vs_original",
    "lead_40",
    "lead_80",
)
CELL_COMPARISONS = {
    "powered_candidate_vs_champion": "candidate-vs-champion-powered",
    "powered_candidate_vs_original": "candidate-vs-original-powered",
    "standard_candidate_vs_original": "candidate-vs-original-standard",
    "lead_40": "candidate-vs-champion-powered-lead-40",
    "lead_80": "candidate-vs-champion-powered-lead-80",
}
CELL_SUITES = {
    "powered_candidate_vs_champion": "confirmation",
    "powered_candidate_vs_original": "confirmation",
    "standard_candidate_vs_original": "confirmation",
    "lead_40": "lead-40",
    "lead_80": "lead-80",
}
RISK_CELL_BINDINGS = {
    "final_20": "powered_candidate_vs_champion",
    "final_50": "powered_candidate_vs_champion",
    "high_confidence_loss": "powered_candidate_vs_champion",
    "lead_40_loss": "lead_40",
    "targeted_lead_40_suite_loss": "lead_40",
    "lead_80_loss": "lead_80",
    "targeted_lead_80_suite_loss": "lead_80",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PromotionEvidenceError(ValueError):
    """An evidence input is incomplete, malformed, or contradicts provenance."""


@dataclass(frozen=True)
class FinalizedRunnerCell:
    cell_name: str
    manifest_path: Path
    manifest_hash: str
    manifest: Mapping[str, Any]
    results_path: Path
    results: Tuple[Dict[str, Any], ...]
    moves_path: Path
    moves: Tuple[Dict[str, Any], ...]
    manifest_cell: Mapping[str, Any]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, source: str) -> str:
    if not _is_sha256(value):
        raise PromotionEvidenceError(
            f"{source} must be a lowercase 64-character SHA-256"
        )
    return value


def _require_string(value: Any, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\n" in value
        or "\r" in value
        or "\x00" in value
    ):
        raise PromotionEvidenceError(f"{source} must be a nonempty single-line string")
    return value


def _require_count(value: Any, source: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise PromotionEvidenceError(f"{source} must be a {qualifier} integer")
    return value


def _reject_constant(value: str) -> None:
    raise PromotionEvidenceError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PromotionEvidenceError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _decode_json(data: bytes, source: Path) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, PromotionEvidenceError) as exc:
        raise PromotionEvidenceError(f"{source}: invalid JSON: {exc}") from exc


def _load_canonical_json(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PromotionEvidenceError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise PromotionEvidenceError(f"cannot read {role} {source}: {exc}") from exc
    value = _decode_json(data, source)
    if not isinstance(value, dict):
        raise PromotionEvidenceError(f"{role} must have an object root")
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise PromotionEvidenceError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value


def _load_canonical_jsonl(path: Path, role: str) -> Tuple[Dict[str, Any], ...]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PromotionEvidenceError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise PromotionEvidenceError(f"cannot read {role} {source}: {exc}") from exc
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line:
            raise PromotionEvidenceError(
                f"{source}:{line_number}: blank JSONL lines are forbidden"
            )
        value = _decode_json(raw_line, source)
        if not isinstance(value, dict):
            raise PromotionEvidenceError(
                f"{source}:{line_number}: row must be an object"
            )
        rows.append(value)
    canonical = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    if data != canonical:
        raise PromotionEvidenceError(f"{role} must be canonical JSONL")
    if not rows:
        raise PromotionEvidenceError(f"{role} must not be empty")
    return tuple(rows)


def _relative_artifact(root: Path, value: Any, role: str) -> Path:
    relative = Path(_require_string(value, role))
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise PromotionEvidenceError(f"{role} must be a normalized relative path")
    path = root / relative
    if (
        path.resolve().parent != root.resolve()
        and root.resolve() not in path.resolve().parents
    ):
        raise PromotionEvidenceError(f"{role} escapes runner output directory")
    return path


def _verify_manifest_payload(manifest: Mapping[str, Any], role: str) -> None:
    payload = dict(manifest)
    stored_hash = payload.pop("manifestPayloadSha256", None)
    if stored_hash != canonical_sha256(payload):
        raise PromotionEvidenceError(f"{role} payload SHA-256 is invalid")


def _validate_result_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule_id: str,
    pair_count: int,
    position_ids: Sequence[str],
    cluster_ids: Sequence[str],
) -> None:
    pairs: Dict[str, List[Mapping[str, Any]]] = {}
    game_ids = set()
    for index, row in enumerate(rows):
        game_id = _require_string(row.get("gameId"), f"result row {index} gameId")
        pair_id = _require_string(row.get("pairId"), f"result row {index} pairId")
        if game_id in game_ids:
            raise PromotionEvidenceError(f"duplicate finalized gameId {game_id!r}")
        game_ids.add(game_id)
        if row.get("scheduleId") != schedule_id:
            raise PromotionEvidenceError(
                "result scheduleId contradicts runner manifest"
            )
        if row.get("resignation") is not False or row.get("hitTurnLimit") is not False:
            raise PromotionEvidenceError(
                "finalized promotion results may not contain resignations or turn limits"
            )
        if not isinstance(row.get("noResult"), bool):
            raise PromotionEvidenceError("result noResult must be boolean")
        pairs.setdefault(pair_id, []).append(row)
    if len(pairs) != pair_count or any(len(group) != 2 for group in pairs.values()):
        raise PromotionEvidenceError(
            "finalized results do not contain the exact complete pair set"
        )
    first_rows = [group[0] for group in pairs.values()]
    actual_positions = sorted(row.get("positionId") for row in first_rows)
    actual_clusters = [
        row.get("independentClusterId", row.get("positionSemanticSha256"))
        for row in first_rows
    ]
    if actual_positions != list(position_ids):
        raise PromotionEvidenceError(
            "finalized result position IDs contradict manifest cell"
        )
    if actual_clusters != list(cluster_ids):
        raise PromotionEvidenceError(
            "finalized result independent clusters contradict manifest cell"
        )
    for pair_id, group in pairs.items():
        first, second = group
        if (
            first.get("positionId") != second.get("positionId")
            or first.get("independentClusterId", first.get("positionSemanticSha256"))
            != second.get("independentClusterId", second.get("positionSemanticSha256"))
            or first.get("blackBot") != second.get("whiteBot")
            or first.get("whiteBot") != second.get("blackBot")
        ):
            raise PromotionEvidenceError(
                f"finalized pair {pair_id!r} is not a color-reversed cluster pair"
            )


def _validate_move_coverage(
    moves: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> None:
    result_by_game = {row["gameId"]: row for row in results}
    turns: Dict[str, List[int]] = {game_id: [] for game_id in result_by_game}
    rows_by_game: Dict[str, List[Mapping[str, Any]]] = {
        game_id: [] for game_id in result_by_game
    }
    seen = set()
    for index, row in enumerate(moves):
        game_id = row.get("gameId")
        if game_id not in result_by_game:
            raise PromotionEvidenceError(
                f"move row {index} references unknown gameId {game_id!r}"
            )
        turn = row.get("turnNumber")
        if type(turn) is not int or turn < 0:
            raise PromotionEvidenceError(
                f"move row {index} turnNumber must be nonnegative"
            )
        identity = (game_id, turn)
        if identity in seen:
            raise PromotionEvidenceError(f"duplicate move trace identity {identity!r}")
        seen.add(identity)
        result = result_by_game[game_id]
        for field in ("scheduleId", "pairId", "positionId", "seed"):
            if row.get(field) != result.get(field):
                raise PromotionEvidenceError(
                    f"move row {index} {field} contradicts finalized result"
                )
        player = row.get("player")
        if player not in ("B", "W"):
            raise PromotionEvidenceError(f"move row {index} player must be B or W")
        expected_bot = (
            result.get("blackBot") if player == "B" else result.get("whiteBot")
        )
        if row.get("bot") != expected_bot:
            raise PromotionEvidenceError(
                f"move row {index} bot contradicts result color assignment"
            )
        score_lead = row.get("scoreLead")
        win_probability = row.get("winProbability")
        if (
            isinstance(score_lead, bool)
            or not isinstance(score_lead, (int, float))
            or not math.isfinite(float(score_lead))
        ):
            raise PromotionEvidenceError(
                f"move row {index} requires finite scoreLead diagnostics"
            )
        if (
            isinstance(win_probability, bool)
            or not isinstance(win_probability, (int, float))
            or not math.isfinite(float(win_probability))
            or not 0.0 <= float(win_probability) <= 1.0
        ):
            raise PromotionEvidenceError(
                f"move row {index} requires bounded winProbability diagnostics"
            )
        turns[game_id].append(turn)
        rows_by_game[game_id].append(row)
    for game_id, result in result_by_game.items():
        start = _require_count(
            result.get("startTurnNumber"), f"{game_id} startTurnNumber"
        )
        count = _require_count(result.get("moveCount"), f"{game_id} moveCount")
        expected = list(range(start, start + count))
        if sorted(turns[game_id]) != expected:
            raise PromotionEvidenceError(
                f"move trace for {game_id!r} is not exact and contiguous"
            )
        ordered_rows = sorted(rows_by_game[game_id], key=lambda row: row["turnNumber"])
        if any(
            first["player"] == second["player"]
            for first, second in zip(ordered_rows, ordered_rows[1:])
        ):
            raise PromotionEvidenceError(
                f"move trace for {game_id!r} does not alternate players"
            )


def load_finalized_runner_cell(
    manifest_path: Path,
    *,
    cell_name: str,
    suite_manifest_path: Path,
    policy_hash: str,
) -> FinalizedRunnerCell:
    """Load one finalized runner bundle and recheck immutable artifact bindings."""

    if cell_name not in CELL_ORDER:
        raise PromotionEvidenceError(f"unknown confirmation cell {cell_name!r}")
    path = Path(manifest_path)
    manifest = _load_canonical_json(path, f"{cell_name} runner manifest")
    _verify_manifest_payload(manifest, f"{cell_name} runner manifest")
    if manifest.get("schemaVersion") != 1:
        raise PromotionEvidenceError(f"{cell_name} runner schemaVersion is unsupported")
    if manifest.get("runnerContract") != RUNNER_CONTRACT:
        raise PromotionEvidenceError(f"{cell_name} runner contract is unsupported")
    spec = manifest.get("evaluationSpec")
    execution = manifest.get("execution")
    schedule = manifest.get("schedule")
    cell = manifest.get("cell")
    if not all(isinstance(value, dict) for value in (spec, execution, schedule, cell)):
        raise PromotionEvidenceError(
            f"{cell_name} runner manifest is missing spec/execution/cell/schedule"
        )
    expected_coordinate = {
        "comparison": CELL_COMPARISONS[cell_name],
        "suite": CELL_SUITES[cell_name],
    }
    for key, expected in expected_coordinate.items():
        if spec.get(key) != expected or cell.get(key) != expected:
            raise PromotionEvidenceError(
                f"{cell_name} runner {key} contradicts required matrix"
            )
    if spec.get("policy_sha") != policy_hash:
        raise PromotionEvidenceError(f"{cell_name} runner policy hash changed")
    suite_manifest_hash = file_sha256(suite_manifest_path)
    if (
        spec.get("suite_manifest_sha") != suite_manifest_hash
        or schedule.get("suiteManifestSha256") != suite_manifest_hash
    ):
        raise PromotionEvidenceError(
            f"{cell_name} runner suite manifest binding changed"
        )
    manifest_cell = resolve_manifest_cell(
        suite_manifest_path,
        stage=spec.get("stage"),
        look=spec.get("look"),
        comparison=spec.get("comparison"),
        suite=spec.get("suite"),
    )
    if schedule.get("manifestCell") != manifest_cell:
        raise PromotionEvidenceError(
            f"{cell_name} runner does not contain the resolved manifest cell"
        )
    if schedule.get("manifestCellSha256") != canonical_sha256(manifest_cell):
        raise PromotionEvidenceError(
            f"{cell_name} runner manifest-cell hash is invalid"
        )
    expected_schedule = {
        "sha256": manifest_cell["schedule_hash"],
        "scheduleId": manifest_cell["schedule_id"],
        "rowCount": manifest_cell["schedule_row_count"],
        "pairCount": manifest_cell["color_pairs"],
        "suiteBankSha256": manifest_cell["bank_hash"],
    }
    if any(schedule.get(key) != value for key, value in expected_schedule.items()):
        raise PromotionEvidenceError(
            f"{cell_name} runner schedule contradicts authoritative manifest cell"
        )
    if (
        spec.get("schedule_sha") != manifest_cell["schedule_hash"]
        or spec.get("schedule_id") != manifest_cell["schedule_id"]
        or spec.get("suite_bank_sha") != manifest_cell["bank_hash"]
    ):
        raise PromotionEvidenceError(
            f"{cell_name} EvaluationSpec contradicts authoritative manifest cell"
        )
    if execution.get("moveTraces") is not True:
        raise PromotionEvidenceError(
            f"{cell_name} was not finalized with full move traces"
        )
    _require_sha256(execution.get("katagoBinarySha256"), "KataGo binary hash")

    root = path.parent
    results_manifest = manifest.get("results")
    moves_manifest = manifest.get("moves")
    if not isinstance(results_manifest, dict) or not isinstance(moves_manifest, dict):
        raise PromotionEvidenceError(
            f"{cell_name} runner manifest requires results and moves"
        )
    results_path = _relative_artifact(
        root, results_manifest.get("path"), f"{cell_name} result path"
    )
    moves_path = _relative_artifact(
        root, moves_manifest.get("path"), f"{cell_name} move path"
    )
    for artifact_path, artifact, role in (
        (results_path, results_manifest, "results"),
        (moves_path, moves_manifest, "moves"),
    ):
        expected_hash = _require_sha256(
            artifact.get("sha256"), f"{cell_name} {role} hash"
        )
        if file_sha256(artifact_path) != expected_hash:
            raise PromotionEvidenceError(
                f"{cell_name} {role} artifact hash contradicts runner manifest"
            )
    results = _load_canonical_jsonl(results_path, f"{cell_name} results")
    moves = _load_canonical_jsonl(moves_path, f"{cell_name} moves")
    if results_manifest.get("rowCount") != len(results):
        raise PromotionEvidenceError(f"{cell_name} result row count changed")
    if moves_manifest.get("rowCount") != len(moves):
        raise PromotionEvidenceError(f"{cell_name} move row count changed")
    _validate_result_pairs(
        results,
        schedule_id=manifest_cell["schedule_id"],
        pair_count=manifest_cell["color_pairs"],
        position_ids=manifest_cell["position_ids"],
        cluster_ids=manifest_cell["independent_cluster_ids"],
    )
    _validate_move_coverage(moves, results)
    return FinalizedRunnerCell(
        cell_name=cell_name,
        manifest_path=path,
        manifest_hash=file_sha256(path),
        manifest=manifest,
        results_path=results_path,
        results=results,
        moves_path=moves_path,
        moves=moves,
        manifest_cell=manifest_cell,
    )


def _look_number(value: Any) -> int:
    look = _require_string(value, "confirmation look")
    if not look.startswith("look-"):
        raise PromotionEvidenceError("confirmation look must be look-<number>")
    try:
        number = int(look.split("-", 1)[1])
    except ValueError as exc:
        raise PromotionEvidenceError("confirmation look is malformed") from exc
    return _require_count(number, "confirmation look number", positive=True)


def _search_settings(policy: Mapping[str, Any], powered: bool) -> Dict[str, Any]:
    if not powered:
        return {"use_score_maximizing_utility": False}
    objective = policy.get("objective")
    if not isinstance(objective, dict):
        raise PromotionEvidenceError("policy objective is missing")
    return {
        "use_score_maximizing_utility": True,
        "win_weight": objective["win_weight"],
        "score_power": objective["score_power"],
        "score_scale": objective["score_scale"],
    }


def _statistics_binding(cell: FinalizedRunnerCell) -> Dict[str, Any]:
    spec = cell.manifest["evaluationSpec"]
    execution = cell.manifest["execution"]
    return {
        "candidate_hash": spec["candidate_model_sha"],
        "reference_hash": spec["reference_model_sha"],
        "comparison": spec["comparison"],
        "suite": spec["suite"],
        "suite_hash": spec["suite_bank_sha"],
        "schedule_id": spec["schedule_id"],
        "schedule_hash": spec["schedule_sha"],
        "config_hash": spec["config_sha"],
        "runner_manifest_hash": cell.manifest_hash,
        "execution_hash": canonical_sha256(execution),
        "katago_binary_hash": execution["katagoBinarySha256"],
    }


def finalize_runner_statistics(
    cell: FinalizedRunnerCell,
    *,
    policy: Mapping[str, Any],
    look_number: int,
    candidate_bot: str = "candidate",
    bootstrap_replications: Optional[int] = None,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute and finalize one cell's paired statistics from runner artifacts."""

    report = compute_paired_statistics(
        cell.results,
        candidate_bot=candidate_bot,
        move_records=cell.moves,
        policy=dict(policy),
        look_number=look_number,
        bootstrap_replications=bootstrap_replications,
        bootstrap_seed=bootstrap_seed,
        data_binding=_statistics_binding(cell),
        finalized=True,
    )
    finalized = finalize_statistics_artifact(report, cell_name=cell.cell_name)
    if (
        finalized["statistics_manifest"]["color_pairs"]
        != cell.manifest_cell["color_pairs"]
    ):
        raise PromotionEvidenceError(
            f"{cell.cell_name} statistics pair count contradicts manifest"
        )
    if (
        finalized["statistics_manifest"]["position_ids"]
        != cell.manifest_cell["position_ids"]
    ):
        raise PromotionEvidenceError(
            f"{cell.cell_name} statistics positions contradict manifest"
        )
    return finalized


def _load_generic_runner_cell(
    manifest_path: Path,
    *,
    cell_name: str,
    policy_hash: str,
) -> FinalizedRunnerCell:
    """Load a non-confirmation runner bundle whose bank binding is legacy-flat."""

    path = Path(manifest_path)
    manifest = _load_canonical_json(path, f"{cell_name} runner manifest")
    _verify_manifest_payload(manifest, f"{cell_name} runner manifest")
    if (
        manifest.get("runnerContract") != RUNNER_CONTRACT
        or manifest.get("schemaVersion") != 1
    ):
        raise PromotionEvidenceError(f"{cell_name} runner contract is unsupported")
    spec = manifest.get("evaluationSpec")
    execution = manifest.get("execution")
    schedule = manifest.get("schedule")
    if not all(isinstance(item, dict) for item in (spec, execution, schedule)):
        raise PromotionEvidenceError(f"{cell_name} runner manifest is incomplete")
    if spec.get("policy_sha") != policy_hash:
        raise PromotionEvidenceError(f"{cell_name} runner policy hash changed")
    if execution.get("moveTraces") is not True:
        raise PromotionEvidenceError(f"{cell_name} requires full move traces")
    root = path.parent
    results_manifest = manifest.get("results")
    moves_manifest = manifest.get("moves")
    if not isinstance(results_manifest, dict) or not isinstance(moves_manifest, dict):
        raise PromotionEvidenceError(f"{cell_name} runner artifacts are incomplete")
    results_path = _relative_artifact(
        root, results_manifest.get("path"), f"{cell_name} result path"
    )
    moves_path = _relative_artifact(
        root, moves_manifest.get("path"), f"{cell_name} move path"
    )
    if file_sha256(results_path) != results_manifest.get("sha256") or file_sha256(
        moves_path
    ) != moves_manifest.get("sha256"):
        raise PromotionEvidenceError(f"{cell_name} runner artifact hash changed")
    results = _load_canonical_jsonl(results_path, f"{cell_name} results")
    moves = _load_canonical_jsonl(moves_path, f"{cell_name} moves")
    if (
        len(results) != schedule.get("rowCount")
        or len(results) != results_manifest.get("rowCount")
        or len(moves) != moves_manifest.get("rowCount")
    ):
        raise PromotionEvidenceError(f"{cell_name} runner row count changed")
    pair_first: Dict[str, Mapping[str, Any]] = {}
    for row in results:
        pair_first.setdefault(row.get("pairId"), row)
    position_ids = sorted(row.get("positionId") for row in pair_first.values())
    cluster_ids = [
        row.get("independentClusterId", row.get("positionSemanticSha256"))
        for row in pair_first.values()
    ]
    if not all(_is_sha256(value) for value in cluster_ids):
        raise PromotionEvidenceError(
            f"{cell_name} results lack independent cluster provenance"
        )
    _validate_result_pairs(
        results,
        schedule_id=schedule.get("scheduleId"),
        pair_count=schedule.get("pairCount"),
        position_ids=position_ids,
        cluster_ids=cluster_ids,
    )
    _validate_move_coverage(moves, results)
    manifest_cell = {
        "cell_name": cell_name,
        "stage": spec.get("stage"),
        "look": spec.get("look"),
        "comparison": spec.get("comparison"),
        "suite": spec.get("suite"),
        "color_pairs": schedule.get("pairCount"),
        "position_ids": position_ids,
        "independent_cluster_ids": cluster_ids,
        "independent_cluster_ids_hash": canonical_sha256(cluster_ids),
        "bank_hash": spec.get("suite_bank_sha"),
        "schedule_hash": spec.get("schedule_sha"),
        "schedule_id": spec.get("schedule_id"),
    }
    return FinalizedRunnerCell(
        cell_name=cell_name,
        manifest_path=path,
        manifest_hash=file_sha256(path),
        manifest=manifest,
        results_path=results_path,
        results=results,
        moves_path=moves_path,
        moves=moves,
        manifest_cell=manifest_cell,
    )


def build_nonconfirmation_controller_evidence(
    plan: Mapping[str, Any],
    *,
    runner_manifests: Optional[Mapping[str, Path]] = None,
    policy_path: Path = DEFAULT_POLICY_PATH,
    stage0_probe_path: Optional[Path] = None,
    stage0_probe_sha256: Optional[str] = None,
    stage0_request_path: Optional[Path] = None,
    stage0_request_sha256: Optional[str] = None,
    bootstrap_replications: Optional[int] = None,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Derive a non-confirmation stage gate; no caller-supplied decision is used."""

    stage = plan.get("stage")
    stage_to_controller = {
        "stage-0": "integrity",
        "stage-1": "screen",
        "stage-2": "finalist",
    }
    if stage not in stage_to_controller:
        raise PromotionEvidenceError("non-confirmation plan stage is unsupported")
    specs = plan.get("specs")
    if not isinstance(specs, list) or not specs:
        raise PromotionEvidenceError("controller plan has no EvaluationSpecs")
    candidate_hash = specs[0].get("candidate_model_sha")
    original_hash = specs[0].get("original_model_sha")
    champion_spec = next(
        (
            spec
            for spec in specs
            if isinstance(spec, dict)
            and spec.get("comparison") == "candidate-vs-champion-powered"
        ),
        None,
    )
    if not isinstance(champion_spec, dict):
        raise PromotionEvidenceError("controller plan has no champion comparison")
    champion_hash = champion_spec.get("reference_model_sha")
    for value, source in (
        (candidate_hash, "candidate hash"),
        (original_hash, "original hash"),
        (champion_hash, "champion hash"),
        (plan.get("policyHash"), "policy hash"),
        (plan.get("configHash"), "config bundle hash"),
        (plan.get("scheduleHash"), "schedule bundle hash"),
        (plan.get("selfplayConfigHash"), "self-play config hash"),
        (plan.get("suiteManifestHash"), "suite manifest hash"),
    ):
        _require_sha256(value, source)
    policy = load_policy(Path(policy_path))
    if canonical_sha256(policy) != plan["policyHash"]:
        raise PromotionEvidenceError("non-confirmation policy contradicts plan")
    if str(Path(policy_path)) != plan["policyPath"]:
        raise PromotionEvidenceError("non-confirmation policy path contradicts plan")
    suite_manifest_path = Path(plan["suiteManifestPath"])
    if file_sha256(suite_manifest_path) != plan["suiteManifestHash"]:
        raise PromotionEvidenceError(
            "non-confirmation suite manifest hash contradicts plan"
        )
    load_suite_manifest(suite_manifest_path)

    derived_artifacts: Dict[str, Any] = {}
    reason_codes: List[str] = []
    if stage == "stage-0":
        if stage0_probe_path is None or stage0_probe_sha256 is None:
            raise PromotionEvidenceError("Stage-0 probe output is required")
        stage0 = validate_stage0_probe(
            stage0_probe_path,
            expected_sha256=stage0_probe_sha256,
            policy=policy,
            candidate_hash=candidate_hash,
            champion_hash=champion_hash,
            original_hash=original_hash,
            request_path=stage0_request_path,
            request_sha256=stage0_request_sha256,
        )
        decision = "PASS" if stage0["stage_0_passed"] else "FAIL"
        if decision == "FAIL":
            reason_codes.append("STAGE_0_PROBE_FAILED")
        derived_artifacts["stage_0"] = stage0
    else:
        if not isinstance(runner_manifests, Mapping) or not runner_manifests:
            raise PromotionEvidenceError(
                "non-confirmation game stages require runner manifests"
            )
        cells = {
            name: _load_generic_runner_cell(
                Path(path), cell_name=name, policy_hash=plan["policyHash"]
            )
            for name, path in sorted(runner_manifests.items())
        }
        runner_specs = [cell.manifest["evaluationSpec"] for cell in cells.values()]
        if sorted(canonical_json(value) for value in runner_specs) != sorted(
            canonical_json(value) for value in specs
        ):
            raise PromotionEvidenceError(
                "non-confirmation runner cells do not match the exact controller plan"
            )
        finalized: Dict[str, Dict[str, Any]] = {}
        for name, cell in cells.items():
            report = compute_paired_statistics(
                cell.results,
                candidate_bot="candidate",
                move_records=cell.moves,
                policy=policy,
                look_number=1,
                bootstrap_replications=bootstrap_replications,
                bootstrap_seed=bootstrap_seed,
                data_binding={
                    **_statistics_binding(cell),
                    "independent_cluster_ids": cell.manifest_cell[
                        "independent_cluster_ids"
                    ],
                    "independent_cluster_ids_hash": cell.manifest_cell[
                        "independent_cluster_ids_hash"
                    ],
                },
                finalized=True,
            )
            finalized[name] = finalize_statistics_artifact(report, cell_name=name)
        champion_cell = next(
            (
                value
                for value in finalized.values()
                if value["statistics_artifact"]["data_binding"]["comparison"]
                == "candidate-vs-champion-powered"
            ),
            None,
        )
        if champion_cell is None:
            raise PromotionEvidenceError(
                "non-confirmation statistics omit champion powered comparison"
            )
        artifact = champion_cell["statistics_artifact"]
        metric = artifact.get("metrics", {}).get("realized_utility")
        valid = artifact.get("validation", {}).get("promotion_valid") is True
        available = (
            isinstance(metric, dict)
            and metric.get("available") is True
            and metric.get("complete") is True
        )
        if not valid:
            decision = "FAIL"
            reason_codes.append("MATCH_VALIDATION_FAILED")
        elif not available:
            decision = "INCONCLUSIVE"
            reason_codes.append("UTILITY_INFERENCE_UNAVAILABLE")
        elif stage == "stage-1":
            threshold = policy["evaluation_stages"]["stage_1_cheap_paired_screen"][
                "utility_futility_upper_bound"
            ]
            upper = metric.get("upper_bound")
            if not isinstance(upper, (int, float)) or not math.isfinite(float(upper)):
                decision = "INCONCLUSIVE"
                reason_codes.append("UTILITY_UPPER_BOUND_UNAVAILABLE")
            elif float(upper) <= float(threshold):
                decision = "FAIL"
                reason_codes.append("UTILITY_FUTILITY_BOUND")
            else:
                decision = "PASS"
        else:
            decision = "PASS"
        derived_artifacts["statistics"] = finalized

    metadata = {
        "finalized": True,
        "candidate_hash": candidate_hash,
        "tested_champion_hash": champion_hash,
        "original_hash": original_hash,
        "evaluation_key": plan["evaluationKey"],
        "config_hash": plan["configHash"],
        "schedule_hash": plan["scheduleHash"],
        "policy_hash": plan["policyHash"],
        "selfplay_config_hash": plan["selfplayConfigHash"],
        "topology": plan["topology"],
    }
    stage_gate_payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": STAGE_EVIDENCE_CONTRACT,
        "controller_stage": stage_to_controller[stage],
        "decision": decision,
        "reason_codes": sorted(reason_codes),
        **metadata,
        "derived_artifacts": derived_artifacts,
    }
    stage_gate = {
        **stage_gate_payload,
        "derivation_hash": canonical_sha256(stage_gate_payload),
    }
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "evidence_contract": EVIDENCE_CONTRACT,
        "controller_stage": stage_to_controller[stage],
        **metadata,
        "policy_path": plan["policyPath"],
        "policy_version": plan["policyVersion"],
        "suite_manifest_path": plan["suiteManifestPath"],
        "suite_manifest_hash": plan["suiteManifestHash"],
        "look": plan["look"],
        "schedule_artifacts": json.loads(canonical_json(plan["scheduleArtifacts"])),
        "stage_gate": stage_gate,
    }
    if stage0_request_path is not None and stage0_request_sha256 is not None:
        envelope["stage0_request_path"] = str(stage0_request_path)
        envelope["stage0_request_hash"] = stage0_request_sha256
    if stage0_probe_path is not None and stage0_probe_sha256 is not None:
        envelope["stage0_probe_path"] = str(stage0_probe_path)
        envelope["stage0_probe_hash"] = stage0_probe_sha256
        envelope["stage0_probe_sha256"] = stage0_probe_sha256
    return envelope


def _artifact_hash(value: Mapping[str, Any], role: str) -> str:
    expected = value.get("artifact_hash")
    payload = dict(value)
    payload.pop("artifact_hash", None)
    actual = canonical_sha256(payload)
    if expected != actual:
        raise PromotionEvidenceError(f"{role} artifact_hash is invalid")
    return actual


def derive_discovery_evidence(
    stage_1_statistics: Mapping[str, Any],
    stage_2_statistics: Sequence[Mapping[str, Any]],
    *,
    candidate_hash: str,
    policy: Mapping[str, Any],
    confirmation_cluster_ids: Iterable[str],
) -> Dict[str, Any]:
    """Derive Stage-1/2 decisions from finalized discovery statistics only."""

    candidate_hash = _require_sha256(candidate_hash, "discovery candidate hash")
    policy_hash = canonical_sha256(policy)

    def validate_artifact(
        value: Mapping[str, Any],
        stage: str,
        *,
        expected_candidate_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        artifact = json.loads(canonical_json(value))
        if artifact.get("finalized") is not True:
            raise PromotionEvidenceError(f"{stage} statistics are not finalized")
        if artifact.get("policy_hash") != policy_hash:
            raise PromotionEvidenceError(f"{stage} statistics use another policy")
        binding = artifact.get("data_binding")
        if not isinstance(binding, dict):
            raise PromotionEvidenceError(f"{stage} statistics lack data binding")
        if binding.get("suite") != "discovery":
            raise PromotionEvidenceError(
                f"{stage} statistics must use the discovery holdout"
            )
        bound_candidate = _require_sha256(
            binding.get("candidate_hash"), f"{stage} candidate hash"
        )
        if (
            expected_candidate_hash is not None
            and bound_candidate != expected_candidate_hash
        ):
            raise PromotionEvidenceError(f"{stage} statistics name another candidate")
        cluster_ids = binding.get("independent_cluster_ids")
        if not (
            isinstance(cluster_ids, list)
            and cluster_ids
            and len(cluster_ids) == len(set(cluster_ids))
            and all(_is_sha256(value) for value in cluster_ids)
            and binding.get("independent_cluster_ids_hash")
            == canonical_sha256(cluster_ids)
        ):
            raise PromotionEvidenceError(
                f"{stage} statistics lack exact independent-cluster provenance"
            )
        validation = artifact.get("validation")
        if not isinstance(validation, dict):
            raise PromotionEvidenceError(f"{stage} statistics lack validation")
        return artifact

    stage_1 = validate_artifact(
        stage_1_statistics,
        "Stage-1",
        expected_candidate_hash=candidate_hash,
    )
    stage_2 = [validate_artifact(value, "Stage-2") for value in stage_2_statistics]
    if not stage_2:
        raise PromotionEvidenceError("at least one Stage-2 finalist is required")
    stage_1_metric = stage_1.get("metrics", {}).get("realized_utility")
    futility = policy["evaluation_stages"]["stage_1_cheap_paired_screen"][
        "utility_futility_upper_bound"
    ]
    stage_1_passed = bool(
        stage_1["validation"].get("promotion_valid") is True
        and isinstance(stage_1_metric, dict)
        and stage_1_metric.get("available") is True
        and stage_1_metric.get("complete") is True
        and isinstance(stage_1_metric.get("upper_bound"), (int, float))
        and math.isfinite(float(stage_1_metric["upper_bound"]))
        and float(stage_1_metric["upper_bound"]) > float(futility)
    )

    safe_finalists: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for artifact in stage_2:
        utility = artifact.get("metrics", {}).get("realized_utility")
        risk = artifact.get("risk_differences", {}).get("final_50")
        binding = artifact["data_binding"]
        if not (
            artifact["validation"].get("promotion_valid") is True
            and isinstance(utility, dict)
            and utility.get("available") is True
            and utility.get("complete") is True
            and isinstance(utility.get("lower_bound"), (int, float))
            and math.isfinite(float(utility["lower_bound"]))
            and isinstance(risk, dict)
            and isinstance(risk.get("upper_bound"), (int, float))
            and math.isfinite(float(risk["upper_bound"]))
        ):
            continue
        safe_finalists.append(
            (
                -float(utility["lower_bound"]),
                float(risk["upper_bound"]),
                binding["candidate_hash"],
                artifact,
            )
        )
    safe_finalists.sort(key=lambda item: item[:3])
    maximum = policy["evaluation_stages"]["stage_2_finalist_selection"][
        "maximum_survivors"
    ]
    safe_finalists = safe_finalists[:maximum]
    selected = safe_finalists[0][2] if safe_finalists else None
    selected_artifact = safe_finalists[0][3] if safe_finalists else None
    stage_2_passed = selected == candidate_hash
    discovery_clusters = set()
    for artifact in (stage_1, *(item[3] for item in safe_finalists)):
        discovery_clusters.update(artifact["data_binding"]["independent_cluster_ids"])
    confirmation_clusters = set(confirmation_cluster_ids)
    independent = not discovery_clusters.intersection(confirmation_clusters)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": DISCOVERY_CONTRACT,
        "finalized": True,
        "candidate_hash": candidate_hash,
        "policy_hash": policy_hash,
        "stage_1_statistics": stage_1,
        "stage_1_statistics_hash": canonical_sha256(stage_1),
        "stage_2_statistics": stage_2,
        "stage_2_statistics_hashes": [
            canonical_sha256(artifact) for artifact in stage_2
        ],
        "selected_candidate_hash": selected,
        "selected_statistics_hash": (
            canonical_sha256(selected_artifact)
            if selected_artifact is not None
            else None
        ),
        "summary": {
            "stage_1_passed": stage_1_passed,
            "stage_2_passed": stage_2_passed,
            "dominated_by_later_safe_finalist": (
                selected is not None and selected != candidate_hash
            ),
            "confirmation_schedule_independent": independent,
            "confirmation_candidate_count": 1 if selected is not None else 0,
        },
    }
    return {**payload, "artifact_hash": canonical_sha256(payload)}


def validate_discovery_evidence(
    value: Mapping[str, Any],
    *,
    candidate_hash: str,
    policy: Mapping[str, Any],
    confirmation_cluster_ids: Iterable[str],
) -> Dict[str, Any]:
    artifact = json.loads(canonical_json(value))
    if artifact.get("contract") != DISCOVERY_CONTRACT:
        raise PromotionEvidenceError("discovery evidence contract is unsupported")
    _artifact_hash(artifact, "discovery evidence")
    rebuilt = derive_discovery_evidence(
        artifact.get("stage_1_statistics", {}),
        artifact.get("stage_2_statistics", []),
        candidate_hash=candidate_hash,
        policy=policy,
        confirmation_cluster_ids=confirmation_cluster_ids,
    )
    if artifact != rebuilt:
        raise PromotionEvidenceError(
            "discovery decisions are not derivable from finalized statistics"
        )
    return artifact


def validate_stage0_probe(
    probe_path: Path,
    *,
    expected_sha256: str,
    policy: Mapping[str, Any],
    candidate_hash: str,
    champion_hash: str,
    original_hash: str,
    request_path: Optional[Path] = None,
    request_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate configured Stage-0 measurements and derive the probe decision."""

    expected_sha256 = _require_sha256(expected_sha256, "Stage-0 probe hash")
    if file_sha256(probe_path) != expected_sha256:
        raise PromotionEvidenceError("Stage-0 probe file hash mismatch")
    probe = _load_canonical_json(probe_path, "Stage-0 probe")
    if probe.get("contract") != STAGE0_PROBE_CONTRACT:
        raise PromotionEvidenceError("Stage-0 probe contract is unsupported")
    if "decision" in probe or "stage_0_passed" in probe:
        raise PromotionEvidenceError(
            "Stage-0 probe may not supply an arbitrary decision/PASS marker"
        )
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "finalized": True,
        "candidate_hash": candidate_hash,
        "tested_champion_hash": champion_hash,
        "original_hash": original_hash,
        "policy_hash": canonical_sha256(policy),
    }
    conflicts = [
        key for key, expected in expected_identity.items() if probe.get(key) != expected
    ]
    if conflicts:
        raise PromotionEvidenceError(
            "Stage-0 probe identity conflicts: " + ", ".join(sorted(conflicts))
        )
    request_hash: Optional[str] = None
    if request_path is not None or request_sha256 is not None:
        if request_path is None or request_sha256 is None:
            raise PromotionEvidenceError(
                "Stage-0 request path and hash must be supplied together"
            )
        request_sha256 = _require_sha256(request_sha256, "Stage-0 request file hash")
        if file_sha256(request_path) != request_sha256:
            raise PromotionEvidenceError("Stage-0 request file hash mismatch")
        request = _load_canonical_json(request_path, "Stage-0 request")
        if request.get("contract") != "risk-score-stage-0-request-v1":
            raise PromotionEvidenceError("Stage-0 request contract is unsupported")
        expected_request_identity = {
            "schema_version": SCHEMA_VERSION,
            "candidate_hash": candidate_hash,
            "tested_champion_hash": champion_hash,
            "original_hash": original_hash,
            "policy_hash": canonical_sha256(policy),
            "stage": "stage-0",
        }
        request_conflicts = [
            key
            for key, expected in expected_request_identity.items()
            if request.get(key) != expected
        ]
        if request_conflicts:
            raise PromotionEvidenceError(
                "Stage-0 request identity conflicts: "
                + ", ".join(sorted(request_conflicts))
            )
        for field in (
            "checkpoint_hash",
            "candidate_manifest_hash",
            "suite_manifest_hash",
        ):
            _require_sha256(request.get(field), f"Stage-0 request {field}")
        for field in (
            "policy_path",
            "policy_version",
            "suite_manifest_path",
            "evaluation_key",
            "look",
        ):
            _require_string(request.get(field), f"Stage-0 request {field}")
        if request.get("policy_version") != policy.get("policy_version"):
            raise PromotionEvidenceError(
                "Stage-0 request policy version contradicts policy"
            )
        stage_policy = policy["evaluation_stages"][
            "stage_0_integrity_and_fixed_probes"
        ]
        if request.get("probe_contract") != stage_policy:
            raise PromotionEvidenceError(
                "Stage-0 request probe contract contradicts policy"
            )
        if not isinstance(request.get("schedule_artifacts"), dict):
            raise PromotionEvidenceError(
                "Stage-0 request schedule artifacts are missing"
            )
        request_hash = request_sha256
        if probe.get("request_hash") != request_hash:
            raise PromotionEvidenceError(
                "Stage-0 probe is not bound to the configured request"
            )
    else:
        request_hash = _require_sha256(
            probe.get("request_hash"), "Stage-0 configured request hash"
        )

    stage_policy = policy["evaluation_stages"]["stage_0_integrity_and_fixed_probes"]
    checks = probe.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(
        stage_policy["required_checks"]
    ):
        raise PromotionEvidenceError(
            "Stage-0 probe checks differ from the policy-required checks"
        )
    if not all(isinstance(value, bool) for value in checks.values()):
        raise PromotionEvidenceError("Stage-0 probe checks must be booleans")
    measurements = probe.get("measurements")
    if not isinstance(measurements, dict):
        raise PromotionEvidenceError("Stage-0 probe measurements are missing")
    for field in (
        "fixed_analysis_positions",
        "fixed_analysis_visits",
        "exploitability_sentinel_positions",
        "exploitability_sentinel_visits",
    ):
        if measurements.get(field) != stage_policy[field]:
            raise PromotionEvidenceError(
                f"Stage-0 measurement {field} contradicts policy"
            )
    zero_fields = (
        "hard_tactical_failures",
        "hard_exploitability_failures",
        "unresolved_failures",
        "model_runtime_errors",
        "perspective_violations",
        "clamp_violations",
        "endpoint_violations",
        "nonfinite_violations",
        "decomposition_violations",
    )
    for field in zero_fields:
        _require_count(measurements.get(field), f"Stage-0 {field}")
    for field in (
        "selected_move_endpoint_mass_dominated",
        "visit_stability_acceptable",
    ):
        if not isinstance(measurements.get(field), bool):
            raise PromotionEvidenceError(f"Stage-0 {field} must be boolean")
    passed = (
        all(checks.values())
        and all(measurements[field] == 0 for field in zero_fields)
        and measurements["selected_move_endpoint_mass_dominated"] is False
        and measurements["visit_stability_acceptable"] is True
    )
    return {
        "stage_0_passed": passed,
        "request_hash": request_hash,
        "probe_output_hash": expected_sha256,
        **{
            field: measurements[field]
            for field in (
                "fixed_analysis_positions",
                "fixed_analysis_visits",
                "exploitability_sentinel_positions",
                "exploitability_sentinel_visits",
                "hard_tactical_failures",
                "hard_exploitability_failures",
                "unresolved_failures",
                "model_runtime_errors",
                "selected_move_endpoint_mass_dominated",
                "visit_stability_acceptable",
            )
        },
        "validity": {
            field: measurements[field]
            for field in (
                "perspective_violations",
                "clamp_violations",
                "endpoint_violations",
                "nonfinite_violations",
                "decomposition_violations",
            )
        },
        "checks": dict(sorted(checks.items())),
    }


def _combined_lead_artifact(
    cells: Mapping[str, FinalizedRunnerCell],
    finalized: Mapping[str, Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    look_number: int,
    bootstrap_replications: Optional[int],
    bootstrap_seed: Optional[int],
) -> Dict[str, Any]:
    lead_cells = [cells["lead_40"], cells["lead_80"]]
    source_hashes = {
        name: finalized[name]["statistics_artifact_hash"]
        for name in ("lead_40", "lead_80")
    }
    binding = {
        "candidate_hash": lead_cells[0].manifest["evaluationSpec"][
            "candidate_model_sha"
        ],
        "reference_hash": lead_cells[0].manifest["evaluationSpec"][
            "reference_model_sha"
        ],
        "comparison": "combined-lead",
        "suite": "combined-lead",
        "suite_hash": canonical_sha256(
            sorted(
                cell.manifest["evaluationSpec"]["suite_bank_sha"] for cell in lead_cells
            )
        ),
        "schedule_id": "combined-lead-"
        + canonical_sha256(
            [cell.manifest["evaluationSpec"]["schedule_id"] for cell in lead_cells]
        )[:24],
        "schedule_hash": canonical_sha256(
            sorted(
                cell.manifest["evaluationSpec"]["schedule_sha"] for cell in lead_cells
            )
        ),
        "config_hash": lead_cells[0].manifest["evaluationSpec"]["config_sha"],
        "runner_manifest_hash": canonical_sha256(
            [cell.manifest_hash for cell in lead_cells]
        ),
        "execution_hash": canonical_sha256(
            [canonical_sha256(cell.manifest["execution"]) for cell in lead_cells]
        ),
        "katago_binary_hash": lead_cells[0].manifest["execution"]["katagoBinarySha256"],
    }
    report = compute_paired_statistics(
        tuple(row for cell in lead_cells for row in cell.results),
        candidate_bot="candidate",
        move_records=tuple(row for cell in lead_cells for row in cell.moves),
        policy=dict(policy),
        look_number=look_number,
        bootstrap_replications=bootstrap_replications,
        bootstrap_seed=bootstrap_seed,
        data_binding=binding,
        finalized=True,
    )
    metric = report["suite_metrics"]["combined_lead"]
    position_ids = sorted(
        set(finalized["lead_40"]["statistics_manifest"]["position_ids"]).union(
            finalized["lead_80"]["statistics_manifest"]["position_ids"]
        )
    )
    if (
        metric["color_pairs"]
        != sum(
            finalized[name]["statistics_manifest"]["color_pairs"]
            for name in ("lead_40", "lead_80")
        )
        or sorted(row["position_id"] for row in metric["position_values"])
        != position_ids
    ):
        raise PromotionEvidenceError(
            "combined Lead statistics do not bind the exact union of both suites"
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "finalized": True,
        "policy_hash": canonical_sha256(policy),
        "look": report["look"],
        "source_statistics_artifact_hashes": source_hashes,
        "counts": {
            "color_pairs": metric["color_pairs"],
            "position_clusters": metric["position_clusters"],
        },
        "position_ids": position_ids,
        "metrics": {"combined_lead_realized_utility": metric},
    }
    artifact_hash = canonical_sha256(artifact)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "finalized": True,
        "source_cells": ["lead_40", "lead_80"],
        "source_statistics_artifact_hashes": source_hashes,
        "color_pairs": metric["color_pairs"],
        "position_ids": position_ids,
        "metric_names": ["combined_lead_realized_utility"],
        "statistics_artifact_hash": artifact_hash,
    }
    return {
        "combined_lead_artifact": artifact,
        "combined_lead_artifact_hash": artifact_hash,
        "combined_lead_manifest": manifest,
        "combined_lead_manifest_hash": canonical_sha256(manifest),
    }


def build_promotion_evidence(
    runner_manifests: Mapping[str, Path],
    *,
    suite_manifest_path: Path,
    policy_path: Path = DEFAULT_POLICY_PATH,
    stage0_probe_path: Path,
    stage0_probe_sha256: str,
    discovery_evidence: Mapping[str, Any],
    attempt: Mapping[str, Any],
    stage0_request_path: Optional[Path] = None,
    stage0_request_sha256: Optional[str] = None,
    bootstrap_replications: Optional[int] = None,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build gate-grade Stage-3 evidence from the exact five runner bundles."""

    if set(runner_manifests) != set(CELL_ORDER):
        raise PromotionEvidenceError(
            "runner_manifests must contain exactly the five confirmation cells"
        )
    policy = load_policy(Path(policy_path))
    policy_hash = canonical_sha256(policy)
    suite_manifest_path = Path(suite_manifest_path)
    suite_manifest = load_suite_manifest(suite_manifest_path)
    suite_manifest_hash = file_sha256(suite_manifest_path)
    if suite_manifest.get("policy_hash") != policy_hash:
        raise PromotionEvidenceError(
            "suite manifest is bound to a different promotion policy"
        )
    cells = {
        name: load_finalized_runner_cell(
            Path(runner_manifests[name]),
            cell_name=name,
            suite_manifest_path=suite_manifest_path,
            policy_hash=policy_hash,
        )
        for name in CELL_ORDER
    }
    specs = [cells[name].manifest["evaluationSpec"] for name in CELL_ORDER]
    candidates = {spec["candidate_model_sha"] for spec in specs}
    originals = {spec["original_model_sha"] for spec in specs}
    stages = {spec["stage"] for spec in specs}
    looks = {spec["look"] for spec in specs}
    topologies = {spec["topology"] for spec in specs}
    binaries = {
        cells[name].manifest["execution"]["katagoBinarySha256"] for name in CELL_ORDER
    }
    if any(
        len(values) != 1
        for values in (candidates, originals, stages, looks, topologies, binaries)
    ):
        raise PromotionEvidenceError(
            "runner cells disagree on candidate/original/stage/look/topology/binary"
        )
    candidate_hash = next(iter(candidates))
    original_hash = next(iter(originals))
    stage = next(iter(stages))
    look = next(iter(looks))
    topology = next(iter(topologies))
    binary_hash = next(iter(binaries))
    if stage != "stage-3":
        raise PromotionEvidenceError("promotion evidence requires Stage-3 runner cells")
    look_number = _look_number(look)
    champion_hash = specs[0]["reference_model_sha"]
    if (
        specs[1]["reference_model_sha"] != original_hash
        or specs[2]["reference_model_sha"] != original_hash
        or specs[3]["reference_model_sha"] != champion_hash
        or specs[4]["reference_model_sha"] != champion_hash
    ):
        raise PromotionEvidenceError(
            "runner reference hashes contradict the five-cell matrix"
        )
    finalized = {
        name: finalize_runner_statistics(
            cells[name],
            policy=policy,
            look_number=look_number,
            bootstrap_replications=bootstrap_replications,
            bootstrap_seed=bootstrap_seed,
        )
        for name in CELL_ORDER
    }
    stage0 = validate_stage0_probe(
        Path(stage0_probe_path),
        expected_sha256=stage0_probe_sha256,
        policy=policy,
        candidate_hash=candidate_hash,
        champion_hash=champion_hash,
        original_hash=original_hash,
        request_path=stage0_request_path,
        request_sha256=stage0_request_sha256,
    )
    confirmation_cluster_ids = [
        cluster
        for name in CELL_ORDER
        for cluster in cells[name].manifest_cell["independent_cluster_ids"]
    ]
    discovery = validate_discovery_evidence(
        discovery_evidence,
        candidate_hash=candidate_hash,
        policy=policy,
        confirmation_cluster_ids=confirmation_cluster_ids,
    )
    attempt_value = json.loads(canonical_json(attempt))
    _require_string(attempt_value.get("generation_id"), "attempt generation_id")
    attempt_number = _require_count(
        attempt_value.get("attempt_number"), "attempt number", positive=True
    )
    _require_count(
        attempt_value.get("promotions_for_generation"),
        "attempt promotions_for_generation",
    )
    maximum_attempts = policy["attempt_budget"][
        "maximum_confirmation_attempts_per_generation"
    ]
    if attempt_number > maximum_attempts:
        raise PromotionEvidenceError("confirmation attempt exceeds policy budget")
    if attempt_number > 1 and (
        attempt_value.get("new_holdout_block") is not True
        or attempt_value.get("new_alpha_allocation") is not True
    ):
        raise PromotionEvidenceError(
            "fallback confirmation requires new holdout and alpha allocation"
        )

    stage_3 = policy["evaluation_stages"]["stage_3_promotion_confirmation"]
    policy_matrix = policy["required_confirmation_matrix"]
    matrix: Dict[str, Any] = {}
    for name in CELL_ORDER:
        cell = cells[name]
        spec = cell.manifest["evaluationSpec"]
        execution = cell.manifest["execution"]
        statistics = finalized[name]
        search_mode = policy_matrix[name]["search_mode"]
        settings = _search_settings(policy, search_mode == "powered")
        matrix[name] = {
            "comparison": spec["comparison"],
            "suite": spec["suite"],
            "stage": spec["stage"],
            "look": spec["look"],
            "topology": spec["topology"],
            "search_mode": search_mode,
            "candidate_hash": spec["candidate_model_sha"],
            "reference_hash": spec["reference_model_sha"],
            "visits": (
                stage_3["powered_visits"]
                if search_mode == "powered"
                else stage_3["standard_visits"]
            ),
            "color_pairs": cell.manifest_cell["color_pairs"],
            "config_hash": spec["config_sha"],
            "schedule_id": spec["schedule_id"],
            "schedule_hash": spec["schedule_sha"],
            "suite_hash": spec["suite_bank_sha"],
            "independent_cluster_ids": cell.manifest_cell["independent_cluster_ids"],
            "independent_cluster_ids_hash": cell.manifest_cell[
                "independent_cluster_ids_hash"
            ],
            "katago_binary_hash": execution["katagoBinarySha256"],
            "runner_manifest": cell.manifest,
            "runner_manifest_hash": cell.manifest_hash,
            "execution_manifest": execution,
            "execution_hash": canonical_sha256(execution),
            **statistics,
            "candidate_search": json.loads(canonical_json(settings)),
            "reference_search": json.loads(canonical_json(settings)),
            "validation": statistics["statistics_artifact"]["validation"],
        }

    combined = _combined_lead_artifact(
        cells,
        finalized,
        policy=policy,
        look_number=look_number,
        bootstrap_replications=bootstrap_replications,
        bootstrap_seed=bootstrap_seed,
    )
    risk_differences = {
        risk_name: matrix[cell_name]["statistics_artifact"]["risk_differences"][
            risk_name
        ]
        for risk_name, cell_name in sorted(RISK_CELL_BINDINGS.items())
    }
    validations = [
        matrix[name]["statistics_artifact"]["validation"] for name in CELL_ORDER
    ]
    total_games = sum(
        matrix[name]["statistics_artifact"]["counts"]["games"] for name in CELL_ORDER
    )
    true_no_results = sum(
        matrix[name]["statistics_artifact"]["counts"]["true_no_results"]
        for name in CELL_ORDER
    )
    sum_fields = (
        "missing_games",
        "duplicate_game_ids",
        "incomplete_pairs",
        "duplicate_pair_members",
        "resignations",
        "turn_limits",
        "unresolved_rows",
        "structural_errors",
        "resolved_missing_numeric_scores",
    )
    validity = {
        "promotion_valid": all(
            validation.get("promotion_valid") is True for validation in validations
        ),
        **{
            field: sum(
                _require_count(validation.get(field), f"statistics validation {field}")
                for validation in validations
            )
            for field in sum_fields
        },
        **stage0["validity"],
        "true_no_results": true_no_results,
        "total_games": total_games,
        "true_no_result_rate": (true_no_results / total_games if total_games else None),
        "full_move_diagnostics": True,
    }
    banks = {
        bank.get("name"): bank
        for bank in suite_manifest.get("banks", [])
        if isinstance(bank, dict) and isinstance(bank.get("name"), str)
    }
    for required_bank in ("tactical", "exploitability"):
        if required_bank not in banks or not isinstance(
            banks[required_bank].get("positions"), dict
        ):
            raise PromotionEvidenceError(
                f"suite manifest lacks required {required_bank} provenance bank"
            )
    config_hashes = {
        "powered_match": specs[0]["config_sha"],
        "standard_match": specs[2]["config_sha"],
    }
    schedule_hashes = {name: matrix[name]["schedule_hash"] for name in CELL_ORDER}
    suite_hashes = {
        **{name: matrix[name]["suite_hash"] for name in CELL_ORDER},
        "tactical": banks["tactical"]["positions"]["sha256"],
        "exploitability": banks["exploitability"]["positions"]["sha256"],
    }
    promotion_evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_contract": EVIDENCE_CONTRACT,
        "policy_version": policy["policy_version"],
        "policy_hash": policy_hash,
        "candidate_hash": candidate_hash,
        "champion_hash": champion_hash,
        "original_hash": original_hash,
        "confirmation_finalized": True,
        "evaluation_key": "matrix-" + canonical_sha256(specs),
        "config_hash": canonical_sha256(sorted(set(config_hashes.values()))),
        "schedule_hash": canonical_sha256(sorted(set(schedule_hashes.values()))),
        "look_number": look_number,
        "evaluation_stage": 3,
        "thresholds_overridden": False,
        "alpha_allocation_overridden": False,
        "attempt": attempt_value,
        "confirmation_matrix": matrix,
        **combined,
        "risk_differences": risk_differences,
        "discovery": discovery["summary"],
        "validity": validity,
        "exploitability": {
            key: value for key, value in stage0.items() if key != "validity"
        },
        "provenance": {
            "complete": True,
            "immutable_inputs": True,
            "immutable_original": True,
            "candidate_hash": candidate_hash,
            "champion_hash": champion_hash,
            "original_hash": original_hash,
            "policy_hash": policy_hash,
            "source_revision_hash": policy["frozen_plan"]["source_revision"],
            "binary_hash": binary_hash,
            "config_hashes": config_hashes,
            "schedule_hashes": schedule_hashes,
            "suite_hashes": suite_hashes,
            "discovery_schedule_hash": suite_manifest["discovery_schedule_hash"],
            "suite_manifest": suite_manifest,
            "suite_manifest_hash": suite_manifest_hash,
            "stage_0_probe_hash": stage0_probe_sha256,
            "stage_0_request_hash": stage0["request_hash"],
            "discovery_evidence_hash": discovery["artifact_hash"],
        },
    }
    return promotion_evidence


def build_controller_evidence(
    plan: Mapping[str, Any],
    promotion_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """Wrap promotion evidence in the exact controller adapter envelope."""

    expected_plan_keys = (
        "evaluationKey",
        "configHash",
        "scheduleHash",
        "policyHash",
        "policyPath",
        "policyVersion",
        "selfplayConfigHash",
        "topology",
        "stage",
        "look",
        "suiteManifestPath",
        "suiteManifestHash",
        "scheduleArtifacts",
        "specs",
    )
    missing = [key for key in expected_plan_keys if key not in plan]
    if missing:
        raise PromotionEvidenceError(f"controller plan is missing keys: {missing}")
    if plan["stage"] != "stage-3":
        raise PromotionEvidenceError(
            "promotion controller evidence requires a Stage-3 plan"
        )
    expected = {
        "candidate_hash": promotion_evidence.get("candidate_hash"),
        "tested_champion_hash": promotion_evidence.get("champion_hash"),
        "original_hash": promotion_evidence.get("original_hash"),
        "evaluation_key": promotion_evidence.get("evaluation_key"),
        "config_hash": promotion_evidence.get("config_hash"),
        "schedule_hash": promotion_evidence.get("schedule_hash"),
        "policy_hash": promotion_evidence.get("policy_hash"),
    }
    plan_expected = {
        "evaluation_key": plan["evaluationKey"],
        "config_hash": plan["configHash"],
        "schedule_hash": plan["scheduleHash"],
        "policy_hash": plan["policyHash"],
    }
    conflicts = [key for key, value in plan_expected.items() if expected[key] != value]
    if conflicts:
        raise PromotionEvidenceError(
            "promotion evidence contradicts controller plan: "
            + ", ".join(sorted(conflicts))
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_contract": EVIDENCE_CONTRACT,
        "finalized": True,
        "controller_stage": "confirmation",
        **expected,
        "policy_path": plan["policyPath"],
        "policy_version": plan["policyVersion"],
        "suite_manifest_path": plan["suiteManifestPath"],
        "suite_manifest_hash": plan["suiteManifestHash"],
        "look": plan["look"],
        "selfplay_config_hash": plan["selfplayConfigHash"],
        "topology": plan["topology"],
        "schedule_artifacts": json.loads(canonical_json(plan["scheduleArtifacts"])),
        "promotion_evidence": json.loads(canonical_json(promotion_evidence)),
    }


def publish_controller_evidence(output_path: Path, evidence: Mapping[str, Any]) -> bool:
    """Atomically create canonical evidence; exact retries are read-only success."""

    path = Path(output_path)
    if not path.parent.is_dir():
        raise PromotionEvidenceError(
            f"evidence output parent does not exist: {path.parent}"
        )
    data = (canonical_json(evidence) + "\n").encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise PromotionEvidenceError(
                f"existing evidence contradicts requested publication: {path}"
            )
        return True
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise PromotionEvidenceError(
                    f"concurrent evidence publication conflicts: {path}"
                )
            return True
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return False
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


# Stable descriptive API aliases for evaluator adapters.
assemble_promotion_evidence = build_promotion_evidence
assemble_controller_evidence = build_controller_evidence
publish_evidence = publish_controller_evidence
load_runner_cell = load_finalized_runner_cell
derive_stage_evidence = build_nonconfirmation_controller_evidence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble finalized EvaluationRunner bundles into evidence.json."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--runner-manifests",
        type=Path,
        help="Canonical JSON object mapping cell names to runner manifests",
    )
    parser.add_argument("--suite-manifest", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--stage0-probe", type=Path)
    parser.add_argument("--stage0-probe-sha256")
    parser.add_argument("--stage0-request", type=Path)
    parser.add_argument("--stage0-request-sha256")
    parser.add_argument("--discovery-evidence", type=Path)
    parser.add_argument("--discovery-evidence-sha256")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--bootstrap-replications", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("-o", "--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        plan = _load_canonical_json(args.plan, "controller plan")
        manifest_map: Dict[str, Any] = {}
        if args.runner_manifests is not None:
            manifest_map = _load_canonical_json(
                args.runner_manifests, "runner manifest map"
            )
            if not all(isinstance(value, str) for value in manifest_map.values()):
                raise PromotionEvidenceError(
                    "runner manifest map values must be file paths"
                )
        policy_path = args.policy or Path(plan["policyPath"])
        if plan.get("stage") == "stage-3":
            if (
                args.runner_manifests is None
                or args.stage0_probe is None
                or args.stage0_probe_sha256 is None
                or args.discovery_evidence is None
                or args.discovery_evidence_sha256 is None
                or args.attempt is None
            ):
                raise PromotionEvidenceError(
                    "Stage-3 requires runner manifests, Stage-0 probe, "
                    "discovery evidence, and attempt metadata"
                )
            discovery = _load_canonical_json(
                args.discovery_evidence, "discovery evidence"
            )
            discovery_hash = _require_sha256(
                args.discovery_evidence_sha256,
                "discovery evidence file hash",
            )
            if file_sha256(args.discovery_evidence) != discovery_hash:
                raise PromotionEvidenceError("discovery evidence file hash mismatch")
            attempt = _load_canonical_json(args.attempt, "attempt metadata")
            suite_manifest_path = args.suite_manifest or Path(plan["suiteManifestPath"])
            promotion = build_promotion_evidence(
                {name: Path(value) for name, value in manifest_map.items()},
                suite_manifest_path=suite_manifest_path,
                policy_path=policy_path,
                stage0_probe_path=args.stage0_probe,
                stage0_probe_sha256=args.stage0_probe_sha256,
                discovery_evidence=discovery,
                attempt=attempt,
                stage0_request_path=args.stage0_request,
                stage0_request_sha256=args.stage0_request_sha256,
                bootstrap_replications=args.bootstrap_replications,
                bootstrap_seed=args.bootstrap_seed,
            )
            evidence = build_controller_evidence(plan, promotion)
        else:
            evidence = build_nonconfirmation_controller_evidence(
                plan,
                runner_manifests=(
                    {name: Path(value) for name, value in manifest_map.items()}
                    if manifest_map
                    else None
                ),
                policy_path=policy_path,
                stage0_probe_path=args.stage0_probe,
                stage0_probe_sha256=args.stage0_probe_sha256,
                stage0_request_path=args.stage0_request,
                stage0_request_sha256=args.stage0_request_sha256,
                bootstrap_replications=args.bootstrap_replications,
                bootstrap_seed=args.bootstrap_seed,
            )
        reused = publish_controller_evidence(args.output, evidence)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha256(args.output),
                "reused": reused,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
