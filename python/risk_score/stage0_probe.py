#!/usr/bin/env python3
"""Execute request-bound Stage-0 integrity and fixed-position probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from risk_score.curate_position_bank import (
    run_analysis,
    validate_deterministic_analysis_config,
)
from risk_score.paired_stats import load_policy
from risk_score.position_samples import (
    build_analysis_query,
    canonical_json,
    canonical_sha256,
    file_sha256,
    normalize_position_sample,
    semantic_position_sha256,
)
from risk_score.promotion_host import (
    HostCommandError,
    atomic_write_json,
)


def _load_json(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise HostCommandError(f"{role} must be a regular non-symlink file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HostCommandError(f"{role} must have an object root")
    return value


def _load_positions(path: Path) -> list[Dict[str, Any]]:
    positions = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            positions.append(
                normalize_position_sample(value, f"{path}:{line_number}")
            )
    return positions


def _suite_positions(
    suite_manifest_path: Path, bank_name: str, count: int
) -> list[Dict[str, Any]]:
    manifest = _load_json(suite_manifest_path, "suite manifest")
    matches = [
        bank
        for bank in manifest.get("banks", [])
        if isinstance(bank, Mapping)
        and bank.get("qualifiedName", bank.get("name")) == bank_name
    ]
    if len(matches) != 1:
        raise HostCommandError(f"suite manifest must contain one {bank_name!r} bank")
    relative = matches[0].get("positions", {}).get("path")
    if not isinstance(relative, str):
        raise HostCommandError(f"suite bank {bank_name!r} has no positions path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise HostCommandError("suite position path is unsafe")
    source = Path(suite_manifest_path).parent / path
    expected_hash = matches[0].get("positions", {}).get("sha256")
    if file_sha256(source) != expected_hash:
        raise HostCommandError(f"suite bank {bank_name!r} positions changed")
    positions = _load_positions(source)
    if len(positions) < count:
        raise HostCommandError(
            f"suite bank {bank_name!r} has {len(positions)} rows, needs {count}"
        )
    return positions[:count]


def _analysis_is_valid(record: Mapping[str, Any]) -> bool:
    root = record.get("rootInfo")
    moves = record.get("moveInfos")
    if not isinstance(root, Mapping) or not isinstance(moves, list) or not moves:
        return False
    for key in ("winrate", "scoreLead", "utility", "visits"):
        value = root.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return False
    for key in ("resultUtility", "scoreUtility", "otherUtility"):
        value = root.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return False
    if not _utility_decomposition_valid(root):
        return False
    if not 0.0 <= float(root["winrate"]) <= 1.0:
        return False
    top = moves[0]
    return isinstance(top, Mapping) and isinstance(top.get("move"), str)


def _utility_decomposition_valid(root: Mapping[str, Any]) -> bool:
    try:
        residual = abs(
            float(root["utility"])
            - float(root["resultUtility"])
            - float(root["scoreUtility"])
            - float(root["otherUtility"])
        )
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(residual) and residual <= 1e-4


def _policy_vector_valid(record: Mapping[str, Any], board_area: int = 361) -> bool:
    policy = record.get("policy")
    if not isinstance(policy, list) or len(policy) != board_area + 1:
        return False
    legal = [
        float(value)
        for value in policy
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and float(value) >= 0.0
    ]
    return (
        len(legal) > 0
        and all(math.isfinite(value) and value <= 1.0 for value in legal)
        and abs(sum(legal) - 1.0) <= 0.02
    )


def _hint_failures(
    positions: Sequence[Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int]:
    failures = 0
    unresolved = 0
    for position in positions:
        hint = position.get("hintLoc")
        record = results.get(semantic_position_sha256(position))
        if hint in {None, "", "null"} or record is None or not _analysis_is_valid(record):
            unresolved += 1
            continue
        top = record["moveInfos"][0]["move"]
        failures += int(str(top).upper() != str(hint).upper())
    return failures, unresolved


def _read_analysis(path: Path) -> Dict[str, Dict[str, Any]]:
    values = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        value = json.loads(raw)
        record_id = value.get("id")
        if not isinstance(record_id, str) or record_id in values:
            raise HostCommandError("analysis results have invalid IDs")
        values[record_id] = value
    return values


def _run_queries(
    *,
    root: Path,
    role: str,
    positions: Iterable[Mapping[str, Any]],
    visits: int,
    powered: bool,
    model: Path,
    katago: Path,
    config: Path,
    env: Mapping[str, str],
    subprocess_runner: Any,
) -> Dict[str, Dict[str, Any]]:
    query_path = root / f"{role}.queries.jsonl"
    rows = []
    for position in positions:
        query_id = semantic_position_sha256(position)
        rows.append(
            build_analysis_query(
                position,
                query_id=query_id,
                max_visits=visits,
                powered=powered,
            )
        )
    query_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = root / f"{role}.results.jsonl"
    run_analysis(
        katago=katago,
        config=config,
        model=model,
        queries=query_path,
        output=output,
        env=env,
        subprocess_runner=subprocess_runner,
    )
    values = _read_analysis(output)
    if set(values) != {row["id"] for row in rows}:
        raise HostCommandError(f"{role} analysis result IDs are incomplete")
    return values


def _mean_policy_distance(
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> float:
    distances = []
    for record_id in sorted(candidate):
        left = candidate[record_id].get("policy")
        right = reference[record_id].get("policy")
        if (
            not isinstance(left, list)
            or not isinstance(right, list)
            or len(left) != len(right)
        ):
            raise HostCommandError("analysis policy vectors are missing")
        distance = 0.5 * sum(
            abs(float(a) - float(b))
            for a, b in zip(left, right)
            if float(a) >= 0.0 and float(b) >= 0.0
        )
        if not math.isfinite(distance):
            raise HostCommandError("policy distance is non-finite")
        distances.append(distance)
    return sum(distances) / len(distances) if distances else 0.0


def run_stage0_probe(
    *,
    request_path: Path,
    request_sha256: str,
    katago: Path,
    analysis_config: Path,
    candidate_dir: Path,
    candidate_model: Path,
    candidate_model_sha256: str,
    champion_model: Path,
    champion_model_sha256: str,
    original_model: Path,
    original_model_sha256: str,
    powered_config: Path,
    powered_config_sha256: str,
    standard_config: Path,
    standard_config_sha256: str,
    policy_path: Path,
    policy_sha256: str,
    suite_manifest_path: Path,
    suite_manifest_sha256: str,
    output: Path,
    gpu_index: int = 7,
    subprocess_runner: Any = subprocess.run,
) -> Mapping[str, Any]:
    if type(gpu_index) is not int or gpu_index < 0:
        raise HostCommandError("Stage-0 GPU index must be nonnegative")
    validate_deterministic_analysis_config(analysis_config)
    request = _load_json(request_path, "Stage-0 request")
    if file_sha256(request_path) != request_sha256:
        raise HostCommandError("Stage-0 request hash mismatch")
    policy = load_policy(policy_path)
    if canonical_sha256(policy) != policy_sha256:
        raise HostCommandError("Stage-0 policy hash mismatch")
    files = (
        (candidate_model, candidate_model_sha256, "candidate model"),
        (champion_model, champion_model_sha256, "champion model"),
        (original_model, original_model_sha256, "original model"),
        (powered_config, powered_config_sha256, "powered config"),
        (standard_config, standard_config_sha256, "standard config"),
        (suite_manifest_path, suite_manifest_sha256, "suite manifest"),
    )
    for path, expected, role in files:
        source = Path(path)
        if source.is_symlink() or not source.is_file() or file_sha256(source) != expected:
            raise HostCommandError(f"{role} hash/path mismatch")
    checkpoint = Path(candidate_dir) / "model.ckpt"
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_file()
        or file_sha256(checkpoint) != request.get("checkpoint_hash")
    ):
        raise HostCommandError("candidate checkpoint contradicts Stage-0 request")
    from risk_score.promotion_controller import inspect_candidate

    if inspect_candidate(Path(candidate_dir)).directory_manifest_hash != request.get(
        "candidate_manifest_hash"
    ):
        raise HostCommandError("candidate directory manifest contradicts Stage-0 request")
    expected_request = {
        "candidate_hash": candidate_model_sha256,
        "tested_champion_hash": champion_model_sha256,
        "original_hash": original_model_sha256,
        "policy_hash": policy_sha256,
        "suite_manifest_hash": suite_manifest_sha256,
        "powered_config_hash": powered_config_sha256,
        "standard_config_hash": standard_config_sha256,
    }
    if "katago_binary_hash" in request:
        expected_request.update(
            {
                "katago_binary_path": str(Path(katago)),
                "katago_binary_hash": file_sha256(Path(katago)),
                "analysis_config_path": str(Path(analysis_config)),
                "analysis_config_hash": file_sha256(Path(analysis_config)),
            }
        )
    if any(request.get(key) != value for key, value in expected_request.items()):
        raise HostCommandError("Stage-0 request identities contradict argv")
    stage = policy["evaluation_stages"]["stage_0_integrity_and_fixed_probes"]
    fixed_count = stage["fixed_analysis_positions"]
    fixed_visits = stage["fixed_analysis_visits"]
    machine_review_v3 = (
        policy.get("policy_version") == "risk-seeking-checkpoint-promotion-v3"
    )
    exploit_count = (
        0 if machine_review_v3 else stage["exploitability_sentinel_positions"]
    )
    exploit_visits = 0 if machine_review_v3 else stage["exploitability_sentinel_visits"]
    fixed = _suite_positions(suite_manifest_path, "audit", fixed_count)
    exploit = (
        []
        if machine_review_v3
        else _suite_positions(suite_manifest_path, "exploitability", exploit_count)
    )
    tactical = (
        []
        if machine_review_v3
        else _suite_positions(suite_manifest_path, "tactical", 1)
    )
    analysis_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
    with tempfile.TemporaryDirectory(prefix="risk-score-stage0-") as temporary:
        root = Path(temporary)
        candidate_standard = _run_queries(
            root=root,
            role="candidate-standard-fixed",
            positions=fixed,
            visits=fixed_visits,
            powered=False,
            model=candidate_model,
            katago=katago,
            config=analysis_config,
            env=analysis_env,
            subprocess_runner=subprocess_runner,
        )
        candidate_powered = _run_queries(
            root=root,
            role="candidate-powered-fixed",
            positions=fixed,
            visits=fixed_visits,
            powered=True,
            model=candidate_model,
            katago=katago,
            config=analysis_config,
            env=analysis_env,
            subprocess_runner=subprocess_runner,
        )
        champion_standard = _run_queries(
            root=root,
            role="champion-standard-fixed",
            positions=fixed,
            visits=fixed_visits,
            powered=False,
            model=champion_model,
            katago=katago,
            config=analysis_config,
            env=analysis_env,
            subprocess_runner=subprocess_runner,
        )
        original_smoke = _run_queries(
            root=root,
            role="original-smoke",
            positions=fixed[:1],
            visits=4,
            powered=False,
            model=original_model,
            katago=katago,
            config=analysis_config,
            env=analysis_env,
            subprocess_runner=subprocess_runner,
        )
        exploit_candidate = (
            {}
            if machine_review_v3
            else _run_queries(
                root=root,
                role="candidate-powered-exploit",
                positions=exploit,
                visits=exploit_visits,
                powered=True,
                model=candidate_model,
                katago=katago,
                config=analysis_config,
                env=analysis_env,
                subprocess_runner=subprocess_runner,
            )
        )
        tactical_candidate = (
            {}
            if machine_review_v3
            else _run_queries(
                root=root,
                role="candidate-powered-tactical",
                positions=tactical,
                visits=exploit_visits,
                powered=True,
                model=candidate_model,
                katago=katago,
                config=analysis_config,
                env=analysis_env,
                subprocess_runner=subprocess_runner,
            )
        )
        stability_positions = fixed[: min(32, len(fixed))]
        stability_candidate = _run_queries(
            root=root,
            role="candidate-powered-stability",
            positions=stability_positions,
            visits=max(fixed_visits + 1, fixed_visits * 2),
            powered=True,
            model=candidate_model,
            katago=katago,
            config=analysis_config,
            env=analysis_env,
            subprocess_runner=subprocess_runner,
        )
        all_records = (
            list(candidate_standard.values())
            + list(candidate_powered.values())
            + list(champion_standard.values())
            + list(original_smoke.values())
            + list(exploit_candidate.values())
            + list(tactical_candidate.values())
            + list(stability_candidate.values())
        )
        invalid = sum(not _analysis_is_valid(record) for record in all_records)
        policy_violations = sum(
            not _policy_vector_valid(record) for record in all_records
        )
        distance = _mean_policy_distance(candidate_standard, champion_standard)
        endpoint_violations = sum(
            abs(float(record["rootInfo"]["scoreLead"])) > 1000.0
            for record in all_records
            if _analysis_is_valid(record)
        )
        tactical_failures, tactical_unresolved = _hint_failures(
            tactical, tactical_candidate
        )
        exploit_failures, exploit_unresolved = _hint_failures(
            exploit, exploit_candidate
        )
        stability_failures = 0
        for position in stability_positions:
            record_id = semantic_position_sha256(position)
            low = candidate_powered[record_id]
            high = stability_candidate[record_id]
            if (
                low["moveInfos"][0]["move"] != high["moveInfos"][0]["move"]
                or abs(
                    float(low["rootInfo"]["scoreLead"])
                    - float(high["rootInfo"]["scoreLead"])
                )
                > 20.0
            ):
                stability_failures += 1
        decomposition_violations = sum(
            not isinstance(record.get("rootInfo"), Mapping)
            or not _utility_decomposition_valid(record["rootInfo"])
            for record in all_records
        )
        expected_players = {
            semantic_position_sha256(position): (
                position["nextPla"]
                if len(position["movePlas"]) % 2 == 0
                else ("W" if position["nextPla"] == "B" else "B")
            )
            for position in fixed
        }
        perspective_violations = sum(
            candidate_standard[record_id]["rootInfo"].get("currentPlayer")
            != player
            for record_id, player in expected_players.items()
        )
        win_weight = float(policy["objective"]["win_weight"])
        clamp_violations = 0
        for record in all_records:
            if not _analysis_is_valid(record):
                continue
            root_info = record["rootInfo"]
            clamp_violations += int(
                abs(float(root_info["resultUtility"])) > win_weight + 1e-6
            )
            for key in ("lowerScoreTailProb", "upperScoreTailProb"):
                if key in root_info:
                    clamp_violations += int(
                        not 0.0 <= float(root_info[key]) <= 1.0
                    )
    calculated_checks = {
        "architecture_compatibility": policy_violations == 0,
        "checkpoint_hash": True,
        "cuda_load": len(all_records) > 0,
        "endpoint_tail": endpoint_violations == 0,
        "finite_outputs": invalid == 0,
        "legal_bounds": (
            invalid == 0 and policy_violations == 0 and clamp_violations == 0
        ),
        "model_hash": True,
        "perspective": perspective_violations == 0,
        "policy_distance": math.isfinite(distance) and 0.0 <= distance <= 1.0,
        "utility_decomposition": decomposition_violations == 0,
    }
    checks = {
        name: calculated_checks.get(name, False)
        for name in stage["required_checks"]
    }
    measurements = {
        "fixed_analysis_positions": fixed_count,
        "fixed_analysis_visits": fixed_visits,
        "exploitability_sentinel_positions": exploit_count,
        "exploitability_sentinel_visits": exploit_visits,
        "hard_tactical_failures": tactical_failures,
        "hard_exploitability_failures": exploit_failures,
        "unresolved_failures": (
            invalid + tactical_unresolved + exploit_unresolved
        ),
        "model_runtime_errors": 0,
        "perspective_violations": perspective_violations,
        "clamp_violations": clamp_violations,
        "endpoint_violations": endpoint_violations,
        "nonfinite_violations": invalid + policy_violations,
        "decomposition_violations": decomposition_violations,
        "selected_move_endpoint_mass_dominated": endpoint_violations > 0,
        "visit_stability_acceptable": stability_failures == 0,
        "mean_policy_distance": distance,
        "visit_stability_failures": stability_failures,
    }
    result = {
        "schema_version": 1,
        "contract": "risk-score-stage-0-probe-output-v1",
        "finalized": True,
        "candidate_hash": candidate_model_sha256,
        "tested_champion_hash": champion_model_sha256,
        "original_hash": original_model_sha256,
        "policy_hash": policy_sha256,
        "request_hash": request_sha256,
        "katago_binary_hash": file_sha256(Path(katago)),
        "analysis_config_hash": file_sha256(Path(analysis_config)),
        "checks": dict(sorted(checks.items())),
        "measurements": measurements,
    }
    atomic_write_json(output, result)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--katago", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--candidate-model-sha256", required=True)
    parser.add_argument("--champion-model", required=True, type=Path)
    parser.add_argument("--champion-model-sha256", required=True)
    parser.add_argument("--original-model", required=True, type=Path)
    parser.add_argument("--original-model-sha256", required=True)
    parser.add_argument("--powered-config", required=True, type=Path)
    parser.add_argument("--powered-config-sha256", required=True)
    parser.add_argument("--standard-config", required=True, type=Path)
    parser.add_argument("--standard-config-sha256", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--suite-manifest", required=True, type=Path)
    parser.add_argument("--suite-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-index", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_stage0_probe(
            request_path=args.request,
            request_sha256=args.request_sha256,
            katago=args.katago,
            analysis_config=args.analysis_config,
            candidate_dir=args.candidate_dir,
            candidate_model=args.candidate_model,
            candidate_model_sha256=args.candidate_model_sha256,
            champion_model=args.champion_model,
            champion_model_sha256=args.champion_model_sha256,
            original_model=args.original_model,
            original_model_sha256=args.original_model_sha256,
            powered_config=args.powered_config,
            powered_config_sha256=args.powered_config_sha256,
            standard_config=args.standard_config,
            standard_config_sha256=args.standard_config_sha256,
            policy_path=args.policy,
            policy_sha256=args.policy_sha256,
            suite_manifest_path=args.suite_manifest,
            suite_manifest_sha256=args.suite_manifest_sha256,
            output=args.output,
            gpu_index=args.gpu_index,
        )
    except (HostCommandError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
