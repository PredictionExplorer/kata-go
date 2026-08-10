#!/usr/bin/env python3
"""Select high-agreement sources before expensive machine consensus."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from risk_score.board_symmetry import symmetry_orbit_sha256
from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    _analysis_map,
    _buffered_consensus_label,
    _canonical_jsonl,
    _normalized_positions,
    _publish_file,
    _suggest_specialized,
    analysis_features,
)
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256


PREFILTER_CONTRACT = "risk-score-consensus-source-prefilter-v1"
PREFILTER_ROLES = (
    "original/standard-2000",
    "original/powered-2000",
    "champion/standard-2000",
    "champion/powered-2000",
)
ALLOWED_LABELS = frozenset({"ordinary", "lead-40", "lead-80"})


def _require_regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    return source


def _analysis_identity(path: Path, role: str) -> Dict[str, Any]:
    source = _require_regular_file(path, f"{role} analysis")
    manifest_path = _require_regular_file(
        Path(str(source) + ".manifest.json"), f"{role} analysis manifest"
    )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("contract") != ANALYSIS_RUN_CONTRACT
        or value.get("output_path") != str(source.resolve())
        or value.get("output_sha256") != file_sha256(source)
    ):
        raise ValueError(f"{role} analysis manifest does not bind its output")
    query_path = value.get("query_path")
    query_hash = value.get("query_sha256")
    model_hash = value.get("model_sha256")
    if (
        not isinstance(query_path, str)
        or not isinstance(query_hash, str)
        or not isinstance(model_hash, str)
    ):
        raise ValueError(f"{role} analysis manifest identity is incomplete")
    query = _require_regular_file(Path(query_path), f"{role} query")
    if file_sha256(query) != query_hash:
        raise ValueError(f"{role} query changed after analysis")
    return {
        "path": str(source.resolve()),
        "sha256": file_sha256(source),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "query_path": str(query.resolve()),
        "query_sha256": query_hash,
        "model_sha256": model_hash,
        "katago_sha256": value.get("katago_sha256"),
        "config_sha256": value.get("config_sha256"),
    }


def _inside_buffered_band(label: str, score: float, buffer: float) -> bool:
    if label == "ordinary":
        return abs(score) < 25.0 - buffer
    if label == "lead-40":
        return 45.0 + buffer <= score < 75.0 - buffer
    if label == "lead-80":
        return score >= 85.0 + buffer
    raise ValueError(f"unsupported prefilter label: {label}")


def prefilter_consensus_sources(
    *,
    normalized_path: Path,
    analysis_paths: Mapping[str, Path],
    label: str,
    output_path: Path,
    manifest_path: Path,
    maximum_score_spread: float = 3.0,
    threshold_buffer: float = 5.0,
    limit: Optional[int] = None,
) -> Mapping[str, Any]:
    """Publish deterministic candidates likely to survive full v3 consensus.

    This is only an efficiency filter. Its output never satisfies machine review;
    every selected row must still pass the complete eight-role, all-symmetry
    consensus contract.
    """

    if label not in ALLOWED_LABELS:
        raise ValueError(f"label must be one of {sorted(ALLOWED_LABELS)}")
    if set(analysis_paths) != set(PREFILTER_ROLES):
        raise ValueError("prefilter requires exactly four model/mode analysis roles")
    if (
        isinstance(maximum_score_spread, bool)
        or not isinstance(maximum_score_spread, (int, float))
        or not math.isfinite(float(maximum_score_spread))
        or maximum_score_spread <= 0
    ):
        raise ValueError("maximum score spread must be finite and positive")
    if (
        isinstance(threshold_buffer, bool)
        or not isinstance(threshold_buffer, (int, float))
        or not math.isfinite(float(threshold_buffer))
        or threshold_buffer < 0
        or threshold_buffer >= 15
    ):
        raise ValueError("threshold buffer must be finite and between 0 and 15")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")

    normalized = _require_regular_file(normalized_path, "normalized positions")
    positions = _normalized_positions(normalized)
    position_ids = [position["semanticSha256"] for position in positions]
    identities = {
        role: _analysis_identity(Path(analysis_paths[role]), role)
        for role in PREFILTER_ROLES
    }
    model_hashes = {
        model: {
            identities[f"{model}/{mode}-2000"]["model_sha256"]
            for mode in ("standard", "powered")
        }
        for model in ("original", "champion")
    }
    if any(len(values) != 1 for values in model_hashes.values()):
        raise ValueError("prefilter model roles do not bind one model each")
    analyses = {
        role: _analysis_map(Path(analysis_paths[role]), position_ids, role)
        for role in PREFILTER_ROLES
    }

    selected = []
    selected_orbits = set()
    rejections: collections.Counter[str] = collections.Counter()
    for position in sorted(positions, key=lambda row: row["semanticSha256"]):
        semantic_hash = position["semanticSha256"]
        features = {
            role: analysis_features(
                analyses[role][semantic_hash], role, expected_visits=2000
            )
            for role in PREFILTER_ROLES
        }
        scores = [features[role]["score_lead"] for role in PREFILTER_ROLES]
        reasons = set()
        if max(scores) - min(scores) > float(maximum_score_spread):
            reasons.add("score_spread")
        if any(_buffered_consensus_label(score) != label for score in scores):
            reasons.add("label_disagreement")
        if any(
            not _inside_buffered_band(label, score, float(threshold_buffer))
            for score in scores
        ):
            reasons.add("threshold_buffer")
        if len({features[role]["top_move"] for role in PREFILTER_ROLES}) != 1:
            reasons.add("top_move_disagreement")
        for model in ("original", "champion"):
            if _suggest_specialized(
                features[f"{model}/standard-2000"],
                features[f"{model}/powered-2000"],
            ):
                reasons.add("specialized_signal")
        orbit_hash = symmetry_orbit_sha256(position)
        if orbit_hash in selected_orbits:
            reasons.add("symmetry_duplicate")
        if reasons:
            rejections.update(reasons)
            continue
        selected_orbits.add(orbit_hash)
        selected.append(position)
        if limit is not None and len(selected) >= limit:
            break

    if not selected:
        raise ValueError("consensus prefilter selected no positions")
    _publish_file(output_path, _canonical_jsonl(selected))
    manifest = {
        "schema_version": 1,
        "contract": PREFILTER_CONTRACT,
        "advisory_only": True,
        "requires_full_machine_consensus": True,
        "label": label,
        "maximum_score_spread": float(maximum_score_spread),
        "threshold_buffer": float(threshold_buffer),
        "limit": limit,
        "normalized": {
            "path": str(normalized.resolve()),
            "sha256": file_sha256(normalized),
            "row_count": len(positions),
            "semantic_ids_sha256": canonical_sha256(position_ids),
        },
        "analyses": identities,
        "model_hashes": {
            model: next(iter(values)) for model, values in model_hashes.items()
        },
        "selected": {
            "path": str(Path(output_path).resolve()),
            "sha256": file_sha256(output_path),
            "row_count": len(selected),
            "symmetry_orbit_count": len(selected_orbits),
        },
        "rejection_counts": dict(sorted(rejections.items())),
    }
    _publish_file(
        manifest_path, (canonical_json(manifest) + "\n").encode("utf-8")
    )
    return manifest


def _analysis_arguments(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path or role in result:
            raise ValueError("analysis arguments must be unique ROLE=PATH values")
        result[role] = Path(raw_path)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized", type=Path)
    parser.add_argument("--analysis", action="append", required=True)
    parser.add_argument("--label", choices=sorted(ALLOWED_LABELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-score-spread", type=float, default=3.0)
    parser.add_argument("--threshold-buffer", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prefilter_consensus_sources(
            normalized_path=args.normalized,
            analysis_paths=_analysis_arguments(args.analysis),
            label=args.label,
            output_path=args.output,
            manifest_path=args.manifest,
            maximum_score_spread=args.maximum_score_spread,
            threshold_buffer=args.threshold_buffer,
            limit=args.limit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
