#!/usr/bin/env python3
"""Select high-agreement sources before expensive machine consensus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from risk_score.board_symmetry import symmetry_orbit_sha256
from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    _analysis_map,
    _buffered_consensus_label,
    _canonical_jsonl,
    _load_json,
    _normalized_positions,
    _publish_bundle,
    _publish_file,
    _suggest_specialized,
    analysis_features,
    build_analysis_query,
    validate_deterministic_analysis_config,
)
from risk_score.paired_stats import load_policy
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256


PREFILTER_CONTRACT = "risk-score-consensus-source-prefilter-v1"
PREFILTER_QUERY_BUNDLE_CONTRACT = "risk-score-consensus-prefilter-query-bundle-v1"
PREFILTER_ROLES = (
    "original/standard-2000",
    "original/powered-2000",
    "champion/standard-2000",
    "champion/powered-2000",
)
ALLOWED_LABELS = frozenset({"ordinary", "lead-40", "lead-80"})


def generate_prefilter_query_bundle(
    *,
    normalized_path: Path,
    output_dir: Path,
    katago_path: Path,
    config_path: Path,
    model_path: Path,
    policy_path: Path,
) -> Mapping[str, Any]:
    """Publish the two 2,000-visit roles used by the advisory prefilter.

    Unlike full machine consensus, this intentionally analyzes only each
    normalized position's canonical orientation. The selected rows remain
    advisory and must later pass all-symmetry, eight-role consensus.
    """

    frozen = {
        "normalized": _require_regular_file(normalized_path, "normalized positions"),
        "katago": _require_regular_file(katago_path, "KataGo binary"),
        "config": _require_regular_file(config_path, "analysis config"),
        "model": _require_regular_file(model_path, "prefilter model"),
        "policy": _require_regular_file(policy_path, "promotion policy"),
    }
    validate_deterministic_analysis_config(frozen["config"])
    policy = load_policy(frozen["policy"])
    positions = _normalized_positions(frozen["normalized"])
    ids = [position["semanticSha256"] for position in positions]
    if len(ids) != len(set(ids)):
        raise ValueError("normalized positions contain duplicate semantic IDs")
    frozen_hashes = {name: file_sha256(path) for name, path in frozen.items()}
    files: Dict[str, bytes] = {}
    queries = {}
    for role, powered in (("standard-2000", False), ("powered-2000", True)):
        data = _canonical_jsonl(
            build_analysis_query(
                position,
                query_id=position["semanticSha256"],
                max_visits=2000,
                powered=powered,
            )
            for position in positions
        )
        relative = f"queries/{role}.jsonl"
        files[relative] = data
        queries[role] = {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "row_count": len(positions),
            "ids_sha256": canonical_sha256(ids),
            "powered": powered,
            "visits": 2000,
        }
    manifest = {
        "schema_version": 1,
        "contract": PREFILTER_QUERY_BUNDLE_CONTRACT,
        "source_path": str(frozen["normalized"].resolve()),
        "source_sha256": frozen_hashes["normalized"],
        "position_count": len(positions),
        "semantic_ids_sha256": canonical_sha256(ids),
        "katago_path": str(frozen["katago"].resolve()),
        "katago_sha256": frozen_hashes["katago"],
        "config_path": str(frozen["config"].resolve()),
        "config_sha256": frozen_hashes["config"],
        "model_path": str(frozen["model"].resolve()),
        "model_sha256": frozen_hashes["model"],
        "policy_path": str(frozen["policy"].resolve()),
        "policy_sha256": frozen_hashes["policy"],
        "policy_hash": canonical_sha256(policy),
        "queries": queries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    files["manifest.json"] = (canonical_json(manifest) + "\n").encode("utf-8")
    for name, path in frozen.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != frozen_hashes[name]
        ):
            raise ValueError(f"{name} changed while generating prefilter queries")
    _publish_bundle(Path(output_dir), files)
    return manifest


def _require_regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    return source


def _require_sha256(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _canonical_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise ValueError(f"{role} must be an absolute canonical path")
    return path


def _analysis_artifact(
    path: Path,
    role: str,
    positions: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source = _require_regular_file(path, f"{role} analysis")
    manifest_path = _require_regular_file(
        Path(str(source) + ".manifest.json"), f"{role} analysis manifest"
    )
    value = _load_json(manifest_path, f"{role} analysis manifest")
    if value.get("contract") != ANALYSIS_RUN_CONTRACT:
        raise ValueError(f"{role} analysis manifest contract is unsupported")
    if "schema_version" in value and value.get("schema_version") != 1:
        raise ValueError(f"{role} analysis manifest schema is unsupported")
    if "manifest_sha256" in value:
        payload = dict(value)
        identity = _require_sha256(
            payload.pop("manifest_sha256"), f"{role} analysis manifest self-hash"
        )
        if identity != canonical_sha256(payload):
            raise ValueError(f"{role} analysis manifest self-hash is invalid")

    output_hash = file_sha256(source)
    output_path = _canonical_path(
        value.get("output_path"), f"{role} analysis output path"
    )
    bound_output_hash = _require_sha256(
        value.get("output_sha256"), f"{role} analysis output hash"
    )
    if output_path != source.resolve() or bound_output_hash != output_hash:
        raise ValueError(f"{role} analysis manifest does not bind its output")

    query = _require_regular_file(
        _canonical_path(value.get("query_path"), f"{role} query path"),
        f"{role} query",
    )
    query_hash = _require_sha256(value.get("query_sha256"), f"{role} query hash")
    if file_sha256(query) != query_hash:
        raise ValueError(f"{role} query changed after analysis")

    powered = role.endswith("/powered-2000")
    expected_queries = [
        build_analysis_query(
            position,
            query_id=position["semanticSha256"],
            max_visits=2000,
            powered=powered,
        )
        for position in positions
    ]
    if query.read_bytes() != _canonical_jsonl(expected_queries):
        raise ValueError(f"{role} query positions, settings, or exact IDs are invalid")

    position_ids = [position["semanticSha256"] for position in positions]
    analyses = _analysis_map(source, position_ids, role)
    if "row_count" in value and (
        type(value.get("row_count")) is not int
        or value.get("row_count") != len(analyses)
    ):
        raise ValueError(f"{role} analysis manifest row count is invalid")

    model_hash = _require_sha256(value.get("model_sha256"), f"{role} model hash")
    katago_hash = _require_sha256(value.get("katago_sha256"), f"{role} KataGo hash")
    config_hash = _require_sha256(value.get("config_sha256"), f"{role} config hash")
    return {
        "path": str(source.resolve()),
        "sha256": output_hash,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "query_path": str(query.resolve()),
        "query_sha256": query_hash,
        "model_sha256": model_hash,
        "katago_sha256": katago_hash,
        "config_sha256": config_hash,
    }, analyses


def _inside_buffered_band(label: str, score: float, buffer: float) -> bool:
    if label == "ordinary":
        return abs(score) < 25.0 - buffer
    if label == "lead-40":
        return 45.0 + buffer <= score < 75.0 - buffer
    if label == "lead-80":
        return score >= 85.0 + buffer
    raise ValueError(f"unsupported prefilter label: {label}")


def _validate_selection_arguments(
    *,
    label: Any,
    maximum_score_spread: Any,
    threshold_buffer: Any,
    limit: Any,
) -> Tuple[str, float, float, Optional[int]]:
    if not isinstance(label, str) or label not in ALLOWED_LABELS:
        raise ValueError(f"label must be one of {sorted(ALLOWED_LABELS)}")
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
    return (
        str(label),
        float(maximum_score_spread),
        float(threshold_buffer),
        limit,
    )


def _load_prefilter_inputs(
    *,
    positions: Sequence[Mapping[str, Any]],
    analysis_paths: Mapping[str, Path],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, str],
]:
    if set(analysis_paths) != set(PREFILTER_ROLES):
        raise ValueError("prefilter requires exactly four model/mode analysis roles")
    identities: Dict[str, Dict[str, Any]] = {}
    analyses: Dict[str, Dict[str, Any]] = {}
    for role in PREFILTER_ROLES:
        identity, records = _analysis_artifact(
            Path(analysis_paths[role]), role, positions
        )
        identities[role] = identity
        analyses[role] = records

    model_hashes: Dict[str, str] = {}
    for model in ("original", "champion"):
        values = {
            identities[f"{model}/{mode}-2000"]["model_sha256"]
            for mode in ("standard", "powered")
        }
        if len(values) != 1:
            raise ValueError("prefilter model roles do not bind one model each")
        model_hashes[model] = next(iter(values))
    if len({identity["config_sha256"] for identity in identities.values()}) != 1:
        raise ValueError("prefilter analysis roles do not bind one config")
    if len({identity["katago_sha256"] for identity in identities.values()}) != 1:
        raise ValueError("prefilter analysis roles do not bind one KataGo binary")
    return identities, analyses, model_hashes


def _compute_prefilter_selection(
    *,
    positions: Sequence[Mapping[str, Any]],
    analyses: Mapping[str, Mapping[str, Mapping[str, Any]]],
    label: str,
    maximum_score_spread: float,
    threshold_buffer: float,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Purely recompute all advisory prefilter decisions."""

    selected: List[Mapping[str, Any]] = []
    selected_orbits = set()
    rejections: collections.Counter[str] = collections.Counter()
    for position in sorted(positions, key=lambda row: row["semanticSha256"]):
        semantic_hash = position["semanticSha256"]
        try:
            features = {
                role: analysis_features(
                    analyses[role][semantic_hash], role, expected_visits=2000
                )
                for role in PREFILTER_ROLES
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"analysis features for {semantic_hash} are malformed"
            ) from exc
        scores = [features[role]["score_lead"] for role in PREFILTER_ROLES]
        reasons = set()
        if max(scores) - min(scores) > maximum_score_spread:
            reasons.add("score_spread")
        if any(_buffered_consensus_label(score) != label for score in scores):
            reasons.add("label_disagreement")
        if any(
            not _inside_buffered_band(label, score, threshold_buffer)
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

    data = _canonical_jsonl(selected)
    return {
        "selected": selected,
        "selected_data": data,
        "selected_sha256": hashlib.sha256(data).hexdigest(),
        "selected_orbit_count": len(selected_orbits),
        "rejection_counts": dict(sorted(rejections.items())),
    }


def _prefilter_manifest_payload(
    *,
    normalized_path: Path,
    normalized_sha256: str,
    positions: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    model_hashes: Mapping[str, str],
    label: str,
    maximum_score_spread: float,
    threshold_buffer: float,
    limit: Optional[int],
    output_path: Path,
    computation: Mapping[str, Any],
) -> Dict[str, Any]:
    position_ids = [position["semanticSha256"] for position in positions]
    return {
        "schema_version": 1,
        "contract": PREFILTER_CONTRACT,
        "advisory_only": True,
        "requires_full_machine_consensus": True,
        "label": label,
        "maximum_score_spread": maximum_score_spread,
        "threshold_buffer": threshold_buffer,
        "limit": limit,
        "normalized": {
            "path": str(normalized_path.resolve()),
            "sha256": normalized_sha256,
            "row_count": len(positions),
            "semantic_ids_sha256": canonical_sha256(position_ids),
        },
        "analyses": dict(identities),
        "model_hashes": dict(model_hashes),
        "selected": {
            "path": str(output_path.resolve()),
            "sha256": computation["selected_sha256"],
            "row_count": len(computation["selected"]),
            "symmetry_orbit_count": computation["selected_orbit_count"],
        },
        "rejection_counts": computation["rejection_counts"],
    }


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
    allow_empty: bool = False,
) -> Mapping[str, Any]:
    """Publish deterministic candidates likely to survive full v3 consensus.

    This is only an efficiency filter. Its output never satisfies machine review;
    every selected row must still pass the complete eight-role, all-symmetry
    consensus contract.
    """

    label, maximum_score_spread, threshold_buffer, limit = (
        _validate_selection_arguments(
            label=label,
            maximum_score_spread=maximum_score_spread,
            threshold_buffer=threshold_buffer,
            limit=limit,
        )
    )
    if type(allow_empty) is not bool:
        raise ValueError("allow_empty must be a boolean")

    normalized = _require_regular_file(normalized_path, "normalized positions")
    positions = _normalized_positions(normalized)
    identities, analyses, model_hashes = _load_prefilter_inputs(
        positions=positions, analysis_paths=analysis_paths
    )
    computation = _compute_prefilter_selection(
        positions=positions,
        analyses=analyses,
        label=label,
        maximum_score_spread=maximum_score_spread,
        threshold_buffer=threshold_buffer,
        limit=limit,
    )
    if not computation["selected"] and not allow_empty:
        raise ValueError("consensus prefilter selected no positions")
    _publish_file(output_path, computation["selected_data"])
    manifest = _prefilter_manifest_payload(
        normalized_path=normalized,
        normalized_sha256=file_sha256(normalized),
        positions=positions,
        identities=identities,
        model_hashes=model_hashes,
        label=label,
        maximum_score_spread=maximum_score_spread,
        threshold_buffer=threshold_buffer,
        limit=limit,
        output_path=Path(output_path),
        computation=computation,
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _publish_file(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def validate_prefilter_artifact(
    manifest_path: Path,
    *,
    expected_label: Optional[str] = None,
    expected_model_hashes: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    """Recompute and validate a published advisory prefilter artifact."""

    manifest_file = _require_regular_file(manifest_path, "consensus prefilter manifest")
    manifest = _load_json(manifest_file, "consensus prefilter manifest")
    payload = dict(manifest)
    has_self_hash = "manifest_sha256" in payload
    supplied_self_hash = payload.pop("manifest_sha256", None)
    if has_self_hash:
        identity = _require_sha256(
            supplied_self_hash, "consensus prefilter manifest self-hash"
        )
        if identity != canonical_sha256(payload):
            raise ValueError("consensus prefilter manifest self-hash is invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != PREFILTER_CONTRACT
        or payload.get("advisory_only") is not True
        or payload.get("requires_full_machine_consensus") is not True
    ):
        raise ValueError("consensus prefilter manifest contract is unsupported")

    label, maximum_score_spread, threshold_buffer, limit = (
        _validate_selection_arguments(
            label=payload.get("label"),
            maximum_score_spread=payload.get("maximum_score_spread"),
            threshold_buffer=payload.get("threshold_buffer"),
            limit=payload.get("limit"),
        )
    )
    if expected_label is not None:
        if not isinstance(expected_label, str) or expected_label not in ALLOWED_LABELS:
            raise ValueError(f"expected_label must be one of {sorted(ALLOWED_LABELS)}")
        if label != expected_label:
            raise ValueError("consensus prefilter label does not match expectation")

    normalized_value = payload.get("normalized")
    if not isinstance(normalized_value, Mapping):
        raise ValueError("consensus prefilter normalized binding is malformed")
    normalized = _require_regular_file(
        _canonical_path(
            normalized_value.get("path"),
            "consensus prefilter normalized path",
        ),
        "consensus prefilter normalized positions",
    )
    positions = _normalized_positions(normalized)

    analyses_value = payload.get("analyses")
    if not isinstance(analyses_value, Mapping) or set(analyses_value) != set(
        PREFILTER_ROLES
    ):
        raise ValueError("consensus prefilter analysis inventory is incomplete")
    analysis_paths: Dict[str, Path] = {}
    for role in PREFILTER_ROLES:
        identity = analyses_value[role]
        if not isinstance(identity, Mapping):
            raise ValueError(f"consensus prefilter analysis {role} is malformed")
        analysis_paths[role] = _canonical_path(
            identity.get("path"), f"consensus prefilter analysis {role} path"
        )
    identities, analyses, model_hashes = _load_prefilter_inputs(
        positions=positions, analysis_paths=analysis_paths
    )

    if expected_model_hashes is not None:
        if not isinstance(expected_model_hashes, Mapping) or set(
            expected_model_hashes
        ) != {"original", "champion"}:
            raise ValueError(
                "expected_model_hashes must name exactly original and champion"
            )
        expected_models = {
            model: _require_sha256(
                expected_model_hashes[model], f"expected {model} model hash"
            )
            for model in ("original", "champion")
        }
        if model_hashes != expected_models:
            raise ValueError(
                "consensus prefilter model hashes do not match expectation"
            )

    computation = _compute_prefilter_selection(
        positions=positions,
        analyses=analyses,
        label=label,
        maximum_score_spread=maximum_score_spread,
        threshold_buffer=threshold_buffer,
        limit=limit,
    )
    selected_value = payload.get("selected")
    if not isinstance(selected_value, Mapping):
        raise ValueError("consensus prefilter selected binding is malformed")
    selected_path = _require_regular_file(
        _canonical_path(
            selected_value.get("path"), "consensus prefilter selected path"
        ),
        "consensus prefilter selected output",
    )
    if selected_path.read_bytes() != computation["selected_data"]:
        raise ValueError(
            "consensus prefilter selected JSONL bytes do not match recomputation"
        )

    expected_payload = _prefilter_manifest_payload(
        normalized_path=normalized,
        normalized_sha256=file_sha256(normalized),
        positions=positions,
        identities=identities,
        model_hashes=model_hashes,
        label=label,
        maximum_score_spread=maximum_score_spread,
        threshold_buffer=threshold_buffer,
        limit=limit,
        output_path=selected_path,
        computation=computation,
    )
    if canonical_json(payload) != canonical_json(expected_payload):
        qualifier = "legacy " if not has_self_hash else ""
        raise ValueError(
            f"{qualifier}consensus prefilter manifest fields do not match recomputation"
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
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "generate-queries":
        parser = argparse.ArgumentParser(
            description="Generate canonical-orientation advisory prefilter queries."
        )
        parser.add_argument("normalized", type=Path)
        parser.add_argument("--output-dir", required=True, type=Path)
        parser.add_argument("--katago", required=True, type=Path)
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("--model", required=True, type=Path)
        parser.add_argument("--policy", required=True, type=Path)
        result = parser.parse_args(values[1:])
        result.command = "generate-queries"
        return result
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized", type=Path)
    parser.add_argument("--analysis", action="append", required=True)
    parser.add_argument("--label", choices=sorted(ALLOWED_LABELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-score-spread", type=float, default=3.0)
    parser.add_argument("--threshold-buffer", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-empty", action="store_true")
    result = parser.parse_args(values)
    result.command = "select"
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "generate-queries":
            manifest = generate_prefilter_query_bundle(
                normalized_path=args.normalized,
                output_dir=args.output_dir,
                katago_path=args.katago,
                config_path=args.config,
                model_path=args.model,
                policy_path=args.policy,
            )
        else:
            manifest = prefilter_consensus_sources(
                normalized_path=args.normalized,
                analysis_paths=_analysis_arguments(args.analysis),
                label=args.label,
                output_path=args.output,
                manifest_path=args.manifest,
                maximum_score_spread=args.maximum_score_spread,
                threshold_buffer=args.threshold_buffer,
                limit=args.limit,
                allow_empty=args.allow_empty,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
