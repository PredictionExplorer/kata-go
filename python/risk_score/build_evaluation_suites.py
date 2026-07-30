#!/usr/bin/env python3
"""Build immutable, content-addressed risk-score evaluation suites."""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from risk_score.generate_schedule import (
    GENERATOR_CONTRACT as SCHEDULE_GENERATOR_CONTRACT,
)
from risk_score.generate_schedule import (
    build_schedule,
    validate_position,
)
from risk_score.position_samples import (
    semantic_position as _shared_semantic_position,
    semantic_position_sha256 as _shared_semantic_position_sha256,
)

SCHEMA_VERSION = 2
GENERATOR_CONTRACT = "risk-score-content-addressed-evaluation-suites-v2"
LEGACY_GENERATOR_CONTRACT = "risk-score-content-addressed-evaluation-suites-v1"
MANIFEST_CONTRACT = "risk-score-authoritative-evaluation-manifest-v2"
_V1_POLICY_PATH = Path(__file__).with_name("promotion_policy_v1.json")
_V2_POLICY_PATH = Path(__file__).with_name("promotion_policy_v2.json")
DEFAULT_POLICY_PATH = _V2_POLICY_PATH if _V2_POLICY_PATH.exists() else _V1_POLICY_PATH
ORDINARY_LABEL = "ordinary"
ORDINARY_BANKS = ("discovery", "confirmation", "audit")
POLICY_HOLDOUTS = ORDINARY_BANKS
RISK_LABELS = ("lead-40", "lead-80")
SPECIALIZED_LABELS = (
    "lead-40",
    "lead-80",
    "tactical",
    "exploitability",
    "baits",
    "tails",
    "sacrifice",
    "small-gain",
    "adversarial",
)
ALL_LABELS = frozenset((ORDINARY_LABEL,) + SPECIALIZED_LABELS)
STAGE_3_CELL_ORDER = (
    "powered_candidate_vs_champion",
    "powered_candidate_vs_original",
    "standard_candidate_vs_original",
    "lead_40",
    "lead_80",
)
STAGE_1_CELL_ORDER = ("powered_candidate_vs_champion",)
STAGE_2_CELL_ORDER = (
    "powered_candidate_vs_champion",
    "powered_candidate_vs_original",
    "lead_40",
    "lead_80",
)
STAGE_3_CELL_DEFAULTS = {
    "powered_candidate_vs_champion": {
        "comparison": "candidate-vs-champion-powered",
        "suite": "confirmation",
        "search_mode": "powered",
    },
    "powered_candidate_vs_original": {
        "comparison": "candidate-vs-original-powered",
        "suite": "confirmation",
        "search_mode": "powered",
    },
    "standard_candidate_vs_original": {
        "comparison": "candidate-vs-original-standard",
        "suite": "confirmation",
        "search_mode": "standard",
    },
    "lead_40": {
        "comparison": "candidate-vs-champion-powered-lead-40",
        "suite": "lead-40",
        "search_mode": "powered",
    },
    "lead_80": {
        "comparison": "candidate-vs-champion-powered-lead-80",
        "suite": "lead-80",
        "search_mode": "powered",
    },
}
_STAGE_3_PAIR_KEYS = {
    "powered_candidate_vs_champion": "powered_ordinary_color_pairs_per_matchup",
    "powered_candidate_vs_original": "powered_ordinary_color_pairs_per_matchup",
    "standard_candidate_vs_original": "standard_ordinary_color_pairs",
    "lead_40": "lead_40_color_pairs",
    "lead_80": "lead_80_color_pairs",
}
_LABEL_ALIASES = {
    "lead40": "lead-40",
    "lead80": "lead-80",
    "bait": "baits",
    "low-probability-high-score-bait": "baits",
    "low-probability-high-score-baits": "baits",
    "tail": "tails",
    "exaggerated-score-tail": "tails",
    "exaggerated-score-tails": "tails",
    "sacrifices": "sacrifice",
    "whole-board-sacrifice-trap": "sacrifice",
    "whole-board-sacrifice-traps": "sacrifice",
    "small-gain-large-lead-risk": "small-gain",
    "small-gain-large-lead-risks": "small-gain",
    "ordinary-tactical-refutation": "tactical",
    "ordinary-tactical-refutations": "tactical",
    "adversarial-continuation": "adversarial",
    "adversarial-continuations": "adversarial",
}


@dataclass(frozen=True)
class SuiteBuildResult:
    """The identity and location of a published suite bundle."""

    output_dir: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    reused: bool = False


@dataclass(frozen=True)
class _LabeledPosition:
    position: Dict[str, Any]
    labels: Tuple[str, ...]
    content_sha256: str
    semantic_sha256: str
    source_name: str
    source_line: int


@dataclass(frozen=True)
class _PolicySuitePlan:
    policy: Mapping[str, Any]
    policy_hash: str
    policy_version: str
    source_revision: str
    exact_quota_contract: bool
    holdout_quotas: Mapping[str, Mapping[str, int]]
    stage_3_looks: Tuple[Mapping[str, Any], ...]


def canonical_json(value: Any) -> str:
    """Serialize JSON according to the suite's hashing contract."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_position(position: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only fields that determine the replayed gameplay position.

    Labels, sampling weights, hints, and provenance are deliberately excluded so
    the same board/history cannot leak across independent holdout banks merely
    by changing annotations.
    """

    return _shared_semantic_position(position)


def semantic_position_sha256(position: Mapping[str, Any]) -> str:
    return _shared_semantic_position_sha256(position)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _positive_policy_count(value: Any, source: str, *, allow_zero: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (0 if allow_zero else 1)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{source} must be a {qualifier} integer")
    return value


def _look_cluster_map(
    stage_3: Mapping[str, Any],
    look: Mapping[str, Any],
    look_number: int,
) -> Optional[Mapping[str, Any]]:
    direct = look.get("minimum_independent_position_clusters")
    if direct is not None:
        if not isinstance(direct, dict):
            raise ValueError(
                "Stage-3 look minimum_independent_position_clusters must be an object"
            )
        return direct

    configured = stage_3.get("minimum_independent_position_clusters")
    if configured is None:
        return None
    if not isinstance(configured, dict):
        raise ValueError(
            "Stage-3 minimum_independent_position_clusters must be an object"
        )
    for key in (
        str(look_number),
        f"look-{look_number}",
        f"look_{look_number}",
    ):
        value = configured.get(key)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(
                    "Stage-3 per-look independent-cluster quota must be an object"
                )
            return value
    if set(STAGE_3_CELL_ORDER).issubset(configured):
        return configured
    return None


def _look_pair_map(look: Mapping[str, Any]) -> Dict[str, int]:
    declared = look.get("exact_color_pairs")
    if declared is None:
        declared = look.get("color_pairs")
    if declared is not None and not isinstance(declared, dict):
        raise ValueError("Stage-3 look exact color-pair quotas must be an object")

    result: Dict[str, int] = {}
    for cell_name, legacy_key in _STAGE_3_PAIR_KEYS.items():
        value = declared.get(cell_name) if isinstance(declared, dict) else None
        if value is None:
            value = look.get(legacy_key)
        result[cell_name] = _positive_policy_count(
            value,
            f"Stage-3 look {look.get('look_number')!r} {cell_name} color pairs",
        )
    return result


def _load_policy_binding(path: Path) -> _PolicySuitePlan:
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load promotion policy {path}: {exc}") from exc
    if not isinstance(policy, dict):
        raise ValueError("promotion policy root must be an object")
    policy_version = policy.get("policy_version")
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("promotion policy has no nonempty policy_version")
    frozen_plan = policy.get("frozen_plan")
    source_revision = (
        frozen_plan.get("source_revision") if isinstance(frozen_plan, dict) else None
    )
    if (
        not isinstance(source_revision, str)
        or len(source_revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("promotion policy has no valid frozen source revision")

    evaluation_stages = policy.get("evaluation_stages")
    if not isinstance(evaluation_stages, dict):
        raise ValueError("promotion policy has no evaluation_stages object")
    stage_3 = evaluation_stages.get("stage_3_promotion_confirmation")
    if not isinstance(stage_3, dict):
        raise ValueError("promotion policy has no Stage-3 confirmation object")
    raw_looks = stage_3.get("looks")
    if not isinstance(raw_looks, list) or not raw_looks:
        raise ValueError("promotion policy Stage-3 looks must be a nonempty list")

    parsed_looks: List[Dict[str, Any]] = []
    seen_looks: Set[int] = set()
    exact_quota_contract = False
    previous_pairs: Optional[Dict[str, int]] = None
    previous_clusters: Optional[Dict[str, int]] = None
    for raw_look in raw_looks:
        if not isinstance(raw_look, dict):
            raise ValueError("promotion policy Stage-3 look must be an object")
        look_number = _positive_policy_count(
            raw_look.get("look_number"), "Stage-3 look_number"
        )
        if look_number in seen_looks:
            raise ValueError(f"duplicate Stage-3 look_number {look_number}")
        seen_looks.add(look_number)
        pairs = _look_pair_map(raw_look)
        cluster_map = _look_cluster_map(stage_3, raw_look, look_number)
        if cluster_map is None:
            clusters = dict(pairs)
        else:
            exact_quota_contract = True
            missing = sorted(set(STAGE_3_CELL_ORDER).difference(cluster_map))
            extra = sorted(set(cluster_map).difference(STAGE_3_CELL_ORDER))
            if missing or extra:
                raise ValueError(
                    "Stage-3 minimum_independent_position_clusters keys differ; "
                    f"missing={missing}, extra={extra}"
                )
            clusters = {
                cell_name: _positive_policy_count(
                    cluster_map[cell_name],
                    (
                        f"Stage-3 look {look_number} {cell_name} "
                        "minimum independent clusters"
                    ),
                )
                for cell_name in STAGE_3_CELL_ORDER
            }
        for cell_name in STAGE_3_CELL_ORDER:
            if clusters[cell_name] > pairs[cell_name]:
                raise ValueError(
                    f"Stage-3 look {look_number} {cell_name} requires "
                    f"{clusters[cell_name]} independent clusters for only "
                    f"{pairs[cell_name]} color pairs"
                )
        if previous_pairs is not None:
            for cell_name in STAGE_3_CELL_ORDER:
                if pairs[cell_name] < previous_pairs[cell_name]:
                    raise ValueError(
                        "Stage-3 looks must be cumulative; "
                        f"{cell_name} decreases at look {look_number}"
                    )
                if clusters[cell_name] < previous_clusters[cell_name]:
                    raise ValueError(
                        "Stage-3 independent-cluster minima must be cumulative; "
                        f"{cell_name} decreases at look {look_number}"
                    )
        parsed_looks.append(
            {
                "look_number": look_number,
                "color_pairs": pairs,
                "minimum_independent_position_clusters": clusters,
            }
        )
        previous_pairs = pairs
        previous_clusters = clusters
    parsed_looks.sort(key=lambda value: value["look_number"])
    if [look["look_number"] for look in parsed_looks] != list(
        range(1, len(parsed_looks) + 1)
    ):
        raise ValueError("Stage-3 look numbers must be contiguous starting at one")

    if "v2" in policy_version.lower() and not exact_quota_contract:
        raise ValueError(
            "v2 promotion policy must declare "
            "minimum_independent_position_clusters per Stage-3 cell"
        )

    stage_1 = evaluation_stages.get("stage_1_cheap_paired_screen", {})
    stage_2 = evaluation_stages.get("stage_2_finalist_selection", {})
    deep_audit = evaluation_stages.get("deep_audit", {})
    for name, value in (
        ("Stage-1", stage_1),
        ("Stage-2", stage_2),
        ("deep-audit", deep_audit),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"promotion policy {name} settings must be an object")

    latest = parsed_looks[-1]["color_pairs"]
    holdout_quotas = {
        ORDINARY_LABEL: {
            "discovery": max(
                _positive_policy_count(
                    stage_1.get("ordinary_color_pairs"),
                    "Stage-1 ordinary_color_pairs",
                ),
                _positive_policy_count(
                    stage_2.get("ordinary_color_pairs"),
                    "Stage-2 ordinary_color_pairs",
                ),
            ),
            "confirmation": max(
                latest["powered_candidate_vs_champion"],
                latest["powered_candidate_vs_original"],
                latest["standard_candidate_vs_original"],
            ),
            "audit": _positive_policy_count(
                deep_audit.get("ordinary_color_pairs"),
                "deep-audit ordinary_color_pairs",
            ),
        },
        "lead-40": {
            "discovery": _positive_policy_count(
                stage_2.get("lead_40_color_pairs"),
                "Stage-2 lead_40_color_pairs",
            ),
            "confirmation": latest["lead_40"],
            "audit": _positive_policy_count(
                deep_audit.get("lead_40_color_pairs", 0),
                "deep-audit lead_40_color_pairs",
                allow_zero=True,
            ),
        },
        "lead-80": {
            "discovery": _positive_policy_count(
                stage_2.get("lead_80_color_pairs"),
                "Stage-2 lead_80_color_pairs",
            ),
            "confirmation": latest["lead_80"],
            "audit": _positive_policy_count(
                deep_audit.get("lead_80_color_pairs", 0),
                "deep-audit lead_80_color_pairs",
                allow_zero=True,
            ),
        },
    }
    return _PolicySuitePlan(
        policy=policy,
        policy_hash=canonical_sha256(policy),
        policy_version=policy_version,
        source_revision=source_revision,
        exact_quota_contract=exact_quota_contract,
        holdout_quotas=holdout_quotas,
        stage_3_looks=tuple(parsed_looks),
    )


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _normalize_label(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: labels must be nonempty strings")
    label = value.strip().lower().replace("_", "-")
    label = _LABEL_ALIASES.get(label, label)
    if label not in ALL_LABELS:
        raise ValueError(
            f"{source}: unsupported evaluation label {value!r}; expected one of "
            f"{', '.join(sorted(ALL_LABELS))}"
        )
    return label


def _labels_from_value(value: Any, source: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_normalize_label(value, source)]
    if isinstance(value, (list, tuple)):
        return [_normalize_label(item, source) for item in value]
    raise ValueError(
        f"{source}: label declaration must be a string or array of strings"
    )


def _extract_labels(
    row: Mapping[str, Any], position: Mapping[str, Any], source: str
) -> Tuple[str, ...]:
    declarations: List[str] = []
    for container in (row, position):
        if "label" in container:
            declarations.extend(_labels_from_value(container["label"], source))
        if "labels" in container:
            declarations.extend(_labels_from_value(container["labels"], source))

    metadata = position.get("metadata")
    if isinstance(metadata, (str, list, tuple)):
        declarations.extend(_labels_from_value(metadata, source))
    elif isinstance(metadata, dict):
        if "label" in metadata:
            declarations.extend(_labels_from_value(metadata["label"], source))
        if "labels" in metadata:
            declarations.extend(_labels_from_value(metadata["labels"], source))
    elif metadata is not None:
        raise ValueError(
            f"{source}: metadata must be a label string, label array, "
            "or object with labels"
        )

    labels = tuple(sorted(set(declarations)))
    if not labels:
        raise ValueError(f"{source}: position has no evaluation label")
    if ORDINARY_LABEL in labels and len(labels) != 1:
        raise ValueError(
            f"{source}: ordinary cannot be combined with specialized labels"
        )
    return labels


def _load_labeled_positions(
    paths: Sequence[Path],
) -> Tuple[List[_LabeledPosition], List[Dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one labeled PositionSample JSONL source is required")

    positions: List[_LabeledPosition] = []
    sources: List[Dict[str, Any]] = []
    seen_content: Dict[str, str] = {}
    seen_semantic: Dict[str, str] = {}
    source_names: Set[str] = set()

    for path in paths:
        source_name = path.name
        if source_name in source_names:
            raise ValueError(
                f"source basenames must be unique for stable manifests: {source_name!r}"
            )
        source_names.add(source_name)
        row_count = 0
        blank_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    blank_count += 1
                    continue
                row_count += 1
                source = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{source}: row must be a JSON object")

                if "position" in row:
                    position_value = row["position"]
                    if not isinstance(position_value, dict):
                        raise ValueError(
                            f"{source}: position wrapper value must be an object"
                        )
                    position = dict(position_value)
                else:
                    position = dict(row)
                validate_position(position, source)
                labels = _extract_labels(row, position, source)
                content_hash = canonical_sha256(position)
                semantic_hash = semantic_position_sha256(position)
                previous_source = seen_content.get(content_hash)
                if previous_source is not None:
                    raise ValueError(
                        f"{source}: duplicate position content SHA-256 {content_hash}; "
                        f"first seen at {previous_source}"
                    )
                semantic_source = seen_semantic.get(semantic_hash)
                if semantic_source is not None:
                    raise ValueError(
                        f"{source}: duplicate gameplay-semantic position SHA-256 "
                        f"{semantic_hash}; first seen at {semantic_source}"
                    )
                seen_content[content_hash] = source
                seen_semantic[semantic_hash] = source
                positions.append(
                    _LabeledPosition(
                        position=position,
                        labels=labels,
                        content_sha256=content_hash,
                        semantic_sha256=semantic_hash,
                        source_name=source_name,
                        source_line=line_number,
                    )
                )

        sources.append(
            {
                "name": source_name,
                "sha256": file_sha256(path),
                "rowCount": row_count,
                "blankLineCount": blank_count,
            }
        )

    if not positions:
        raise ValueError("no labeled PositionSample rows were loaded")
    return positions, sorted(sources, key=lambda source: source["name"])


def _validate_sha256(value: str, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{source} must be a 64-character SHA-256")
    return value.lower()


def _ordinary_counts(total: int, weights: Mapping[str, float]) -> Dict[str, int]:
    if total < len(ORDINARY_BANKS):
        raise ValueError(
            f"at least {len(ORDINARY_BANKS)} ordinary positions are required to keep "
            "discovery, confirmation, and audit independent"
        )
    if set(weights) != set(ORDINARY_BANKS):
        raise ValueError(
            "ordinary weights must contain exactly discovery, confirmation, and audit"
        )
    checked: Dict[str, float] = {}
    for bank in ORDINARY_BANKS:
        value = weights[bank]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"ordinary weight {bank!r} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"ordinary weight {bank!r} must be finite and positive")
        checked[bank] = number

    remaining = total - len(ORDINARY_BANKS)
    weight_sum = sum(checked.values())
    exact = {bank: remaining * checked[bank] / weight_sum for bank in ORDINARY_BANKS}
    counts = {bank: 1 + int(math.floor(exact[bank])) for bank in ORDINARY_BANKS}
    unassigned = total - sum(counts.values())
    remainder_order = sorted(
        ORDINARY_BANKS,
        key=lambda bank: (
            -(exact[bank] - math.floor(exact[bank])),
            ORDINARY_BANKS.index(bank),
        ),
    )
    for bank in remainder_order[:unassigned]:
        counts[bank] += 1
    return counts


def _semantic_order(
    positions: Iterable[_LabeledPosition],
    *,
    seed: str,
    purpose: str,
    bank: str,
) -> List[_LabeledPosition]:
    return sorted(
        positions,
        key=lambda item: (
            canonical_sha256(
                {
                    "generatorContract": GENERATOR_CONTRACT,
                    "purpose": purpose,
                    "seed": seed,
                    "bank": bank,
                    "semanticSha256": item.semantic_sha256,
                }
            ),
            item.semantic_sha256,
            item.content_sha256,
        ),
    )


def _split_exact_policy_holdouts(
    positions: Sequence[_LabeledPosition],
    *,
    seed: str,
    holdout_quotas: Mapping[str, Mapping[str, int]],
) -> Tuple[Dict[str, List[_LabeledPosition]], Set[str]]:
    banks: Dict[str, List[_LabeledPosition]] = {}
    assigned_content_hashes: Set[str] = set()
    for label in (ORDINARY_LABEL,) + RISK_LABELS:
        quotas = holdout_quotas.get(label)
        if not isinstance(quotas, Mapping) or set(quotas) != set(POLICY_HOLDOUTS):
            raise ValueError(
                f"policy quotas for {label!r} must contain exactly "
                "discovery, confirmation, and audit"
            )
        selected = [
            item
            for item in positions
            if (
                item.labels == (ORDINARY_LABEL,)
                if label == ORDINARY_LABEL
                else label in item.labels
            )
        ]
        if label in RISK_LABELS:
            overlapping = [
                item
                for item in selected
                if all(risk_label in item.labels for risk_label in RISK_LABELS)
            ]
            if overlapping:
                raise ValueError(
                    "policy holdout positions may not carry both lead-40 and "
                    "lead-80 labels because combined Lead inference requires "
                    "independent clusters"
                )
        ordered = _semantic_order(
            selected,
            seed=seed,
            purpose="policy-holdout-assignment",
            bank=label,
        )
        required = sum(int(quotas[holdout]) for holdout in POLICY_HOLDOUTS)
        if len(ordered) < required:
            raise ValueError(
                f"insufficient independent {label} positions for policy quotas: "
                f"need {required}, found {len(ordered)}"
            )
        offset = 0
        for holdout in POLICY_HOLDOUTS:
            count = int(quotas[holdout])
            group = ordered[offset : offset + count]
            offset += count
            bank_name = holdout if label == ORDINARY_LABEL else f"{label}-{holdout}"
            banks[bank_name] = _semantic_order(
                group,
                seed=seed,
                purpose="policy-holdout-order",
                bank=bank_name,
            )
            assigned_content_hashes.update(item.content_sha256 for item in group)

    for label in SPECIALIZED_LABELS:
        if label in RISK_LABELS:
            continue
        selected = [item for item in positions if label in item.labels]
        if selected:
            banks[label] = _semantic_order(
                selected,
                seed=seed,
                purpose="specialized-bank-order",
                bank=label,
            )
            assigned_content_hashes.update(item.content_sha256 for item in selected)
    return banks, assigned_content_hashes


def split_labeled_positions(
    positions: Sequence[_LabeledPosition],
    *,
    seed: str,
    ordinary_weights: Optional[Mapping[str, float]] = None,
    holdout_quotas: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> Dict[str, List[_LabeledPosition]]:
    """Deterministically form disjoint ordinary banks and specialized suites."""

    if not isinstance(seed, str) or not seed or "\n" in seed or "\r" in seed:
        raise ValueError("seed must be a nonempty single-line string")
    if holdout_quotas is not None:
        banks, _ = _split_exact_policy_holdouts(
            positions,
            seed=seed,
            holdout_quotas=holdout_quotas,
        )
        return banks
    weights = (
        dict(ordinary_weights)
        if ordinary_weights is not None
        else {bank: 1.0 for bank in ORDINARY_BANKS}
    )

    ordinary = [item for item in positions if item.labels == (ORDINARY_LABEL,)]
    counts = _ordinary_counts(len(ordinary), weights)
    ordinary_order = _semantic_order(
        ordinary,
        seed=seed,
        purpose="ordinary-bank-assignment",
        bank=ORDINARY_LABEL,
    )

    banks: Dict[str, List[_LabeledPosition]] = {}
    offset = 0
    for bank in ORDINARY_BANKS:
        selected = ordinary_order[offset : offset + counts[bank]]
        offset += counts[bank]
        banks[bank] = _semantic_order(
            selected,
            seed=seed,
            purpose="bank-order",
            bank=bank,
        )

    for label in SPECIALIZED_LABELS:
        selected = [item for item in positions if label in item.labels]
        if selected:
            banks[label] = _semantic_order(
                selected,
                seed=seed,
                purpose="specialized-bank-order",
                bank=label,
            )
    return banks


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_existing_bundle(
    output_dir: Path, expected_manifest: Mapping[str, Any]
) -> None:
    if not output_dir.is_dir():
        raise FileExistsError(f"refusing to overwrite non-directory {output_dir}")
    manifest_path = output_dir / "manifest.json"
    expected_data = (canonical_json(expected_manifest) + "\n").encode("utf-8")
    try:
        existing_data = manifest_path.read_bytes()
        existing = json.loads(existing_data.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"refusing to overwrite unverifiable suite bundle {output_dir}: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise FileExistsError(
            f"refusing to overwrite non-UTF-8 suite manifest {manifest_path}: {exc}"
        ) from exc
    if existing != expected_manifest or existing_data != expected_data:
        raise FileExistsError(
            "refusing to overwrite suite bundle with contradictory manifest: "
            f"{output_dir}"
        )
    for bank in existing.get("banks", []):
        for artifact_name in ("positions", "schedule"):
            artifact = bank[artifact_name]
            path = output_dir / artifact["path"]
            if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                raise ValueError(
                    f"existing suite artifact contradicts manifest: {artifact['path']}"
                )
    verified_schedules: Set[str] = set()
    for cell in existing.get("cells", []):
        if not isinstance(cell, dict):
            raise ValueError("existing suite manifest has malformed cell entry")
        relative = cell.get("schedule_path")
        expected_hash = cell.get("schedule_hash")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("existing suite manifest cell has no schedule binding")
        if relative in verified_schedules:
            continue
        path = output_dir / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(
                f"existing suite cell schedule contradicts manifest: {relative}"
            )
        verified_schedules.add(relative)


def build_evaluation_suites(
    source_paths: Sequence[Path],
    output_dir: Path,
    *,
    seed: str,
    ordinary_weights: Optional[Mapping[str, float]] = None,
    pairs_per_position: int = 1,
    bot_a_index: int = 0,
    bot_b_index: int = 1,
    exclude_content_hashes: Sequence[str] = (),
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> SuiteBuildResult:
    """Build and atomically publish deterministic position banks and schedules."""

    output_dir = Path(output_dir)
    sources = [Path(path) for path in source_paths]
    if pairs_per_position <= 0:
        raise ValueError("pairs_per_position must be positive")

    policy_plan = _load_policy_binding(Path(policy_path))
    if policy_plan.exact_quota_contract and pairs_per_position != 1:
        raise ValueError(
            "exact policy suite builds require pairs_per_position=1 so every "
            "Stage-3 color pair is an independent gameplay-semantic cluster"
        )
    positions, source_manifest = _load_labeled_positions(sources)
    exclusions = {
        _validate_sha256(value, "excluded content hash")
        for value in exclude_content_hashes
    }
    known_hashes = {item.content_sha256 for item in positions}
    unknown_exclusions = sorted(exclusions.difference(known_hashes))
    if unknown_exclusions:
        raise ValueError(
            f"excluded content hashes were not found: {unknown_exclusions}"
        )

    excluded_rows = [item for item in positions if item.content_sha256 in exclusions]
    included_rows = [
        item for item in positions if item.content_sha256 not in exclusions
    ]
    weights = (
        dict(ordinary_weights)
        if ordinary_weights is not None
        else {bank: 1.0 for bank in ORDINARY_BANKS}
    )
    if policy_plan.exact_quota_contract:
        banks, assigned_content_hashes = _split_exact_policy_holdouts(
            included_rows,
            seed=seed,
            holdout_quotas=policy_plan.holdout_quotas,
        )
    else:
        banks = split_labeled_positions(
            included_rows,
            seed=seed,
            ordinary_weights=weights,
        )
        assigned_content_hashes = {
            item.content_sha256 for selected in banks.values() for item in selected
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-", dir=str(output_dir.parent)
        )
    )
    try:
        assert temporary is not None
        positions_dir = temporary / "position-banks"
        schedules_dir = temporary / "schedules"
        positions_dir.mkdir()
        schedules_dir.mkdir()
        bank_manifest: List[Dict[str, Any]] = []
        bank_artifacts: Dict[str, Dict[str, Any]] = {}
        schedule_rows_by_bank: Dict[str, Tuple[Dict[str, Any], ...]] = {}

        if policy_plan.exact_quota_contract:
            exact_order = (
                *ORDINARY_BANKS,
                "lead-40-discovery",
                "lead-40-confirmation",
                "lead-40-audit",
                "lead-80-discovery",
                "lead-80-confirmation",
                "lead-80-audit",
            )
            bank_order = tuple(dict.fromkeys((*exact_order, *sorted(banks))))
        else:
            bank_order = ORDINARY_BANKS + SPECIALIZED_LABELS

        for qualified_bank_name in bank_order:
            selected = banks.get(qualified_bank_name)
            if not selected:
                continue
            if (
                policy_plan.exact_quota_contract
                and qualified_bank_name.startswith("lead-")
                and qualified_bank_name.rsplit("-", 1)[-1] in POLICY_HOLDOUTS
            ):
                source_label, holdout = qualified_bank_name.rsplit("-", 1)
                bank_name = (
                    source_label if holdout == "confirmation" else qualified_bank_name
                )
                suite_name = source_label
            else:
                source_label = (
                    ORDINARY_LABEL
                    if qualified_bank_name in ORDINARY_BANKS
                    else qualified_bank_name
                )
                holdout = (
                    qualified_bank_name
                    if qualified_bank_name in ORDINARY_BANKS
                    else None
                )
                bank_name = qualified_bank_name
                suite_name = qualified_bank_name
            position_rows = [item.position for item in selected]
            position_relative = Path("position-banks") / f"{qualified_bank_name}.jsonl"
            position_data = _canonical_jsonl(position_rows)
            position_bank_sha = hashlib.sha256(position_data).hexdigest()
            _write_fsynced(temporary / position_relative, position_data)

            schedule_seed = "risk-score-suite-v2-" + canonical_sha256(
                {
                    "generatorContract": GENERATOR_CONTRACT,
                    "purpose": "schedule-seed",
                    "masterSeed": seed,
                    "bank": qualified_bank_name,
                    "bankSha256": position_bank_sha,
                }
            )
            schedule_rows = build_schedule(
                position_rows,
                bot_a_index=bot_a_index,
                bot_b_index=bot_b_index,
                pairs_per_position=pairs_per_position,
                base_seed=schedule_seed,
            )
            for schedule_row in schedule_rows:
                selected_position = selected[schedule_row["sourcePositionIndex"]]
                schedule_row.update(
                    {
                        "suite": suite_name,
                        "suiteBank": suite_name,
                        "suiteBankSha256": position_bank_sha,
                        "suiteQualifiedName": qualified_bank_name,
                        "suiteHoldout": holdout,
                        "positionContentSha256": selected_position.content_sha256,
                        "positionSemanticSha256": selected_position.semantic_sha256,
                        "independentClusterId": selected_position.semantic_sha256,
                    }
                )
            schedule_relative = Path("schedules") / f"{qualified_bank_name}.jsonl"
            schedule_data = _canonical_jsonl(schedule_rows)
            _write_fsynced(temporary / schedule_relative, schedule_data)
            schedule_hash = hashlib.sha256(schedule_data).hexdigest()
            position_ids = list(
                dict.fromkeys(row["positionId"] for row in schedule_rows)
            )
            independent_cluster_ids = [item.semantic_sha256 for item in selected]

            entry = {
                "name": bank_name,
                "qualifiedName": qualified_bank_name,
                "sourceLabel": source_label,
                "holdout": holdout,
                "kind": (
                    "ordinary" if source_label == ORDINARY_LABEL else "specialized"
                ),
                "contentSha256s": [item.content_sha256 for item in selected],
                "semanticSha256s": independent_cluster_ids,
                "independentClusterIds": independent_cluster_ids,
                "independentClusterIdsSha256": canonical_sha256(
                    independent_cluster_ids
                ),
                "positionIds": position_ids,
                "positionIdsSha256": canonical_sha256(position_ids),
                "positions": {
                    "path": position_relative.as_posix(),
                    "sha256": position_bank_sha,
                    "rowCount": len(position_rows),
                },
                "schedule": {
                    "path": schedule_relative.as_posix(),
                    "sha256": schedule_hash,
                    "rowCount": len(schedule_rows),
                    "pairCount": len(schedule_rows) // 2,
                    "scheduleId": schedule_rows[0]["scheduleId"],
                    "baseSeed": schedule_seed,
                },
            }
            bank_manifest.append(entry)
            bank_artifacts[qualified_bank_name] = entry
            schedule_rows_by_bank[qualified_bank_name] = tuple(schedule_rows)

        cell_manifest: List[Dict[str, Any]] = []
        prefix_dir = schedules_dir / "prefixes"
        prefix_artifacts: Dict[Tuple[str, int], Dict[str, Any]] = {}

        def schedule_prefix(
            qualified_bank_name: str, pair_count: int
        ) -> Dict[str, Any]:
            cache_key = (qualified_bank_name, pair_count)
            cached = prefix_artifacts.get(cache_key)
            if cached is not None:
                return cached
            bank = bank_artifacts.get(qualified_bank_name)
            rows = schedule_rows_by_bank.get(qualified_bank_name)
            if bank is None or rows is None:
                raise ValueError(
                    f"policy cell references missing bank {qualified_bank_name!r}"
                )
            full_pair_count = len(rows) // 2
            if pair_count > full_pair_count:
                raise ValueError(
                    f"policy cell needs {pair_count} pairs from "
                    f"{qualified_bank_name}, which has {full_pair_count}"
                )
            prefix_rows = rows[: 2 * pair_count]
            if len(prefix_rows) != 2 * pair_count:
                raise AssertionError("complete-pair schedule prefix was truncated")
            if pair_count == full_pair_count:
                artifact = dict(bank["schedule"])
            else:
                if not prefix_dir.exists():
                    prefix_dir.mkdir()
                relative = (
                    Path("schedules")
                    / "prefixes"
                    / f"{qualified_bank_name}-pairs-{pair_count}.jsonl"
                )
                data = _canonical_jsonl(prefix_rows)
                _write_fsynced(temporary / relative, data)
                artifact = {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "rowCount": len(prefix_rows),
                    "pairCount": pair_count,
                    "scheduleId": prefix_rows[0]["scheduleId"],
                    "baseSeed": bank["schedule"]["baseSeed"],
                }
            artifact["rows"] = prefix_rows
            prefix_artifacts[cache_key] = artifact
            return artifact

        if policy_plan.exact_quota_contract:
            stages = policy_plan.policy["evaluation_stages"]
            stage_1 = stages["stage_1_cheap_paired_screen"]
            stage_2 = stages["stage_2_finalist_selection"]
            stage_3 = stages["stage_3_promotion_confirmation"]
            matrix = policy_plan.policy.get("required_confirmation_matrix")
            if not isinstance(matrix, dict) or set(matrix) != set(STAGE_3_CELL_ORDER):
                raise ValueError(
                    "exact policy required_confirmation_matrix must contain "
                    "the five Stage-3 cells"
                )
            latest_look_number = policy_plan.stage_3_looks[-1]["look_number"]

            def append_cell(
                *,
                cell_name: str,
                stage: str,
                look: str,
                comparison: str,
                suite: str,
                search_mode: str,
                qualified_bank_name: str,
                pair_count: int,
                minimum_clusters: int,
                visits: int,
                maximal_look_schedule: bool,
            ) -> None:
                for value, source in (
                    (comparison, "comparison"),
                    (suite, "suite"),
                    (search_mode, "search_mode"),
                ):
                    if not isinstance(value, str) or not value:
                        raise ValueError(
                            f"{stage} {cell_name} {source} must be nonempty"
                        )
                visits = _positive_policy_count(
                    visits, f"{stage} {cell_name} visits"
                )
                artifact = schedule_prefix(qualified_bank_name, pair_count)
                prefix_rows = artifact["rows"]
                pair_ids = list(dict.fromkeys(row["pairId"] for row in prefix_rows))
                position_ids = sorted(set(row["positionId"] for row in prefix_rows))
                independent_cluster_ids = list(
                    dict.fromkeys(
                        row["independentClusterId"] for row in prefix_rows
                    )
                )
                if len(pair_ids) != pair_count:
                    raise AssertionError(f"{stage} prefix has wrong pair count")
                if len(independent_cluster_ids) != pair_count:
                    raise ValueError(
                        f"{stage} {cell_name} does not use one pair per "
                        "independent gameplay-semantic cluster"
                    )
                if len(independent_cluster_ids) < minimum_clusters:
                    raise ValueError(
                        f"{stage} {cell_name} has only "
                        f"{len(independent_cluster_ids)} independent clusters; "
                        f"policy requires {minimum_clusters}"
                    )
                bank = bank_artifacts[qualified_bank_name]
                cell_payload = {
                    "cell_name": cell_name,
                    "stage": stage,
                    "look": look,
                    "comparison": comparison,
                    "suite": suite,
                    "search_mode": search_mode,
                    "visits": visits,
                    "color_pairs": pair_count,
                    "minimum_independent_position_clusters": minimum_clusters,
                    "independent_cluster_ids": independent_cluster_ids,
                    "independent_cluster_ids_hash": canonical_sha256(
                        independent_cluster_ids
                    ),
                    "position_ids": position_ids,
                    "position_ids_hash": canonical_sha256(position_ids),
                    "bank_name": qualified_bank_name,
                    "bank_path": bank["positions"]["path"],
                    "bank_hash": bank["positions"]["sha256"],
                    "schedule_path": artifact["path"],
                    "schedule_hash": artifact["sha256"],
                    "schedule_id": artifact["scheduleId"],
                    "schedule_row_count": artifact["rowCount"],
                    "maximal_look_schedule": maximal_look_schedule,
                    "policy_hash": policy_plan.policy_hash,
                    "policy_version": policy_plan.policy_version,
                    "source_revision": policy_plan.source_revision,
                }
                cell_manifest.append(
                    {
                        "cell_id": "suite-cell-" + canonical_sha256(cell_payload),
                        **cell_payload,
                    }
                )

            stage_1_pairs = _positive_policy_count(
                stage_1.get("ordinary_color_pairs"),
                "Stage-1 ordinary_color_pairs",
            )
            append_cell(
                cell_name="powered_candidate_vs_champion",
                stage="stage-1",
                look="automatic",
                comparison="candidate-vs-champion-powered",
                suite="discovery",
                search_mode="powered",
                qualified_bank_name="discovery",
                pair_count=stage_1_pairs,
                minimum_clusters=stage_1_pairs,
                visits=stage_1.get("visits"),
                maximal_look_schedule=False,
            )

            stage_2_ordinary_pairs = _positive_policy_count(
                stage_2.get("ordinary_color_pairs"),
                "Stage-2 ordinary_color_pairs",
            )
            stage_2_lead_40_pairs = _positive_policy_count(
                stage_2.get("lead_40_color_pairs"),
                "Stage-2 lead_40_color_pairs",
            )
            stage_2_lead_80_pairs = _positive_policy_count(
                stage_2.get("lead_80_color_pairs"),
                "Stage-2 lead_80_color_pairs",
            )
            for cell_name, comparison in (
                (
                    "powered_candidate_vs_champion",
                    "candidate-vs-champion-powered",
                ),
                (
                    "powered_candidate_vs_original",
                    "candidate-vs-original-powered",
                ),
            ):
                append_cell(
                    cell_name=cell_name,
                    stage="stage-2",
                    look="automatic",
                    comparison=comparison,
                    suite="discovery",
                    search_mode="powered",
                    qualified_bank_name="discovery",
                    pair_count=stage_2_ordinary_pairs,
                    minimum_clusters=stage_2_ordinary_pairs,
                    visits=stage_2.get("ordinary_visits"),
                    maximal_look_schedule=True,
                )
            for cell_name, suite, pair_count in (
                ("lead_40", "lead-40", stage_2_lead_40_pairs),
                ("lead_80", "lead-80", stage_2_lead_80_pairs),
            ):
                append_cell(
                    cell_name=cell_name,
                    stage="stage-2",
                    look="automatic",
                    comparison=f"candidate-vs-champion-powered-{suite}",
                    suite=suite,
                    search_mode="powered",
                    qualified_bank_name=f"{suite}-discovery",
                    pair_count=pair_count,
                    minimum_clusters=pair_count,
                    visits=stage_2.get("lead_visits"),
                    maximal_look_schedule=True,
                )

            for look in policy_plan.stage_3_looks:
                look_number = look["look_number"]
                for cell_name in STAGE_3_CELL_ORDER:
                    policy_cell = matrix[cell_name]
                    if not isinstance(policy_cell, dict):
                        raise ValueError(
                            f"required_confirmation_matrix {cell_name} must be an object"
                        )
                    defaults = STAGE_3_CELL_DEFAULTS[cell_name]
                    comparison = policy_cell.get("comparison", defaults["comparison"])
                    suite = policy_cell.get("suite", defaults["suite"])
                    search_mode = policy_cell.get(
                        "search_mode", defaults["search_mode"]
                    )
                    qualified_bank_name = (
                        "confirmation"
                        if cell_name
                        in {
                            "powered_candidate_vs_champion",
                            "powered_candidate_vs_original",
                            "standard_candidate_vs_original",
                        }
                        else f"{suite}-confirmation"
                    )
                    pair_count = look["color_pairs"][cell_name]
                    minimum_clusters = look["minimum_independent_position_clusters"][
                        cell_name
                    ]
                    append_cell(
                        cell_name=cell_name,
                        stage="stage-3",
                        look=f"look-{look_number}",
                        comparison=comparison,
                        suite=suite,
                        search_mode=search_mode,
                        qualified_bank_name=qualified_bank_name,
                        pair_count=pair_count,
                        minimum_clusters=minimum_clusters,
                        visits=(
                            stage_3["powered_visits"]
                            if search_mode == "powered"
                            else stage_3["standard_visits"]
                        ),
                        maximal_look_schedule=(
                            look_number == latest_look_number
                        ),
                    )

        unassigned_rows = [
            item
            for item in included_rows
            if item.content_sha256 not in assigned_content_hashes
        ]

        manifest_payload: Dict[str, Any] = {
            "schemaVersion": (
                SCHEMA_VERSION if policy_plan.exact_quota_contract else 1
            ),
            "manifestContract": MANIFEST_CONTRACT,
            "generatorContract": (
                GENERATOR_CONTRACT
                if policy_plan.exact_quota_contract
                else LEGACY_GENERATOR_CONTRACT
            ),
            "scheduleGeneratorContract": SCHEDULE_GENERATOR_CONTRACT,
            "canonicalJsonContract": "utf-8-sort-keys-compact-json-lines-v1",
            "ordinaryAssignmentContract": (
                "policy-exact-seeded-semantic-hash-disjoint-holdouts-v2"
                if policy_plan.exact_quota_contract
                else "seeded-semantic-hash-balanced-disjoint-v1"
            ),
            "semanticPositionContract": (
                "canonical-xSize-ySize-board-nextPla-moveLocs-movePlas-"
                "initialTurnNumber-v1"
            ),
            "seed": seed,
            "policy_hash": policy_plan.policy_hash,
            "policy_version": policy_plan.policy_version,
            "source_revision": policy_plan.source_revision,
            "exactPolicyQuotas": policy_plan.exact_quota_contract,
            "policyHoldoutQuotas": (
                {
                    label: dict(quotas)
                    for label, quotas in sorted(policy_plan.holdout_quotas.items())
                }
                if policy_plan.exact_quota_contract
                else None
            ),
            "ordinaryWeights": {bank: float(weights[bank]) for bank in ORDINARY_BANKS},
            "pairsPerPosition": pairs_per_position,
            "botAIndex": bot_a_index,
            "botBIndex": bot_b_index,
            "acceptedLabels": sorted(ALL_LABELS),
            "sources": source_manifest,
            "inputRowCount": len(positions),
            "includedRowCount": len(included_rows),
            "assignedRowCount": len(assigned_content_hashes),
            "unassigned": [
                {
                    "contentSha256": item.content_sha256,
                    "semanticSha256": item.semantic_sha256,
                    "labels": list(item.labels),
                    "reason": "outside-exact-policy-quota",
                    "source": {
                        "name": item.source_name,
                        "line": item.source_line,
                    },
                }
                for item in sorted(
                    unassigned_rows, key=lambda item: item.content_sha256
                )
            ],
            "exclusions": [
                {
                    "contentSha256": item.content_sha256,
                    "semanticSha256": item.semantic_sha256,
                    "labels": list(item.labels),
                    "reason": "explicit-content-hash-exclusion",
                    "source": {"name": item.source_name, "line": item.source_line},
                }
                for item in sorted(excluded_rows, key=lambda item: item.content_sha256)
            ],
            "banks": bank_manifest,
            "cells": cell_manifest,
            "discovery_schedule_hash": (
                bank_artifacts["discovery"]["schedule"]["sha256"]
                if "discovery" in bank_artifacts
                else None
            ),
        }
        manifest_payload_sha = canonical_sha256(manifest_payload)
        manifest = dict(manifest_payload)
        manifest["manifestPayloadSha256"] = manifest_payload_sha
        manifest_data = (canonical_json(manifest) + "\n").encode("utf-8")
        _write_fsynced(temporary / "manifest.json", manifest_data)

        _fsync_directory(positions_dir)
        if prefix_dir.exists():
            _fsync_directory(prefix_dir)
        _fsync_directory(schedules_dir)
        _fsync_directory(temporary)

        if output_dir.exists():
            _verify_existing_bundle(output_dir, manifest)
            reused = True
        else:
            try:
                os.rename(str(temporary), str(output_dir))
                _fsync_directory(output_dir.parent)
                reused = False
                temporary = None
            except OSError:
                if not output_dir.exists():
                    raise
                _verify_existing_bundle(output_dir, manifest)
                reused = True

        return SuiteBuildResult(
            output_dir=output_dir,
            manifest_path=output_dir / "manifest.json",
            manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
            manifest=manifest,
            reused=reused,
        )
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable labeled risk-score evaluation suites."
    )
    parser.add_argument(
        "sources", nargs="+", type=Path, help="Labeled PositionSample JSONL"
    )
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument(
        "--seed", required=True, help="Explicit split and schedule seed"
    )
    parser.add_argument("--pairs-per-position", type=int, default=1)
    parser.add_argument("--bot-a-index", type=int, default=0)
    parser.add_argument("--bot-b-index", type=int, default=1)
    parser.add_argument("--discovery-weight", type=float, default=1.0)
    parser.add_argument("--confirmation-weight", type=float, default=1.0)
    parser.add_argument("--audit-weight", type=float, default=1.0)
    parser.add_argument(
        "--exclude-content-hash",
        action="append",
        default=[],
        help="Explicit PositionSample content SHA-256 to exclude",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Frozen promotion policy to bind into the suite manifest",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = build_evaluation_suites(
            args.sources,
            args.output_dir,
            seed=args.seed,
            ordinary_weights={
                "discovery": args.discovery_weight,
                "confirmation": args.confirmation_weight,
                "audit": args.audit_weight,
            },
            pairs_per_position=args.pairs_per_position,
            bot_a_index=args.bot_a_index,
            bot_b_index=args.bot_b_index,
            exclude_content_hashes=args.exclude_content_hash,
            policy_path=args.policy,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    action = "Reused" if result.reused else "Published"
    print(
        f"{action} evaluation suites at {result.output_dir} "
        f"(manifest SHA-256 {result.manifest_sha256})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
