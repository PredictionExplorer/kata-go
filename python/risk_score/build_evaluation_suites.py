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

SCHEMA_VERSION = 1
GENERATOR_CONTRACT = "risk-score-content-addressed-evaluation-suites-v1"
DEFAULT_POLICY_PATH = Path(__file__).with_name("promotion_policy_v1.json")
ORDINARY_LABEL = "ordinary"
ORDINARY_BANKS = ("discovery", "confirmation", "audit")
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

    validate_position(dict(position), "semantic position")
    return {
        key: position[key]
        for key in (
            "xSize",
            "ySize",
            "board",
            "nextPla",
            "moveLocs",
            "movePlas",
            "initialTurnNumber",
        )
    }


def semantic_position_sha256(position: Mapping[str, Any]) -> str:
    return canonical_sha256(semantic_position(position))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_policy_binding(path: Path) -> Tuple[str, str]:
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load promotion policy {path}: {exc}") from exc
    if not isinstance(policy, dict):
        raise ValueError("promotion policy root must be an object")
    frozen_plan = policy.get("frozen_plan")
    source_revision = (
        frozen_plan.get("source_revision")
        if isinstance(frozen_plan, dict)
        else None
    )
    if (
        not isinstance(source_revision, str)
        or len(source_revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("promotion policy has no valid frozen source revision")
    return canonical_sha256(policy), source_revision


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


def split_labeled_positions(
    positions: Sequence[_LabeledPosition],
    *,
    seed: str,
    ordinary_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, List[_LabeledPosition]]:
    """Deterministically form disjoint ordinary banks and specialized suites."""

    if not isinstance(seed, str) or not seed or "\n" in seed or "\r" in seed:
        raise ValueError("seed must be a nonempty single-line string")
    weights = (
        dict(ordinary_weights)
        if ordinary_weights is not None
        else {bank: 1.0 for bank in ORDINARY_BANKS}
    )

    ordinary = [item for item in positions if item.labels == (ORDINARY_LABEL,)]
    counts = _ordinary_counts(len(ordinary), weights)
    ordinary_order = sorted(
        ordinary,
        key=lambda item: (
            canonical_sha256(
                {
                    "generatorContract": GENERATOR_CONTRACT,
                    "purpose": "ordinary-bank-assignment",
                    "seed": seed,
                    "semanticSha256": item.semantic_sha256,
                }
            ),
            item.semantic_sha256,
            item.content_sha256,
        ),
    )

    banks: Dict[str, List[_LabeledPosition]] = {}
    offset = 0
    for bank in ORDINARY_BANKS:
        selected = ordinary_order[offset : offset + counts[bank]]
        offset += counts[bank]
        banks[bank] = sorted(
            selected,
            key=lambda item: (
                canonical_sha256(
                    {
                        "generatorContract": GENERATOR_CONTRACT,
                        "purpose": "bank-order",
                        "seed": seed,
                        "bank": bank,
                        "semanticSha256": item.semantic_sha256,
                    }
                ),
                item.semantic_sha256,
                item.content_sha256,
            ),
        )

    for label in SPECIALIZED_LABELS:
        selected = [item for item in positions if label in item.labels]
        if selected:
            banks[label] = sorted(
                selected,
                key=lambda item: (
                    canonical_sha256(
                        {
                            "generatorContract": GENERATOR_CONTRACT,
                            "purpose": "specialized-bank-order",
                            "seed": seed,
                            "bank": label,
                            "semanticSha256": item.semantic_sha256,
                        }
                    ),
                    item.semantic_sha256,
                    item.content_sha256,
                ),
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

    policy_hash, source_revision = _load_policy_binding(Path(policy_path))
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
    banks = split_labeled_positions(
        included_rows,
        seed=seed,
        ordinary_weights=weights,
    )

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

        for bank_name in ORDINARY_BANKS + SPECIALIZED_LABELS:
            selected = banks.get(bank_name)
            if not selected:
                continue
            position_rows = [item.position for item in selected]
            position_relative = Path("position-banks") / f"{bank_name}.jsonl"
            position_data = _canonical_jsonl(position_rows)
            position_bank_sha = hashlib.sha256(position_data).hexdigest()
            _write_fsynced(temporary / position_relative, position_data)

            schedule_seed = "risk-score-suite-v1-" + canonical_sha256(
                {
                    "generatorContract": GENERATOR_CONTRACT,
                    "purpose": "schedule-seed",
                    "masterSeed": seed,
                    "bank": bank_name,
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
                        "suite": bank_name,
                        "suiteBank": bank_name,
                        "suiteBankSha256": position_bank_sha,
                        "positionContentSha256": selected_position.content_sha256,
                        "positionSemanticSha256": selected_position.semantic_sha256,
                    }
                )
            schedule_relative = Path("schedules") / f"{bank_name}.jsonl"
            schedule_data = _canonical_jsonl(schedule_rows)
            _write_fsynced(temporary / schedule_relative, schedule_data)

            bank_manifest.append(
                {
                    "name": bank_name,
                    "kind": "ordinary"
                    if bank_name in ORDINARY_BANKS
                    else "specialized",
                    "contentSha256s": [item.content_sha256 for item in selected],
                    "semanticSha256s": [item.semantic_sha256 for item in selected],
                    "positions": {
                        "path": position_relative.as_posix(),
                        "sha256": position_bank_sha,
                        "rowCount": len(position_rows),
                    },
                    "schedule": {
                        "path": schedule_relative.as_posix(),
                        "sha256": hashlib.sha256(schedule_data).hexdigest(),
                        "rowCount": len(schedule_rows),
                        "pairCount": len(schedule_rows) // 2,
                        "scheduleId": schedule_rows[0]["scheduleId"],
                        "baseSeed": schedule_seed,
                    },
                }
            )

        manifest_payload: Dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "generatorContract": GENERATOR_CONTRACT,
            "scheduleGeneratorContract": SCHEDULE_GENERATOR_CONTRACT,
            "canonicalJsonContract": "utf-8-sort-keys-compact-json-lines-v1",
            "ordinaryAssignmentContract": "seeded-semantic-hash-balanced-disjoint-v1",
            "semanticPositionContract": (
                "canonical-xSize-ySize-board-nextPla-moveLocs-movePlas-"
                "initialTurnNumber-v1"
            ),
            "seed": seed,
            "policy_hash": policy_hash,
            "source_revision": source_revision,
            "ordinaryWeights": {bank: float(weights[bank]) for bank in ORDINARY_BANKS},
            "pairsPerPosition": pairs_per_position,
            "botAIndex": bot_a_index,
            "botBIndex": bot_b_index,
            "acceptedLabels": sorted(ALL_LABELS),
            "sources": source_manifest,
            "inputRowCount": len(positions),
            "includedRowCount": len(included_rows),
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
        }
        manifest_payload_sha = canonical_sha256(manifest_payload)
        manifest = dict(manifest_payload)
        manifest["manifestPayloadSha256"] = manifest_payload_sha
        manifest_data = (canonical_json(manifest) + "\n").encode("utf-8")
        _write_fsynced(temporary / "manifest.json", manifest_data)

        _fsync_directory(positions_dir)
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
