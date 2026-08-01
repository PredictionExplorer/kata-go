#!/usr/bin/env python3
"""Deterministically curate reviewed PositionSample pools for promotion suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from risk_score.build_evaluation_suites import ALL_LABELS
from risk_score.paired_stats import load_policy
from risk_score.position_samples import (
    build_analysis_query,
    canonical_json,
    canonical_sha256,
    file_sha256,
    normalize_position_sample,
    semantic_position_sha256,
)


CURATION_CONTRACT = "risk-score-position-bank-curation-v1"
HARVEST_PLAN_CONTRACT = "risk-score-position-harvest-plan-v1"
HARVEST_RECEIPT_CONTRACT = "risk-score-position-harvest-receipt-v1"
QUERY_BUNDLE_CONTRACT = "risk-score-position-analysis-query-bundle-v1"
ANALYSIS_RUN_CONTRACT = "risk-score-position-analysis-run-v1"
LABELING_CONTRACT = "risk-score-position-bank-labeling-v1"
FINAL_MANIFEST_CONTRACT = "risk-score-reviewed-position-bank-v1"
REVIEW_ONLY_LABELS = frozenset(
    {
        "tactical",
        "exploitability",
        "baits",
        "tails",
        "sacrifice",
        "small-gain",
        "adversarial",
    }
)
AUTO_LABELS = frozenset({"ordinary", "lead-40", "lead-80"})


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_json(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {role} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must have an object root")
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_jsonl(
    path: Path, role: str, *, allow_empty: bool = False
) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: row must be an object")
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"{role} contains no rows")
    return rows


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file(path: Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise ValueError(f"existing immutable output conflicts: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_bundle(output_dir: Path, files: Mapping[str, bytes]) -> Path:
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        for relative, data in sorted(files.items()):
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o444)
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary)
        if output.exists():
            existing_files = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            }
            if existing_files != set(files):
                raise ValueError(f"existing immutable bundle has unexpected files: {output}")
            for relative, data in files.items():
                existing = output / relative
                if (
                    existing.is_symlink()
                    or not existing.is_file()
                    or existing.read_bytes() != data
                ):
                    raise ValueError(f"existing immutable bundle conflicts: {output}")
            return output
        os.rename(temporary, output)
        _fsync_directory(output.parent)
        temporary = None
        return output
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _source_inventory(
    *,
    sgfs_dirs: Sequence[Path],
    sgf_dirs: Sequence[Path],
) -> List[Dict[str, Any]]:
    inventory = []
    for kind, directories, suffixes in (
        ("sgfs", sgfs_dirs, {".sgfs"}),
        ("sgf", sgf_dirs, {".sgf"}),
    ):
        for directory in map(Path, directories):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(
                    f"{kind.upper()} source is not a regular directory: {directory}"
                )
            files = sorted(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in suffixes
            )
            if not files:
                raise ValueError(f"{kind.upper()} source has no input files: {directory}")
            inventory.append(
                {
                    "kind": kind,
                    "directory": str(directory.resolve()),
                    "files": [
                        {
                            "path": str(path.relative_to(directory)),
                            "size": path.stat().st_size,
                            "sha256": file_sha256(path),
                        }
                        for path in files
                    ],
                }
            )
    return inventory


def validate_deterministic_analysis_config(path: Path) -> None:
    values = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        values[key] = value.lower()
    required = {
        "forDeterministicTesting": "true",
        "numAnalysisThreads": "1",
        "nnRandomize": "false",
        "rootNoiseEnabled": "false",
        "rootNumSymmetriesToSample": "1",
        "useUncertainty": "false",
        "cpuctUtilityStdevScale": "0",
        "reportAnalysisWinratesAs": "sidetomove",
    }
    conflicts = {
        key: {"actual": values.get(key), "expected": expected}
        for key, expected in required.items()
        if values.get(key) != expected
    }
    if conflicts:
        raise ValueError(
            f"analysis config is not deterministic and perspective-fixed: {conflicts}"
        )


def normalize_sources(
    sources: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not sources:
        raise ValueError("at least one PositionSample source is required")
    records = []
    seen: Dict[str, str] = {}
    source_manifest = []
    for source in map(Path, sources):
        rows = _load_jsonl(source, "PositionSample source")
        source_manifest.append(
            {
                "path": str(source.resolve()),
                "sha256": file_sha256(source),
                "rowCount": len(rows),
            }
        )
        for line_number, row in enumerate(rows, start=1):
            origin = f"{source}:{line_number}"
            position = normalize_position_sample(row, origin)
            semantic_hash = semantic_position_sha256(position)
            previous = seen.get(semantic_hash)
            if previous is not None:
                raise ValueError(
                    f"{origin}: duplicate semantic position; first seen at {previous}"
                )
            seen[semantic_hash] = origin
            records.append(
                {
                    **position,
                    "semanticSha256": semantic_hash,
                    "curationSource": {
                        "path": str(source.resolve()),
                        "line": line_number,
                    },
                }
            )
    records.sort(key=lambda row: row["semanticSha256"])
    manifest = {
        "schema_version": 1,
        "contract": CURATION_CONTRACT,
        "stage": "normalized",
        "sources": sorted(source_manifest, key=lambda item: item["path"]),
        "row_count": len(records),
        "semantic_hashes_sha256": canonical_sha256(
            [row["semanticSha256"] for row in records]
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return records, manifest


def publish_normalized(
    sources: Sequence[Path], output: Path, manifest_path: Path
) -> Mapping[str, Any]:
    records, manifest = normalize_sources(sources)
    data = _canonical_jsonl(records)
    published_manifest = {
        **manifest,
        "output_path": str(Path(output).resolve()),
        "output_sha256": hashlib.sha256(data).hexdigest(),
    }
    payload = dict(published_manifest)
    payload.pop("manifest_sha256", None)
    published_manifest["manifest_sha256"] = canonical_sha256(payload)
    _publish_file(Path(output), data)
    _publish_file(
        Path(manifest_path),
        (canonical_json(published_manifest) + "\n").encode("utf-8"),
    )
    return published_manifest


def build_harvest_argv(
    *,
    katago: Path,
    sgfs_dirs: Sequence[Path],
    sgf_dirs: Sequence[Path],
    training_input_roots: Sequence[Path],
    output_dir: Path,
    threads: int,
) -> Tuple[str, ...]:
    binary = Path(katago)
    if binary.is_symlink() or not binary.is_file():
        raise ValueError("KataGo harvester must be a regular non-symlink file")
    if not sgfs_dirs and not sgf_dirs:
        raise ValueError("at least one SGF or SGFS directory is required")
    if not training_input_roots:
        raise ValueError("at least one training/shuffler input root is required")
    roots = []
    for root in map(Path, training_input_roots):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"training input root is not a regular directory: {root}")
        roots.append(root.resolve())
    if type(threads) is not int or threads != 1:
        raise ValueError(
            "deterministic samplesgfs harvesting currently requires exactly one thread"
        )
    argv = [str(binary.resolve()), "samplesgfs"]
    for directory in map(Path, sgfs_dirs):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"SGFS source is not a regular directory: {directory}")
        if any(_paths_overlap(directory, root) for root in roots):
            raise ValueError(
                f"SGFS source is inside a training/shuffler input root: {directory}"
            )
        argv.extend(["--sgfsdir", str(directory.resolve())])
    for directory in map(Path, sgf_dirs):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"SGF source is not a regular directory: {directory}")
        if any(_paths_overlap(directory, root) for root in roots):
            raise ValueError(
                f"SGF source is inside a training/shuffler input root: {directory}"
            )
        argv.extend(["--sgfdir", str(directory.resolve())])
    argv.extend(
        [
            "--outdir",
            str(Path(output_dir).resolve()),
            "--sample-prob",
            "1",
            "--turn-weight-lambda",
            "0",
            "--min-turn-number-board-area-prop",
            "0.05",
            "--max-turn-number-board-area-prop",
            "0.95",
            "--max-handicap",
            "0",
            "--max-komi",
            "7.5",
            "--num-threads",
            str(threads),
            "--for-testing",
        ]
    )
    return tuple(argv)


def publish_harvest_plan(
    *,
    katago: Path,
    sgfs_dirs: Sequence[Path],
    sgf_dirs: Sequence[Path],
    training_input_roots: Sequence[Path],
    output_dir: Path,
    manifest_path: Path,
    threads: int,
) -> Mapping[str, Any]:
    argv = build_harvest_argv(
        katago=katago,
        sgfs_dirs=sgfs_dirs,
        sgf_dirs=sgf_dirs,
        training_input_roots=training_input_roots,
        output_dir=output_dir,
        threads=threads,
    )
    if Path(output_dir).exists():
        raise ValueError("planned harvest output directory must not already exist")
    inputs = _source_inventory(sgfs_dirs=sgfs_dirs, sgf_dirs=sgf_dirs)
    manifest = {
        "schema_version": 1,
        "contract": HARVEST_PLAN_CONTRACT,
        "katago_path": str(Path(katago).resolve()),
        "katago_sha256": file_sha256(Path(katago)),
        "argv": list(argv),
        "inputs": inputs,
        "training_input_roots": sorted(
            str(Path(root).resolve()) for root in training_input_roots
        ),
        "output_dir": str(Path(output_dir).resolve()),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _publish_file(
        Path(manifest_path), (canonical_json(manifest) + "\n").encode("utf-8")
    )
    return manifest


def _validate_harvest_plan(plan_path: Path) -> Dict[str, Any]:
    plan = _load_json(plan_path, "harvest plan")
    if plan.get("contract") != HARVEST_PLAN_CONTRACT:
        raise ValueError("harvest plan contract is unsupported")
    payload = dict(plan)
    expected_hash = payload.pop("manifest_sha256", None)
    if expected_hash != canonical_sha256(payload):
        raise ValueError("harvest plan self-hash is invalid")
    binary = Path(plan.get("katago_path", ""))
    if (
        binary.is_symlink()
        or not binary.is_file()
        or file_sha256(binary) != plan.get("katago_sha256")
    ):
        raise ValueError("harvest plan KataGo binary changed")
    inputs = plan.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("harvest plan has no frozen source inventory")
    sgfs_dirs = [
        Path(item["directory"])
        for item in inputs
        if isinstance(item, Mapping) and item.get("kind") == "sgfs"
    ]
    sgf_dirs = [
        Path(item["directory"])
        for item in inputs
        if isinstance(item, Mapping) and item.get("kind") == "sgf"
    ]
    if _source_inventory(sgfs_dirs=sgfs_dirs, sgf_dirs=sgf_dirs) != inputs:
        raise ValueError("harvest source inventory changed after planning")
    roots = plan.get("training_input_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("harvest plan has no training input exclusions")
    for source in (*sgfs_dirs, *sgf_dirs):
        if any(_paths_overlap(source, Path(root)) for root in roots):
            raise ValueError("harvest source now aliases a training input root")
    return plan


def _harvest_output_inventory(root: Path) -> List[Dict[str, Any]]:
    values = []
    for path in sorted(root.rglob("*")):
        if path.name == "receipt.json":
            continue
        if path.is_symlink():
            raise ValueError(f"harvest output contains a symlink: {path}")
        if path.is_file():
            values.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    if not any(item["path"].endswith(".startposes.txt") for item in values):
        raise ValueError("harvest produced no PositionSample output")
    return values


def execute_harvest_plan(
    plan_path: Path,
    *,
    subprocess_runner: Any = subprocess.run,
) -> Mapping[str, Any]:
    plan = _validate_harvest_plan(Path(plan_path))
    plan_file_hash = file_sha256(Path(plan_path))
    output = Path(plan["output_dir"])
    receipt_path = output / "receipt.json"
    if output.exists():
        receipt = _load_json(receipt_path, "harvest receipt")
        payload = dict(receipt)
        receipt_hash = payload.pop("manifest_sha256", None)
        if (
            receipt.get("contract") != HARVEST_RECEIPT_CONTRACT
            or receipt_hash != canonical_sha256(payload)
            or receipt.get("plan_sha256") != plan_file_hash
            or receipt.get("outputs") != _harvest_output_inventory(output)
        ):
            raise ValueError("existing harvest output contradicts reviewed plan")
        return {**receipt, "reused": True}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.harvest-", dir=str(output.parent))
    )
    try:
        argv = list(plan["argv"])
        out_index = argv.index("--outdir") + 1
        argv[out_index] = str(temporary)
        result = subprocess_runner(tuple(argv), shell=False, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"samplesgfs failed with status {result.returncode}")
        # Close the verify/use window over growing live SGF files.
        _validate_harvest_plan(Path(plan_path))
        outputs = _harvest_output_inventory(temporary)
        receipt = {
            "schema_version": 1,
            "contract": HARVEST_RECEIPT_CONTRACT,
            "plan_path": str(Path(plan_path).resolve()),
            "plan_sha256": plan_file_hash,
            "output_dir": str(output.resolve()),
            "outputs": outputs,
        }
        receipt["manifest_sha256"] = canonical_sha256(receipt)
        receipt_data = (canonical_json(receipt) + "\n").encode("utf-8")
        with (temporary / "receipt.json").open("xb") as handle:
            handle.write(receipt_data)
            handle.flush()
            os.fsync(handle.fileno())
        for path in temporary.rglob("*"):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary)
        os.rename(temporary, output)
        _fsync_directory(output.parent)
        temporary = None
        return {**receipt, "reused": False}
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _normalized_positions(path: Path) -> List[Dict[str, Any]]:
    rows = _load_jsonl(path, "normalized PositionSample file")
    seen = set()
    normalized = []
    for index, row in enumerate(rows, start=1):
        position = normalize_position_sample(row, f"{path}:{index}")
        digest = semantic_position_sha256(position)
        if row.get("semanticSha256") not in {None, digest}:
            raise ValueError(f"{path}:{index}: semantic hash is invalid")
        if digest in seen:
            raise ValueError(f"{path}:{index}: duplicate semantic position")
        seen.add(digest)
        normalized.append({**position, "semanticSha256": digest})
    normalized.sort(key=lambda row: row["semanticSha256"])
    return normalized


def generate_query_bundle(
    normalized_path: Path,
    output_dir: Path,
    *,
    katago_binary: Path,
    analysis_config: Path,
    reference_model: Path,
    policy_path: Path,
) -> Mapping[str, Any]:
    for path, role in (
        (katago_binary, "KataGo binary"),
        (analysis_config, "analysis config"),
        (reference_model, "reference model"),
    ):
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"{role} must be a regular non-symlink file")
    validate_deterministic_analysis_config(Path(analysis_config))
    model = Path(reference_model)
    positions = _normalized_positions(Path(normalized_path))
    policy = load_policy(Path(policy_path))
    stages = policy["evaluation_stages"]
    visits = sorted(
        {
            stages["stage_0_integrity_and_fixed_probes"]["fixed_analysis_visits"],
            stages["stage_2_finalist_selection"]["ordinary_visits"],
            stages["stage_0_integrity_and_fixed_probes"][
                "exploitability_sentinel_visits"
            ],
        }
    )
    roles = [
        (f"{mode}-{visit_count}", mode == "powered", visit_count)
        for visit_count in visits
        for mode in ("standard", "powered")
        if not (mode == "powered" and visit_count == min(visits))
    ]
    files: Dict[str, bytes] = {}
    query_manifest = {}
    for role, powered, visit_count in roles:
        rows = [
            build_analysis_query(
                position,
                query_id=position["semanticSha256"],
                max_visits=visit_count,
                powered=powered,
            )
            for position in positions
        ]
        relative = f"queries/{role}.jsonl"
        data = _canonical_jsonl(rows)
        files[relative] = data
        query_manifest[role] = {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "row_count": len(rows),
            "powered": powered,
            "visits": visit_count,
        }
    manifest = {
        "schema_version": 1,
        "contract": QUERY_BUNDLE_CONTRACT,
        "normalized_path": str(Path(normalized_path).resolve()),
        "normalized_sha256": file_sha256(Path(normalized_path)),
        "katago_path": str(Path(katago_binary).resolve()),
        "katago_sha256": file_sha256(Path(katago_binary)),
        "analysis_config_path": str(Path(analysis_config).resolve()),
        "analysis_config_sha256": file_sha256(Path(analysis_config)),
        "reference_model_path": str(model.resolve()),
        "reference_model_sha256": file_sha256(model),
        "policy_path": str(Path(policy_path).resolve()),
        "policy_hash": canonical_sha256(policy),
        "position_count": len(positions),
        "semantic_hashes_sha256": canonical_sha256(
            [position["semanticSha256"] for position in positions]
        ),
        "queries": query_manifest,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    files["manifest.json"] = (canonical_json(manifest) + "\n").encode("utf-8")
    _publish_bundle(Path(output_dir), files)
    return manifest


def run_analysis(
    *,
    katago: Path,
    config: Path,
    model: Path,
    queries: Path,
    output: Path,
    env: Optional[Mapping[str, str]] = None,
    subprocess_runner: Any = subprocess.run,
) -> Mapping[str, Any]:
    for path, role in (
        (katago, "KataGo binary"),
        (config, "analysis config"),
        (model, "analysis model"),
        (queries, "analysis queries"),
    ):
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"{role} must be a regular non-symlink file")
    query_rows = _load_jsonl(Path(queries), "analysis queries")
    expected_ids = [row.get("id") for row in query_rows]
    if (
        any(not isinstance(value, str) or not value for value in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise ValueError("analysis query IDs must be unique nonempty strings")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.analysis-", dir=str(output.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    argv = (
        str(Path(katago).resolve()),
        "analysis",
        "-config",
        str(Path(config).resolve()),
        "-model",
        str(Path(model).resolve()),
    )
    with Path(queries).open("rb") as source, temporary.open("wb") as destination:
        result = subprocess_runner(
            argv,
            stdin=source,
            stdout=destination,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            env=(None if env is None else dict(env)),
        )
        destination.flush()
        os.fsync(destination.fileno())
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"KataGo analysis failed with status {result.returncode}")
    responses = _load_jsonl(temporary, "analysis responses")
    by_id = {}
    for row in responses:
        record_id = row.get("id")
        if not isinstance(record_id, str) or record_id in by_id:
            temporary.unlink(missing_ok=True)
            raise ValueError("analysis responses have missing or duplicate IDs")
        if "error" in row:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"analysis response {record_id} contains an error")
        by_id[record_id] = row
    if set(by_id) != set(expected_ids):
        temporary.unlink(missing_ok=True)
        raise ValueError("analysis response IDs do not match query IDs")
    data = _canonical_jsonl(by_id[record_id] for record_id in sorted(by_id))
    temporary.unlink(missing_ok=True)
    _publish_file(output, data)
    manifest = {
        "schema_version": 1,
        "contract": ANALYSIS_RUN_CONTRACT,
        "argv": list(argv),
        "katago_sha256": file_sha256(Path(katago)),
        "config_sha256": file_sha256(Path(config)),
        "model_sha256": file_sha256(Path(model)),
        "query_path": str(Path(queries).resolve()),
        "query_sha256": file_sha256(Path(queries)),
        "output_path": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "row_count": len(by_id),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = Path(str(output) + ".manifest.json")
    _publish_file(
        manifest_path, (canonical_json(manifest) + "\n").encode("utf-8")
    )
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def _finite(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{role} must be finite")
    return number


def analysis_features(
    record: Mapping[str, Any], role: str, *, expected_visits: int
) -> Dict[str, Any]:
    root = record.get("rootInfo")
    moves = record.get("moveInfos")
    if not isinstance(root, Mapping) or not isinstance(moves, list) or not moves:
        raise ValueError(f"{role} lacks rootInfo or moveInfos")
    ordered = sorted(
        (move for move in moves if isinstance(move, Mapping)),
        key=lambda move: (move.get("order", 1 << 30), -move.get("visits", 0)),
    )
    if not ordered:
        raise ValueError(f"{role} has no valid moveInfos")
    actual_visits = root.get("visits")
    if (
        type(expected_visits) is not int
        or expected_visits <= 0
        or type(actual_visits) is not int
        or actual_visits < expected_visits
    ):
        raise ValueError(
            f"{role} effective visits {actual_visits!r} are below "
            f"the bound budget {expected_visits}"
        )
    top = ordered[0]
    return {
        "visits": actual_visits,
        "score_lead": _finite(root.get("scoreLead"), f"{role} root scoreLead"),
        "winrate": _finite(root.get("winrate"), f"{role} root winrate"),
        "top_move": str(top.get("move")),
        "top_prior": _finite(top.get("prior", 0.0), f"{role} top prior"),
        "top_score_selfplay": _finite(
            top.get("scoreSelfplay", top.get("scoreLead")),
            f"{role} top scoreSelfplay",
        ),
        "top_score_stdev": _finite(
            top.get("scoreStdev", 0.0), f"{role} top scoreStdev"
        ),
    }


def _analysis_map(path: Path, expected_ids: Iterable[str], role: str) -> Dict[str, Any]:
    rows = _load_jsonl(path, f"{role} analysis")
    values = {}
    for row in rows:
        record_id = row.get("id")
        if not isinstance(record_id, str) or record_id in values:
            raise ValueError(f"{role} analysis IDs are missing or duplicated")
        values[record_id] = row
    if set(values) != set(expected_ids):
        raise ValueError(f"{role} analysis IDs do not match normalized positions")
    return values


def _suggest_specialized(
    standard: Mapping[str, Any], powered: Mapping[str, Any]
) -> List[str]:
    suggestions = []
    if standard["top_move"] != powered["top_move"]:
        suggestions.append("adversarial")
    if (
        powered["top_prior"] < 0.05
        and powered["top_score_selfplay"] - standard["top_score_selfplay"] >= 10.0
    ):
        suggestions.append("baits")
    if max(standard["top_score_stdev"], powered["top_score_stdev"]) >= 30.0:
        suggestions.append("tails")
    return sorted(set(suggestions))


def label_positions(
    *,
    normalized_path: Path,
    query_manifest_path: Path,
    analysis_paths: Mapping[str, Path],
    output_dir: Path,
    stability_margin: float = 5.0,
) -> Mapping[str, Any]:
    if not math.isfinite(stability_margin) or stability_margin < 0:
        raise ValueError("stability margin must be finite and nonnegative")
    positions = _normalized_positions(Path(normalized_path))
    position_by_id = {row["semanticSha256"]: row for row in positions}
    query_manifest = _load_json(query_manifest_path, "query manifest")
    if query_manifest.get("contract") != QUERY_BUNDLE_CONTRACT:
        raise ValueError("query manifest contract is unsupported")
    query_payload = dict(query_manifest)
    query_manifest_hash = query_payload.pop("manifest_sha256", None)
    if query_manifest_hash != canonical_sha256(query_payload):
        raise ValueError("query manifest self-hash is invalid")
    if query_manifest.get("normalized_sha256") != file_sha256(Path(normalized_path)):
        raise ValueError("query manifest names another normalized position file")
    queries = query_manifest.get("queries")
    if not isinstance(queries, dict) or set(analysis_paths) != set(queries):
        raise ValueError("analysis result roles must exactly match query manifest")
    expected_ids = set(position_by_id)
    for role, artifact in queries.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"query manifest role {role} is malformed")
        relative = artifact.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"query manifest role {role} has no path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"query manifest role {role} path is unsafe")
        query_path = Path(query_manifest_path).parent / relative_path
        if (
            query_path.is_symlink()
            or not query_path.is_file()
            or file_sha256(query_path) != artifact.get("sha256")
        ):
            raise ValueError(f"query manifest role {role} artifact changed")
        query_ids = {row.get("id") for row in _load_jsonl(query_path, role)}
        if query_ids != expected_ids or len(query_ids) != artifact.get("row_count"):
            raise ValueError(f"query manifest role {role} IDs changed")
    analyses = {
        role: _analysis_map(Path(path), position_by_id, role)
        for role, path in analysis_paths.items()
    }
    execution_manifests = {}
    for role, path in analysis_paths.items():
        execution_path = Path(str(path) + ".manifest.json")
        execution = _load_json(execution_path, f"{role} analysis run manifest")
        if execution.get("contract") != ANALYSIS_RUN_CONTRACT:
            raise ValueError(f"{role} analysis run contract is unsupported")
        execution_payload = dict(execution)
        execution_hash = execution_payload.pop("manifest_sha256", None)
        if execution_hash != canonical_sha256(execution_payload):
            raise ValueError(f"{role} analysis run manifest self-hash is invalid")
        if (
            execution.get("model_sha256")
            != query_manifest.get("reference_model_sha256")
            or execution.get("katago_sha256")
            != query_manifest.get("katago_sha256")
            or execution.get("config_sha256")
            != query_manifest.get("analysis_config_sha256")
            or execution.get("query_sha256") != queries[role].get("sha256")
            or execution.get("output_path") != str(Path(path).resolve())
            or execution.get("output_sha256") != file_sha256(Path(path))
            or execution.get("row_count") != len(analyses[role])
        ):
            raise ValueError(f"{role} analysis run provenance changed")
        execution_manifests[role] = {
            "path": str(execution_path.resolve()),
            "sha256": file_sha256(execution_path),
            "katago_sha256": execution.get("katago_sha256"),
            "config_sha256": execution.get("config_sha256"),
        }
    required = {
        "standard-200",
        "standard-800",
        "standard-2000",
        "powered-800",
        "powered-2000",
    }
    if not required.issubset(analyses):
        raise ValueError(f"analysis bundle is missing roles: {sorted(required-analyses.keys())}")

    auto_rows = []
    review_rows = []
    for semantic_hash in sorted(position_by_id):
        position = normalize_position_sample(
            position_by_id[semantic_hash], f"normalized {semantic_hash}"
        )
        features = {
            role: analysis_features(
                analyses[role][semantic_hash],
                role,
                expected_visits=queries[role]["visits"],
            )
            for role in sorted(required)
        }
        standard_scores = [
            features[role]["score_lead"]
            for role in ("standard-200", "standard-800", "standard-2000")
        ]
        powered_scores = [
            features[role]["score_lead"]
            for role in ("powered-800", "powered-2000")
        ]
        stable = (
            max(standard_scores) - min(standard_scores) <= stability_margin
            and max(powered_scores) - min(powered_scores) <= stability_margin
        )
        scores = tuple(standard_scores)
        auto_label = None
        if stable and min(scores) >= 80.0:
            auto_label = "lead-80"
        elif stable and min(scores) >= 40.0 and max(scores) < 80.0:
            auto_label = "lead-40"
        elif stable and max(abs(value) for value in scores) < 30.0:
            auto_label = "ordinary"
        suggestions = sorted(
            set(
                _suggest_specialized(
                    features["standard-800"], features["powered-800"]
                )
            ).union(
                _suggest_specialized(
                    features["standard-2000"], features["powered-2000"]
                )
            )
        )
        summary = {
            **{role.replace("-", "_"): value for role, value in features.items()},
            "visit_stable": stable,
            "stability_margin": stability_margin,
        }
        if auto_label is not None and not suggestions:
            auto_rows.append(
                {
                    **position,
                    "labels": [auto_label],
                    "curation": {
                        "classification": "automatic",
                        "semanticSha256": semantic_hash,
                        "analysis": summary,
                    },
                }
            )
        else:
            review_rows.append(
                {
                    "semantic_sha256": semantic_hash,
                    "position": position,
                    "recommended_auto_label": auto_label,
                    "suggested_specialized_labels": suggestions,
                    "allowed_labels": sorted(ALL_LABELS),
                    "analysis": summary,
                }
            )
    analysis_manifest = {
        role: {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(Path(path)),
            "row_count": len(analyses[role]),
            "execution_manifest": execution_manifests[role],
        }
        for role, path in sorted(analysis_paths.items())
    }
    auto_data = _canonical_jsonl(auto_rows)
    review_data = _canonical_jsonl(review_rows)
    manifest = {
        "schema_version": 1,
        "contract": LABELING_CONTRACT,
        "normalized_path": str(Path(normalized_path).resolve()),
        "normalized_sha256": file_sha256(Path(normalized_path)),
        "query_manifest_path": str(Path(query_manifest_path).resolve()),
        "query_manifest_sha256": file_sha256(Path(query_manifest_path)),
        "reference_model_sha256": query_manifest.get("reference_model_sha256"),
        "policy_hash": query_manifest.get("policy_hash"),
        "stability_margin": stability_margin,
        "automatic_count": len(auto_rows),
        "review_count": len(review_rows),
        "automatic_sha256": hashlib.sha256(auto_data).hexdigest(),
        "review_queue_sha256": hashlib.sha256(review_data).hexdigest(),
        "analysis": analysis_manifest,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    files = {
        "auto-labeled.jsonl": auto_data,
        "review-queue.jsonl": review_data,
        "manifest.json": (canonical_json(manifest) + "\n").encode("utf-8"),
    }
    _publish_bundle(Path(output_dir), files)
    return manifest


def policy_pool_minima(policy: Mapping[str, Any]) -> Dict[str, int]:
    stages = policy["evaluation_stages"]
    stage_1 = stages["stage_1_cheap_paired_screen"]
    stage_2 = stages["stage_2_finalist_selection"]
    latest = max(
        stages["stage_3_promotion_confirmation"]["looks"],
        key=lambda look: look["look_number"],
    )
    audit = stages["deep_audit"]
    stage_0 = stages["stage_0_integrity_and_fixed_probes"]
    return {
        "ordinary": (
            max(
                stage_1["ordinary_color_pairs"],
                stage_2["ordinary_color_pairs"],
            )
            + max(
                latest["powered_ordinary_color_pairs_per_matchup"],
                latest["standard_ordinary_color_pairs"],
            )
            + audit["ordinary_color_pairs"]
        ),
        "lead-40": (
            stage_2["lead_40_color_pairs"]
            + latest["lead_40_color_pairs"]
            + audit["lead_40_color_pairs"]
        ),
        "lead-80": (
            stage_2["lead_80_color_pairs"]
            + latest["lead_80_color_pairs"]
            + audit["lead_80_color_pairs"]
        ),
        "exploitability": max(
            stage_0["exploitability_sentinel_positions"],
            audit["exploitability_positions"],
        ),
        "tactical": 1,
        "baits": 1,
        "tails": 1,
        "sacrifice": 1,
        "small-gain": 1,
        "adversarial": 1,
    }


def finalize_reviewed_bank(
    *,
    auto_path: Path,
    review_queue_path: Path,
    decisions_path: Path,
    labeling_manifest_path: Path,
    policy_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> Mapping[str, Any]:
    auto_rows = _load_jsonl(auto_path, "automatic labels", allow_empty=True)
    review_rows = _load_jsonl(review_queue_path, "review queue", allow_empty=True)
    decisions = _load_jsonl(decisions_path, "review decisions", allow_empty=True)
    labeling_manifest = _load_json(labeling_manifest_path, "labeling manifest")
    policy = load_policy(Path(policy_path))
    active_policy_hash = canonical_sha256(policy)
    if labeling_manifest.get("contract") != LABELING_CONTRACT:
        raise ValueError("labeling manifest contract is unsupported")
    labeling_payload = dict(labeling_manifest)
    labeling_manifest_hash = labeling_payload.pop("manifest_sha256", None)
    if labeling_manifest_hash != canonical_sha256(labeling_payload):
        raise ValueError("labeling manifest self-hash is invalid")
    if labeling_manifest.get("policy_hash") != active_policy_hash:
        raise ValueError("labeling manifest is bound to another promotion policy")
    if (
        labeling_manifest.get("automatic_count") != len(auto_rows)
        or labeling_manifest.get("review_count") != len(review_rows)
        or labeling_manifest.get("automatic_sha256") != file_sha256(Path(auto_path))
        or labeling_manifest.get("review_queue_sha256")
        != file_sha256(Path(review_queue_path))
    ):
        raise ValueError("labeling manifest inputs changed")
    queue = {row.get("semantic_sha256"): row for row in review_rows}
    if None in queue or len(queue) != len(review_rows):
        raise ValueError("review queue semantic identities are invalid")
    decision_map = {row.get("semantic_sha256"): row for row in decisions}
    if (
        None in decision_map
        or len(decision_map) != len(decisions)
        or set(decision_map) != set(queue)
    ):
        raise ValueError("every review queue row requires exactly one decision")

    final_rows = []
    seen = set()
    for row in auto_rows:
        position = normalize_position_sample(row, "automatic labeled position")
        position.pop("metadata", None)
        semantic_hash = semantic_position_sha256(position)
        labels = row.get("labels")
        if (
            not isinstance(labels, list)
            or len(labels) != 1
            or labels[0] not in AUTO_LABELS
        ):
            raise ValueError("automatic row has an invalid label")
        if semantic_hash in seen:
            raise ValueError("automatic rows contain a semantic duplicate")
        seen.add(semantic_hash)
        final_rows.append(
            {
                **position,
                "labels": labels,
                "curation": {
                    "classification": "automatic",
                    "semanticSha256": semantic_hash,
                },
            }
        )
    for semantic_hash in sorted(queue):
        decision = decision_map[semantic_hash]
        approved = decision.get("approved")
        labels = decision.get("labels", [])
        if type(approved) is not bool or not isinstance(labels, list):
            raise ValueError("review decision must declare approved and labels")
        if not approved:
            if labels:
                raise ValueError("rejected review decision may not retain labels")
            continue
        if (
            not labels
            or len(labels) != len(set(labels))
            or any(label not in ALL_LABELS for label in labels)
        ):
            raise ValueError("approved review decision has invalid labels")
        if any(label in AUTO_LABELS for label in labels) and len(labels) != 1:
            raise ValueError(
                "ordinary and Lead labels may not bridge into specialized banks"
            )
        position = normalize_position_sample(
            queue[semantic_hash]["position"], f"reviewed {semantic_hash}"
        )
        hint_loc = decision.get("hint_loc")
        if any(label in {"tactical", "exploitability"} for label in labels):
            if not isinstance(hint_loc, str) or not hint_loc:
                raise ValueError(
                    "tactical/exploitability review requires an explicit hint_loc"
                )
        if hint_loc is not None:
            position["hintLoc"] = hint_loc
            position = normalize_position_sample(
                position, f"reviewed hint {semantic_hash}"
            )
        position.pop("metadata", None)
        if semantic_position_sha256(position) != semantic_hash:
            raise ValueError("review decision semantic identity changed")
        if semantic_hash in seen:
            raise ValueError("reviewed row duplicates an automatic row")
        seen.add(semantic_hash)
        final_rows.append(
            {
                **position,
                "labels": sorted(labels),
                "curation": {
                    "classification": "human-reviewed",
                    "semanticSha256": semantic_hash,
                },
            }
        )
    final_rows.sort(key=lambda row: row["curation"]["semanticSha256"])
    counts = Counter(
        label for row in final_rows for label in row.get("labels", [])
    )
    minima = policy_pool_minima(policy)
    deficits = {
        label: minimum - counts.get(label, 0)
        for label, minimum in minima.items()
        if counts.get(label, 0) < minimum
    }
    if deficits:
        raise ValueError(f"reviewed position pool is below policy minima: {deficits}")
    data = _canonical_jsonl(final_rows)
    manifest = {
        "schema_version": 1,
        "contract": FINAL_MANIFEST_CONTRACT,
        "policy_path": str(Path(policy_path).resolve()),
        "policy_hash": active_policy_hash,
        "labeling_manifest_path": str(Path(labeling_manifest_path).resolve()),
        "labeling_manifest_sha256": file_sha256(Path(labeling_manifest_path)),
        "automatic_input_sha256": file_sha256(Path(auto_path)),
        "review_queue_sha256": file_sha256(Path(review_queue_path)),
        "review_decisions_sha256": file_sha256(Path(decisions_path)),
        "row_count": len(final_rows),
        "label_counts": dict(sorted(counts.items())),
        "required_minima": minima,
        "semantic_hashes_sha256": canonical_sha256(
            [row["curation"]["semanticSha256"] for row in final_rows]
        ),
        "output_path": str(Path(output_path).resolve()),
        "output_sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _publish_file(Path(output_path), data)
    _publish_file(
        Path(manifest_path), (canonical_json(manifest) + "\n").encode("utf-8")
    )
    return manifest


def _analysis_arguments(values: Sequence[str]) -> Dict[str, Path]:
    result = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path or role in result:
            raise ValueError("--analysis values must be unique ROLE=PATH entries")
        result[role] = Path(raw_path)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate immutable risk-score evaluation position pools."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("sources", nargs="+", type=Path)
    normalize.add_argument("-o", "--output", required=True, type=Path)
    normalize.add_argument("--manifest", required=True, type=Path)

    harvest = subparsers.add_parser("harvest-plan")
    harvest.add_argument("--katago", required=True, type=Path)
    harvest.add_argument("--sgfs-dir", action="append", default=[], type=Path)
    harvest.add_argument("--sgf-dir", action="append", default=[], type=Path)
    harvest.add_argument(
        "--training-input-root",
        action="append",
        required=True,
        type=Path,
    )
    harvest.add_argument("--output-dir", required=True, type=Path)
    harvest.add_argument("--manifest", required=True, type=Path)
    harvest.add_argument("--threads", type=int, default=1)

    harvest_execute = subparsers.add_parser("harvest-execute")
    harvest_execute.add_argument("plan", type=Path)

    queries = subparsers.add_parser("queries")
    queries.add_argument("normalized", type=Path)
    queries.add_argument("--output-dir", required=True, type=Path)
    queries.add_argument("--katago", required=True, type=Path)
    queries.add_argument("--analysis-config", required=True, type=Path)
    queries.add_argument("--reference-model", required=True, type=Path)
    queries.add_argument("--policy", required=True, type=Path)

    analyze = subparsers.add_parser("run-analysis")
    analyze.add_argument("--katago", required=True, type=Path)
    analyze.add_argument("--config", required=True, type=Path)
    analyze.add_argument("--model", required=True, type=Path)
    analyze.add_argument("--queries", required=True, type=Path)
    analyze.add_argument("-o", "--output", required=True, type=Path)

    label = subparsers.add_parser("label")
    label.add_argument("normalized", type=Path)
    label.add_argument("--query-manifest", required=True, type=Path)
    label.add_argument("--analysis", action="append", default=[], required=True)
    label.add_argument("--output-dir", required=True, type=Path)
    label.add_argument("--stability-margin", type=float, default=5.0)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--auto", required=True, type=Path)
    finalize.add_argument("--review-queue", required=True, type=Path)
    finalize.add_argument("--decisions", required=True, type=Path)
    finalize.add_argument("--labeling-manifest", required=True, type=Path)
    finalize.add_argument("--policy", required=True, type=Path)
    finalize.add_argument("-o", "--output", required=True, type=Path)
    finalize.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "normalize":
            result = publish_normalized(args.sources, args.output, args.manifest)
        elif args.command == "harvest-plan":
            result = publish_harvest_plan(
                katago=args.katago,
                sgfs_dirs=args.sgfs_dir,
                sgf_dirs=args.sgf_dir,
                training_input_roots=args.training_input_root,
                output_dir=args.output_dir,
                manifest_path=args.manifest,
                threads=args.threads,
            )
        elif args.command == "harvest-execute":
            result = execute_harvest_plan(args.plan)
        elif args.command == "queries":
            result = generate_query_bundle(
                args.normalized,
                args.output_dir,
                katago_binary=args.katago,
                analysis_config=args.analysis_config,
                reference_model=args.reference_model,
                policy_path=args.policy,
            )
        elif args.command == "run-analysis":
            result = run_analysis(
                katago=args.katago,
                config=args.config,
                model=args.model,
                queries=args.queries,
                output=args.output,
            )
        elif args.command == "label":
            result = label_positions(
                normalized_path=args.normalized,
                query_manifest_path=args.query_manifest,
                analysis_paths=_analysis_arguments(args.analysis),
                output_dir=args.output_dir,
                stability_margin=args.stability_margin,
            )
        else:
            result = finalize_reviewed_bank(
                auto_path=args.auto,
                review_queue_path=args.review_queue,
                decisions_path=args.decisions,
                labeling_manifest_path=args.labeling_manifest,
                policy_path=args.policy,
                output_path=args.output,
                manifest_path=args.manifest,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
