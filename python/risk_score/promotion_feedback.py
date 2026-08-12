#!/usr/bin/env python3
"""Generation provenance receipts and promotion feedback reconciliation.

The immutable receipts in this module are the bridge between promoted
self-play, shuffled training data, and trainer checkpoints/exports. Mutable
watermarks are only atomic projections of those receipts and can always be
reconstructed by a watcher after a restart.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from risk_score.adaptive_training import (
    AdaptiveTrainingError,
    load_adaptive_observation,
    load_adaptive_service_spec,
    load_policy as load_adaptive_policy,
    publish_adaptive_observation,
)
from risk_score.promotion_state import PromotionStateError, load_champion


SCHEMA_VERSION = 1
SHUFFLE_MANIFEST_FILENAME = "generation-provenance.json"
SHUFFLE_MANIFEST_CONTRACT = "katago-generation-shuffle-manifest-v1"
SHUFFLE_PROVENANCE_CONTRACT = SHUFFLE_MANIFEST_CONTRACT
TRAINER_RECEIPT_CONTRACT = "katago-generation-trainer-receipt-v1"
DATA_RECEIPT_CONTRACT = "risk-score-generation-data-receipt-v1"
DATA_WATERMARK_CONTRACT = "risk-score-generation-data-watermark-v1"
SHUFFLE_WATERMARK_CONTRACT = "risk-score-generation-shuffle-watermark-v1"
FEEDBACK_EVIDENCE_CONTRACT = "risk-score-promotion-feedback-evidence-v1"
FEEDBACK_DELIVERY_CONTRACT = "risk-score-promotion-feedback-delivery-v1"
AUTONOMY_SERVICE_SPEC_CONTRACT = "risk-score-host-services-v3"
LEGACY_SERVICE_SPEC_CONTRACT = "risk-score-host-services-v2"
ADAPTIVE_SERVICE_BINDING_CONTRACT = (
    "risk-score-adaptive-observation-service-binding-v1"
)
ADAPTIVE_DATA_SNAPSHOT_CONTRACT = (
    "risk-score-adaptive-admitted-data-snapshot-v1"
)
ADAPTIVE_BASELINE_CONTRACT = "risk-score-adaptive-champion-baseline-v1"
ADAPTIVE_PRODUCER_STATUS_CONTRACT = (
    "risk-score-adaptive-observation-producer-status-v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


class PromotionFeedbackError(RuntimeError):
    """A provenance input is malformed, ambiguous, or has changed."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PromotionFeedbackError(
            f"value is not canonical-JSON compatible: {exc}"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def utc_timestamp(
    now: Optional[datetime.datetime] = None,
) -> str:
    current = now or datetime.datetime.now(datetime.timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise PromotionFeedbackError("timestamp must be timezone-aware")
    return (
        current.astimezone(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a mutable canonical JSON projection."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise PromotionFeedbackError(
            f"JSON destination is not a regular non-symlink file: {target}"
        )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Create immutable canonical JSON, accepting an exact retry.

    Returns ``True`` when an existing byte-identical receipt was reused.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(value)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != data
        ):
            raise PromotionFeedbackError(f"immutable receipt conflicts: {target}")
        return True
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != data
            ):
                raise PromotionFeedbackError(
                    f"concurrent immutable receipt conflicts: {target}"
                )
            return True
        _fsync_directory(target.parent)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise PromotionFeedbackError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PromotionFeedbackError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_canonical_json(path: Path, role: str = "JSON artifact") -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PromotionFeedbackError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionFeedbackError(f"cannot load {role} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionFeedbackError(f"{role} must have an object root")
    if data != _canonical_bytes(value):
        raise PromotionFeedbackError(
            f"{role} must be canonical newline-terminated JSON: {source}"
        )
    return value


def _require_hash(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PromotionFeedbackError(
            f"{role} must be a lowercase 64-character SHA-256"
        )
    return value


def _safe_id(value: Any, role: str = "identifier") -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PromotionFeedbackError(f"{role} is not a safe single path component")
    return value


def _absolute_directory(
    path: Path,
    role: str,
    *,
    create: bool = False,
    required: bool = True,
) -> Path:
    value = Path(path)
    if not value.is_absolute() or value.is_symlink():
        raise PromotionFeedbackError(
            f"{role} must be an absolute non-symlink directory"
        )
    if create:
        value.mkdir(parents=True, exist_ok=True)
    if required and not value.is_dir():
        raise PromotionFeedbackError(f"{role} does not exist: {value}")
    if value.exists() and not value.is_dir():
        raise PromotionFeedbackError(f"{role} is not a directory: {value}")
    return value


def _stable_file_record(
    path: Path,
    root: Path,
    *,
    known: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PromotionFeedbackError(f"inventory contains a nonregular file: {source}")
    metadata = {
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "device": before.st_dev,
        "inode": before.st_ino,
    }
    reusable = (
        isinstance(known, Mapping)
        and all(known.get(key) == value for key, value in metadata.items())
        and isinstance(known.get("sha256"), str)
        and _SHA256_RE.fullmatch(known["sha256"]) is not None
    )
    digest = known["sha256"] if reusable else file_sha256(source)
    after = source.lstat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise PromotionFeedbackError(f"file changed while hashing: {source}")
    return {
        "path": source.relative_to(root).as_posix(),
        "sha256": digest,
        **metadata,
    }


def _stable_file_binding(path: Path, role: str) -> Dict[str, Any]:
    """Hash a regular file while rejecting path replacement or mutation."""

    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PromotionFeedbackError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    digest = file_sha256(source)
    after = source.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise PromotionFeedbackError(f"{role} changed while hashing: {source}")
    return {
        "path": str(source),
        "sha256": digest,
        "size": before.st_size,
    }


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _immutable_checkpoint_snapshot(source_path: Path, root: Path) -> Path:
    """Create or reuse a content-addressed copy of a mutable checkpoint."""

    source = Path(source_path)
    destination_root = _absolute_directory(
        Path(root), "adaptive checkpoint snapshot root", create=True
    )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(source), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PromotionFeedbackError(
                f"trainer checkpoint is not a regular file: {source}"
            )
        source_before = source.lstat()
        if (
            stat.S_ISLNK(source_before.st_mode)
            or _descriptor_identity(source_before) != _descriptor_identity(before)
        ):
            raise PromotionFeedbackError(
                f"trainer checkpoint path changed before snapshot: {source}"
            )
        digest_before = _hash_descriptor(descriptor)
        destination = destination_root / f"{digest_before}.ckpt"
        if destination.exists() or destination.is_symlink():
            metadata = destination.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != before.st_size
                or metadata.st_mode & 0o222
                or file_sha256(destination) != digest_before
            ):
                raise PromotionFeedbackError(
                    f"immutable checkpoint snapshot conflicts: {destination}"
                )
        else:
            temporary_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".checkpoint.",
                suffix=".tmp",
                dir=str(destination_root),
            )
            temporary = Path(temporary_name)
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                with os.fdopen(os.dup(descriptor), "rb") as source_handle:
                    with os.fdopen(temporary_descriptor, "wb") as target_handle:
                        shutil.copyfileobj(
                            source_handle,
                            target_handle,
                            length=1024 * 1024,
                        )
                        target_handle.flush()
                        os.fsync(target_handle.fileno())
                os.chmod(temporary, 0o444)
                temporary_metadata = temporary.lstat()
                if (
                    not stat.S_ISREG(temporary_metadata.st_mode)
                    or temporary_metadata.st_size != before.st_size
                    or file_sha256(temporary) != digest_before
                ):
                    raise PromotionFeedbackError(
                        "checkpoint snapshot copy does not match its source"
                    )
                try:
                    os.link(temporary, destination, follow_symlinks=False)
                except FileExistsError:
                    metadata = destination.lstat()
                    if (
                        stat.S_ISLNK(metadata.st_mode)
                        or not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_size != before.st_size
                        or metadata.st_mode & 0o222
                        or file_sha256(destination) != digest_before
                    ):
                        raise PromotionFeedbackError(
                            "concurrent immutable checkpoint snapshot conflicts: "
                            f"{destination}"
                        )
                _fsync_directory(destination_root)
            finally:
                temporary.unlink(missing_ok=True)

        digest_after = _hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        source_after = source.lstat()
        if (
            digest_after != digest_before
            or _descriptor_identity(after) != _descriptor_identity(before)
            or stat.S_ISLNK(source_after.st_mode)
            or _descriptor_identity(source_after) != _descriptor_identity(before)
        ):
            raise PromotionFeedbackError(
                f"trainer checkpoint changed while snapshotting: {source}"
            )
        snapshot = _stable_file_binding(
            destination, "immutable checkpoint snapshot"
        )
        if (
            snapshot["sha256"] != digest_before
            or snapshot["size"] != before.st_size
        ):
            raise PromotionFeedbackError(
                "immutable checkpoint snapshot failed final verification"
            )
        return destination
    finally:
        os.close(descriptor)


def tree_inventory(
    root: Path,
    *,
    suffix: Optional[str] = None,
    exclude_names: Sequence[str] = (),
    ignore_temporary_npz: bool = False,
    known_inventory: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Hash a regular-file-only tree in deterministic path order."""

    base = _absolute_directory(Path(root), "inventory root")
    records: List[Dict[str, Any]] = []
    known_by_path = {
        item["path"]: item
        for item in (known_inventory or ())
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        base, followlinks=False, onerror=raise_walk_error
    ):
        directory_path = Path(directory)
        kept = []
        for name in sorted(dirnames):
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise PromotionFeedbackError(
                    f"inventory contains a symlink directory: {child}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            if name in exclude_names:
                continue
            if suffix is not None and not name.endswith(suffix):
                continue
            if (
                ignore_temporary_npz
                and name.endswith(".npz")
                and "_" in name
            ):
                continue
            path = directory_path / name
            relative = path.relative_to(base).as_posix()
            records.append(
                _stable_file_record(
                    path,
                    base,
                    known=known_by_path.get(relative),
                )
            )
    records.sort(key=lambda item: item["path"])
    return records


def _finalize_receipt(
    payload: Mapping[str, Any], *, hash_field: str = "receipt_sha256"
) -> Dict[str, Any]:
    if hash_field in payload:
        raise PromotionFeedbackError(f"payload already contains {hash_field}")
    value = dict(payload)
    value[hash_field] = canonical_sha256(value)
    return value


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    contract: str,
    role: str,
    hash_field: str = "receipt_sha256",
) -> Dict[str, Any]:
    result = dict(value)
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("contract") != contract
    ):
        raise PromotionFeedbackError(f"{role} contract is unsupported")
    stored = result.pop(hash_field, None)
    if stored != canonical_sha256(result):
        raise PromotionFeedbackError(f"{role} self-hash is invalid")
    result[hash_field] = stored
    return result


def load_data_watermark(path: Path) -> Dict[str, Any]:
    value = load_canonical_json(path, "data watermark")
    return _validate_receipt(
        value,
        contract=DATA_WATERMARK_CONTRACT,
        role="data watermark",
        hash_field="watermark_sha256",
    )


def load_shuffle_watermark(path: Path) -> Dict[str, Any]:
    value = load_canonical_json(path, "shuffle watermark")
    return _validate_receipt(
        value,
        contract=SHUFFLE_WATERMARK_CONTRACT,
        role="shuffle watermark",
        hash_field="watermark_sha256",
    )


def _watermark_lineage_index(
    watermark: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    generations = watermark.get("generations")
    if not isinstance(generations, list):
        raise PromotionFeedbackError("data watermark generations must be an array")
    index: Dict[str, Dict[str, Any]] = {}
    generation_candidates: Dict[str, str] = {}
    for generation in generations:
        if not isinstance(generation, dict):
            raise PromotionFeedbackError("data watermark generation is malformed")
        generation_id = _safe_id(
            generation.get("generation_id"), "watermark generation_id"
        )
        candidate_hash = _require_hash(
            generation.get("candidate_hash"), "watermark candidate_hash"
        )
        receipt_path = Path(generation.get("receipt_path", ""))
        if not receipt_path.is_absolute():
            raise PromotionFeedbackError(
                "data watermark receipt path must be absolute"
            )
        receipt = _validate_receipt(
            load_canonical_json(receipt_path, "generation data receipt"),
            contract=DATA_RECEIPT_CONTRACT,
            role="generation data receipt",
        )
        if (
            receipt.get("receipt_sha256")
            != generation.get("receipt_sha256")
            or receipt.get("generation_id") != generation_id
            or receipt.get("candidate_hash") != candidate_hash
            or receipt.get("roots") != generation.get("roots")
        ):
            raise PromotionFeedbackError(
                "data watermark contradicts its immutable generation receipt"
            )
        previous = generation_candidates.setdefault(generation_id, candidate_hash)
        if previous != candidate_hash:
            raise PromotionFeedbackError(
                f"generation {generation_id} has multiple candidate hashes"
            )
        roots = generation.get("roots")
        if not isinstance(roots, list):
            raise PromotionFeedbackError("watermark roots must be an array")
        for root_value in roots:
            if not isinstance(root_value, dict):
                raise PromotionFeedbackError("watermark root is malformed")
            root = Path(root_value.get("path", ""))
            if not root.is_absolute():
                raise PromotionFeedbackError("watermark root path must be absolute")
            inventory = root_value.get("inventory")
            if not isinstance(inventory, list):
                raise PromotionFeedbackError("watermark inventory must be an array")
            if root_value.get("inventory_sha256") != canonical_sha256(inventory):
                raise PromotionFeedbackError("watermark inventory hash is invalid")
            for record in inventory:
                if not isinstance(record, dict):
                    raise PromotionFeedbackError("watermark file record is malformed")
                relative = Path(record.get("path", ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise PromotionFeedbackError(
                        "watermark inventory path must be relative"
                    )
                if type(record.get("size")) is not int or record["size"] < 0:
                    raise PromotionFeedbackError(
                        "watermark source file size is invalid"
                    )
                for field in ("mtime_ns", "ctime_ns", "device", "inode"):
                    if type(record.get(field)) is not int or record[field] < 0:
                        raise PromotionFeedbackError(
                            f"watermark source file {field} is invalid"
                        )
                absolute = str(root / relative)
                bound = {
                    "generation_id": generation_id,
                    "candidate_hash": candidate_hash,
                    "data_receipt_path": generation.get("receipt_path"),
                    "data_receipt_sha256": generation.get("receipt_sha256"),
                    "sha256": _require_hash(
                        record.get("sha256"), "watermark source file hash"
                    ),
                    "size": record.get("size"),
                    "lineage_status": "admitted",
                }
                existing = index.get(absolute)
                if existing is not None and existing != bound:
                    raise PromotionFeedbackError(
                        f"source file has ambiguous generation lineage: {absolute}"
                    )
                index[absolute] = bound
    historical_root = watermark.get("historical_source_root")
    historical = watermark.get("historical_sources", [])
    if historical_root is not None or historical:
        root = Path(historical_root or "")
        if not root.is_absolute() or not isinstance(historical, list):
            raise PromotionFeedbackError(
                "historical source watermark is malformed"
            )
        if watermark.get("historical_sources_sha256") != canonical_sha256(
            historical
        ):
            raise PromotionFeedbackError(
                "historical source inventory hash is invalid"
            )
        for record in historical:
            if not isinstance(record, dict):
                raise PromotionFeedbackError(
                    "historical source inventory row is malformed"
                )
            relative = Path(record.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise PromotionFeedbackError(
                    "historical source inventory path is unsafe"
                )
            if type(record.get("size")) is not int or record["size"] < 0:
                raise PromotionFeedbackError(
                    "historical source size is invalid"
                )
            for field in ("mtime_ns", "ctime_ns", "device", "inode"):
                if type(record.get(field)) is not int or record[field] < 0:
                    raise PromotionFeedbackError(
                        f"historical source {field} is invalid"
                    )
            absolute = str(root / relative)
            bound = {
                "generation_id": None,
                "candidate_hash": None,
                "data_receipt_path": None,
                "data_receipt_sha256": None,
                "sha256": _require_hash(
                    record.get("sha256"), "historical source hash"
                ),
                "size": record.get("size"),
                "lineage_status": "historical-baseline",
            }
            existing = index.get(absolute)
            if existing is not None and existing != bound:
                raise PromotionFeedbackError(
                    f"historical source has ambiguous lineage: {absolute}"
                )
            index[absolute] = bound
    return index


def bind_source_inventory(
    input_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    *,
    data_watermark_path: Optional[Path],
    strict: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Bind gate inventory rows to admitted generation identities."""

    root = _absolute_directory(Path(input_root), "shuffle input root")
    index: Dict[str, Dict[str, Any]] = {}
    if data_watermark_path is not None:
        watermark = load_data_watermark(Path(data_watermark_path))
        index = _watermark_lineage_index(watermark)
    elif strict:
        raise PromotionFeedbackError(
            "strict shuffle provenance requires a data watermark"
        )

    bound: List[Dict[str, Any]] = []
    candidates: Dict[str, str] = {}
    for item in inventory:
        record = dict(item)
        relative = Path(record.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise PromotionFeedbackError("shuffle input inventory path is unsafe")
        expected_hash = _require_hash(
            record.get("sha256"), "shuffle input file hash"
        )
        absolute = str(root / relative)
        lineage = index.get(absolute)
        if lineage is None:
            if strict:
                raise PromotionFeedbackError(
                    f"shuffle input is not in the admitted data watermark: {absolute}"
                )
            record.update(
                {
                    "generation_id": None,
                    "candidate_hash": None,
                    "data_receipt_path": None,
                    "data_receipt_sha256": None,
                    "lineage_status": "historical-unbound",
                }
            )
        else:
            if lineage["sha256"] != expected_hash or lineage["size"] != record.get(
                "size"
            ):
                raise PromotionFeedbackError(
                    f"admitted source hash drift: {absolute}"
                )
            if lineage["lineage_status"] == "admitted":
                generation_id = lineage["generation_id"]
                candidate_hash = lineage["candidate_hash"]
                previous = candidates.setdefault(generation_id, candidate_hash)
                if previous != candidate_hash:
                    raise PromotionFeedbackError(
                        f"mixed candidate identity for generation {generation_id}"
                    )
            record.update(lineage)
        bound.append(record)
    bound.sort(key=lambda item: item["path"])
    if strict and not bound:
        raise PromotionFeedbackError(
            "strict shuffle provenance requires at least one bound source file"
        )
    return bound, dict(sorted(candidates.items()))


def build_shuffle_manifest(
    *,
    output_path: Path,
    input_root: Path,
    source_inventory: Sequence[Mapping[str, Any]],
    gate_fingerprint_sha256: str,
    command_sha256: str,
    data_watermark_path: Optional[Path],
    strict: bool,
    now: Optional[datetime.datetime] = None,
    manifest_filename: str = SHUFFLE_MANIFEST_FILENAME,
) -> Dict[str, Any]:
    """Build a hash-bound manifest for one completed shuffle directory."""

    output = _absolute_directory(Path(output_path), "shuffle output")
    source_root = _absolute_directory(Path(input_root), "shuffle input root")
    _require_hash(gate_fingerprint_sha256, "gate fingerprint")
    _require_hash(command_sha256, "shuffle command hash")
    watermark_file_hash = (
        file_sha256(Path(data_watermark_path))
        if data_watermark_path is not None
        else None
    )
    bound, candidate_hashes = bind_source_inventory(
        source_root,
        source_inventory,
        data_watermark_path=data_watermark_path,
        strict=strict,
    )
    if data_watermark_path is not None and file_sha256(
        Path(data_watermark_path)
    ) != watermark_file_hash:
        raise PromotionFeedbackError(
            "data watermark changed while binding shuffle provenance"
        )
    output_inventory = tree_inventory(
        output, exclude_names=(manifest_filename,)
    )
    if strict and not any(item["path"].endswith(".npz") for item in output_inventory):
        raise PromotionFeedbackError("strict shuffle output contains no NPZ files")
    generation_ids = sorted(candidate_hashes)
    unbound_count = sum(
        item.get("lineage_status") == "historical-unbound"
        for item in bound
    )
    historical_count = sum(
        item.get("lineage_status") == "historical-baseline" for item in bound
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": SHUFFLE_MANIFEST_CONTRACT,
        "created_at_utc": utc_timestamp(now),
        "output_path": str(output),
        "output_inventory": output_inventory,
        "output_inventory_sha256": canonical_sha256(output_inventory),
        "source_root": str(source_root),
        "source_inventory": bound,
        "source_inventory_sha256": canonical_sha256(bound),
        "generation_ids": generation_ids,
        "candidate_hashes": candidate_hashes,
        "mixed_generation": len(generation_ids) > 1,
        "unbound_source_count": unbound_count,
        "historical_source_count": historical_count,
        "strict": bool(strict),
        "gate_fingerprint_sha256": gate_fingerprint_sha256,
        "command_sha256": command_sha256,
        "data_watermark_path": (
            str(Path(data_watermark_path))
            if data_watermark_path is not None
            else None
        ),
        "data_watermark_file_sha256": (
            watermark_file_hash
        ),
    }
    return _finalize_receipt(payload)


def load_shuffle_manifest(
    path: Path,
    *,
    verify_output: bool = True,
    verify_sources: bool = False,
    manifest_filename: str = SHUFFLE_MANIFEST_FILENAME,
) -> Dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / manifest_filename
    value = _validate_receipt(
        load_canonical_json(source, "shuffle provenance manifest"),
        contract=SHUFFLE_MANIFEST_CONTRACT,
        role="shuffle provenance manifest",
    )
    output = Path(value.get("output_path", ""))
    if not output.is_absolute() or output.resolve() != source.parent.resolve():
        raise PromotionFeedbackError(
            "shuffle manifest output path does not name its containing directory"
        )
    output_inventory = value.get("output_inventory")
    source_inventory = value.get("source_inventory")
    if not isinstance(output_inventory, list) or not isinstance(
        source_inventory, list
    ):
        raise PromotionFeedbackError("shuffle manifest inventories are malformed")
    if value.get("output_inventory_sha256") != canonical_sha256(output_inventory):
        raise PromotionFeedbackError("shuffle output inventory hash is invalid")
    if value.get("source_inventory_sha256") != canonical_sha256(source_inventory):
        raise PromotionFeedbackError("shuffle source inventory hash is invalid")
    for inventory, role in (
        (output_inventory, "shuffle output"),
        (source_inventory, "shuffle source"),
    ):
        seen_paths = set()
        for record in inventory:
            if not isinstance(record, dict):
                raise PromotionFeedbackError(f"{role} inventory row is malformed")
            relative = Path(record.get("path", ""))
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.as_posix() in seen_paths
            ):
                raise PromotionFeedbackError(
                    f"{role} inventory contains an unsafe or duplicate path"
                )
            seen_paths.add(relative.as_posix())
            if type(record.get("size")) is not int or record["size"] < 0:
                raise PromotionFeedbackError(
                    f"{role} inventory size is invalid"
                )
            for field in ("mtime_ns", "ctime_ns", "device", "inode"):
                if type(record.get(field)) is not int or record[field] < 0:
                    raise PromotionFeedbackError(
                        f"{role} inventory {field} is invalid"
                    )
            _require_hash(record.get("sha256"), f"{role} inventory hash")
    candidates = value.get("candidate_hashes")
    generation_ids = value.get("generation_ids")
    if (
        not isinstance(candidates, dict)
        or not isinstance(generation_ids, list)
        or generation_ids != sorted(candidates)
        or value.get("mixed_generation") != (len(generation_ids) > 1)
    ):
        raise PromotionFeedbackError("shuffle generation summary is inconsistent")
    for generation_id, candidate_hash in candidates.items():
        _safe_id(generation_id, "shuffle generation_id")
        _require_hash(candidate_hash, "shuffle candidate hash")
    actual_candidates = {}
    unbound_count = 0
    historical_count = 0
    for record in source_inventory:
        lineage_status = record.get("lineage_status")
        if lineage_status == "admitted":
            generation_id = _safe_id(
                record.get("generation_id"), "shuffle source generation_id"
            )
            candidate_hash = _require_hash(
                record.get("candidate_hash"), "shuffle source candidate hash"
            )
            previous = actual_candidates.setdefault(generation_id, candidate_hash)
            if previous != candidate_hash:
                raise PromotionFeedbackError(
                    f"shuffle generation {generation_id} has mixed candidates"
                )
            if candidates.get(generation_id) != candidate_hash:
                raise PromotionFeedbackError(
                    "shuffle source rows contradict the generation summary"
                )
            receipt_hash = record.get("data_receipt_sha256")
            if receipt_hash is not None:
                _require_hash(receipt_hash, "shuffle data receipt hash")
        elif lineage_status == "historical-unbound":
            unbound_count += 1
            if record.get("generation_id") is not None or record.get(
                "candidate_hash"
            ) is not None:
                raise PromotionFeedbackError(
                    "historical shuffle source carries an unverified identity"
                )
        elif lineage_status == "historical-baseline":
            historical_count += 1
            if (
                record.get("generation_id") is not None
                or record.get("candidate_hash") is not None
                or record.get("data_receipt_path") is not None
                or record.get("data_receipt_sha256") is not None
            ):
                raise PromotionFeedbackError(
                    "historical baseline source carries a generation identity"
                )
        else:
            raise PromotionFeedbackError(
                "shuffle source lineage status is unsupported"
            )
    if actual_candidates != candidates or value.get(
        "unbound_source_count"
    ) != unbound_count or value.get(
        "historical_source_count", 0
    ) != historical_count:
        raise PromotionFeedbackError("shuffle source lineage summary is inconsistent")
    if value.get("strict") is True and unbound_count:
        raise PromotionFeedbackError(
            "strict shuffle manifest contains unbound source lineage"
        )
    if verify_output:
        current = tree_inventory(
            output,
            exclude_names=(manifest_filename,),
            known_inventory=output_inventory,
        )
        if current != output_inventory:
            raise PromotionFeedbackError(
                f"shuffle output inventory or hashes changed: {output}"
            )
    if verify_sources:
        root = Path(value.get("source_root", ""))
        for record in source_inventory:
            relative = Path(record.get("path", ""))
            source_path = root / relative
            if (
                not source_path.is_file()
                or source_path.is_symlink()
                or file_sha256(source_path) != record.get("sha256")
                or source_path.stat().st_size != record.get("size")
            ):
                raise PromotionFeedbackError(
                    f"shuffle source inventory or hashes changed: {source_path}"
                )
    return value


def publish_shuffle_manifest(
    output_path: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_filename: str = SHUFFLE_MANIFEST_FILENAME,
) -> Path:
    """Publish one immutable manifest, accepting an exact semantic retry."""

    destination = Path(output_path) / manifest_filename
    if destination.exists():
        existing = load_shuffle_manifest(
            destination,
            verify_output=True,
            manifest_filename=manifest_filename,
        )
        identity_fields = (
            "output_path",
            "output_inventory_sha256",
            "source_root",
            "source_inventory_sha256",
            "generation_ids",
            "candidate_hashes",
            "gate_fingerprint_sha256",
            "command_sha256",
            "data_watermark_file_sha256",
        )
        if any(existing.get(key) != manifest.get(key) for key in identity_fields):
            raise PromotionFeedbackError(
                f"existing shuffle provenance conflicts: {destination}"
            )
        return destination
    atomic_create_json(destination, dict(manifest))
    load_shuffle_manifest(
        destination,
        verify_output=True,
        manifest_filename=manifest_filename,
    )
    return destination


class TrainerProvenanceRecorder:
    """Persist trainer consumption/checkpoint/export lineage receipts."""

    def __init__(
        self,
        root: Path,
        *,
        strict: bool = False,
        now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(
            datetime.timezone.utc
        ),
        manifest_filename: str = SHUFFLE_MANIFEST_FILENAME,
    ):
        self.root = _absolute_directory(
            Path(root), "trainer provenance root", create=True
        )
        self.strict = bool(strict)
        self.now = now
        self.manifest_filename = manifest_filename
        self.binding: Optional[Dict[str, Any]] = None

    def bind_shuffle(
        self, shuffle_path: Path, *, required: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        directory = _absolute_directory(Path(shuffle_path), "trainer shuffle")
        manifest_path = directory / self.manifest_filename
        must_exist = self.strict if required is None else bool(required)
        if not manifest_path.exists():
            if must_exist:
                raise PromotionFeedbackError(
                    f"shuffle has no generation provenance manifest: {directory}"
                )
            self.binding = None
            return None
        manifest = load_shuffle_manifest(
            manifest_path,
            verify_output=True,
            manifest_filename=self.manifest_filename,
        )
        if self.strict and manifest.get("strict") is not True:
            raise PromotionFeedbackError(
                f"trainer rejects non-strict shuffle provenance: {manifest_path}"
            )
        manifest_file_hash = file_sha256(manifest_path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": TRAINER_RECEIPT_CONTRACT,
            "kind": "shuffle-binding",
            "observed_at_utc": utc_timestamp(self.now()),
            "shuffle_path": str(directory),
            "shuffle_manifest_path": str(manifest_path),
            "shuffle_manifest_file_sha256": manifest_file_hash,
            "shuffle_manifest_receipt_sha256": manifest["receipt_sha256"],
            "generation_ids": manifest["generation_ids"],
            "candidate_hashes": manifest["candidate_hashes"],
        }
        key = canonical_sha256(
            {key: value for key, value in payload.items() if key != "observed_at_utc"}
        )
        path = self.root / "bindings" / f"{key}.json"
        if path.exists():
            receipt = _validate_receipt(
                load_canonical_json(path, "trainer shuffle binding"),
                contract=TRAINER_RECEIPT_CONTRACT,
                role="trainer shuffle binding",
            )
            existing_identity = {
                key: value
                for key, value in receipt.items()
                if key not in {"observed_at_utc", "receipt_sha256"}
            }
            requested_identity = {
                key: value
                for key, value in payload.items()
                if key != "observed_at_utc"
            }
            if existing_identity != requested_identity:
                raise PromotionFeedbackError(
                    "trainer shuffle binding retry changed immutable identity"
                )
        else:
            receipt = _finalize_receipt(payload)
            atomic_create_json(path, receipt)
        self.binding = {
            "shuffle_path": directory,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "manifest_file_sha256": manifest_file_hash,
            "binding_path": path,
            "binding_receipt_sha256": receipt["receipt_sha256"],
        }
        return dict(self.binding)

    def _require_binding(self) -> Dict[str, Any]:
        if self.binding is None:
            raise PromotionFeedbackError(
                "trainer provenance operation has no bound shuffle manifest"
            )
        current = file_sha256(self.binding["manifest_path"])
        if current != self.binding["manifest_file_sha256"]:
            raise PromotionFeedbackError("bound shuffle manifest hash changed")
        load_shuffle_manifest(
            self.binding["manifest_path"],
            verify_output=True,
            manifest_filename=self.manifest_filename,
        )
        return self.binding

    def record_consumption(
        self,
        files: Sequence[Path | str],
        *,
        samples_before: int,
        samples_after: int,
    ) -> Path:
        binding = self._require_binding()
        if (
            type(samples_before) is not int
            or type(samples_after) is not int
            or samples_before < 0
            or samples_after <= samples_before
        ):
            raise PromotionFeedbackError(
                "trainer consumption sample range must be increasing integers"
            )
        if not files:
            raise PromotionFeedbackError("trainer consumption file set is empty")
        output_records = {
            item["path"]: item
            for item in binding["manifest"]["output_inventory"]
        }
        selected = []
        seen = set()
        for raw_path in files:
            path = Path(raw_path)
            if not path.is_absolute():
                path = binding["shuffle_path"] / path
            try:
                relative = path.resolve().relative_to(
                    binding["shuffle_path"].resolve()
                ).as_posix()
            except ValueError as exc:
                raise PromotionFeedbackError(
                    f"trainer selected a file outside the bound shuffle: {path}"
                ) from exc
            if relative in seen:
                raise PromotionFeedbackError(
                    f"trainer consumption contains duplicate file: {relative}"
                )
            seen.add(relative)
            expected = output_records.get(relative)
            if expected is None or not relative.endswith(".npz"):
                raise PromotionFeedbackError(
                    f"trainer file is absent from shuffle manifest: {relative}"
                )
            actual = _stable_file_record(path, binding["shuffle_path"])
            if actual != expected:
                raise PromotionFeedbackError(
                    f"trainer input hash drift: {path}"
                )
            selected.append(actual)
        selected.sort(key=lambda item: item["path"])
        identity = {
            "shuffle_manifest_file_sha256": binding["manifest_file_sha256"],
            "samples_before": samples_before,
            "samples_after": samples_after,
            "selected_files": selected,
        }
        key = canonical_sha256(identity)
        destination = (
            self.root
            / "consumption"
            / f"{samples_before:020d}-{samples_after:020d}-{key[:20]}.json"
        )
        for path, existing in self._consumption_receipts():
            existing_identity = {
                "shuffle_manifest_file_sha256": existing.get(
                    "shuffle_manifest_file_sha256"
                ),
                "samples_before": existing.get("samples_before"),
                "samples_after": existing.get("samples_after"),
                "selected_files": existing.get("selected_files"),
            }
            if existing_identity == identity:
                return path
            existing_before = existing.get("samples_before")
            existing_after = existing.get("samples_after")
            if (
                type(existing_before) is int
                and type(existing_after) is int
                and samples_before < existing_after
                and existing_before < samples_after
            ):
                raise PromotionFeedbackError(
                    "trainer consumption sample ranges overlap with different lineage"
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": TRAINER_RECEIPT_CONTRACT,
            "kind": "consumption",
            "observed_at_utc": utc_timestamp(self.now()),
            "shuffle_path": str(binding["shuffle_path"]),
            "shuffle_manifest_path": str(binding["manifest_path"]),
            "shuffle_manifest_file_sha256": binding["manifest_file_sha256"],
            "shuffle_manifest_receipt_sha256": binding["manifest"][
                "receipt_sha256"
            ],
            "generation_ids": binding["manifest"]["generation_ids"],
            "candidate_hashes": binding["manifest"]["candidate_hashes"],
            "samples_before": samples_before,
            "samples_after": samples_after,
            "selected_files": selected,
            "selected_files_sha256": canonical_sha256(selected),
        }
        atomic_create_json(destination, _finalize_receipt(payload))
        self._replace_trainer_projection()
        return destination

    def _consumption_receipts(self) -> List[tuple[Path, Dict[str, Any]]]:
        directory = self.root / "consumption"
        if not directory.exists():
            return []
        receipts = []
        for path in sorted(directory.glob("*.json")):
            value = _validate_receipt(
                load_canonical_json(path, "trainer consumption receipt"),
                contract=TRAINER_RECEIPT_CONTRACT,
                role="trainer consumption receipt",
            )
            if value.get("kind") != "consumption":
                raise PromotionFeedbackError(
                    f"non-consumption receipt in consumption directory: {path}"
                )
            receipts.append((path, value))
        return receipts

    def _latest_consumption(
        self, sample_count: int, manifest_hash: str
    ) -> Optional[tuple[Path, Dict[str, Any]]]:
        matches = [
            item
            for item in self._consumption_receipts()
            if item[1].get("shuffle_manifest_file_sha256") == manifest_hash
            and type(item[1].get("samples_after")) is int
            and item[1]["samples_after"] <= sample_count
        ]
        return max(matches, key=lambda item: item[1]["samples_after"], default=None)

    def _consumption_lineage(self, sample_count: int) -> List[Dict[str, Any]]:
        lineage = []
        for path, value in self._consumption_receipts():
            samples_after = value.get("samples_after")
            if type(samples_after) is not int or samples_after > sample_count:
                continue
            lineage.append(
                {
                    "path": str(path),
                    "receipt_sha256": value["receipt_sha256"],
                    "samples_before": value["samples_before"],
                    "samples_after": samples_after,
                    "shuffle_manifest_file_sha256": value[
                        "shuffle_manifest_file_sha256"
                    ],
                    "generation_ids": value["generation_ids"],
                    "candidate_hashes": value["candidate_hashes"],
                }
            )
        lineage.sort(
            key=lambda item: (
                item["samples_before"],
                item["samples_after"],
                item["path"],
            )
        )
        for first, second in zip(lineage, lineage[1:]):
            if second["samples_before"] < first["samples_after"]:
                raise PromotionFeedbackError(
                    "trainer consumption lineage contains overlapping sample ranges"
                )
        return lineage

    def record_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        sample_count: int,
        kind: str = "checkpoint",
    ) -> Path:
        if kind not in {"checkpoint", "longterm-checkpoint"}:
            raise PromotionFeedbackError(f"unsupported checkpoint kind: {kind}")
        binding = self._require_binding()
        if type(sample_count) is not int or sample_count < 0:
            raise PromotionFeedbackError("checkpoint sample count must be nonnegative")
        source = Path(checkpoint_path)
        if not source.is_absolute():
            source = source.resolve()
        if source.is_symlink() or not source.is_file():
            raise PromotionFeedbackError(
                f"trainer checkpoint is not a regular file: {source}"
            )
        consumption = self._latest_consumption(
            sample_count, binding["manifest_file_sha256"]
        )
        if consumption is None:
            if self.strict:
                raise PromotionFeedbackError(
                    "strict checkpoint provenance has no prior consumption receipt"
                )
            consumption_path = None
            consumption_hash = None
        else:
            consumption_path = str(consumption[0])
            consumption_hash = consumption[1]["receipt_sha256"]
        consumption_lineage = self._consumption_lineage(sample_count)
        artifact = {
            "path": str(source),
            "size": source.stat().st_size,
            "sha256": file_sha256(source),
        }
        identity = {
            "kind": kind,
            "sample_count": sample_count,
            "artifact": artifact,
            "shuffle_manifest_file_sha256": binding["manifest_file_sha256"],
            "consumption_receipt_sha256": consumption_hash,
            "consumption_lineage_sha256": canonical_sha256(
                consumption_lineage
            ),
        }
        key = canonical_sha256(identity)
        destination = (
            self.root
            / "checkpoints"
            / f"{sample_count:020d}-{key[:20]}.json"
        )
        checkpoints_root = self.root / "checkpoints"
        if checkpoints_root.exists():
            for path in sorted(checkpoints_root.glob("*.json")):
                existing = _validate_receipt(
                    load_canonical_json(path, "trainer checkpoint receipt"),
                    contract=TRAINER_RECEIPT_CONTRACT,
                    role="trainer checkpoint receipt",
                )
                existing_identity = {
                    field: existing.get(field)
                    for field in (
                        "kind",
                        "sample_count",
                        "artifact",
                        "shuffle_manifest_file_sha256",
                        "consumption_receipt_sha256",
                        "consumption_lineage_sha256",
                    )
                }
                if existing_identity == identity:
                    return path
                if (
                    existing.get("kind") == kind
                    and existing.get("sample_count") == sample_count
                ):
                    raise PromotionFeedbackError(
                        "checkpoint sample already has different immutable lineage"
                    )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": TRAINER_RECEIPT_CONTRACT,
            "observed_at_utc": utc_timestamp(self.now()),
            **identity,
            "shuffle_manifest_path": str(binding["manifest_path"]),
            "shuffle_manifest_receipt_sha256": binding["manifest"][
                "receipt_sha256"
            ],
            "generation_ids": binding["manifest"]["generation_ids"],
            "candidate_hashes": binding["manifest"]["candidate_hashes"],
            "consumption_receipt_path": consumption_path,
            "consumption_lineage": consumption_lineage,
        }
        atomic_create_json(destination, _finalize_receipt(payload))
        self._replace_trainer_projection()
        return destination

    def record_export(
        self, export_path: Path, *, sample_count: int
    ) -> Path:
        binding = self._require_binding()
        if type(sample_count) is not int or sample_count < 0:
            raise PromotionFeedbackError("export sample count must be nonnegative")
        source = _absolute_directory(Path(export_path), "trainer export")
        inventory = tree_inventory(source)
        if not inventory:
            raise PromotionFeedbackError("trainer export is empty")
        consumption = self._latest_consumption(
            sample_count, binding["manifest_file_sha256"]
        )
        if consumption is None and self.strict:
            raise PromotionFeedbackError(
                "strict export provenance has no prior consumption receipt"
            )
        consumption_lineage = self._consumption_lineage(sample_count)
        identity = {
            "kind": "export",
            "sample_count": sample_count,
            "artifact": {
                "path": str(source),
                "inventory": inventory,
                "inventory_sha256": canonical_sha256(inventory),
            },
            "shuffle_manifest_file_sha256": binding["manifest_file_sha256"],
            "consumption_receipt_sha256": (
                consumption[1]["receipt_sha256"]
                if consumption is not None
                else None
            ),
            "consumption_lineage_sha256": canonical_sha256(
                consumption_lineage
            ),
        }
        key = canonical_sha256(identity)
        destination = (
            self.root / "exports" / f"{sample_count:020d}-{key[:20]}.json"
        )
        exports_root = self.root / "exports"
        if exports_root.exists():
            for path in sorted(exports_root.glob("*.json")):
                existing = _validate_receipt(
                    load_canonical_json(path, "trainer export receipt"),
                    contract=TRAINER_RECEIPT_CONTRACT,
                    role="trainer export receipt",
                )
                existing_identity = {
                    field: existing.get(field)
                    for field in (
                        "kind",
                        "sample_count",
                        "artifact",
                        "shuffle_manifest_file_sha256",
                        "consumption_receipt_sha256",
                        "consumption_lineage_sha256",
                    )
                }
                if existing_identity == identity:
                    return path
                if existing.get("sample_count") == sample_count:
                    raise PromotionFeedbackError(
                        "export sample already has different immutable lineage"
                    )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": TRAINER_RECEIPT_CONTRACT,
            "observed_at_utc": utc_timestamp(self.now()),
            **identity,
            "shuffle_manifest_path": str(binding["manifest_path"]),
            "shuffle_manifest_receipt_sha256": binding["manifest"][
                "receipt_sha256"
            ],
            "generation_ids": binding["manifest"]["generation_ids"],
            "candidate_hashes": binding["manifest"]["candidate_hashes"],
            "consumption_receipt_path": (
                str(consumption[0]) if consumption is not None else None
            ),
            "consumption_lineage": consumption_lineage,
        }
        atomic_create_json(destination, _finalize_receipt(payload))
        self._replace_trainer_projection()
        return destination

    def _replace_trainer_projection(self) -> None:
        receipts = []
        for directory in ("consumption", "checkpoints", "exports"):
            root = self.root / directory
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json")):
                value = _validate_receipt(
                    load_canonical_json(path, "trainer lineage receipt"),
                    contract=TRAINER_RECEIPT_CONTRACT,
                    role="trainer lineage receipt",
                )
                receipts.append(
                    {
                        "path": str(path),
                        "kind": value["kind"],
                        "receipt_sha256": value["receipt_sha256"],
                        "sample_count": value.get(
                            "sample_count", value.get("samples_after")
                        ),
                        "generation_ids": value.get("generation_ids", []),
                    }
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": "katago-generation-trainer-watermark-v1",
            "receipts": receipts,
        }
        value = dict(payload)
        value["watermark_sha256"] = canonical_sha256(payload)
        atomic_replace_json(self.root / "trainer-watermark.json", value)


class PromotionFeedbackWatcher:
    """Reconcile immutable lineage and deliver first-milestone feedback."""

    def __init__(
        self,
        *,
        promotion_root: Path,
        admitted_root: Path,
        shuffle_root: Path,
        trainer_provenance_root: Path,
        state_root: Path,
        data_watermark_path: Path,
        shuffle_watermark_path: Path,
        rollout_root: Optional[Path] = None,
        runtime_config: Optional[Path] = None,
        strict: bool = False,
        feedback_recorder: Optional[
            Callable[[str, str, Path], Mapping[str, Any] | None]
        ] = None,
        now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(
            datetime.timezone.utc
        ),
        sleeper: Callable[[float], None] = time.sleep,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.promotion_root = _absolute_directory(
            Path(promotion_root), "promotion root"
        )
        self.admitted_root = _absolute_directory(
            Path(admitted_root), "admitted self-play root"
        )
        self.shuffle_root = _absolute_directory(
            Path(shuffle_root), "shuffle root"
        )
        self.trainer_root = _absolute_directory(
            Path(trainer_provenance_root),
            "trainer provenance root",
            create=True,
        )
        self.state_root = _absolute_directory(
            Path(state_root), "feedback state root", create=True
        )
        self.data_watermark_path = Path(data_watermark_path)
        self.shuffle_watermark_path = Path(shuffle_watermark_path)
        for path, role in (
            (self.data_watermark_path, "data watermark"),
            (self.shuffle_watermark_path, "shuffle watermark"),
        ):
            if not path.is_absolute() or path.is_symlink():
                raise PromotionFeedbackError(
                    f"{role} path must be absolute and non-symlink"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
        self.rollout_root = (
            Path(rollout_root)
            if rollout_root is not None
            else self.promotion_root / "rollouts"
        )
        if not self.rollout_root.is_absolute() or self.rollout_root.is_symlink():
            raise PromotionFeedbackError(
                "rollout root must be absolute and non-symlink"
            )
        self.runtime_config = (
            Path(runtime_config) if runtime_config is not None else None
        )
        self.strict = bool(strict)
        self.feedback_recorder = feedback_recorder
        self.now = now
        self.sleeper = sleeper
        self.command_runner = command_runner

    def _transactions(self) -> Dict[str, Dict[str, Any]]:
        root = self.promotion_root / "transactions"
        generations: Dict[str, Dict[str, Any]] = {}
        intent_paths = sorted(root.glob("*/intent.json")) if root.is_dir() else []
        for intent_path in intent_paths:
            intent = load_canonical_json(intent_path, "promotion intent")
            generation_id = _safe_id(
                intent.get("generation_id"), "promotion generation_id"
            )
            candidate_hash = _require_hash(
                intent.get("candidate_hash"), "promotion candidate_hash"
            )
            if intent_path.parent.name != generation_id:
                raise PromotionFeedbackError(
                    "promotion intent directory contradicts generation_id"
                )
            if generation_id in generations:
                raise PromotionFeedbackError(
                    f"duplicate promotion transaction: {generation_id}"
                )
            generations[generation_id] = {
                "generation_id": generation_id,
                "candidate_hash": candidate_hash,
                "intent": intent,
                "intent_path": intent_path,
                "transaction": intent_path.parent,
                "rolled_back": (intent_path.parent / "rolled-back.json").is_file(),
                "pending": not (intent_path.parent / "complete.json").is_file(),
            }
        if not any(item.get("bootstrap") for item in generations.values()):
            bootstrap_source = None
            bootstrap_value = None
            events_root = self.promotion_root / "events"
            if events_root.is_dir():
                for event_path in sorted(events_root.glob("*.json")):
                    event = load_canonical_json(
                        event_path, "promotion lifecycle event"
                    )
                    if event.get("transition") != "champion.bootstrapped":
                        continue
                    body = dict(event)
                    event_hash = body.pop("event_hash", None)
                    if event_hash != canonical_sha256(body):
                        raise PromotionFeedbackError(
                            "bootstrap lifecycle event hash is invalid"
                        )
                    bootstrap_source = event_path
                    bootstrap_value = event
                    break
            champion_path = self.promotion_root / "champion.json"
            if bootstrap_value is None and champion_path.is_file():
                champion = load_canonical_json(
                    champion_path, "champion projection"
                )
                if champion.get("bootstrap") is True:
                    body = dict(champion)
                    record_hash = body.pop("record_hash", None)
                    if record_hash != canonical_sha256(body):
                        raise PromotionFeedbackError(
                            "bootstrap champion record hash is invalid"
                        )
                    bootstrap_source = champion_path
                    bootstrap_value = {
                        "champion_hash": champion.get("champion_hash"),
                        "payload": {
                            "generation_id": champion.get("generation_id")
                        },
                    }
            if bootstrap_source is not None and bootstrap_value is not None:
                payload = bootstrap_value.get("payload")
                if not isinstance(payload, dict):
                    raise PromotionFeedbackError(
                        "bootstrap lifecycle event payload is malformed"
                    )
                generation_id = _safe_id(
                    payload.get("generation_id"),
                    "bootstrap generation_id",
                )
                candidate_hash = _require_hash(
                    bootstrap_value.get("champion_hash"),
                    "bootstrap champion hash",
                )
                if generation_id not in generations:
                    generations[generation_id] = {
                        "generation_id": generation_id,
                        "candidate_hash": candidate_hash,
                        "intent": bootstrap_value,
                        "intent_path": bootstrap_source,
                        "transaction": (
                            self.promotion_root
                            / "transactions"
                            / generation_id
                        ),
                        "rolled_back": False,
                        "pending": False,
                        "bootstrap": True,
                    }
        return generations

    def _generation_data_roots(self, generation_id: str) -> List[Path]:
        roots = [
            self.admitted_root / generation_id,
            self.admitted_root / "continuous" / generation_id,
        ]
        return [path for path in roots if path.is_dir() and not path.is_symlink()]

    def _archive_watermark(
        self, kind: str, value: Mapping[str, Any]
    ) -> None:
        digest = value["watermark_sha256"]
        atomic_create_json(
            self.state_root / "watermark-history" / kind / f"{digest}.json",
            dict(value),
        )

    def _reconcile_data_watermark(
        self,
        generations: Mapping[str, Mapping[str, Any]],
        *,
        frozen: bool,
    ) -> Dict[str, Any]:
        previous = (
            load_data_watermark(self.data_watermark_path)
            if self.data_watermark_path.is_file()
            else None
        )
        if frozen:
            if previous is None:
                raise PromotionFeedbackError(
                    "promotion transaction cannot freeze a missing data watermark"
                )
            for generation in previous.get("generations", []):
                for root_value in generation.get("roots", []):
                    root = Path(root_value["path"])
                    current = tree_inventory(
                        root,
                        suffix=".npz",
                        ignore_temporary_npz=True,
                        known_inventory=root_value["inventory"],
                    )
                    if current != root_value["inventory"]:
                        raise PromotionFeedbackError(
                            "frozen admitted data changed during promotion: "
                            f"{root}"
                        )
            historical_root = previous.get("historical_source_root")
            for record in previous.get("historical_sources", []):
                root = Path(historical_root)
                current = _stable_file_record(
                    root / record["path"],
                    root,
                    known=record,
                )
                if current != record:
                    raise PromotionFeedbackError(
                        "frozen historical source changed during promotion: "
                        f"{root / record['path']}"
                    )
            return previous
        previous_index = (
            _watermark_lineage_index(previous) if previous is not None else {}
        )
        previous_generations = {
            item["generation_id"]: item
            for item in (previous or {}).get("generations", [])
        }
        rows = []
        for generation_id, generation in sorted(generations.items()):
            if generation["rolled_back"]:
                continue
            old_generation = previous_generations.get(generation_id)
            old_roots = {
                item["path"]: item
                for item in (old_generation or {}).get("roots", [])
            }
            root_rows = []
            for root in self._generation_data_roots(generation_id):
                known_root = old_roots.get(str(root))
                inventory = tree_inventory(
                    root,
                    suffix=".npz",
                    ignore_temporary_npz=True,
                    known_inventory=(
                        known_root.get("inventory", [])
                        if known_root is not None
                        else ()
                    ),
                )
                for record in inventory:
                    absolute = str(root / record["path"])
                    old = previous_index.get(absolute)
                    if old is not None and (
                        old["sha256"] != record["sha256"]
                        or old["size"] != record["size"]
                    ):
                        raise PromotionFeedbackError(
                            f"admitted data hash drift: {absolute}"
                        )
                root_rows.append(
                    {
                        "path": str(root),
                        "inventory": inventory,
                        "inventory_sha256": canonical_sha256(inventory),
                    }
                )
            if not root_rows:
                continue
            admission_marker = (
                generation["transaction"] / "generation-data-admitted.json"
            )
            marker_hash = (
                file_sha256(admission_marker)
                if admission_marker.is_file() and not admission_marker.is_symlink()
                else None
            )
            if old_generation is not None:
                old_receipt = _validate_receipt(
                    load_canonical_json(
                        Path(old_generation["receipt_path"]),
                        "previous generation data receipt",
                    ),
                    contract=DATA_RECEIPT_CONTRACT,
                    role="previous generation data receipt",
                )
                if (
                    old_receipt.get("promotion_intent_sha256")
                    != file_sha256(generation["intent_path"])
                    or old_receipt.get("admission_marker_sha256")
                    != marker_hash
                ):
                    raise PromotionFeedbackError(
                        f"generation control artifact hash drift: {generation_id}"
                    )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "contract": DATA_RECEIPT_CONTRACT,
                "generation_id": generation_id,
                "candidate_hash": generation["candidate_hash"],
                "promotion_intent_path": str(generation["intent_path"]),
                "promotion_intent_sha256": file_sha256(
                    generation["intent_path"]
                ),
                "admission_marker_path": (
                    str(admission_marker) if marker_hash is not None else None
                ),
                "admission_marker_sha256": marker_hash,
                "roots": root_rows,
                "source_inventory_sha256": canonical_sha256(root_rows),
            }
            receipt = _finalize_receipt(payload)
            receipt_path = (
                self.state_root
                / "data-receipts"
                / generation_id
                / f"{receipt['receipt_sha256']}.json"
            )
            atomic_create_json(receipt_path, receipt)
            rows.append(
                {
                    "generation_id": generation_id,
                    "candidate_hash": generation["candidate_hash"],
                    "roots": root_rows,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": receipt["receipt_sha256"],
                    "admitted": marker_hash is not None,
                }
            )
        rows.sort(key=lambda item: item["generation_id"])
        generation_paths = {
            str(Path(root["path"]) / record["path"])
            for generation in rows
            for root in generation["roots"]
            for record in root["inventory"]
        }
        known_for_admitted_root = []
        for generation in rows:
            for root_value in generation["roots"]:
                root = Path(root_value["path"])
                for record in root_value["inventory"]:
                    transformed = dict(record)
                    transformed["path"] = (
                        root / record["path"]
                    ).relative_to(self.admitted_root).as_posix()
                    known_for_admitted_root.append(transformed)
        previous_historical = (previous or {}).get("historical_sources", [])
        known_for_admitted_root.extend(previous_historical)
        complete_inventory = tree_inventory(
            self.admitted_root,
            suffix=".npz",
            ignore_temporary_npz=True,
            known_inventory=known_for_admitted_root,
        )
        historical_sources = [
            record
            for record in complete_inventory
            if str(self.admitted_root / record["path"]) not in generation_paths
        ]
        if previous is not None and "historical_sources" in previous:
            old_historical = {
                item["path"]: item for item in previous_historical
            }
            current_historical = {
                item["path"]: item for item in historical_sources
            }
            changed = sorted(
                path
                for path in set(old_historical).intersection(current_historical)
                if old_historical[path] != current_historical[path]
            )
            added = sorted(set(current_historical) - set(old_historical))
            removed_historical = sorted(
                set(old_historical) - set(current_historical)
            )
            if self.strict and (changed or added or removed_historical):
                raise PromotionFeedbackError(
                    "historical source baseline changed; "
                    f"changed={changed[:3]}, added={added[:3]}, "
                    f"removed={removed_historical[:3]}"
                )
        current_paths = generation_paths.union(
            str(self.admitted_root / record["path"])
            for record in historical_sources
        )
        if previous is not None:
            removed = sorted(set(previous_index) - current_paths)
            live_removed = [
                path
                for path in removed
                if previous_index[path]["generation_id"] in {
                    generation_id
                    for generation_id, generation in generations.items()
                    if not generation["rolled_back"]
                }
            ]
            if live_removed and self.strict:
                raise PromotionFeedbackError(
                    "admitted data disappeared without rollback: "
                    + ", ".join(live_removed[:5])
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": DATA_WATERMARK_CONTRACT,
            "generations": rows,
            "historical_source_root": str(self.admitted_root),
            "historical_sources": historical_sources,
            "historical_sources_sha256": canonical_sha256(
                historical_sources
            ),
            "generation_paths": sorted(
                {
                    root["path"]
                    for generation in rows
                    for root in generation["roots"]
                }
            ),
        }
        value = dict(payload)
        value["watermark_sha256"] = canonical_sha256(payload)
        self._archive_watermark("data", value)
        atomic_replace_json(self.data_watermark_path, value)
        return value

    def _reconcile_shuffle_watermark(
        self,
        generations: Mapping[str, Mapping[str, Any]],
        *,
        frozen: bool,
    ) -> Dict[str, Any]:
        previous = (
            load_shuffle_watermark(self.shuffle_watermark_path)
            if self.shuffle_watermark_path.is_file()
            else None
        )
        if frozen:
            if previous is None:
                raise PromotionFeedbackError(
                    "promotion transaction cannot freeze a missing shuffle watermark"
                )
            for item in previous.get("manifests", []):
                manifest_path = Path(item["manifest_path"])
                if (
                    file_sha256(manifest_path)
                    != item.get("manifest_file_sha256")
                ):
                    raise PromotionFeedbackError(
                        f"frozen shuffle manifest changed: {manifest_path}"
                    )
                load_shuffle_manifest(manifest_path, verify_output=True)
            return previous
        previous_historical = {
            item["path"]: item
            for item in (previous or {}).get("historical_paths", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        previous_manifests = {
            item["path"]: item
            for item in (previous or {}).get("manifests", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        manifests = []
        historical = []
        by_generation: Dict[str, List[str]] = {}
        for directory in sorted(self.shuffle_root.iterdir()):
            if directory.name.endswith(".tmp") or directory.is_symlink():
                continue
            if not directory.is_dir():
                continue
            manifest_path = directory / SHUFFLE_MANIFEST_FILENAME
            if not manifest_path.is_file():
                old = previous_historical.get(str(directory))
                if previous is not None and old is None and self.strict:
                    raise PromotionFeedbackError(
                        f"new shuffle directory lacks provenance: {directory}"
                    )
                if old is None:
                    inventory = tree_inventory(directory)
                    old = {
                        "path": str(directory),
                        "inventory_sha256": canonical_sha256(inventory),
                    }
                historical.append(old)
                continue
            manifest = load_shuffle_manifest(manifest_path, verify_output=True)
            if self.strict and manifest.get("strict") is not True:
                raise PromotionFeedbackError(
                    f"strict watcher rejects non-strict shuffle manifest: "
                    f"{manifest_path}"
                )
            manifest_hash = file_sha256(manifest_path)
            old = previous_manifests.get(str(directory))
            if old is not None and old.get("manifest_file_sha256") != manifest_hash:
                raise PromotionFeedbackError(
                    f"shuffle manifest hash drift: {manifest_path}"
                )
            row = {
                "path": str(directory),
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": manifest_hash,
                "manifest_receipt_sha256": manifest["receipt_sha256"],
                "source_inventory_sha256": manifest["source_inventory_sha256"],
                "output_inventory_sha256": manifest["output_inventory_sha256"],
                "generation_ids": manifest["generation_ids"],
                "mixed_generation": manifest["mixed_generation"],
            }
            manifests.append(row)
            for generation_id in manifest["generation_ids"]:
                by_generation.setdefault(generation_id, []).append(str(directory))
        manifests.sort(key=lambda item: item["path"])
        historical.sort(key=lambda item: item["path"])
        derived_paths = sorted(
            item["path"] for item in manifests if item["generation_ids"]
        )
        active_generation_ids = {
            generation_id
            for generation_id, generation in generations.items()
            if not generation["rolled_back"]
        }
        consumption_by_generation: Dict[str, List[str]] = {}
        for _, receipt in self._trainer_consumption_rows(
            active_generation_ids
        ):
            for generation_id in receipt.get("generation_ids", []):
                consumption_by_generation.setdefault(
                    generation_id, []
                ).append(receipt["receipt_sha256"])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": SHUFFLE_WATERMARK_CONTRACT,
            # The flat field is retained for PromotionController v1. The
            # per-generation map is authoritative for precise reconciliation.
            "derived_paths": derived_paths,
            "derived_paths_by_generation": {
                generation_id: sorted(paths)
                for generation_id, paths in sorted(by_generation.items())
            },
            "trainer_consumed_by_generation": {
                generation_id: bool(
                    consumption_by_generation.get(generation_id)
                )
                for generation_id in sorted(active_generation_ids)
            },
            "trainer_consumption_receipts_by_generation": {
                generation_id: sorted(set(receipts))
                for generation_id, receipts in sorted(
                    consumption_by_generation.items()
                )
            },
            "manifests": manifests,
            "historical_paths": historical,
        }
        value = dict(payload)
        value["watermark_sha256"] = canonical_sha256(payload)
        self._archive_watermark("shuffle", value)
        atomic_replace_json(self.shuffle_watermark_path, value)
        return value

    def _candidate_roots(self, generation_id: str) -> List[Path]:
        roots = [
            self.rollout_root / generation_id / "data",
            *self._generation_data_roots(generation_id),
        ]
        return [
            path
            for path in roots
            if path.is_dir() and not path.is_symlink()
        ]

    def _iter_suffix(self, roots: Sequence[Path], suffix: str) -> List[Path]:
        values = []
        for root in roots:
            inventory = tree_inventory(
                root,
                suffix=suffix,
                ignore_temporary_npz=suffix == ".npz",
            )
            values.extend(root / item["path"] for item in inventory)
        return sorted(set(values), key=lambda path: str(path))

    def _evidence_path(self, generation_id: str, milestone: str) -> Path:
        return (
            self.state_root
            / "evidence"
            / generation_id
            / f"{milestone}.json"
        )

    def _publish_evidence(
        self,
        generation: Mapping[str, Any],
        milestone: str,
        details: Mapping[str, Any],
    ) -> Path:
        path = self._evidence_path(generation["generation_id"], milestone)
        if path.exists():
            _validate_receipt(
                load_canonical_json(path, "promotion feedback evidence"),
                contract=FEEDBACK_EVIDENCE_CONTRACT,
                role="promotion feedback evidence",
                hash_field="evidence_sha256",
            )
            return path
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": FEEDBACK_EVIDENCE_CONTRACT,
            "generation_id": generation["generation_id"],
            "candidate_hash": generation["candidate_hash"],
            "milestone": milestone,
            "observed_at_utc": utc_timestamp(self.now()),
            **dict(details),
        }
        evidence = _finalize_receipt(payload, hash_field="evidence_sha256")
        atomic_create_json(path, evidence)
        return path

    def _deliver(
        self, generation_id: str, kind: str, evidence_path: Path
    ) -> bool:
        delivery_path = (
            self.state_root / "deliveries" / generation_id / f"{kind}.json"
        )
        evidence_hash = file_sha256(evidence_path)
        if delivery_path.exists():
            delivery = _validate_receipt(
                load_canonical_json(delivery_path, "feedback delivery receipt"),
                contract=FEEDBACK_DELIVERY_CONTRACT,
                role="feedback delivery receipt",
            )
            if delivery.get("evidence_file_sha256") != evidence_hash:
                raise PromotionFeedbackError(
                    "delivered feedback evidence hash changed"
                )
            return False
        if self.feedback_recorder is not None:
            result = self.feedback_recorder(generation_id, kind, evidence_path)
            outcome = dict(result or {})
        elif self.runtime_config is not None:
            completed = self.command_runner(
                [
                    sys.executable,
                    "-m",
                    "risk_score.promotion_host",
                    "feedback-record",
                    "--runtime-config",
                    str(self.runtime_config),
                    "--generation",
                    generation_id,
                    "--kind",
                    kind,
                    "--evidence",
                    str(evidence_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise PromotionFeedbackError(
                    f"feedback-record failed for {generation_id}/{kind}: "
                    f"{completed.stderr.strip()}"
                )
            try:
                outcome = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise PromotionFeedbackError(
                    "feedback-record did not return JSON evidence"
                ) from exc
        else:
            return False
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": FEEDBACK_DELIVERY_CONTRACT,
            "generation_id": generation_id,
            "kind": kind,
            "evidence_path": str(evidence_path),
            "evidence_file_sha256": evidence_hash,
            "delivered_at_utc": utc_timestamp(self.now()),
            "outcome": outcome,
        }
        atomic_create_json(delivery_path, _finalize_receipt(payload))
        return True

    def _trainer_consumption_rows(
        self, active_generation_ids: Iterable[str]
    ) -> List[tuple[Path, Dict[str, Any]]]:
        root = self.trainer_root / "consumption"
        rows = []
        active = set(active_generation_ids)
        if not root.exists():
            return rows
        for path in sorted(root.glob("*.json")):
            value = _validate_receipt(
                load_canonical_json(path, "trainer consumption receipt"),
                contract=TRAINER_RECEIPT_CONTRACT,
                role="trainer consumption receipt",
            )
            if active.isdisjoint(value.get("generation_ids", [])):
                continue
            manifest_path = Path(value.get("shuffle_manifest_path", ""))
            if (
                not manifest_path.is_absolute()
                or file_sha256(manifest_path)
                != value.get("shuffle_manifest_file_sha256")
            ):
                raise PromotionFeedbackError(
                    f"trainer consumption shuffle binding changed: {path}"
                )
            load_shuffle_manifest(manifest_path, verify_output=True)
            rows.append((path, value))
        return rows

    def _milestones(
        self,
        generations: Mapping[str, Mapping[str, Any]],
        data_watermark: Mapping[str, Any],
        shuffle_watermark: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        events = []
        data_rows = {
            item["generation_id"]: item
            for item in data_watermark.get("generations", [])
        }
        shuffle_rows: Dict[str, List[Mapping[str, Any]]] = {}
        for item in shuffle_watermark.get("manifests", []):
            for generation_id in item.get("generation_ids", []):
                shuffle_rows.setdefault(generation_id, []).append(item)
        consumption_rows = self._trainer_consumption_rows(
            generation_id
            for generation_id, generation in generations.items()
            if not generation["rolled_back"]
        )
        for generation_id, generation in sorted(generations.items()):
            if generation["rolled_back"]:
                continue
            roots = self._candidate_roots(generation_id)
            game_files = self._iter_suffix(roots, ".sgfs")
            game_details = None
            for path in game_files:
                with path.open("rb") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if line.strip():
                            game_details = {
                                "source_path": str(path),
                                "first_nonempty_line_number": line_number,
                                "first_nonempty_line_sha256": hashlib.sha256(
                                    line.strip()
                                ).hexdigest(),
                            }
                            break
                if game_details is not None:
                    break
            candidates = [
                ("first-game", game_details),
            ]
            tdata_files = self._iter_suffix(roots, ".npz")
            if tdata_files:
                first_tdata = tdata_files[0]
                candidates.append(
                    (
                        "first-tdata",
                        {
                            "source_path": str(first_tdata),
                            "source_size": first_tdata.stat().st_size,
                            "source_sha256": file_sha256(first_tdata),
                        },
                    )
                )
            admission = data_rows.get(generation_id)
            if admission is not None and admission.get("admitted") is True:
                candidates.append(
                    (
                        "admission",
                        {
                            "data_receipt_path": admission["receipt_path"],
                            "data_receipt_sha256": admission["receipt_sha256"],
                            "admitted_roots": [
                                root["path"] for root in admission["roots"]
                            ],
                        },
                    )
                )
            generation_shuffles = sorted(
                shuffle_rows.get(generation_id, []),
                key=lambda item: item["path"],
            )
            if generation_shuffles:
                first = generation_shuffles[0]
                candidates.append(
                    (
                        "first-shuffle",
                        {
                            "shuffle_path": first["path"],
                            "shuffle_manifest_path": first["manifest_path"],
                            "shuffle_manifest_file_sha256": first[
                                "manifest_file_sha256"
                            ],
                            "shuffle_manifest_receipt_sha256": first[
                                "manifest_receipt_sha256"
                            ],
                            "mixed_generation": first["mixed_generation"],
                        },
                    )
                )
            matching_consumption = sorted(
                (
                    (path, value)
                    for path, value in consumption_rows
                    if generation_id in value.get("generation_ids", [])
                    and value.get("candidate_hashes", {}).get(generation_id)
                    == generation["candidate_hash"]
                ),
                key=lambda item: (
                    item[1].get("samples_before", 0),
                    str(item[0]),
                ),
            )
            if matching_consumption:
                path, value = matching_consumption[0]
                candidates.append(
                    (
                        "first-training-consumption",
                        {
                            "trainer_receipt_path": str(path),
                            "trainer_receipt_file_sha256": file_sha256(path),
                            "trainer_receipt_sha256": value["receipt_sha256"],
                            "samples_before": value["samples_before"],
                            "samples_after": value["samples_after"],
                            "shuffle_manifest_file_sha256": value[
                                "shuffle_manifest_file_sha256"
                            ],
                        },
                    )
                )
            for kind, details in candidates:
                if details is None:
                    continue
                evidence_path = self._publish_evidence(
                    generation, kind, details
                )
                delivered = (
                    self._deliver(generation_id, kind, evidence_path)
                    if kind
                    in {
                        "first-game",
                        "first-tdata",
                        "first-shuffle",
                        "first-training-consumption",
                    }
                    and generation.get("bootstrap") is not True
                    else False
                )
                events.append(
                    {
                        "generation_id": generation_id,
                        "kind": kind,
                        "evidence_path": str(evidence_path),
                        "delivered": delivered,
                    }
                )
        return events

    @property
    def _adaptive_root(self) -> Path:
        return self.promotion_root / "adaptive"

    @property
    def _adaptive_parents_root(self) -> Path:
        return self._adaptive_root / "parents"

    @property
    def _adaptive_producer_status_path(self) -> Path:
        return self._adaptive_root / "observation-producer-status.json"

    def _now_unix(self) -> float:
        current = self.now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise PromotionFeedbackError(
                "adaptive observation timestamp must be timezone-aware"
            )
        value = current.timestamp()
        if value < 0:
            raise PromotionFeedbackError(
                "adaptive observation timestamp must be nonnegative"
            )
        return value

    def _promotion_services_path(self) -> Path:
        return self.promotion_root.parent / "configs" / "promotion-services.json"

    def _write_adaptive_producer_status(
        self,
        *,
        state: str,
        observed_at_unix: float,
        service_spec: Optional[Mapping[str, Any]],
        observation: Optional[Mapping[str, Any]],
        blocked_reason: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        root = _absolute_directory(
            self._adaptive_root,
            "adaptive observation producer root",
            create=True,
        )
        if root != self._adaptive_root:
            raise PromotionFeedbackError(
                "adaptive observation producer root changed unexpectedly"
            )
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": ADAPTIVE_PRODUCER_STATUS_CONTRACT,
            "state": state,
            "observed_at_unix": observed_at_unix,
            "service_spec": (
                None if service_spec is None else dict(service_spec)
            ),
            "observation": (
                None if observation is None else dict(observation)
            ),
            "blocked_reason": blocked_reason,
            "error_code": error_code,
        }
        value = dict(payload)
        value["status_sha256"] = canonical_sha256(payload)
        atomic_replace_json(self._adaptive_producer_status_path, value)
        return value

    def _retained_observation_summary(
        self, observation_path: Optional[Path]
    ) -> Optional[Dict[str, Any]]:
        if observation_path is None:
            return None
        try:
            binding_before = _stable_file_binding(
                observation_path, "retained adaptive observation"
            )
            value = load_adaptive_observation(
                observation_path, expected_path=observation_path
            )
            binding_after = _stable_file_binding(
                observation_path, "retained adaptive observation"
            )
            if binding_before != binding_after:
                return None
        except (AdaptiveTrainingError, OSError, PromotionFeedbackError):
            return None
        return {
            **binding_before,
            "observation_sha256": value.get("observation_sha256"),
            "admitted_samples": value.get("admitted_samples"),
            "current_champion_model_sha256": value.get(
                "current_champion_model_sha256"
            ),
        }

    def _publish_blocked_adaptive_status(
        self,
        *,
        observed_at_unix: float,
        error: BaseException,
        service_spec: Optional[Mapping[str, Any]] = None,
        observation_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        status = self._write_adaptive_producer_status(
            state="BLOCKED",
            observed_at_unix=observed_at_unix,
            service_spec=service_spec,
            observation=self._retained_observation_summary(
                observation_path
            ),
            blocked_reason=str(error),
            error_code=str(
                getattr(error, "code", type(error).__name__)
            ),
        )
        return {
            "status": "BLOCKED",
            "status_path": str(self._adaptive_producer_status_path),
            "blocked_reason": status["blocked_reason"],
            "error_code": status["error_code"],
            "observation_retained": status["observation"] is not None,
        }

    def _load_adaptive_service_binding(
        self,
        services_path: Path,
        services: Mapping[str, Any],
    ) -> tuple[Any, Dict[str, Any]]:
        if (
            services.get("schema_version") != 3
            or services.get("contract") != AUTONOMY_SERVICE_SPEC_CONTRACT
            or services.get("full_autonomy") is not True
            or services.get("mutation_enabled") is not True
        ):
            raise PromotionFeedbackError(
                "full-autonomy v3 promotion service specification is invalid"
            )
        raw_inputs = services.get("service_inputs")
        expected_inputs = {
            "autonomy_policy",
            "executor_spec",
            "adaptive_spec",
            "suite_registry_spec",
        }
        if (
            not isinstance(raw_inputs, Mapping)
            or set(raw_inputs) != expected_inputs
        ):
            raise PromotionFeedbackError(
                "full-autonomy service_inputs are incomplete"
            )
        raw_binding = raw_inputs.get("adaptive_spec")
        if (
            not isinstance(raw_binding, Mapping)
            or set(raw_binding) != {"path", "sha256"}
        ):
            raise PromotionFeedbackError(
                "adaptive service input binding is malformed"
            )
        expected_file_hash = _require_hash(
            raw_binding.get("sha256"), "adaptive service input file hash"
        )
        raw_path = raw_binding.get("path")
        if not isinstance(raw_path, str):
            raise PromotionFeedbackError(
                "adaptive service input path is malformed"
            )
        adaptive_path = Path(raw_path)
        if (
            not adaptive_path.is_absolute()
            or adaptive_path != Path(os.path.abspath(str(adaptive_path)))
        ):
            raise PromotionFeedbackError(
                "adaptive service input path must be normalized and absolute"
            )
        services_binding = _stable_file_binding(
            services_path, "promotion services specification"
        )
        adaptive_binding = _stable_file_binding(
            adaptive_path, "adaptive service specification"
        )
        if adaptive_binding["sha256"] != expected_file_hash:
            raise PromotionFeedbackError(
                "adaptive service specification changed from service_inputs hash"
            )
        spec = load_adaptive_service_spec(adaptive_path)
        final_binding = _stable_file_binding(
            adaptive_path, "adaptive service specification"
        )
        if (
            spec.file_sha256 != expected_file_hash
            or final_binding != adaptive_binding
        ):
            raise PromotionFeedbackError(
                "adaptive service specification changed while loading"
            )
        binding_payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": ADAPTIVE_SERVICE_BINDING_CONTRACT,
            "promotion_services": services_binding,
            "adaptive_service_spec": {
                **adaptive_binding,
                "spec_sha256": spec.spec_sha256,
                "observation_path": str(spec.observation_path),
            },
        }
        binding_receipt = _finalize_receipt(binding_payload)
        _absolute_directory(
            self._adaptive_root,
            "adaptive observation producer root",
            create=True,
        )
        _absolute_directory(
            self._adaptive_parents_root,
            "adaptive observation parent root",
            create=True,
        )
        binding_path = self._adaptive_parents_root / "service-binding.json"
        if binding_path.exists() or binding_path.is_symlink():
            binding_metadata = binding_path.lstat()
            if (
                stat.S_ISLNK(binding_metadata.st_mode)
                or not stat.S_ISREG(binding_metadata.st_mode)
                or binding_metadata.st_mode & 0o222
            ):
                raise PromotionFeedbackError(
                    "adaptive service binding receipt is not immutable"
                )
            existing_binding_before = _stable_file_binding(
                binding_path, "adaptive service binding receipt"
            )
            existing = _validate_receipt(
                load_canonical_json(
                    binding_path, "adaptive service binding receipt"
                ),
                contract=ADAPTIVE_SERVICE_BINDING_CONTRACT,
                role="adaptive service binding receipt",
            )
            existing_binding_after = _stable_file_binding(
                binding_path, "adaptive service binding receipt"
            )
            if existing_binding_before != existing_binding_after:
                raise PromotionFeedbackError(
                    "adaptive service binding receipt changed while loading"
                )
            if existing != binding_receipt:
                raise PromotionFeedbackError(
                    "adaptive service specification binding changed"
                )
        else:
            atomic_create_json(binding_path, binding_receipt)
        summary = {
            "promotion_services_path": str(services_path),
            "promotion_services_sha256": services_binding["sha256"],
            "adaptive_service_spec_path": str(adaptive_path),
            "adaptive_service_spec_file_sha256": expected_file_hash,
            "adaptive_service_spec_sha256": spec.spec_sha256,
            "binding_receipt_path": str(binding_path),
            "binding_receipt_sha256": binding_receipt["receipt_sha256"],
            "observation_path": str(spec.observation_path),
        }
        return spec, summary

    def _known_adaptive_observation_path(self) -> Optional[Path]:
        candidates = []
        status_path = self._adaptive_producer_status_path
        if status_path.is_file() and not status_path.is_symlink():
            try:
                status = load_canonical_json(
                    status_path, "adaptive observation producer status"
                )
                body = dict(status)
                supplied_hash = body.pop("status_sha256", None)
                if (
                    status.get("schema_version") == SCHEMA_VERSION
                    and status.get("contract")
                    == ADAPTIVE_PRODUCER_STATUS_CONTRACT
                    and supplied_hash == canonical_sha256(body)
                ):
                    service = status.get("service_spec")
                    if isinstance(service, Mapping):
                        candidates.append(service.get("observation_path"))
            except (OSError, PromotionFeedbackError):
                pass
        binding_path = self._adaptive_parents_root / "service-binding.json"
        if binding_path.is_file() and not binding_path.is_symlink():
            try:
                binding = _validate_receipt(
                    load_canonical_json(
                        binding_path, "adaptive service binding receipt"
                    ),
                    contract=ADAPTIVE_SERVICE_BINDING_CONTRACT,
                    role="adaptive service binding receipt",
                )
                adaptive = binding.get("adaptive_service_spec")
                if isinstance(adaptive, Mapping):
                    candidates.append(adaptive.get("observation_path"))
            except (OSError, PromotionFeedbackError):
                pass
        for raw in candidates:
            if not isinstance(raw, str):
                continue
            path = Path(raw)
            if (
                path.is_absolute()
                and path == Path(os.path.abspath(str(path)))
            ):
                return path
        return None

    def _runtime_adaptive_paths(self) -> Dict[str, Path]:
        if self.runtime_config is None:
            raise PromotionFeedbackError(
                "full-autonomy adaptive observation requires a runtime config"
            )
        runtime = load_canonical_json(
            self.runtime_config, "promotion runtime config"
        )
        paths = runtime.get("paths")
        if not isinstance(paths, Mapping):
            raise PromotionFeedbackError(
                "promotion runtime config has no paths object"
            )

        def runtime_path(name: str, role: str) -> Path:
            raw = paths.get(name)
            if not isinstance(raw, str):
                raise PromotionFeedbackError(
                    f"promotion runtime config has no {name} path"
                )
            path = Path(raw)
            if (
                not path.is_absolute()
                or path != Path(os.path.abspath(str(path)))
            ):
                raise PromotionFeedbackError(
                    f"{role} path must be normalized and absolute"
                )
            for component in (path, *path.parents):
                if component.exists() and component.is_symlink():
                    raise PromotionFeedbackError(
                        f"{role} path contains a symlink: {component}"
                    )
            return path

        champion = runtime_path("champion", "champion projection")
        if champion != self.promotion_root / "champion.json":
            raise PromotionFeedbackError(
                "runtime champion path is not the canonical champion projection"
            )
        return {
            "champion": champion,
            "trainer_checkpoint": runtime_path(
                "trainerCheckpoint", "trainer checkpoint"
            ),
            "candidate_inbox": runtime_path(
                "candidateInbox", "candidate inbox"
            ),
        }

    def _candidate_inbox_depth(self, inbox: Path) -> int:
        root = _absolute_directory(Path(inbox), "candidate inbox")
        count = 0
        with os.scandir(root) as scan:
            entries = sorted(scan, key=lambda item: item.name)
        for entry in entries:
            if entry.name.startswith(".") or entry.name.endswith(
                (".tmp", ".partial", ".exported")
            ):
                continue
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise PromotionFeedbackError(
                    f"unexpected non-directory candidate inbox entry: {entry.path}"
                )
            count += 1
        return count

    def _trainer_receipt_rows(self) -> List[Dict[str, Any]]:
        directory_kinds = {
            "consumption": {"consumption"},
            "checkpoints": {"checkpoint", "longterm-checkpoint"},
            "exports": {"export"},
        }
        rows: List[Dict[str, Any]] = []
        manifest_cache: Dict[str, tuple[Dict[str, Any], Dict[str, Any]]] = {}
        for directory_name, expected_kinds in directory_kinds.items():
            root = self.trainer_root / directory_name
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                raise PromotionFeedbackError(
                    f"trainer receipt root is unsafe: {root}"
                )
            for path in sorted(root.glob("*.json")):
                metadata = path.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_mode & 0o222
                ):
                    raise PromotionFeedbackError(
                        f"trainer receipt is not immutable: {path}"
                    )
                receipt_binding_before = _stable_file_binding(
                    path, "trainer provenance receipt"
                )
                value = _validate_receipt(
                    load_canonical_json(path, "trainer provenance receipt"),
                    contract=TRAINER_RECEIPT_CONTRACT,
                    role="trainer provenance receipt",
                )
                receipt_binding_after = _stable_file_binding(
                    path, "trainer provenance receipt"
                )
                if receipt_binding_before != receipt_binding_after:
                    raise PromotionFeedbackError(
                        f"trainer receipt changed while loading: {path}"
                    )
                kind = value.get("kind")
                if kind not in expected_kinds:
                    raise PromotionFeedbackError(
                        f"trainer receipt kind is misplaced: {path}"
                    )
                sample_count = (
                    value.get("samples_after")
                    if kind == "consumption"
                    else value.get("sample_count")
                )
                if type(sample_count) is not int or sample_count < 0:
                    raise PromotionFeedbackError(
                        f"trainer receipt sample watermark is invalid: {path}"
                    )
                manifest_path = Path(
                    value.get("shuffle_manifest_path", "")
                )
                if not manifest_path.is_absolute():
                    raise PromotionFeedbackError(
                        f"trainer receipt shuffle path is invalid: {path}"
                    )
                cache_key = str(manifest_path)
                cached = manifest_cache.get(cache_key)
                if cached is None:
                    manifest_binding = _stable_file_binding(
                        manifest_path, "trainer-bound shuffle manifest"
                    )
                    manifest = load_shuffle_manifest(
                        manifest_path, verify_output=True
                    )
                    final_manifest_binding = _stable_file_binding(
                        manifest_path, "trainer-bound shuffle manifest"
                    )
                    if manifest_binding != final_manifest_binding:
                        raise PromotionFeedbackError(
                            "trainer-bound shuffle manifest changed while loading"
                        )
                    cached = (manifest, manifest_binding)
                    manifest_cache[cache_key] = cached
                manifest, manifest_binding = cached
                if (
                    value.get("shuffle_manifest_file_sha256")
                    != manifest_binding["sha256"]
                    or value.get("shuffle_manifest_receipt_sha256")
                    != manifest.get("receipt_sha256")
                    or value.get("generation_ids")
                    != manifest.get("generation_ids")
                    or value.get("candidate_hashes")
                    != manifest.get("candidate_hashes")
                ):
                    raise PromotionFeedbackError(
                        f"trainer receipt shuffle binding is invalid: {path}"
                    )
                if kind == "consumption":
                    before = value.get("samples_before")
                    selected = value.get("selected_files")
                    if (
                        type(before) is not int
                        or before < 0
                        or before >= sample_count
                        or not isinstance(selected, list)
                        or not selected
                        or value.get("selected_files_sha256")
                        != canonical_sha256(selected)
                    ):
                        raise PromotionFeedbackError(
                            f"trainer consumption receipt is invalid: {path}"
                        )
                    expected_output = {
                        item["path"]: item
                        for item in manifest["output_inventory"]
                    }
                    selected_paths = set()
                    for item in selected:
                        if not isinstance(item, dict):
                            raise PromotionFeedbackError(
                                f"trainer selected-file binding is invalid: {path}"
                            )
                        selected_path = item.get("path")
                        if (
                            not isinstance(selected_path, str)
                            or selected_path in selected_paths
                            or expected_output.get(selected_path) != item
                        ):
                            raise PromotionFeedbackError(
                                f"trainer selected-file binding changed: {path}"
                            )
                        selected_paths.add(selected_path)
                else:
                    lineage = value.get("consumption_lineage")
                    if (
                        not isinstance(lineage, list)
                        or value.get("consumption_lineage_sha256")
                        != canonical_sha256(lineage)
                    ):
                        raise PromotionFeedbackError(
                            f"trainer receipt consumption lineage is invalid: {path}"
                        )
                rows.append(
                    {
                        "path": path,
                        "file_sha256": receipt_binding_before["sha256"],
                        "value": value,
                        "kind": kind,
                        "sample_count": sample_count,
                        "manifest": manifest,
                        "manifest_binding": manifest_binding,
                    }
                )
        by_path = {str(item["path"]): item for item in rows}
        for row in rows:
            if row["kind"] == "consumption":
                continue
            value = row["value"]
            lineage = value["consumption_lineage"]
            for binding in lineage:
                if not isinstance(binding, Mapping):
                    raise PromotionFeedbackError(
                        f"trainer consumption lineage row is invalid: {row['path']}"
                    )
                consumption = by_path.get(str(binding.get("path")))
                if (
                    consumption is None
                    or consumption["kind"] != "consumption"
                    or consumption["value"].get("receipt_sha256")
                    != binding.get("receipt_sha256")
                    or consumption["value"].get("samples_before")
                    != binding.get("samples_before")
                    or consumption["value"].get("samples_after")
                    != binding.get("samples_after")
                    or consumption["value"].get(
                        "shuffle_manifest_file_sha256"
                    )
                    != binding.get("shuffle_manifest_file_sha256")
                ):
                    raise PromotionFeedbackError(
                        f"trainer consumption lineage binding changed: {row['path']}"
                    )
            receipt_path = value.get("consumption_receipt_path")
            receipt_hash = value.get("consumption_receipt_sha256")
            if (receipt_path is None) != (receipt_hash is None):
                raise PromotionFeedbackError(
                    f"trainer latest-consumption binding is incomplete: {row['path']}"
                )
            if receipt_path is not None:
                consumption = by_path.get(str(receipt_path))
                if (
                    consumption is None
                    or consumption["kind"] != "consumption"
                    or consumption["value"].get("receipt_sha256")
                    != receipt_hash
                    or consumption["sample_count"] > row["sample_count"]
                ):
                    raise PromotionFeedbackError(
                        f"trainer latest-consumption binding changed: {row['path']}"
                    )
        return rows

    def _latest_trainer_receipt(
        self, trainer_checkpoint: Path
    ) -> Dict[str, Any]:
        rows = self._trainer_receipt_rows()
        if not rows:
            raise PromotionFeedbackError(
                "adaptive observation has no immutable trainer provenance receipt"
            )
        precedence = {
            "consumption": 0,
            "export": 1,
            "longterm-checkpoint": 2,
            "checkpoint": 3,
        }
        selected = max(
            rows,
            key=lambda item: (
                item["sample_count"],
                precedence[item["kind"]],
                str(item["path"]),
            ),
        )
        if selected["manifest"].get("strict") is not True:
            raise PromotionFeedbackError(
                "latest trainer receipt is not bound to strict shuffle provenance"
            )
        admitted_samples = selected["sample_count"]
        checkpoint_binding = _stable_file_binding(
            trainer_checkpoint, "current trainer checkpoint"
        )
        checkpoints = [
            item
            for item in rows
            if item["kind"] in {"checkpoint", "longterm-checkpoint"}
            and item["sample_count"] == admitted_samples
        ]
        if checkpoints:
            identities = {
                canonical_json(item["value"].get("artifact"))
                for item in checkpoints
            }
            if len(identities) != 1:
                raise PromotionFeedbackError(
                    "trainer checkpoint receipts disagree at the admitted watermark"
                )
            artifact = checkpoints[-1]["value"].get("artifact")
            if not isinstance(artifact, Mapping):
                raise PromotionFeedbackError(
                    "trainer checkpoint receipt artifact is malformed"
                )
            if (
                artifact.get("path") != str(trainer_checkpoint)
                or artifact.get("sha256") != checkpoint_binding["sha256"]
                or artifact.get("size") != checkpoint_binding["size"]
            ):
                raise PromotionFeedbackError(
                    "current trainer checkpoint contradicts its immutable receipt"
                )
        elif selected["kind"] == "checkpoint":
            raise PromotionFeedbackError(
                "latest trainer checkpoint receipt cannot be resolved"
            )
        if selected["kind"] == "export":
            artifact = selected["value"].get("artifact")
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("inventory"), list)
                or artifact.get("inventory_sha256")
                != canonical_sha256(artifact["inventory"])
            ):
                raise PromotionFeedbackError(
                    "latest trainer export receipt artifact is malformed"
                )
            export_path = Path(artifact.get("path", ""))
            current = tree_inventory(
                export_path, known_inventory=artifact["inventory"]
            )
            if current != artifact["inventory"]:
                raise PromotionFeedbackError(
                    "latest trainer export artifact changed"
                )
        return {
            "admitted_samples": admitted_samples,
            "path": str(selected["path"]),
            "file_sha256": selected["file_sha256"],
            "receipt_sha256": selected["value"]["receipt_sha256"],
            "kind": selected["kind"],
            "shuffle_manifest_file_sha256": selected["value"][
                "shuffle_manifest_file_sha256"
            ],
            "checkpoint_sha256": checkpoint_binding["sha256"],
            "checkpoint_size": checkpoint_binding["size"],
        }

    def _latest_strict_shuffle_manifest(
        self, shuffle_watermark: Mapping[str, Any]
    ) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
        candidates = []
        rows = shuffle_watermark.get("manifests")
        if not isinstance(rows, list):
            raise PromotionFeedbackError(
                "shuffle watermark manifest inventory is malformed"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise PromotionFeedbackError(
                    "shuffle watermark manifest row is malformed"
                )
            path = Path(row.get("manifest_path", ""))
            if not path.is_absolute():
                raise PromotionFeedbackError(
                    "shuffle watermark manifest path is invalid"
                )
            binding = _stable_file_binding(
                path, "adaptive admitted-data shuffle manifest"
            )
            manifest = load_shuffle_manifest(
                path, verify_output=True, verify_sources=True
            )
            final_binding = _stable_file_binding(
                path, "adaptive admitted-data shuffle manifest"
            )
            if binding != final_binding:
                raise PromotionFeedbackError(
                    "adaptive admitted-data shuffle manifest changed while loading"
                )
            if (
                row.get("manifest_file_sha256") != binding["sha256"]
                or row.get("manifest_receipt_sha256")
                != manifest.get("receipt_sha256")
            ):
                raise PromotionFeedbackError(
                    "shuffle watermark contradicts its manifest binding"
                )
            if manifest.get("strict") is True:
                candidates.append((path, manifest, binding))
        if not candidates:
            raise PromotionFeedbackError(
                "adaptive observation has no strict shuffle provenance manifest"
            )
        selected = max(
            candidates,
            key=lambda item: (
                str(item[1].get("created_at_utc", "")),
                str(item[0]),
            ),
        )
        if (
            selected[1].get("unbound_source_count") != 0
            or not selected[1].get("source_inventory")
        ):
            raise PromotionFeedbackError(
                "latest strict shuffle has incomplete source provenance"
            )
        return selected

    def _shuffle_source_provenance(
        self, manifest: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        source_root = Path(manifest.get("source_root", ""))
        bindings: Dict[str, Dict[str, Any]] = {}
        for row in manifest.get("source_inventory", []):
            if not isinstance(row, Mapping):
                raise PromotionFeedbackError(
                    "strict shuffle source inventory row is malformed"
                )
            status = row.get("lineage_status")
            if status == "historical-baseline":
                if any(
                    row.get(field) is not None
                    for field in (
                        "generation_id",
                        "candidate_hash",
                        "data_receipt_path",
                        "data_receipt_sha256",
                    )
                ):
                    raise PromotionFeedbackError(
                        "historical shuffle source has ambiguous provenance"
                    )
                continue
            if status != "admitted":
                raise PromotionFeedbackError(
                    "strict shuffle source provenance is incomplete"
                )
            receipt_path = Path(row.get("data_receipt_path", ""))
            if not receipt_path.is_absolute():
                raise PromotionFeedbackError(
                    "shuffle source data receipt path is invalid"
                )
            receipt_metadata = receipt_path.lstat()
            if (
                stat.S_ISLNK(receipt_metadata.st_mode)
                or not stat.S_ISREG(receipt_metadata.st_mode)
                or receipt_metadata.st_mode & 0o222
            ):
                raise PromotionFeedbackError(
                    f"shuffle source data receipt is not immutable: {receipt_path}"
                )
            file_binding_before = _stable_file_binding(
                receipt_path, "shuffle source data receipt"
            )
            receipt = _validate_receipt(
                load_canonical_json(
                    receipt_path, "shuffle source data receipt"
                ),
                contract=DATA_RECEIPT_CONTRACT,
                role="shuffle source data receipt",
            )
            file_binding_after = _stable_file_binding(
                receipt_path, "shuffle source data receipt"
            )
            if file_binding_before != file_binding_after:
                raise PromotionFeedbackError(
                    "shuffle source data receipt changed while loading"
                )
            if (
                receipt.get("receipt_sha256")
                != row.get("data_receipt_sha256")
                or receipt.get("generation_id") != row.get("generation_id")
                or receipt.get("candidate_hash") != row.get("candidate_hash")
            ):
                raise PromotionFeedbackError(
                    "shuffle source data receipt binding changed"
                )
            roots = receipt.get("roots")
            if (
                not isinstance(roots, list)
                or receipt.get("source_inventory_sha256")
                != canonical_sha256(roots)
            ):
                raise PromotionFeedbackError(
                    "shuffle source data receipt inventory is malformed"
                )
            source_path = source_root / str(row.get("path", ""))
            matched = False
            for root in roots:
                if not isinstance(root, Mapping):
                    raise PromotionFeedbackError(
                        "shuffle source data receipt root is malformed"
                    )
                inventory = root.get("inventory")
                if (
                    not isinstance(inventory, list)
                    or root.get("inventory_sha256")
                    != canonical_sha256(inventory)
                ):
                    raise PromotionFeedbackError(
                        "shuffle source data receipt root hash is invalid"
                    )
                root_path = Path(root.get("path", ""))
                for item in inventory:
                    if (
                        isinstance(item, Mapping)
                        and root_path / str(item.get("path", ""))
                        == source_path
                        and item.get("sha256") == row.get("sha256")
                        and item.get("size") == row.get("size")
                    ):
                        matched = True
            if not matched:
                raise PromotionFeedbackError(
                    "shuffle source is absent from its data receipt"
                )
            binding = {
                "path": str(receipt_path),
                "file_sha256": file_binding_before["sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "generation_id": receipt["generation_id"],
                "candidate_hash": receipt["candidate_hash"],
            }
            prior = bindings.setdefault(str(receipt_path), binding)
            if prior != binding:
                raise PromotionFeedbackError(
                    "shuffle source data receipt has conflicting bindings"
                )
        return [bindings[key] for key in sorted(bindings)]

    def _admitted_data_snapshot(
        self, shuffle_watermark: Mapping[str, Any]
    ) -> tuple[Path, Dict[str, Any]]:
        manifest_path, manifest, manifest_binding = (
            self._latest_strict_shuffle_manifest(shuffle_watermark)
        )
        provenance = self._shuffle_source_provenance(manifest)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": ADAPTIVE_DATA_SNAPSHOT_CONTRACT,
            "shuffle_manifest": {
                "path": str(manifest_path),
                "file_sha256": manifest_binding["sha256"],
                "receipt_sha256": manifest["receipt_sha256"],
                "source_inventory_sha256": manifest[
                    "source_inventory_sha256"
                ],
                "output_inventory_sha256": manifest[
                    "output_inventory_sha256"
                ],
            },
            "source_root": manifest["source_root"],
            "source_inventory": manifest["source_inventory"],
            "source_inventory_sha256": manifest[
                "source_inventory_sha256"
            ],
            "generation_ids": manifest["generation_ids"],
            "candidate_hashes": manifest["candidate_hashes"],
            "provenance_receipts": provenance,
            "provenance_receipts_sha256": canonical_sha256(provenance),
        }
        value = dict(payload)
        value["snapshot_sha256"] = canonical_sha256(payload)
        snapshot_root = _absolute_directory(
            self._adaptive_parents_root / "admitted-data",
            "adaptive admitted-data snapshot root",
            create=True,
        )
        destination = (
            snapshot_root / f"{value['snapshot_sha256']}.json"
        )
        atomic_create_json(destination, value)
        destination_metadata = destination.lstat()
        if (
            stat.S_ISLNK(destination_metadata.st_mode)
            or not stat.S_ISREG(destination_metadata.st_mode)
            or destination_metadata.st_mode & 0o222
        ):
            raise PromotionFeedbackError(
                "adaptive admitted-data snapshot is not immutable"
            )
        existing = load_canonical_json(
            destination, "adaptive admitted-data snapshot"
        )
        if existing != value:
            raise PromotionFeedbackError(
                "adaptive admitted-data snapshot conflicts"
            )
        return destination, value

    def _load_prior_adaptive_observation(
        self, observation_path: Path
    ) -> Optional[Dict[str, Any]]:
        if not observation_path.exists() and not observation_path.is_symlink():
            return None
        binding_before = _stable_file_binding(
            observation_path, "prior adaptive observation"
        )
        observation = load_adaptive_observation(
            observation_path, expected_path=observation_path
        )
        binding_after = _stable_file_binding(
            observation_path, "prior adaptive observation"
        )
        if binding_before != binding_after:
            raise PromotionFeedbackError(
                "prior adaptive observation changed while loading"
            )
        return observation

    def _champion_baseline(
        self,
        *,
        champion_path: Path,
        admitted_samples: int,
        trainer_receipt: Mapping[str, Any],
        service_summary: Mapping[str, Any],
        prior_observation: Optional[Mapping[str, Any]],
    ) -> tuple[Any, Dict[str, Any], Path]:
        champion_binding_before = _stable_file_binding(
            champion_path, "canonical champion projection"
        )
        champion = load_champion(champion_path)
        champion_binding_after = _stable_file_binding(
            champion_path, "canonical champion projection"
        )
        if champion_binding_before != champion_binding_after:
            raise PromotionFeedbackError(
                "canonical champion projection changed while loading"
            )
        baseline_root = _absolute_directory(
            self._adaptive_parents_root / "baselines",
            "adaptive champion baseline root",
            create=True,
        )
        destination = (
            baseline_root / f"{champion.record_hash}.json"
        )
        if destination.exists() or destination.is_symlink():
            baseline_metadata = destination.lstat()
            if (
                stat.S_ISLNK(baseline_metadata.st_mode)
                or not stat.S_ISREG(baseline_metadata.st_mode)
                or baseline_metadata.st_mode & 0o222
            ):
                raise PromotionFeedbackError(
                    "adaptive champion baseline receipt is not immutable"
                )
            baseline_binding_before = _stable_file_binding(
                destination, "adaptive champion baseline receipt"
            )
            baseline = _validate_receipt(
                load_canonical_json(
                    destination, "adaptive champion baseline receipt"
                ),
                contract=ADAPTIVE_BASELINE_CONTRACT,
                role="adaptive champion baseline receipt",
                hash_field="baseline_sha256",
            )
            baseline_binding_after = _stable_file_binding(
                destination, "adaptive champion baseline receipt"
            )
            if baseline_binding_before != baseline_binding_after:
                raise PromotionFeedbackError(
                    "adaptive champion baseline changed while loading"
                )
            if (
                baseline.get("champion_record_hash")
                != champion.record_hash
                or baseline.get("champion_model_sha256")
                != champion.champion_hash
                or baseline.get("generation_id") != champion.generation_id
                or baseline.get("champion_projection_sha256")
                != champion_binding_before["sha256"]
            ):
                raise PromotionFeedbackError(
                    "adaptive champion baseline contradicts champion.json"
                )
            baseline_samples = baseline.get(
                "last_promotion_admitted_samples"
            )
            if (
                type(baseline_samples) is not int
                or baseline_samples < 0
                or baseline_samples > admitted_samples
            ):
                raise PromotionFeedbackError(
                    "adaptive champion baseline would rewind admitted samples"
                )
            captured = baseline.get("captured_trainer_receipt")
            if not isinstance(captured, Mapping):
                raise PromotionFeedbackError(
                    "adaptive champion baseline trainer binding is malformed"
                )
            captured_path = Path(captured.get("path", ""))
            captured_metadata = captured_path.lstat()
            if (
                stat.S_ISLNK(captured_metadata.st_mode)
                or not stat.S_ISREG(captured_metadata.st_mode)
                or captured_metadata.st_mode & 0o222
            ):
                raise PromotionFeedbackError(
                    "baseline-bound trainer receipt is not immutable"
                )
            captured_binding = _stable_file_binding(
                captured_path, "baseline-bound trainer receipt"
            )
            captured_value = _validate_receipt(
                load_canonical_json(
                    captured_path, "baseline-bound trainer receipt"
                ),
                contract=TRAINER_RECEIPT_CONTRACT,
                role="baseline-bound trainer receipt",
            )
            if (
                captured_binding["sha256"]
                != captured.get("file_sha256")
                or captured_value.get("receipt_sha256")
                != captured.get("receipt_sha256")
            ):
                raise PromotionFeedbackError(
                    "adaptive champion baseline trainer binding changed"
                )
            return champion, baseline, destination

        baseline_samples = admitted_samples
        if (
            prior_observation is not None
            and prior_observation.get("current_champion_model_sha256")
            == champion.champion_hash
        ):
            baseline_samples = prior_observation[
                "last_promotion_admitted_samples"
            ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract": ADAPTIVE_BASELINE_CONTRACT,
            "champion_projection_path": str(champion_path),
            "champion_projection_sha256": champion_binding_before["sha256"],
            "champion_record_hash": champion.record_hash,
            "champion_model_sha256": champion.champion_hash,
            "generation_id": champion.generation_id,
            "previous_champion_model_sha256": (
                champion.previous_champion_hash
            ),
            "champion_activated_at_utc": champion.activated_at_utc,
            "last_promotion_admitted_samples": baseline_samples,
            "captured_trainer_receipt": dict(trainer_receipt),
            "adaptive_service_spec_sha256": service_summary[
                "adaptive_service_spec_sha256"
            ],
            "adaptive_service_spec_file_sha256": service_summary[
                "adaptive_service_spec_file_sha256"
            ],
        }
        baseline = _finalize_receipt(
            payload, hash_field="baseline_sha256"
        )
        atomic_create_json(destination, baseline)
        return champion, baseline, destination

    def _produce_adaptive_observation(
        self,
        *,
        spec: Any,
        service_summary: Mapping[str, Any],
        shuffle_watermark: Mapping[str, Any],
        observed_at_unix: float,
    ) -> Dict[str, Any]:
        runtime_paths = self._runtime_adaptive_paths()
        prior = self._load_prior_adaptive_observation(
            spec.observation_path
        )
        trainer = self._latest_trainer_receipt(
            runtime_paths["trainer_checkpoint"]
        )
        admitted_samples = trainer["admitted_samples"]
        if (
            prior is not None
            and prior["admitted_samples"] > admitted_samples
        ):
            raise PromotionFeedbackError(
                "adaptive admitted-sample watermark would rewind"
            )
        admitted_snapshot_path, admitted_snapshot = (
            self._admitted_data_snapshot(shuffle_watermark)
        )
        champion, baseline, baseline_path = self._champion_baseline(
            champion_path=runtime_paths["champion"],
            admitted_samples=admitted_samples,
            trainer_receipt={
                "path": trainer["path"],
                "file_sha256": trainer["file_sha256"],
                "receipt_sha256": trainer["receipt_sha256"],
                "kind": trainer["kind"],
                "sample_count": admitted_samples,
            },
            service_summary=service_summary,
            prior_observation=prior,
        )
        last_promotion = baseline["last_promotion_admitted_samples"]
        if (
            prior is not None
            and prior["current_champion_model_sha256"]
            == champion.champion_hash
            and prior["last_promotion_admitted_samples"]
            != last_promotion
        ):
            raise PromotionFeedbackError(
                "adaptive champion baseline changed without promotion"
            )
        queue_depth = self._candidate_inbox_depth(
            runtime_paths["candidate_inbox"]
        )
        adaptive_status_path = spec.root / "status.json"
        adaptive_status: Optional[Mapping[str, Any]] = None
        if adaptive_status_path.is_file() and not adaptive_status_path.is_symlink():
            candidate_status = load_canonical_json(
                adaptive_status_path, "adaptive training status"
            )
            status_body = dict(candidate_status)
            status_hash = status_body.pop("status_sha256", None)
            if (
                candidate_status.get("contract")
                != "risk-score-adaptive-training-status-v1"
                or status_hash != canonical_sha256(status_body)
            ):
                raise PromotionFeedbackError(
                    "adaptive training status self-hash is invalid"
                )
            adaptive_status = candidate_status
        last_epoch_samples = (
            adaptive_status.get("last_epoch_admitted_samples")
            if adaptive_status is not None
            else None
        )
        if last_epoch_samples is not None and (
            type(last_epoch_samples) is not int
            or last_epoch_samples < 0
            or last_epoch_samples > admitted_samples
        ):
            raise PromotionFeedbackError(
                "adaptive trial sample watermark is invalid"
            )
        trigger_baseline = max(
            last_promotion,
            last_epoch_samples
            if type(last_epoch_samples) is int
            else last_promotion,
        )
        policy = load_adaptive_policy(spec.autonomy_policy_path)
        required_samples = policy["trigger"][
            "minimum_admitted_samples_without_promotion"
        ]
        maximum_queue = policy["queue"]["maximum_candidate_queue_depth"]
        active_epoch = (
            adaptive_status.get("active_epoch_id")
            if adaptive_status is not None
            else None
        )
        prior_matches_champion = bool(
            prior is not None
            and prior.get("current_champion_model_sha256")
            == champion.champion_hash
        )
        prior_already_due = bool(
            prior_matches_champion
            and prior is not None
            and prior.get("admitted_samples", 0) - trigger_baseline
            >= required_samples
            and prior.get("candidate_queue_depth", maximum_queue + 1)
            <= maximum_queue
        )
        snapshot_due = (
            active_epoch is None
            and admitted_samples - trigger_baseline >= required_samples
            and queue_depth <= maximum_queue
            and not prior_already_due
        )
        checkpoint_snapshot: Optional[Path] = None
        if prior_matches_champion and not snapshot_due:
            prior_checkpoint = prior.get("champion_checkpoint")
            if isinstance(prior_checkpoint, Mapping):
                raw_path = prior_checkpoint.get("path")
                raw_hash = prior_checkpoint.get("sha256")
                if (
                    isinstance(raw_path, str)
                    and isinstance(raw_hash, str)
                ):
                    candidate_snapshot = Path(raw_path)
                    binding = _stable_file_binding(
                        candidate_snapshot,
                        "retained adaptive checkpoint snapshot",
                    )
                    if (
                        binding["sha256"] == raw_hash
                    ):
                        checkpoint_snapshot = candidate_snapshot
        if checkpoint_snapshot is None:
            checkpoint_snapshot = _immutable_checkpoint_snapshot(
                runtime_paths["trainer_checkpoint"],
                self._adaptive_parents_root / "checkpoints",
            )
            checkpoint_snapshot_binding = _stable_file_binding(
                checkpoint_snapshot, "immutable checkpoint snapshot"
            )
            if (
                checkpoint_snapshot_binding["sha256"]
                != trainer["checkpoint_sha256"]
                or checkpoint_snapshot_binding["size"]
                != trainer["checkpoint_size"]
            ):
                raise PromotionFeedbackError(
                    "trainer checkpoint changed between provenance validation "
                    "and snapshot creation"
                )
        final_champion_binding = _stable_file_binding(
            runtime_paths["champion"], "canonical champion projection"
        )
        if (
            final_champion_binding["sha256"]
            != baseline["champion_projection_sha256"]
        ):
            raise PromotionFeedbackError(
                "canonical champion projection changed before observation publication"
            )
        for path_field, hash_field, role in (
            (
                "promotion_services_path",
                "promotion_services_sha256",
                "promotion services specification",
            ),
            (
                "adaptive_service_spec_path",
                "adaptive_service_spec_file_sha256",
                "adaptive service specification",
            ),
        ):
            final_service_binding = _stable_file_binding(
                Path(service_summary[path_field]), role
            )
            if final_service_binding["sha256"] != service_summary[hash_field]:
                raise PromotionFeedbackError(
                    f"{role} changed before observation publication"
                )
        observation = publish_adaptive_observation(
            spec.observation_path,
            admitted_samples=admitted_samples,
            last_promotion_admitted_samples=last_promotion,
            candidate_queue_depth=queue_depth,
            current_champion_model_sha256=champion.champion_hash,
            champion_checkpoint_path=checkpoint_snapshot,
            admitted_data_manifest_path=admitted_snapshot_path,
            updated_at_unix=observed_at_unix,
        )
        status = self._write_adaptive_producer_status(
            state="PUBLISHED",
            observed_at_unix=observed_at_unix,
            service_spec=service_summary,
            observation={
                "path": str(spec.observation_path),
                "observation_sha256": observation[
                    "observation_sha256"
                ],
                "admitted_samples": admitted_samples,
                "last_promotion_admitted_samples": last_promotion,
                "current_champion_model_sha256": champion.champion_hash,
                "checkpoint_snapshot_path": str(checkpoint_snapshot),
                "admitted_data_snapshot_path": str(
                    admitted_snapshot_path
                ),
                "admitted_data_snapshot_sha256": admitted_snapshot[
                    "snapshot_sha256"
                ],
                "baseline_path": str(baseline_path),
                "baseline_sha256": baseline["baseline_sha256"],
                "trainer_receipt_path": trainer["path"],
                "trainer_receipt_sha256": trainer["receipt_sha256"],
            },
        )
        return {
            "status": "PUBLISHED",
            "status_path": str(self._adaptive_producer_status_path),
            "observation_path": str(spec.observation_path),
            "observation_sha256": observation["observation_sha256"],
            "admitted_samples": admitted_samples,
            "last_promotion_admitted_samples": last_promotion,
            "candidate_queue_depth": queue_depth,
            "checkpoint_snapshot_path": str(checkpoint_snapshot),
            "admitted_data_snapshot_path": str(admitted_snapshot_path),
            "baseline_path": str(baseline_path),
            "producer_status_sha256": status["status_sha256"],
        }

    def _reconcile_adaptive_observation(
        self, shuffle_watermark: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        services_path = self._promotion_services_path()
        if not services_path.exists() and not services_path.is_symlink():
            return None
        try:
            services = load_canonical_json(
                services_path, "promotion services specification"
            )
        except (OSError, PromotionFeedbackError) as exc:
            return self._publish_blocked_adaptive_status(
                observed_at_unix=self._now_unix(),
                error=exc,
            )
        if (
            services.get("schema_version") == 2
            and services.get("contract") == LEGACY_SERVICE_SPEC_CONTRACT
        ):
            return None
        observed_at_unix = self._now_unix()
        spec = None
        service_summary = None
        try:
            spec, service_summary = self._load_adaptive_service_binding(
                services_path, services
            )
            return self._produce_adaptive_observation(
                spec=spec,
                service_summary=service_summary,
                shuffle_watermark=shuffle_watermark,
                observed_at_unix=observed_at_unix,
            )
        except (
            AdaptiveTrainingError,
            OSError,
            PromotionFeedbackError,
            PromotionStateError,
            TypeError,
            ValueError,
        ) as exc:
            return self._publish_blocked_adaptive_status(
                observed_at_unix=observed_at_unix,
                error=exc,
                service_spec=service_summary,
                observation_path=(
                    spec.observation_path
                    if spec is not None
                    else self._known_adaptive_observation_path()
                ),
            )

    def scan_once(self) -> Dict[str, Any]:
        lock_path = self.state_root / "watcher.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(lock_path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PromotionFeedbackError(
                    "promotion feedback watcher lock is not a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            generations = self._transactions()
            frozen = any(
                not generation["rolled_back"]
                and generation.get("pending") is True
                for generation in generations.values()
            )
            data = self._reconcile_data_watermark(
                generations, frozen=frozen
            )
            shuffle = self._reconcile_shuffle_watermark(
                generations, frozen=frozen
            )
            milestones = self._milestones(generations, data, shuffle)
            adaptive = self._reconcile_adaptive_observation(shuffle)
            result = {
                "status": "OK",
                "generation_count": len(generations),
                "data_watermark_path": str(self.data_watermark_path),
                "data_watermark_sha256": file_sha256(
                    self.data_watermark_path
                ),
                "shuffle_watermark_path": str(self.shuffle_watermark_path),
                "shuffle_watermark_sha256": file_sha256(
                    self.shuffle_watermark_path
                ),
                "watermarks_frozen": frozen,
                "milestones": milestones,
            }
            if adaptive is not None:
                result["adaptive_observation"] = adaptive
            return result
        finally:
            os.close(descriptor)

    def watch(self, *, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise PromotionFeedbackError("watch interval must be positive")
        while True:
            print(canonical_json(self.scan_once()), flush=True)
            self.sleeper(interval_seconds)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("once", "watch"), default="once")
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Training run root; supplies shorthand defaults for service mode",
    )
    parser.add_argument("--promotion-root", type=Path)
    parser.add_argument("--admitted-root", type=Path)
    parser.add_argument("--shuffle-root", type=Path)
    parser.add_argument("--trainer-provenance-root", type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--data-watermark", type=Path)
    parser.add_argument("--shuffle-watermark", type=Path)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--emit-only",
        action="store_true",
        help="Emit canonical evidence without invoking promotion_host",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--interval-seconds",
        "--interval",
        dest="interval_seconds",
        type=float,
        default=5.0,
    )
    args = parser.parse_args(argv)
    if not args.emit_only and args.runtime_config is None:
        parser.error("--runtime-config is required unless --emit-only is used")
    if args.runtime_config is None and any(
        value is None
        for value in (
            args.promotion_root,
            args.admitted_root,
            args.shuffle_root,
            args.trainer_provenance_root,
            args.state_root,
            args.data_watermark,
            args.shuffle_watermark,
        )
    ):
        parser.error(
            "explicit roots/watermarks are required without --runtime-config"
        )
    if args.runtime_config is not None and args.run_root is None and (
        args.shuffle_root is None or args.trainer_provenance_root is None
    ):
        parser.error(
            "--run-root supplies shuffle/trainer defaults for --runtime-config"
        )
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def _resolve_cli_paths(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    runtime_paths: Mapping[str, Any] = {}
    if args.runtime_config is not None:
        runtime = load_canonical_json(
            args.runtime_config, "promotion runtime config"
        )
        value = runtime.get("paths")
        if not isinstance(value, dict):
            raise PromotionFeedbackError(
                "promotion runtime config has no paths object"
            )
        runtime_paths = value

    def selected(
        explicit: Optional[Path],
        runtime_name: str,
        fallback: Optional[Path] = None,
    ) -> Optional[Path]:
        if explicit is not None:
            return Path(explicit)
        runtime_value = runtime_paths.get(runtime_name)
        if isinstance(runtime_value, str) and runtime_value:
            return Path(runtime_value)
        return fallback

    promotion_root = selected(args.promotion_root, "promotionRoot")
    run_root = Path(args.run_root) if args.run_root is not None else None
    values: Dict[str, Optional[Path]] = {
        "promotion_root": promotion_root,
        "admitted_root": selected(args.admitted_root, "admittedSelfplay"),
        "shuffle_root": selected(
            args.shuffle_root,
            "",
            run_root / "shuffleddata" if run_root is not None else None,
        ),
        "trainer_provenance_root": selected(
            args.trainer_provenance_root,
            "",
            (
                promotion_root / "provenance" / "trainer"
                if promotion_root is not None
                else None
            ),
        ),
        "state_root": selected(
            args.state_root,
            "",
            (
                promotion_root / "provenance" / "feedback"
                if promotion_root is not None
                else None
            ),
        ),
        "data_watermark_path": selected(
            args.data_watermark, "dataWatermark"
        ),
        "shuffle_watermark_path": selected(
            args.shuffle_watermark, "shuffleWatermark"
        ),
        "rollout_root": selected(args.rollout_root, "rolloutQuarantine"),
    }
    missing = sorted(key for key, value in values.items() if value is None)
    if missing:
        raise PromotionFeedbackError(
            "cannot resolve feedback service paths: " + ", ".join(missing)
        )
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        paths = _resolve_cli_paths(args)
        watcher = PromotionFeedbackWatcher(
            promotion_root=paths["promotion_root"],
            admitted_root=paths["admitted_root"],
            shuffle_root=paths["shuffle_root"],
            trainer_provenance_root=paths["trainer_provenance_root"],
            state_root=paths["state_root"],
            data_watermark_path=paths["data_watermark_path"],
            shuffle_watermark_path=paths["shuffle_watermark_path"],
            rollout_root=paths["rollout_root"],
            runtime_config=(
                None if args.emit_only else args.runtime_config
            ),
            strict=args.strict,
        )
        if args.mode == "watch":
            watcher.watch(interval_seconds=args.interval_seconds)
            return 0
        result = watcher.scan_once()
    except (OSError, PromotionFeedbackError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
