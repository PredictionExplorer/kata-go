"""Bind frozen extreme-score worker shards through shuffle and trainer lineage."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from risk_score.extreme_score_league import (
    _publish_immutable_json,
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_plan,
    load_worker_receipt,
)
from risk_score.promotion_feedback import (
    SHUFFLE_MANIFEST_CONTRACT,
    SHUFFLE_MANIFEST_FILENAME,
    load_shuffle_manifest,
    publish_shuffle_manifest,
    tree_inventory,
)

EXTREME_SHUFFLE_CONTRACT = "risk-score-extreme-shuffle-provenance-v1"
INPUT_CLAIM_CONTRACT = "risk-score-extreme-shuffle-input-claim-v1"


class ExtremeScoreProvenanceError(RuntimeError):
    """Extreme-score worker or shuffle lineage is incomplete or changed."""


@contextmanager
def _exclusive_lock(path: Path):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ExtremeScoreProvenanceError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    return source.resolve()


def _regular_directory(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise ExtremeScoreProvenanceError(
            f"{role} must be a regular non-symlink directory: {source}"
        )
    return source.resolve()


def _load_command(path: Path) -> tuple[list[str], dict[str, str]]:
    source = _regular_file(path, "shuffle command JSON")
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtremeScoreProvenanceError(
            f"shuffle command JSON is invalid: {exc}"
        ) from exc
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(part, str)
            or not part
            or part != part.strip()
            or any(character in part for character in ("\x00", "\n", "\r"))
            for part in value
        )
    ):
        raise ExtremeScoreProvenanceError(
            "shuffle command must be a nonempty JSON string array"
        )
    executable = Path(value[0])
    if not executable.is_absolute():
        raise ExtremeScoreProvenanceError("shuffle command executable must be absolute")
    _regular_file(executable, "shuffle command executable")
    return value, {
        "path": str(source),
        "file_sha256": file_sha256(source),
        "argv_sha256": canonical_sha256(value),
    }


def _source_record(
    path: Path,
    root: Path,
    shard: Mapping[str, Any],
    *,
    generation_id: str,
    candidate_hash: str,
    receipt_path: Path,
    receipt_file_sha256: str,
) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ExtremeScoreProvenanceError(f"worker shard is not a regular file: {path}")
    if (
        shard.get("sha256") != file_sha256(path)
        or shard.get("size_bytes") != metadata.st_size
    ):
        raise ExtremeScoreProvenanceError(f"worker shard changed: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": shard["sha256"],
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "generation_id": generation_id,
        "candidate_hash": candidate_hash,
        "data_receipt_path": str(receipt_path),
        "data_receipt_sha256": receipt_file_sha256,
        "lineage_status": "admitted",
    }


def _collect_receipt_lineage(
    plans: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    source_roots = {plan["output_root"] for _, plan in plans}
    if len(source_roots) != 1:
        raise ExtremeScoreProvenanceError(
            "all league plans in one shuffle must share an output root"
        )
    source_root = _regular_directory(Path(next(iter(source_roots))), "source root")
    policies = {canonical_json(plan["policy"]) for _, plan in plans}
    group_sizes = {plan["group_size"] for _, plan in plans}
    if len(policies) != 1 or len(group_sizes) != 1:
        raise ExtremeScoreProvenanceError(
            "one shuffle cannot mix training policies or curriculum cohort sizes"
        )
    source_inventory: list[dict[str, Any]] = []
    receipt_bindings: list[dict[str, Any]] = []
    plan_bindings: list[dict[str, Any]] = []
    candidate_hashes: dict[str, str] = {}
    seen_source_paths: set[str] = set()
    for plan_path, plan in plans:
        generation_id = plan["generation_id"]
        candidate_hash = plan["focal_model"]["model_sha256"]
        previous = candidate_hashes.setdefault(generation_id, candidate_hash)
        if previous != candidate_hash:
            raise ExtremeScoreProvenanceError(
                f"generation {generation_id} has multiple focal models"
            )
        plan_bindings.append(
            {
                "path": str(plan_path),
                "file_sha256": file_sha256(plan_path),
                "plan_sha256": plan["plan_sha256"],
            }
        )
        for worker in plan["workers"]:
            receipt = load_worker_receipt(plan, worker["worker_id"])
            outcome = receipt["process_outcome"]
            if (
                outcome["status"] != "succeeded"
                or outcome["returncode"] != 0
                or receipt["artifact_verification"]["artifacts_unchanged"] is not True
            ):
                raise ExtremeScoreProvenanceError(
                    f"worker {worker['worker_id']} did not complete successfully"
                )
            receipt_path = _regular_file(
                Path(worker["output_directory"]) / "worker-execution-receipt.json",
                f"worker {worker['worker_id']} receipt",
            )
            receipt_file_digest = file_sha256(receipt_path)
            receipt_bindings.append(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "worker_id": worker["worker_id"],
                    "path": str(receipt_path),
                    "file_sha256": receipt_file_digest,
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
            worker_output = Path(worker["output_directory"])
            for shard in receipt["output_shards"]:
                relative = Path(shard["relative_path"])
                if relative.suffix != ".npz":
                    continue
                record = _source_record(
                    (worker_output / relative).resolve(),
                    source_root,
                    shard,
                    generation_id=generation_id,
                    candidate_hash=candidate_hash,
                    receipt_path=receipt_path,
                    receipt_file_sha256=receipt_file_digest,
                )
                if record["path"] in seen_source_paths:
                    raise ExtremeScoreProvenanceError(
                        f"duplicate worker source shard: {record['path']}"
                    )
                seen_source_paths.add(record["path"])
                source_inventory.append(record)
    if not source_inventory:
        raise ExtremeScoreProvenanceError(
            "completed workers produced no NPZ training shards"
        )
    source_inventory.sort(key=lambda item: item["path"])
    receipt_bindings.sort(key=lambda item: (item["plan_sha256"], item["worker_id"]))
    plan_bindings.sort(key=lambda item: item["plan_sha256"])
    return {
        "source_root": source_root,
        "source_inventory": source_inventory,
        "receipt_bindings": receipt_bindings,
        "plan_bindings": plan_bindings,
        "candidate_hashes": dict(sorted(candidate_hashes.items())),
        "policy": plans[0][1]["policy"],
        "curriculum_state": plans[0][1]["curriculum_state"],
        "cohort_size": next(iter(group_sizes)),
    }


def _lineage_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "path",
            "sha256",
            "size",
            "generation_id",
            "candidate_hash",
            "data_receipt_path",
            "data_receipt_sha256",
            "lineage_status",
        )
    }


def _finalize_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _snapshot_receipt_sources(
    lineage: Mapping[str, Any],
    destination: Path,
) -> list[dict[str, Any]]:
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ExtremeScoreProvenanceError(
            f"shuffle input snapshot already exists without a claim: {target}"
        )
    target.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for source_record in lineage["source_inventory"]:
            relative = Path(source_record["path"])
            source = _regular_file(
                Path(lineage["source_root"]) / relative,
                "receipt-bound source shard",
            )
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_handle, output.open("xb") as target_handle:
                while True:
                    block = source_handle.read(1024 * 1024)
                    if not block:
                        break
                    target_handle.write(block)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            if file_sha256(output) != source_record["sha256"]:
                raise ExtremeScoreProvenanceError(
                    f"shuffle input snapshot copy changed: {relative}"
                )
            os.chmod(output, 0o444)
            metadata = output.lstat()
            record = dict(source_record)
            record.update(
                {
                    "size": metadata.st_size,
                    "mtime_ns": metadata.st_mtime_ns,
                    "ctime_ns": metadata.st_ctime_ns,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                }
            )
            records.append(record)
        for directory, _, _ in os.walk(target, topdown=False):
            os.chmod(directory, 0o555)
    except Exception:
        # The claim has not yet been published, so this private partial snapshot
        # is safe to remove and retry.
        import shutil

        shutil.rmtree(target, ignore_errors=True)
        raise
    records.sort(key=lambda item: item["path"])
    return records


def _load_input_claim(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "shuffle input claim")
    if source.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ExtremeScoreProvenanceError("shuffle input claim must be read-only")
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtremeScoreProvenanceError(
            f"shuffle input claim is invalid: {exc}"
        ) from exc
    supplied = value.get("claim_sha256") if isinstance(value, Mapping) else None
    payload = dict(value) if isinstance(value, Mapping) else {}
    payload.pop("claim_sha256", None)
    if (
        value.get("schema_version") != 1
        or value.get("contract") != INPUT_CLAIM_CONTRACT
        or supplied != canonical_sha256(payload)
    ):
        raise ExtremeScoreProvenanceError("shuffle input claim identity is invalid")
    return dict(value)


def build_extreme_shuffle_manifest(
    *,
    plan_paths: Sequence[Path],
    shuffle_output: Path,
    shuffle_command_json: Path,
    source_snapshot_root: Path | None = None,
    source_snapshot_inventory: Sequence[Mapping[str, Any]] | None = None,
    input_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute and bind all worker and shuffled NPZ files."""
    if not plan_paths:
        raise ExtremeScoreProvenanceError("at least one league plan is required")
    plans: list[tuple[Path, dict[str, Any]]] = []
    for index, raw_path in enumerate(plan_paths):
        path = _regular_file(raw_path, f"league plan {index}")
        plans.append((path, load_plan(path)))

    source_roots = {plan["output_root"] for _, plan in plans}
    if len(source_roots) != 1:
        raise ExtremeScoreProvenanceError(
            "all league plans in one shuffle must share an output root"
        )
    source_root = _regular_directory(Path(next(iter(source_roots))), "source root")
    policies = {canonical_json(plan["policy"]) for _, plan in plans}
    group_sizes = {plan["group_size"] for _, plan in plans}
    if len(policies) != 1 or len(group_sizes) != 1:
        raise ExtremeScoreProvenanceError(
            "one shuffle cannot mix training policies or curriculum cohort sizes"
        )

    source_inventory: list[dict[str, Any]] = []
    receipt_bindings: list[dict[str, Any]] = []
    plan_bindings: list[dict[str, Any]] = []
    candidate_hashes: dict[str, str] = {}
    seen_source_paths: set[str] = set()
    for plan_path, plan in plans:
        generation_id = plan["generation_id"]
        candidate_hash = plan["focal_model"]["model_sha256"]
        previous = candidate_hashes.setdefault(generation_id, candidate_hash)
        if previous != candidate_hash:
            raise ExtremeScoreProvenanceError(
                f"generation {generation_id} has multiple focal models"
            )
        plan_bindings.append(
            {
                "path": str(plan_path),
                "file_sha256": file_sha256(plan_path),
                "plan_sha256": plan["plan_sha256"],
            }
        )
        for worker in plan["workers"]:
            receipt = load_worker_receipt(plan, worker["worker_id"])
            outcome = receipt["process_outcome"]
            if (
                outcome["status"] != "succeeded"
                or outcome["returncode"] != 0
                or receipt["artifact_verification"]["artifacts_unchanged"] is not True
            ):
                raise ExtremeScoreProvenanceError(
                    f"worker {worker['worker_id']} did not complete successfully"
                )
            receipt_path = _regular_file(
                Path(worker["output_directory"]) / "worker-execution-receipt.json",
                f"worker {worker['worker_id']} receipt",
            )
            receipt_file_digest = file_sha256(receipt_path)
            receipt_bindings.append(
                {
                    "plan_sha256": plan["plan_sha256"],
                    "worker_id": worker["worker_id"],
                    "path": str(receipt_path),
                    "file_sha256": receipt_file_digest,
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
            worker_output = Path(worker["output_directory"])
            for shard in receipt["output_shards"]:
                relative = Path(shard["relative_path"])
                if relative.suffix != ".npz":
                    continue
                path = (worker_output / relative).resolve()
                record = _source_record(
                    path,
                    source_root,
                    shard,
                    generation_id=generation_id,
                    candidate_hash=candidate_hash,
                    receipt_path=receipt_path,
                    receipt_file_sha256=receipt_file_digest,
                )
                if record["path"] in seen_source_paths:
                    raise ExtremeScoreProvenanceError(
                        f"duplicate worker source shard: {record['path']}"
                    )
                seen_source_paths.add(record["path"])
                source_inventory.append(record)
    if not source_inventory:
        raise ExtremeScoreProvenanceError(
            "completed workers produced no NPZ training shards"
        )

    source_inventory.sort(key=lambda item: item["path"])
    receipt_bindings.sort(key=lambda item: (item["plan_sha256"], item["worker_id"]))
    plan_bindings.sort(key=lambda item: item["plan_sha256"])
    if (
        source_snapshot_root is None
        or source_snapshot_inventory is None
        or input_claim is None
    ):
        raise ExtremeScoreProvenanceError(
            "extreme shuffle manifests require a pre-execution input claim"
        )
    snapshot_inventory = [dict(item) for item in source_snapshot_inventory]
    snapshot_inventory.sort(key=lambda item: item["path"])
    if [_lineage_identity(item) for item in snapshot_inventory] != [
        _lineage_identity(item) for item in source_inventory
    ]:
        raise ExtremeScoreProvenanceError(
            "snapshot inventory differs from worker-receipt lineage"
        )
    source_root = _regular_directory(source_snapshot_root, "source snapshot root")
    source_inventory = snapshot_inventory
    output = _regular_directory(shuffle_output, "completed shuffle output")
    output_inventory = tree_inventory(
        output, exclude_names=(SHUFFLE_MANIFEST_FILENAME,)
    )
    if not any(item["path"].endswith(".npz") for item in output_inventory):
        raise ExtremeScoreProvenanceError(
            "completed shuffle output contains no NPZ files"
        )
    argv, command_binding = _load_command(shuffle_command_json)
    policy = plans[0][1]["policy"]
    curriculum_state = plans[0][1]["curriculum_state"]
    extreme_binding = {
        "schema_version": 1,
        "contract": EXTREME_SHUFFLE_CONTRACT,
        "policy": policy,
        "cohort_size": next(iter(group_sizes)),
        "curriculum_state": curriculum_state,
        "league_plans": plan_bindings,
        "worker_receipts": receipt_bindings,
        "source_inventory_sha256": canonical_sha256(source_inventory),
        "shuffle_command": command_binding,
        "input_claim": dict(input_claim),
    }
    now = (
        datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    generation_ids = sorted(candidate_hashes)
    payload = {
        "schema_version": 1,
        "contract": SHUFFLE_MANIFEST_CONTRACT,
        "created_at_utc": now,
        "output_path": str(output),
        "output_inventory": output_inventory,
        "output_inventory_sha256": canonical_sha256(output_inventory),
        "source_root": str(source_root),
        "source_inventory": source_inventory,
        "source_inventory_sha256": canonical_sha256(source_inventory),
        "generation_ids": generation_ids,
        "candidate_hashes": dict(sorted(candidate_hashes.items())),
        "mixed_generation": len(generation_ids) > 1,
        "unbound_source_count": 0,
        "historical_source_count": 0,
        "strict": True,
        "gate_fingerprint_sha256": canonical_sha256(receipt_bindings),
        "command_sha256": canonical_sha256(argv),
        "data_watermark_path": None,
        "data_watermark_file_sha256": None,
        "extreme_score": extreme_binding,
    }
    return _finalize_receipt(payload)


def publish_extreme_shuffle_manifest(
    *,
    plan_paths: Sequence[Path],
    shuffle_output: Path,
    shuffle_command_json: Path,
    source_snapshot_root: Path,
    source_snapshot_inventory: Sequence[Mapping[str, Any]],
    input_claim: Mapping[str, Any],
) -> Path:
    manifest = build_extreme_shuffle_manifest(
        plan_paths=plan_paths,
        shuffle_output=shuffle_output,
        shuffle_command_json=shuffle_command_json,
        source_snapshot_root=source_snapshot_root,
        source_snapshot_inventory=source_snapshot_inventory,
        input_claim=input_claim,
    )
    destination = publish_shuffle_manifest(shuffle_output, manifest)
    validate_extreme_shuffle_manifest(
        destination,
        expected_policy=manifest["extreme_score"]["policy"],
        expected_cohort_size=manifest["extreme_score"]["cohort_size"],
    )
    return destination


def validate_extreme_shuffle_manifest(
    path: Path,
    *,
    expected_policy: Mapping[str, Any],
    expected_cohort_size: int,
) -> dict[str, Any]:
    manifest_path = _regular_file(path, "extreme-score shuffle manifest")
    try:
        manifest = load_shuffle_manifest(
            manifest_path, verify_output=True, verify_sources=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtremeScoreProvenanceError(
            f"extreme-score shuffle manifest verification failed: {exc}"
        ) from exc
    extreme = manifest.get("extreme_score")
    expected_keys = {
        "schema_version",
        "contract",
        "policy",
        "cohort_size",
        "curriculum_state",
        "league_plans",
        "worker_receipts",
        "source_inventory_sha256",
        "shuffle_command",
        "input_claim",
    }
    if (
        not isinstance(extreme, Mapping)
        or set(extreme) != expected_keys
        or extreme.get("schema_version") != 1
        or extreme.get("contract") != EXTREME_SHUFFLE_CONTRACT
        or not isinstance(extreme.get("policy"), Mapping)
        or not isinstance(extreme.get("league_plans"), list)
        or not isinstance(extreme.get("worker_receipts"), list)
    ):
        raise ExtremeScoreProvenanceError(
            "shuffle has no valid extreme-score provenance binding"
        )
    if extreme["cohort_size"] != expected_cohort_size:
        raise ExtremeScoreProvenanceError(
            "shuffle cohort size differs from trainer curriculum"
        )
    for field in ("file_sha256", "contract", "policy_version"):
        if extreme["policy"].get(field) != expected_policy.get(field):
            raise ExtremeScoreProvenanceError(
                f"shuffle training policy differs from trainer in {field}"
            )
    if extreme["source_inventory_sha256"] != canonical_sha256(
        manifest["source_inventory"]
    ):
        raise ExtremeScoreProvenanceError(
            "extreme-score source inventory hash is invalid"
        )
    claim_binding = extreme["input_claim"]
    if not isinstance(claim_binding, Mapping) or set(claim_binding) != {
        "path",
        "file_sha256",
        "claim_sha256",
    }:
        raise ExtremeScoreProvenanceError("shuffle input claim binding is malformed")
    claim_path = _regular_file(Path(claim_binding["path"]), "shuffle input claim")
    if file_sha256(claim_path) != claim_binding["file_sha256"]:
        raise ExtremeScoreProvenanceError("shuffle input claim file changed")
    claim = _load_input_claim(claim_path)
    if (
        claim["claim_sha256"] != claim_binding["claim_sha256"]
        or claim.get("created_before_shuffle") is not True
        or Path(claim.get("snapshot_root", "")).resolve()
        != Path(manifest["source_root"]).resolve()
        or claim.get("snapshot_inventory") != manifest["source_inventory"]
        or claim.get("snapshot_inventory_sha256")
        != canonical_sha256(manifest["source_inventory"])
    ):
        raise ExtremeScoreProvenanceError(
            "shuffle input claim contradicts the bound source snapshot"
        )
    command = extreme["shuffle_command"]
    if not isinstance(command, Mapping) or set(command) != {
        "path",
        "file_sha256",
        "argv_sha256",
    }:
        raise ExtremeScoreProvenanceError("shuffle command binding is malformed")
    argv, current_command = _load_command(Path(command["path"]))
    if current_command != dict(command) or manifest[
        "command_sha256"
    ] != canonical_sha256(argv):
        raise ExtremeScoreProvenanceError("shuffle command binding changed")

    plan_by_sha: dict[str, dict[str, Any]] = {}
    plan_sources: list[tuple[Path, dict[str, Any]]] = []
    for binding in extreme["league_plans"]:
        if not isinstance(binding, Mapping):
            raise ExtremeScoreProvenanceError("league plan binding is malformed")
        plan_path = _regular_file(Path(binding.get("path", "")), "league plan")
        if file_sha256(plan_path) != binding.get("file_sha256"):
            raise ExtremeScoreProvenanceError("league plan file changed")
        plan = load_plan(plan_path)
        if plan["plan_sha256"] != binding.get("plan_sha256"):
            raise ExtremeScoreProvenanceError("league plan identity changed")
        if (
            plan["group_size"] != expected_cohort_size
            or plan["policy"] != extreme["policy"]
        ):
            raise ExtremeScoreProvenanceError(
                "league plan policy or curriculum differs from its shuffle"
            )
        plan_by_sha[plan["plan_sha256"]] = plan
        plan_sources.append((plan_path, plan))
    if not plan_by_sha:
        raise ExtremeScoreProvenanceError("shuffle binds no league plans")

    seen_workers: set[tuple[str, str]] = set()
    for binding in extreme["worker_receipts"]:
        if not isinstance(binding, Mapping):
            raise ExtremeScoreProvenanceError("worker receipt binding is malformed")
        plan_sha = binding.get("plan_sha256")
        worker_id = binding.get("worker_id")
        key = (plan_sha, worker_id)
        if key in seen_workers or plan_sha not in plan_by_sha:
            raise ExtremeScoreProvenanceError(
                "worker receipt binding is duplicate or unplanned"
            )
        seen_workers.add(key)
        receipt_path = _regular_file(Path(binding.get("path", "")), "worker receipt")
        if file_sha256(receipt_path) != binding.get("file_sha256"):
            raise ExtremeScoreProvenanceError("worker receipt file changed")
        receipt = load_worker_receipt(plan_by_sha[plan_sha], worker_id)
        if receipt["receipt_sha256"] != binding.get("receipt_sha256"):
            raise ExtremeScoreProvenanceError("worker receipt identity changed")
        if (
            receipt["process_outcome"]["status"] != "succeeded"
            or receipt["process_outcome"]["returncode"] != 0
        ):
            raise ExtremeScoreProvenanceError(f"worker {worker_id} did not succeed")
    expected_workers = {
        (plan_sha, worker["worker_id"])
        for plan_sha, plan in plan_by_sha.items()
        for worker in plan["workers"]
    }
    if seen_workers != expected_workers:
        raise ExtremeScoreProvenanceError(
            "shuffle does not bind every planned worker receipt"
        )
    lineage = _collect_receipt_lineage(plan_sources)
    if (
        lineage["plan_bindings"] != extreme["league_plans"]
        or lineage["receipt_bindings"] != extreme["worker_receipts"]
        or [_lineage_identity(item) for item in lineage["source_inventory"]]
        != [_lineage_identity(item) for item in manifest["source_inventory"]]
    ):
        raise ExtremeScoreProvenanceError(
            "shuffle source lineage differs from worker receipts"
        )
    return dict(extreme)


def run_extreme_shuffle(
    *,
    plan_paths: Sequence[Path],
    shuffle_command_json: Path,
    shuffled_root: Path,
    lock_path: Path,
    output_id: str,
    claim_root: Path,
    executor: Callable[..., Any] = lambda argv, **kwargs: subprocess.run(
        list(argv), check=False, **kwargs
    ),
) -> Path:
    """Run or recover one transactionally claimed extreme-score shuffle."""
    if (
        not isinstance(output_id, str)
        or not output_id
        or output_id != output_id.strip()
        or "/" in output_id
        or output_id in {".", ".."}
    ):
        raise ExtremeScoreProvenanceError("shuffle output_id is unsafe")
    argv, command_binding = _load_command(shuffle_command_json)
    plans = [
        (_regular_file(path, "league plan"), load_plan(path)) for path in plan_paths
    ]
    if not plans:
        raise ExtremeScoreProvenanceError("at least one league plan is required")
    if len(argv) < 2:
        raise ExtremeScoreProvenanceError("shuffle command omits its base directory")
    root = _regular_directory(shuffled_root, "shuffled data root")
    if root != Path(argv[1]).resolve() / "shuffleddata":
        raise ExtremeScoreProvenanceError(
            "shuffled_root differs from the shuffle command base directory"
        )
    claim_parent = Path(claim_root).resolve()
    claim_parent.mkdir(parents=True, exist_ok=True)
    claim_parent = _regular_directory(claim_parent, "shuffle claim root")
    output = root / output_id
    temporary_output = root / f"{output_id}.tmp"
    with _exclusive_lock(lock_path):
        lineage = _collect_receipt_lineage(plans)
        claim_identity = {
            "plan_sha256s": sorted(plan["plan_sha256"] for _, plan in plans),
            "worker_receipts_sha256": canonical_sha256(lineage["receipt_bindings"]),
            "source_inventory_sha256": canonical_sha256(
                [_lineage_identity(item) for item in lineage["source_inventory"]]
            ),
            "shuffle_command": command_binding,
            "output_id": output_id,
            "output_path": str(output),
        }
        claim_id = canonical_sha256(claim_identity)
        claim_directory = claim_parent / claim_id
        snapshot_root = claim_directory / "input"
        claim_path = claim_directory / "claim.json"
        claim_exists = claim_path.exists() or claim_path.is_symlink()
        if not claim_exists and (
            output.exists()
            or output.is_symlink()
            or temporary_output.exists()
            or temporary_output.is_symlink()
        ):
            raise ExtremeScoreProvenanceError(
                "shuffle output exists without its pre-execution claim"
            )
        if claim_exists:
            claim = _load_input_claim(claim_path)
            if claim["claim_identity"] != claim_identity:
                raise ExtremeScoreProvenanceError("existing shuffle claim conflicts")
            snapshot_inventory = claim["snapshot_inventory"]
        else:
            if claim_directory.is_symlink():
                raise ExtremeScoreProvenanceError(
                    "unpublished shuffle claim directory is a symlink"
                )
            if claim_directory.exists():
                import shutil

                shutil.rmtree(claim_directory)
            claim_directory.mkdir(parents=True, exist_ok=False)
            snapshot_inventory = _snapshot_receipt_sources(lineage, snapshot_root)
            claim = {
                "schema_version": 1,
                "contract": INPUT_CLAIM_CONTRACT,
                "claim_identity": claim_identity,
                "snapshot_root": str(snapshot_root),
                "snapshot_inventory": snapshot_inventory,
                "snapshot_inventory_sha256": canonical_sha256(snapshot_inventory),
                "created_before_shuffle": True,
            }
            claim["claim_sha256"] = canonical_sha256(claim)
            _publish_immutable_json(claim_path, claim)
        if canonical_sha256(snapshot_inventory) != claim["snapshot_inventory_sha256"]:
            raise ExtremeScoreProvenanceError(
                "shuffle input snapshot inventory changed"
            )
        claim_binding = {
            "path": str(claim_path),
            "file_sha256": file_sha256(claim_path),
            "claim_sha256": claim["claim_sha256"],
        }

        manifest_path = output / SHUFFLE_MANIFEST_FILENAME
        if manifest_path.exists() or manifest_path.is_symlink():
            validate_extreme_shuffle_manifest(
                manifest_path,
                expected_policy=lineage["policy"],
                expected_cohort_size=lineage["cohort_size"],
            )
            return manifest_path
        if output.exists() or output.is_symlink():
            _regular_directory(output, "claimed shuffle output")
            return publish_extreme_shuffle_manifest(
                plan_paths=plan_paths,
                shuffle_output=output,
                shuffle_command_json=shuffle_command_json,
                source_snapshot_root=snapshot_root,
                source_snapshot_inventory=snapshot_inventory,
                input_claim=claim_binding,
            )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise ExtremeScoreProvenanceError(
                "claimed shuffle has an incomplete temporary output; "
                "use a new generation-specific output_id"
            )

        environment = os.environ.copy()
        environment.update(
            {
                "KATAGO_SHUFFLE_GATE_BYPASS": "1",
                "KATAGO_SHUFFLE_INPUT_ROOT": str(snapshot_root),
                "KATAGO_SHUFFLE_OUTPUT_ROOT": str(root),
                "KATAGO_SHUFFLE_OUTPUT_ID": output_id,
            }
        )
        outcome = executor(tuple(argv), env=environment)
        returncode = (
            outcome if type(outcome) is int else getattr(outcome, "returncode", None)
        )
        if type(returncode) is not int or returncode != 0:
            raise ExtremeScoreProvenanceError(
                f"shuffle command failed with return code {returncode!r}"
            )
        _, after_command_binding = _load_command(shuffle_command_json)
        if after_command_binding != command_binding:
            raise ExtremeScoreProvenanceError(
                "shuffle command changed during execution"
            )
        if temporary_output.exists() or temporary_output.is_symlink():
            raise ExtremeScoreProvenanceError(
                "shuffle command returned success with an incomplete output"
            )
        _regular_directory(output, "claimed shuffle output")
        return publish_extreme_shuffle_manifest(
            plan_paths=plan_paths,
            shuffle_output=output,
            shuffle_command_json=shuffle_command_json,
            source_snapshot_root=snapshot_root,
            source_snapshot_inventory=snapshot_inventory,
            input_claim=claim_binding,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True, type=Path, action="append")
    run.add_argument("--shuffle-command-json", required=True, type=Path)
    run.add_argument("--shuffled-root", required=True, type=Path)
    run.add_argument("--lock", required=True, type=Path)
    run.add_argument("--output-id", required=True)
    run.add_argument("--claim-root", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--policy", required=True, type=Path)
    verify.add_argument("--cohort-size", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            path = run_extreme_shuffle(
                plan_paths=args.plan,
                shuffle_command_json=args.shuffle_command_json,
                shuffled_root=args.shuffled_root,
                lock_path=args.lock,
                output_id=args.output_id,
                claim_root=args.claim_root,
            )
            result = {
                "state": "PUBLISHED",
                "manifest": str(path),
                "file_sha256": file_sha256(path),
            }
        else:
            from katago.train.extreme_score_policy import (
                load_extreme_score_training_policy,
            )

            policy = load_extreme_score_training_policy(args.policy, args.cohort_size)
            value = validate_extreme_shuffle_manifest(
                args.manifest,
                expected_policy=policy,
                expected_cohort_size=args.cohort_size,
            )
            result = {
                "manifest": str(args.manifest.resolve()),
                "extreme_score": value,
            }
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
