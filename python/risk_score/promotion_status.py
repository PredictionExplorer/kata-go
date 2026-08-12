#!/usr/bin/env python3
"""Read-only status summary for the closed-loop risk-training pipeline."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_state import canonical_json_bytes


class StatusError(RuntimeError):
    """The live status tree is missing or internally inconsistent."""


BACKPRESSURE_ALLOWANCE_SECONDS = 120.0
GPU_LEASE_NON_TRAINER_STALE_SECONDS = 120.0
SELFPLAY_TRAINING_DATA_STALE_SECONDS = 300.0
TRAINER_OBSERVATION_STALE_SECONDS = 30.0
FUTURE_TIMESTAMP_TOLERANCE_SECONDS = 5.0
CONTROLLER_STATUS_FRESH_SECONDS = 90.0
SELFPLAY_FALLBACK_MODEL_LIMIT = 128
SELFPLAY_FALLBACK_FILE_LIMIT = 256
AUTONOMY_STATUS_STALE_SECONDS = 90.0

AUTONOMY_BOOTSTRAP_SPEC_CONTRACT = "risk-score-autonomy-bootstrap-spec-v1"
AUTONOMY_BOOTSTRAP_STATUS_CONTRACT = "risk-score-autonomy-bootstrap-status-v1"
AUTONOMY_BOOTSTRAP_HALT_CONTRACT = "risk-score-autonomy-safety-halt-v1"
AUTONOMY_RUNTIME_RECEIPT_CONTRACT = "risk-score-autonomy-runtime-receipt-v1"
AUTONOMY_ACTIVATION_VERIFICATION_CONTRACT = (
    "risk-score-autonomy-activation-verification-v1"
)
AUTONOMY_SERVICE_SPEC_CONTRACT = "risk-score-host-services-v3"
ACTIVATION_RECEIPT_CONTRACT = "risk-score-systemd-activation-receipt-v1"
CLUSTER_EXECUTOR_SPEC_CONTRACT = "risk-score-cluster-executor-spec-v1"
CLUSTER_EXECUTOR_STATUS_CONTRACT = "risk-score-cluster-executor-status-v1"
CLUSTER_EXECUTOR_HEARTBEAT_CONTRACT = "risk-score-cluster-heartbeat-v1"
CLUSTER_EXECUTOR_QUARANTINE_CONTRACT = "risk-score-cluster-quarantine-v1"
CLUSTER_EXECUTOR_HALT_CONTRACT = "risk-score-cluster-safety-halt-v1"
ADAPTIVE_STATUS_CONTRACT = "risk-score-adaptive-training-status-v1"
ADAPTIVE_SERVICE_SPEC_CONTRACT = "risk-score-adaptive-training-service-spec-v1"
ADAPTIVE_SERVICE_STATUS_CONTRACT = "risk-score-adaptive-training-service-status-v1"
ADAPTIVE_OBSERVATION_CONTRACT = "risk-score-adaptive-training-observation-v1"
ADAPTIVE_RECIPE_BINDING_CONTRACT = "risk-score-active-training-recipe-v1"
ADAPTIVE_ROLLBACK_CONTRACT = "risk-score-training-recipe-rollback-v1"
ADAPTIVE_HANDOFF_CONTRACT = "risk-score-adaptive-candidate-handoff-v1"
SUITE_REGISTRY_SPEC_CONTRACT = "risk-score-evaluation-suite-registry-spec-v1"
SUITE_ROTATION_STATUS_CONTRACT = "risk-score-evaluation-suite-rotation-status-v1"
SUITE_ROTATION_SERVICE_SPEC_CONTRACT = (
    "risk-score-suite-rotation-service-spec-v1"
)
SUITE_ROTATION_SERVICE_STATUS_CONTRACT = (
    "risk-score-suite-rotation-service-status-v1"
)
ACTIVE_SUITE_CONTRACT = "risk-score-active-evaluation-suite-v1"

CURATION_PIPELINE_SPEC_CONTRACT = "risk-score-curation-pipeline-spec-v1"
CURATION_PIPELINE_STATUS_CONTRACT = "risk-score-curation-pipeline-status-v1"
LEGACY_CURATION_STATUS_CONTRACTS = frozenset(
    {
        "risk-score-machine-consensus-curation-status-v1",
        "risk-score-curation-supplement-status-v1",
    }
)

GPU_EVALUATION_STAGE_DEADLINES_SECONDS = {
    "integrity": 60.0 * 60.0,
    "screen": 2.0 * 60.0 * 60.0,
    "finalist": 4.0 * 60.0 * 60.0,
    "confirmation": 6.0 * 60.0 * 60.0,
}
GPU_EVALUATION_STAGE_ALIASES = {
    "stage-0": "integrity",
    "stage-1": "screen",
    "stage-2": "finalist",
    "stage-3": "confirmation",
    "evaluating_integrity": "integrity",
    "evaluating_screen": "screen",
    "evaluating_finalist": "finalist",
    "evaluating_confirmation": "confirmation",
}

GPU_LEASE_PHASES = frozenset(
    {
        "draining_trainer",
        "trainer_drained",
        "leased",
        "evaluator_starting",
        "evaluating",
        "draining_evaluators",
        "evaluator_drained",
        "releasing",
        "release_gpu_verified",
        "restoring_trainer",
        "trainer_running",
        "safety_halt",
    }
)


def _load_canonical(path: Path, role: str) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise StatusError(f"{role} is not a regular file")
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StatusError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or data != canonical_json_bytes(value) + b"\n":
        raise StatusError(f"{role} is not canonical JSON")
    return value


def _file_observation(path: Path, now: float) -> Mapping[str, Any]:
    if not path.exists():
        return {"present": False, "path": str(path)}
    if path.is_symlink() or not path.is_file():
        raise StatusError(f"status artifact is not a regular file: {path}")
    stat_result = path.stat()
    return {
        "present": True,
        "path": str(path),
        "size": stat_result.st_size,
        "mtime_unix": stat_result.st_mtime,
        "mtime_utc": datetime.datetime.fromtimestamp(
            stat_result.st_mtime, datetime.timezone.utc
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "age_seconds": max(0.0, now - stat_result.st_mtime),
    }


def _latest_file(root: Path, pattern: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = [
        path for path in root.glob(pattern) if path.is_file() and not path.is_symlink()
    ]
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
        default=None,
    )


def _latest_named_file(root: Path, names: Sequence[str]) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = [
        path
        for name in names
        for path in root.glob(f"**/{name}")
        if path.is_file() and not path.is_symlink()
    ]
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
        default=None,
    )


def _absolute_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StatusError(f"{role} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise StatusError(f"{role} must be a non-empty absolute path")
    return path


def _validate_self_hash(value: Mapping[str, Any], field: str, role: str) -> None:
    payload = dict(value)
    supplied = payload.pop(field, None)
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise StatusError(f"{role} self-hash is invalid")


def _require_sha256_string(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StatusError(f"{role} is not a lowercase SHA-256")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalized_absolute_path(value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if path != Path(os.path.abspath(os.fspath(path))) or ".." in path.parts:
        raise StatusError(f"{role} must be lexically normalized")
    current = path
    while True:
        if current.is_symlink():
            raise StatusError(f"{role} has a symlinked path component")
        if current != path and current.exists() and not current.is_dir():
            raise StatusError(f"{role} has a non-directory path component")
        if current.parent == current:
            break
        current = current.parent
    return path


def _state_path(value: Any, root: Path, role: str) -> Path:
    path = _normalized_absolute_path(value, role)
    if path == root or not _is_within(path, root):
        raise StatusError(f"{role} is outside the run root")
    return path


def _bound_canonical_file(
    value: Any,
    role: str,
    *,
    hash_key: str = "sha256",
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", hash_key}:
        raise StatusError(f"{role} binding is malformed")
    path = _normalized_absolute_path(value.get("path"), f"{role} path")
    expected = _require_sha256_string(value.get(hash_key), f"{role} binding hash")
    loaded = _load_canonical(path, role)
    if loaded is None or file_sha256(path) != expected:
        raise StatusError(f"{role} binding changed")
    return path, loaded


def _validate_contract(
    value: Mapping[str, Any],
    *,
    contract: str,
    role: str,
    hash_field: Optional[str] = None,
) -> None:
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("contract") != contract
    ):
        raise StatusError(f"{role} contract is invalid")
    if hash_field is not None:
        _validate_self_hash(value, hash_field, role)


def _optional_self_hash(value: Mapping[str, Any], field: str, role: str) -> None:
    if field in value:
        _validate_self_hash(value, field, role)


def _binding_summary(
    path: Path, value: Mapping[str, Any], identity: str
) -> Mapping[str, str]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "identity": identity,
    }


def _observation_freshness(
    value: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    now: float,
    maximum_age_seconds: float,
    timestamp_keys: Sequence[str],
    role: str,
) -> Mapping[str, Any]:
    for key in timestamp_keys:
        if key in value:
            updated_at = _epoch_seconds(value[key], f"{role} {key}")
            source = key
            break
    else:
        updated_at = float(observation["mtime_unix"])
        source = "mtime"
    return _freshness(
        updated_at=updated_at,
        now=now,
        source=source,
        maximum_age_seconds=maximum_age_seconds,
    )


def _mapping_reason(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    if not isinstance(value, Mapping):
        return None
    reason = value.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    error = value.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return None


def _discover_curation_status(
    root: Path,
) -> tuple[Optional[Path], Optional[Mapping[str, Any]], Optional[str]]:
    spec_path = root / "configs" / "curation-pipeline.json"
    if spec_path.exists():
        if spec_path.resolve() != spec_path:
            raise StatusError("curation pipeline specification path is not normalized")
        spec = _load_canonical(spec_path, "curation pipeline specification")
        assert spec is not None
        if (
            spec.get("schema_version") != 1
            or spec.get("contract") != CURATION_PIPELINE_SPEC_CONTRACT
        ):
            raise StatusError("curation pipeline specification contract is invalid")
        _validate_self_hash(spec, "spec_sha256", "curation pipeline specification")
        if spec.get("run_root") != str(root):
            raise StatusError("curation pipeline specification run root is invalid")
        work_root = _absolute_path(
            spec.get("work_root"), "curation pipeline specification work_root"
        )
        if (
            work_root.resolve() != work_root
            or work_root == root
            or not _is_within(work_root, root)
        ):
            raise StatusError(
                "curation pipeline specification work_root is outside run root"
            )
        status_path = work_root / "status.json"
        status = _load_canonical(status_path, "curation pipeline status")
        if status is None:
            return status_path, None, "pipeline"
        if (
            status.get("schema_version") != 1
            or status.get("contract") != CURATION_PIPELINE_STATUS_CONTRACT
        ):
            raise StatusError("curation pipeline status contract is invalid")
        _validate_self_hash(status, "status_sha256", "curation pipeline status")
        binding = status.get("spec")
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(spec_path.resolve())
            or binding.get("sha256") != file_sha256(spec_path)
            or binding.get("identity") != spec.get("spec_sha256")
            or status.get("work_root") != str(work_root)
        ):
            raise StatusError("curation pipeline status contradicts its specification")
        return status_path, status, "pipeline"

    legacy_root = root / "evaluation" / "curation" / "machine-consensus-v3"
    status_path = _latest_named_file(
        legacy_root, ("status.json", "original-status.json")
    )
    if status_path is None:
        return None, None, None
    status = _load_canonical(status_path, "legacy curation status")
    assert status is not None
    if (
        status.get("schema_version") != 1
        or status.get("contract") not in LEGACY_CURATION_STATUS_CONTRACTS
    ):
        raise StatusError("legacy curation status contract is invalid")
    _validate_self_hash(status, "status_sha256", "legacy curation status")
    return status_path, status, "legacy"


def _runtime_path(runtime: Optional[Mapping[str, Any]], key: str) -> Optional[Path]:
    if runtime is None:
        return None
    paths = runtime.get("paths")
    if paths is None:
        return None
    if not isinstance(paths, Mapping):
        raise StatusError("promotion runtime paths are malformed")
    value = paths.get(key)
    if value is None:
        return None
    return _absolute_path(value, f"promotion runtime paths.{key}")


def _discover_runtime_config(
    root: Path, supervisor: Optional[Mapping[str, Any]]
) -> tuple[Path, Optional[Mapping[str, Any]]]:
    candidates = []
    if supervisor is not None and supervisor.get("runtime_config") is not None:
        candidates.append(
            _absolute_path(
                supervisor.get("runtime_config"),
                "supervisor runtime_config",
            )
        )
    candidates.extend(
        (
            root / "configs" / "promotion-runtime.json",
            root / "promotion-runtime.json",
        )
    )
    unique_candidates = list(dict.fromkeys(candidates))
    for path in unique_candidates:
        if path.exists():
            return path, _load_canonical(path, "promotion runtime config")
    return unique_candidates[0], None


def _runtime_min_free_bytes(
    runtime: Optional[Mapping[str, Any]],
) -> Optional[int]:
    if runtime is None:
        return None
    limits = runtime.get("limits")
    if not isinstance(limits, Mapping):
        raise StatusError("promotion runtime limits are malformed")
    value = limits.get("minFreeBytes")
    if type(value) is not int or value < 0:
        raise StatusError("promotion runtime limits.minFreeBytes is malformed")
    return value


def _selfplay_path(root: Path, value: Any, role: str) -> Path:
    path = _absolute_path(value, role)
    if ".." in path.parts or path.resolve() != path:
        raise StatusError(f"{role} is not a normalized non-symlink path")
    selfplay_root = root / "selfplay"
    if path != selfplay_root and not _is_within(path, selfplay_root):
        raise StatusError(f"{role} is outside the self-play tree")
    return path


def _latest_selfplay_from_summary(root: Path, summary_path: Path) -> Optional[Path]:
    summary = _load_canonical(summary_path, "self-play summary")
    if summary is None:
        return None
    candidates = []
    for raw_directory, raw_details in summary.items():
        directory = _selfplay_path(root, raw_directory, "self-play summary directory")
        if isinstance(raw_details, Mapping):
            entries = raw_details.get("filename_mtime_num_rowss")
        else:
            entries = raw_details
        if not isinstance(entries, list):
            raise StatusError("self-play summary entries are malformed")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                raise StatusError("self-play summary file entry is malformed")
            filename, raw_mtime, row_count = entry
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".npz")
            ):
                raise StatusError("self-play summary filename is malformed")
            if row_count is None:
                continue
            if type(row_count) is not int or row_count <= 0:
                raise StatusError("self-play summary row count is malformed")
            mtime = _epoch_seconds(raw_mtime, "self-play summary file mtime")
            candidates.append((mtime, directory / filename))
    if not candidates:
        return None
    reported_mtime, path = max(
        candidates, key=lambda item: (item[0], item[1].as_posix())
    )
    if path.is_symlink() or not path.is_file():
        raise StatusError("newest self-play summary file is not a regular file")
    actual_mtime = path.stat().st_mtime
    if not math.isclose(actual_mtime, reported_mtime, abs_tol=1e-6):
        raise StatusError("newest self-play summary file mtime contradicts summary")
    return path


def _latest_selfplay_from_watermark(root: Path, watermark_path: Path) -> Optional[Path]:
    watermark = _load_canonical(watermark_path, "self-play data watermark")
    if watermark is None:
        return None
    if (
        watermark.get("schema_version") != 1
        or watermark.get("contract") != "risk-score-generation-data-watermark-v1"
    ):
        raise StatusError("self-play data watermark contract is invalid")
    _validate_self_hash(watermark, "watermark_sha256", "self-play data watermark")
    candidates = []

    def add_inventory(
        raw_root: Any, inventory: Any, inventory_hash: Any, role: str
    ) -> None:
        source_root = _selfplay_path(root, raw_root, f"{role} root")
        if not isinstance(inventory, list):
            raise StatusError(f"{role} inventory is malformed")
        if inventory_hash != canonical_sha256(inventory):
            raise StatusError(f"{role} inventory hash is invalid")
        for record in inventory:
            if not isinstance(record, Mapping):
                raise StatusError(f"{role} inventory record is malformed")
            relative = Path(record.get("path", ""))
            mtime_ns = record.get("mtime_ns")
            size = record.get("size")
            if (
                relative.is_absolute()
                or relative == Path(".")
                or ".." in relative.parts
                or not relative.name.endswith(".npz")
                or type(mtime_ns) is not int
                or mtime_ns < 0
                or type(size) is not int
                or size < 0
            ):
                raise StatusError(f"{role} inventory record is malformed")
            candidates.append((mtime_ns, source_root / relative, size))

    generations = watermark.get("generations")
    if not isinstance(generations, list):
        raise StatusError("self-play data watermark generations are malformed")
    for generation in generations:
        if not isinstance(generation, Mapping):
            raise StatusError("self-play data watermark generation is malformed")
        roots = generation.get("roots")
        if not isinstance(roots, list):
            raise StatusError("self-play data watermark roots are malformed")
        for root_value in roots:
            if not isinstance(root_value, Mapping):
                raise StatusError("self-play data watermark root is malformed")
            add_inventory(
                root_value.get("path"),
                root_value.get("inventory"),
                root_value.get("inventory_sha256"),
                "generation self-play",
            )
    historical = watermark.get("historical_sources", [])
    if historical:
        add_inventory(
            watermark.get("historical_source_root"),
            historical,
            watermark.get("historical_sources_sha256"),
            "historical self-play",
        )
    if not candidates:
        return None
    mtime_ns, path, expected_size = max(
        candidates, key=lambda item: (item[0], item[1].as_posix())
    )
    if path.is_symlink() or not path.is_file():
        raise StatusError("newest watermarked self-play file is not a regular file")
    metadata = path.stat()
    if metadata.st_mtime_ns != mtime_ns or metadata.st_size != expected_size:
        raise StatusError("newest watermarked self-play file contradicts its inventory")
    return path


def _bounded_selfplay_fallback(root: Path) -> Optional[Path]:
    selfplay_root = root / "selfplay"
    if selfplay_root.is_symlink() or not selfplay_root.is_dir():
        return None
    tdata_directories = []
    for model in islice(selfplay_root.iterdir(), SELFPLAY_FALLBACK_MODEL_LIMIT):
        if model.is_symlink() or not model.is_dir():
            continue
        tdata_directories.append(model / "tdata")
        if model.name == "continuous":
            for generation in islice(model.iterdir(), SELFPLAY_FALLBACK_MODEL_LIMIT):
                if generation.is_symlink() or not generation.is_dir():
                    continue
                tdata_directories.append(generation / "tdata")
    candidates = []
    for directory in tdata_directories:
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in islice(directory.iterdir(), SELFPLAY_FALLBACK_FILE_LIMIT):
            if (
                path.name.endswith(".npz")
                and "_" not in path.name
                and path.is_file()
                and not path.is_symlink()
            ):
                candidates.append(path)
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
        default=None,
    )


def _discover_selfplay_training_data(
    root: Path, runtime: Optional[Mapping[str, Any]]
) -> tuple[Optional[Path], Optional[str]]:
    summary_path = root / "selfplay.summary.json"
    if summary_path.exists():
        candidate = _latest_selfplay_from_summary(root, summary_path)
        if candidate is not None:
            return candidate, "selfplay.summary.json"
    watermark_path = _runtime_path(runtime, "dataWatermark")
    if watermark_path is None:
        watermark_path = root / "promotion" / "watermarks" / "data.json"
    if watermark_path.exists():
        candidate = _latest_selfplay_from_watermark(root, watermark_path)
        if candidate is not None:
            return candidate, "generation-data-watermark"
    candidate = _bounded_selfplay_fallback(root)
    return candidate, "bounded-layout-fallback" if candidate is not None else None


def _discover_gpu_lease_paths(
    root: Path, runtime: Optional[Mapping[str, Any]]
) -> tuple[Optional[Path], Path]:
    gpu_config_candidates = []
    configured = _runtime_path(runtime, "gpuLeaseConfig")
    if configured is not None:
        gpu_config_candidates.append(configured)
    gpu_config_candidates.extend(
        (
            root / "configs" / "gpu-lease-runtime.json",
            root / "gpu-lease-runtime.json",
        )
    )
    for config_path in dict.fromkeys(gpu_config_candidates):
        if not config_path.exists():
            continue
        config = _load_canonical(config_path, "GPU lease runtime config")
        assert config is not None
        paths = config.get("paths")
        if not isinstance(paths, Mapping):
            raise StatusError("GPU lease runtime paths are malformed")
        return config_path, _absolute_path(
            paths.get("leaseState"), "GPU lease runtime paths.leaseState"
        )
    return None, root / "promotion" / "gpu-lease.json"


def _epoch_seconds(value: Any, role: str) -> float:
    if isinstance(value, str):
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise StatusError(f"{role} is invalid: {exc}") from exc
        if parsed.tzinfo is None:
            raise StatusError(f"{role} must include a timezone")
        result = parsed.timestamp()
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        raise StatusError(f"{role} is not a timestamp")
    if not math.isfinite(result):
        raise StatusError(f"{role} is not finite")
    return result


def _freshness(
    *,
    updated_at: float,
    now: float,
    source: str,
    maximum_age_seconds: float,
) -> Mapping[str, Any]:
    raw_age = now - updated_at
    stale = (
        raw_age > maximum_age_seconds or raw_age < -FUTURE_TIMESTAMP_TOLERANCE_SECONDS
    )
    return {
        "freshness_source": source,
        "freshness_age_seconds": max(0.0, raw_age),
        "maximum_age_seconds": maximum_age_seconds,
        "future_dated": raw_age < -FUTURE_TIMESTAMP_TOLERANCE_SECONDS,
        "stale": stale,
    }


def _backpressure_freshness(
    value: Mapping[str, Any],
    observation: Mapping[str, Any],
    now: float,
) -> Mapping[str, Any]:
    for key in ("updated_at_utc", "updated_at", "updatedAt"):
        if key in value:
            updated_at = _epoch_seconds(value[key], f"backpressure {key}")
            source = key
            break
    else:
        updated_at = float(observation["mtime_unix"])
        source = "mtime"
    return _freshness(
        updated_at=updated_at,
        now=now,
        source=source,
        maximum_age_seconds=BACKPRESSURE_ALLOWANCE_SECONDS,
    )


def _normalized_evaluation_stage(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return GPU_EVALUATION_STAGE_ALIASES.get(value, value)


def _active_evaluation_evidence(
    *,
    lease: Mapping[str, Any],
    controller: Optional[Mapping[str, Any]],
    controller_result: Mapping[str, Any],
    controller_observation: Mapping[str, Any],
    now: float,
) -> Mapping[str, Any]:
    if controller is None:
        return {"corroborated": False, "reason": "controller-status-missing"}
    try:
        controller_updated = _epoch_seconds(
            controller.get("observed_at_utc", controller_observation.get("mtime_unix")),
            "controller status observation time",
        )
    except StatusError:
        return {"corroborated": False, "reason": "controller-time-invalid"}
    controller_freshness = _freshness(
        updated_at=controller_updated,
        now=now,
        source="controller-status",
        maximum_age_seconds=CONTROLLER_STATUS_FRESH_SECONDS,
    )
    if controller_freshness["stale"]:
        return {
            "corroborated": False,
            "reason": "controller-status-stale",
            "controller_age_seconds": controller_freshness["freshness_age_seconds"],
        }
    owner = controller_result.get("leaseOwner")
    lease_owner = lease.get("ownerId")
    if (
        not isinstance(owner, str)
        or not owner
        or not isinstance(lease_owner, str)
        or owner != lease_owner
    ):
        return {
            "corroborated": False,
            "reason": "controller-lease-owner-mismatch",
            "controller_age_seconds": controller_freshness["freshness_age_seconds"],
        }
    stage = _normalized_evaluation_stage(controller_result.get("activeStage"))
    deadline = GPU_EVALUATION_STAGE_DEADLINES_SECONDS.get(stage or "")
    active = controller_result.get("activeEvaluations")
    if deadline is None or not isinstance(active, list):
        return {
            "corroborated": False,
            "reason": "controller-active-stage-missing",
            "controller_age_seconds": controller_freshness["freshness_age_seconds"],
        }
    matching = next(
        (
            item
            for item in active
            if isinstance(item, Mapping)
            and _normalized_evaluation_stage(item.get("stage")) == stage
        ),
        None,
    )
    if matching is None:
        return {
            "corroborated": False,
            "reason": "controller-active-stage-contradiction",
            "controller_age_seconds": controller_freshness["freshness_age_seconds"],
        }
    try:
        started_at = _epoch_seconds(
            matching.get("startedAtUtc"),
            "controller active evaluation startedAtUtc",
        )
    except StatusError:
        return {
            "corroborated": False,
            "reason": "controller-active-stage-time-invalid",
            "controller_age_seconds": controller_freshness["freshness_age_seconds"],
        }
    stage_age = now - started_at
    deadline_exceeded = (
        stage_age > deadline or stage_age < -FUTURE_TIMESTAMP_TOLERANCE_SECONDS
    )
    return {
        "corroborated": True,
        "controller_age_seconds": controller_freshness["freshness_age_seconds"],
        "owner_id": owner,
        "stage": stage,
        "stage_started_at_unix": started_at,
        "stage_age_seconds": max(0.0, stage_age),
        "stage_deadline_seconds": deadline,
        "deadline_exceeded": deadline_exceeded,
    }


def _gpu_lease_summary(
    value: Mapping[str, Any],
    path: Path,
    now: float,
    *,
    controller: Optional[Mapping[str, Any]],
    controller_result: Mapping[str, Any],
    controller_observation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if value.get("schemaVersion") != 1:
        raise StatusError("GPU lease state schema is invalid")
    phase = value.get("phase")
    if not isinstance(phase, str) or phase not in GPU_LEASE_PHASES:
        raise StatusError("GPU lease state phase is invalid")
    safety_halt = value.get("safetyHalt")
    if type(safety_halt) is not bool:
        raise StatusError("GPU lease state safetyHalt is invalid")
    updated_at = _epoch_seconds(value.get("updatedAt"), "GPU lease state updatedAt")
    freshness = _freshness(
        updated_at=updated_at,
        now=now,
        source="updatedAt",
        maximum_age_seconds=GPU_LEASE_NON_TRAINER_STALE_SECONDS,
    )
    non_trainer_phase = phase != "trainer_running"
    active_evaluation: Mapping[str, Any] = {}
    stale_after_seconds = GPU_LEASE_NON_TRAINER_STALE_SECONDS
    stale_non_trainer_phase = non_trainer_phase and freshness["stale"]
    if phase == "evaluating" and not safety_halt:
        active_evaluation = _active_evaluation_evidence(
            lease=value,
            controller=controller,
            controller_result=controller_result,
            controller_observation=controller_observation,
            now=now,
        )
        if active_evaluation.get("corroborated"):
            stale_after_seconds = active_evaluation["stage_deadline_seconds"]
            stale_non_trainer_phase = active_evaluation["deadline_exceeded"]
    return {
        "path": str(path),
        "lease_id": value.get("leaseId"),
        "owner_id": value.get("ownerId"),
        "phase": phase,
        "non_trainer_phase": non_trainer_phase,
        "safety_halt": safety_halt or phase == "safety_halt",
        "safety_reason": value.get("safetyReason"),
        "updated_at_unix": updated_at,
        "age_seconds": freshness["freshness_age_seconds"],
        "stale_after_seconds": stale_after_seconds,
        "stale_non_trainer_phase": stale_non_trainer_phase,
        "future_dated": freshness["future_dated"],
        "active_evaluation": active_evaluation,
    }


def _trainer_observation_summary(
    value: Mapping[str, Any], path: Path, now: float
) -> Mapping[str, Any]:
    if (
        value.get("schema_version") != 1
        or value.get("contract") != "risk-score-host-trainer-observation-v1"
        or value.get("role") != "trainer"
    ):
        raise StatusError("trainer observation contract is invalid")
    observation = value.get("observation")
    decision = value.get("decision")
    if not isinstance(observation, str) or not isinstance(decision, str):
        raise StatusError("trainer observation state is malformed")
    updated = _epoch_seconds(
        value.get("updated_at_unix"), "trainer observation updated_at_unix"
    )
    decision_since = _epoch_seconds(
        value.get("decision_since_unix"),
        "trainer observation decision_since_unix",
    )
    if decision_since > updated:
        raise StatusError("trainer observation decision begins after its update")
    freshness = _freshness(
        updated_at=updated,
        now=now,
        source="updated_at_unix",
        maximum_age_seconds=TRAINER_OBSERVATION_STALE_SECONDS,
    )
    return {
        "path": str(path),
        "observation": observation,
        "decision": decision,
        "updated_at_unix": updated,
        "age_seconds": freshness["freshness_age_seconds"],
        "stale": freshness["stale"],
        "future_dated": freshness["future_dated"],
        "decision_since_unix": decision_since,
        "decision_duration_seconds": max(0.0, now - decision_since),
        "restart_not_before_unix": value.get("restart_not_before_unix"),
        "consecutive_short_clean_exits": value.get("consecutive_short_clean_exits", 0),
    }


def _directory_count(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(
        path.is_dir()
        and not path.is_symlink()
        and not path.name.startswith(".")
        and not path.name.endswith((".tmp", ".partial", ".exported"))
        for path in root.iterdir()
    )


def _scheduler_summary(value: Mapping[str, Any]) -> Mapping[str, Any]:
    body = dict(value)
    expected_hash = body.pop("state_sha256", None)
    if (
        value.get("schema_version") != 1
        or not isinstance(expected_hash, str)
        or canonical_sha256(body) != expected_hash
    ):
        raise StatusError("scheduler state hash or schema is invalid")
    work = value.get("work")
    claims = value.get("claims")
    idle = value.get("idle")
    idle_events = value.get("idle_events")
    if (
        not isinstance(work, Mapping)
        or not isinstance(claims, Mapping)
        or not isinstance(idle, Mapping)
        or not isinstance(idle_events, list)
    ):
        raise StatusError("scheduler state is malformed")
    state_counts = {}
    kind_counts = {}
    for raw in work.values():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("item"), Mapping):
            raise StatusError("scheduler work record is malformed")
        state = raw.get("state")
        kind = raw["item"].get("kind")
        if not isinstance(state, str) or not isinstance(kind, str):
            raise StatusError("scheduler work state/kind is malformed")
        state_counts[state] = state_counts.get(state, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    events_by_id = {
        event.get("event_id"): event
        for event in idle_events
        if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
    }
    idle_reasons = {}
    for gpu, event_id in idle.items():
        event = events_by_id.get(event_id)
        if not isinstance(gpu, str) or not isinstance(event, Mapping):
            raise StatusError("scheduler idle index is malformed")
        idle_reasons[gpu] = event.get("reason")
    return {
        "revision": value.get("revision"),
        "state_sha256": expected_hash,
        "gpu_ids": value.get("gpu_ids"),
        "active_claims": len(claims),
        "claims": {
            gpu: dict(claim)
            for gpu, claim in sorted(claims.items())
            if isinstance(claim, Mapping)
        },
        "owners": {
            gpu: claim.get("owner_id")
            for gpu, claim in claims.items()
            if isinstance(claim, Mapping)
        },
        "work_by_state": dict(sorted(state_counts.items())),
        "work_by_kind": dict(sorted(kind_counts.items())),
        "idle_reasons": dict(sorted(idle_reasons.items())),
        "safety_halt": value.get("safety_halt"),
        "gpu_safety_halts": value.get("gpu_safety_halts"),
    }


def _bootstrap_observability(root: Path, now: float) -> Mapping[str, Any]:
    state_root = _state_path(
        str(root / "promotion" / "autonomy-bootstrap"),
        root,
        "autonomy bootstrap state root",
    )
    status_path = _normalized_absolute_path(
        str(state_root / "status.json"), "autonomy bootstrap status"
    )
    runtime_receipt_path = _normalized_absolute_path(
        str(state_root / "receipts" / "runtime.json"),
        "autonomy runtime receipt",
    )
    activation_verification_path = _normalized_absolute_path(
        str(state_root / "receipts" / "activation-verification.json"),
        "autonomy activation verification",
    )
    observations = {
        "autonomy_bootstrap": _file_observation(status_path, now),
        "autonomy_runtime_receipt": _file_observation(runtime_receipt_path, now),
        "autonomy_activation_verification": _file_observation(
            activation_verification_path, now
        ),
    }
    status = _load_canonical(status_path, "autonomy bootstrap status")
    spec_path: Optional[Path] = None
    spec: Optional[Mapping[str, Any]] = None
    service_candidates: list[Path] = []
    activation_receipt_path: Optional[Path] = None
    terminal_reasons: list[Mapping[str, str]] = []
    summary: dict[str, Any] = {}

    if status is not None:
        _validate_contract(
            status,
            contract=AUTONOMY_BOOTSTRAP_STATUS_CONTRACT,
            role="autonomy bootstrap status",
            hash_field="status_sha256",
        )
        spec_binding = status.get("spec")
        if not isinstance(spec_binding, Mapping) or set(spec_binding) != {
            "path",
            "file_sha256",
            "identity",
        }:
            raise StatusError("autonomy bootstrap specification binding is malformed")
        spec_path, spec = _bound_canonical_file(
            {
                "path": spec_binding["path"],
                "file_sha256": spec_binding["file_sha256"],
            },
            "autonomy bootstrap specification",
            hash_key="file_sha256",
        )
        _validate_contract(
            spec,
            contract=AUTONOMY_BOOTSTRAP_SPEC_CONTRACT,
            role="autonomy bootstrap specification",
            hash_field="spec_sha256",
        )
        if spec_binding.get("identity") != spec.get("spec_sha256"):
            raise StatusError(
                "autonomy bootstrap status specification identity changed"
            )
        configured_state_root = _state_path(
            spec.get("state_root"),
            root,
            "autonomy bootstrap specification state_root",
        )
        if configured_state_root != state_root:
            raise StatusError(
                "autonomy bootstrap specification state root is not canonical"
            )
        runtime_spec = spec.get("runtime")
        if not isinstance(runtime_spec, Mapping):
            raise StatusError("autonomy bootstrap runtime specification is malformed")
        runtime_output = _state_path(
            runtime_spec.get("output_dir"),
            root,
            "autonomy bootstrap runtime output_dir",
        )
        if not _is_within(runtime_output, state_root):
            raise StatusError(
                "autonomy bootstrap runtime output is outside bootstrap state"
            )
        service_candidates.append(runtime_output / "promotion-services.json")
        activation = spec.get("activation")
        if not isinstance(activation, Mapping):
            raise StatusError(
                "autonomy bootstrap activation specification is malformed"
            )
        activation_receipt_path = _state_path(
            activation.get("receipt"),
            root,
            "autonomy bootstrap activation receipt",
        )
        if not _is_within(activation_receipt_path, state_root):
            raise StatusError(
                "autonomy bootstrap activation receipt is outside bootstrap state"
            )

        safety_halt = status.get("safety_halt")
        halt_reason = None
        if safety_halt is not None:
            if not isinstance(safety_halt, Mapping):
                raise StatusError("autonomy bootstrap safety halt is malformed")
            _validate_contract(
                safety_halt,
                contract=AUTONOMY_BOOTSTRAP_HALT_CONTRACT,
                role="autonomy bootstrap safety halt",
                hash_field="halt_sha256",
            )
            halt_reason = _mapping_reason(safety_halt)
            if status.get("state") != "safety-halt" or halt_reason is None:
                raise StatusError("autonomy bootstrap safety halt contradicts status")
            terminal_reasons.append(
                {"source": "autonomy-bootstrap", "reason": halt_reason}
            )
        elif status.get("state") == "safety-halt":
            raise StatusError("autonomy bootstrap status has no safety halt receipt")

        completed = status.get("completed_gates")
        total = status.get("total_gates")
        if (
            status.get("state")
            not in {
                "gates",
                "waiting",
                "runtime",
                "activation",
                "active",
                "safety-halt",
            }
            or not isinstance(completed, list)
            or any(not isinstance(gate, str) or not gate for gate in completed)
            or len(completed) != len(set(completed))
            or type(total) is not int
            or total < len(completed)
        ):
            raise StatusError("autonomy bootstrap gate progress is malformed")
        summary = {
            "path": str(status_path),
            "state": status.get("state"),
            "completed_gates": list(completed),
            "completed_gate_count": len(completed),
            "total_gates": total,
            "next_gate": status.get("next_gate"),
            "waiting_gate": status.get("waiting_gate"),
            "selected_evaluator_processes": status.get("selected_evaluator_processes"),
            "runtime_ready": status.get("runtime_ready"),
            "activation_verified": status.get("activation_verified"),
            "safety_halt": safety_halt is not None,
            "safety_halt_reason": halt_reason,
            "spec": _binding_summary(spec_path, spec, str(spec.get("spec_sha256"))),
        }

    runtime_receipt = _load_canonical(runtime_receipt_path, "autonomy runtime receipt")
    runtime_receipt_summary: Mapping[str, Any] = {}
    if runtime_receipt is not None:
        _validate_contract(
            runtime_receipt,
            contract=AUTONOMY_RUNTIME_RECEIPT_CONTRACT,
            role="autonomy runtime receipt",
            hash_field="receipt_sha256",
        )
        result = runtime_receipt.get("result")
        if not isinstance(result, Mapping):
            raise StatusError("autonomy runtime receipt result is malformed")
        if "result_sha256" in runtime_receipt and runtime_receipt.get(
            "result_sha256"
        ) != canonical_sha256(result):
            raise StatusError("autonomy runtime receipt result hash is invalid")
        outputs = runtime_receipt.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, list):
                raise StatusError(
                    "autonomy runtime receipt output inventory is malformed"
                )
            for output in outputs:
                if not isinstance(output, Mapping) or set(output) != {"path", "sha256"}:
                    raise StatusError(
                        "autonomy runtime receipt output binding is malformed"
                    )
                output_path = _normalized_absolute_path(
                    output.get("path"),
                    "autonomy runtime receipt output",
                )
                if (
                    not output_path.is_file()
                    or output_path.is_symlink()
                    or file_sha256(output_path) != output.get("sha256")
                ):
                    raise StatusError("autonomy runtime receipt output binding changed")
            if runtime_receipt.get("output_set_sha256") != canonical_sha256(outputs):
                raise StatusError(
                    "autonomy runtime receipt output inventory hash is invalid"
                )
        argv = runtime_receipt.get("argv")
        if argv is not None and runtime_receipt.get("argv_sha256") != canonical_sha256(
            argv
        ):
            raise StatusError("autonomy runtime receipt argv hash is invalid")
        raw_service_path = result.get("service_spec")
        if raw_service_path is not None:
            service_candidates.append(
                _normalized_absolute_path(
                    raw_service_path, "autonomy runtime receipt service_spec"
                )
            )
        runtime_receipt_summary = {
            "path": str(runtime_receipt_path),
            "decision": runtime_receipt.get("decision"),
            "identity": runtime_receipt.get("receipt_sha256"),
            "full_autonomy": result.get("full_autonomy"),
            "mutation_enabled": result.get("mutation_enabled"),
            "service_spec": result.get("service_spec"),
            "service_spec_sha256": result.get("service_spec_sha256"),
            "promotion_runtime": result.get("promotion_runtime"),
            "promotion_runtime_sha256": result.get("promotion_runtime_sha256"),
        }

    activation_verification = _load_canonical(
        activation_verification_path,
        "autonomy activation verification",
    )
    activation_verification_summary: Mapping[str, Any] = {}
    if activation_verification is not None:
        _validate_contract(
            activation_verification,
            contract=AUTONOMY_ACTIVATION_VERIFICATION_CONTRACT,
            role="autonomy activation verification",
            hash_field="receipt_sha256",
        )
        activation_verification_summary = {
            "path": str(activation_verification_path),
            "decision": activation_verification.get("decision"),
            "identity": activation_verification.get("receipt_sha256"),
            "runtime_receipt_sha256": activation_verification.get(
                "runtime_receipt_sha256"
            ),
            "active": activation_verification.get("active"),
        }
        if runtime_receipt is not None and activation_verification.get(
            "runtime_receipt_sha256"
        ) != runtime_receipt.get("receipt_sha256"):
            raise StatusError(
                "autonomy activation verification runtime binding changed"
            )

    return {
        "summary": summary,
        "status": status,
        "spec": spec,
        "spec_path": spec_path,
        "service_candidates": service_candidates,
        "activation_receipt_path": activation_receipt_path,
        "runtime_receipt": runtime_receipt,
        "runtime_receipt_summary": runtime_receipt_summary,
        "activation_verification": activation_verification,
        "activation_verification_summary": activation_verification_summary,
        "observations": observations,
        "terminal_reasons": terminal_reasons,
        "state_root": state_root,
    }


def _discover_autonomy_service_spec(
    root: Path,
    runtime_config_path: Path,
    bootstrap: Mapping[str, Any],
) -> tuple[Optional[Path], Optional[Mapping[str, Any]]]:
    candidates = list(bootstrap["service_candidates"])
    candidates.extend(
        (
            runtime_config_path.parent / "promotion-services.json",
            root
            / "promotion"
            / "autonomy-bootstrap"
            / "runtime"
            / "promotion-services.json",
            root / "configs" / "promotion-services.json",
            root / "promotion-services.json",
        )
    )
    legacy: Optional[tuple[Path, Mapping[str, Any]]] = None
    for candidate in dict.fromkeys(candidates):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        path = _normalized_absolute_path(
            str(candidate), "promotion service specification"
        )
        value = _load_canonical(path, "promotion service specification")
        assert value is not None
        if (
            value.get("schema_version") == 3
            and value.get("contract") == AUTONOMY_SERVICE_SPEC_CONTRACT
        ):
            return path, value
        if (
            value.get("schema_version") == 2
            and value.get("contract") == "risk-score-host-services-v2"
        ):
            if legacy is None:
                legacy = (path, value)
            continue
        raise StatusError("promotion service specification contract is invalid")
    return legacy if legacy is not None else (None, None)


def _validate_autonomy_service_inputs(
    root: Path,
    service_spec: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        service_spec.get("schema_version") != 3
        or service_spec.get("contract") != AUTONOMY_SERVICE_SPEC_CONTRACT
        or service_spec.get("full_autonomy") is not True
        or service_spec.get("mutation_enabled") is not True
    ):
        raise StatusError("full-autonomy service specification is invalid")
    systemd_units = service_spec.get("systemd_units")
    if systemd_units is not None:
        if not isinstance(systemd_units, Mapping):
            raise StatusError("full-autonomy systemd unit inventory is malformed")
        for name, binding in systemd_units.items():
            if (
                not isinstance(name, str)
                or not isinstance(binding, Mapping)
                or set(binding) != {"path", "sha256"}
            ):
                raise StatusError("full-autonomy systemd unit binding is malformed")
            unit_path = _normalized_absolute_path(
                binding.get("path"), f"systemd unit {name}"
            )
            if (
                not unit_path.is_file()
                or unit_path.is_symlink()
                or file_sha256(unit_path) != binding.get("sha256")
            ):
                raise StatusError(f"full-autonomy systemd unit {name} binding changed")
    raw_inputs = service_spec.get("service_inputs")
    expected = {
        "autonomy_policy",
        "executor_spec",
        "adaptive_spec",
        "suite_registry_spec",
    }
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != expected:
        raise StatusError("full-autonomy service inputs are incomplete")
    inputs: dict[str, Any] = {}
    for name in sorted(expected):
        path, value = _bound_canonical_file(
            raw_inputs[name], f"autonomy service input {name}"
        )
        inputs[name] = {
            "path": path,
            "value": value,
            "binding": {
                "path": str(path),
                "sha256": file_sha256(path),
            },
        }

    executor = inputs["executor_spec"]["value"]
    _validate_contract(
        executor,
        contract=CLUSTER_EXECUTOR_SPEC_CONTRACT,
        role="cluster executor specification",
        hash_field="spec_sha256",
    )
    scheduler_directory = _state_path(
        executor.get("scheduler_directory"),
        root,
        "cluster executor scheduler_directory",
    )
    state_directory = _state_path(
        executor.get("state_directory"),
        root,
        "cluster executor state_directory",
    )
    if scheduler_directory == state_directory:
        raise StatusError(
            "cluster executor scheduler and state directories are identical"
        )

    suite_input = inputs["suite_registry_spec"]
    suite = suite_input["value"]
    if suite.get("contract") == SUITE_ROTATION_SERVICE_SPEC_CONTRACT:
        _validate_contract(
            suite,
            contract=SUITE_ROTATION_SERVICE_SPEC_CONTRACT,
            role="suite rotation service specification",
            hash_field="spec_sha256",
        )
        suite_scheduler = _state_path(
            suite.get("scheduler_directory"),
            root,
            "suite rotation scheduler_directory",
        )
        if suite_scheduler != scheduler_directory:
            raise StatusError(
                "suite rotation service scheduler binding changed"
            )
        registry_binding = suite.get("registry_spec")
        if not isinstance(registry_binding, Mapping):
            raise StatusError(
                "suite rotation service registry binding is malformed"
            )
        registry_path, registry_value = _bound_canonical_file(
            registry_binding,
            "evaluation suite registry specification",
        )
        inputs["suite_rotation_service_spec"] = suite_input
        inputs["suite_registry_spec"] = {
            "path": registry_path,
            "value": registry_value,
            "binding": {
                "path": str(registry_path),
                "sha256": file_sha256(registry_path),
            },
        }
        suite = registry_value
    _validate_contract(
        suite,
        contract=SUITE_REGISTRY_SPEC_CONTRACT,
        role="evaluation suite registry specification",
        hash_field="spec_sha256",
    )
    _state_path(
        suite.get("registry_root"),
        root,
        "evaluation suite registry_root",
    )

    adaptive = inputs["adaptive_spec"]["value"]
    if adaptive.get("contract") == ADAPTIVE_SERVICE_SPEC_CONTRACT:
        _validate_contract(
            adaptive,
            contract=ADAPTIVE_SERVICE_SPEC_CONTRACT,
            role="adaptive training service specification",
            hash_field="spec_sha256",
        )
        adaptive_root = _state_path(
            adaptive.get("root"),
            root,
            "adaptive training service root",
        )
        if adaptive_root != root / "promotion" / "adaptive":
            raise StatusError(
                "adaptive training service root is not the canonical promotion root"
            )
        adaptive_scheduler = _state_path(
            adaptive.get("scheduler_directory"),
            root,
            "adaptive training scheduler_directory",
        )
        policy_path = _normalized_absolute_path(
            adaptive.get("autonomy_policy_path"),
            "adaptive training autonomy_policy_path",
        )
        policy_value = inputs["autonomy_policy"]["value"]
        if (
            adaptive_scheduler != scheduler_directory
            or policy_path != inputs["autonomy_policy"]["path"]
            or adaptive.get("autonomy_policy_sha256") != canonical_sha256(policy_value)
        ):
            raise StatusError("adaptive training service input binding changed")
    else:
        _optional_self_hash(adaptive, "spec_sha256", "adaptive training specification")
    return inputs


def _autonomy_runtime_observability(
    root: Path,
    now: float,
    runtime_config_path: Path,
    runtime: Optional[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> Mapping[str, Any]:
    service_path, service_spec = _discover_autonomy_service_spec(
        root, runtime_config_path, bootstrap
    )
    observations: dict[str, Mapping[str, Any]] = {}
    if service_path is not None:
        observations["autonomy_service_spec"] = _file_observation(service_path, now)
    v3_service_spec = bool(
        service_spec is not None
        and service_spec.get("schema_version") == 3
        and service_spec.get("contract") == AUTONOMY_SERVICE_SPEC_CONTRACT
    )
    claimed_full_autonomy = bool(
        bootstrap["runtime_receipt"] is not None
        and bootstrap["runtime_receipt"]["result"].get("full_autonomy") is True
    )
    if claimed_full_autonomy and not v3_service_spec:
        raise StatusError("autonomy runtime receipt has no v3 service specification")
    if v3_service_spec and (
        service_spec.get("full_autonomy") is not True
        or service_spec.get("mutation_enabled") is not True
    ):
        raise StatusError("v3 promotion service specification is not full autonomy")
    full_autonomy = v3_service_spec
    inputs: Mapping[str, Any] = {}
    if full_autonomy:
        assert service_spec is not None
        inputs = _validate_autonomy_service_inputs(root, service_spec)
        for name, record in inputs.items():
            observations[f"autonomy_input_{name}"] = _file_observation(
                record["path"], now
            )

    runtime_receipt = bootstrap["runtime_receipt"]
    promotion_runtime_path: Optional[Path] = None
    autonomy_runtime = runtime
    if runtime_receipt is not None:
        result = runtime_receipt["result"]
        raw_path = result.get("promotion_runtime")
        raw_hash = result.get("promotion_runtime_sha256")
        if raw_path is not None:
            promotion_runtime_path = _normalized_absolute_path(
                raw_path, "autonomy promotion runtime"
            )
            loaded = _load_canonical(
                promotion_runtime_path, "autonomy promotion runtime"
            )
            if (
                loaded is None
                or not isinstance(raw_hash, str)
                or file_sha256(promotion_runtime_path) != raw_hash
            ):
                raise StatusError("autonomy promotion runtime binding changed")
            autonomy_runtime = loaded
    if promotion_runtime_path is None and service_path is not None:
        candidate = service_path.parent / "promotion-runtime.json"
        if candidate.exists() or candidate.is_symlink():
            promotion_runtime_path = _normalized_absolute_path(
                str(candidate), "autonomy promotion runtime"
            )
            autonomy_runtime = _load_canonical(
                promotion_runtime_path, "autonomy promotion runtime"
            )
    if promotion_runtime_path is None and runtime is not None:
        promotion_runtime_path = _normalized_absolute_path(
            str(runtime_config_path), "promotion runtime config"
        )
    if full_autonomy and autonomy_runtime is None:
        raise StatusError("full-autonomy promotion runtime config is missing")
    if promotion_runtime_path is not None:
        observations["autonomy_runtime_config"] = _file_observation(
            promotion_runtime_path, now
        )

    if runtime_receipt is not None and service_path is not None:
        result = runtime_receipt["result"]
        if result.get("service_spec") != str(service_path) or result.get(
            "service_spec_sha256"
        ) != file_sha256(service_path):
            raise StatusError("autonomy runtime receipt service binding changed")

    activation_path = bootstrap["activation_receipt_path"]
    if activation_path is None:
        activation_path = root / "promotion" / "autonomy-bootstrap" / "activation.json"
    activation_path = _state_path(
        str(activation_path), root, "autonomy activation receipt"
    )
    observations["autonomy_activation_receipt"] = _file_observation(
        activation_path, now
    )
    observe_activation = bool(
        full_autonomy
        or bootstrap["status"] is not None
        or bootstrap["activation_verification"] is not None
    )
    activation = (
        _load_canonical(activation_path, "systemd activation receipt")
        if observe_activation
        else None
    )
    activation_summary: Mapping[str, Any] = {}
    if activation is not None:
        _validate_contract(
            activation,
            contract=ACTIVATION_RECEIPT_CONTRACT,
            role="systemd activation receipt",
            hash_field="receipt_sha256",
        )
        active = activation.get("active")
        installed = activation.get("installed_units")
        inventory = activation.get("unit_inventory")
        if (
            not isinstance(active, Mapping)
            or not isinstance(installed, Mapping)
            or not isinstance(inventory, list)
            or any(not isinstance(name, str) for name in inventory)
            or len(inventory) != len(set(inventory))
            or set(installed) != set(inventory)
            or set(active) != set(inventory)
        ):
            raise StatusError("systemd activation receipt inventory is malformed")
        for unit, binding in installed.items():
            if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                raise StatusError(
                    f"systemd activation receipt unit {unit} is malformed"
                )
            installed_path = _normalized_absolute_path(
                binding.get("path"),
                f"installed systemd unit {unit}",
            )
            if (
                not installed_path.is_file()
                or installed_path.is_symlink()
                or file_sha256(installed_path) != binding.get("sha256")
            ):
                raise StatusError(f"installed systemd unit {unit} binding changed")
        if service_path is not None and activation.get(
            "service_spec_sha256"
        ) != file_sha256(service_path):
            raise StatusError("systemd activation receipt service binding changed")
        activation_summary = {
            "path": str(activation_path),
            "identity": activation.get("receipt_sha256"),
            "service_spec_sha256": activation.get("service_spec_sha256"),
            "target_unit": activation.get("target_unit"),
            "unit_inventory": activation.get("unit_inventory"),
            "active": dict(active),
            "all_active": bool(active)
            and all(value == "active" for value in active.values()),
            "restart_occurred": activation.get("restart_occurred"),
        }

    verification = bootstrap["activation_verification"]
    if verification is not None:
        binding = verification.get("activation_receipt")
        if (
            activation is None
            or not isinstance(binding, Mapping)
            or binding.get("path") != str(activation_path)
            or binding.get("sha256") != file_sha256(activation_path)
            or binding.get("identity") != activation.get("receipt_sha256")
            or verification.get("active") != activation.get("active")
        ):
            raise StatusError(
                "autonomy activation verification receipt binding changed"
            )

    return {
        "summary": {
            "full_autonomy": full_autonomy,
            "mutation_enabled": (
                service_spec.get("mutation_enabled")
                if service_spec is not None
                else None
            ),
            "service_spec_path": (
                str(service_path) if service_path is not None else None
            ),
            "service_spec_sha256": (
                file_sha256(service_path) if service_path is not None else None
            ),
            "promotion_runtime_path": (
                str(promotion_runtime_path)
                if promotion_runtime_path is not None
                else None
            ),
            "service_inputs": {
                name: record["binding"] for name, record in inputs.items()
            },
            "runtime_receipt": bootstrap["runtime_receipt_summary"],
            "activation_verification": bootstrap["activation_verification_summary"],
            "activation_receipt": activation_summary,
            "activation_verified": bool(
                activation_summary.get("all_active")
                and (
                    bootstrap["status"] is None
                    or bootstrap["status"].get("activation_verified") is True
                )
            ),
        },
        "full_autonomy": full_autonomy,
        "service_path": service_path,
        "service_spec": service_spec,
        "inputs": inputs,
        "promotion_runtime_path": promotion_runtime_path,
        "promotion_runtime": autonomy_runtime,
        "activation": activation,
        "observations": observations,
    }


def _cluster_executor_observability(
    root: Path,
    now: float,
    inputs: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "executor_spec" not in inputs:
        return {
            "summary": {},
            "status": None,
            "heartbeat": None,
            "observations": {},
            "disagreement": False,
            "terminal_reasons": [],
        }
    spec_record = inputs["executor_spec"]
    spec_path = spec_record["path"]
    spec = spec_record["value"]
    state_directory = _state_path(
        spec.get("state_directory"),
        root,
        "cluster executor state_directory",
    )
    scheduler_directory = _state_path(
        spec.get("scheduler_directory"),
        root,
        "cluster executor scheduler_directory",
    )
    owner_id = spec.get("owner_id")
    stale_after = spec.get("stale_after_seconds")
    gpu_ids = spec.get("gpu_ids")
    if (
        not isinstance(owner_id, str)
        or not owner_id
        or isinstance(stale_after, bool)
        or not isinstance(stale_after, (int, float))
        or not math.isfinite(float(stale_after))
        or float(stale_after) <= 0
        or not isinstance(gpu_ids, list)
        or not gpu_ids
        or any(not isinstance(item, str) or not item for item in gpu_ids)
    ):
        raise StatusError("cluster executor specification runtime fields are malformed")

    heartbeat_name = hashlib.sha256(owner_id.encode("utf-8")).hexdigest() + ".json"
    heartbeat_path = _normalized_absolute_path(
        str(state_directory / "heartbeats" / heartbeat_name),
        "cluster executor heartbeat",
    )
    status_candidates = (
        state_directory / "status.json",
        state_directory / "executor-status.json",
    )
    status_path = next(
        (path for path in status_candidates if path.exists() or path.is_symlink()),
        status_candidates[0],
    )
    status_path = _normalized_absolute_path(str(status_path), "cluster executor status")
    observations = {
        "cluster_executor_heartbeat": _file_observation(heartbeat_path, now),
        "cluster_executor_status": _file_observation(status_path, now),
    }

    heartbeat = _load_canonical(heartbeat_path, "cluster executor heartbeat")
    heartbeat_summary: Mapping[str, Any] = {}
    if heartbeat is not None:
        _validate_contract(
            heartbeat,
            contract=CLUSTER_EXECUTOR_HEARTBEAT_CONTRACT,
            role="cluster executor heartbeat",
            hash_field="state_sha256",
        )
        if heartbeat.get("owner_id") != owner_id or heartbeat.get(
            "executor_spec_sha256"
        ) != spec.get("spec_sha256"):
            raise StatusError("cluster executor heartbeat binding changed")
        heartbeat_freshness = _observation_freshness(
            heartbeat,
            observations["cluster_executor_heartbeat"],
            now=now,
            maximum_age_seconds=float(stale_after),
            timestamp_keys=("updated_at_unix",),
            role="cluster executor heartbeat",
        )
        heartbeat_summary = {
            "path": str(heartbeat_path),
            "state": heartbeat.get("state"),
            "updated_at_unix": heartbeat.get("updated_at_unix"),
            "age_seconds": heartbeat_freshness["freshness_age_seconds"],
            "maximum_age_seconds": heartbeat_freshness["maximum_age_seconds"],
            "future_dated": heartbeat_freshness["future_dated"],
            "stale": heartbeat_freshness["stale"],
        }

    status = _load_canonical(status_path, "cluster executor status")
    claims: list[Mapping[str, Any]] = []
    quarantines: list[Mapping[str, Any]] = []
    status_quarantine_ids: list[Any] = []
    status_freshness: Mapping[str, Any] = {}
    status_owners: dict[str, Any] = {}
    if status is not None:
        _validate_contract(
            status,
            contract=CLUSTER_EXECUTOR_STATUS_CONTRACT,
            role="cluster executor status",
        )
        _optional_self_hash(status, "status_sha256", "cluster executor status")
        if status.get("owner_id") != owner_id or status.get(
            "executor_spec_sha256"
        ) != spec.get("spec_sha256"):
            raise StatusError("cluster executor status binding changed")
        raw_gpus = status.get("gpus")
        raw_quarantines = status.get("quarantines")
        if not isinstance(raw_gpus, list) or not isinstance(raw_quarantines, list):
            raise StatusError("cluster executor status inventory is malformed")
        observed_gpu_ids: list[str] = []
        for row in raw_gpus:
            if not isinstance(row, Mapping):
                raise StatusError("cluster executor GPU status is malformed")
            gpu_id = row.get("gpu_id")
            if not isinstance(gpu_id, str) or gpu_id not in gpu_ids:
                raise StatusError("cluster executor GPU status is outside inventory")
            observed_gpu_ids.append(gpu_id)
            claim = row.get("claim")
            if claim is not None:
                if not isinstance(claim, Mapping):
                    raise StatusError("cluster executor claim status is malformed")
                claims.append(dict(claim))
                status_owners[gpu_id] = claim.get("owner_id")
        if sorted(observed_gpu_ids) != sorted(gpu_ids) or len(observed_gpu_ids) != len(
            set(observed_gpu_ids)
        ):
            raise StatusError("cluster executor GPU status inventory is incomplete")
        for quarantine in raw_quarantines:
            if not isinstance(quarantine, Mapping):
                raise StatusError("cluster executor quarantine receipt is malformed")
            _validate_contract(
                quarantine,
                contract=CLUSTER_EXECUTOR_QUARANTINE_CONTRACT,
                role="cluster executor quarantine receipt",
                hash_field="receipt_sha256",
            )
            status_quarantine_ids.append(quarantine.get("receipt_sha256"))
            quarantines.append(
                {
                    "work_id": quarantine.get("work_id"),
                    "claim_id": quarantine.get("claim_id"),
                    "failure_count": quarantine.get("failure_count"),
                    "retry_budget": quarantine.get("retry_budget"),
                    "reason": quarantine.get("reason"),
                    "quarantined_at_unix": quarantine.get("quarantined_at_unix"),
                }
            )
        status_freshness = _observation_freshness(
            status,
            observations["cluster_executor_status"],
            now=now,
            maximum_age_seconds=float(stale_after),
            timestamp_keys=("observed_at_unix", "updated_at_unix"),
            role="cluster executor status",
        )
    elif scheduler:
        scheduler_claims = scheduler.get("claims")
        if isinstance(scheduler_claims, Mapping):
            claims = [
                dict(claim)
                for _, claim in sorted(scheduler_claims.items())
                if isinstance(claim, Mapping)
            ]
        status_owners = dict(scheduler.get("owners", {}))

    quarantine_root = state_directory / "quarantine"
    scanned_quarantines: list[Mapping[str, Any]] = []
    scanned_quarantine_ids: list[Any] = []
    if quarantine_root.exists() or quarantine_root.is_symlink():
        if quarantine_root.is_symlink() or not quarantine_root.is_dir():
            raise StatusError("cluster executor quarantine directory is unsafe")
        for quarantine_path in sorted(quarantine_root.glob("*.json")):
            quarantine = _load_canonical(
                quarantine_path, "cluster executor quarantine receipt"
            )
            assert quarantine is not None
            _validate_contract(
                quarantine,
                contract=CLUSTER_EXECUTOR_QUARANTINE_CONTRACT,
                role="cluster executor quarantine receipt",
                hash_field="receipt_sha256",
            )
            scanned_quarantine_ids.append(quarantine.get("receipt_sha256"))
            scanned_quarantines.append(
                {
                    "path": str(quarantine_path),
                    "work_id": quarantine.get("work_id"),
                    "claim_id": quarantine.get("claim_id"),
                    "failure_count": quarantine.get("failure_count"),
                    "retry_budget": quarantine.get("retry_budget"),
                    "reason": quarantine.get("reason"),
                    "quarantined_at_unix": quarantine.get("quarantined_at_unix"),
                }
            )
    if status is not None and sorted(status_quarantine_ids) != sorted(
        scanned_quarantine_ids
    ):
        raise StatusError("cluster executor status quarantine inventory changed")
    quarantines = scanned_quarantines

    halt_receipts: list[Mapping[str, Any]] = []
    terminal_reasons: list[Mapping[str, str]] = []
    halt_root = state_directory / "halts"
    if halt_root.exists() or halt_root.is_symlink():
        if halt_root.is_symlink() or not halt_root.is_dir():
            raise StatusError("cluster executor halt directory is unsafe")
        for halt_path in sorted(halt_root.glob("*.json")):
            halt = _load_canonical(halt_path, "cluster executor safety halt")
            assert halt is not None
            _validate_contract(
                halt,
                contract=CLUSTER_EXECUTOR_HALT_CONTRACT,
                role="cluster executor safety halt",
                hash_field="state_sha256",
            )
            gpu_id = halt.get("gpu_id")
            reason = _mapping_reason(halt)
            expected_name = (
                hashlib.sha256(str(gpu_id).encode("utf-8")).hexdigest() + ".json"
            )
            if (
                not isinstance(gpu_id, str)
                or gpu_id not in gpu_ids
                or halt_path.name != expected_name
                or reason is None
            ):
                raise StatusError("cluster executor safety halt is malformed")
            halt_receipts.append(
                {
                    "path": str(halt_path),
                    "gpu_id": gpu_id,
                    "claim_id": halt.get("claim_id"),
                    "work_id": halt.get("work_id"),
                    "reason": reason,
                    "halted_at_unix": halt.get("halted_at_unix"),
                }
            )
            terminal_reasons.append(
                {"source": f"cluster-executor:{gpu_id}", "reason": reason}
            )

    if status is not None:
        global_reason = _mapping_reason(status.get("safety_halt"))
        if global_reason is not None:
            terminal_reasons.append(
                {"source": "cluster-executor", "reason": global_reason}
            )
        gpu_halts = status.get("gpu_safety_halts")
        if gpu_halts is not None and not isinstance(gpu_halts, Mapping):
            raise StatusError("cluster executor GPU safety halts are malformed")
        for gpu_id, reason_value in sorted((gpu_halts or {}).items()):
            reason = _mapping_reason(reason_value)
            if not isinstance(gpu_id, str) or gpu_id not in gpu_ids or reason is None:
                raise StatusError("cluster executor GPU safety halt is malformed")
            terminal_reasons.append(
                {
                    "source": f"cluster-executor:{gpu_id}",
                    "reason": reason,
                }
            )

    deduplicated_terminal = [
        dict(item)
        for item in {
            (item["source"], item["reason"]): item for item in terminal_reasons
        }.values()
    ]
    deduplicated_terminal.sort(key=lambda item: (item["source"], item["reason"]))

    expected_scheduler_directory = root / "promotion" / "scheduler"
    disagreement = scheduler_directory != expected_scheduler_directory
    if status is not None:
        if not scheduler:
            disagreement = True
        else:
            disagreement = disagreement or any(
                (
                    status.get("scheduler_revision") != scheduler.get("revision"),
                    status.get("scheduler_state_sha256")
                    != scheduler.get("state_sha256"),
                    len(claims) != scheduler.get("active_claims"),
                    status_owners != scheduler.get("owners"),
                )
            )

    summary = {
        "spec": _binding_summary(spec_path, spec, str(spec.get("spec_sha256"))),
        "state_directory": str(state_directory),
        "scheduler_directory": str(scheduler_directory),
        "owner_id": owner_id,
        "heartbeat": heartbeat_summary,
        "status_path": str(status_path),
        "status_source": ("published" if status is not None else "durable-state"),
        "status_age_seconds": status_freshness.get("freshness_age_seconds"),
        "status_stale": status_freshness.get("stale"),
        "scheduler_revision": (
            status.get("scheduler_revision")
            if status is not None
            else scheduler.get("revision")
        ),
        "scheduler_state_sha256": (
            status.get("scheduler_state_sha256")
            if status is not None
            else scheduler.get("state_sha256")
        ),
        "claims": claims,
        "active_claims": len(claims),
        "quarantines": quarantines,
        "quarantine_count": len(quarantines),
        "safety_halt": (
            status.get("safety_halt")
            if status is not None
            else scheduler.get("safety_halt")
        ),
        "gpu_safety_halts": (
            status.get("gpu_safety_halts")
            if status is not None
            else scheduler.get("gpu_safety_halts", {})
        ),
        "halt_receipts": halt_receipts,
        "scheduler_disagreement": disagreement,
    }
    return {
        "summary": summary,
        "status": status,
        "heartbeat": heartbeat,
        "observations": observations,
        "disagreement": disagreement,
        "terminal_reasons": deduplicated_terminal,
    }


def _adaptive_budget_summary(
    status: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]],
    now: float,
) -> Mapping[str, Any]:
    if policy is None:
        return {}
    budget = policy.get("gpu_budget")
    halving = policy.get("successive_halving")
    if not isinstance(budget, Mapping) or not isinstance(halving, Mapping):
        raise StatusError("adaptive policy budget is malformed")
    window = budget.get("rolling_window_seconds")
    host_gpus = budget.get("host_gpu_count")
    fraction = budget.get("maximum_fraction")
    rounds = halving.get("round_gpu_seconds")
    if (
        type(window) is not int
        or window <= 0
        or type(host_gpus) is not int
        or host_gpus <= 0
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or float(fraction) <= 0
        or not isinstance(rounds, list)
        or not rounds
        or isinstance(rounds[0], bool)
        or not isinstance(rounds[0], (int, float))
        or float(rounds[0]) <= 0
    ):
        raise StatusError("adaptive policy budget is malformed")
    usage = status.get("gpu_usage")
    if not isinstance(usage, list):
        raise StatusError("adaptive GPU usage inventory is malformed")
    cutoff = now - window
    rolling = 0.0
    for interval in usage:
        if not isinstance(interval, Mapping):
            raise StatusError("adaptive GPU usage interval is malformed")
        started = _epoch_seconds(interval.get("started_at"), "adaptive GPU usage start")
        ended = _epoch_seconds(interval.get("ended_at"), "adaptive GPU usage end")
        gpu_count = interval.get("gpu_count")
        if ended < started or type(gpu_count) is not int or gpu_count <= 0:
            raise StatusError("adaptive GPU usage interval is malformed")
        overlap = max(0.0, min(ended, now) - max(started, cutoff))
        rolling += overlap * gpu_count
    maximum = float(window * host_gpus) * float(fraction)
    requested = float(rounds[0])
    return {
        "allowed": rolling + requested <= maximum,
        "host_gpu_count": host_gpus,
        "maximum_fraction": float(fraction),
        "maximum_gpu_seconds": maximum,
        "projected_gpu_seconds": rolling + requested,
        "remaining_gpu_seconds": max(0.0, maximum - rolling),
        "requested_gpu_seconds": requested,
        "rolling_gpu_seconds": rolling,
        "rolling_window_seconds": window,
    }


def _adaptive_trigger_summary(
    status: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]],
    budget: Mapping[str, Any],
    observation: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if policy is None or observation is None:
        policy_trigger = (
            policy.get("trigger")
            if isinstance(policy, Mapping)
            and isinstance(policy.get("trigger"), Mapping)
            else {}
        )
        return {
            "active_epoch_id": status.get("active_epoch_id"),
            "active_trial_id": status.get("active_trial_id"),
            "blocked_by_active_epoch": status.get("active_epoch_id") is not None,
            "blocked_by_active_trial": status.get("active_trial_id") is not None,
            "minimum_admitted_samples_without_promotion": policy_trigger.get(
                "minimum_admitted_samples_without_promotion"
            ),
            "last_epoch_admitted_samples": status.get("last_epoch_admitted_samples"),
            "gpu_budget_allows_next_trial": budget.get("allowed"),
        }
    trigger_policy = policy.get("trigger")
    queue_policy = policy.get("queue")
    trials_policy = policy.get("trials")
    if not all(
        isinstance(item, Mapping)
        for item in (trigger_policy, queue_policy, trials_policy)
    ):
        raise StatusError("adaptive trigger policy is malformed")
    admitted = observation.get("admitted_samples")
    promoted = observation.get("last_promotion_admitted_samples")
    queue_depth = observation.get("candidate_queue_depth")
    last_epoch = status.get("last_epoch_admitted_samples")
    if (
        type(admitted) is not int
        or admitted < 0
        or type(promoted) is not int
        or promoted < 0
        or type(queue_depth) is not int
        or queue_depth < 0
        or (last_epoch is not None and (type(last_epoch) is not int or last_epoch < 0))
    ):
        raise StatusError("adaptive trigger observation is malformed")
    baseline = max(promoted, last_epoch or 0)
    required = trigger_policy.get("minimum_admitted_samples_without_promotion")
    maximum_queue = queue_policy.get("maximum_candidate_queue_depth")
    maximum_active = trials_policy.get("maximum_active")
    if (
        type(required) is not int
        or required < 0
        or type(maximum_queue) is not int
        or maximum_queue < 0
        or type(maximum_active) is not int
        or maximum_active < 1
        or baseline > admitted
    ):
        raise StatusError("adaptive trigger policy or watermark is malformed")
    active_count = 1 if status.get("active_trial_id") is not None else 0
    reasons = []
    if admitted - baseline < required:
        reasons.append("INSUFFICIENT_ADMITTED_SAMPLES")
    if queue_depth > maximum_queue:
        reasons.append("CANDIDATE_QUEUE_UNBOUNDED")
    if active_count >= maximum_active:
        reasons.append("ACTIVE_TRIAL_EXISTS")
    if budget.get("allowed") is False:
        reasons.append("GPU_BUDGET_EXHAUSTED")
    if status.get("active_epoch_id") is not None:
        reasons.append("ACTIVE_EPOCH_EXISTS")
    return {
        "active_trial_count": active_count,
        "admitted_samples_since_baseline": admitted - baseline,
        "budget": dict(budget),
        "candidate_queue_depth": queue_depth,
        "eligible": not reasons,
        "maximum_candidate_queue_depth": maximum_queue,
        "reason_codes": reasons,
        "required_admitted_samples": required,
    }


def _adaptive_observability(
    root: Path,
    now: float,
    inputs: Mapping[str, Any],
    champion: Optional[Mapping[str, Any]],
    controller_champion_hash: Any,
) -> Mapping[str, Any]:
    adaptive_spec = (
        inputs["adaptive_spec"]["value"] if "adaptive_spec" in inputs else None
    )
    configured_adaptive_root = (
        adaptive_spec.get("root")
        if isinstance(adaptive_spec, Mapping)
        and adaptive_spec.get("contract") == ADAPTIVE_SERVICE_SPEC_CONTRACT
        else str(root / "promotion" / "adaptive")
    )
    adaptive_root = _state_path(
        configured_adaptive_root,
        root,
        "adaptive training state root",
    )
    status_path = adaptive_root / "status.json"
    recipe_path = adaptive_root / "active-recipe.json"
    service_status_path = adaptive_root / "service-status.json"
    observations = {
        "adaptive_training": _file_observation(status_path, now),
        "adaptive_active_recipe": _file_observation(recipe_path, now),
        "adaptive_service": _file_observation(service_status_path, now),
    }
    status = _load_canonical(status_path, "adaptive training status")
    store_status = status
    if store_status is not None:
        _validate_contract(
            store_status,
            contract=ADAPTIVE_STATUS_CONTRACT,
            role="adaptive training status",
            hash_field="status_sha256",
        )
    service_status = _load_canonical(
        service_status_path, "adaptive training service status"
    )
    service_observation: Optional[Mapping[str, Any]] = None
    nested_status: Optional[Mapping[str, Any]] = None
    service_freshness: Mapping[str, Any] = {}
    if service_status is not None:
        _validate_contract(
            service_status,
            contract=ADAPTIVE_SERVICE_STATUS_CONTRACT,
            role="adaptive training service status",
            hash_field="status_sha256",
        )
        if adaptive_spec is not None and service_status.get(
            "service_spec_sha256"
        ) != adaptive_spec.get("spec_sha256"):
            raise StatusError(
                "adaptive training service status specification binding changed"
            )
        raw_nested_status = service_status.get("adaptive_status")
        if not isinstance(raw_nested_status, Mapping):
            raise StatusError("adaptive training service nested status is malformed")
        nested_status = raw_nested_status
        _validate_contract(
            nested_status,
            contract=ADAPTIVE_STATUS_CONTRACT,
            role="adaptive training service nested status",
            hash_field="status_sha256",
        )
        if store_status is not None and nested_status != store_status:
            raise StatusError(
                "adaptive training service status contradicts store status"
            )
        raw_observation = service_status.get("observation")
        if raw_observation is not None:
            if not isinstance(raw_observation, Mapping):
                raise StatusError("adaptive training service observation is malformed")
            _validate_contract(
                raw_observation,
                contract=ADAPTIVE_OBSERVATION_CONTRACT,
                role="adaptive training service observation",
                hash_field="observation_sha256",
            )
            service_observation = raw_observation
        service_freshness = _observation_freshness(
            service_status,
            observations["adaptive_service"],
            now=now,
            maximum_age_seconds=AUTONOMY_STATUS_STALE_SECONDS,
            timestamp_keys=("observed_at_unix",),
            role="adaptive training service status",
        )
    effective_status = status if status is not None else nested_status
    summary: dict[str, Any] = {}
    status_freshness: Mapping[str, Any] = {}
    if effective_status is not None:
        status = effective_status
        _validate_contract(
            status,
            contract=ADAPTIVE_STATUS_CONTRACT,
            role="adaptive training status",
            hash_field="status_sha256",
        )
        epochs = status.get("epochs")
        trials = status.get("trials")
        if not isinstance(epochs, Mapping) or not isinstance(trials, Mapping):
            raise StatusError("adaptive training status lifecycle is malformed")
        active_epoch_id = status.get("active_epoch_id")
        active_trial_id = status.get("active_trial_id")
        if active_epoch_id is not None and (
            not isinstance(active_epoch_id, str) or active_epoch_id not in epochs
        ):
            raise StatusError("adaptive active epoch is malformed")
        if active_trial_id is not None and (
            not isinstance(active_trial_id, str) or active_trial_id not in trials
        ):
            raise StatusError("adaptive active trial is malformed")
        status_freshness = (
            service_freshness
            if service_status is not None
            else _observation_freshness(
                status,
                observations["adaptive_training"],
                now=now,
                maximum_age_seconds=AUTONOMY_STATUS_STALE_SECONDS,
                timestamp_keys=("generated_at_utc", "observed_at_unix"),
                role="adaptive training status",
            )
        )
        policy = (
            inputs["autonomy_policy"]["value"] if "autonomy_policy" in inputs else None
        )
        if policy is not None and status.get("policy_hash") != canonical_sha256(policy):
            raise StatusError("adaptive training status policy binding changed")
        budget = _adaptive_budget_summary(status, policy, now)
        trigger_value = status.get("trigger", status.get("trigger_decision"))
        if trigger_value is not None and not isinstance(trigger_value, Mapping):
            raise StatusError("adaptive trigger status is malformed")
        if trigger_value is None:
            trigger_value = _adaptive_trigger_summary(
                status,
                policy,
                budget,
                service_observation,
            )

        handoffs: list[Mapping[str, Any]] = []
        seen_handoff_paths: set[Path] = set()
        for trial_id, trial in sorted(trials.items()):
            if not isinstance(trial_id, str) or not isinstance(trial, Mapping):
                raise StatusError("adaptive trial status is malformed")
            raw_handoff_path = trial.get("handoff_path")
            if raw_handoff_path is None:
                continue
            handoff_path = _state_path(
                raw_handoff_path,
                root,
                "adaptive candidate handoff",
            )
            if not _is_within(handoff_path, adaptive_root):
                raise StatusError(
                    "adaptive candidate handoff is outside adaptive state"
                )
            if handoff_path in seen_handoff_paths:
                continue
            seen_handoff_paths.add(handoff_path)
            handoff = _load_canonical(handoff_path, "adaptive candidate handoff")
            if handoff is None:
                raise StatusError("adaptive candidate handoff is missing")
            _validate_contract(
                handoff,
                contract=ADAPTIVE_HANDOFF_CONTRACT,
                role="adaptive candidate handoff",
                hash_field="manifest_sha256",
            )
            if handoff.get("trial_id") != trial_id:
                raise StatusError("adaptive candidate handoff contradicts trial status")
            candidate = handoff.get("candidate")
            if (
                handoff.get("direct_promotion_permitted") is not False
                or not isinstance(handoff.get("handoff_id"), str)
                or not isinstance(candidate, Mapping)
            ):
                raise StatusError("adaptive candidate handoff is malformed")
            _require_sha256_string(
                candidate.get("sha256"),
                "adaptive candidate handoff candidate hash",
            )
            handoffs.append(
                {
                    "path": str(handoff_path),
                    "handoff_id": handoff.get("handoff_id"),
                    "trial_id": trial_id,
                    "epoch_id": handoff.get("epoch_id"),
                    "candidate_sha256": (
                        candidate.get("sha256")
                        if isinstance(candidate, Mapping)
                        else None
                    ),
                    "parent_champion_model_sha256": handoff.get(
                        "parent_champion_model_sha256"
                    ),
                    "recipe_sha256": handoff.get("recipe_sha256"),
                    "manifest_sha256": handoff.get("manifest_sha256"),
                    "direct_promotion_permitted": handoff.get(
                        "direct_promotion_permitted"
                    ),
                }
            )
        summary = {
            "path": str(status_path),
            "status_age_seconds": status_freshness["freshness_age_seconds"],
            "status_stale": status_freshness["stale"],
            "active_epoch_id": active_epoch_id,
            "active_epoch": (
                dict(epochs[active_epoch_id]) if active_epoch_id is not None else None
            ),
            "active_trial_id": active_trial_id,
            "active_trial": (
                dict(trials[active_trial_id]) if active_trial_id is not None else None
            ),
            "gpu_budget": budget,
            "trigger": dict(trigger_value),
            "handoffs": handoffs,
            "latest_handoff": handoffs[-1] if handoffs else None,
            "last_sequence": status.get("last_sequence"),
            "last_event_hash": status.get("last_event_hash"),
            "service": {
                "path": str(service_status_path),
                "age_seconds": service_freshness.get("freshness_age_seconds"),
                "stale": service_freshness.get("stale"),
                "blocked_reason": (
                    service_status.get("blocked_reason")
                    if service_status is not None
                    else None
                ),
                "active_work_id": (
                    service_status.get("active_work_id")
                    if service_status is not None
                    else None
                ),
                "actions": (
                    service_status.get("actions") if service_status is not None else []
                ),
                "error": (
                    service_status.get("error") if service_status is not None else None
                ),
            },
        }

    recipe = _load_canonical(recipe_path, "active training recipe")
    active_recipe: Mapping[str, Any] = {}
    recipe_champion_mismatch = False
    if recipe is not None:
        _validate_contract(
            recipe,
            contract=ADAPTIVE_RECIPE_BINDING_CONTRACT,
            role="active training recipe",
            hash_field="record_sha256",
        )
        for field in (
            "admitted_data_manifest_sha256",
            "champion_checkpoint_sha256",
            "champion_model_sha256",
            "recipe_sha256",
        ):
            _require_sha256_string(recipe.get(field), f"active training recipe {field}")
        previous_record = recipe.get("previous_record_sha256")
        if previous_record is not None:
            _require_sha256_string(
                previous_record,
                "active training recipe previous_record_sha256",
            )
        watermarks = recipe.get("data_watermark_sha256s")
        if not isinstance(watermarks, Mapping) or not watermarks:
            raise StatusError("active training recipe watermark bindings are malformed")
        for name, digest in watermarks.items():
            if not isinstance(name, str) or not name:
                raise StatusError("active training recipe watermark name is malformed")
            _require_sha256_string(digest, f"active training recipe watermark {name}")
        if (
            not isinstance(recipe.get("generation_id"), str)
            or not recipe.get("generation_id")
            or not isinstance(recipe.get("recipe_path"), str)
            or not recipe.get("recipe_path")
        ):
            raise StatusError("active training recipe identity is malformed")
        _epoch_seconds(
            recipe.get("activated_at_utc"),
            "active training recipe activated_at_utc",
        )
        rollback = recipe.get("rollback")
        if rollback is not None:
            if not isinstance(rollback, Mapping):
                raise StatusError("active training recipe rollback is malformed")
            _validate_contract(
                rollback,
                contract=ADAPTIVE_ROLLBACK_CONTRACT,
                role="active training recipe rollback",
                hash_field="rollback_sha256",
            )
        champion_hash = None
        champion_generation = None
        if isinstance(champion, Mapping):
            for key in ("championHash", "champion_hash", "sha256"):
                if isinstance(champion.get(key), str):
                    champion_hash = champion[key]
                    break
            for key in ("generationId", "generation_id"):
                if isinstance(champion.get(key), str):
                    champion_generation = champion[key]
                    break
        if champion_hash is None and isinstance(controller_champion_hash, str):
            champion_hash = controller_champion_hash
        recipe_champion_mismatch = bool(
            champion_hash is not None
            and recipe.get("champion_model_sha256") != champion_hash
        )
        if (
            champion_generation is not None
            and recipe.get("generation_id") != champion_generation
        ):
            recipe_champion_mismatch = True
        active_recipe = {
            "path": str(recipe_path),
            "recipe_sha256": recipe.get("recipe_sha256"),
            "recipe_path": recipe.get("recipe_path"),
            "champion_model_sha256": recipe.get("champion_model_sha256"),
            "champion_checkpoint_sha256": recipe.get("champion_checkpoint_sha256"),
            "generation_id": recipe.get("generation_id"),
            "activated_at_utc": recipe.get("activated_at_utc"),
            "record_sha256": recipe.get("record_sha256"),
            "previous_record_sha256": recipe.get("previous_record_sha256"),
            "champion_matches": not recipe_champion_mismatch,
        }
    summary["active_recipe"] = active_recipe
    return {
        "summary": summary,
        "status": store_status,
        "service_status": service_status,
        "recipe": recipe,
        "observations": observations,
        "recipe_champion_mismatch": recipe_champion_mismatch,
    }


def _suite_rotation_observability(
    root: Path,
    now: float,
    inputs: Mapping[str, Any],
    promotion_runtime: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if "suite_registry_spec" not in inputs:
        return {
            "summary": {},
            "status": None,
            "active_pointer": None,
            "observations": {},
            "runtime_divergence": False,
            "projection_inconsistent": False,
        }
    spec_record = inputs["suite_registry_spec"]
    spec_path = spec_record["path"]
    spec = spec_record["value"]
    registry_root = _state_path(
        spec.get("registry_root"),
        root,
        "evaluation suite registry_root",
    )
    status_path = registry_root / "status.json"
    active_path = registry_root / "active-suite.json"
    observations = {
        "suite_rotation": _file_observation(status_path, now),
        "suite_active_pointer": _file_observation(active_path, now),
    }
    service_status: Optional[Mapping[str, Any]] = None
    service_summary: Mapping[str, Any] = {}
    service_spec_record = inputs.get("suite_rotation_service_spec")
    if isinstance(service_spec_record, Mapping):
        service_spec = service_spec_record.get("value")
        service_spec_path = service_spec_record.get("path")
        if not isinstance(service_spec, Mapping) or not isinstance(
            service_spec_path, Path
        ):
            raise StatusError(
                "suite rotation service specification record is malformed"
            )
        results = service_spec.get("results")
        status_binding = (
            results.get("status")
            if isinstance(results, Mapping)
            else None
        )
        status_template = (
            status_binding.get("path")
            if isinstance(status_binding, Mapping)
            else None
        )
        if (
            not isinstance(status_template, str)
            or "{" in status_template
            or "}" in status_template
        ):
            raise StatusError(
                "suite rotation service status path is not fixed"
            )
        service_status_path = _state_path(
            status_template,
            root,
            "suite rotation service status",
        )
        observations["suite_rotation_service"] = _file_observation(
            service_status_path, now
        )
        service_status = _load_canonical(
            service_status_path,
            "suite rotation service status",
        )
        if service_status is not None:
            _validate_contract(
                service_status,
                contract=SUITE_ROTATION_SERVICE_STATUS_CONTRACT,
                role="suite rotation service status",
                hash_field="status_sha256",
            )
            binding = service_status.get("service_spec")
            if (
                not isinstance(binding, Mapping)
                or binding.get("path") != str(service_spec_path)
                or binding.get("sha256") != file_sha256(service_spec_path)
                or binding.get("identity")
                != service_spec.get("spec_sha256")
            ):
                raise StatusError(
                    "suite rotation service status specification binding changed"
                )
            service_freshness = _observation_freshness(
                service_status,
                observations["suite_rotation_service"],
                now=now,
                maximum_age_seconds=AUTONOMY_STATUS_STALE_SECONDS,
                timestamp_keys=("generated_at_utc", "observed_at_unix"),
                role="suite rotation service status",
            )
            service_summary = {
                "path": str(service_status_path),
                "state": service_status.get("state"),
                "next_action": service_status.get("next_action"),
                "candidate_suite_id": service_status.get(
                    "candidate_suite_id"
                ),
                "activation_performed": service_status.get(
                    "activation_performed"
                ),
                "active_pointer_mutated": service_status.get(
                    "active_pointer_mutated"
                ),
                "error": service_status.get("error"),
                "age_seconds": service_freshness.get(
                    "freshness_age_seconds"
                ),
                "stale": service_freshness.get("stale"),
            }
    status = _load_canonical(status_path, "evaluation suite rotation status")
    active_pointer = _load_canonical(active_path, "active evaluation suite pointer")
    status_freshness: Mapping[str, Any] = {}
    if status is not None:
        _validate_contract(
            status,
            contract=SUITE_ROTATION_STATUS_CONTRACT,
            role="evaluation suite rotation status",
            hash_field="status_sha256",
        )
        status_spec = status.get("spec")
        if (
            not isinstance(status_spec, Mapping)
            or status_spec.get("path") != str(spec_path)
            or status_spec.get("sha256") != file_sha256(spec_path)
            or status_spec.get("identity") != spec.get("spec_sha256")
        ):
            raise StatusError(
                "evaluation suite rotation status specification binding changed"
            )
        if not isinstance(status.get("in_flight_evaluations"), list):
            raise StatusError("evaluation suite rotation pin inventory is malformed")
        if any(not isinstance(pin, Mapping) for pin in status["in_flight_evaluations"]):
            raise StatusError("evaluation suite rotation pin is malformed")
        if not isinstance(status.get("retained_suites"), list):
            raise StatusError(
                "evaluation suite rotation version inventory is malformed"
            )
        cadence = status.get("cadence")
        if not isinstance(cadence, Mapping):
            raise StatusError("evaluation suite rotation cadence is malformed")
        status_freshness = _observation_freshness(
            status,
            observations["suite_rotation"],
            now=now,
            maximum_age_seconds=AUTONOMY_STATUS_STALE_SECONDS,
            timestamp_keys=("generated_at_utc", "observed_at_unix"),
            role="evaluation suite rotation status",
        )

    if active_pointer is not None:
        _validate_contract(
            active_pointer,
            contract=ACTIVE_SUITE_CONTRACT,
            role="active evaluation suite pointer",
            hash_field="record_sha256",
        )
        for field in (
            "spec_sha256",
            "suite_id",
            "version_sha256",
            "manifest_sha256",
            "manifest_identity",
            "activation_champion_sha256",
            "event_sha256",
        ):
            _require_sha256_string(
                active_pointer.get(field),
                f"active evaluation suite {field}",
            )
        if active_pointer.get("spec_sha256") != spec.get("spec_sha256"):
            raise StatusError("active evaluation suite specification binding changed")
        if active_pointer.get("suite_id") != active_pointer.get("manifest_sha256"):
            raise StatusError("active evaluation suite content identity is malformed")
        active_manifest_path = _state_path(
            active_pointer.get("manifest_path"),
            root,
            "active evaluation suite manifest",
        )
        active_manifest = _load_canonical(
            active_manifest_path, "active evaluation suite manifest"
        )
        if active_manifest is None or file_sha256(
            active_manifest_path
        ) != active_pointer.get("manifest_sha256"):
            raise StatusError("active evaluation suite manifest binding changed")
        manifest_payload = dict(active_manifest)
        manifest_identity = manifest_payload.pop("manifestPayloadSha256", None)
        if manifest_identity is not None and (
            manifest_identity != canonical_sha256(manifest_payload)
            or active_pointer.get("manifest_identity") != manifest_identity
        ):
            raise StatusError("active evaluation suite manifest identity changed")

    projection_inconsistent = False
    status_active = (
        status.get("active_suite")
        if status is not None and isinstance(status.get("active_suite"), Mapping)
        else None
    )
    if (
        status is not None
        and status.get("active_suite") is not None
        and (status_active is None)
    ):
        raise StatusError("evaluation suite active status is malformed")
    if status_active is None:
        projection_inconsistent = active_pointer is not None
    elif active_pointer is None:
        projection_inconsistent = True
    else:
        projection_inconsistent = status_active.get("suite_id") != active_pointer.get(
            "suite_id"
        ) or status_active.get("version_sha256") != active_pointer.get("version_sha256")
    if status is not None and status.get("active_projection_consistent") is False:
        projection_inconsistent = True

    frozen_suite: Mapping[str, Any] = {}
    runtime_divergence = False
    if promotion_runtime is not None:
        paths = promotion_runtime.get("paths")
        hashes = promotion_runtime.get("hashes")
        if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
            raise StatusError("autonomy promotion runtime suite binding is malformed")
        raw_suite_root = paths.get("suites")
        raw_suite_hash = hashes.get("suiteManifest")
        if raw_suite_root is not None or raw_suite_hash is not None:
            suite_root = _state_path(
                raw_suite_root,
                root,
                "autonomy promotion runtime suites",
            )
            manifest_path = suite_root / "manifest.json"
            if (
                not isinstance(raw_suite_hash, str)
                or manifest_path.is_symlink()
                or not manifest_path.is_file()
                or file_sha256(manifest_path) != raw_suite_hash
            ):
                raise StatusError("autonomy promotion runtime suite binding changed")
            observations["frozen_runtime_suite"] = _file_observation(manifest_path, now)
            frozen_suite = {
                "root": str(suite_root),
                "manifest_path": str(manifest_path),
                "manifest_sha256": raw_suite_hash,
            }
            if active_pointer is not None:
                runtime_divergence = (
                    active_pointer.get("manifest_sha256") != raw_suite_hash
                )

    desired_suite_id = None
    desired_version = None
    cadence: Mapping[str, Any] = {}
    pins: list[Any] = []
    pending_continuity: Mapping[str, Any] = {}
    state = None
    next_action = None
    if status is not None:
        state = status.get("state")
        next_action = status.get("next_action")
        candidate_suite_id = status.get("candidate_suite_id")
        desired_suite_id = (
            candidate_suite_id
            if isinstance(candidate_suite_id, str)
            else (status_active.get("suite_id") if status_active is not None else None)
        )
        for retained in status["retained_suites"]:
            if not isinstance(retained, Mapping):
                raise StatusError("evaluation suite retained version is malformed")
            if retained.get("suite_id") == desired_suite_id:
                desired_version = retained.get("version_sha256")
        cadence = dict(status["cadence"])
        pins = list(status["in_flight_evaluations"])
        continuity_required = state == "continuity-pending"
        pending_continuity = {
            "required": continuity_required,
            "request_id": (
                status.get("current_request_id") if continuity_required else None
            ),
            "suite_id": (candidate_suite_id if continuity_required else None),
        }

    active_summary: Mapping[str, Any] = {}
    if active_pointer is not None:
        active_summary = {
            "path": str(active_path),
            "suite_id": active_pointer.get("suite_id"),
            "version_sha256": active_pointer.get("version_sha256"),
            "manifest_path": active_pointer.get("manifest_path"),
            "manifest_sha256": active_pointer.get("manifest_sha256"),
            "activation_champion_sha256": active_pointer.get(
                "activation_champion_sha256"
            ),
            "activation_generation_id": active_pointer.get("activation_generation_id"),
            "activated_at_utc": active_pointer.get("activated_at_utc"),
            "record_sha256": active_pointer.get("record_sha256"),
        }
    summary = {
        "spec": _binding_summary(spec_path, spec, str(spec.get("spec_sha256"))),
        "registry_root": str(registry_root),
        "path": str(status_path),
        "status_age_seconds": status_freshness.get("freshness_age_seconds"),
        "status_stale": status_freshness.get("stale"),
        "state": state,
        "next_action": next_action,
        "desired_suite_id": desired_suite_id,
        "desired_version_sha256": desired_version,
        "active_suite": active_summary,
        "cadence": cadence,
        "pins": pins,
        "pin_count": len(pins),
        "pending_continuity": pending_continuity,
        "frozen_runtime_suite": frozen_suite,
        "active_projection_consistent": not projection_inconsistent,
        "runtime_divergence": runtime_divergence,
        "service": service_summary,
    }
    return {
        "summary": summary,
        "status": status,
        "active_pointer": active_pointer,
        "observations": observations,
        "runtime_divergence": runtime_divergence,
        "projection_inconsistent": projection_inconsistent,
        "service_status": service_status,
    }


def _collect_autonomy_status(
    root: Path,
    *,
    now: float,
    runtime_config_path: Path,
    runtime: Optional[Mapping[str, Any]],
    champion: Optional[Mapping[str, Any]],
    controller_champion_hash: Any,
    scheduler: Mapping[str, Any],
    gpu_lease: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], set[str]]:
    bootstrap = _bootstrap_observability(root, now)
    runtime_status = _autonomy_runtime_observability(
        root,
        now,
        runtime_config_path,
        runtime,
        bootstrap,
    )
    full_autonomy = runtime_status["full_autonomy"]
    inputs = runtime_status["inputs"]
    executor = _cluster_executor_observability(root, now, inputs, scheduler)
    adaptive = _adaptive_observability(
        root,
        now,
        inputs,
        champion,
        controller_champion_hash,
    )
    suite = _suite_rotation_observability(
        root,
        now,
        inputs,
        runtime_status["promotion_runtime"],
    )
    observations = {
        **bootstrap["observations"],
        **runtime_status["observations"],
        **executor["observations"],
        **adaptive["observations"],
        **suite["observations"],
    }
    warnings: set[str] = set()

    if full_autonomy:
        bootstrap_status = bootstrap["status"]
        if bootstrap_status is None:
            warnings.add("autonomy-bootstrap-status-missing")
        elif (
            bootstrap_status.get("state") not in {"active", "safety-halt"}
            and bootstrap["observations"]["autonomy_bootstrap"].get("age_seconds", 0.0)
            > AUTONOMY_STATUS_STALE_SECONDS
        ):
            warnings.add("autonomy-bootstrap-status-stale")

        if executor["heartbeat"] is None:
            warnings.add("cluster-executor-heartbeat-missing")
        elif executor["summary"]["heartbeat"].get("stale"):
            warnings.add("cluster-executor-heartbeat-stale")
        if executor["status"] is not None and executor["summary"].get("status_stale"):
            warnings.add("cluster-executor-status-stale")

        if adaptive["status"] is None:
            warnings.add("adaptive-training-status-missing")
        elif adaptive["summary"].get("status_stale"):
            warnings.add("adaptive-training-status-stale")

        if suite["status"] is None:
            warnings.add("suite-rotation-status-missing")
        elif suite["summary"].get("status_stale"):
            warnings.add("suite-rotation-status-stale")
        if "suite_rotation_service_spec" in inputs:
            if suite["service_status"] is None:
                warnings.add("suite-rotation-service-status-missing")
            elif suite["summary"].get("service", {}).get("stale"):
                warnings.add("suite-rotation-service-status-stale")
            if suite["summary"].get("service", {}).get("error"):
                warnings.add("suite-rotation-service-failed")

    if adaptive["recipe_champion_mismatch"]:
        warnings.add("adaptive-recipe-champion-mismatch")
    if suite["runtime_divergence"]:
        warnings.add("suite-active-pointer-runtime-divergence")
    if suite["projection_inconsistent"]:
        warnings.add("suite-active-projection-inconsistent")
    if executor["disagreement"]:
        warnings.add("scheduler-executor-disagreement")

    terminal_reasons = [
        *bootstrap["terminal_reasons"],
        *executor["terminal_reasons"],
    ]
    if full_autonomy and scheduler:
        scheduler_reason = _mapping_reason(scheduler.get("safety_halt"))
        if scheduler_reason is not None:
            terminal_reasons.append(
                {"source": "cluster-scheduler", "reason": scheduler_reason}
            )
        raw_gpu_halts = scheduler.get("gpu_safety_halts")
        if isinstance(raw_gpu_halts, Mapping):
            for gpu_id, halt in sorted(raw_gpu_halts.items()):
                reason = _mapping_reason(halt)
                if reason is not None:
                    terminal_reasons.append(
                        {
                            "source": f"cluster-scheduler:{gpu_id}",
                            "reason": reason,
                        }
                    )
    if full_autonomy and gpu_lease.get("safety_halt"):
        terminal_reasons.append(
            {
                "source": "gpu-lease",
                "reason": str(
                    gpu_lease.get("safety_reason") or "GPU lease entered safety halt"
                ),
            }
        )
    terminal_reasons = [
        dict(item)
        for item in {
            (item["source"], item["reason"]): item for item in terminal_reasons
        }.values()
    ]
    terminal_reasons.sort(key=lambda item: (item["source"], item["reason"]))
    if terminal_reasons:
        warnings.add("autonomy-terminal-halt")

    terminal = {
        "halted": bool(terminal_reasons),
        "reasons": terminal_reasons,
        "remediation_reason": (
            terminal_reasons[0]["reason"] if terminal_reasons else None
        ),
    }
    result = {
        "full_autonomy": full_autonomy,
        "runtime": runtime_status["summary"],
        "bootstrap": bootstrap["summary"],
        "executor": executor["summary"],
        "adaptive": adaptive["summary"],
        "suite_rotation": suite["summary"],
        "terminal": terminal,
        "terminal_remediation_reason": terminal["remediation_reason"],
    }
    return result, observations, warnings


def collect_status(run_root: Path, *, now: Optional[float] = None) -> Mapping[str, Any]:
    root = Path(run_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise StatusError("run root must be an absolute non-symlink directory")
    observed = time.time() if now is None else float(now)
    promotion = root / "promotion"
    controller_path = promotion / "status.json"
    supervisor_path = promotion / "supervisor" / "service.json"
    trainer_observation_path = promotion / "supervisor" / "trainer-observation.json"
    backpressure_path = promotion / "operations" / "backpressure.json"
    champion_path = promotion / "champion.json"
    scheduler_path = promotion / "scheduler" / "state.json"
    controller = _load_canonical(controller_path, "controller status")
    supervisor = _load_canonical(supervisor_path, "supervisor heartbeat")
    trainer_observation_value = _load_canonical(
        trainer_observation_path, "trainer observation"
    )
    backpressure = _load_canonical(backpressure_path, "backpressure status")
    champion = _load_canonical(champion_path, "champion projection")
    scheduler_value = _load_canonical(scheduler_path, "cluster scheduler state")
    runtime_config_path, runtime = _discover_runtime_config(root, supervisor)
    gpu_config_path, gpu_lease_path = _discover_gpu_lease_paths(root, runtime)
    gpu_lease_value = _load_canonical(gpu_lease_path, "GPU lease state")

    controller_result = (
        controller.get("result")
        if isinstance(controller, Mapping)
        and isinstance(controller.get("result"), Mapping)
        else {}
    )
    runtime_min_free_bytes = _runtime_min_free_bytes(runtime)
    disk_free_bytes = int(shutil.disk_usage(root).free)
    disk = {
        "free_bytes": disk_free_bytes,
        "runtime_min_free_bytes": runtime_min_free_bytes,
        "below_runtime_minimum": (
            runtime_min_free_bytes is not None
            and disk_free_bytes < runtime_min_free_bytes
        ),
        "runtime_config_path": (
            str(runtime_config_path) if runtime is not None else None
        ),
    }
    backpressure_observation = _file_observation(backpressure_path, observed)
    backpressure_freshness: Mapping[str, Any] = {}
    if backpressure is not None:
        if (
            type(backpressure.get("allowExport")) is not bool
            or type(backpressure.get("allowEvaluation")) is not bool
        ):
            raise StatusError("backpressure allowances are malformed")
        backpressure_freshness = _backpressure_freshness(
            backpressure, backpressure_observation, observed
        )
        backpressure_observation = {
            **backpressure_observation,
            **backpressure_freshness,
        }
    observations = {
        "controller": _file_observation(controller_path, observed),
        "supervisor": _file_observation(supervisor_path, observed),
        "trainer_observation": _file_observation(trainer_observation_path, observed),
        "backpressure": backpressure_observation,
        "champion": _file_observation(champion_path, observed),
        "scheduler": _file_observation(scheduler_path, observed),
        "selfplay": _file_observation(root / "selfplay.summary.json", observed),
        "shuffle": _file_observation(root / "shuffle-input-state.json", observed),
        "runtime_config": _file_observation(runtime_config_path, observed),
        "gpu_lease": _file_observation(gpu_lease_path, observed),
    }
    if gpu_config_path is not None:
        observations["gpu_lease_config"] = _file_observation(gpu_config_path, observed)
    configured_checkpoint = _runtime_path(runtime, "trainerCheckpoint")
    checkpoint = configured_checkpoint or _latest_file(
        root / "train", "**/checkpoint.ckpt"
    )
    latest_selfplay_training_data, selfplay_training_data_source = (
        _discover_selfplay_training_data(root, runtime)
    )
    latest_report = _latest_file(promotion / "reports", "**/*.json")
    (
        curation_status_path,
        curation_status,
        curation_status_source,
    ) = _discover_curation_status(root)
    observations["checkpoint"] = (
        _file_observation(checkpoint, observed)
        if checkpoint is not None
        else {"present": False}
    )
    selfplay_training_data_observation = (
        _file_observation(latest_selfplay_training_data, observed)
        if latest_selfplay_training_data is not None
        else {"present": False}
    )
    observations["selfplay_training_data"] = {
        **selfplay_training_data_observation,
        "source": selfplay_training_data_source,
    }
    observations["latest_report"] = (
        _file_observation(latest_report, observed)
        if latest_report is not None
        else {"present": False}
    )
    observations["curation"] = (
        _file_observation(curation_status_path, observed)
        if curation_status_path is not None
        else {"present": False}
    )

    raw_warnings = controller_result.get("warnings", [])
    warnings = (
        set(raw_warnings)
        if isinstance(raw_warnings, list)
        and all(isinstance(item, str) for item in raw_warnings)
        else {"controller-warnings-invalid"}
    )
    if controller is None:
        warnings.add("controller-status-missing")
    elif observations["controller"]["age_seconds"] > 90:
        warnings.add("controller-status-stale")
    if supervisor is None:
        warnings.add("supervisor-heartbeat-missing")
    else:
        updated = supervisor.get("updated_at_unix")
        if not isinstance(updated, (int, float)) or observed - float(updated) > 30:
            warnings.add("supervisor-heartbeat-stale")
    trainer_observation: Mapping[str, Any] = {}
    if trainer_observation_value is not None:
        trainer_observation = _trainer_observation_summary(
            trainer_observation_value, trainer_observation_path, observed
        )
        if trainer_observation["stale"]:
            warnings.add("trainer-observation-stale")
        if trainer_observation["decision"] == "abnormal-exit":
            warnings.add("trainer-abnormal-exit")
    elif supervisor is not None and supervisor.get("mutation_enabled") is True:
        warnings.add("trainer-observation-missing")
    if observations["selfplay"].get("present") and (
        observations["selfplay"]["age_seconds"] > 300
    ):
        warnings.add("selfplay-summary-stale")
    if not observations["selfplay_training_data"].get("present"):
        warnings.add("selfplay-training-data-missing")
    elif (
        observations["selfplay_training_data"]["age_seconds"]
        > SELFPLAY_TRAINING_DATA_STALE_SECONDS
    ):
        warnings.add("selfplay-training-data-stale")
    if observations["shuffle"].get("present") and (
        observations["shuffle"]["age_seconds"] > 3600
    ):
        warnings.add("shuffle-state-stale")
    if backpressure_freshness.get("stale"):
        warnings.add("backpressure-status-stale")
    if disk["below_runtime_minimum"]:
        warnings.add("disk-free-below-runtime-minimum")

    scheduler: Mapping[str, Any] = {}
    if scheduler_value is not None:
        scheduler = _scheduler_summary(scheduler_value)
        queued = scheduler.get("work_by_state", {}).get("queued", 0)
        if queued and scheduler.get("active_claims") == 0:
            warnings.add("scheduler-runnable-work-unclaimed")
        if scheduler.get("safety_halt") or scheduler.get("gpu_safety_halts"):
            warnings.add("scheduler-safety-halt")

    gpu_lease: Mapping[str, Any] = {}
    if gpu_lease_value is not None:
        gpu_lease = _gpu_lease_summary(
            gpu_lease_value,
            gpu_lease_path,
            observed,
            controller=controller,
            controller_result=controller_result,
            controller_observation=observations["controller"],
        )
        if gpu_lease["safety_halt"]:
            warnings.add("gpu-lease-safety-halt")
        if gpu_lease["stale_non_trainer_phase"]:
            warnings.add("gpu-lease-non-trainer-phase-stale")

    curation: Mapping[str, Any] = {}
    if curation_status_path is not None and curation_status is not None:
        curation = {
            "path": str(curation_status_path),
            "source": curation_status_source,
            "contract": curation_status.get("contract"),
            "state": curation_status.get("state"),
            "ready_for_labeling": curation_status.get(
                "ready_for_labeling",
                curation_status.get("state") == "complete",
            ),
            "progress": curation_status.get("progress"),
            "next_stage": curation_status.get("next_stage"),
            "accepted_counts": curation_status.get("accepted_counts"),
            "deficits": curation_status.get("deficits"),
            "error": curation_status.get("error"),
        }

    improvement: Mapping[str, Any] = {}
    if latest_report is not None:
        report = _load_canonical(latest_report, "latest promotion report")
        if report is not None:
            improvement = {
                "path": str(latest_report),
                "decision": report.get("decision"),
                "candidate_hash": report.get("candidate_hash"),
                "tested_champion_hash": report.get("tested_champion_hash"),
                "ranking_summary": report.get("ranking_summary"),
            }

    raw_checkpoint_backlog = _directory_count(root / "torchmodels_toexport")
    candidate_inbox = _runtime_path(runtime, "candidateInbox")
    candidate_inbox_depth = _directory_count(
        candidate_inbox or root / "modelstobetested"
    )
    accepted_model_count = _directory_count(root / "models")
    autonomy, autonomy_observations, autonomy_warnings = _collect_autonomy_status(
        root,
        now=observed,
        runtime_config_path=runtime_config_path,
        runtime=runtime,
        champion=champion,
        controller_champion_hash=controller_result.get("championHash"),
        scheduler=scheduler,
        gpu_lease=gpu_lease,
    )
    observations.update(autonomy_observations)
    warnings.update(autonomy_warnings)

    return {
        "schema_version": 1,
        "contract": "risk-score-training-status-v1",
        "observed_at_utc": datetime.datetime.fromtimestamp(
            observed, datetime.timezone.utc
        )
        .isoformat()
        .replace("+00:00", "Z"),
        "run_root": str(root.resolve()),
        "healthy": not warnings,
        "champion": champion,
        "controller": {
            "mode": controller_result.get("mode"),
            "champion_hash": controller_result.get("championHash"),
            "generation_id": controller_result.get("currentGenerationId"),
            "queue_depth": controller_result.get("queueDepth"),
            "active_stage": controller_result.get("activeStage"),
            "active_look": controller_result.get("activeLook"),
            "lease_owner": controller_result.get("leaseOwner"),
            "worker_acknowledgements": controller_result.get("workerAcknowledgements"),
            "promotion_feedback": controller_result.get("promotionFeedback"),
        },
        "backpressure": backpressure,
        "backpressure_freshness": backpressure_freshness,
        "gpu_lease": gpu_lease,
        "disk": disk,
        "trainer": {
            "checkpoint_present": observations["checkpoint"].get("present", False),
            "checkpoint_path": observations["checkpoint"].get("path"),
            "checkpoint_age_seconds": observations["checkpoint"].get("age_seconds"),
            "observation": trainer_observation,
        },
        "backlogs": {
            "raw_checkpoint_depth": raw_checkpoint_backlog,
            "candidate_inbox_depth": candidate_inbox_depth,
        },
        "scheduler": scheduler,
        "autonomy": autonomy,
        "curation": curation,
        "latest_improvement": improvement,
        "pipeline": {
            "raw_checkpoint_backlog": raw_checkpoint_backlog,
            "candidate_inbox_depth": candidate_inbox_depth,
            "accepted_model_count": accepted_model_count,
            "reviewed_position_bank_ready": (
                root / "evaluation" / "source-positions.manifest.json"
            ).is_file(),
            "v3_suite_ready": (
                root / "evaluation" / "promotion-suites-v3" / "manifest.json"
            ).is_file(),
        },
        "artifacts": observations,
        "warnings": sorted(warnings),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = collect_status(args.run_root)
    except (OSError, TypeError, ValueError, StatusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
