#!/usr/bin/env python3
"""Read-only status summary for the closed-loop risk-training pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
import math
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
