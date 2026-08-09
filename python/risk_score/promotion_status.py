#!/usr/bin/env python3
"""Read-only status summary for the closed-loop risk-training pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from risk_score.position_samples import canonical_json
from risk_score.promotion_state import canonical_json_bytes


class StatusError(RuntimeError):
    """The live status tree is missing or internally inconsistent."""


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
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def collect_status(run_root: Path, *, now: Optional[float] = None) -> Mapping[str, Any]:
    root = Path(run_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise StatusError("run root must be an absolute non-symlink directory")
    observed = time.time() if now is None else float(now)
    promotion = root / "promotion"
    controller_path = promotion / "status.json"
    supervisor_path = promotion / "supervisor" / "service.json"
    backpressure_path = promotion / "operations" / "backpressure.json"
    champion_path = promotion / "champion.json"
    controller = _load_canonical(controller_path, "controller status")
    supervisor = _load_canonical(supervisor_path, "supervisor heartbeat")
    backpressure = _load_canonical(backpressure_path, "backpressure status")
    champion = _load_canonical(champion_path, "champion projection")

    controller_result = (
        controller.get("result")
        if isinstance(controller, Mapping)
        and isinstance(controller.get("result"), Mapping)
        else {}
    )
    observations = {
        "controller": _file_observation(controller_path, observed),
        "supervisor": _file_observation(supervisor_path, observed),
        "backpressure": _file_observation(backpressure_path, observed),
        "champion": _file_observation(champion_path, observed),
        "selfplay": _file_observation(root / "selfplay.summary.json", observed),
        "shuffle": _file_observation(root / "shuffle-input-state.json", observed),
    }
    checkpoint = _latest_file(root / "train", "*/checkpoint.ckpt")
    latest_report = _latest_file(promotion / "reports", "**/*.json")
    curation_status_path = _latest_file(
        root / "evaluation" / "curation" / "machine-consensus-v3",
        "**/status.json",
    )
    observations["checkpoint"] = (
        _file_observation(checkpoint, observed)
        if checkpoint is not None
        else {"present": False}
    )
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
    if observations["selfplay"].get("present") and (
        observations["selfplay"]["age_seconds"] > 300
    ):
        warnings.add("selfplay-summary-stale")
    if observations["shuffle"].get("present") and (
        observations["shuffle"]["age_seconds"] > 3600
    ):
        warnings.add("shuffle-state-stale")

    curation: Mapping[str, Any] = {}
    if curation_status_path is not None:
        curation_status = _load_canonical(
            curation_status_path, "machine-consensus curation status"
        )
        if curation_status is not None:
            curation = {
                "path": str(curation_status_path),
                "state": curation_status.get("state"),
                "ready_for_labeling": curation_status.get("ready_for_labeling"),
                "progress": curation_status.get("progress"),
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
        "curation": curation,
        "latest_improvement": improvement,
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
