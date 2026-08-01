#!/usr/bin/env python3
"""Load one KataGo model and require finite deterministic analysis output."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from risk_score.curate_position_bank import (
    run_analysis,
    validate_deterministic_analysis_config,
)
from risk_score.position_samples import build_analysis_query, canonical_json, file_sha256
from risk_score.promotion_host import HostCommandError, atomic_write_json


def _finite_analysis(record: Mapping[str, Any]) -> bool:
    root = record.get("rootInfo")
    moves = record.get("moveInfos")
    if not isinstance(root, Mapping) or not isinstance(moves, list) or not moves:
        return False
    for key in ("winrate", "scoreLead", "utility", "visits"):
        value = root.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return False
    return 0.0 <= float(root["winrate"]) <= 1.0


def _require_gpu_idle(gpu_index: int) -> str:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if inventory.returncode != 0:
        raise HostCommandError("cannot inventory CUDA GPUs for model probe")
    uuids = [line.strip() for line in inventory.stdout.splitlines() if line.strip()]
    if not 0 <= gpu_index < len(uuids):
        raise HostCommandError("model probe GPU index is absent")
    expected_uuid = uuids[gpu_index]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if processes.returncode != 0:
        raise HostCommandError("cannot query CUDA processes for model probe")
    if any(
        line.split(",", 1)[0].strip() == expected_uuid
        for line in processes.stdout.splitlines()
        if line.strip()
    ):
        raise HostCommandError("model probe GPU is not exclusively idle")
    return expected_uuid


def probe_model(
    *,
    katago: Path,
    config: Path,
    model: Path,
    expected_model_sha256: Optional[str] = None,
    output: Optional[Path] = None,
    gpu_index: int = 7,
    require_idle_gpu: bool = True,
    subprocess_runner: Any = subprocess.run,
) -> Mapping[str, Any]:
    model_path = Path(model)
    if type(gpu_index) is not int or gpu_index < 0:
        raise HostCommandError("model probe GPU index must be nonnegative")
    gpu_uuid = _require_gpu_idle(gpu_index) if require_idle_gpu else None
    validate_deterministic_analysis_config(Path(config))
    actual_hash = file_sha256(model_path)
    if expected_model_sha256 is not None and actual_hash != expected_model_sha256:
        raise HostCommandError("model probe hash mismatch")
    position = {
        "xSize": 19,
        "ySize": 19,
        "board": "/".join(["." * 19] * 19),
        "nextPla": "B",
        "moveLocs": [],
        "movePlas": [],
        "initialTurnNumber": 0,
        "hintLoc": "null",
    }
    with tempfile.TemporaryDirectory(prefix="risk-score-model-probe-") as temporary:
        root = Path(temporary)
        query = root / "query.jsonl"
        query.write_text(
            canonical_json(
                build_analysis_query(
                    position,
                    query_id="model-probe",
                    max_visits=4,
                    powered=False,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        result_path = root / "result.jsonl"
        run_analysis(
            katago=Path(katago),
            config=Path(config),
            model=model_path,
            queries=query,
            output=result_path,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)},
            subprocess_runner=subprocess_runner,
        )
        records = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(records) != 1 or not _finite_analysis(records[0]):
            raise HostCommandError("model probe returned malformed/non-finite analysis")
        result = {
            "schema_version": 1,
            "contract": "risk-score-model-probe-v1",
            "model_sha256": actual_hash,
            "katago_sha256": file_sha256(Path(katago)),
            "config_sha256": file_sha256(Path(config)),
            "finite": True,
            "gpu_uuid": gpu_uuid,
        }
    if output is not None:
        atomic_write_json(Path(output), result)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--katago", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-index", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = probe_model(
            katago=args.katago,
            config=args.config,
            model=args.model,
            expected_model_sha256=args.model_sha256,
            output=args.output,
            gpu_index=args.gpu_index,
        )
    except (HostCommandError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
