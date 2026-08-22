#!/usr/bin/env python3
"""Durably keep one provenance-bound extreme-score generation in flight."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


RUN = Path(
    os.environ.get(
        "KATAGO_EXTREME_RUN",
        "/home/ubuntu/kata-go-artifacts/runs/extreme-pilot-20260817-0023",
    )
)
SOURCE = RUN / "source"
PYTHON_ROOT = SOURCE / "python"
GENERATIONS = RUN / "generations"
TEMPLATE = GENERATIONS / "extreme-train-g0014"
ACCEPTED_STATE = RUN / "state/controller-original/accepted-current.json"
POLICY = PYTHON_ROOT / "risk_score/extreme_score_training_policy_v1.json"
WATCHDOG_ROOT = RUN / "operations/extreme-pipeline-watchdog"
WATCHDOG_LOCK = RUN / "state/extreme-pipeline-watchdog.lock"
STOP_FILE = RUN / "state/extreme-pipeline.stop"
STATUS_FILE = RUN / "state/extreme-pipeline-watchdog-status.json"
GPU7_LOCK = RUN / "state/gpu7-training.lock"
GENERATION_PATTERN = re.compile(r"^extreme-train-g([0-9]{4,})$")
MINIMUM_FREE_BYTES = 500 * 1024**3
EXPECTED_GPU_COUNT = 8
WATCHDOG_VERSION = 2

sys.path.insert(0, str(PYTHON_ROOT))

from risk_score.extreme_score_controller import load_accepted_state  # noqa: E402
from risk_score.extreme_score_league import (  # noqa: E402
    canonical_json,
    file_sha256,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def run_command(
    argv: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        env={
            **os.environ,
            "PYTHONPATH": str(PYTHON_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def immutable_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"immutable artifact conflicts: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def immutable_json(path: Path, value: Any, mode: int = 0o444) -> None:
    immutable_write(path, (canonical_json(value) + "\n").encode(), mode)


def atomic_status(value: dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(value) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{STATUS_FILE.name}.", suffix=".partial", dir=STATUS_FILE.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, STATUS_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def generation_directories() -> list[tuple[int, Path]]:
    result = []
    for path in GENERATIONS.iterdir():
        match = GENERATION_PATTERN.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            result.append((int(match.group(1)), path))
    return sorted(result)


def generation_id(number: int) -> str:
    return f"extreme-train-g{number:04d}"


def unit_name(number: int) -> str:
    return f"katago-extreme-training-g{number}"


def service_state(number: int) -> dict[str, str]:
    output = run_command(
        [
            "systemctl",
            "show",
            f"{unit_name(number)}.service",
            "--no-pager",
            "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus",
        ],
        check=False,
    ).stdout
    result: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def quick_worker_state(directory: Path) -> str:
    plan_path = directory / "league-plan.json"
    if not plan_path.is_file():
        return "UNPLANNED"
    plan = json.loads(plan_path.read_text())
    states = []
    for worker in plan["workers"]:
        receipt_path = (
            Path(worker["output_directory"]) / "worker-execution-receipt.json"
        )
        if not receipt_path.is_file():
            states.append("WAIT")
            continue
        receipt = json.loads(receipt_path.read_text())
        outcome = receipt.get("process_outcome", {})
        states.append(
            "OK"
            if outcome.get("status") == "succeeded"
            and outcome.get("returncode") == 0
            else "FAILED"
        )
    if "FAILED" in states:
        return "FAILED"
    if states and all(state == "OK" for state in states):
        return "SUCCEEDED"
    return "RUNNING"


def gpu_occupancy() -> tuple[int, set[int]]:
    gpu_lines = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.splitlines()
    uuid_to_index: dict[str, int] = {}
    for line in gpu_lines:
        raw_index, raw_uuid = line.split(",", 1)
        uuid_to_index[raw_uuid.strip()] = int(raw_index.strip())
    app_lines = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.splitlines()
    occupied = set()
    for line in app_lines:
        raw_uuid, _ = line.split(",", 1)
        if raw_uuid.strip() in uuid_to_index:
            occupied.add(uuid_to_index[raw_uuid.strip()])
    return len(gpu_lines), occupied


def assert_launch_capacity() -> dict[str, Any]:
    usage = shutil.disk_usage(RUN)
    if usage.free < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"free disk below watchdog floor: {usage.free} < {MINIMUM_FREE_BYTES}"
        )
    gpu_count, occupied = gpu_occupancy()
    if gpu_count != EXPECTED_GPU_COUNT:
        raise RuntimeError(
            f"GPU count changed: {gpu_count} != {EXPECTED_GPU_COUNT}"
        )
    collisions = occupied.intersection(range(7))
    if collisions:
        raise RuntimeError(
            f"refusing to launch over occupied self-play GPUs: {sorted(collisions)}"
        )
    return {
        "disk_free_bytes": usage.free,
        "gpu_count": gpu_count,
        "occupied_gpu_indices": sorted(occupied),
    }


def round_script(number: int, identifier: str) -> str:
    short = f"g{number}"
    return f"""#!/bin/bash
set -euo pipefail

PILOT={RUN}
SRC="$PILOT/source"
GEN="$PILOT/generations/{identifier}"
PLAN="$GEN/league-plan.json"
export PYTHONPATH="$SRC/python"
export PYTHONDONTWRITEBYTECODE=1
cd "$SRC/python"

mapfile -t workers < <(python3 - "$PLAN" <<'PYWORKERS'
import json
import sys
for worker in json.load(open(sys.argv[1], encoding="utf-8"))["workers"]:
    print(worker["worker_id"])
PYWORKERS
)

pids=()
cleanup() {{
  for pid in "${{pids[@]:-}}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

for worker in "${{workers[@]}}"; do
  python3 -m risk_score.extreme_score_league run-worker \
    --plan "$PLAN" \
    --worker-id "$worker" \
    > "$PILOT/logs/{short}-worker-${{worker}}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${{pids[@]}}"; do
  wait "$pid" || failed=1
done
pids=()
trap - EXIT INT TERM
if [ "$failed" -ne 0 ]; then
  echo "one or more workers failed" >&2
  exit 1
fi

python3 -m risk_score.extreme_score_league status \
  --plan "$PLAN" \
  > "$PILOT/logs/{short}-worker-status.json"
python3 - "$PILOT/logs/{short}-worker-status.json" <<'PYSTATUS'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert all(worker["state"] == "SUCCEEDED" for worker in value["workers"]), value
PYSTATUS

python3 -m risk_score.extreme_score_provenance run \
  --plan "$PLAN" \
  --shuffle-command-json "$GEN/shuffle-command.json" \
  --shuffled-root "$PILOT/shuffleddata" \
  --output-id {identifier} \
  --claim-root "$PILOT/shuffle-claims" \
  --lock "$PILOT/state/shuffle.lock" \
  | tee "$PILOT/logs/{short}-shuffle.json"

SAMPLES_PER_EPOCH=$(python3 - "$PILOT/shuffleddata/{identifier}/train" "$GEN/adaptive-epoch.json" <<'PYADAPT'
import json
import os
import sys
from pathlib import Path

import numpy as np

data_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
files = sorted(data_dir.glob("*.npz"))
if not files:
    raise SystemExit("adaptive epoch selection found no shuffled training files")
rows = 0
for path in files:
    with np.load(path) as value:
        rows += int(value["binaryInputNCHWPacked"].shape[0])
usable = (rows // 192) * 192
samples = min(7680, usable)
if samples < 192:
    raise SystemExit(f"adaptive epoch has fewer than one usable batch: {{rows}} rows")
value = {{
    "schema_version": 1,
    "contract": "risk-score-extreme-adaptive-epoch-v1",
    "generation_id": "{identifier}",
    "available_rows": rows,
    "batch_size": 192,
    "samples_per_epoch": samples,
}}
data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\\n").encode()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
print(samples)
PYADAPT
)
echo "Adaptive samples per epoch: $SAMPLES_PER_EPOCH"

exec 9>"{GPU7_LOCK}"
flock -x 9
BEFORE_SELECTED=$(python3 - <<'PYBEFORE'
import json
from pathlib import Path
p=Path("{RUN}/train/extreme-pilot-original/extreme-score-progress.json")
print(json.loads(p.read_text())["selected_training_samples"])
PYBEFORE
)

export CUDA_VISIBLE_DEVICES=7
export LD_LIBRARY_PATH=/usr/lib/python3/dist-packages/torch/lib
python3 "$SRC/python/train.py" \
  -traindir "$PILOT/train/extreme-pilot-original" \
  -latestdatadir "$PILOT/shuffleddata" \
  -exportdir "$PILOT/torchmodels_toexport" \
  -exportprefix extreme-pilot-original \
  -pos-len 19 \
  -batch-size 192 \
  -model-kind b40c768nbt \
  -lr-scale 0.25 \
  -use-muon \
  -use-bf16 \
  -samples-per-epoch "$SAMPLES_PER_EPOCH" \
  -swa-period-samples 100000 \
  -epochs-per-export 5 \
  -export-min-sample-interval 50000 \
  -fixed-val-datadir "$PILOT/evaluations/fixed-validation-g2" \
  -fixed-val-manifest "$PILOT/evaluations/fixed-validation-g2.manifest.json" \
  -max-val-samples 64 \
  -max-train-bucket-per-new-data 4 \
  -max-train-bucket-size 5000000 \
  -no-repeat-files \
  -stop-when-train-bucket-limited \
  -max-epochs-this-instance 1 \
  -quit-if-no-data \
  -extreme-score-only \
  -extreme-score-training-policy "$SRC/python/risk_score/extreme_score_training_policy_v1.json" \
  -extreme-score-cohort-size 1 \
  -generation-provenance-dir "$PILOT/provenance/trainer" \
  -require-shuffle-provenance \
  2>&1 | tee "$PILOT/logs/{short}-trainer.log"

python3 - "$BEFORE_SELECTED" <<'PYADVANCE'
import json
import sys
from pathlib import Path
p=Path("{RUN}/train/extreme-pilot-original/extreme-score-progress.json")
after=json.loads(p.read_text())["selected_training_samples"]
before=int(sys.argv[1])
if after <= before:
    raise SystemExit(f"training made no selected-sample progress: {{before}} -> {{after}}")
print(f"selected training samples advanced: {{before}} -> {{after}}")
PYADVANCE

python3 - <<'PYDONE'
import datetime as dt
import json
import os
from pathlib import Path
pilot = Path("{RUN}")
progress = json.loads(
    (pilot / "train/extreme-pilot-original/extreme-score-progress.json").read_text()
)
value = {{
    "schema_version": 1,
    "contract": "risk-score-extreme-round-complete-v1",
    "generation_id": "{identifier}",
    "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "progress_sha256": progress["progress_sha256"],
    "selected_training_samples": progress["selected_training_samples"],
}}
data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\\n").encode()
path = pilot / "generations/{identifier}/round-complete.json"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(value, sort_keys=True))
PYDONE
"""


def prepare_generation(number: int) -> Path:
    identifier = generation_id(number)
    target = GENERATIONS / identifier
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing existing generation path: {target}")
    target.mkdir(mode=0o755)

    config_name = "extreme-score-selfplay-128v.cfg"
    config_source = TEMPLATE / config_name
    config_target = target / config_name
    immutable_write(config_target, config_source.read_bytes(), 0o444)

    state = load_accepted_state(ACCEPTED_STATE)
    request = json.loads((TEMPLATE / "league-request.json").read_text())
    request["generation_id"] = identifier
    request["output_root"] = str(target / "selfplay")
    request["config"] = {
        "path": str(config_target),
        "sha256": file_sha256(config_target),
    }
    request["accepted_state"] = {
        "path": str(ACCEPTED_STATE),
        "file_sha256": file_sha256(ACCEPTED_STATE),
        "state_sha256": state["state_sha256"],
    }
    for opponent in request["opponents"]:
        snapshot = opponent["model"]["snapshot_id"]
        opponent["model"]["snapshot_id"] = re.sub(
            r"^g[0-9]+-", f"g{number}-", snapshot
        )
    immutable_json(target / "league-request.json", request)

    shuffle_command = [
        str(PYTHON_ROOT / "selfplay/shuffle.sh"),
        str(RUN),
        str(RUN / f"shuffle-scratch-g{number}"),
        "32",
        "-min-rows",
        "1",
    ]
    immutable_json(target / "shuffle-command.json", shuffle_command)

    run_command(
        [
            sys.executable,
            "-m",
            "risk_score.extreme_score_league",
            "plan",
            "--request",
            str(target / "league-request.json"),
            "--policy",
            str(POLICY),
            "--output",
            str(target / "league-plan.json"),
        ]
    )

    script_path = target / "run-round.sh"
    immutable_write(script_path, round_script(number, identifier).encode(), 0o555)
    ready = {
        "schema_version": 1,
        "contract": "risk-score-extreme-generation-ready-v1",
        "watchdog_version": WATCHDOG_VERSION,
        "generation_id": identifier,
        "created_at_utc": utc_now(),
        "accepted_state_sha256": state["state_sha256"],
        "artifacts": {
            name: {
                "path": str(target / name),
                "file_sha256": file_sha256(target / name),
            }
            for name in (
                config_name,
                "league-request.json",
                "league-plan.json",
                "shuffle-command.json",
                "run-round.sh",
            )
        },
    }
    ready["ready_sha256"] = hashlib.sha256(
        canonical_json(ready).encode()
    ).hexdigest()
    immutable_json(target / "generation-ready.json", ready)
    return target


def launch_generation(number: int, target: Path) -> None:
    script = target / "run-round.sh"
    run_command(
        [
            "sudo",
            "-n",
            "systemd-run",
            "--no-block",
            "--quiet",
            "--unit",
            unit_name(number),
            "--property=Type=exec",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=300",
            str(script),
        ]
    )
    launch = {
        "schema_version": 1,
        "contract": "risk-score-extreme-generation-launch-v1",
        "generation_id": generation_id(number),
        "launched_at_utc": utc_now(),
        "unit": f"{unit_name(number)}.service",
        "ready_file_sha256": file_sha256(target / "generation-ready.json"),
    }
    launch["launch_sha256"] = hashlib.sha256(
        canonical_json(launch).encode()
    ).hexdigest()
    immutable_json(target / "generation-launch.json", launch)


def create_and_launch(number: int) -> dict[str, Any]:
    capacity = assert_launch_capacity()
    target = prepare_generation(number)
    launch_generation(number, target)
    return {
        "action": "LAUNCHED",
        "generation_id": generation_id(number),
        "unit": f"{unit_name(number)}.service",
        "capacity": capacity,
    }


def watchdog_once() -> dict[str, Any]:
    if STOP_FILE.exists() or STOP_FILE.is_symlink():
        return {"action": "PAUSED", "reason": f"stop file present: {STOP_FILE}"}
    generations = generation_directories()
    if not generations:
        raise RuntimeError("no template generation exists")
    number, latest = generations[-1]
    service = service_state(number)
    workers = quick_worker_state(latest)
    round_complete = (latest / "round-complete.json").is_file()
    ready = (latest / "generation-ready.json").is_file()
    launched = (latest / "generation-launch.json").is_file()
    active = service.get("ActiveState") == "active"
    worker_outputs_exist = False
    plan_path = latest / "league-plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        worker_outputs_exist = any(
            Path(worker["output_directory"]).exists() for worker in plan["workers"]
        )

    observation = {
        "latest_generation": latest.name,
        "latest_service": service,
        "latest_workers": workers,
        "latest_round_complete": round_complete,
    }
    if active and workers not in {"SUCCEEDED", "FAILED"}:
        return {"action": "MONITORING", **observation}
    if active and workers == "FAILED":
        return {
            "action": "WAITING_FOR_FAILED_UNIT_EXIT",
            "reason": "immutable failed worker receipt",
            **observation,
        }
    if (
        not active
        and ready
        and not launched
        and workers in {"UNPLANNED", "RUNNING"}
        and not worker_outputs_exist
    ):
        launch_generation(number, latest)
        return {
            "action": "LAUNCHED_PREPARED",
            "generation_id": latest.name,
            **observation,
        }
    return {**create_and_launch(number + 1), **observation}


def main() -> int:
    WATCHDOG_ROOT.mkdir(parents=True, exist_ok=True)
    WATCHDOG_LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(WATCHDOG_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(canonical_json({"action": "SKIPPED", "reason": "watchdog lock held"}))
        os.close(descriptor)
        return 0
    status: dict[str, Any]
    try:
        result = watchdog_once()
        status = {
            "schema_version": 1,
            "contract": "risk-score-extreme-pipeline-watchdog-status-v1",
            "watchdog_version": WATCHDOG_VERSION,
            "observed_at_utc": utc_now(),
            "status": "OK",
            **result,
        }
        atomic_status(status)
        print(canonical_json(status))
        return 0
    except Exception as exc:
        status = {
            "schema_version": 1,
            "contract": "risk-score-extreme-pipeline-watchdog-status-v1",
            "watchdog_version": WATCHDOG_VERSION,
            "observed_at_utc": utc_now(),
            "status": "ERROR",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        atomic_status(status)
        print(canonical_json(status), file=sys.stderr)
        return 1
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
