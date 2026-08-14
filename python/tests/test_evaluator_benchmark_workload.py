import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from risk_score.evaluator_benchmark_workload import (
    BENCHMARK_RECEIPT_CONTRACT,
    PROCESS_COUNTS,
    WorkloadConflictError,
    WorkloadExecutionError,
    WorkloadSpecError,
    benchmark_input_bindings,
    build_benchmark_argv_template,
    canonical_json,
    canonical_sha256,
    publish_topology_specs,
    publish_workload_spec,
    run_workload,
)
from risk_score.evaluator_topology_benchmark import (
    BenchmarkExecutionError,
    run_benchmark_gate,
)


DETERMINISTIC_CONFIG = """\
forDeterministicTesting = true
numAnalysisThreads = 1
numSearchThreadsPerAnalysisThread = 1
nnRandomize = false
rootNoiseEnabled = false
rootNumSymmetriesToSample = 1
useUncertainty = false
cpuctUtilityStdevScale = 0
reportAnalysisWinratesAs = sidetomove
"""


FAKE_KATAGO = r"""#!__PYTHON__
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODE = __MODE__
GPU_INDEX = __GPU_INDEX__
GPU_STATE = Path(__GPU_STATE__)
PID_LOG = Path(__PID_LOG__)
ORPHAN_LOG = Path(__ORPHAN_LOG__)

if (
    len(sys.argv) != 6
    or sys.argv[1] != "analysis"
    or sys.argv[2] != "-config"
    or sys.argv[4] != "-model"
):
    print("unexpected KataGo argv", file=sys.stderr)
    raise SystemExit(11)
if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
    print("wrong CUDA_VISIBLE_DEVICES", file=sys.stderr)
    raise SystemExit(12)

pid = os.getpid()
marker = GPU_STATE / f"{pid}.pid"
marker.write_text("katago\n", encoding="utf-8")
with PID_LOG.open("a", encoding="utf-8") as stream:
    stream.write(f"{pid},{os.getpgrp()}\n")
try:
    rows = [json.loads(line) for line in sys.stdin if line.strip()]
    time.sleep(0.12)
    if MODE == "failure" and any(row["id"] == "query-000" for row in rows):
        print("intentional analysis failure", file=sys.stderr)
        raise SystemExit(9)
    if MODE in {"timeout", "orphan"}:
        orphan = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        with ORPHAN_LOG.open("a", encoding="utf-8") as stream:
            stream.write(f"{orphan.pid}\n")
    if MODE == "timeout":
        time.sleep(30)

    for row in reversed(rows):
        response = {
            "id": row["id"],
            "echo": row["id"],
            "rootInfo": {"visits": row["maxVisits"]},
            "moveInfos": [{"move": "pass", "visits": row["maxVisits"]}],
            "turnNumber": len(row["moves"]),
            "isDuringSearch": MODE == "streaming",
        }
        if MODE == "missing-result-field":
            response.pop("moveInfos")
        copies = 2 if MODE == "multi-result" else 1
        for _copy in range(copies):
            print(
                json.dumps(
                    response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
finally:
    marker.unlink(missing_ok=True)
"""


FAKE_NVIDIA_SMI = r"""#!__PYTHON__
import os
import sys
from pathlib import Path

INDEX = __GPU_INDEX__
UUID = __OBSERVED_UUID__
FOREIGN = __FOREIGN__
REPORT_CHILDREN = __REPORT_CHILDREN__
MIG = __MIG__
MIG_TARGET = __MIG_TARGET__
MIG_MODERN = __MIG_MODERN__
MIG_MAPPING = __MIG_MAPPING__
GPU_STATE = Path(__GPU_STATE__)

parent = UUID if MIG_TARGET else "GPU-other-mig"
mig_uuid = (
    ("MIG-modern-target" if MIG_TARGET else "MIG-modern-other")
    if MIG_MODERN
    else f"MIG-{parent}/1/2"
)
has_other_gpu = MIG and (not MIG_TARGET or MIG_MAPPING == "ambiguous")

if "--query-gpu=index,uuid" in sys.argv:
    print(f"{INDEX}, {UUID}")
    if has_other_gpu:
        print(f"{INDEX + 1}, GPU-other-mig")
elif "-L" in sys.argv:
    print(f"GPU {INDEX}: NVIDIA H100 (UUID: {UUID})")
    if MIG_MAPPING == "malformed":
        print("  malformed MIG topology row")
    elif MIG and (MIG_TARGET or MIG_MAPPING == "ambiguous"):
        print(f"  MIG 1g.10gb Device 0: (UUID: {mig_uuid})")
    if has_other_gpu:
        print(f"GPU {INDEX + 1}: NVIDIA H100 (UUID: GPU-other-mig)")
        if MIG and (not MIG_TARGET or MIG_MAPPING == "ambiguous"):
            print(f"  MIG 1g.10gb Device 0: (UUID: {mig_uuid})")
elif "--query-compute-apps=gpu_uuid,pid,process_name" in sys.argv:
    if MIG:
        print(f"{mig_uuid}, 888888, mig-process")
    if REPORT_CHILDREN:
        for marker in sorted(GPU_STATE.glob("*.pid")):
            pid = int(marker.stem)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                marker.unlink(missing_ok=True)
            else:
                print(f"{UUID}, {pid}, katago")
    if FOREIGN:
        print(f"{UUID}, 999999, foreign-process")
else:
    print("unsupported nvidia-smi query", file=sys.stderr)
    raise SystemExit(3)
"""

def make_fixture(
    tmp_path,
    *,
    mode="success",
    observed_uuid="GPU-test-0007",
    expected_uuid="GPU-test-0007",
    foreign=False,
    report_children=True,
    mig=False,
    mig_target=False,
    mig_modern=False,
    mig_mapping="valid",
    timeout_seconds=5,
    query_mutator=None,
    config_text=DETERMINISTIC_CONFIG,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    gpu_index = 7
    gpu_state = (tmp_path / "gpu-state").resolve()
    gpu_state.mkdir()
    pid_log = (tmp_path / "child-pids.log").resolve()
    orphan_log = (tmp_path / "orphan-pids.log").resolve()
    katago = (tmp_path / "katago").resolve()
    katago.write_text(
        FAKE_KATAGO.replace(
            "__PYTHON__", str(Path(sys.executable).resolve())
        )
        .replace("__MODE__", repr(mode))
        .replace("__GPU_INDEX__", str(gpu_index))
        .replace("__GPU_STATE__", repr(str(gpu_state)))
        .replace("__PID_LOG__", repr(str(pid_log)))
        .replace("__ORPHAN_LOG__", repr(str(orphan_log))),
        encoding="utf-8",
    )
    katago.chmod(0o700)
    nvidia_smi = (tmp_path / "nvidia-smi").resolve()
    nvidia_smi.write_text(
        FAKE_NVIDIA_SMI.replace(
            "__PYTHON__", str(Path(sys.executable).resolve())
        )
        .replace("__GPU_INDEX__", str(gpu_index))
        .replace("__OBSERVED_UUID__", repr(observed_uuid))
        .replace("__FOREIGN__", repr(foreign))
        .replace("__REPORT_CHILDREN__", repr(report_children))
        .replace("__MIG__", repr(mig))
        .replace("__MIG_TARGET__", repr(mig_target))
        .replace("__MIG_MODERN__", repr(mig_modern))
        .replace("__MIG_MAPPING__", repr(mig_mapping))
        .replace("__GPU_STATE__", repr(str(gpu_state))),
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o700)
    config = (tmp_path / "analysis.cfg").resolve()
    config.write_text(config_text, encoding="utf-8")
    model = (tmp_path / "model.bin.gz").resolve()
    model.write_bytes(b"frozen-model")

    # Deliberately not ID-sorted, so a merge-by-ID implementation fails.
    order = list(range(0, 32, 2)) + list(range(1, 32, 2))
    rows = [
        {
            "id": f"query-{index:03d}",
            "moves": [],
            "initialStones": [],
            "initialPlayer": "B",
            "rules": "tromp-taylor",
            "komi": 7.5,
            "boardXSize": 19,
            "boardYSize": 19,
            "includePolicy": True,
            "maxVisits": 8,
            "overrideSettings": {
                "useScoreMaximizingUtility": bool(position % 2),
                "scorePower": 1.5,
                "scoreScale": 20.0,
                "winWeight": 4.0,
                "rootNoiseEnabled": False,
                "rootNumSymmetriesToSample": 1,
            },
        }
        for position, index in enumerate(order)
    ]
    if query_mutator is not None:
        query_mutator(rows[0])
    queries = (tmp_path / "queries.jsonl").resolve()
    queries.write_bytes(
        b"".join(
            (canonical_json(row) + "\n").encode("utf-8")
            for row in rows
        )
    )
    spec_path = (tmp_path / "workload-spec.json").resolve()
    spec = publish_workload_spec(
        spec_path,
        katago=katago,
        analysis_config=config,
        model=model,
        queries=queries,
        nvidia_smi=nvidia_smi,
        gpu_index=gpu_index,
        expected_gpu_uuid=expected_uuid,
        row_count=len(rows),
        max_visits=8,
        timeout_seconds=timeout_seconds,
    )
    return {
        "katago": katago,
        "nvidia_smi": nvidia_smi,
        "config": config,
        "model": model,
        "queries": queries,
        "rows": rows,
        "gpu_state": gpu_state,
        "pid_log": pid_log,
        "orphan_log": orphan_log,
        "spec_path": spec_path,
        "spec": spec,
    }


def run_once(fixture, tmp_path, process_count):
    root = (tmp_path / f"run-p{process_count}").resolve()
    root.mkdir()
    output = root / "benchmark-receipt.json"
    receipt = run_workload(
        fixture["spec"],
        process_count=process_count,
        output=output,
        work_root=root,
        expected_spec_sha256=fixture["spec"].file_sha256,
    )
    return root, output, receipt


def assert_processes_exit(path, timeout=3):
    pids = [
        int(line.split(",", 1)[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert pids
    deadline = time.monotonic() + timeout
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        for pid in tuple(remaining):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                remaining.remove(pid)
        if remaining:
            time.sleep(0.02)
    assert not remaining


def test_p4_p8_p16_outputs_are_bit_identical_and_in_query_order(tmp_path):
    fixture = make_fixture(tmp_path / "fixture")
    results = [
        run_once(fixture, tmp_path, process_count)
        for process_count in PROCESS_COUNTS
    ]

    manifests = [receipt["output_manifest"] for _, _, receipt in results]
    assert manifests[0] == manifests[1] == manifests[2]
    assert len(
        {receipt["output_manifest_sha256"] for _, _, receipt in results}
    ) == 1
    artifact_bytes = [
        (root / "artifacts" / "analysis.jsonl").read_bytes()
        for root, _, _ in results
    ]
    assert artifact_bytes[0] == artifact_bytes[1] == artifact_bytes[2]
    output_rows = [
        json.loads(line)
        for line in artifact_bytes[0].decode("utf-8").splitlines()
    ]
    assert [row["id"] for row in output_rows] == [
        row["id"] for row in fixture["rows"]
    ]
    assert [row["echo"] for row in output_rows] == [
        row["id"] for row in fixture["rows"]
    ]
    assert all(
        receipt["contract"] == BENCHMARK_RECEIPT_CONTRACT
        and receipt["completed_work_count"] == len(fixture["rows"])
        and receipt["elapsed_seconds"] > 0
        for _, _, receipt in results
    )
    assert len(
        {receipt["elapsed_seconds"] for _, _, receipt in results}
    ) == len(PROCESS_COUNTS)
    assert all(
        not (root / ".evaluator-benchmark-workload-attempt").exists()
        for root, _, _ in results
    )
    process_groups = {
        int(line.split(",", 1)[1])
        for line in fixture["pid_log"].read_text(
            encoding="utf-8"
        ).splitlines()
    }
    assert process_groups == {os.getpgrp()}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update({"action": "terminate"}),
        lambda row: row.update({"maxTime": 1.0}),
        lambda row: row.update({"reportDuringSearchEvery": 0.1}),
        lambda row: row.update({"priority": 99}),
        lambda row: row.update({"terminateId": "other"}),
        lambda row: row.update({"analyzeTurns": [0]}),
        lambda row: row["overrideSettings"].update(
            {"rootNoiseEnabled": True}
        ),
        lambda row: row["overrideSettings"].update(
            {"unknownOverride": 1}
        ),
    ],
)
def test_rejects_control_streaming_and_nondeterministic_queries(
    tmp_path, mutation
):
    with pytest.raises(WorkloadSpecError):
        make_fixture(
            tmp_path / f"invalid-{id(mutation)}",
            query_mutator=mutation,
        )


def test_requires_single_search_thread_in_config(tmp_path):
    unsafe = DETERMINISTIC_CONFIG.replace(
        "numSearchThreadsPerAnalysisThread = 1\n", ""
    )
    with pytest.raises(
        WorkloadSpecError, match="numSearchThreadsPerAnalysisThread"
    ):
        make_fixture(tmp_path / "unsafe-config", config_text=unsafe)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("streaming", "final normal analysis"),
        ("multi-result", "duplicate"),
        ("missing-result-field", "final normal analysis"),
    ],
)
def test_rejects_streaming_multi_result_and_incomplete_results(
    tmp_path, mode, message
):
    fixture = make_fixture(tmp_path / mode, mode=mode)
    root = (tmp_path / f"{mode}-run").resolve()
    root.mkdir()
    with pytest.raises(WorkloadExecutionError, match=message):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=root / "receipt.json",
            work_root=root,
        )


def test_requires_positive_gpu_attribution_and_ignores_unrelated_mig(tmp_path):
    empty = make_fixture(
        tmp_path / "empty-attribution", report_children=False
    )
    empty_root = (tmp_path / "empty-run").resolve()
    empty_root.mkdir()
    with pytest.raises(
        WorkloadExecutionError, match="positive GPU attribution"
    ):
        run_workload(
            empty["spec"],
            process_count=4,
            output=empty_root / "receipt.json",
            work_root=empty_root,
        )

    mig = make_fixture(tmp_path / "unrelated-mig", mig=True)
    root, _, receipt = run_once(mig, tmp_path, 4)
    assert receipt["completed_work_count"] == len(mig["rows"])
    assert (root / "artifacts" / "analysis.jsonl").is_file()

    target_mig = make_fixture(
        tmp_path / "target-mig", mig=True, mig_target=True
    )
    target_root = (tmp_path / "target-mig-run").resolve()
    target_root.mkdir()
    with pytest.raises(
        WorkloadExecutionError, match="target CUDA GPU.*MIG"
    ):
        run_workload(
            target_mig["spec"],
            process_count=4,
            output=target_root / "receipt.json",
            work_root=target_root,
        )


def test_maps_modern_opaque_mig_rows_to_physical_gpus(tmp_path):
    unrelated = make_fixture(
        tmp_path / "modern-unrelated",
        mig=True,
        mig_modern=True,
    )
    root, _, receipt = run_once(unrelated, tmp_path, 4)
    assert receipt["completed_work_count"] == len(unrelated["rows"])
    assert (root / "artifacts" / "analysis.jsonl").is_file()

    target = make_fixture(
        tmp_path / "modern-target",
        mig=True,
        mig_target=True,
        mig_modern=True,
    )
    target_root = (tmp_path / "modern-target-run").resolve()
    target_root.mkdir()
    with pytest.raises(
        WorkloadExecutionError, match="target CUDA GPU.*MIG"
    ):
        run_workload(
            target["spec"],
            process_count=4,
            output=target_root / "receipt.json",
            work_root=target_root,
        )


@pytest.mark.parametrize(
    ("mapping_mode", "message"),
    [
        ("malformed", "topology inventory is malformed"),
        ("ambiguous", "maps one MIG UUID more than once"),
    ],
)
def test_rejects_malformed_or_ambiguous_mig_mapping(
    tmp_path, mapping_mode, message
):
    fixture = make_fixture(
        tmp_path / mapping_mode,
        mig=True,
        mig_modern=True,
        mig_mapping=mapping_mode,
    )
    root = (tmp_path / f"{mapping_mode}-run").resolve()
    root.mkdir()
    with pytest.raises(WorkloadExecutionError, match=message):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=root / "receipt.json",
            work_root=root,
        )


def test_hash_drift_and_wrong_gpu_fail_closed(tmp_path):
    drift = make_fixture(tmp_path / "drift")
    drift["model"].write_bytes(b"changed-model")
    drift_root = (tmp_path / "drift-run").resolve()
    drift_root.mkdir()
    with pytest.raises(WorkloadSpecError, match="hash-bound input changed"):
        run_workload(
            drift["spec"],
            process_count=4,
            output=drift_root / "receipt.json",
            work_root=drift_root,
        )
    assert list(drift_root.iterdir()) == []

    wrong_gpu = make_fixture(
        tmp_path / "wrong-gpu",
        observed_uuid="GPU-other-0008",
    )
    gpu_root = (tmp_path / "gpu-run").resolve()
    gpu_root.mkdir()
    with pytest.raises(WorkloadExecutionError, match="UUID differs"):
        run_workload(
            wrong_gpu["spec"],
            process_count=4,
            output=gpu_root / "receipt.json",
            work_root=gpu_root,
        )
    assert list(gpu_root.iterdir()) == []

    foreign = make_fixture(tmp_path / "foreign-gpu", foreign=True)
    foreign_root = (tmp_path / "foreign-run").resolve()
    foreign_root.mkdir()
    with pytest.raises(WorkloadExecutionError, match="foreign compute"):
        run_workload(
            foreign["spec"],
            process_count=4,
            output=foreign_root / "receipt.json",
            work_root=foreign_root,
        )
    assert list(foreign_root.iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "timeout_seconds", "message"),
    [
        ("failure", 5, "status 9"),
        ("timeout", 1.0, "timed out|timeout"),
    ],
)
def test_child_failure_and_timeout_preserve_forensics(
    tmp_path, mode, timeout_seconds, message
):
    fixture = make_fixture(
        tmp_path / mode,
        mode=mode,
        timeout_seconds=timeout_seconds,
    )
    root = (tmp_path / f"{mode}-run").resolve()
    root.mkdir()
    output = root / "receipt.json"
    with pytest.raises(WorkloadExecutionError, match=message):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=output,
            work_root=root,
        )
    attempt = root / ".evaluator-benchmark-workload-attempt"
    assert attempt.is_dir()
    assert sorted((attempt / "queries").glob("*.jsonl"))
    assert sorted((attempt / "responses").glob("*.raw.jsonl"))
    assert sorted((attempt / "stderr").glob("*.stderr"))
    assert not output.exists()
    if mode == "timeout":
        assert_processes_exit(fixture["orphan_log"])


def test_detects_and_kills_orphaned_child_descendants(tmp_path):
    fixture = make_fixture(tmp_path / "orphan", mode="orphan")
    root = (tmp_path / "orphan-run").resolve()
    root.mkdir()
    with pytest.raises(
        WorkloadExecutionError, match="descendants remained"
    ):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=root / "receipt.json",
            work_root=root,
        )
    assert_processes_exit(fixture["orphan_log"])


def test_path_escape_and_symlink_inputs_are_rejected(tmp_path):
    fixture = make_fixture(tmp_path / "fixture")
    linked_queries = (tmp_path / "linked-queries.jsonl").resolve(
        strict=False
    )
    linked_queries.symlink_to(fixture["queries"])
    with pytest.raises(WorkloadSpecError, match="symlink"):
        publish_workload_spec(
            (tmp_path / "linked-spec.json").resolve(),
            katago=fixture["katago"],
            analysis_config=fixture["config"],
            model=fixture["model"],
            queries=linked_queries,
            nvidia_smi=fixture["nvidia_smi"],
            gpu_index=7,
            expected_gpu_uuid="GPU-test-0007",
            row_count=32,
            max_visits=8,
            timeout_seconds=5,
        )

    root = (tmp_path / "safe-root").resolve()
    root.mkdir()
    outside = (tmp_path / "outside-receipt.json").resolve()
    with pytest.raises(WorkloadSpecError, match="strictly beneath"):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=outside,
            work_root=root,
        )

    target = (tmp_path / "real-root").resolve()
    target.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(WorkloadSpecError, match="symlink"):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=linked_root / "receipt.json",
            work_root=linked_root,
        )


def test_exact_replay_succeeds_and_conflicting_replay_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path / "fixture")
    root, output, receipt = run_once(fixture, tmp_path, 4)
    receipt_bytes = output.read_bytes()
    artifact_bytes = (root / "artifacts" / "analysis.jsonl").read_bytes()

    replay = run_workload(
        fixture["spec"],
        process_count=4,
        output=output,
        work_root=root,
    )
    assert replay == receipt
    assert output.read_bytes() == receipt_bytes
    assert (root / "artifacts" / "analysis.jsonl").read_bytes() == artifact_bytes

    conflicting = dict(receipt)
    conflicting["elapsed_seconds"] = receipt["elapsed_seconds"] + 1
    # Deliberately retain the old self-hash: canonical bytes exist, but are not
    # the exact validated replay state.
    output.write_text(canonical_json(conflicting) + "\n", encoding="utf-8")
    with pytest.raises(WorkloadConflictError, match="self-hash"):
        run_workload(
            fixture["spec"],
            process_count=4,
            output=output,
            work_root=root,
        )


def test_cli_template_is_accepted_by_generic_topology_runner(tmp_path):
    fixture = make_fixture(tmp_path / "fixture")
    adapter = (
        Path(__file__).resolve().parents[1]
        / "risk_score"
        / "evaluator_benchmark_workload.py"
    ).resolve()
    python = Path(sys.executable).resolve()
    work_root = (tmp_path / "topology-work").resolve()
    evidence = work_root / "evidence.json"
    workload_path = (tmp_path / "provisioned-workload.json").resolve()
    outer_path = (tmp_path / "topology-spec.json").resolve()
    publication = publish_topology_specs(
        workload_spec_path=workload_path,
        benchmark_spec_path=outer_path,
        python_executable=python,
        adapter_path=adapter,
        katago_binary=fixture["katago"],
        models=[fixture["spec"].model.spec_value()],
        model_probe_config=fixture["config"],
        process_counts=PROCESS_COUNTS,
        work_root=work_root,
        evidence_output=evidence,
        timeout_seconds=30,
        queries=fixture["queries"],
        nvidia_smi=fixture["nvidia_smi"],
        gpu_index=7,
        expected_gpu_uuid="GPU-test-0007",
        row_count=len(fixture["rows"]),
        max_visits=8,
    )
    assert publication["workload_spec"]["path"] == str(workload_path)
    assert publication["benchmark_spec"]["path"] == str(outer_path)
    outer_value = json.loads(outer_path.read_text(encoding="utf-8"))
    workload_value = json.loads(
        workload_path.read_text(encoding="utf-8")
    )
    input_hashes = {
        row["path"]: row["sha256"] for row in outer_value["inputs"]
    }
    argv = outer_value["benchmark_argv_template"]
    assert Path(argv[1]).is_absolute() and Path(argv[3]).is_absolute()
    assert argv[1] == str(adapter)
    assert argv[3] == str(workload_path)
    assert input_hashes[str(adapter)] == hashlib.sha256(
        adapter.read_bytes()
    ).hexdigest()
    assert input_hashes[str(workload_path)] == hashlib.sha256(
        workload_path.read_bytes()
    ).hexdigest()
    assert workload_value["timeout_seconds"] < outer_value["timeout_seconds"]

    result = run_benchmark_gate(
        outer_path,
        expected_spec_sha256=hashlib.sha256(
            outer_path.read_bytes()
        ).hexdigest(),
    )

    benchmarks = result["checks"]["benchmarks"]
    assert [row["process_count"] for row in benchmarks] == [4, 8, 16]
    assert len({row["output_sha256"] for row in benchmarks}) == 1
    assert all(
        row["output_sha256"] == row["repeat_output_sha256"]
        for row in benchmarks
    )
    assert json.loads(evidence.read_text(encoding="utf-8")) == result


def test_outer_timeout_kills_shared_process_group(tmp_path):
    fixture = make_fixture(
        tmp_path / "fixture",
        mode="timeout",
        timeout_seconds=10,
    )
    adapter = (
        Path(__file__).resolve().parents[1]
        / "risk_score"
        / "evaluator_benchmark_workload.py"
    ).resolve()
    python = Path(sys.executable).resolve()
    argv = build_benchmark_argv_template(
        fixture["spec"],
        python_executable=python,
        adapter_path=adapter,
    )
    work_root = (tmp_path / "outer-timeout-work").resolve()
    evidence = work_root / "evidence.json"
    outer = {
        "schema_version": 1,
        "contract": "risk-score-evaluator-topology-benchmark-spec-v1",
        "benchmark_argv_template": list(argv),
        "inputs": list(
            benchmark_input_bindings(
                fixture["spec"],
                python_executable=python,
                adapter_path=adapter,
            )
        ),
        "timeout_seconds": 0.8,
        "work_root": str(work_root),
        "evidence_output": str(evidence),
    }
    outer["spec_sha256"] = canonical_sha256(outer)
    outer_path = (tmp_path / "outer-timeout-spec.json").resolve()
    outer_path.write_text(
        canonical_json(outer) + "\n", encoding="utf-8"
    )

    with pytest.raises(BenchmarkExecutionError, match="timed out"):
        run_benchmark_gate(outer_path)

    assert_processes_exit(fixture["pid_log"])
    assert_processes_exit(fixture["orphan_log"])
    assert not evidence.exists()
