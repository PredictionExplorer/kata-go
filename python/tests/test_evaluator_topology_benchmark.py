import hashlib
import json
import sys
from pathlib import Path

import pytest

from risk_score.autonomy_bootstrap import select_evaluator_topology
from risk_score.evaluator_topology_benchmark import (
    BENCHMARK_RECEIPT_CONTRACT,
    COMPLETION_RECEIPT_CONTRACT,
    GATE_EVIDENCE_CONTRACT,
    PROCESS_COUNTS,
    SPEC_CONTRACT,
    BenchmarkConflictError,
    BenchmarkExecutionError,
    BenchmarkSpecError,
    EvaluatorTopologyBenchmark,
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_benchmark_spec,
    main,
    run_benchmark_gate,
)


FAKE_BENCHMARK = r"""#!__PYTHON__
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--process-count", required=True, type=int)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--work-root", required=True, type=Path)
parser.add_argument("--rates", required=True)
parser.add_argument("--mode", required=True)
args = parser.parse_args()

if not args.output.is_absolute() or not args.work_root.is_absolute():
    raise SystemExit("paths are not absolute")
if args.output.parent != args.work_root:
    raise SystemExit("receipt is outside the isolated work root")
if args.mode == "failure" and args.process_count == 8:
    print("intentional benchmark failure", file=sys.stderr)
    raise SystemExit(7)
if args.mode == "timeout":
    time.sleep(10)
if args.mode == "underreport-time":
    time.sleep(1.0)
if args.mode == "mutate-input":
    Path(__file__).write_text(
        Path(__file__).read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )

rates = {
    int(pair.split(":", 1)[0]): float(pair.split(":", 1)[1])
    for pair in args.rates.split(",")
}
completed = 10
elapsed = completed / rates[args.process_count]
if args.mode == "underreport-time":
    elapsed = 0.001
else:
    time.sleep(elapsed)
payload = b"deterministic evaluator output\n" * completed
if args.mode == "determinism-mismatch":
    payload = (
        f"topology={args.process_count}\n".encode("ascii")
        + b"deterministic evaluator output\n" * (completed - 1)
    )

artifact = args.work_root / "artifacts" / "result.bin"
artifact.parent.mkdir()
if args.mode == "symlink-output":
    artifact.symlink_to(Path(__file__))
    artifact_data = Path(__file__).read_bytes()
else:
    artifact.write_bytes(payload)
    artifact_data = payload

artifact_sha256 = hashlib.sha256(artifact_data).hexdigest()
if args.mode == "bad-artifact-hash":
    artifact_sha256 = "0" * 64
manifest = [
    {
        "path": "artifacts/result.bin",
        "sha256": artifact_sha256,
        "size_bytes": len(artifact_data),
        "row_count": completed,
    }
]
receipt = {
    "schema_version": 1,
    "contract": "__RECEIPT_CONTRACT__",
    "process_count": (
        args.process_count + 1
        if args.mode == "wrong-process-count"
        else args.process_count
    ),
    "completed_work_count": completed,
    "elapsed_seconds": elapsed,
    "output_manifest": manifest,
    "output_manifest_sha256": canonical_sha256(manifest),
}
receipt["receipt_sha256"] = canonical_sha256(receipt)
args.output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
"""


def write_canonical(path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def make_fixture(
    tmp_path,
    *,
    mode="success",
    rates=(10, 20, 15),
    timeout_seconds=10,
    evidence_output=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake = tmp_path / "fake-benchmark"
    fake.write_text(
        FAKE_BENCHMARK.replace("__PYTHON__", str(Path(sys.executable).resolve()))
        .replace("__RECEIPT_CONTRACT__", BENCHMARK_RECEIPT_CONTRACT),
        encoding="utf-8",
    )
    fake.chmod(0o700)
    fake = fake.resolve()

    work_root = (tmp_path / "topology-work").resolve()
    evidence = (
        (work_root / "gate-evidence.json").resolve()
        if evidence_output is None
        else Path(evidence_output).resolve()
    )
    rate_argument = ",".join(
        f"{count}:{rate}" for count, rate in zip(PROCESS_COUNTS, rates, strict=True)
    )
    spec_value = {
        "schema_version": 1,
        "contract": SPEC_CONTRACT,
        "benchmark_argv_template": [
            str(fake),
            "--process-count",
            "{process_count}",
            "--output",
            "{output}",
            "--work-root",
            "{work_root}",
            "--rates",
            rate_argument,
            "--mode",
            mode,
        ],
        "inputs": [{"path": str(fake), "sha256": file_sha256(fake)}],
        "timeout_seconds": timeout_seconds,
        "work_root": str(work_root),
        "evidence_output": str(evidence),
    }
    spec_value["spec_sha256"] = canonical_sha256(spec_value)
    spec_path = (tmp_path / "topology-spec.json").resolve()
    write_canonical(spec_path, spec_value)
    return {
        "fake": fake,
        "work_root": work_root,
        "evidence": evidence,
        "spec_path": spec_path,
        "spec_value": spec_value,
        "spec_file_sha256": file_sha256(spec_path),
    }


def assert_no_partial_publication(fixture):
    assert not fixture["evidence"].exists()
    assert not (
        fixture["work_root"]
        / ".evaluator-topology-benchmark-completion.json"
    ).exists()
    if fixture["work_root"].exists():
        assert not any(
            child.name.startswith(".evaluator-topology-run-")
            for child in fixture["work_root"].iterdir()
        )


def test_runs_two_isolated_repetitions_and_publishes_generic_evidence(tmp_path):
    fixture = make_fixture(tmp_path)
    spec = load_benchmark_spec(
        fixture["spec_path"],
        expected_spec_sha256=fixture["spec_file_sha256"],
    )

    evidence = EvaluatorTopologyBenchmark(spec).run()

    assert evidence["contract"] == GATE_EVIDENCE_CONTRACT
    assert evidence["gate_id"] == "evaluator-topology-benchmark"
    assert evidence["decision"] == "PASS"
    assert evidence["evidence_sha256"] == canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    benchmarks = evidence["checks"]["benchmarks"]
    assert [row["process_count"] for row in benchmarks] == [4, 8, 16]
    assert benchmarks[1]["throughput_per_second"] > benchmarks[0][
        "throughput_per_second"
    ]
    assert benchmarks[1]["throughput_per_second"] > benchmarks[2][
        "throughput_per_second"
    ]
    assert len({row["output_sha256"] for row in benchmarks}) == 1
    assert all(
        row["output_sha256"] == row["repeat_output_sha256"]
        for row in benchmarks
    )
    assert evidence["checks"]["selected_process_count"] == 8
    assert select_evaluator_topology(benchmarks) == 8

    assert fixture["evidence"].read_bytes() == (
        canonical_json(evidence) + "\n"
    ).encode("utf-8")
    completion_path = (
        fixture["work_root"]
        / ".evaluator-topology-benchmark-completion.json"
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["contract"] == COMPLETION_RECEIPT_CONTRACT
    assert len(completion["repetitions"]) == 6
    assert [
        (row["process_count"], row["repetition"])
        for row in completion["repetitions"]
    ] == [(4, 1), (4, 2), (8, 1), (8, 2), (16, 1), (16, 2)]
    assert not any(
        child.name.startswith(".evaluator-topology-run-")
        for child in fixture["work_root"].iterdir()
    )


def test_replays_valid_completion_and_recovers_missing_generic_evidence(tmp_path):
    fixture = make_fixture(tmp_path)
    spec = load_benchmark_spec(fixture["spec_path"])
    expected = EvaluatorTopologyBenchmark(spec).run()
    fixture["evidence"].unlink()
    fixture["fake"].chmod(0o600)

    replayed = EvaluatorTopologyBenchmark(spec).run()

    assert replayed == expected
    assert json.loads(fixture["evidence"].read_text(encoding="utf-8")) == expected


def test_replay_recovers_owned_temporary_hard_link(tmp_path):
    fixture = make_fixture(tmp_path)
    spec = load_benchmark_spec(fixture["spec_path"])
    expected = EvaluatorTopologyBenchmark(spec).run()
    completion = (
        fixture["work_root"]
        / ".evaluator-topology-benchmark-completion.json"
    )
    orphan = completion.with_name(
        f".{completion.name}.crash.tmp"
    )
    orphan.hardlink_to(completion)
    fixture["evidence"].unlink()

    replayed = EvaluatorTopologyBenchmark(spec).run()

    assert replayed == expected
    assert not orphan.exists()
    assert completion.stat().st_nlink == 1


def test_determinism_mismatch_never_publishes_pass_evidence(tmp_path):
    fixture = make_fixture(tmp_path, mode="determinism-mismatch")

    with pytest.raises(BenchmarkExecutionError, match="changed across process counts"):
        run_benchmark_gate(fixture["spec_path"])

    assert_no_partial_publication(fixture)


@pytest.mark.parametrize(
    ("mode", "timeout_seconds", "message"),
    [
        ("failure", 10, "exit code 7"),
        ("timeout", 0.05, "timed out"),
        ("underreport-time", 10, "independently measured command lifetime"),
    ],
)
def test_timeout_and_command_failure_are_bounded_and_clean(
    tmp_path, mode, timeout_seconds, message
):
    fixture = make_fixture(
        tmp_path,
        mode=mode,
        timeout_seconds=timeout_seconds,
    )

    with pytest.raises(BenchmarkExecutionError, match=message):
        run_benchmark_gate(fixture["spec_path"])

    assert_no_partial_publication(fixture)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("bad-artifact-hash", "artifact hash is invalid"),
        ("symlink-output", "metadata is invalid"),
        ("wrong-process-count", "process_count does not match"),
    ],
)
def test_receipt_artifacts_and_invocation_bindings_are_independently_verified(
    tmp_path, mode, message
):
    fixture = make_fixture(tmp_path, mode=mode)

    with pytest.raises(BenchmarkExecutionError, match=message):
        run_benchmark_gate(fixture["spec_path"])

    assert_no_partial_publication(fixture)


def test_hash_bound_input_mutation_is_detected_before_publication(tmp_path):
    fixture = make_fixture(tmp_path, mode="mutate-input")

    with pytest.raises(BenchmarkSpecError, match="hash-bound input changed"):
        run_benchmark_gate(fixture["spec_path"])

    assert_no_partial_publication(fixture)


def test_spec_rejects_noncanonical_unsafe_and_symlinked_paths(tmp_path):
    outside = (tmp_path / "outside-evidence.json").resolve()
    fixture = make_fixture(tmp_path, evidence_output=outside)
    with pytest.raises(BenchmarkSpecError, match="strictly beneath work_root"):
        load_benchmark_spec(fixture["spec_path"])

    fixture = make_fixture(tmp_path / "noncanonical")
    value = fixture["spec_value"]
    fixture["spec_path"].write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(BenchmarkSpecError, match="canonical"):
        load_benchmark_spec(fixture["spec_path"])

    target_root = (tmp_path / "real-root").resolve()
    target_root.mkdir()
    symlink_root = tmp_path / "linked-root"
    symlink_root.symlink_to(target_root, target_is_directory=True)
    fixture = make_fixture(
        tmp_path / "symlink-case",
        evidence_output=target_root / "gate-evidence.json",
    )
    value = dict(fixture["spec_value"])
    value["work_root"] = str(symlink_root)
    value["evidence_output"] = str(symlink_root / "gate-evidence.json")
    value.pop("spec_sha256")
    value["spec_sha256"] = canonical_sha256(value)
    write_canonical(fixture["spec_path"], value)
    with pytest.raises(BenchmarkSpecError, match="symlink"):
        load_benchmark_spec(fixture["spec_path"])


def test_existing_unbound_evidence_and_stale_attempt_fail_closed(tmp_path):
    fixture = make_fixture(tmp_path)
    fixture["work_root"].mkdir()
    write_canonical(fixture["evidence"], {"untrusted": True})

    with pytest.raises(BenchmarkConflictError, match="without.*completion"):
        run_benchmark_gate(fixture["spec_path"])

    fixture["evidence"].unlink()
    stale = fixture["work_root"] / ".evaluator-topology-run-abandoned"
    stale.mkdir()
    with pytest.raises(BenchmarkConflictError, match="explicit cleanup"):
        run_benchmark_gate(fixture["spec_path"])
    assert stale.is_dir()


def test_topology_selection_uses_authoritative_bootstrap_function(tmp_path):
    fixture = make_fixture(tmp_path, rates=(10, 21, 20))

    evidence = run_benchmark_gate(fixture["spec_path"])
    benchmarks = evidence["checks"]["benchmarks"]

    assert evidence["checks"]["selected_process_count"] == 8
    assert select_evaluator_topology(benchmarks) == 8


def test_cli_requires_and_checks_complete_spec_file_hash(tmp_path, capsys):
    fixture = make_fixture(tmp_path)

    assert (
        main(
            [
                "--spec",
                str(fixture["spec_path"]),
                "--expected-spec-sha256",
                fixture["spec_file_sha256"],
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["checks"]["selected_process_count"] == 8

    second = make_fixture(tmp_path / "wrong-hash")
    assert (
        main(
            [
                "--spec",
                str(second["spec_path"]),
                "--expected-spec-sha256",
                hashlib.sha256(b"wrong").hexdigest(),
            ]
        )
        == 2
    )
    assert "does not match expected" in capsys.readouterr().err
