import json
import os
from pathlib import Path

import pytest
from risk_score.extreme_score_evaluation_run import (
    REQUEST_CONTRACT,
    ExtremeScoreEvaluationRunError,
    finalize_spec,
    load_execution_attestation,
    load_spec,
    publish_execution_attestation,
    run_evaluation,
)
from risk_score.extreme_score_evaluator import canonical_sha256, file_sha256

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _request(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "katago"
    config = tmp_path / "match.cfg"
    candidate = tmp_path / "candidate.bin.gz"
    reference = tmp_path / "reference.bin.gz"
    opponent = tmp_path / "opponent.bin.gz"
    for path, payload in (
        (binary, b"binary"),
        (config, b"config"),
        (candidate, b"candidate"),
        (reference, b"reference"),
        (opponent, b"opponent"),
    ):
        path.write_bytes(payload)
    binary.chmod(0o755)
    return {
        "schema_version": 1,
        "contract": REQUEST_CONTRACT,
        "katago_binary": _binding(binary),
        "match_config": _binding(config),
        "focal_models": [_binding(candidate), _binding(reference)],
        "opponent_models": [_binding(opponent)],
        "output_root": str((tmp_path / "runner-output").resolve()),
        "topology": "single-gpu-isolated-match-processes",
        "process_count": 2,
        "gpu": {
            "index": 7,
            "uuid": GPU_UUID,
            "lease_provenance": "lease:test-gpu-7",
        },
    }


def _finalized_spec(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request(tmp_path)), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    value = finalize_spec(request_path, spec_path)
    return value, spec_path


def test_finalized_evaluation_run_spec_is_hash_bound_and_read_only(tmp_path):
    value, spec_path = _finalized_spec(tmp_path)
    assert load_spec(spec_path) == value
    assert spec_path.stat().st_mode & 0o222 == 0
    assert len(value["spec_sha256"]) == 64

    binary = Path(value["katago_binary"]["path"])
    binary.write_bytes(b"changed")
    with pytest.raises(ExtremeScoreEvaluationRunError, match="SHA-256"):
        load_spec(spec_path)


def test_run_wires_bound_spec_into_match_runner_and_evaluator(tmp_path):
    value, spec_path = _finalized_spec(tmp_path)
    observed = {}

    class FakeRunner:
        pass

    runner = FakeRunner()

    def runner_factory(spec):
        observed["runner_spec"] = spec
        return runner

    def evaluator(plan_path, output_path, *, runner, raw_lifetime_records):
        observed["plan_path"] = plan_path
        observed["output_path"] = output_path
        observed["runner"] = runner
        observed["diagnostics"] = raw_lifetime_records
        return {
            "report_sha256": "1" * 64,
            "decision": "INCONCLUSIVE",
            "promotion_recommended": False,
        }

    report = run_evaluation(
        spec_path=spec_path,
        plan_path=tmp_path / "plan.json",
        output_path=tmp_path / "report.json",
        raw_lifetime_records={"record": 10},
        runner_factory=runner_factory,
        evaluator=evaluator,
    )
    assert report["decision"] == "INCONCLUSIVE"
    assert observed["runner"] is runner
    assert observed["runner_spec"].expected_gpu_uuid == GPU_UUID
    assert observed["runner_spec"].gpu_index == 7
    assert observed["runner_spec"].gpu_lease_provenance == "lease:test-gpu-7"
    assert set(observed["runner_spec"].focal_models) == {
        item["sha256"] for item in value["focal_models"]
    }
    assert observed["diagnostics"] == {"record": 10}


def test_writable_or_conflicting_finalized_spec_is_rejected(tmp_path):
    _, spec_path = _finalized_spec(tmp_path)
    os.chmod(spec_path, 0o644)
    with pytest.raises(ExtremeScoreEvaluationRunError, match="read-only"):
        load_spec(spec_path)

    request_path = tmp_path / "other-request.json"
    request = _request(tmp_path / "other")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ExtremeScoreEvaluationRunError, match="conflicts"):
        finalize_spec(request_path, spec_path)


def test_attestation_transitively_binds_arm_and_cell_receipts(tmp_path):
    _, spec_path = _finalized_spec(tmp_path)
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"
    plan_path.write_bytes(b"plan")
    report_path.write_bytes(b"report")
    plan_path.chmod(0o444)
    report_path.chmod(0o444)
    provenance_root = tmp_path / "runner" / "plans" / ("1" * 64)
    arm_receipts = {}
    result_sources = {}
    for arm in ("candidate", "reference"):
        arm_dir = provenance_root / "arms" / arm
        cell_dir = provenance_root / "cells" / arm / "cell-a"
        arm_dir.mkdir(parents=True)
        cell_dir.mkdir(parents=True)
        results = arm_dir / "results.jsonl"
        arm_receipt = arm_dir / "receipt.json"
        cell_receipt = cell_dir / "receipt.json"
        results.write_bytes((arm + "\n").encode())
        arm_receipt.write_bytes((arm + "-receipt\n").encode())
        cell_receipt.write_bytes((arm + "-cell\n").encode())
        arm_receipt.chmod(0o444)
        cell_receipt.chmod(0o444)
        result_sources[arm] = {
            "path": str(results.resolve()),
            "file_sha256": file_sha256(results),
        }
        arm_receipts[arm] = {
            "path": str(arm_receipt.resolve()),
            "file_sha256": file_sha256(arm_receipt),
            "receipt_sha256": ("a" if arm == "candidate" else "b") * 64,
            "cell_receipts": [
                {
                    "cell_id": "cell-a",
                    "receipt_path": str(cell_receipt.relative_to(provenance_root)),
                    "receipt_file_sha256": file_sha256(cell_receipt),
                }
            ],
        }
    provenance = {
        "schema_version": 1,
        "contract": "risk-score-extreme-score-match-execution-provenance-v1",
        "runner_binding": {"bound": True},
        "runner_spec_sha256": "c" * 64,
        "result_sources": result_sources,
        "arm_receipts": arm_receipts,
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)

    class Runner:
        execution_provenance = provenance

    attestation_path = tmp_path / "attestation.json"
    published = publish_execution_attestation(
        output_path=attestation_path,
        spec_path=spec_path,
        plan_path=plan_path,
        report_path=report_path,
        runner=Runner(),
    )
    assert load_execution_attestation(attestation_path) == published

    cell_path = Path(
        published["execution_provenance"]["arm_receipts"]["candidate"]["cell_receipts"][
            0
        ]["receipt_path"]
    )
    absolute_cell = provenance_root / cell_path
    absolute_cell.chmod(0o644)
    absolute_cell.write_bytes(b"changed")
    with pytest.raises(
        ExtremeScoreEvaluationRunError, match="cell receipt (changed|must be read-only)"
    ):
        load_execution_attestation(attestation_path)
