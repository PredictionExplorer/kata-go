"""Run the production held-out evaluator through its hash-bound match adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from risk_score.extreme_score_evaluator import (
    build_runner_jobs,
    canonical_json,
    canonical_sha256,
    evaluate_plan_file,
    file_sha256,
)
from risk_score.extreme_score_match_runner import (
    ExtremeScoreMatchRunner,
    ExtremeScoreMatchRunnerSpec,
)

REQUEST_CONTRACT = "risk-score-extreme-evaluation-run-request-v1"
SPEC_CONTRACT = "risk-score-extreme-evaluation-run-spec-v1"
ATTESTATION_CONTRACT = "risk-score-extreme-evaluation-attestation-v1"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ExtremeScoreEvaluationRunError(RuntimeError):
    """A production evaluator launch specification is invalid."""


def _absolute_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExtremeScoreEvaluationRunError(f"{role} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ExtremeScoreEvaluationRunError(f"{role} must be absolute and normalized")
    return path


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ExtremeScoreEvaluationRunError(
            f"{role} must be a regular non-symlink file: {source}"
        )
    return source.resolve()


def _file_binding(value: Any, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ExtremeScoreEvaluationRunError(f"{role} must be a path/SHA-256 binding")
    path = _regular_file(_absolute_path(value["path"], f"{role}.path"), role)
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or file_sha256(path) != digest
    ):
        raise ExtremeScoreEvaluationRunError(f"{role} SHA-256 is invalid")
    return {"path": str(path), "sha256": digest}


def _model_bindings(value: Any, role: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ExtremeScoreEvaluationRunError(f"{role} must be a nonempty array")
    result = [
        _file_binding(item, f"{role} {index}") for index, item in enumerate(value)
    ]
    hashes = [item["sha256"] for item in result]
    if len(set(hashes)) != len(hashes):
        raise ExtremeScoreEvaluationRunError(f"{role} contains duplicate models")
    return sorted(result, key=lambda item: item["sha256"])


def _validate_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "contract",
        "katago_binary",
        "match_config",
        "focal_models",
        "opponent_models",
        "output_root",
        "topology",
        "process_count",
        "gpu",
    }
    if not isinstance(request, Mapping) or set(request) != expected_keys:
        raise ExtremeScoreEvaluationRunError(
            "evaluation run request keys differ from contract"
        )
    if request["schema_version"] != 1 or request["contract"] not in {
        REQUEST_CONTRACT,
        SPEC_CONTRACT,
    }:
        raise ExtremeScoreEvaluationRunError(
            "evaluation run request identity is invalid"
        )
    output_root = _absolute_path(request["output_root"], "output_root")
    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise ExtremeScoreEvaluationRunError(
            "output_root parent must be an existing non-symlink directory"
        )
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise ExtremeScoreEvaluationRunError(
            "output_root must be a non-symlink directory"
        )
    topology = request["topology"]
    if (
        not isinstance(topology, str)
        or not topology
        or topology != topology.strip()
        or any(character in topology for character in ("\x00", "\n", "\r"))
    ):
        raise ExtremeScoreEvaluationRunError("topology is invalid")
    process_count = request["process_count"]
    if type(process_count) is not int or process_count <= 0:
        raise ExtremeScoreEvaluationRunError("process_count must be a positive integer")
    gpu = request["gpu"]
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "index",
        "uuid",
        "lease_provenance",
    }:
        raise ExtremeScoreEvaluationRunError("GPU binding is malformed")
    if type(gpu["index"]) is not int or gpu["index"] < 0:
        raise ExtremeScoreEvaluationRunError("GPU index must be nonnegative")
    for field in ("uuid", "lease_provenance"):
        if (
            not isinstance(gpu[field], str)
            or not gpu[field]
            or gpu[field] != gpu[field].strip()
            or any(character in gpu[field] for character in ("\x00", "\n", "\r"))
        ):
            raise ExtremeScoreEvaluationRunError(f"GPU {field} is invalid")
    if re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu["uuid"]) is None:
        raise ExtremeScoreEvaluationRunError("GPU uuid is invalid")
    katago_binary = _file_binding(request["katago_binary"], "KataGo binary")
    if not os.access(katago_binary["path"], os.X_OK):
        raise ExtremeScoreEvaluationRunError("KataGo binary must be executable")
    return {
        "schema_version": 1,
        "contract": SPEC_CONTRACT,
        "katago_binary": katago_binary,
        "match_config": _file_binding(request["match_config"], "match config"),
        "focal_models": _model_bindings(request["focal_models"], "focal models"),
        "opponent_models": _model_bindings(
            request["opponent_models"], "opponent models"
        ),
        "output_root": str(output_root),
        "topology": topology,
        "process_count": process_count,
        "gpu": dict(gpu),
    }


def _decode_json(path: Path, role: str) -> dict[str, Any]:
    source = _regular_file(path, role)
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtremeScoreEvaluationRunError(f"cannot decode {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtremeScoreEvaluationRunError(f"{role} must have an object root")
    return value


def _publish_read_only_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    data = (canonical_json(value) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ExtremeScoreEvaluationRunError("spec output parent is unsafe")
    if target.exists() or target.is_symlink():
        source = _regular_file(target, "evaluation run spec")
        if source.stat().st_mode & _WRITE_BITS or source.read_bytes() != data:
            raise ExtremeScoreEvaluationRunError(
                "existing evaluation run spec conflicts or is writable"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_spec(request_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _validate_payload(_decode_json(request_path, "evaluation run request"))
    payload["spec_sha256"] = canonical_sha256(payload)
    _publish_read_only_json(output_path, payload)
    return payload


def load_spec(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "evaluation run spec")
    if source.stat().st_mode & _WRITE_BITS:
        raise ExtremeScoreEvaluationRunError("evaluation run spec must be read-only")
    value = _decode_json(source, "evaluation run spec")
    supplied = value.get("spec_sha256")
    payload = dict(value)
    payload.pop("spec_sha256", None)
    checked = _validate_payload(payload)
    if supplied != canonical_sha256(checked):
        raise ExtremeScoreEvaluationRunError("evaluation run spec self-hash is invalid")
    checked["spec_sha256"] = supplied
    return checked


def _runner_spec(value: Mapping[str, Any]) -> ExtremeScoreMatchRunnerSpec:
    gpu = value["gpu"]
    return ExtremeScoreMatchRunnerSpec(
        katago_binary=Path(value["katago_binary"]["path"]),
        focal_models={
            item["sha256"]: Path(item["path"]) for item in value["focal_models"]
        },
        opponent_models={
            item["sha256"]: Path(item["path"]) for item in value["opponent_models"]
        },
        match_config=Path(value["match_config"]["path"]),
        output_root=Path(value["output_root"]),
        topology=value["topology"],
        process_count=value["process_count"],
        expected_gpu_uuid=gpu["uuid"],
        gpu_lease_provenance=gpu["lease_provenance"],
        gpu_index=gpu["index"],
    )


def _artifact_binding(path: Path, role: str) -> dict[str, str]:
    source = _regular_file(path, role)
    return {"path": str(source), "file_sha256": file_sha256(source)}


def publish_execution_attestation(
    *,
    output_path: Path,
    spec_path: Path,
    plan_path: Path,
    report_path: Path,
    runner: Any,
) -> dict[str, Any]:
    provenance = getattr(runner, "execution_provenance", None)
    if not isinstance(provenance, Mapping):
        raise ExtremeScoreEvaluationRunError(
            "production runner did not expose execution provenance"
        )
    value = {
        "schema_version": 1,
        "contract": ATTESTATION_CONTRACT,
        "run_spec": {
            **_artifact_binding(spec_path, "evaluation run spec"),
            "spec_sha256": load_spec(spec_path)["spec_sha256"],
        },
        "plan": _artifact_binding(plan_path, "expected-max plan"),
        "report": _artifact_binding(report_path, "expected-max report"),
        "execution_provenance": json.loads(canonical_json(provenance)),
    }
    value["attestation_sha256"] = canonical_sha256(value)
    _publish_read_only_json(output_path, value)
    return value


def load_execution_attestation(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "execution attestation")
    if source.stat().st_mode & _WRITE_BITS:
        raise ExtremeScoreEvaluationRunError("execution attestation must be read-only")
    value = _decode_json(source, "execution attestation")
    expected = {
        "schema_version",
        "contract",
        "run_spec",
        "plan",
        "report",
        "execution_provenance",
        "attestation_sha256",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("contract") != ATTESTATION_CONTRACT
    ):
        raise ExtremeScoreEvaluationRunError(
            "execution attestation keys or contract are invalid"
        )
    supplied = value["attestation_sha256"]
    payload = dict(value)
    payload.pop("attestation_sha256")
    if supplied != canonical_sha256(payload):
        raise ExtremeScoreEvaluationRunError(
            "execution attestation self-hash is invalid"
        )
    for role in ("run_spec", "plan", "report"):
        binding = value[role]
        expected_keys = {"path", "file_sha256"}
        if role == "run_spec":
            expected_keys.add("spec_sha256")
        if not isinstance(binding, Mapping) or set(binding) != expected_keys:
            raise ExtremeScoreEvaluationRunError(
                f"execution attestation {role} binding is malformed"
            )
        artifact = _regular_file(Path(binding["path"]), role)
        if artifact.stat().st_mode & _WRITE_BITS:
            raise ExtremeScoreEvaluationRunError(
                f"execution attestation {role} artifact must be read-only"
            )
        if file_sha256(artifact) != binding["file_sha256"]:
            raise ExtremeScoreEvaluationRunError(
                f"execution attestation {role} artifact changed"
            )
    provenance = value["execution_provenance"]
    if not isinstance(provenance, Mapping):
        raise ExtremeScoreEvaluationRunError("execution provenance must be an object")
    arm_receipts = provenance.get("arm_receipts")
    if not isinstance(arm_receipts, Mapping) or set(arm_receipts) != {
        "candidate",
        "reference",
    }:
        raise ExtremeScoreEvaluationRunError(
            "execution provenance arm receipts are incomplete"
        )
    for arm, binding in arm_receipts.items():
        receipt_path = _regular_file(
            Path(binding.get("path", "")), f"{arm} arm receipt"
        )
        if receipt_path.stat().st_mode & _WRITE_BITS:
            raise ExtremeScoreEvaluationRunError(f"{arm} arm receipt must be read-only")
        if file_sha256(receipt_path) != binding.get("file_sha256"):
            raise ExtremeScoreEvaluationRunError(f"{arm} arm receipt changed")
        plan_root = receipt_path.parents[2]
        for cell in binding.get("cell_receipts", []):
            cell_path = plan_root / cell.get("receipt_path", "")
            receipt = _regular_file(cell_path, f"{arm} cell receipt")
            if receipt.stat().st_mode & _WRITE_BITS:
                raise ExtremeScoreEvaluationRunError(
                    f"{arm} cell receipt must be read-only"
                )
            if file_sha256(receipt) != cell.get("receipt_file_sha256"):
                raise ExtremeScoreEvaluationRunError(f"{arm} cell receipt changed")
    return value


def verify_execution_attestation(
    *,
    attestation_path: Path,
    spec_path: Path,
    plan_path: Path,
    report_path: Path,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    runner_factory: Callable[
        [ExtremeScoreMatchRunnerSpec], Any
    ] = ExtremeScoreMatchRunner,
) -> dict[str, Any]:
    attestation = load_execution_attestation(attestation_path)
    expected_paths = {
        "run_spec": spec_path,
        "plan": plan_path,
        "report": report_path,
    }
    for role, expected_path in expected_paths.items():
        binding = attestation[role]
        if Path(binding["path"]) != Path(expected_path).resolve() or binding[
            "file_sha256"
        ] != file_sha256(expected_path):
            raise ExtremeScoreEvaluationRunError(
                f"execution attestation is bound to another {role}"
            )
    spec = load_spec(spec_path)
    if attestation["run_spec"]["spec_sha256"] != spec["spec_sha256"]:
        raise ExtremeScoreEvaluationRunError(
            "execution attestation run-spec identity changed"
        )
    runner = runner_factory(_runner_spec(spec))
    for arm in ("candidate", "reference"):
        runner(arm, build_runner_jobs(plan, arm))
    observed = runner.execution_provenance
    if observed != attestation["execution_provenance"]:
        raise ExtremeScoreEvaluationRunError(
            "execution attestation contradicts validated runner receipts"
        )
    for arm in ("candidate", "reference"):
        report_source = report["result_bindings"][arm]["source"]
        runner_source = observed["result_sources"][arm]
        if (
            report_source["file_sha256"] != runner_source["file_sha256"]
            or _regular_file(
                Path(report_source["path"]), f"{arm} report results"
            ).read_bytes()
            != _regular_file(
                Path(runner_source["path"]), f"{arm} runner results"
            ).read_bytes()
        ):
            raise ExtremeScoreEvaluationRunError(
                f"{arm} report results contradict runner receipts"
            )
    return attestation


def run_evaluation(
    *,
    spec_path: Path,
    plan_path: Path,
    output_path: Path,
    attestation_output_path: Path | None = None,
    raw_lifetime_records: Mapping[str, Any] | None = None,
    runner_factory: Callable[
        [ExtremeScoreMatchRunnerSpec], Any
    ] = ExtremeScoreMatchRunner,
    evaluator: Callable[..., dict[str, Any]] = evaluate_plan_file,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    runner = runner_factory(_runner_spec(spec))
    report = evaluator(
        plan_path,
        output_path,
        runner=runner,
        raw_lifetime_records=raw_lifetime_records,
    )
    if attestation_output_path is not None:
        publish_execution_attestation(
            output_path=attestation_output_path,
            spec_path=spec_path,
            plan_path=plan_path,
            report_path=output_path,
            runner=runner,
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--request", required=True, type=Path)
    finalize.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--spec", required=True, type=Path)
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--attestation-output", required=True, type=Path)
    run.add_argument("--raw-lifetime-records", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "finalize":
            value = finalize_spec(args.request, args.output)
            result = {
                "output": str(args.output.resolve()),
                "file_sha256": file_sha256(args.output),
                "spec_sha256": value["spec_sha256"],
            }
        else:
            diagnostics = (
                _decode_json(args.raw_lifetime_records, "raw lifetime records")
                if args.raw_lifetime_records is not None
                else None
            )
            report = run_evaluation(
                spec_path=args.spec,
                plan_path=args.plan,
                output_path=args.output,
                attestation_output_path=args.attestation_output,
                raw_lifetime_records=diagnostics,
            )
            result = {
                "output": str(args.output.resolve()),
                "file_sha256": file_sha256(args.output),
                "report_sha256": report["report_sha256"],
                "decision": report["decision"],
                "promotion_recommended": report["promotion_recommended"],
                "attestation": str(args.attestation_output.resolve()),
                "attestation_file_sha256": file_sha256(args.attestation_output),
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
