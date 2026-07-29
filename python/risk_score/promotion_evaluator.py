#!/usr/bin/env python3
"""Execute a controller evaluation plan and derive promotion evidence.

This adapter is intentionally shell-free.  Every executable input is supplied
as an argv element, every content-bearing input is checked against the
controller plan, and only in-repo evidence derivation code decides PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from risk_score.evaluation_runner import (
    EvaluationError,
    EvaluationResult,
    EvaluationRunner,
    EvaluationSpec,
    canonical_json,
    canonical_sha256,
    file_sha256,
    load_suite_manifest,
)
from risk_score.paired_stats import load_policy
from risk_score.promotion_evidence import (
    CELL_ORDER,
    PromotionEvidenceError,
    build_controller_evidence,
    build_nonconfirmation_controller_evidence,
    build_promotion_evidence,
    derive_discovery_evidence,
    publish_controller_evidence,
    validate_stage0_probe,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_SHARDS = 256
_MAX_PARALLEL = 32
_MAX_ATTEMPTS = 5

_COMPARISON_TO_CELL = {
    "candidate-vs-champion-powered": "powered_candidate_vs_champion",
    "candidate-vs-original-powered": "powered_candidate_vs_original",
    "candidate-vs-original-standard": "standard_candidate_vs_original",
    "candidate-vs-champion-powered-lead-40": "lead_40",
    "candidate-vs-champion-powered-lead-80": "lead_80",
}
_NONCONFIRMATION_CELLS = CELL_ORDER[:3]
_SCHEDULE_ARGUMENTS = {
    "powered_candidate_vs_champion": "powered_champion",
    "powered_candidate_vs_original": "powered_original",
    "standard_candidate_vs_original": "standard_original",
    "lead_40": "lead40",
    "lead_80": "lead80",
}


class PromotionEvaluatorError(ValueError):
    """The configured evaluator inputs contradict their immutable plan."""


@dataclass(frozen=True)
class CellExecution:
    name: str
    spec: EvaluationSpec
    schedule_path: Path
    config_path: Path
    reference_model_path: Path


@dataclass(frozen=True)
class ValidatedInputs:
    plan: Mapping[str, Any]
    cells: Tuple[CellExecution, ...]
    policy: Mapping[str, Any]
    candidate_hash: str
    champion_hash: str
    original_hash: str


def _reject_constant(value: str) -> None:
    raise PromotionEvaluatorError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PromotionEvaluatorError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_canonical_json(path: Path, role: str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PromotionEvaluatorError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionEvaluatorError(f"cannot load {role} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionEvaluatorError(f"{role} must have an object root")
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise PromotionEvaluatorError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PromotionEvaluatorError(
            f"{role} must be a lowercase 64-character SHA-256"
        )
    return value


def _optional_text(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or "\x00" in value or "\n" in value or "\r" in value:
        raise PromotionEvaluatorError("optional argv value is malformed")
    return value


def _optional_path(value: Any) -> Optional[Path]:
    text = _optional_text(value)
    return None if text is None else Path(text)


def _regular_file(path: Path, role: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise PromotionEvaluatorError(f"{role} path must be absolute")
    if source.is_symlink() or not source.is_file():
        raise PromotionEvaluatorError(f"{role} must be a regular non-symlink file")
    return source


def _verify_file(path: Path, expected_hash: str, role: str) -> None:
    _regular_file(path, role)
    actual = file_sha256(path)
    if actual != expected_hash:
        raise PromotionEvaluatorError(
            f"{role} SHA-256 mismatch: found {actual}, expected {expected_hash}"
        )


def _verify_supplied_hash(supplied: Any, expected: str, role: str) -> None:
    supplied_hash = _require_sha256(supplied, f"{role} supplied hash")
    if supplied_hash != expected:
        raise PromotionEvaluatorError(f"{role} supplied hash contradicts plan")


def _single(values: Iterable[Any], role: str) -> Any:
    unique = set(values)
    if len(unique) != 1:
        raise PromotionEvaluatorError(f"plan cells disagree on {role}")
    return next(iter(unique))


def _validate_output_path(path: Path, role: str) -> None:
    if not path.is_absolute():
        raise PromotionEvaluatorError(f"{role} path must be absolute")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise PromotionEvaluatorError(
            f"{role} parent must be an existing non-symlink directory"
        )
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise PromotionEvaluatorError(f"{role} is not a regular file")


def _artifact_schedule(
    args: argparse.Namespace,
    *,
    cell_name: str,
    artifact: Mapping[str, Any],
    spec: EvaluationSpec,
) -> Path:
    prefix = _SCHEDULE_ARGUMENTS[cell_name]
    supplied_path = _optional_text(getattr(args, f"{prefix}_schedule"))
    supplied_hash = _optional_text(getattr(args, f"{prefix}_schedule_sha256"))
    supplied_id = _optional_text(getattr(args, f"{prefix}_schedule_id"))

    plan_path = artifact.get("path")
    plan_hash = artifact.get("sha256")
    plan_id = artifact.get("scheduleId")
    if not isinstance(plan_path, str) or not plan_path:
        raise PromotionEvaluatorError(f"plan schedule {cell_name} has no path")
    _require_sha256(plan_hash, f"plan schedule {cell_name} hash")
    if not isinstance(plan_id, str) or not plan_id:
        raise PromotionEvaluatorError(f"plan schedule {cell_name} has no schedule ID")

    if supplied_path is not None and supplied_path != plan_path:
        raise PromotionEvaluatorError(
            f"{cell_name} supplied schedule path contradicts plan"
        )
    if supplied_hash is not None and supplied_hash != plan_hash:
        raise PromotionEvaluatorError(
            f"{cell_name} supplied schedule hash contradicts plan"
        )
    if supplied_id is not None and supplied_id != plan_id:
        raise PromotionEvaluatorError(
            f"{cell_name} supplied schedule ID contradicts plan"
        )
    if (
        artifact.get("cell") != cell_name
        or artifact.get("comparison") != spec.comparison
        or artifact.get("stage") != spec.stage
        or artifact.get("look") != spec.look
        or plan_hash != spec.schedule_sha
        or plan_id != spec.schedule_id
        or artifact.get("suiteBankSha256") != spec.suite_bank_sha
    ):
        raise PromotionEvaluatorError(
            f"{cell_name} schedule artifact contradicts EvaluationSpec"
        )
    path = Path(plan_path)
    _verify_file(path, plan_hash, f"{cell_name} schedule")
    return path


def _validate_plan(args: argparse.Namespace) -> ValidatedInputs:
    plan_path = _regular_file(args.plan, "controller plan")
    if _optional_text(args.plan_sha256) is not None:
        _verify_file(
            plan_path,
            _require_sha256(args.plan_sha256, "controller plan hash"),
            "controller plan",
        )
    plan = _load_canonical_json(plan_path, "controller plan")
    required_plan_keys = {
        "evaluationKey",
        "configHash",
        "scheduleHash",
        "policyHash",
        "policyPath",
        "policyVersion",
        "selfplayConfigHash",
        "topology",
        "stage",
        "look",
        "suiteManifestPath",
        "suiteManifestHash",
        "scheduleArtifacts",
        "specs",
    }
    missing = sorted(required_plan_keys.difference(plan))
    if missing:
        raise PromotionEvaluatorError(f"controller plan is missing keys: {missing}")
    stage = plan.get("stage")
    if stage not in {"stage-0", "stage-1", "stage-2", "stage-3"}:
        raise PromotionEvaluatorError(f"unsupported controller stage {stage!r}")
    for key in (
        "configHash",
        "scheduleHash",
        "policyHash",
        "selfplayConfigHash",
        "suiteManifestHash",
    ):
        _require_sha256(plan.get(key), f"controller plan {key}")
    for key in (
        "evaluationKey",
        "policyPath",
        "policyVersion",
        "topology",
        "look",
        "suiteManifestPath",
    ):
        value = plan.get(key)
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or "\n" in value
            or "\r" in value
        ):
            raise PromotionEvaluatorError(
                f"controller plan {key} must be a nonempty single-line string"
            )
    expected_cells = (
        tuple(CELL_ORDER) if stage == "stage-3" else tuple(_NONCONFIRMATION_CELLS)
    )

    raw_specs = plan.get("specs")
    if not isinstance(raw_specs, list):
        raise PromotionEvaluatorError("controller plan specs must be an array")
    try:
        specs = tuple(
            EvaluationSpec.from_dict(value)
            for value in raw_specs
            if isinstance(value, Mapping)
        )
    except ValueError as exc:
        raise PromotionEvaluatorError(f"invalid EvaluationSpec: {exc}") from exc
    if len(specs) != len(raw_specs) or len(specs) != len(expected_cells):
        raise PromotionEvaluatorError("controller plan has the wrong EvaluationSpec count")
    if plan["evaluationKey"] != "matrix-" + canonical_sha256(raw_specs):
        raise PromotionEvaluatorError("controller plan evaluationKey is invalid")
    if plan["configHash"] != canonical_sha256(
        sorted({spec.config_sha for spec in specs})
    ):
        raise PromotionEvaluatorError("controller plan configHash is invalid")
    if plan["scheduleHash"] != canonical_sha256(
        sorted({spec.schedule_sha for spec in specs})
    ):
        raise PromotionEvaluatorError("controller plan scheduleHash is invalid")

    by_cell: Dict[str, EvaluationSpec] = {}
    for spec in specs:
        cell_name = _COMPARISON_TO_CELL.get(spec.comparison)
        if cell_name is None or cell_name in by_cell:
            raise PromotionEvaluatorError(
                f"plan has unknown or duplicate comparison {spec.comparison!r}"
            )
        by_cell[cell_name] = spec
    if set(by_cell) != set(expected_cells):
        raise PromotionEvaluatorError(
            "controller plan does not contain the exact required evaluation matrix"
        )

    candidate_hash = _single(
        (spec.candidate_model_sha for spec in specs), "candidate model hash"
    )
    original_hash = _single(
        (spec.original_model_sha for spec in specs), "original model hash"
    )
    champion_hash = _single(
        (
            spec.reference_model_sha
            for spec in specs
            if "champion" in spec.comparison
        ),
        "tested champion model hash",
    )
    if any(
        spec.stage != stage
        or spec.look != plan["look"]
        or spec.topology != plan["topology"]
        or spec.policy_sha != plan["policyHash"]
        or spec.suite_manifest_sha != plan["suiteManifestHash"]
        for spec in specs
    ):
        raise PromotionEvaluatorError(
            "EvaluationSpecs contradict plan stage/look/topology/policy/manifest"
        )

    policy_path_text = str(args.policy)
    if policy_path_text != plan["policyPath"]:
        raise PromotionEvaluatorError("supplied policy path contradicts plan")
    policy_path = _regular_file(args.policy, "promotion policy")
    policy = load_policy(policy_path)
    policy_hash = canonical_sha256(policy)
    _require_sha256(plan["policyHash"], "plan policy hash")
    if policy_hash != plan["policyHash"]:
        raise PromotionEvaluatorError("promotion policy canonical hash contradicts plan")
    _verify_supplied_hash(args.policy_sha256, policy_hash, "promotion policy")
    if policy.get("policy_version") != plan["policyVersion"]:
        raise PromotionEvaluatorError("promotion policy version contradicts plan")

    suite_path_text = str(args.suite_manifest)
    if suite_path_text != plan["suiteManifestPath"]:
        raise PromotionEvaluatorError("supplied suite manifest path contradicts plan")
    suite_manifest_path = _regular_file(args.suite_manifest, "suite manifest")
    suite_hash = _require_sha256(plan["suiteManifestHash"], "plan suite manifest hash")
    _verify_supplied_hash(
        args.suite_manifest_sha256, suite_hash, "suite manifest"
    )
    _verify_file(suite_manifest_path, suite_hash, "suite manifest")
    suite_manifest = load_suite_manifest(suite_manifest_path)
    if suite_manifest.get("policy_hash") != policy_hash:
        raise PromotionEvaluatorError("suite manifest is bound to another policy")

    _verify_supplied_hash(
        args.candidate_model_sha256, candidate_hash, "candidate model"
    )
    _verify_supplied_hash(
        args.champion_model_sha256, champion_hash, "champion model"
    )
    _verify_supplied_hash(
        args.original_model_sha256, original_hash, "original model"
    )
    _verify_file(args.candidate_model, candidate_hash, "candidate model")
    _verify_file(args.champion_model, champion_hash, "champion model")
    _verify_file(args.original_model, original_hash, "original model")

    powered_hash = _single(
        (
            spec.config_sha
            for spec in specs
            if spec.comparison != "candidate-vs-original-standard"
        ),
        "powered config hash",
    )
    standard_hash = by_cell["standard_candidate_vs_original"].config_sha
    _verify_supplied_hash(
        args.powered_config_sha256, powered_hash, "powered config"
    )
    _verify_supplied_hash(
        args.standard_config_sha256, standard_hash, "standard config"
    )
    _verify_file(args.powered_config, powered_hash, "powered config")
    _verify_file(args.standard_config, standard_hash, "standard config")

    artifacts = plan.get("scheduleArtifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_cells):
        raise PromotionEvaluatorError(
            "plan scheduleArtifacts do not match the exact evaluation matrix"
        )
    for absent_cell in set(CELL_ORDER).difference(expected_cells):
        prefix = _SCHEDULE_ARGUMENTS[absent_cell]
        if any(
            _optional_text(getattr(args, f"{prefix}_schedule{suffix}")) is not None
            for suffix in ("", "_sha256", "_id")
        ):
            raise PromotionEvaluatorError(
                f"unexpected schedule arguments for absent cell {absent_cell}"
            )

    executions = []
    for cell_name in expected_cells:
        spec = by_cell[cell_name]
        artifact = artifacts[cell_name]
        if not isinstance(artifact, Mapping):
            raise PromotionEvaluatorError(
                f"plan schedule artifact {cell_name} is malformed"
            )
        schedule_path = _artifact_schedule(
            args, cell_name=cell_name, artifact=artifact, spec=spec
        )
        is_standard = spec.comparison == "candidate-vs-original-standard"
        config_path = args.standard_config if is_standard else args.powered_config
        reference_path = (
            args.champion_model
            if "champion" in spec.comparison
            else args.original_model
        )
        expected_reference = champion_hash if "champion" in spec.comparison else original_hash
        if spec.reference_model_sha != expected_reference:
            raise PromotionEvaluatorError(
                f"{cell_name} reference hash contradicts comparison role"
            )
        executions.append(
            CellExecution(
                name=cell_name,
                spec=spec,
                schedule_path=schedule_path,
                config_path=config_path,
                reference_model_path=reference_path,
            )
        )
    return ValidatedInputs(
        plan=plan,
        cells=tuple(executions),
        policy=policy,
        candidate_hash=candidate_hash,
        champion_hash=champion_hash,
        original_hash=original_hash,
    )


def _load_stage0(
    args: argparse.Namespace,
    inputs: ValidatedInputs,
) -> Tuple[Path, str, Path, str]:
    probe_path = _optional_path(args.stage0_probe)
    probe_hash = _optional_text(args.stage0_probe_sha256)
    request_path = _optional_path(args.stage0_request)
    request_hash = _optional_text(args.stage0_request_sha256)
    if None in (probe_path, probe_hash, request_path, request_hash):
        raise PromotionEvaluatorError(
            "every stage requires the finalized hash-bound Stage-0 probe and request"
        )
    assert probe_path is not None
    assert probe_hash is not None
    assert request_path is not None
    assert request_hash is not None
    probe_hash = _require_sha256(probe_hash, "Stage-0 probe hash")
    request_hash = _require_sha256(request_hash, "Stage-0 request hash")
    validate_stage0_probe(
        probe_path,
        expected_sha256=probe_hash,
        policy=inputs.policy,
        candidate_hash=inputs.candidate_hash,
        champion_hash=inputs.champion_hash,
        original_hash=inputs.original_hash,
        request_path=request_path,
        request_sha256=request_hash,
    )
    request = _load_canonical_json(request_path, "Stage-0 request")
    expected_request_bindings = {
        "policy_path": inputs.plan["policyPath"],
        "policy_hash": inputs.plan["policyHash"],
        "policy_version": inputs.plan["policyVersion"],
        "suite_manifest_path": inputs.plan["suiteManifestPath"],
        "suite_manifest_hash": inputs.plan["suiteManifestHash"],
    }
    conflicts = [
        key
        for key, expected in expected_request_bindings.items()
        if request.get(key) != expected
    ]
    if conflicts:
        raise PromotionEvaluatorError(
            "Stage-0 request contradicts current immutable inputs: "
            + ", ".join(sorted(conflicts))
        )
    return probe_path, probe_hash, request_path, request_hash


def _load_bound_json(
    path_value: Any,
    hash_value: Any,
    role: str,
) -> Tuple[Path, str, Dict[str, Any]]:
    path = _optional_path(path_value)
    digest = _optional_text(hash_value)
    if path is None or digest is None:
        raise PromotionEvaluatorError(f"{role} path and hash are required")
    digest = _require_sha256(digest, f"{role} hash")
    _verify_file(path, digest, role)
    return path, digest, _load_canonical_json(path, role)


def _champion_statistics(
    evidence: Mapping[str, Any],
    *,
    controller_stage: str,
    inputs: ValidatedInputs,
) -> Mapping[str, Any]:
    expected = {
        "finalized": True,
        "controller_stage": controller_stage,
        "candidate_hash": inputs.candidate_hash,
        "tested_champion_hash": inputs.champion_hash,
        "original_hash": inputs.original_hash,
        "policy_hash": inputs.plan["policyHash"],
    }
    conflicts = [
        key for key, value in expected.items() if evidence.get(key) != value
    ]
    if conflicts:
        raise PromotionEvaluatorError(
            "prior stage evidence contradicts current identity: "
            + ", ".join(sorted(conflicts))
        )
    stage_gate = evidence.get("stage_gate")
    if not isinstance(stage_gate, Mapping):
        raise PromotionEvaluatorError("prior stage evidence has no derived stage gate")
    derivation_hash = stage_gate.get("derivation_hash")
    payload = dict(stage_gate)
    payload.pop("derivation_hash", None)
    if derivation_hash != canonical_sha256(payload):
        raise PromotionEvaluatorError("prior stage derivation hash is invalid")
    statistics = stage_gate.get("derived_artifacts", {}).get("statistics")
    if not isinstance(statistics, Mapping):
        raise PromotionEvaluatorError("prior stage evidence has no statistics")
    for finalized in statistics.values():
        if not isinstance(finalized, Mapping):
            continue
        artifact = finalized.get("statistics_artifact")
        manifest = finalized.get("statistics_manifest")
        if (
            isinstance(artifact, Mapping)
            and isinstance(manifest, Mapping)
            and finalized.get("statistics_artifact_hash")
            == canonical_sha256(artifact)
            and finalized.get("statistics_manifest_hash")
            == canonical_sha256(manifest)
            and artifact.get("data_binding", {}).get("comparison")
            == "candidate-vs-champion-powered"
        ):
            return artifact
    raise PromotionEvaluatorError(
        "prior stage statistics omit the champion powered comparison"
    )


def _confirmation_cluster_ids(suite_manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    clusters = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            stage = value.get("stage")
            ids = value.get(
                "independent_cluster_ids",
                value.get("independentClusterIds"),
            )
            if stage == "stage-3" and isinstance(ids, list):
                clusters.extend(ids)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(suite_manifest.get("cells"))
    if not clusters:
        for bank in suite_manifest.get("banks", []):
            if not isinstance(bank, Mapping) or bank.get("name") not in {
                "confirmation",
                "lead-40",
                "lead-80",
            }:
                continue
            ids = bank.get("independentClusterIds")
            if isinstance(ids, list):
                clusters.extend(ids)
    if not clusters or any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in clusters
    ):
        raise PromotionEvaluatorError(
            "suite manifest lacks confirmation independent-cluster provenance"
        )
    return tuple(clusters)


def _derive_and_publish_discovery(
    args: argparse.Namespace,
    inputs: ValidatedInputs,
    evidence: Mapping[str, Any],
) -> Tuple[Path, str]:
    _, _, stage1 = _load_bound_json(
        args.stage1_evidence,
        args.stage1_evidence_sha256,
        "Stage-1 evaluator evidence",
    )
    stage1_statistics = _champion_statistics(
        stage1, controller_stage="screen", inputs=inputs
    )
    stage2_statistics = _champion_statistics(
        evidence, controller_stage="finalist", inputs=inputs
    )
    suite_manifest = load_suite_manifest(Path(inputs.plan["suiteManifestPath"]))
    discovery = derive_discovery_evidence(
        stage1_statistics,
        [stage2_statistics],
        candidate_hash=inputs.candidate_hash,
        policy=inputs.policy,
        confirmation_cluster_ids=_confirmation_cluster_ids(suite_manifest),
    )
    output = _optional_path(args.discovery_output)
    if output is None:
        raise PromotionEvaluatorError("Stage-2 requires --discovery-output")
    _validate_output_path(output, "discovery evidence output")
    publish_controller_evidence(output, discovery)
    return output, file_sha256(output)


def _validate_limits(args: argparse.Namespace) -> None:
    for value, role, maximum in (
        (args.shards, "shards", _MAX_SHARDS),
        (args.max_parallel, "max parallelism", _MAX_PARALLEL),
        (args.max_attempts, "max attempts", _MAX_ATTEMPTS),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise PromotionEvaluatorError(
                f"{role} must be between 1 and {maximum}"
            )
    if args.max_parallel > args.shards:
        raise PromotionEvaluatorError("max parallelism may not exceed shard count")


def evaluate(
    args: argparse.Namespace,
    *,
    subprocess_runner: Callable[..., Any] = subprocess.run,
) -> Mapping[str, Any]:
    """Run every plan cell and publish its derived controller evidence."""

    _validate_limits(args)
    inputs = _validate_plan(args)
    katago = _regular_file(args.katago, "KataGo binary")
    katago_hash = file_sha256(katago)
    evaluation_root = Path(args.evaluation_root)
    if not evaluation_root.is_absolute():
        raise PromotionEvaluatorError("evaluation root must be absolute")
    if evaluation_root.exists() and (
        evaluation_root.is_symlink() or not evaluation_root.is_dir()
    ):
        raise PromotionEvaluatorError(
            "evaluation root must be a non-symlink directory"
        )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    if evaluation_root.is_symlink():
        raise PromotionEvaluatorError("evaluation root may not be a symlink")

    runner_map_path = Path(args.runner_manifests)
    evidence_output = Path(args.output)
    _validate_output_path(runner_map_path, "runner manifest map")
    _validate_output_path(evidence_output, "controller evidence output")
    if runner_map_path == evidence_output:
        raise PromotionEvaluatorError(
            "runner manifest map and evidence output must be distinct"
        )

    probe_path, probe_hash, request_path, request_hash = _load_stage0(args, inputs)

    manifest_map: Dict[str, str] = {}
    for cell in inputs.cells:
        runner = EvaluationRunner(
            katago_binary=katago,
            config_path=cell.config_path,
            output_root=evaluation_root,
            shard_count=args.shards,
            max_parallel=args.max_parallel,
            max_attempts=args.max_attempts,
            include_move_traces=True,
            subprocess_runner=subprocess_runner,
        )
        result = runner.run(
            cell.spec,
            cell.schedule_path,
            args.candidate_model,
            cell.reference_model_path,
            original_model_path=args.original_model,
            policy_path=args.policy,
            suite_manifest_path=args.suite_manifest,
        )
        if not isinstance(result, EvaluationResult):
            raise PromotionEvaluatorError(f"{cell.name} did not finalize")
        if file_sha256(katago) != katago_hash:
            raise PromotionEvaluatorError(
                "KataGo binary changed during matrix execution"
            )
        manifest_map[cell.name] = str(result.manifest_path)

    # Close the verify/use window before publishing a matrix-level artifact.
    revalidated = _validate_plan(args)
    if revalidated.plan != inputs.plan or revalidated.cells != inputs.cells:
        raise PromotionEvaluatorError(
            "controller inputs changed during matrix execution"
        )
    if _load_stage0(args, revalidated) != (
        probe_path,
        probe_hash,
        request_path,
        request_hash,
    ):
        raise PromotionEvaluatorError(
            "Stage-0 artifacts changed during matrix execution"
        )
    publish_controller_evidence(runner_map_path, manifest_map)
    runner_map_hash = file_sha256(runner_map_path)
    runner_paths = {name: Path(path) for name, path in manifest_map.items()}

    stage = inputs.plan["stage"]
    common = {
        "runner_manifests_path": str(runner_map_path),
        "runner_manifests_hash": runner_map_hash,
        "stage0_request_path": str(request_path),
        "stage0_request_hash": request_hash,
        "stage0_probe_path": str(probe_path),
        "stage0_probe_hash": probe_hash,
        "stage0_probe_sha256": probe_hash,
    }
    if stage == "stage-3":
        discovery_path, discovery_hash, discovery = _load_bound_json(
            args.discovery_evidence,
            args.discovery_evidence_sha256,
            "discovery evidence",
        )
        attempt_path, attempt_hash, attempt = _load_bound_json(
            args.attempt, args.attempt_sha256, "attempt metadata"
        )
        promotion = build_promotion_evidence(
            runner_paths,
            suite_manifest_path=args.suite_manifest,
            policy_path=args.policy,
            stage0_probe_path=probe_path,
            stage0_probe_sha256=probe_hash,
            stage0_request_path=request_path,
            stage0_request_sha256=request_hash,
            discovery_evidence=discovery,
            attempt=attempt,
            bootstrap_replications=args.bootstrap_replications,
            bootstrap_seed=args.bootstrap_seed,
        )
        evidence: Dict[str, Any] = build_controller_evidence(inputs.plan, promotion)
        evidence.update(
            {
                **common,
                "discovery_evidence_path": str(discovery_path),
                "discovery_evidence_hash": discovery_hash,
                "discovery_evidence_sha256": discovery_hash,
                "attempt_path": str(attempt_path),
                "attempt_hash": attempt_hash,
            }
        )
    else:
        evidence = build_nonconfirmation_controller_evidence(
            inputs.plan,
            runner_manifests=runner_paths,
            policy_path=args.policy,
            stage0_probe_path=probe_path,
            stage0_probe_sha256=probe_hash,
            stage0_request_path=request_path,
            stage0_request_sha256=request_hash,
            bootstrap_replications=args.bootstrap_replications,
            bootstrap_seed=args.bootstrap_seed,
        )
        evidence.update(common)
        if stage == "stage-2":
            discovery_path, discovery_hash = _derive_and_publish_discovery(
                args, inputs, evidence
            )
            evidence.update(
                {
                    "discovery_evidence_path": str(discovery_path),
                    "discovery_evidence_hash": discovery_hash,
                    "discovery_evidence_sha256": discovery_hash,
                }
            )

    publish_controller_evidence(evidence_output, evidence)
    return {
        "output": str(evidence_output),
        "sha256": file_sha256(evidence_output),
        "runnerManifests": str(runner_map_path),
        "runnerManifestsSha256": runner_map_hash,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a controller plan and derive promotion evidence."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", default="")
    parser.add_argument("--katago", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--candidate-model-sha256", required=True)
    parser.add_argument("--champion-model", required=True, type=Path)
    parser.add_argument("--champion-model-sha256", required=True)
    parser.add_argument("--original-model", required=True, type=Path)
    parser.add_argument("--original-model-sha256", required=True)
    parser.add_argument("--powered-config", required=True, type=Path)
    parser.add_argument("--powered-config-sha256", required=True)
    parser.add_argument("--standard-config", required=True, type=Path)
    parser.add_argument("--standard-config-sha256", required=True)
    parser.add_argument("--suite-manifest", required=True, type=Path)
    parser.add_argument("--suite-manifest-sha256", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--policy-sha256", required=True)

    for prefix in _SCHEDULE_ARGUMENTS.values():
        option = prefix.replace("_", "-")
        parser.add_argument(f"--{option}-schedule", default="")
        parser.add_argument(f"--{option}-schedule-sha256", default="")
        parser.add_argument(f"--{option}-schedule-id", default="")

    parser.add_argument("--stage0-probe", default="")
    parser.add_argument("--stage0-probe-sha256", default="")
    parser.add_argument("--stage0-request", default="")
    parser.add_argument("--stage0-request-sha256", default="")
    parser.add_argument("--stage1-evidence", default="")
    parser.add_argument("--stage1-evidence-sha256", default="")
    parser.add_argument("--discovery-output", default="")
    parser.add_argument("--discovery-evidence", default="")
    parser.add_argument("--discovery-evidence-sha256", default="")
    parser.add_argument("--attempt", default="")
    parser.add_argument("--attempt-sha256", default="")

    parser.add_argument(
        "--evaluation-root",
        "--output-root",
        dest="evaluation_root",
        required=True,
        type=Path,
    )
    parser.add_argument("--runner-manifests", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--bootstrap-replications", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("-o", "--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    subprocess_runner: Optional[Callable[..., Any]] = None,
) -> int:
    args = parse_args(argv)
    try:
        result = evaluate(
            args,
            subprocess_runner=(
                subprocess.run if subprocess_runner is None else subprocess_runner
            ),
        )
    except (
        OSError,
        EvaluationError,
        KeyError,
        TypeError,
        ValueError,
        PromotionEvidenceError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
