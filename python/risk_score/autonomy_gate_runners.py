#!/usr/bin/env python3
"""Non-destructive evidence publishers for autonomy bootstrap gates.

The commands in this module only inspect production artifacts, execute bounded
model probes, and atomically replace their requested evidence file.  They do
not create candidates, alter controller state, or enable backpressure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from risk_score.autonomy_bootstrap import publish_gate_evidence
from risk_score.build_live_runtime import verify_deployment_manifest
from risk_score.model_probe import probe_model
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256
from risk_score.promotion_controller import inventory_candidates
from risk_score.promotion_host import HostCommandError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_ACTIVE_CANDIDATE_STATES = frozenset(
    {
        "claimed",
        "evaluating_integrity",
        "evaluating_screen",
        "evaluating_finalist",
        "evaluating_confirmation",
    }
)
_EVALUATING_CANDIDATE_STATES = _ACTIVE_CANDIDATE_STATES.difference({"claimed"})

SubprocessRunner = Callable[..., Any]


class GateRunnerError(HostCommandError):
    """An input or observation cannot safely support gate evidence."""


class _GateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise GateRunnerError(message)


@dataclass(frozen=True)
class ModelBinding:
    path: Path
    sha256: str


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _reject_symlink_components(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise GateRunnerError(
                f"{role} has a symlinked path component: {current}"
            )
        if current.parent == current:
            return
        current = current.parent


def _absolute_path(value: Path | str, role: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise GateRunnerError(f"{role} must be an absolute path")
    path = Path(value)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or normalized != path:
        raise GateRunnerError(f"{role} must be an absolute lexically-normal path")
    _reject_symlink_components(path, role)
    return path


def _regular_file(value: Path | str, role: str) -> Path:
    path = _absolute_path(value, role)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise GateRunnerError(f"{role} is missing: {path}") from exc
    if not stat.S_ISREG(mode):
        raise GateRunnerError(f"{role} must be a regular non-symlink file")
    return path


def _regular_directory(value: Path | str, role: str) -> Path:
    path = _absolute_path(value, role)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise GateRunnerError(f"{role} is missing: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise GateRunnerError(f"{role} must be a regular non-symlink directory")
    return path


def _optional_artifact_path(value: Path | str, role: str) -> Path:
    path = _absolute_path(value, role)
    if _lexists(path):
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise GateRunnerError(f"{role} must be a regular non-symlink file")
    return path


def _evidence_output(value: Path | str, role: str = "evidence output") -> Path:
    path = _absolute_path(value, role)
    existing_parent = path.parent
    while not _lexists(existing_parent):
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir() or existing_parent.is_symlink():
        raise GateRunnerError(
            f"{role} parent must descend from a non-symlink directory"
        )
    if _lexists(path) and not stat.S_ISREG(path.lstat().st_mode):
        raise GateRunnerError(f"{role} must be a regular non-symlink file")
    return path


def _require_distinct_output(output: Path, inputs: Sequence[Path]) -> None:
    if output in set(inputs):
        raise GateRunnerError("evidence output must not replace an inspected input")


def _strictly_within(path: Path, root: Path) -> bool:
    if path == root:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateRunnerError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise GateRunnerError(f"non-finite JSON value is forbidden: {value}")


def _load_canonical_object(
    path: Path,
    role: str,
) -> Tuple[Dict[str, Any], str]:
    source = _regular_file(path, role)
    size = source.stat().st_size
    if size > _MAX_JSON_BYTES:
        raise GateRunnerError(f"{role} exceeds the {_MAX_JSON_BYTES}-byte limit")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateRunnerError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GateRunnerError(f"{role} must have an object root")
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise GateRunnerError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value, hashlib.sha256(data).hexdigest()


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateRunnerError(f"{role} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateRunnerError(f"{role} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, role: str) -> bool:
    if type(value) is not bool:
        raise GateRunnerError(f"{role} must be a boolean")
    return value


def _status_readiness(status_path: Path, suite_manifest: Path) -> bool:
    if not _lexists(status_path):
        return False
    status, _ = _load_canonical_object(
        status_path,
        "curation pipeline status",
    )
    payload = dict(status)
    supplied_hash = payload.pop("status_sha256", None)
    if (
        status.get("schema_version") != 1
        or status.get("contract") != "risk-score-curation-pipeline-status-v1"
        or supplied_hash != canonical_sha256(payload)
    ):
        raise GateRunnerError(
            "curation pipeline status contract or self-hash is invalid"
        )
    state = status.get("state")
    error = status.get("error")
    artifacts = status.get("artifacts")
    if (
        not isinstance(state, str)
        or not state
        or (error is not None and not isinstance(error, Mapping))
        or not isinstance(artifacts, Mapping)
        or not isinstance(artifacts.get("reviewed_bank"), Mapping)
        or not isinstance(artifacts.get("suite"), Mapping)
    ):
        raise GateRunnerError("curation pipeline status is structurally invalid")
    reviewed_complete = _require_bool(
        artifacts["reviewed_bank"].get("complete"),
        "curation reviewed-bank completeness",
    )
    suite_record = artifacts["suite"]
    suite_complete = _require_bool(
        suite_record.get("complete"),
        "curation suite completeness",
    )
    declared_suite_path = _absolute_path(
        suite_record.get("path"),
        "curation status suite path",
    )
    if declared_suite_path != suite_manifest.parent:
        raise GateRunnerError(
            "curation status suite path differs from the inspected manifest"
        )
    complete = (
        state == "complete"
        and error is None
        and reviewed_complete
        and suite_complete
    )
    if state == "complete" and not complete:
        raise GateRunnerError(
            "curation status claims completion without complete artifacts"
        )
    return complete


def _suite_readiness(
    suite_manifest: Path,
    suite_registry_spec: Optional[Path] = None,
    *,
    suite_validator: Optional[Callable[..., Any]] = None,
) -> bool:
    if not _lexists(suite_manifest):
        return False
    suite, _ = _load_canonical_object(
        suite_manifest,
        "authoritative evaluation suite manifest",
    )
    payload = dict(suite)
    supplied_hash = payload.pop("manifestPayloadSha256", None)
    if (
        suite.get("schemaVersion") != 3
        or suite.get("manifestContract")
        != "risk-score-authoritative-evaluation-manifest-v3"
        or suite.get("machineReviewOnly") is not True
        or supplied_hash != canonical_sha256(payload)
    ):
        raise GateRunnerError(
            "evaluation suite manifest is not authoritative machine-review v3"
        )
    if suite_registry_spec is not None:
        registry_path = _regular_file(
            suite_registry_spec,
            "suite registry specification",
        )
        if suite_validator is None:
            from risk_score.suite_rotation import validate_suite_manifest

            suite_validator = validate_suite_manifest
        try:
            suite_validator(suite_manifest, registry_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise GateRunnerError(
                f"evaluation suite transitive validation failed: {exc}"
            ) from exc
    return True


def curation_suite_readiness(
    *,
    curation_status: Path,
    suite_manifest: Path,
    output: Path,
    suite_registry_spec: Optional[Path] = None,
    suite_validator: Optional[Callable[..., Any]] = None,
) -> Mapping[str, Any]:
    """Publish PASS only for two independently verified readiness artifacts."""

    status_path = _optional_artifact_path(
        curation_status,
        "curation status path",
    )
    manifest_path = _optional_artifact_path(
        suite_manifest,
        "suite manifest path",
    )
    registry_path = (
        None
        if suite_registry_spec is None
        else _regular_file(suite_registry_spec, "suite registry specification")
    )
    evidence_path = _evidence_output(output)
    _require_distinct_output(
        evidence_path,
        tuple(
            path
            for path in (status_path, manifest_path, registry_path)
            if path is not None
        ),
    )
    checks = {
        "curation_complete": _status_readiness(status_path, manifest_path),
        "suite_complete": _suite_readiness(
            manifest_path,
            registry_path,
            suite_validator=suite_validator,
        ),
    }
    decision = "PASS" if all(checks.values()) else "WAIT"
    return publish_gate_evidence(
        evidence_path,
        "curation-suite-readiness",
        checks,
        decision=decision,
    )


def deployment_hash_validation(
    *,
    manifest: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Verify every hash-bound deployment artifact before publishing PASS."""

    manifest_path = _regular_file(manifest, "deployment manifest")
    evidence_path = _evidence_output(output)
    _require_distinct_output(evidence_path, (manifest_path,))
    canonical_value, before_hash = _load_canonical_object(
        manifest_path,
        "deployment manifest",
    )
    verified = verify_deployment_manifest(manifest_path)
    files = canonical_value.get("files")
    if not isinstance(files, Mapping):
        raise GateRunnerError("deployment manifest has no file inventory")
    for name, record in files.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise GateRunnerError("deployment manifest file inventory is malformed")
        artifact_path = _regular_file(
            record.get("path"),
            f"deployment artifact {name}",
        )
        if file_sha256(artifact_path) != record.get("sha256"):
            raise GateRunnerError(f"deployment artifact {name} changed")
    if verified != canonical_value or file_sha256(manifest_path) != before_hash:
        raise GateRunnerError("deployment manifest changed during verification")
    return publish_gate_evidence(
        evidence_path,
        "deployment-hash-validation",
        {"deployment_valid": True},
    )


def parse_model_binding(value: str) -> ModelBinding:
    """Parse one ``MODEL_PATH=SHA256`` binding without resolving its path."""

    if not isinstance(value, str):
        raise GateRunnerError("model binding must be MODEL_PATH=SHA256")
    raw_path, separator, raw_hash = value.rpartition("=")
    if not separator or not raw_path:
        raise GateRunnerError("model binding must be MODEL_PATH=SHA256")
    path = _regular_file(Path(raw_path), "model binding path")
    expected_hash = _require_sha256(raw_hash, "model binding hash")
    if file_sha256(path) != expected_hash:
        raise GateRunnerError(f"model binding hash mismatch: {path}")
    return ModelBinding(path, expected_hash)


def _coerce_model_bindings(
    values: Sequence[str | ModelBinding | Mapping[str, Any]],
) -> Tuple[ModelBinding, ...]:
    if not values:
        raise GateRunnerError("at least one model binding is required")
    bindings = []
    for index, value in enumerate(values):
        if isinstance(value, ModelBinding):
            path = _regular_file(value.path, f"model binding {index} path")
            digest = _require_sha256(value.sha256, f"model binding {index} hash")
            binding = ModelBinding(path, digest)
        elif isinstance(value, str):
            binding = parse_model_binding(value)
        elif isinstance(value, Mapping):
            if set(value) != {"path", "sha256"}:
                raise GateRunnerError(
                    f"model binding {index} must contain only path and sha256"
                )
            path = _regular_file(value["path"], f"model binding {index} path")
            digest = _require_sha256(
                value["sha256"],
                f"model binding {index} hash",
            )
            binding = ModelBinding(path, digest)
        else:
            raise GateRunnerError(
                f"model binding {index} must be MODEL_PATH=SHA256"
            )
        if file_sha256(binding.path) != binding.sha256:
            raise GateRunnerError(f"model binding hash mismatch: {binding.path}")
        bindings.append(binding)
    if len({item.path for item in bindings}) != len(bindings):
        raise GateRunnerError("model binding paths must be unique")
    if len({item.sha256 for item in bindings}) != len(bindings):
        raise GateRunnerError("model binding hashes must be unique")
    return tuple(sorted(bindings, key=lambda item: item.sha256))


def _run_text_command(
    argv: Sequence[str],
    *,
    role: str,
    subprocess_runner: SubprocessRunner,
) -> str:
    result = subprocess_runner(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise GateRunnerError(f"{role} returned no integer process status")
    if returncode != 0:
        stderr = str(getattr(result, "stderr", "")).strip()
        raise GateRunnerError(
            f"{role} failed with status {returncode}: {stderr}"
        )
    if not isinstance(stdout, str):
        raise GateRunnerError(f"{role} stdout must be text")
    return stdout


def _require_target_gpu_idle(
    gpu_index: int,
    expected_gpu_uuid: str,
    *,
    subprocess_runner: SubprocessRunner,
) -> str:
    inventory = _run_text_command(
        (
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ),
        role="CUDA GPU inventory",
        subprocess_runner=subprocess_runner,
    )
    uuids = [line.strip() for line in inventory.splitlines() if line.strip()]
    if (
        not uuids
        or len(set(uuids)) != len(uuids)
        or any(_GPU_UUID_RE.fullmatch(value) is None for value in uuids)
    ):
        raise GateRunnerError("CUDA GPU inventory is malformed or empty")
    if not 0 <= gpu_index < len(uuids):
        raise GateRunnerError("requested CUDA GPU index is absent")
    observed_uuid = uuids[gpu_index]
    if observed_uuid != expected_gpu_uuid:
        raise GateRunnerError(
            "requested CUDA GPU UUID differs from the expected UUID"
        )
    process_output = _run_text_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ),
        role="CUDA process inventory",
        subprocess_runner=subprocess_runner,
    )
    for line in process_output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if (
            len(fields) != 3
            or _GPU_UUID_RE.fullmatch(fields[0]) is None
            or not fields[1]
            or not fields[2]
        ):
            raise GateRunnerError("CUDA process inventory is malformed")
        if fields[0] == observed_uuid:
            raise GateRunnerError("target CUDA GPU is not exclusively idle")
    return observed_uuid


def _canonical_analysis_output_sha256(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateRunnerError("KataGo analysis output is not UTF-8") from exc
    by_id: Dict[str, Mapping[str, Any]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise GateRunnerError(
                f"KataGo analysis output line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise GateRunnerError("KataGo analysis output row is not an object")
        record_id = row.get("id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in by_id
        ):
            raise GateRunnerError(
                "KataGo analysis output IDs must be unique nonempty strings"
            )
        by_id[record_id] = row
    if not by_id:
        raise GateRunnerError("KataGo analysis output is empty")
    canonical = "".join(
        canonical_json(by_id[record_id]) + "\n"
        for record_id in sorted(by_id)
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class _CanonicalOutputCapturingRunner:
    def __init__(self, delegate: SubprocessRunner) -> None:
        self.delegate = delegate
        self.calls = 0
        self.output_sha256: Optional[str] = None

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls != 1:
            raise GateRunnerError("one model probe launched multiple analysis commands")
        result = self.delegate(argv, **kwargs)
        if getattr(result, "returncode", None) == 0:
            destination = kwargs.get("stdout")
            if destination is None or not hasattr(destination, "name"):
                raise GateRunnerError("analysis runner exposed no output file")
            destination.flush()
            self.output_sha256 = _canonical_analysis_output_sha256(
                Path(destination.name).read_bytes()
            )
        return result


def cuda_model_probes(
    *,
    katago: Path,
    config: Path,
    gpu_index: int,
    expected_gpu_uuid: str,
    output: Path,
    model_bindings: Optional[
        Sequence[str | ModelBinding | Mapping[str, Any]]
    ] = None,
    models: Optional[Sequence[str | ModelBinding | Mapping[str, Any]]] = None,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> Mapping[str, Any]:
    """Probe each hash-bound model twice on one verified idle CUDA GPU."""

    if model_bindings is not None and models is not None:
        raise GateRunnerError("provide model_bindings or models, not both")
    raw_bindings = model_bindings if model_bindings is not None else models
    if raw_bindings is None:
        raise GateRunnerError("at least one model binding is required")
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        raise GateRunnerError("GPU index must be a nonnegative integer")
    if (
        not isinstance(expected_gpu_uuid, str)
        or _GPU_UUID_RE.fullmatch(expected_gpu_uuid) is None
    ):
        raise GateRunnerError("expected GPU UUID is malformed")
    katago_path = _regular_file(katago, "KataGo binary")
    config_path = _regular_file(config, "deterministic analysis config")
    bindings = _coerce_model_bindings(raw_bindings)
    evidence_path = _evidence_output(output)
    _require_distinct_output(
        evidence_path,
        (katago_path, config_path, *(binding.path for binding in bindings)),
    )
    frozen_katago_hash = file_sha256(katago_path)
    frozen_config_hash = file_sha256(config_path)
    observed_uuid = _require_target_gpu_idle(
        gpu_index,
        expected_gpu_uuid,
        subprocess_runner=subprocess_runner,
    )
    model_checks = []
    for binding in bindings:
        output_hashes = []
        for _repeat in range(2):
            if file_sha256(binding.path) != binding.sha256:
                raise GateRunnerError(
                    f"model changed before deterministic probe: {binding.path}"
                )
            capturing_runner = _CanonicalOutputCapturingRunner(subprocess_runner)
            result = probe_model(
                katago=katago_path,
                config=config_path,
                model=binding.path,
                expected_model_sha256=binding.sha256,
                gpu_index=gpu_index,
                require_idle_gpu=False,
                subprocess_runner=capturing_runner,
            )
            if (
                result.get("model_sha256") != binding.sha256
                or result.get("katago_sha256") != frozen_katago_hash
                or result.get("config_sha256") != frozen_config_hash
                or result.get("finite") is not True
                or capturing_runner.output_sha256 is None
            ):
                raise GateRunnerError("model probe result contradicts frozen inputs")
            output_hashes.append(capturing_runner.output_sha256)
            observed_uuid = _require_target_gpu_idle(
                gpu_index,
                expected_gpu_uuid,
                subprocess_runner=subprocess_runner,
            )
        if output_hashes[0] != output_hashes[1]:
            raise GateRunnerError(
                f"model analysis output is not deterministic: {binding.path}"
            )
        if file_sha256(binding.path) != binding.sha256:
            raise GateRunnerError(f"model changed during probe: {binding.path}")
        model_checks.append(
            {
                "model_sha256": binding.sha256,
                "finite": True,
                "deterministic": True,
            }
        )
    if (
        file_sha256(katago_path) != frozen_katago_hash
        or file_sha256(config_path) != frozen_config_hash
    ):
        raise GateRunnerError("KataGo binary or analysis config changed during probes")
    checks = {
        "cuda_available": True,
        "gpu_exclusive": True,
        "gpu_uuid": observed_uuid,
        "models": model_checks,
    }
    return publish_gate_evidence(
        evidence_path,
        "cuda-model-probes",
        checks,
    )


def _candidate_snapshot(
    inbox: Path,
) -> Tuple[Mapping[str, Any], int, int]:
    candidates, ignored = inventory_candidates(inbox)
    rows = [
        {
            "name": candidate.name,
            "path": str(candidate.path),
            "model_sha256": candidate.model_hash,
            "checkpoint_sha256": candidate.checkpoint_hash,
            "directory_manifest_sha256": candidate.directory_manifest_hash,
            "sample_count": candidate.sample_count,
            "data_count": candidate.data_count,
            "size_bytes": candidate.size_bytes,
            "files": [list(item) for item in candidate.files],
        }
        for candidate in candidates
    ]
    snapshot = {
        "candidates": rows,
        "ignored": list(ignored),
        "inventory_sha256": canonical_sha256(
            {"candidates": rows, "ignored": list(ignored)}
        ),
    }
    return snapshot, len(candidates), len(ignored)


def _controller_active_queue(
    status: Mapping[str, Any],
    *,
    ready_count: int,
    ignored_count: int,
) -> Tuple[int, int, str, str, Mapping[str, Any]]:
    if (
        status.get("schema_version") != 1
        or status.get("contract") != "risk-score-controller-status-v1"
    ):
        raise GateRunnerError("controller status contract is unsupported")
    policy_hash = _require_sha256(
        status.get("policy_hash"),
        "controller status policy hash",
    )
    controller_hash = _require_sha256(
        status.get("controller_hash"),
        "controller status controller hash",
    )
    result = status.get("result")
    if not isinstance(result, Mapping) or result.get("mode") != "automatic":
        raise GateRunnerError(
            "controller status must contain an automatic-mode result"
        )
    queue = result.get("queue")
    candidates = result.get("candidates")
    active_evaluations = result.get("activeEvaluations")
    backpressure = result.get("backpressure")
    if (
        not isinstance(queue, Mapping)
        or not isinstance(candidates, list)
        or not isinstance(active_evaluations, list)
        or not isinstance(backpressure, Mapping)
    ):
        raise GateRunnerError(
            "controller status lacks queue, candidate, or backpressure artifacts"
        )

    candidate_states: Dict[str, str] = {}
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise GateRunnerError(f"controller candidate row {index} is malformed")
        candidate_hash = _require_sha256(
            row.get("hash"),
            f"controller candidate row {index} hash",
        )
        state_value = row.get("state")
        if not isinstance(state_value, str) or not state_value:
            raise GateRunnerError(
                f"controller candidate row {index} state is malformed"
            )
        _require_bool(
            row.get("present"),
            f"controller candidate row {index} presence",
        )
        if candidate_hash in candidate_states:
            raise GateRunnerError("controller status repeats a candidate hash")
        candidate_states[candidate_hash] = state_value

    evaluation_hashes = set()
    for index, row in enumerate(active_evaluations):
        if not isinstance(row, Mapping):
            raise GateRunnerError(
                f"controller active-evaluation row {index} is malformed"
            )
        candidate_hash = _require_sha256(
            row.get("candidateHash"),
            f"controller active-evaluation row {index} hash",
        )
        if candidate_hash in evaluation_hashes:
            raise GateRunnerError(
                "controller active evaluations repeat a candidate hash"
            )
        evaluation_hashes.add(candidate_hash)

    active_hashes = {
        digest
        for digest, state_value in candidate_states.items()
        if state_value in _ACTIVE_CANDIDATE_STATES
    }
    evaluating_hashes = {
        digest
        for digest, state_value in candidate_states.items()
        if state_value in _EVALUATING_CANDIDATE_STATES
    }
    if evaluation_hashes != evaluating_hashes:
        raise GateRunnerError(
            "controller active evaluations contradict candidate states"
        )

    pending_count = ready_count + ignored_count
    queue_depth = _require_int(queue.get("depth"), "controller queue depth")
    queue_pending = _require_int(
        queue.get("pendingDepth"),
        "controller pending queue depth",
    )
    queue_ready = _require_int(
        queue.get("readyDepth"),
        "controller ready queue depth",
    )
    queue_ignored = _require_int(
        queue.get("ignoredDepth"),
        "controller ignored queue depth",
    )
    queue_active = _require_int(
        queue.get("activeDepth"),
        "controller active evaluation depth",
    )
    if (
        queue_pending != pending_count
        or queue_ready != ready_count
        or queue_ignored != ignored_count
        or queue_active != len(evaluation_hashes)
        or queue_depth != queue_pending + queue_active
    ):
        raise GateRunnerError(
            "controller queue status does not match the candidate inbox"
        )

    active_depth = _require_int(
        backpressure.get("evaluationBacklogDepth"),
        "controller active queue depth",
    )
    configured_limit = _require_int(
        backpressure.get("maximumActiveEvaluatorEntries"),
        "controller active queue limit",
        minimum=1,
    )
    if active_depth != len(active_hashes):
        raise GateRunnerError(
            "controller backpressure depth contradicts active candidate states"
        )
    for key in (
        "allowExport",
        "allowEvaluation",
        "exportPaused",
        "evaluationPaused",
    ):
        _require_bool(
            backpressure.get(key),
            f"controller backpressure {key}",
        )
    return (
        active_depth,
        configured_limit,
        policy_hash,
        controller_hash,
        backpressure,
    )


def _backpressure_is_enforced(
    value: Mapping[str, Any],
    *,
    policy_hash: str,
    controller_hash: str,
    controller_status: Optional[Mapping[str, Any]],
    controller_limit: int,
    maximum_active_queue: int,
) -> bool:
    if value.get("schema_version") != 1:
        raise GateRunnerError("backpressure artifact schema is unsupported")
    observed_policy_hash = _require_sha256(
        value.get("policy_hash"),
        "backpressure policy hash",
    )
    observed_controller_hash = _require_sha256(
        value.get("controller_hash"),
        "backpressure controller hash",
    )
    allow_export = _require_bool(
        value.get("allowExport"),
        "backpressure allowExport",
    )
    allow_evaluation = _require_bool(
        value.get("allowEvaluation"),
        "backpressure allowEvaluation",
    )
    export_paused = _require_bool(
        value.get("exportPaused"),
        "backpressure exportPaused",
    )
    evaluation_paused = _require_bool(
        value.get("evaluationPaused"),
        "backpressure evaluationPaused",
    )
    _require_int(
        value.get("exportBacklogDepth"),
        "backpressure export backlog depth",
    )
    _require_int(
        value.get("evaluationBacklogDepth"),
        "backpressure evaluation backlog depth",
    )
    artifact_limit = _require_int(
        value.get("maximumActiveEvaluatorEntries"),
        "backpressure active evaluator limit",
        minimum=1,
    )
    reasons = value.get("reasons")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise GateRunnerError(
            "fail-closed backpressure must contain at least one reason"
        )
    shared_fields = (
        "allowExport",
        "allowEvaluation",
        "exportPaused",
        "evaluationPaused",
        "exportBacklogDepth",
        "evaluationBacklogDepth",
        "maximumActiveEvaluatorEntries",
        "importantQueueWarningDepth",
        "reasons",
    )
    status_agrees = controller_status is None or all(
        value.get(key) == controller_status.get(key) for key in shared_fields
    )
    return (
        observed_policy_hash == policy_hash
        and observed_controller_hash == controller_hash
        and controller_limit == maximum_active_queue
        and artifact_limit == maximum_active_queue
        and status_agrees
        and allow_export is False
        and allow_evaluation is False
        and export_paused is True
        and evaluation_paused is True
    )


def _production_target_is_inactive(
    unit: str,
    *,
    subprocess_runner: SubprocessRunner,
) -> bool:
    if not isinstance(unit, str) or not unit or any(
        character.isspace() for character in unit
    ):
        raise GateRunnerError("training target unit is malformed")
    result = subprocess_runner(
        ["systemctl", "is-active", unit],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    returncode = getattr(result, "returncode", None)
    status = str(getattr(result, "stdout", "")).strip()
    return (
        isinstance(returncode, int)
        and not isinstance(returncode, bool)
        and returncode != 0
        and status in {"inactive", "unknown", "not-found"}
    )


def backlog_bound(
    *,
    inbox: Path,
    controller_status: Optional[Path],
    backpressure: Path,
    maximum_candidates: int,
    maximum_active_queue: int,
    output: Path,
    training_target_unit: str = "katago-risk-training.target",
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> Mapping[str, Any]:
    """Publish actual stable backlog counts and observed fail-closed controls."""

    candidate_inbox = _regular_directory(inbox, "candidate inbox")
    status_path = (
        None
        if controller_status is None
        else _regular_file(controller_status, "controller status")
    )
    backpressure_path = _regular_file(
        backpressure,
        "backpressure artifact",
    )
    evidence_path = _evidence_output(output)
    _require_distinct_output(
        evidence_path,
        tuple(
            path
            for path in (status_path, backpressure_path)
            if path is not None
        ),
    )
    if _strictly_within(evidence_path, candidate_inbox):
        raise GateRunnerError("evidence output must not be inside the candidate inbox")
    maximum_candidate_count = _require_int(
        maximum_candidates,
        "maximum candidate count",
    )
    maximum_queue_count = _require_int(
        maximum_active_queue,
        "maximum active queue count",
        minimum=1,
    )

    first_snapshot, ready_count, ignored_count = _candidate_snapshot(
        candidate_inbox
    )
    backpressure_value, backpressure_file_hash = _load_canonical_object(
        backpressure_path,
        "backpressure artifact",
    )
    if status_path is None:
        if not _production_target_is_inactive(
            training_target_unit,
            subprocess_runner=subprocess_runner,
        ):
            raise GateRunnerError(
                "automatic controller status is absent while the training "
                "target is not provably inactive"
            )
        active_depth = 0
        controller_limit = maximum_queue_count
        policy_hash = _require_sha256(
            backpressure_value.get("policy_hash"),
            "backpressure policy hash",
        )
        controller_hash = _require_sha256(
            backpressure_value.get("controller_hash"),
            "backpressure controller hash",
        )
        controller_backpressure = None
        status_file_hash = None
    else:
        status, status_file_hash = _load_canonical_object(
            status_path,
            "controller status",
        )
        (
            active_depth,
            controller_limit,
            policy_hash,
            controller_hash,
            controller_backpressure,
        ) = _controller_active_queue(
            status,
            ready_count=ready_count,
            ignored_count=ignored_count,
        )
    backpressure_enforced = _backpressure_is_enforced(
        backpressure_value,
        policy_hash=policy_hash,
        controller_hash=controller_hash,
        controller_status=controller_backpressure,
        controller_limit=controller_limit,
        maximum_active_queue=maximum_queue_count,
    )

    second_snapshot, second_ready, second_ignored = _candidate_snapshot(
        candidate_inbox
    )
    final_status_hash = None
    if status_path is not None:
        _, final_status_hash = _load_canonical_object(
            status_path,
            "controller status",
        )
    _, final_backpressure_hash = _load_canonical_object(
        backpressure_path,
        "backpressure artifact",
    )
    evidence_preserved = (
        first_snapshot == second_snapshot
        and status_file_hash == final_status_hash
        and backpressure_file_hash == final_backpressure_hash
    )
    candidate_count = second_ready + second_ignored
    checks = {
        "candidate_count": candidate_count,
        "maximum_candidates": maximum_candidate_count,
        "active_queue_depth": active_depth,
        "maximum_active_queue": maximum_queue_count,
        "backpressure_enforced": backpressure_enforced,
        "evidence_preserved": evidence_preserved,
    }
    decision = (
        "PASS"
        if (
            candidate_count <= maximum_candidate_count
            and active_depth <= maximum_queue_count
            and backpressure_enforced
            and evidence_preserved
        )
        else "FAIL"
    )
    return publish_gate_evidence(
        evidence_path,
        "backlog-bound",
        checks,
        decision=decision,
    )


# Descriptive aliases for callers that treat each function as a gate command.
run_curation_suite_readiness = curation_suite_readiness
run_deployment_hash_validation = deployment_hash_validation
run_cuda_model_probes = cuda_model_probes
run_backlog_bound = backlog_bound


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = _GateArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    readiness = subparsers.add_parser(
        "curation-suite-readiness",
    )
    readiness.add_argument("--curation-status", required=True, type=Path)
    readiness.add_argument("--suite-manifest", required=True, type=Path)
    readiness.add_argument("--suite-registry-spec", type=Path)
    readiness.add_argument("--output", required=True, type=Path)

    deployment = subparsers.add_parser(
        "deployment-hash-validation",
    )
    deployment.add_argument("--manifest", required=True, type=Path)
    deployment.add_argument("--output", required=True, type=Path)

    cuda = subparsers.add_parser(
        "cuda-model-probes",
    )
    cuda.add_argument("--katago", required=True, type=Path)
    cuda.add_argument("--config", required=True, type=Path)
    cuda.add_argument("--gpu-index", required=True, type=int)
    cuda.add_argument("--expected-gpu-uuid", required=True)
    cuda.add_argument(
        "--model",
        "--model-binding",
        dest="model_bindings",
        action="append",
        required=True,
        metavar="MODEL_PATH=SHA256",
    )
    cuda.add_argument("--output", required=True, type=Path)

    backlog = subparsers.add_parser(
        "backlog-bound",
    )
    backlog.add_argument(
        "--inbox",
        "--candidate-inbox",
        dest="inbox",
        required=True,
        type=Path,
    )
    backlog.add_argument("--controller-status", type=Path)
    backlog.add_argument(
        "--backpressure",
        "--backpressure-status",
        dest="backpressure",
        required=True,
        type=Path,
    )
    backlog.add_argument("--maximum-candidates", required=True, type=int)
    backlog.add_argument("--maximum-active-queue", required=True, type=int)
    backlog.add_argument(
        "--training-target-unit",
        default="katago-risk-training.target",
    )
    backlog.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> int:
    try:
        args = parse_args(argv)
        if args.command == "curation-suite-readiness":
            result = curation_suite_readiness(
                curation_status=args.curation_status,
                suite_manifest=args.suite_manifest,
                suite_registry_spec=args.suite_registry_spec,
                output=args.output,
            )
        elif args.command == "deployment-hash-validation":
            result = deployment_hash_validation(
                manifest=args.manifest,
                output=args.output,
            )
        elif args.command == "cuda-model-probes":
            result = cuda_model_probes(
                katago=args.katago,
                config=args.config,
                gpu_index=args.gpu_index,
                expected_gpu_uuid=args.expected_gpu_uuid,
                model_bindings=args.model_bindings,
                output=args.output,
                subprocess_runner=subprocess_runner,
            )
        else:
            result = backlog_bound(
                inbox=args.inbox,
                controller_status=args.controller_status,
                backpressure=args.backpressure,
                maximum_candidates=args.maximum_candidates,
                maximum_active_queue=args.maximum_active_queue,
                output=args.output,
                training_target_unit=args.training_target_unit,
                subprocess_runner=subprocess_runner,
            )
    except (
        GateRunnerError,
        HostCommandError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
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
