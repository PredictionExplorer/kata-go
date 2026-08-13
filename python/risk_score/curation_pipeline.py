"""Restartable coordination from advisory curation sources to frozen suites.

The coordinator intentionally contains no position-classification logic.  It
validates a frozen, canonical specification and composes the existing
``queries-consensus``, :class:`CurationOrchestrator`, ``label-consensus``,
``merge-labeling-consensus``, ``finalize-consensus``, and suite-builder
primitives.

Specification contract ``risk-score-curation-pipeline-spec-v1``::

    {
      "schema_version": 1,
      "contract": "risk-score-curation-pipeline-spec-v1",
      "deployment": {
        "repository_path": "/absolute/deployed/repository",
        "source_revision": "<git object id>",
        "source_sha256": "<sha256(source_revision)>"
      },
      "deployment_manifest": {
        "path": "/absolute/deployment-manifest.json",
        "sha256": "..."
      },
      "run_root": "/absolute/live/run/root",
      "policy": {"path": "/absolute/policy.json", "sha256": "..."},
      "katago": {"path": "/absolute/katago", "sha256": "..."},
      "analysis_config": {"path": "/absolute/analysis.cfg", "sha256": "..."},
      "models": {
        "original": {"path": "/absolute/original.bin.gz", "sha256": "..."},
        "champion": {"path": "/absolute/champion.bin.gz", "sha256": "..."}
      },
      "sources": [{
        "name": "ordinary-primary",
        "label": "ordinary",
        "selected": {"path": "/absolute/selected.jsonl", "sha256": "..."},
        "prefilter_manifest": {
          "path": "/absolute/selected.manifest.json",
          "sha256": "..."
        }
      }],
      "work_root": "/absolute/live/run/root/evaluation/curation/pipeline",
      "outputs": {
        "reviewed_bank": "/absolute/.../source-positions.jsonl",
        "reviewed_manifest": "/absolute/.../source-positions.manifest.json",
        "suite_directory": "/absolute/.../promotion-suites-v3"
      },
      "quotas": {"ordinary": 3200, "lead-40": 2080, "lead-80": 4128},
      "topology": {
        "shards_per_role": 8,
        "gpus": ["0", "1"],
        "per_gpu_parallelism": 1
      },
      "suite_seed": "risk-score-promotion-v3",
      "spec_sha256": "<canonical payload sha256>"
    }

All JSON specifications and mutable status snapshots are compact, sorted-key,
newline-terminated canonical JSON.  Immutable artifacts, rather than status,
are authoritative on every restart.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production curation runs on Unix.
    fcntl = None  # type: ignore[assignment]

from risk_score import consensus_prefilter as _consensus_prefilter
from risk_score.board_symmetry import symmetry_orbit_sha256
from risk_score.build_evaluation_suites import (
    MACHINE_GENERATOR_CONTRACT,
    MACHINE_MANIFEST_CONTRACT,
    build_evaluation_suites,
)
from risk_score.build_live_runtime import verify_deployment_manifest
from risk_score.consensus_prefilter import (
    ALLOWED_LABELS,
    PREFILTER_CONTRACT,
    PREFILTER_ROLES,
)
from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    CONSENSUS_COMBINED_LABELING_CONTRACT,
    CONSENSUS_FINAL_MANIFEST_CONTRACT,
    CONSENSUS_LABELING_CONTRACT,
    _canonical_jsonl,
    _load_consensus_labeling_artifacts,
    _machine_curation_policy,
    _normalized_positions,
    _validate_consensus_query_bundle,
    finalize_consensus_reviewed_bank,
    generate_consensus_query_bundle,
    label_positions_consensus,
    load_policy,
    merge_consensus_labeling_bundles,
    policy_pool_minima,
    validate_deterministic_analysis_config,
)
from risk_score.curation_orchestrator import (
    STATUS_CONTRACT as ORCHESTRATOR_STATUS_CONTRACT,
)
from risk_score.curation_orchestrator import (
    CurationOrchestrator,
)
from risk_score.curation_supplement import (
    SUMMARY_CONTRACT as SUPPLEMENT_SUMMARY_CONTRACT,
)
from risk_score.curation_supplement import load_supplement_spec
from risk_score.gpu_lease import GpuLeaseError, LeaseRecord, ProcessIdentity
from risk_score.position_samples import (
    canonical_json,
    canonical_sha256,
    file_sha256,
    semantic_position_sha256,
)

SPEC_SCHEMA_VERSION = 1
SPEC_CONTRACT = "risk-score-curation-pipeline-spec-v1"
STATUS_SCHEMA_VERSION = 1
STATUS_CONTRACT = "risk-score-curation-pipeline-status-v1"
GPU_OWNERSHIP_SCHEMA_VERSION = 1
GPU_OWNERSHIP_CONTRACT = "risk-score-curation-gpu-ownership-v1"
POLICY_MINIMA: Mapping[str, int] = {
    "ordinary": 3200,
    "lead-40": 2080,
    "lead-80": 4128,
}
EXPECTED_LABELS = tuple(sorted(POLICY_MINIMA))
PIPELINE_SPEC_CONTRACT = SPEC_CONTRACT
PIPELINE_STATUS_CONTRACT = STATUS_CONTRACT
REQUIRED_MINIMA = POLICY_MINIMA

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SOURCE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_EXECUTED_MODULES = (
    "board_symmetry.py",
    "build_evaluation_suites.py",
    "curate_position_bank.py",
    "curation_orchestrator.py",
    "curation_pipeline.py",
    "curation_supplement.py",
    "position_samples.py",
)

_SPEC_KEYS = {
    "schema_version",
    "contract",
    "deployment",
    "deployment_manifest",
    "run_root",
    "policy",
    "katago",
    "analysis_config",
    "models",
    "sources",
    "work_root",
    "outputs",
    "quotas",
    "topology",
    "suite_seed",
    "spec_sha256",
}


class CurationPipelineError(RuntimeError):
    """Base class for coordinator failures."""


class PipelineSpecError(CurationPipelineError, ValueError):
    """The frozen pipeline specification is malformed or stale."""


class PipelineContradiction(CurationPipelineError, ValueError):
    """An existing artifact contradicts the frozen pipeline specification."""


class PipelineBusy(CurationPipelineError):
    """Another coordinator currently owns the pipeline lock."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class DeploymentBinding:
    repository_path: Path
    source_revision: str
    source_sha256: str


@dataclass(frozen=True)
class SourceSpec:
    name: str
    label: str
    selected: FileBinding
    prefilter_manifest: FileBinding
    supplement_summary: FileBinding | None = None


@dataclass(frozen=True)
class Topology:
    shards_per_role: int
    gpus: tuple[str, ...]
    per_gpu_parallelism: int


@dataclass(frozen=True)
class OutputSpec:
    reviewed_bank: Path
    reviewed_manifest: Path
    suite_directory: Path


@dataclass(frozen=True)
class PipelineSpec:
    path: Path
    file_sha256: str
    identity: str
    raw: Mapping[str, Any]
    deployment: DeploymentBinding
    deployment_manifest: FileBinding
    run_root: Path
    policy: FileBinding
    katago: FileBinding
    analysis_config: FileBinding
    models: Mapping[str, FileBinding]
    sources: tuple[SourceSpec, ...]
    work_root: Path
    outputs: OutputSpec
    quotas: Mapping[str, int]
    topology: Topology
    suite_seed: str


@dataclass(frozen=True)
class SourceLayout:
    root: Path
    query_directory: Path
    query_manifest: Path
    consensus_work: Path
    labeling_directory: Path


@dataclass(frozen=True)
class PipelineLayout:
    sources: Mapping[str, SourceLayout]
    combined_labeling: Path
    gpu_ownership: Path
    status: Path
    lock: Path


@dataclass(frozen=True)
class SourceInventory:
    name: str
    label: str
    row_count: int
    semantic_ids: tuple[str, ...]
    symmetry_orbits: tuple[str, ...]


@dataclass(frozen=True)
class SourceProgress:
    name: str
    label: str
    selected_count: int
    queries_complete: bool
    consensus_complete: bool
    labeling_complete: bool
    accepted_count: int | None = None
    rejected_count: int | None = None


@dataclass(frozen=True)
class PipelineSnapshot:
    selected_counts: Mapping[str, int]
    deficits: Mapping[str, int]
    sources: tuple[SourceProgress, ...]
    combined_required: bool
    combined_complete: bool
    reviewed_complete: bool
    suite_complete: bool
    accepted_counts: Mapping[str, int] | None = None


@dataclass(frozen=True)
class StageAction:
    kind: str
    source: str | None = None


def _default_consensus_runner(
    *,
    query_manifest_path: Path,
    work_dir: Path,
    shard_count: int,
    gpus: Sequence[str],
    per_gpu_parallelism: int,
    poll_interval: float,
) -> Mapping[str, Any]:
    return CurationOrchestrator(
        query_manifest_path=query_manifest_path,
        work_dir=work_dir,
        shard_count=shard_count,
        gpus=gpus,
        per_gpu_parallelism=per_gpu_parallelism,
    ).watch(poll_interval=poll_interval)


@dataclass(frozen=True)
class PipelineRunners:
    """Injectable stage boundaries; defaults are the trusted production APIs."""

    queries: Callable[..., Mapping[str, Any]] = generate_consensus_query_bundle
    consensus: Callable[..., Mapping[str, Any]] = _default_consensus_runner
    label: Callable[..., Mapping[str, Any]] = label_positions_consensus
    merge: Callable[..., Mapping[str, Any]] = merge_consensus_labeling_bundles
    finalize: Callable[..., Mapping[str, Any]] = finalize_consensus_reviewed_bank
    suites: Callable[..., Any] = build_evaluation_suites


DEFAULT_RUNNERS = PipelineRunners()


@dataclass(frozen=True)
class GpuComputeProcess:
    gpu_uuid: str
    pid: int
    process_name: str


@dataclass(frozen=True)
class GpuOccupancy:
    index_to_uuid: Mapping[str, str]
    processes: tuple[GpuComputeProcess, ...] = ()


def nvidia_smi_gpu_occupancy() -> GpuOccupancy:
    """Inventory CUDA index/UUID bindings and active compute processes."""

    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if inventory.returncode != 0:
        raise PipelineContradiction("cannot inventory CUDA GPUs with nvidia-smi")
    index_to_uuid: dict[str, str] = {}
    for line in inventory.stdout.splitlines():
        columns = [column.strip() for column in line.split(",", 1)]
        if len(columns) != 2 or not columns[0] or not columns[1]:
            raise PipelineContradiction("nvidia-smi GPU inventory is malformed")
        if columns[0] in index_to_uuid or columns[1] in index_to_uuid.values():
            raise PipelineContradiction("nvidia-smi GPU inventory contains duplicates")
        index_to_uuid[columns[0]] = columns[1]
    processes_result = subprocess.run(
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
    if processes_result.returncode != 0:
        raise PipelineContradiction("cannot inspect CUDA compute processes")
    processes: list[GpuComputeProcess] = []
    for line in processes_result.stdout.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.split(",", 2)]
        if len(columns) != 3 or not columns[0] or not columns[1]:
            raise PipelineContradiction("nvidia-smi process inventory is malformed")
        try:
            pid = int(columns[1])
        except ValueError as exc:
            raise PipelineContradiction(
                "nvidia-smi process inventory contains an invalid PID"
            ) from exc
        processes.append(GpuComputeProcess(columns[0], pid, columns[2]))
    return GpuOccupancy(index_to_uuid, tuple(processes))


def production_target_is_inactive() -> bool:
    """Conservatively prove the production training target is not using GPUs.

    The target is installed only when the closed-loop services are activated.
    Before activation, systemd reports ``not-found`` (exit code 4), which is
    equivalent to inactive for GPU ownership. Unexpected states still fail
    closed.
    """

    result = subprocess.run(
        ["systemctl", "is-active", "katago-risk-training.target"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode == 0:
        return False
    return result.stdout.strip() in {"inactive", "not-found", "unknown"}


def gpu7_lease_is_trainer_safe(run_root: Path) -> bool:
    """Accept an absent lease or a non-handoff trainer state."""

    path = Path(run_root) / "promotion" / "gpu-lease.json"
    if not _lexists(path):
        return True
    if path.is_symlink() or not path.is_file():
        return False
    try:
        value = _load_json_object(path, "GPU7 lease state", canonical=True)
        record = LeaseRecord.from_dict(value)
    except (GpuLeaseError, OSError, TypeError, ValueError):
        return False
    return (
        record.phase == "trainer_running"
        and not record.safety_halt
        and not record.evaluators
    )


def current_process_identity() -> ProcessIdentity:
    return ProcessIdentity.capture(os.getpid())


@dataclass(frozen=True)
class OwnershipProbes:
    gpu_occupancy: Callable[[], GpuOccupancy] = nvidia_smi_gpu_occupancy
    current_process: Callable[[], ProcessIdentity] = current_process_identity
    process_identity: Callable[[int], ProcessIdentity] = ProcessIdentity.capture
    gpu7_lease_safe: Callable[[Path], bool] = gpu7_lease_is_trainer_safe
    production_target_inactive: Callable[[], bool] = production_target_is_inactive
    sleep: Callable[[float], None] = time.sleep
    global_lock_path: Path = Path("/run/lock/kata-go-risk-curation-gpu.lock")


DEFAULT_OWNERSHIP_PROBES = OwnershipProbes()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _load_json_object(path: Path, role: str, *, canonical: bool) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {role} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must have an object root")
    if canonical and data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_canonical_jsonl(
    path: Path, role: str, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot load {role} {source}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"{role} contains a blank row at line {line_number}")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{role} has invalid JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{role} row {line_number} must be an object")
        rows.append(row)
    expected = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    if data != expected or (not rows and not allow_empty):
        qualifier = "nonempty " if not allow_empty else ""
        raise ValueError(f"{role} must be {qualifier}canonical JSONL")
    return rows


def _require_exact_keys(value: Any, keys: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineSpecError(f"{role} must be an object")
    missing = sorted(keys.difference(value))
    extra = sorted(set(value).difference(keys))
    if missing or extra:
        raise PipelineSpecError(
            f"{role} keys differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PipelineSpecError(f"{role} must be a lowercase 64-character SHA-256")
    return value


def _canonical_path(raw: Any, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PipelineSpecError(f"{role} must be a nonempty absolute path")
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve()) != raw:
        raise PipelineSpecError(
            f"{role} must be an absolute canonical path with no symlink components"
        )
    return path


def _required_directory(raw: Any, role: str) -> Path:
    path = _canonical_path(raw, role)
    if path.is_symlink() or not path.is_dir():
        raise PipelineSpecError(f"{role} must be an existing non-symlink directory")
    return path


def _future_directory(raw: Any, role: str) -> Path:
    path = _canonical_path(raw, role)
    if _lexists(path) and (path.is_symlink() or not path.is_dir()):
        raise PipelineSpecError(f"{role} must be a non-symlink directory when present")
    return path


def _future_file(raw: Any, role: str) -> Path:
    path = _canonical_path(raw, role)
    if _lexists(path) and (path.is_symlink() or not path.is_file()):
        raise PipelineSpecError(f"{role} must be a non-symlink file when present")
    return path


def _file_binding(value: Any, role: str) -> FileBinding:
    binding = _require_exact_keys(value, {"path", "sha256"}, role)
    path = _canonical_path(binding["path"], f"{role} path")
    digest = _require_sha256(binding["sha256"], f"{role} hash")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise PipelineSpecError(f"{role} file is missing or does not match its hash")
    return FileBinding(path=path, sha256=digest)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def git_revision(repository: Path) -> str:
    """Return the deployed checkout's current Git object id."""

    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise PipelineSpecError(
            f"cannot resolve deployment revision: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def git_status_porcelain(repository: Path) -> str:
    """Return tracked and untracked deployment checkout changes."""

    completed = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        raise PipelineSpecError(
            f"cannot inspect deployment checkout: {completed.stderr.strip()}"
        )
    return completed.stdout


def _verify_bound_deployment_manifest(
    binding: FileBinding,
    deployment: DeploymentBinding,
    *,
    error_type: type[CurationPipelineError],
) -> Mapping[str, Any]:
    if (
        binding.path.is_symlink()
        or not binding.path.is_file()
        or file_sha256(binding.path) != binding.sha256
    ):
        raise error_type("deployment manifest is missing or changed")
    try:
        manifest = verify_deployment_manifest(binding.path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise error_type(f"deployment manifest verification failed: {exc}") from exc
    if (
        manifest.get("source_revision") != deployment.source_revision
        or manifest.get("source_sha256") != deployment.source_sha256
    ):
        raise error_type("deployment manifest does not bind the frozen revision")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise error_type("deployment manifest file inventory is malformed")
    for module_name in _EXECUTED_MODULES:
        key = f"module:{module_name}"
        artifact = files.get(key)
        expected_path = (
            deployment.repository_path / "python" / "risk_score" / module_name
        )
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("path") != str(expected_path)
            or artifact.get("sha256") != file_sha256(expected_path)
        ):
            raise error_type(f"deployment manifest does not bind {module_name}")
    return manifest


def load_pipeline_spec(
    path: Path,
    *,
    revision_reader: Callable[[Path], str] = git_revision,
    repository_status_reader: Callable[[Path], str] = git_status_porcelain,
) -> PipelineSpec:
    """Load and fully validate the strict canonical v1 specification."""

    spec_path = Path(path)
    try:
        raw = _load_json_object(
            spec_path, "curation pipeline specification", canonical=True
        )
    except ValueError as exc:
        raise PipelineSpecError(str(exc)) from exc
    _require_exact_keys(raw, _SPEC_KEYS, "pipeline specification")
    if raw.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise PipelineSpecError("pipeline specification schema_version must be 1")
    if raw.get("contract") != SPEC_CONTRACT:
        raise PipelineSpecError("pipeline specification contract is unsupported")
    payload = dict(raw)
    supplied_identity = payload.pop("spec_sha256", None)
    if _require_sha256(
        supplied_identity, "pipeline specification identity"
    ) != canonical_sha256(payload):
        raise PipelineSpecError("pipeline specification self-hash is invalid")

    deployment_value = _require_exact_keys(
        raw["deployment"],
        {"repository_path", "source_revision", "source_sha256"},
        "deployment binding",
    )
    repository = _required_directory(
        deployment_value["repository_path"], "deployment repository"
    )
    revision = deployment_value["source_revision"]
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise PipelineSpecError(
            "deployment source_revision must be a lowercase Git hash"
        )
    revision_hash = _require_sha256(
        deployment_value["source_sha256"], "deployment source hash"
    )
    if revision_hash != hashlib.sha256(revision.encode("utf-8")).hexdigest():
        raise PipelineSpecError("deployment source hash does not bind source_revision")
    actual_revision = revision_reader(repository)
    if actual_revision != revision:
        raise PipelineSpecError(
            f"deployment repository is at {actual_revision!r}, expected {revision!r}"
        )
    deployment = DeploymentBinding(repository, revision, revision_hash)
    if repository_status_reader(repository):
        raise PipelineSpecError("deployment repository has uncommitted changes")
    deployment_manifest = _file_binding(
        raw["deployment_manifest"], "deployment manifest"
    )
    _verify_bound_deployment_manifest(
        deployment_manifest,
        deployment,
        error_type=PipelineSpecError,
    )

    run_root = _required_directory(raw["run_root"], "run root")
    policy = _file_binding(raw["policy"], "promotion policy")
    katago = _file_binding(raw["katago"], "KataGo binary")
    analysis_config = _file_binding(raw["analysis_config"], "analysis config")
    try:
        validate_deterministic_analysis_config(analysis_config.path)
    except ValueError as exc:
        raise PipelineSpecError(f"analysis config is not deterministic: {exc}") from exc

    models_value = _require_exact_keys(
        raw["models"], {"original", "champion"}, "model bindings"
    )
    models = {
        role: _file_binding(models_value[role], f"{role} model")
        for role in ("original", "champion")
    }
    if (
        models["original"].path == models["champion"].path
        or models["original"].sha256 == models["champion"].sha256
    ):
        raise PipelineSpecError("original and champion models must be distinct")

    sources_value = raw["sources"]
    if not isinstance(sources_value, list) or not sources_value:
        raise PipelineSpecError("sources must be a nonempty array")
    sources: list[SourceSpec] = []
    names = set()
    for index, source_value in enumerate(sources_value):
        if not isinstance(source_value, Mapping):
            raise PipelineSpecError(f"source {index} must be an object")
        required_source_keys = {"name", "label", "selected", "prefilter_manifest"}
        allowed_source_keys = required_source_keys | {"supplement_summary"}
        if not required_source_keys.issubset(source_value) or not set(
            source_value
        ).issubset(allowed_source_keys):
            raise PipelineSpecError(f"source {index} keys differ from the contract")
        source = dict(
            source_value,
        )
        name = source["name"]
        if not isinstance(name, str) or _SOURCE_NAME_RE.fullmatch(name) is None:
            raise PipelineSpecError(
                f"source {index} name must be a lowercase filesystem-safe identifier"
            )
        if name in names:
            raise PipelineSpecError(f"duplicate source name: {name}")
        names.add(name)
        label = source["label"]
        if label not in ALLOWED_LABELS:
            raise PipelineSpecError(
                f"source {name} label must be one of {sorted(ALLOWED_LABELS)}"
            )
        sources.append(
            SourceSpec(
                name=name,
                label=label,
                selected=_file_binding(source["selected"], f"source {name} selected"),
                prefilter_manifest=_file_binding(
                    source["prefilter_manifest"],
                    f"source {name} prefilter manifest",
                ),
                supplement_summary=(
                    None
                    if "supplement_summary" not in source
                    else _file_binding(
                        source["supplement_summary"],
                        f"source {name} supplement summary",
                    )
                ),
            )
        )
    if [source.name for source in sources] != sorted(source.name for source in sources):
        raise PipelineSpecError("sources must be sorted by name")

    work_root = _future_directory(raw["work_root"], "pipeline work root")
    if not _is_within(work_root, run_root) or work_root == run_root:
        raise PipelineSpecError("pipeline work root must be strictly beneath run root")

    outputs_value = _require_exact_keys(
        raw["outputs"],
        {"reviewed_bank", "reviewed_manifest", "suite_directory"},
        "pipeline outputs",
    )
    outputs = OutputSpec(
        reviewed_bank=_future_file(outputs_value["reviewed_bank"], "reviewed bank"),
        reviewed_manifest=_future_file(
            outputs_value["reviewed_manifest"], "reviewed bank manifest"
        ),
        suite_directory=_future_directory(
            outputs_value["suite_directory"], "suite directory"
        ),
    )
    if outputs.reviewed_bank.name != "source-positions.jsonl":
        raise PipelineSpecError("reviewed bank must be named source-positions.jsonl")
    if outputs.reviewed_manifest.name != "source-positions.manifest.json":
        raise PipelineSpecError(
            "reviewed manifest must be named source-positions.manifest.json"
        )
    if outputs.reviewed_bank.parent != outputs.reviewed_manifest.parent:
        raise PipelineSpecError("reviewed bank and manifest must share a directory")
    for output, role in (
        (outputs.reviewed_bank, "reviewed bank"),
        (outputs.reviewed_manifest, "reviewed manifest"),
        (outputs.suite_directory, "suite directory"),
    ):
        if not _is_within(output, run_root) or output == run_root:
            raise PipelineSpecError(f"{role} must be strictly beneath run root")
        if _paths_overlap(output, work_root):
            raise PipelineSpecError(f"{role} may not overlap pipeline work root")
    if (
        outputs.reviewed_bank == outputs.reviewed_manifest
        or _paths_overlap(outputs.suite_directory, outputs.reviewed_bank)
        or _paths_overlap(outputs.suite_directory, outputs.reviewed_manifest)
    ):
        raise PipelineSpecError("pipeline output paths overlap")

    quotas_value = _require_exact_keys(
        raw["quotas"], set(POLICY_MINIMA), "source quotas"
    )
    quotas = {label: quotas_value[label] for label in EXPECTED_LABELS}
    if quotas != dict(sorted(POLICY_MINIMA.items())):
        raise PipelineSpecError(
            f"source quotas must equal frozen v3 minima {dict(POLICY_MINIMA)}"
        )

    topology_value = _require_exact_keys(
        raw["topology"],
        {"shards_per_role", "gpus", "per_gpu_parallelism"},
        "shard/GPU topology",
    )
    shards = topology_value["shards_per_role"]
    parallelism = topology_value["per_gpu_parallelism"]
    if type(shards) is not int or not 1 <= shards <= 64:
        raise PipelineSpecError("shards_per_role must be between 1 and 64")
    if type(parallelism) is not int or not 1 <= parallelism <= 64:
        raise PipelineSpecError("per_gpu_parallelism must be between 1 and 64")
    raw_gpus = topology_value["gpus"]
    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise PipelineSpecError("topology gpus must be a nonempty array")
    gpus: list[str] = []
    for gpu in raw_gpus:
        if (
            not isinstance(gpu, str)
            or not gpu
            or gpu.strip() != gpu
            or "," in gpu
            or any(character.isspace() for character in gpu)
            or gpu in gpus
        ):
            raise PipelineSpecError("GPU identifiers must be unique nonempty tokens")
        gpus.append(gpu)
    topology = Topology(shards, tuple(gpus), parallelism)

    suite_seed = raw["suite_seed"]
    if (
        not isinstance(suite_seed, str)
        or not suite_seed
        or "\n" in suite_seed
        or "\r" in suite_seed
    ):
        raise PipelineSpecError("suite_seed must be a nonempty single-line string")

    try:
        policy_value = load_policy(policy.path)
        _machine_curation_policy(policy_value)
        minima = policy_pool_minima(policy_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineSpecError(
            f"promotion policy is not a valid v3 policy: {exc}"
        ) from exc
    if minima != dict(POLICY_MINIMA):
        raise PipelineSpecError(
            f"promotion policy minima are {minima}, expected {dict(POLICY_MINIMA)}"
        )

    protected_inputs = {
        deployment_manifest.path,
        policy.path,
        katago.path,
        analysis_config.path,
        models["original"].path,
        models["champion"].path,
        *(source.selected.path for source in sources),
        *(source.prefilter_manifest.path for source in sources),
        *(
            source.supplement_summary.path
            for source in sources
            if source.supplement_summary is not None
        ),
    }
    if (
        outputs.reviewed_bank in protected_inputs
        or outputs.reviewed_manifest in protected_inputs
    ):
        raise PipelineSpecError("pipeline outputs may not replace frozen inputs")

    return PipelineSpec(
        path=spec_path.resolve(),
        file_sha256=file_sha256(spec_path),
        identity=supplied_identity,
        raw=raw,
        deployment=deployment,
        deployment_manifest=deployment_manifest,
        run_root=run_root,
        policy=policy,
        katago=katago,
        analysis_config=analysis_config,
        models=models,
        sources=tuple(sources),
        work_root=work_root,
        outputs=outputs,
        quotas=quotas,
        topology=topology,
        suite_seed=suite_seed,
    )


# Concise aliases for callers that prefer generic loader names.
load_spec = load_pipeline_spec


def pipeline_layout(spec: PipelineSpec) -> PipelineLayout:
    sources = {}
    for source in spec.sources:
        root = spec.work_root / "sources" / source.name
        query_directory = root / "query-bundle-v2"
        sources[source.name] = SourceLayout(
            root=root,
            query_directory=query_directory,
            query_manifest=query_directory / "manifest.json",
            consensus_work=root / "consensus-work-v2",
            labeling_directory=root / "labeling-v2",
        )
    return PipelineLayout(
        sources=sources,
        combined_labeling=spec.work_root / "labeling-combined-v2",
        gpu_ownership=spec.run_root / "evaluation" / ".curation-gpu-ownership.json",
        status=spec.work_root / "status.json",
        lock=spec.work_root / ".pipeline.lock",
    )


def _bound_regular_file(path: Path, expected_hash: str, role: str) -> Path:
    if (
        not path.is_absolute()
        or str(path.resolve()) != str(path)
        or path.is_symlink()
        or not path.is_file()
        or file_sha256(path) != expected_hash
    ):
        raise PipelineContradiction(f"{role} changed or has an unsafe path")
    return path


def _manifest_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PipelineContradiction(f"{role} path is missing")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise PipelineContradiction(f"{role} path is not absolute and canonical")
    return path


def _validate_prefilter_with_central_api(
    spec: PipelineSpec,
    source: SourceSpec,
) -> None:
    """Narrow adapter for the shared prefilter validator when present."""

    validator = getattr(_consensus_prefilter, "validate_prefilter_artifact", None)
    if validator is None:
        return
    validator(
        source.prefilter_manifest.path,
        expected_label=source.label,
        expected_model_hashes={
            role: spec.models[role].sha256 for role in ("original", "champion")
        },
    )


def _verify_transitive_summary_bindings(value: Any, *, role: str) -> None:
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            if "path" not in value or "sha256" not in value:
                raise PipelineContradiction(
                    f"{role} contains a partial transitive artifact binding"
                )
            path = _manifest_path(value.get("path"), f"{role} artifact")
            digest = _require_sha256(value.get("sha256"), f"{role} artifact hash")
            _bound_regular_file(path, digest, f"{role} artifact")
        for key, nested in value.items():
            _verify_transitive_summary_bindings(nested, role=f"{role}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _verify_transitive_summary_bindings(
                nested,
                role=f"{role}[{index}]",
            )


def _verify_supplement_summary(spec: PipelineSpec, source: SourceSpec) -> None:
    binding = source.supplement_summary
    if binding is None:
        return
    summary_path = _bound_regular_file(
        binding.path,
        binding.sha256,
        f"{source.name} supplement summary",
    )
    summary = _load_json_object(
        summary_path,
        f"{source.name} supplement summary",
        canonical=True,
    )
    payload = dict(summary)
    supplied_hash = payload.pop("summary_sha256", None)
    expected_summary_keys = {
        "schema_version",
        "contract",
        "spec",
        "state",
        "primary_counts",
        "target_counts",
        "generation_limits",
        "supplemental_counts",
        "final_counts",
        "primary_prefilter_manifests",
        "selfplay",
        "harvest",
        "normalized",
        "query_bundle",
        "analyses",
        "selected",
        "summary_sha256",
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("schema_version") != 1
        or summary.get("contract") != SUPPLEMENT_SUMMARY_CONTRACT
        or supplied_hash != canonical_sha256(payload)
        or summary.get("state") not in {"complete", "insufficient_candidates"}
    ):
        raise PipelineContradiction(
            f"{source.name} supplement summary contract or self-hash changed"
        )
    _verify_transitive_summary_bindings(
        summary,
        role=f"{source.name} supplement summary",
    )
    supplement_spec_binding = summary.get("spec")
    if not isinstance(supplement_spec_binding, Mapping):
        raise PipelineContradiction(
            f"{source.name} supplement summary spec ancestry is missing"
        )
    supplement_spec_path = _manifest_path(
        supplement_spec_binding.get("path"),
        f"{source.name} supplement specification",
    )
    try:
        supplement_spec = load_supplement_spec(supplement_spec_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PipelineContradiction(
            f"{source.name} supplement specification is invalid: {exc}"
        ) from exc
    if (
        supplement_spec_binding.get("sha256") != supplement_spec.file_sha256
        or supplement_spec_binding.get("identity") != supplement_spec.identity
        or supplement_spec.run_root != spec.run_root
        or (supplement_spec.katago.path, supplement_spec.katago.sha256)
        != (spec.katago.path, spec.katago.sha256)
        or (
            supplement_spec.analysis_config.path,
            supplement_spec.analysis_config.sha256,
        )
        != (spec.analysis_config.path, spec.analysis_config.sha256)
        or (supplement_spec.policy.path, supplement_spec.policy.sha256)
        != (spec.policy.path, spec.policy.sha256)
        or any(
            (
                supplement_spec.models[role].path,
                supplement_spec.models[role].sha256,
            )
            != (spec.models[role].path, spec.models[role].sha256)
            for role in ("original", "champion")
        )
    ):
        raise PipelineContradiction(
            f"{source.name} supplement deployment/model/policy ancestry changed"
        )
    expected_primary = [
        {
            "label": primary.label,
            "row_count": primary.row_count,
            "path": str(primary.manifest.path),
            "sha256": primary.manifest.sha256,
        }
        for primary in supplement_spec.primary_prefilters
    ]
    if (
        summary.get("target_counts") != dict(supplement_spec.target_counts)
        or summary.get("primary_prefilter_manifests") != expected_primary
    ):
        raise PipelineContradiction(
            f"{source.name} supplement primary/policy ancestry changed"
        )
    selected = summary.get("selected")
    selected_entry = (
        selected.get(source.label) if isinstance(selected, Mapping) else None
    )
    if not isinstance(selected_entry, Mapping):
        raise PipelineContradiction(
            f"{source.name} supplement selected label is missing"
        )
    if selected_entry.get("output") != {
        "path": str(source.selected.path),
        "sha256": source.selected.sha256,
    } or selected_entry.get("manifest") != {
        "path": str(source.prefilter_manifest.path),
        "sha256": source.prefilter_manifest.sha256,
    }:
        raise PipelineContradiction(
            f"{source.name} supplement selected output binding changed"
        )
    for artifact_role in ("selfplay", "harvest", "normalized", "query_bundle"):
        if not isinstance(summary.get(artifact_role), Mapping):
            raise PipelineContradiction(
                f"{source.name} supplement {artifact_role} ancestry is missing"
            )
    analyses = summary.get("analyses")
    if not isinstance(analyses, Mapping) or set(analyses) != set(PREFILTER_ROLES):
        raise PipelineContradiction(
            f"{source.name} supplement analysis ancestry is incomplete"
        )


def _verify_prefilter_source(spec: PipelineSpec, source: SourceSpec) -> SourceInventory:
    try:
        _validate_prefilter_with_central_api(spec, source)
        _verify_supplement_summary(spec, source)
        selected_path = _bound_regular_file(
            source.selected.path,
            source.selected.sha256,
            f"{source.name} selected source",
        )
        manifest_path = _bound_regular_file(
            source.prefilter_manifest.path,
            source.prefilter_manifest.sha256,
            f"{source.name} prefilter manifest",
        )
        manifest = _load_json_object(
            manifest_path, f"{source.name} prefilter manifest", canonical=True
        )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("contract") != PREFILTER_CONTRACT
            or manifest.get("advisory_only") is not True
            or manifest.get("requires_full_machine_consensus") is not True
            or manifest.get("label") != source.label
        ):
            raise PipelineContradiction(
                f"{source.name} prefilter contract or expected label changed"
            )

        _load_canonical_jsonl(
            selected_path, f"{source.name} selected normalized source"
        )
        positions = _normalized_positions(selected_path)
        if selected_path.read_bytes() != _canonical_jsonl(positions):
            raise PipelineContradiction(
                f"{source.name} selected source is not normalized canonical JSONL"
            )
        semantic_ids = tuple(position["semanticSha256"] for position in positions)
        orbit_ids = tuple(symmetry_orbit_sha256(position) for position in positions)
        if len(orbit_ids) != len(set(orbit_ids)):
            raise PipelineContradiction(
                f"{source.name} selected source contains symmetry duplicates"
            )

        selected = manifest.get("selected")
        if (
            not isinstance(selected, Mapping)
            or selected.get("path") != str(selected_path)
            or selected.get("sha256") != source.selected.sha256
            or selected.get("row_count") != len(positions)
            or selected.get("symmetry_orbit_count") != len(set(orbit_ids))
        ):
            raise PipelineContradiction(
                f"{source.name} prefilter selected inventory changed"
            )

        normalized = manifest.get("normalized")
        if not isinstance(normalized, Mapping):
            raise PipelineContradiction(
                f"{source.name} prefilter normalized provenance is missing"
            )
        normalized_path = _manifest_path(
            normalized.get("path"), f"{source.name} prefilter normalized source"
        )
        normalized_hash = _require_sha256(
            normalized.get("sha256"), f"{source.name} normalized source hash"
        )
        _bound_regular_file(
            normalized_path, normalized_hash, f"{source.name} normalized source"
        )
        normalized_positions = _normalized_positions(normalized_path)
        normalized_ids = [
            position["semanticSha256"] for position in normalized_positions
        ]
        if (
            normalized.get("row_count") != len(normalized_positions)
            or normalized.get("semantic_ids_sha256") != canonical_sha256(normalized_ids)
            or not set(semantic_ids).issubset(normalized_ids)
        ):
            raise PipelineContradiction(
                f"{source.name} prefilter normalized inventory changed"
            )

        expected_models = {
            role: spec.models[role].sha256 for role in ("original", "champion")
        }
        if manifest.get("model_hashes") != expected_models:
            raise PipelineContradiction(
                f"{source.name} prefilter names unexpected model hashes"
            )
        analyses = manifest.get("analyses")
        if not isinstance(analyses, Mapping) or set(analyses) != set(PREFILTER_ROLES):
            raise PipelineContradiction(
                f"{source.name} prefilter analysis inventory is incomplete"
            )
        for role in PREFILTER_ROLES:
            identity = analyses[role]
            if not isinstance(identity, Mapping):
                raise PipelineContradiction(
                    f"{source.name} prefilter analysis {role} is malformed"
                )
            model = role.split("/", 1)[0]
            output_path = _manifest_path(
                identity.get("path"), f"{source.name} {role} analysis"
            )
            output_hash = _require_sha256(
                identity.get("sha256"), f"{source.name} {role} analysis hash"
            )
            execution_path = _manifest_path(
                identity.get("manifest_path"),
                f"{source.name} {role} execution manifest",
            )
            execution_hash = _require_sha256(
                identity.get("manifest_sha256"),
                f"{source.name} {role} execution manifest hash",
            )
            query_path = _manifest_path(
                identity.get("query_path"), f"{source.name} {role} query"
            )
            query_hash = _require_sha256(
                identity.get("query_sha256"), f"{source.name} {role} query hash"
            )
            _bound_regular_file(
                output_path, output_hash, f"{source.name} {role} analysis"
            )
            _bound_regular_file(
                execution_path,
                execution_hash,
                f"{source.name} {role} execution manifest",
            )
            _bound_regular_file(query_path, query_hash, f"{source.name} {role} query")
            if (
                identity.get("model_sha256") != expected_models[model]
                or identity.get("config_sha256") != spec.analysis_config.sha256
            ):
                raise PipelineContradiction(
                    f"{source.name} {role} prefilter execution identity changed"
                )
            execution = _load_json_object(
                execution_path,
                f"{source.name} {role} execution manifest",
                canonical=False,
            )
            if (
                execution.get("contract") != ANALYSIS_RUN_CONTRACT
                or execution.get("output_path") != str(output_path)
                or execution.get("output_sha256") != output_hash
                or execution.get("query_path") != str(query_path)
                or execution.get("query_sha256") != query_hash
                or execution.get("model_sha256") != expected_models[model]
                or execution.get("katago_sha256") != identity.get("katago_sha256")
                or execution.get("config_sha256") != spec.analysis_config.sha256
            ):
                raise PipelineContradiction(
                    f"{source.name} {role} execution provenance changed"
                )
        return SourceInventory(
            name=source.name,
            label=source.label,
            row_count=len(positions),
            semantic_ids=semantic_ids,
            symmetry_orbits=orbit_ids,
        )
    except PipelineContradiction:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PipelineContradiction(
            f"{source.name} prefilter provenance is invalid: {exc}"
        ) from exc


def verify_prefilter_sources(spec: PipelineSpec) -> tuple[SourceInventory, ...]:
    """Verify every advisory source and reject cross-source duplicate positions."""

    inventories = tuple(
        _verify_prefilter_source(spec, source) for source in spec.sources
    )
    seen_semantic: dict[str, str] = {}
    seen_orbits: dict[str, str] = {}
    for inventory in inventories:
        for semantic_id, orbit_id in zip(
            inventory.semantic_ids, inventory.symmetry_orbits, strict=True
        ):
            previous = seen_semantic.get(semantic_id)
            if previous is not None:
                raise PipelineContradiction(
                    f"source {inventory.name} duplicates semantic position "
                    f"from {previous}"
                )
            previous_orbit = seen_orbits.get(orbit_id)
            if previous_orbit is not None:
                raise PipelineContradiction(
                    f"source {inventory.name} duplicates symmetry orbit "
                    f"from {previous_orbit}"
                )
            seen_semantic[semantic_id] = inventory.name
            seen_orbits[orbit_id] = inventory.name
    return inventories


def selected_counts(
    inventories: Sequence[SourceInventory],
) -> Mapping[str, int]:
    return {
        label: sum(item.row_count for item in inventories if item.label == label)
        for label in EXPECTED_LABELS
    }


def source_deficits(
    counts: Mapping[str, int], quotas: Mapping[str, int] = POLICY_MINIMA
) -> Mapping[str, int]:
    return {
        label: int(quotas[label]) - int(counts.get(label, 0))
        for label in EXPECTED_LABELS
        if int(counts.get(label, 0)) < int(quotas[label])
    }


def plan_next_stage(snapshot: PipelineSnapshot) -> StageAction:
    """Return the next authoritative stage using only immutable-artifact state."""

    if snapshot.deficits:
        return StageAction("blocked_insufficient_sources")
    for source in snapshot.sources:
        if not source.queries_complete:
            return StageAction("create_queries_consensus", source.name)
        if not source.consensus_complete:
            return StageAction("run_consensus", source.name)
        if not source.labeling_complete:
            return StageAction("label_consensus", source.name)
    if snapshot.combined_required and not snapshot.combined_complete:
        return StageAction("merge_labeling_consensus")
    if not snapshot.reviewed_complete:
        return StageAction("finalize_consensus")
    if not snapshot.suite_complete:
        return StageAction("build_evaluation_suites")
    return StageAction("complete")


def _query_context(
    spec: PipelineSpec, source: SourceSpec, layout: SourceLayout
) -> Mapping[str, Any] | None:
    if not _lexists(layout.query_directory):
        return None
    _require_regular_directory(layout.query_directory, f"{source.name} query bundle")
    try:
        context = _validate_consensus_query_bundle(
            source.selected.path, layout.query_manifest
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineContradiction(
            f"{source.name} query bundle contradicts the specification: {exc}"
        ) from exc
    manifest = context["manifest"]
    if (
        manifest.get("normalized_path") != str(source.selected.path)
        or manifest.get("normalized_sha256") != source.selected.sha256
        or manifest.get("katago_path") != str(spec.katago.path)
        or manifest.get("katago_sha256") != spec.katago.sha256
        or manifest.get("analysis_config_path") != str(spec.analysis_config.path)
        or manifest.get("analysis_config_sha256") != spec.analysis_config.sha256
        or manifest.get("policy_path") != str(spec.policy.path)
        or manifest.get("policy_sha256") != spec.policy.sha256
        or {
            role: manifest.get("models", {}).get(role, {}).get("sha256")
            for role in ("original", "champion")
        }
        != {role: spec.models[role].sha256 for role in ("original", "champion")}
    ):
        raise PipelineContradiction(
            f"{source.name} query bundle uses different frozen inputs"
        )
    return context


def _inspect_consensus(
    spec: PipelineSpec,
    source: SourceSpec,
    layout: SourceLayout,
    query_context: Mapping[str, Any],
) -> bool:
    if not _lexists(layout.consensus_work):
        return False
    _require_regular_directory(
        layout.consensus_work, f"{source.name} consensus work path"
    )
    status_path = layout.consensus_work / "status.json"
    if not _lexists(status_path):
        return False
    try:
        status = _load_json_object(
            status_path, f"{source.name} consensus status", canonical=True
        )
    except ValueError as exc:
        raise PipelineContradiction(str(exc)) from exc
    payload = dict(status)
    supplied_hash = payload.pop("status_sha256", None)
    query_binding = status.get("query_bundle")
    scheduler = status.get("scheduler")
    if (
        status.get("contract") != ORCHESTRATOR_STATUS_CONTRACT
        or supplied_hash != canonical_sha256(payload)
        or not isinstance(query_binding, Mapping)
        or query_binding.get("path") != str(layout.query_manifest)
        or query_binding.get("sha256") != query_context["manifest_file_sha256"]
        or query_binding.get("identity") != query_context["manifest_identity"]
        or status.get("work_directory") != str(layout.consensus_work)
        or status.get("shards_per_role") != spec.topology.shards_per_role
        or not isinstance(scheduler, Mapping)
        or scheduler.get("gpus") != list(spec.topology.gpus)
        or scheduler.get("per_gpu_parallelism") != spec.topology.per_gpu_parallelism
    ):
        raise PipelineContradiction(
            f"{source.name} consensus status contradicts shard/GPU topology"
        )
    if status.get("state") != "complete":
        return False
    if status.get("ready_for_labeling") is not True:
        raise PipelineContradiction(
            f"{source.name} complete consensus status is not ready for labeling"
        )
    expected_roles = set(query_context["manifest"]["queries"])
    roles = status.get("roles")
    outputs = status.get("analysis_outputs")
    if (
        not isinstance(roles, Mapping)
        or set(roles) != expected_roles
        or not isinstance(outputs, Mapping)
        or set(outputs) != expected_roles
    ):
        raise PipelineContradiction(
            f"{source.name} complete consensus role inventory changed"
        )
    for role in sorted(expected_roles):
        role_status = roles[role]
        if not isinstance(role_status, Mapping):
            raise PipelineContradiction(
                f"{source.name} consensus role {role} is malformed"
            )
        expected_output = layout.consensus_work.joinpath(
            "roles", *role.split("/"), "analysis.jsonl"
        )
        manifest_path = Path(str(expected_output) + ".manifest.json")
        merged = role_status.get("merged")
        model = role.split("/", 1)[0]
        if (
            outputs.get(role) != str(expected_output)
            or role_status.get("merged_output_path") != str(expected_output)
            or role_status.get("model_sha256") != spec.models[model].sha256
            or not isinstance(merged, Mapping)
            or merged.get("path") != str(expected_output)
            or merged.get("manifest_path") != str(manifest_path)
        ):
            raise PipelineContradiction(
                f"{source.name} consensus role {role} output binding changed"
            )
        output_hash = merged.get("sha256")
        manifest_hash = merged.get("manifest_sha256")
        if (
            not isinstance(output_hash, str)
            or _SHA256_RE.fullmatch(output_hash) is None
            or not isinstance(manifest_hash, str)
            or _SHA256_RE.fullmatch(manifest_hash) is None
        ):
            raise PipelineContradiction(
                f"{source.name} consensus role {role} hashes are malformed"
            )
        _bound_regular_file(
            expected_output, output_hash, f"{source.name} {role} merged analysis"
        )
        _bound_regular_file(
            manifest_path,
            manifest_hash,
            f"{source.name} {role} merged analysis manifest",
        )
        try:
            manifest = _load_json_object(
                manifest_path,
                f"{source.name} {role} merged analysis manifest",
                canonical=True,
            )
        except ValueError as exc:
            raise PipelineContradiction(str(exc)) from exc
        manifest_payload = dict(manifest)
        manifest_identity = manifest_payload.pop("manifest_sha256", None)
        query_artifact = query_context["manifest"]["queries"][role]
        if (
            manifest.get("contract") != ANALYSIS_RUN_CONTRACT
            or manifest_identity != canonical_sha256(manifest_payload)
            or manifest.get("katago_sha256") != spec.katago.sha256
            or manifest.get("config_sha256") != spec.analysis_config.sha256
            or manifest.get("model_sha256") != spec.models[model].sha256
            or manifest.get("query_sha256") != query_artifact["sha256"]
            or manifest.get("output_path") != str(expected_output)
            or manifest.get("output_sha256") != output_hash
        ):
            raise PipelineContradiction(
                f"{source.name} consensus role {role} merged provenance changed"
            )
    return True


def _inspect_labeling(
    spec: PipelineSpec,
    source: SourceSpec,
    layout: SourceLayout,
    query_context: Mapping[str, Any],
) -> tuple[bool, int | None, int | None]:
    if not _lexists(layout.labeling_directory):
        return False, None, None
    _require_regular_directory(
        layout.labeling_directory, f"{source.name} labeling bundle"
    )
    try:
        bundle = _load_consensus_labeling_artifacts(
            machine_path=layout.labeling_directory / "machine-labeled.jsonl",
            rejected_path=layout.labeling_directory / "rejected.jsonl",
            manifest_path=layout.labeling_directory / "manifest.json",
            role=f"{source.name} consensus labeling",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineContradiction(
            f"{source.name} labeling bundle contradicts its ancestry: {exc}"
        ) from exc
    manifest = bundle["manifest"]
    if (
        manifest.get("contract") != CONSENSUS_LABELING_CONTRACT
        or manifest.get("normalized_path") != str(source.selected.path)
        or manifest.get("normalized_sha256") != source.selected.sha256
        or manifest.get("query_manifest_path") != str(layout.query_manifest)
        or manifest.get("query_manifest_sha256")
        != query_context["manifest_file_sha256"]
        or manifest.get("query_manifest_identity") != query_context["manifest_identity"]
        or bundle["policy_hash"] != canonical_sha256(load_policy(spec.policy.path))
        or {role: bundle["models"][role]["sha256"] for role in ("original", "champion")}
        != {role: spec.models[role].sha256 for role in ("original", "champion")}
    ):
        raise PipelineContradiction(
            f"{source.name} labeling bundle uses different frozen coordinates"
        )
    for _, row in bundle["machine"]:
        if row.get("labels") != [source.label]:
            raise PipelineContradiction(
                f"{source.name} accepted a row outside expected label {source.label}"
            )
    return True, len(bundle["machine"]), len(bundle["rejected"])


def _inspect_combined(
    spec: PipelineSpec, layout: PipelineLayout
) -> tuple[bool, Mapping[str, Any] | None]:
    if len(spec.sources) == 1:
        if _lexists(layout.combined_labeling):
            raise PipelineContradiction(
                "combined labeling bundle exists although only one source is configured"
            )
        return True, None
    if not _lexists(layout.combined_labeling):
        return False, None
    _require_regular_directory(layout.combined_labeling, "combined labeling path")
    try:
        bundle = _load_consensus_labeling_artifacts(
            machine_path=layout.combined_labeling / "machine-labeled.jsonl",
            rejected_path=layout.combined_labeling / "rejected.jsonl",
            manifest_path=layout.combined_labeling / "manifest.json",
            role="combined consensus labeling",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineContradiction(
            f"combined labeling bundle contradicts source bundles: {exc}"
        ) from exc
    manifest = bundle["manifest"]
    expected_roots = sorted(
        str(layout.sources[source.name].labeling_directory.resolve())
        for source in spec.sources
    )
    actual_roots = sorted(
        item.get("path")
        for item in manifest.get("source_bundles", [])
        if isinstance(item, Mapping)
    )
    if (
        manifest.get("contract") != CONSENSUS_COMBINED_LABELING_CONTRACT
        or actual_roots != expected_roots
        or bundle["policy_hash"] != canonical_sha256(load_policy(spec.policy.path))
        or {role: bundle["models"][role]["sha256"] for role in ("original", "champion")}
        != {role: spec.models[role].sha256 for role in ("original", "champion")}
    ):
        raise PipelineContradiction("combined labeling coordinates changed")
    return True, bundle


def _labeling_input(spec: PipelineSpec, layout: PipelineLayout) -> Path:
    return (
        layout.sources[spec.sources[0].name].labeling_directory
        if len(spec.sources) == 1
        else layout.combined_labeling
    )


def _inspect_reviewed(
    spec: PipelineSpec, layout: PipelineLayout
) -> tuple[bool, Mapping[str, int] | None]:
    output_exists = _lexists(spec.outputs.reviewed_bank)
    manifest_exists = _lexists(spec.outputs.reviewed_manifest)
    if not output_exists and not manifest_exists:
        return False, None
    if output_exists != manifest_exists:
        raise PipelineContradiction(
            "reviewed bank and manifest are only partly present"
        )
    try:
        rows = _load_canonical_jsonl(
            spec.outputs.reviewed_bank, "reviewed source-position bank"
        )
        manifest = _load_json_object(
            spec.outputs.reviewed_manifest,
            "reviewed source-position manifest",
            canonical=True,
        )
    except ValueError as exc:
        raise PipelineContradiction(str(exc)) from exc
    payload = dict(manifest)
    identity = payload.pop("manifest_sha256", None)
    for index, row in enumerate(rows, start=1):
        labels = row.get("labels")
        curation = row.get("curation")
        if (
            not isinstance(labels, list)
            or len(labels) != 1
            or labels[0] not in POLICY_MINIMA
            or not isinstance(curation, Mapping)
            or curation.get("classification") != "machine-reviewed"
            or curation.get("review_mode") != "machine-consensus"
            or curation.get("consensus_rules_version") != 1
            or curation.get("semanticSha256") != semantic_position_sha256(row)
            or curation.get("symmetryOrbitSha256") != symmetry_orbit_sha256(row)
        ):
            raise PipelineContradiction(
                f"reviewed bank row {index} is not a valid machine-reviewed position"
            )
    counts = Counter(label for row in rows for label in row["labels"])
    semantic_ids = [
        row.get("curation", {}).get("semanticSha256")
        if isinstance(row.get("curation"), Mapping)
        else None
        for row in rows
    ]
    orbit_ids = [
        row.get("curation", {}).get("symmetryOrbitSha256")
        if isinstance(row.get("curation"), Mapping)
        else None
        for row in rows
    ]
    labeling_root = _labeling_input(spec, layout)
    labeling_manifest = labeling_root / "manifest.json"
    try:
        labeling_value = _load_json_object(
            labeling_manifest, "final labeling input manifest", canonical=True
        )
    except ValueError as exc:
        raise PipelineContradiction(str(exc)) from exc
    labeling_payload = dict(labeling_value)
    labeling_identity = labeling_payload.pop("manifest_sha256", None)
    expected_counts = {label: counts[label] for label in sorted(counts)}
    models = manifest.get("models")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("contract") != CONSENSUS_FINAL_MANIFEST_CONTRACT
        or identity != canonical_sha256(payload)
        or manifest.get("policy_path") != str(spec.policy.path)
        or manifest.get("policy_sha256") != spec.policy.sha256
        or manifest.get("policy_hash")
        != canonical_sha256(load_policy(spec.policy.path))
        or manifest.get("labeling_manifest_path") != str(labeling_manifest.resolve())
        or manifest.get("labeling_manifest_sha256") != file_sha256(labeling_manifest)
        or manifest.get("labeling_manifest_identity") != labeling_identity
        or manifest.get("output_path") != str(spec.outputs.reviewed_bank)
        or manifest.get("output_sha256") != file_sha256(spec.outputs.reviewed_bank)
        or manifest.get("row_count") != len(rows)
        or manifest.get("label_counts") != expected_counts
        or manifest.get("required_minima") != dict(POLICY_MINIMA)
        or manifest.get("semantic_hashes_sha256") != canonical_sha256(semantic_ids)
        or manifest.get("symmetry_orbits_sha256") != canonical_sha256(orbit_ids)
        or len(semantic_ids) != len(set(semantic_ids))
        or len(orbit_ids) != len(set(orbit_ids))
        or not isinstance(models, Mapping)
        or {
            role: {
                "role": models.get(role, {}).get("role"),
                "path": models.get(role, {}).get("path"),
                "sha256": models.get(role, {}).get("sha256"),
            }
            if isinstance(models.get(role), Mapping)
            else None
            for role in ("original", "champion")
        }
        != {
            "original": {
                "role": "immutable_original",
                "path": str(spec.models["original"].path),
                "sha256": spec.models["original"].sha256,
            },
            "champion": {
                "role": "frozen_champion",
                "path": str(spec.models["champion"].path),
                "sha256": spec.models["champion"].sha256,
            },
        }
    ):
        raise PipelineContradiction("reviewed bank provenance or inventory changed")
    deficits = source_deficits(counts, spec.quotas)
    if deficits:
        raise PipelineContradiction(
            f"reviewed bank is below frozen policy minima: {deficits}"
        )
    return True, {label: counts[label] for label in EXPECTED_LABELS}


def _inspect_suite(spec: PipelineSpec) -> bool:
    output = spec.outputs.suite_directory
    if not _lexists(output):
        return False
    _require_regular_directory(output, "suite output")
    manifest_path = output / "manifest.json"
    try:
        manifest = _load_json_object(
            manifest_path, "evaluation suite manifest", canonical=True
        )
    except ValueError as exc:
        raise PipelineContradiction(str(exc)) from exc
    payload = dict(manifest)
    identity = payload.pop("manifestPayloadSha256", None)
    sources = manifest.get("sources")
    curation_sources = manifest.get("curationSources")
    with spec.outputs.reviewed_bank.open("rb") as reviewed_handle:
        reviewed_row_count = sum(1 for _ in reviewed_handle)
    expected_source = {
        "name": spec.outputs.reviewed_bank.name,
        "sha256": file_sha256(spec.outputs.reviewed_bank),
        "rowCount": reviewed_row_count,
        "blankLineCount": 0,
    }
    if (
        manifest.get("schemaVersion") != 3
        or manifest.get("manifestContract") != MACHINE_MANIFEST_CONTRACT
        or manifest.get("generatorContract") != MACHINE_GENERATOR_CONTRACT
        or identity != canonical_sha256(payload)
        or manifest.get("seed") != spec.suite_seed
        or manifest.get("policy_hash")
        != canonical_sha256(load_policy(spec.policy.path))
        or manifest.get("machineReviewOnly") is not True
        or sources != [expected_source]
        or not isinstance(curation_sources, list)
        or len(curation_sources) != 1
        or not isinstance(curation_sources[0], Mapping)
        or curation_sources[0].get("source_name") != spec.outputs.reviewed_bank.name
        or curation_sources[0].get("contract") != CONSENSUS_FINAL_MANIFEST_CONTRACT
        or curation_sources[0].get("review_mode") != "machine-consensus"
        or curation_sources[0].get("consensus_rules_version") != 1
        or curation_sources[0].get("policy_hash")
        != canonical_sha256(load_policy(spec.policy.path))
        or curation_sources[0].get("output_sha256") != expected_source["sha256"]
        or curation_sources[0].get("manifest_sha256")
        != file_sha256(spec.outputs.reviewed_manifest)
    ):
        raise PipelineContradiction("evaluation suite manifest coordinates changed")
    banks = manifest.get("banks")
    cells = manifest.get("cells")
    if not isinstance(banks, list) or not isinstance(cells, list):
        raise PipelineContradiction("suite bank or cell inventory is malformed")
    for bank in banks:
        if not isinstance(bank, Mapping):
            raise PipelineContradiction("suite bank inventory is malformed")
        for artifact_name in ("positions", "schedule"):
            artifact = bank.get(artifact_name)
            if not isinstance(artifact, Mapping):
                raise PipelineContradiction("suite bank artifact is malformed")
            relative = artifact.get("path")
            digest = artifact.get("sha256")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
            ):
                raise PipelineContradiction("suite bank artifact binding is unsafe")
            _bound_regular_file(
                output / relative, digest, f"suite {artifact_name} artifact"
            )
    checked_schedules = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise PipelineContradiction("suite cell inventory is malformed")
        relative = cell.get("schedule_path")
        digest = cell.get("schedule_hash")
        if relative in checked_schedules:
            continue
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
        ):
            raise PipelineContradiction("suite cell schedule binding is unsafe")
        _bound_regular_file(output / relative, digest, "suite cell schedule")
        checked_schedules.add(relative)
    return True


def _require_regular_directory(path: Path, role: str) -> None:
    if (
        not _lexists(path)
        or path.is_symlink()
        or not path.is_dir()
        or str(path.resolve()) != str(path)
    ):
        raise PipelineContradiction(f"{role} is not a canonical non-symlink directory")


def infer_pipeline_snapshot(
    spec: PipelineSpec,
    *,
    inventories: Sequence[SourceInventory] | None = None,
) -> PipelineSnapshot:
    """Reconstruct progress exclusively from frozen inputs and artifacts."""

    current_inventories = (
        tuple(inventories)
        if inventories is not None
        else verify_prefilter_sources(spec)
    )
    counts = selected_counts(current_inventories)
    deficits = source_deficits(counts, spec.quotas)
    inventory_by_name = {item.name: item for item in current_inventories}
    layout = pipeline_layout(spec)
    if deficits:
        return PipelineSnapshot(
            selected_counts=counts,
            deficits=deficits,
            sources=tuple(
                SourceProgress(
                    name=source.name,
                    label=source.label,
                    selected_count=inventory_by_name[source.name].row_count,
                    queries_complete=False,
                    consensus_complete=False,
                    labeling_complete=False,
                )
                for source in spec.sources
            ),
            combined_required=len(spec.sources) > 1,
            combined_complete=False,
            reviewed_complete=False,
            suite_complete=False,
        )

    progress: list[SourceProgress] = []
    for source in spec.sources:
        source_layout = layout.sources[source.name]
        context = _query_context(spec, source, source_layout)
        if context is None:
            if _lexists(source_layout.consensus_work) or _lexists(
                source_layout.labeling_directory
            ):
                raise PipelineContradiction(
                    f"{source.name} has downstream artifacts without a query bundle"
                )
            progress.append(
                SourceProgress(
                    name=source.name,
                    label=source.label,
                    selected_count=inventory_by_name[source.name].row_count,
                    queries_complete=False,
                    consensus_complete=False,
                    labeling_complete=False,
                )
            )
            continue
        consensus_complete = _inspect_consensus(spec, source, source_layout, context)
        if not consensus_complete and _lexists(source_layout.labeling_directory):
            raise PipelineContradiction(
                f"{source.name} has labeling without complete consensus analysis"
            )
        labeling_complete = False
        accepted = None
        rejected = None
        if consensus_complete:
            labeling_complete, accepted, rejected = _inspect_labeling(
                spec, source, source_layout, context
            )
        progress.append(
            SourceProgress(
                name=source.name,
                label=source.label,
                selected_count=inventory_by_name[source.name].row_count,
                queries_complete=True,
                consensus_complete=consensus_complete,
                labeling_complete=labeling_complete,
                accepted_count=accepted,
                rejected_count=rejected,
            )
        )

    all_labeled = all(item.labeling_complete for item in progress)
    if not all_labeled and _lexists(layout.combined_labeling):
        raise PipelineContradiction(
            "combined labeling exists before every source labeling bundle is complete"
        )
    combined_complete = len(spec.sources) == 1 and all_labeled
    combined_bundle: Mapping[str, Any] | None = None
    if all_labeled:
        combined_complete, combined_bundle = _inspect_combined(spec, layout)

    if not combined_complete and (
        _lexists(spec.outputs.reviewed_bank) or _lexists(spec.outputs.reviewed_manifest)
    ):
        raise PipelineContradiction(
            "reviewed bank exists before labeling merge is complete"
        )
    reviewed_complete = False
    accepted_counts: Mapping[str, int] | None = None
    if combined_complete:
        reviewed_complete, accepted_counts = _inspect_reviewed(spec, layout)

    if not reviewed_complete and _lexists(spec.outputs.suite_directory):
        raise PipelineContradiction("suite exists before the reviewed bank is complete")
    suite_complete = _inspect_suite(spec) if reviewed_complete else False
    if accepted_counts is None and combined_bundle is not None:
        accepted_counter = Counter(
            row["labels"][0] for _, row in combined_bundle["machine"]
        )
        accepted_counts = {label: accepted_counter[label] for label in EXPECTED_LABELS}
    return PipelineSnapshot(
        selected_counts=counts,
        deficits={},
        sources=tuple(progress),
        combined_required=len(spec.sources) > 1,
        combined_complete=combined_complete,
        reviewed_complete=reviewed_complete,
        suite_complete=suite_complete,
        accepted_counts=accepted_counts,
    )


def analysis_outputs_for(
    query_manifest: Mapping[str, Any], source_layout: SourceLayout
) -> Mapping[str, Path]:
    return {
        role: source_layout.consensus_work.joinpath(
            "roles", *role.split("/"), "analysis.jsonl"
        ).resolve()
        for role in sorted(query_manifest["queries"])
    }


def dispatch_stage(
    action: StageAction,
    spec: PipelineSpec,
    *,
    layout: PipelineLayout | None = None,
    runners: PipelineRunners = DEFAULT_RUNNERS,
    poll_interval: float = 30.0,
) -> Any:
    """Dispatch one planned stage; useful for runner-only unit tests."""

    active_layout = pipeline_layout(spec) if layout is None else layout
    source_by_name = {source.name: source for source in spec.sources}
    if action.kind in {
        "create_queries_consensus",
        "run_consensus",
        "label_consensus",
    }:
        if action.source not in source_by_name:
            raise ValueError(f"stage {action.kind} requires a configured source")
        source = source_by_name[action.source]
        source_layout = active_layout.sources[source.name]
    if action.kind == "create_queries_consensus":
        return runners.queries(
            normalized=source.selected.path,
            output=source_layout.query_directory,
            katago=spec.katago.path,
            config=spec.analysis_config.path,
            original_model=spec.models["original"].path,
            champion_model=spec.models["champion"].path,
            policy=spec.policy.path,
        )
    if action.kind == "run_consensus":
        return runners.consensus(
            query_manifest_path=source_layout.query_manifest,
            work_dir=source_layout.consensus_work,
            shard_count=spec.topology.shards_per_role,
            gpus=spec.topology.gpus,
            per_gpu_parallelism=spec.topology.per_gpu_parallelism,
            poll_interval=poll_interval,
        )
    if action.kind == "label_consensus":
        query_manifest = _load_json_object(
            source_layout.query_manifest,
            f"{source.name} query manifest",
            canonical=True,
        )
        return runners.label(
            normalized_path=source.selected.path,
            query_manifest_path=source_layout.query_manifest,
            analysis_paths=analysis_outputs_for(query_manifest, source_layout),
            output_dir=source_layout.labeling_directory,
        )
    if action.kind == "merge_labeling_consensus":
        return runners.merge(
            bundle_dirs=[
                active_layout.sources[source.name].labeling_directory
                for source in spec.sources
            ],
            output_dir=active_layout.combined_labeling,
        )
    if action.kind == "finalize_consensus":
        labeling_root = _labeling_input(spec, active_layout)
        return runners.finalize(
            machine_labeled_path=labeling_root / "machine-labeled.jsonl",
            rejected_path=labeling_root / "rejected.jsonl",
            labeling_manifest_path=labeling_root / "manifest.json",
            policy_path=spec.policy.path,
            output_path=spec.outputs.reviewed_bank,
            manifest_path=spec.outputs.reviewed_manifest,
        )
    if action.kind == "build_evaluation_suites":
        return runners.suites(
            [spec.outputs.reviewed_bank],
            spec.outputs.suite_directory,
            seed=spec.suite_seed,
            policy_path=spec.policy.path,
            curation_manifest_paths=[spec.outputs.reviewed_manifest],
        )
    raise ValueError(f"stage {action.kind!r} is not executable")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise PipelineContradiction("status parent is not a regular directory")
    if _lexists(target) and (target.is_symlink() or not target.is_file()):
        raise PipelineContradiction("status path is not a regular file")
    data = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _status_state(action: StageAction, *, mode: str, running: bool = False) -> str:
    if running:
        return "running_" + action.kind
    if action.kind == "run_consensus" and mode == "once":
        return "awaiting_watch"
    return action.kind


def build_status(
    spec: PipelineSpec,
    snapshot: PipelineSnapshot,
    *,
    mode: str,
    action: StageAction | None = None,
    state: str | None = None,
    error: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    next_action = plan_next_stage(snapshot) if action is None else action
    layout = pipeline_layout(spec)
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "contract": STATUS_CONTRACT,
        "spec": {
            "path": str(spec.path),
            "sha256": spec.file_sha256,
            "identity": spec.identity,
        },
        "mode": mode,
        "state": state or _status_state(next_action, mode=mode),
        "next_stage": (
            None
            if next_action.kind in {"complete", "blocked_insufficient_sources"}
            else {"kind": next_action.kind, "source": next_action.source}
        ),
        "work_root": str(spec.work_root),
        "selected_counts": dict(snapshot.selected_counts),
        "required_counts": dict(spec.quotas),
        "deficits": dict(snapshot.deficits),
        "accepted_counts": (
            None if snapshot.accepted_counts is None else dict(snapshot.accepted_counts)
        ),
        "sources": {
            source.name: {
                "label": source.label,
                "selected_count": source.selected_count,
                "queries_consensus": source.queries_complete,
                "consensus_analysis": source.consensus_complete,
                "label_consensus": source.labeling_complete,
                "accepted_count": source.accepted_count,
                "rejected_count": source.rejected_count,
                "query_manifest": str(layout.sources[source.name].query_manifest),
                "consensus_work": str(layout.sources[source.name].consensus_work),
                "labeling_directory": str(
                    layout.sources[source.name].labeling_directory
                ),
            }
            for source in snapshot.sources
        },
        "artifacts": {
            "gpu_ownership": {
                "path": str(layout.gpu_ownership),
            },
            "combined_labeling": {
                "required": snapshot.combined_required,
                "complete": snapshot.combined_complete,
                "path": str(layout.combined_labeling),
            },
            "reviewed_bank": {
                "complete": snapshot.reviewed_complete,
                "path": str(spec.outputs.reviewed_bank),
                "manifest_path": str(spec.outputs.reviewed_manifest),
            },
            "suite": {
                "complete": snapshot.suite_complete,
                "path": str(spec.outputs.suite_directory),
            },
        },
        "error": None if error is None else dict(error),
    }
    status["status_sha256"] = canonical_sha256(status)
    return status


class GpuOwnershipManager:
    """Reusable, crash-reconciling exclusive ownership for configured GPUs.

    Callers provide the durable claim location and frozen spec identity, GPU
    indices, topology binding, accepted child command hashes, run root, and
    injectable host/process probes.  Hold :meth:`global_lock` across
    :meth:`acquire`, GPU work, and :meth:`release`.
    """

    def __init__(
        self,
        *,
        claim_path: Path,
        spec_path: Path,
        spec_sha256: str,
        spec_identity: str,
        configured_gpu_ids: Sequence[str],
        topology_binding: Mapping[str, Any],
        expected_command_sha256s: Sequence[str],
        run_root: Path,
        probes: OwnershipProbes = DEFAULT_OWNERSHIP_PROBES,
    ) -> None:
        self.claim_path = Path(claim_path)
        self.spec_binding = {
            "path": str(Path(spec_path)),
            "sha256": _require_sha256(spec_sha256, "ownership spec file hash"),
            "identity": _require_sha256(spec_identity, "ownership spec identity"),
        }
        self.configured_gpu_ids = tuple(configured_gpu_ids)
        if (
            not self.claim_path.is_absolute()
            or not self.configured_gpu_ids
            or len(set(self.configured_gpu_ids)) != len(self.configured_gpu_ids)
        ):
            raise PipelineSpecError("GPU ownership configuration is invalid")
        self.topology_binding = dict(topology_binding)
        if self.topology_binding.get("gpus") != list(self.configured_gpu_ids):
            raise PipelineSpecError("GPU ownership topology does not bind GPU IDs")
        self.expected_command_sha256s = tuple(
            sorted(
                _require_sha256(value, "owned process command hash")
                for value in expected_command_sha256s
            )
        )
        if not self.expected_command_sha256s:
            raise PipelineSpecError("GPU ownership requires accepted command hashes")
        self.run_root = Path(run_root)
        self.probes = probes

    def _configured_occupancy(
        self,
    ) -> tuple[Mapping[str, str], tuple[GpuComputeProcess, ...]]:
        occupancy = self.probes.gpu_occupancy()
        if not isinstance(occupancy, GpuOccupancy):
            raise PipelineContradiction("GPU occupancy probe returned an invalid value")
        try:
            gpus = {
                index: occupancy.index_to_uuid[index]
                for index in self.configured_gpu_ids
            }
        except (KeyError, TypeError) as exc:
            raise PipelineContradiction(
                "configured GPU index is absent from nvidia-smi inventory"
            ) from exc
        if any(not isinstance(uuid, str) or not uuid for uuid in gpus.values()) or len(
            set(gpus.values())
        ) != len(gpus):
            raise PipelineContradiction("configured GPU UUID inventory is invalid")
        configured_uuids = set(gpus.values())
        processes = tuple(
            process
            for process in occupancy.processes
            if process.gpu_uuid in configured_uuids
        )
        if any(
            type(process.pid) is not int
            or process.pid <= 0
            or not isinstance(process.process_name, str)
            for process in processes
        ):
            raise PipelineContradiction("GPU process inventory is malformed")
        return dict(sorted(gpus.items())), processes

    def _current_owner(self) -> ProcessIdentity:
        try:
            owner = self.probes.current_process()
        except (GpuLeaseError, OSError, RuntimeError, ValueError) as exc:
            raise PipelineContradiction(
                f"cannot capture coordinator process identity: {exc}"
            ) from exc
        if (
            not isinstance(owner, ProcessIdentity)
            or not owner.is_verifiable
            or owner.boot_id is None
            or owner.process_group_id is None
        ):
            raise PipelineContradiction(
                "coordinator process identity lacks boot/process-group binding"
            )
        return owner

    def _payload(
        self,
        *,
        state: str,
        gpus: Mapping[str, str],
        owner: ProcessIdentity,
        generation: int,
        recovered_from: str | None = None,
        observed_processes: Sequence[Mapping[str, Any]] = (),
        released_by: ProcessIdentity | None = None,
    ) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "schema_version": GPU_OWNERSHIP_SCHEMA_VERSION,
            "contract": GPU_OWNERSHIP_CONTRACT,
            "state": state,
            "spec": self.spec_binding,
            "topology": self.topology_binding,
            "gpus": dict(sorted(gpus.items())),
            "owner": owner.to_dict(),
            "expected_command_sha256s": list(self.expected_command_sha256s),
            "generation": generation,
            "recovered_from_claim_sha256": recovered_from,
            "observed_processes": list(observed_processes),
            "released_by": None if released_by is None else released_by.to_dict(),
        }
        value["claim_sha256"] = canonical_sha256(value)
        return value

    def _load(
        self, expected_gpus: Mapping[str, str]
    ) -> tuple[Mapping[str, Any], ProcessIdentity] | None:
        if not _lexists(self.claim_path):
            return None
        if self.claim_path.is_symlink() or not self.claim_path.is_file():
            raise PipelineContradiction("GPU ownership claim path is unsafe")
        try:
            claim = _load_json_object(
                self.claim_path,
                "GPU ownership claim",
                canonical=True,
            )
            owner_value = claim.get("owner")
            owner = (
                ProcessIdentity.from_dict(owner_value)
                if isinstance(owner_value, Mapping)
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineContradiction(
                f"GPU ownership claim is invalid: {exc}"
            ) from exc
        expected_keys = {
            "schema_version",
            "contract",
            "state",
            "spec",
            "topology",
            "gpus",
            "owner",
            "expected_command_sha256s",
            "generation",
            "recovered_from_claim_sha256",
            "observed_processes",
            "released_by",
            "claim_sha256",
        }
        payload = dict(claim)
        supplied_hash = payload.pop("claim_sha256", None)
        if (
            set(claim) != expected_keys
            or claim.get("schema_version") != GPU_OWNERSHIP_SCHEMA_VERSION
            or claim.get("contract") != GPU_OWNERSHIP_CONTRACT
            or claim.get("state") not in {"claimed", "recovering", "released"}
            or supplied_hash != canonical_sha256(payload)
            or claim.get("spec") != self.spec_binding
            or claim.get("topology") != self.topology_binding
            or claim.get("gpus") != dict(sorted(expected_gpus.items()))
            or claim.get("expected_command_sha256s")
            != list(self.expected_command_sha256s)
            or type(claim.get("generation")) is not int
            or claim["generation"] < 1
            or not isinstance(claim.get("observed_processes"), list)
            or owner is None
            or not owner.is_verifiable
            or owner.boot_id is None
            or owner.process_group_id is None
        ):
            raise PipelineContradiction(
                "stale GPU ownership claim contradicts this pipeline"
            )
        return claim, owner

    def _assert_host_safe(self) -> None:
        try:
            lease_safe = self.probes.gpu7_lease_safe(self.run_root)
            target_inactive = self.probes.production_target_inactive()
        except (OSError, RuntimeError, ValueError) as exc:
            raise PipelineContradiction(f"cannot prove GPU host safety: {exc}") from exc
        if lease_safe is not True:
            raise PipelineContradiction("GPU7 lease state is not trainer-safe")
        if target_inactive is not True:
            raise PipelineContradiction(
                "production systemd target is not provably inactive"
            )

    @contextlib.contextmanager
    def global_lock(self) -> Iterator[None]:
        """Exclude every coordinator using the configured host lock."""

        if fcntl is None:  # pragma: no cover
            raise CurationPipelineError("GPU ownership locking requires fcntl")
        path = Path(self.probes.global_lock_path)
        if (
            not path.is_absolute()
            or path.parent.is_symlink()
            or not path.parent.is_dir()
            or path.is_symlink()
        ):
            raise PipelineContradiction("global GPU ownership lock path is unsafe")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PipelineContradiction(
                    "global GPU ownership lock is not a regular file"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineBusy(
                    "another pipeline holds global GPU ownership"
                ) from exc
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def acquire(self, *, poll_interval: float) -> Mapping[str, Any]:
        """Acquire or safely reconcile a durable ownership claim."""

        current_owner = self._current_owner()
        while True:
            self._assert_host_safe()
            gpus, processes = self._configured_occupancy()
            loaded = self._load(gpus)
            if loaded is None:
                if processes:
                    raise PipelineContradiction(
                        "foreign compute process occupies a configured GPU"
                    )
                claim = self._payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=1,
                )
                _atomic_replace_json(self.claim_path, claim)
                return claim
            prior, prior_owner = loaded
            if prior_owner.boot_id != current_owner.boot_id:
                if processes:
                    raise PipelineContradiction(
                        "old-boot GPU claim has ambiguous occupied devices"
                    )
                claim = self._payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=prior["generation"] + 1,
                    recovered_from=prior["claim_sha256"],
                )
                _atomic_replace_json(self.claim_path, claim)
                return claim
            if prior["state"] == "released":
                if processes:
                    raise PipelineContradiction(
                        "foreign compute process appeared after GPU ownership release"
                    )
                claim = self._payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=prior["generation"] + 1,
                )
                _atomic_replace_json(self.claim_path, claim)
                return claim
            if not processes:
                claim = self._payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=prior["generation"] + 1,
                    recovered_from=(
                        prior.get("recovered_from_claim_sha256")
                        or prior["claim_sha256"]
                    ),
                )
                _atomic_replace_json(self.claim_path, claim)
                return claim
            observations = []
            for process in processes:
                try:
                    identity = self.probes.process_identity(process.pid)
                except (GpuLeaseError, OSError, RuntimeError, ValueError) as exc:
                    raise PipelineContradiction(
                        "cannot verify a configured GPU process identity"
                    ) from exc
                if (
                    identity.boot_id != prior_owner.boot_id
                    or identity.process_group_id != prior_owner.process_group_id
                    or identity.command_sha256 not in self.expected_command_sha256s
                ):
                    raise PipelineContradiction(
                        "foreign compute process occupies a configured GPU"
                    )
                observations.append(
                    {
                        "gpu_uuid": process.gpu_uuid,
                        "pid": process.pid,
                        "process_name": process.process_name,
                        "identity": identity.to_dict(),
                    }
                )
            recovering = self._payload(
                state="recovering",
                gpus=gpus,
                owner=prior_owner,
                generation=prior["generation"],
                recovered_from=(
                    prior.get("recovered_from_claim_sha256") or prior["claim_sha256"]
                ),
                observed_processes=observations,
            )
            _atomic_replace_json(self.claim_path, recovering)
            self.probes.sleep(poll_interval)

    def release(self) -> Mapping[str, Any]:
        """Complete ownership only after proving configured GPUs empty."""

        self._assert_host_safe()
        current_owner = self._current_owner()
        gpus, processes = self._configured_occupancy()
        loaded = self._load(gpus)
        if loaded is None:
            raise PipelineContradiction("GPU ownership claim disappeared")
        claim, owner = loaded
        if claim["state"] != "claimed" or not owner.same_process_as(current_owner):
            raise PipelineContradiction(
                "GPU ownership claim is not held by this coordinator"
            )
        if processes:
            raise PipelineContradiction(
                "configured GPU remains occupied after consensus returned"
            )
        released = self._payload(
            state="released",
            gpus=gpus,
            owner=owner,
            generation=claim["generation"],
            recovered_from=claim.get("recovered_from_claim_sha256"),
            released_by=current_owner,
        )
        _atomic_replace_json(self.claim_path, released)
        return released


class CurationPipeline:
    """Single-writer coordinator with artifact-derived restart semantics."""

    def __init__(
        self,
        spec: PipelineSpec | Path,
        *,
        runners: PipelineRunners = DEFAULT_RUNNERS,
        revision_reader: Callable[[Path], str] = git_revision,
        repository_status_reader: Callable[[Path], str] = git_status_porcelain,
        ownership_probes: OwnershipProbes = DEFAULT_OWNERSHIP_PROBES,
    ) -> None:
        self.revision_reader = revision_reader
        self.repository_status_reader = repository_status_reader
        self.spec = (
            spec
            if isinstance(spec, PipelineSpec)
            else load_pipeline_spec(
                Path(spec),
                revision_reader=revision_reader,
                repository_status_reader=repository_status_reader,
            )
        )
        self.runners = runners
        self.ownership_probes = ownership_probes
        self.layout = pipeline_layout(self.spec)
        self.gpu_ownership = GpuOwnershipManager(
            claim_path=self.layout.gpu_ownership,
            spec_path=self.spec.path,
            spec_sha256=self.spec.file_sha256,
            spec_identity=self.spec.identity,
            configured_gpu_ids=self.spec.topology.gpus,
            topology_binding=self._ownership_topology(),
            expected_command_sha256s=self._expected_analysis_command_hashes(),
            run_root=self.spec.run_root,
            probes=ownership_probes,
        )

    def _expected_analysis_command_hashes(self) -> tuple[str, ...]:
        hashes = []
        for role in ("original", "champion"):
            argv = (
                str(self.spec.katago.path),
                "analysis",
                "-config",
                str(self.spec.analysis_config.path),
                "-model",
                str(self.spec.models[role].path),
            )
            command = b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
            hashes.append(hashlib.sha256(command).hexdigest())
        return tuple(sorted(hashes))

    def _ownership_topology(self) -> Mapping[str, Any]:
        return {
            "gpus": list(self.spec.topology.gpus),
            "per_gpu_parallelism": self.spec.topology.per_gpu_parallelism,
            "shards_per_role": self.spec.topology.shards_per_role,
        }

    def _configured_occupancy(
        self, occupancy: GpuOccupancy
    ) -> tuple[Mapping[str, str], tuple[GpuComputeProcess, ...]]:
        if not isinstance(occupancy, GpuOccupancy):
            raise PipelineContradiction("GPU occupancy probe returned an invalid value")
        try:
            gpus = {
                index: occupancy.index_to_uuid[index]
                for index in self.spec.topology.gpus
            }
        except (KeyError, TypeError) as exc:
            raise PipelineContradiction(
                "configured GPU index is absent from nvidia-smi inventory"
            ) from exc
        if any(not isinstance(uuid, str) or not uuid for uuid in gpus.values()) or len(
            set(gpus.values())
        ) != len(gpus):
            raise PipelineContradiction("configured GPU UUID inventory is invalid")
        configured_uuids = set(gpus.values())
        processes = tuple(
            process
            for process in occupancy.processes
            if process.gpu_uuid in configured_uuids
        )
        if any(
            type(process.pid) is not int
            or process.pid <= 0
            or not isinstance(process.process_name, str)
            for process in processes
        ):
            raise PipelineContradiction("GPU process inventory is malformed")
        return dict(sorted(gpus.items())), processes

    def _current_owner(self) -> ProcessIdentity:
        try:
            owner = self.ownership_probes.current_process()
        except (GpuLeaseError, OSError, RuntimeError, ValueError) as exc:
            raise PipelineContradiction(
                f"cannot capture coordinator process identity: {exc}"
            ) from exc
        if (
            not isinstance(owner, ProcessIdentity)
            or not owner.is_verifiable
            or owner.boot_id is None
            or owner.process_group_id is None
        ):
            raise PipelineContradiction(
                "coordinator process identity lacks boot/process-group binding"
            )
        return owner

    def _ownership_payload(
        self,
        *,
        state: str,
        gpus: Mapping[str, str],
        owner: ProcessIdentity,
        generation: int,
        recovered_from: str | None = None,
        observed_processes: Sequence[Mapping[str, Any]] = (),
        released_by: ProcessIdentity | None = None,
    ) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "schema_version": GPU_OWNERSHIP_SCHEMA_VERSION,
            "contract": GPU_OWNERSHIP_CONTRACT,
            "state": state,
            "spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "topology": self._ownership_topology(),
            "gpus": dict(sorted(gpus.items())),
            "owner": owner.to_dict(),
            "expected_command_sha256s": list(self._expected_analysis_command_hashes()),
            "generation": generation,
            "recovered_from_claim_sha256": recovered_from,
            "observed_processes": list(observed_processes),
            "released_by": None if released_by is None else released_by.to_dict(),
        }
        value["claim_sha256"] = canonical_sha256(value)
        return value

    def _load_ownership_claim(
        self, expected_gpus: Mapping[str, str]
    ) -> tuple[Mapping[str, Any], ProcessIdentity] | None:
        if not _lexists(self.layout.gpu_ownership):
            return None
        if (
            self.layout.gpu_ownership.is_symlink()
            or not self.layout.gpu_ownership.is_file()
        ):
            raise PipelineContradiction("GPU ownership claim path is unsafe")
        try:
            claim = _load_json_object(
                self.layout.gpu_ownership, "GPU ownership claim", canonical=True
            )
        except ValueError as exc:
            raise PipelineContradiction(
                f"GPU ownership claim is invalid: {exc}"
            ) from exc
        expected_keys = {
            "schema_version",
            "contract",
            "state",
            "spec",
            "topology",
            "gpus",
            "owner",
            "expected_command_sha256s",
            "generation",
            "recovered_from_claim_sha256",
            "observed_processes",
            "released_by",
            "claim_sha256",
        }
        payload = dict(claim)
        supplied_hash = payload.pop("claim_sha256", None)
        owner_value = claim.get("owner")
        try:
            owner = (
                ProcessIdentity.from_dict(owner_value)
                if isinstance(owner_value, Mapping)
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineContradiction(
                "GPU ownership owner identity is malformed"
            ) from exc
        if (
            set(claim) != expected_keys
            or claim.get("schema_version") != GPU_OWNERSHIP_SCHEMA_VERSION
            or claim.get("contract") != GPU_OWNERSHIP_CONTRACT
            or claim.get("state") not in {"claimed", "recovering", "released"}
            or supplied_hash != canonical_sha256(payload)
            or claim.get("spec")
            != {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            }
            or claim.get("topology") != self._ownership_topology()
            or claim.get("gpus") != dict(sorted(expected_gpus.items()))
            or claim.get("expected_command_sha256s")
            != list(self._expected_analysis_command_hashes())
            or type(claim.get("generation")) is not int
            or claim["generation"] < 1
            or not isinstance(claim.get("observed_processes"), list)
            or owner is None
            or not owner.is_verifiable
            or owner.boot_id is None
            or owner.process_group_id is None
        ):
            raise PipelineContradiction(
                "stale GPU ownership claim contradicts this pipeline"
            )
        return claim, owner

    def _assert_gpu_safety_prerequisites(self) -> None:
        try:
            lease_safe = self.ownership_probes.gpu7_lease_safe(self.spec.run_root)
            target_inactive = self.ownership_probes.production_target_inactive()
        except (OSError, RuntimeError, ValueError) as exc:
            raise PipelineContradiction(f"cannot prove GPU host safety: {exc}") from exc
        if lease_safe is not True:
            raise PipelineContradiction("GPU7 lease state is not trainer-safe")
        if target_inactive is not True:
            raise PipelineContradiction(
                "production systemd target is not provably inactive"
            )

    @contextlib.contextmanager
    def _global_gpu_lock(self) -> Iterator[None]:
        """Compatibility wrapper around the public ownership helper."""

        with self.gpu_ownership.global_lock():
            yield
        return
        if fcntl is None:  # pragma: no cover
            raise CurationPipelineError("GPU ownership locking requires fcntl")
        path = Path(self.ownership_probes.global_lock_path)
        if (
            not path.is_absolute()
            or path.parent.is_symlink()
            or not path.parent.is_dir()
            or path.is_symlink()
        ):
            raise PipelineContradiction("global GPU ownership lock path is unsafe")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PipelineContradiction(
                    "global GPU ownership lock is not a regular file"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineBusy(
                    "another pipeline holds global GPU ownership"
                ) from exc
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _acquire_gpu_ownership(self, *, poll_interval: float) -> Mapping[str, Any]:
        """Compatibility wrapper around the public ownership helper."""

        return self.gpu_ownership.acquire(poll_interval=poll_interval)
        current_owner = self._current_owner()
        while True:
            self._assert_gpu_safety_prerequisites()
            gpus, processes = self._configured_occupancy(
                self.ownership_probes.gpu_occupancy()
            )
            loaded = self._load_ownership_claim(gpus)
            if loaded is None:
                if processes:
                    raise PipelineContradiction(
                        "foreign compute process occupies a configured GPU"
                    )
                claim = self._ownership_payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=1,
                )
                _atomic_replace_json(self.layout.gpu_ownership, claim)
                return claim
            prior, prior_owner = loaded
            if prior_owner.boot_id != current_owner.boot_id:
                raise PipelineContradiction(
                    "stale GPU ownership claim is from a different boot"
                )
            if prior["state"] == "released":
                if processes:
                    raise PipelineContradiction(
                        "foreign compute process appeared after GPU ownership release"
                    )
                claim = self._ownership_payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=prior["generation"] + 1,
                )
                _atomic_replace_json(self.layout.gpu_ownership, claim)
                return claim
            if not processes:
                claim = self._ownership_payload(
                    state="claimed",
                    gpus=gpus,
                    owner=current_owner,
                    generation=prior["generation"] + 1,
                    recovered_from=(
                        prior.get("recovered_from_claim_sha256")
                        or prior["claim_sha256"]
                    ),
                )
                _atomic_replace_json(self.layout.gpu_ownership, claim)
                return claim
            observations = []
            for process in processes:
                try:
                    identity = self.ownership_probes.process_identity(process.pid)
                except (GpuLeaseError, OSError, RuntimeError, ValueError) as exc:
                    raise PipelineContradiction(
                        "cannot verify a configured GPU process identity"
                    ) from exc
                if (
                    identity.boot_id != prior_owner.boot_id
                    or identity.process_group_id != prior_owner.process_group_id
                    or identity.command_sha256
                    not in self._expected_analysis_command_hashes()
                ):
                    raise PipelineContradiction(
                        "foreign compute process occupies a configured GPU"
                    )
                observations.append(
                    {
                        "gpu_uuid": process.gpu_uuid,
                        "pid": process.pid,
                        "process_name": process.process_name,
                        "identity": identity.to_dict(),
                    }
                )
            recovering = self._ownership_payload(
                state="recovering",
                gpus=gpus,
                owner=prior_owner,
                generation=prior["generation"],
                recovered_from=(
                    prior.get("recovered_from_claim_sha256") or prior["claim_sha256"]
                ),
                observed_processes=observations,
            )
            _atomic_replace_json(self.layout.gpu_ownership, recovering)
            self.ownership_probes.sleep(poll_interval)

    def _release_gpu_ownership(self) -> Mapping[str, Any]:
        """Compatibility wrapper around the public ownership helper."""

        return self.gpu_ownership.release()
        self._assert_gpu_safety_prerequisites()
        current_owner = self._current_owner()
        gpus, processes = self._configured_occupancy(
            self.ownership_probes.gpu_occupancy()
        )
        loaded = self._load_ownership_claim(gpus)
        if loaded is None:
            raise PipelineContradiction("GPU ownership claim disappeared")
        claim, owner = loaded
        if claim["state"] != "claimed" or not owner.same_process_as(current_owner):
            raise PipelineContradiction(
                "GPU ownership claim is not held by this coordinator"
            )
        if processes:
            raise PipelineContradiction(
                "configured GPU remains occupied after consensus returned"
            )
        released = self._ownership_payload(
            state="released",
            gpus=gpus,
            owner=owner,
            generation=claim["generation"],
            recovered_from=claim.get("recovered_from_claim_sha256"),
            released_by=current_owner,
        )
        _atomic_replace_json(self.layout.gpu_ownership, released)
        return released

    def _assert_frozen_inputs(self) -> None:
        if (
            self.spec.path.is_symlink()
            or not self.spec.path.is_file()
            or str(self.spec.path.resolve()) != str(self.spec.path)
            or file_sha256(self.spec.path) != self.spec.file_sha256
        ):
            raise PipelineContradiction(
                "pipeline specification changed during execution"
            )
        _require_regular_directory(
            self.spec.deployment.repository_path, "deployment repository"
        )
        _require_regular_directory(self.spec.run_root, "run root")
        if _lexists(self.spec.work_root):
            _require_regular_directory(self.spec.work_root, "pipeline work root")
        if self.revision_reader(self.spec.deployment.repository_path) != (
            self.spec.deployment.source_revision
        ):
            raise PipelineContradiction("deployment repository revision changed")
        if self.repository_status_reader(self.spec.deployment.repository_path):
            raise PipelineContradiction("deployment repository became dirty")
        _verify_bound_deployment_manifest(
            self.spec.deployment_manifest,
            self.spec.deployment,
            error_type=PipelineContradiction,
        )
        for binding, role in (
            (self.spec.policy, "promotion policy"),
            (self.spec.katago, "KataGo binary"),
            (self.spec.analysis_config, "analysis config"),
            (self.spec.models["original"], "original model"),
            (self.spec.models["champion"], "champion model"),
            *(
                (source.selected, f"{source.name} selected source")
                for source in self.spec.sources
            ),
            *(
                (
                    source.prefilter_manifest,
                    f"{source.name} prefilter manifest",
                )
                for source in self.spec.sources
            ),
            *(
                (
                    source.supplement_summary,
                    f"{source.name} supplement summary",
                )
                for source in self.spec.sources
                if source.supplement_summary is not None
            ),
        ):
            _bound_regular_file(binding.path, binding.sha256, role)

    def _snapshot(self) -> PipelineSnapshot:
        self._assert_frozen_inputs()
        inventories = verify_prefilter_sources(self.spec)
        snapshot = infer_pipeline_snapshot(self.spec, inventories=inventories)
        self._assert_frozen_inputs()
        return snapshot

    def _validate_existing_status(self) -> None:
        if not _lexists(self.layout.status):
            return
        _require_regular_directory(self.spec.work_root, "pipeline work root")
        try:
            status = _load_json_object(
                self.layout.status, "curation pipeline status", canonical=True
            )
        except ValueError as exc:
            raise PipelineContradiction(f"pipeline status is invalid: {exc}") from exc
        payload = dict(status)
        supplied_hash = payload.pop("status_sha256", None)
        binding = status.get("spec")
        if (
            status.get("schema_version") != STATUS_SCHEMA_VERSION
            or status.get("contract") != STATUS_CONTRACT
            or supplied_hash != canonical_sha256(payload)
            or not isinstance(binding, Mapping)
            or binding.get("path") != str(self.spec.path)
            or binding.get("sha256") != self.spec.file_sha256
            or binding.get("identity") != self.spec.identity
            or status.get("work_root") != str(self.spec.work_root)
        ):
            raise PipelineContradiction(
                "existing pipeline status contradicts the frozen specification"
            )

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.spec.work_root.mkdir(parents=True, exist_ok=True)
        _require_regular_directory(self.spec.work_root, "pipeline work root")
        if self.layout.lock.is_symlink():
            raise PipelineContradiction("pipeline lock may not be a symlink")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(self.layout.lock), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PipelineContradiction("pipeline lock is not a regular file")
            if fcntl is None:
                raise CurationPipelineError(
                    "pipeline coordination requires Unix file locking"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineBusy(
                    f"another curation pipeline holds {self.layout.lock}"
                ) from exc
            yield
        finally:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _persist(
        self,
        snapshot: PipelineSnapshot,
        *,
        mode: str,
        action: StageAction | None = None,
        state: str | None = None,
        error: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        status = build_status(
            self.spec,
            snapshot,
            mode=mode,
            action=action,
            state=state,
            error=error,
        )
        _atomic_replace_json(self.layout.status, status)
        return status

    def status(self) -> Mapping[str, Any]:
        """Inspect without creating directories, locks, or status files."""

        self._validate_existing_status()
        snapshot = self._snapshot()
        return build_status(self.spec, snapshot, mode="status")

    def _run_locked(self, *, mode: str, poll_interval: float) -> Mapping[str, Any]:
        self._validate_existing_status()
        while True:
            snapshot = self._snapshot()
            action = plan_next_stage(snapshot)
            if action.kind in {"blocked_insufficient_sources", "complete"}:
                return self._persist(snapshot, mode=mode, action=action)
            if mode == "once" and action.kind == "run_consensus":
                return self._persist(
                    snapshot,
                    mode=mode,
                    action=action,
                    state="awaiting_watch",
                )
            try:
                ownership_guard = (
                    self.gpu_ownership.global_lock()
                    if action.kind == "run_consensus"
                    else contextlib.nullcontext()
                )
                with ownership_guard:
                    if action.kind == "run_consensus":
                        self.gpu_ownership.acquire(poll_interval=poll_interval)
                        refreshed_snapshot = self._snapshot()
                        refreshed_action = plan_next_stage(refreshed_snapshot)
                        if refreshed_action != action:
                            self.gpu_ownership.release()
                            snapshot = refreshed_snapshot
                            action = refreshed_action
                            continue
                    self._persist(
                        snapshot,
                        mode=mode,
                        action=action,
                        state=_status_state(action, mode=mode, running=True),
                    )
                    dispatch_stage(
                        action,
                        self.spec,
                        layout=self.layout,
                        runners=self.runners,
                        poll_interval=poll_interval,
                    )
                    next_snapshot = self._snapshot()
                    if action.kind == "run_consensus":
                        self.gpu_ownership.release()
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    self._persist(
                        snapshot,
                        mode=mode,
                        action=action,
                        state=(
                            "interrupted"
                            if isinstance(exc, KeyboardInterrupt)
                            else "failed"
                        ),
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                raise
            next_action = plan_next_stage(next_snapshot)
            if next_action == action:
                error = CurationPipelineError(
                    f"stage runner returned without publishing {action.kind}"
                )
                self._persist(
                    next_snapshot,
                    mode=mode,
                    action=action,
                    state="failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
                raise error
            status = self._persist(next_snapshot, mode=mode, action=next_action)
            if mode == "once":
                return status

    def once(self) -> Mapping[str, Any]:
        """Advance exactly one non-GPU stage, never the long consensus stage."""

        with self._exclusive_lock():
            return self._run_locked(mode="once", poll_interval=30.0)

    def watch(self, *, poll_interval: float = 30.0) -> Mapping[str, Any]:
        """Resume all stages, including restartable GPU consensus, to completion."""

        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(float(poll_interval))
            or poll_interval < 0
        ):
            raise ValueError("poll_interval must be finite and nonnegative")
        with self._exclusive_lock():
            return self._run_locked(mode="watch", poll_interval=float(poll_interval))


CurationPipelineCoordinator = CurationPipeline
CurationToSuiteCoordinator = CurationPipeline


def coordinate(
    *,
    mode: str,
    spec_path: Path,
    runners: PipelineRunners = DEFAULT_RUNNERS,
    revision_reader: Callable[[Path], str] = git_revision,
    repository_status_reader: Callable[[Path], str] = git_status_porcelain,
    ownership_probes: OwnershipProbes = DEFAULT_OWNERSHIP_PROBES,
    poll_interval: float = 30.0,
) -> Mapping[str, Any]:
    coordinator = CurationPipeline(
        spec_path,
        runners=runners,
        revision_reader=revision_reader,
        repository_status_reader=repository_status_reader,
        ownership_probes=ownership_probes,
    )
    if mode == "status":
        return coordinator.status()
    if mode == "once":
        return coordinator.once()
    if mode == "watch":
        return coordinator.watch(poll_interval=poll_interval)
    raise ValueError(f"unsupported curation pipeline mode: {mode}")


run_pipeline = coordinate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "once", "watch"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runners: PipelineRunners = DEFAULT_RUNNERS,
    revision_reader: Callable[[Path], str] = git_revision,
) -> int:
    args = parse_args(argv)
    try:
        status = coordinate(
            mode=args.mode,
            spec_path=args.spec,
            runners=runners,
            revision_reader=revision_reader,
            poll_interval=args.poll_interval,
        )
    except KeyboardInterrupt:
        print(
            canonical_json(
                {"error": {"type": "KeyboardInterrupt", "message": "interrupted"}}
            ),
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(status))
    if args.mode == "status":
        return 0
    return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
