#!/usr/bin/env python3
"""Restartable production of quarantined supplemental Lead sources.

The strict canonical specification binds a clean deployment, a verified
deployment manifest, the exact finite KataGo self-play command, a one-model
``models`` directory, GPU topology, policy-derived reserve targets, and a
complete primary-prefilter inventory.  All accepted stage products are
immutable. ``status.json`` and attempt journals are the only mutable artifacts;
both are reconstructed or reconciled against immutable evidence on restart.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production curation runs on Unix.
    fcntl = None  # type: ignore[assignment]

from risk_score.board_symmetry import symmetry_orbit_sha256
from risk_score.consensus_prefilter import (
    PREFILTER_CONTRACT,
    PREFILTER_QUERY_BUNDLE_CONTRACT,
    PREFILTER_ROLES,
    generate_prefilter_query_bundle,
    prefilter_consensus_sources,
    validate_prefilter_artifact,
)
from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    CURATION_CONTRACT,
    HARVEST_PLAN_CONTRACT,
    HARVEST_RECEIPT_CONTRACT,
    QUERY_SHARDS_CONTRACT,
    _canonical_jsonl,
    _harvest_output_inventory,
    _load_query_shard_manifest,
    _normalized_positions,
    _source_inventory,
    _validate_harvest_plan,
    build_harvest_argv,
    execute_harvest_plan,
    merge_analysis,
    normalize_sources,
    policy_pool_minima,
    publish_harvest_plan,
    run_analysis,
    split_queries,
    validate_deterministic_analysis_config,
)
from risk_score.build_live_runtime import verify_deployment_manifest
from risk_score.gpu_lease import ProcessIdentity
from risk_score.paired_stats import load_policy
from risk_score.promotion_auditor import PromotionAuditorError, _sgf_properties
from risk_score.position_samples import (
    build_analysis_query,
    canonical_json,
    canonical_sha256,
    file_sha256,
)

SPEC_SCHEMA_VERSION = 1
SPEC_CONTRACT = "risk-score-curation-supplement-spec-v1"
STATUS_SCHEMA_VERSION = 1
STATUS_CONTRACT = "risk-score-curation-supplement-status-v1"
SUMMARY_SCHEMA_VERSION = 1
SUMMARY_CONTRACT = "risk-score-curation-supplement-summary-v1"
SELFPLAY_RECEIPT_CONTRACT = "risk-score-curation-supplement-selfplay-v1"
SELFPLAY_ATTEMPT_CONTRACT = "risk-score-curation-supplement-selfplay-attempt-v1"
STAGE_ATTEMPT_CONTRACT = "risk-score-curation-supplement-stage-attempt-v1"
PRIMARY_INVENTORY_CONTRACT = (
    "risk-score-curation-supplement-primary-prefilter-inventory-v1"
)
REJECTED_DUPLICATES_CONTRACT = "risk-score-curation-supplement-rejected-duplicates-v1"
GPU_OWNERSHIP_CONTRACT = "risk-score-curation-gpu-ownership-v1"
SUPPLEMENT_SPEC_CONTRACT = SPEC_CONTRACT
SUPPLEMENT_STATUS_CONTRACT = STATUS_CONTRACT
SUPPLEMENT_SUMMARY_CONTRACT = SUMMARY_CONTRACT

LEAD_LABELS = ("lead-40", "lead-80")
PRIMARY_LABELS = ("ordinary", *LEAD_LABELS)
PREFILTER_MAXIMUM_SCORE_SPREAD = 3.0
PREFILTER_THRESHOLD_BUFFER = 5.0

_SHA256_LENGTH = 64
_REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_TERMINAL_RESULT_RE = re.compile(
    r"(?:[BW]\+(?:R|Resign|T|Time|F|Forfeit|[0-9]+(?:\.[0-9]+)?)|0|Draw|Void)"
)
_CRITICAL_SELFPLAY_FLAGS = {
    "-max-games-total",
    "-output-dir",
    "-models-dir",
    "-config",
}
_EXECUTED_MODULES = (
    "board_symmetry.py",
    "build_live_runtime.py",
    "consensus_prefilter.py",
    "curate_position_bank.py",
    "curation_orchestrator.py",
    "curation_pipeline.py",
    "curation_supplement.py",
    "gpu_lease.py",
    "position_samples.py",
)
_SPEC_KEYS = {
    "schema_version",
    "contract",
    "deployment",
    "deployment_manifest",
    "run_root",
    "training_input_root",
    "work_root",
    "katago",
    "analysis_config",
    "selfplay_config",
    "selfplay_models_directory",
    "selfplay_override_args",
    "policy",
    "models",
    "game_count",
    "topology",
    "consensus_reserve_fraction",
    "target_counts",
    "primary_prefilter_inventory",
    "primary_prefilter_manifests",
    "round",
    "prior_round_summaries",
    "downstream_accepted_counts",
    "spec_sha256",
}
_LEGACY_READ_ONLY_SPEC_KEYS = {
    "schema_version",
    "contract",
    "run_root",
    "training_input_root",
    "work_root",
    "katago",
    "analysis_config",
    "selfplay_config",
    "policy",
    "models",
    "game_count",
    "topology",
    "target_counts",
    "primary_prefilter_manifests",
    "selfplay_argv_template",
    "spec_sha256",
}


class SupplementError(RuntimeError):
    """Base class for supplemental-source coordination failures."""


class SupplementSpecError(SupplementError, ValueError):
    """The canonical supplement specification is invalid or stale."""


class SupplementContradiction(SupplementError, ValueError):
    """An existing artifact contradicts the frozen specification."""


class SupplementBusy(SupplementError):
    """Another process owns the supplemental coordinator lock."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class DirectoryBinding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class DeploymentBinding:
    repository_path: Path
    source_revision: str
    source_sha256: str


@dataclass(frozen=True)
class PrimaryPrefilter:
    manifest: FileBinding
    label: str
    selected: FileBinding
    row_count: int
    semantic_ids: tuple[str, ...]
    symmetry_orbits: tuple[str, ...]


@dataclass(frozen=True)
class Topology:
    shards_per_role: int
    gpus: tuple[str, ...]
    selfplay_gpus: tuple[str, ...]
    per_gpu_parallelism: int


@dataclass(frozen=True)
class SupplementSpec:
    path: Path
    file_sha256: str
    identity: str
    raw: Mapping[str, Any]
    deployment: DeploymentBinding
    deployment_manifest: FileBinding
    run_root: Path
    training_input_root: Path
    work_root: Path
    katago: FileBinding
    analysis_config: FileBinding
    selfplay_config: FileBinding
    selfplay_models_directory: DirectoryBinding
    selfplay_override_args: tuple[tuple[str, str], ...]
    policy: FileBinding
    models: Mapping[str, FileBinding]
    game_count: int
    topology: Topology
    policy_minima: Mapping[str, int]
    consensus_reserve_fraction: float
    target_counts: Mapping[str, int]
    primary_prefilter_inventory: FileBinding
    primary_prefilters: tuple[PrimaryPrefilter, ...]
    round: int
    prior_round_summaries: tuple[FileBinding, ...]
    downstream_accepted_counts: Mapping[str, int] | None
    frozen_files: tuple[FileBinding, ...]
    legacy_read_only: bool = False


@dataclass(frozen=True)
class SupplementLayout:
    selfplay_attempt_root: Path
    selfplay_attempt_output: Path
    selfplay_attempt_journal: Path
    selfplay_orphans: Path
    selfplay_directory: Path
    selfplay_receipt: Path
    harvest_plan: Path
    harvest_directory: Path
    normalized: Path
    normalized_manifest: Path
    rejected_duplicates: Path
    rejected_duplicates_manifest: Path
    query_directory: Path
    query_manifest: Path
    query_shards: Path
    analysis_shards: Path
    analyses: Path
    selected: Path
    summary: Path
    gpu_ownership_archive: Path
    stage_attempts: Path
    status: Path
    lock: Path


@dataclass(frozen=True)
class AnalysisJob:
    role: str
    model: str
    mode: str
    shard_index: int
    gpu: str
    query_path: Path
    output_path: Path

    @property
    def manifest_path(self) -> Path:
        return Path(str(self.output_path) + ".manifest.json")

    @property
    def key(self) -> tuple[str, int]:
        return self.role, self.shard_index


@dataclass(frozen=True)
class SupplementSnapshot:
    primary_counts: Mapping[str, int]
    deficits: Mapping[str, int]
    selfplay_complete: bool
    harvest_plan_complete: bool
    harvest_complete: bool
    normalized_complete: bool
    queries_complete: bool
    split_roles_complete: int
    analysis_shards_complete: int
    analysis_shards_total: int
    merged_roles_complete: int
    prefilters_complete: Mapping[str, bool]
    supplemental_counts: Mapping[str, int]
    summary_complete: bool
    summary_state: str | None


@dataclass(frozen=True)
class SupplementRunners:
    """Injectable process and trusted stage boundaries."""

    process: Callable[..., Any] = subprocess.run
    launcher: Callable[..., Any] = subprocess.Popen
    harvest_plan: Callable[..., Mapping[str, Any]] = publish_harvest_plan
    harvest: Callable[..., Mapping[str, Any]] = execute_harvest_plan
    queries: Callable[..., Mapping[str, Any]] = generate_prefilter_query_bundle
    split: Callable[..., Mapping[str, Any]] = split_queries
    analysis: Callable[..., Mapping[str, Any]] = run_analysis
    merge: Callable[..., Mapping[str, Any]] = merge_analysis
    prefilter: Callable[..., Mapping[str, Any]] = prefilter_consensus_sources


DEFAULT_RUNNERS = SupplementRunners()


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


def _load_json_object(
    path: Path, role: str, *, canonical: bool = True
) -> dict[str, Any]:
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
        raise SupplementSpecError(f"{role} must be an object")
    missing = sorted(keys.difference(value))
    extra = sorted(set(value).difference(keys))
    if missing or extra:
        raise SupplementSpecError(
            f"{role} keys differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SupplementSpecError(f"{role} must be a lowercase 64-character SHA-256")
    return value


def _canonical_path(raw: Any, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SupplementSpecError(f"{role} must be a nonempty absolute path")
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve()) != raw:
        raise SupplementSpecError(
            f"{role} must be an absolute canonical path with no symlink components"
        )
    return path


def _required_directory(raw: Any, role: str) -> Path:
    path = _canonical_path(raw, role)
    if path.is_symlink() or not path.is_dir():
        raise SupplementSpecError(f"{role} must be an existing non-symlink directory")
    return path


def _future_directory(raw: Any, role: str) -> Path:
    path = _canonical_path(raw, role)
    if _lexists(path) and (path.is_symlink() or not path.is_dir()):
        raise SupplementSpecError(
            f"{role} must be a non-symlink directory when present"
        )
    return path


def _file_binding(value: Any, role: str) -> FileBinding:
    binding = _require_exact_keys(value, {"path", "sha256"}, role)
    path = _canonical_path(binding["path"], f"{role} path")
    digest = _require_sha256(binding["sha256"], f"{role} hash")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise SupplementSpecError(f"{role} file is missing or does not match its hash")
    return FileBinding(path=path, sha256=digest)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _directory_inventory(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_dir():
        raise SupplementSpecError("self-play models directory is missing or unsafe")
    rows: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise SupplementSpecError(
                "self-play models directory may contain only regular files"
            )
        rows.append(
            {
                "path": child.name,
                "size": metadata.st_size,
                "sha256": file_sha256(child),
            }
        )
    return tuple(rows)


def _directory_binding(value: Any, role: str) -> DirectoryBinding:
    binding = _require_exact_keys(value, {"path", "sha256"}, role)
    path = _required_directory(binding["path"], f"{role} path")
    digest = _require_sha256(binding["sha256"], f"{role} hash")
    if canonical_sha256(list(_directory_inventory(path))) != digest:
        raise SupplementSpecError(f"{role} inventory does not match its hash")
    return DirectoryBinding(path, digest)


def _validate_override_args(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise SupplementSpecError("selfplay_override_args must be an argv-pair array")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, pair in enumerate(value):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(
                isinstance(token, str)
                and token
                and "\x00" not in token
                and "\n" not in token
                and "\r" not in token
                for token in pair
            )
            or not pair[0].startswith("-")
        ):
            raise SupplementSpecError(
                f"selfplay override {index} must be an explicit [FLAG, VALUE] pair"
            )
        flag = pair[0]
        if flag in _CRITICAL_SELFPLAY_FLAGS:
            raise SupplementSpecError(
                f"selfplay override duplicates critical flag {flag}"
            )
        if flag in seen:
            raise SupplementSpecError(f"selfplay override flag is repeated: {flag}")
        if flag == "-override-config" and any(
            key in pair[1]
            for key in (
                "cudaDeviceToUseModel",
                "numNNServerThreadsPerModel",
            )
        ):
            raise SupplementSpecError(
                "selfplay overrides may not alter the bound GPU topology"
            )
        seen.add(flag)
        result.append((flag, pair[1]))
    return tuple(result)


def _parse_config_assignments(path: Path) -> Mapping[str, str]:
    assignments: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise SupplementSpecError(
                f"self-play config line {line_number} is not KEY = VALUE"
            )
        if key in assignments:
            raise SupplementSpecError(f"self-play config repeats {key}")
        assignments[key] = value
    return assignments


def _validate_selfplay_gpu_mapping(
    config: FileBinding, selfplay_gpus: Sequence[str]
) -> None:
    assignments = _parse_config_assignments(config.path)
    try:
        thread_count = int(assignments["numNNServerThreadsPerModel"])
    except (KeyError, ValueError) as exc:
        raise SupplementSpecError(
            "self-play config must bind numNNServerThreadsPerModel"
        ) from exc
    expected = {
        f"cudaDeviceToUseModel0Thread{index}": gpu
        for index, gpu in enumerate(selfplay_gpus)
    }
    actual = {
        key: value
        for key, value in assignments.items()
        if key.startswith("cudaDeviceToUseModel0Thread")
    }
    if thread_count != len(selfplay_gpus) or actual != expected:
        raise SupplementSpecError(
            "self-play config GPU mapping contradicts configured topology"
        )


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        raise SupplementSpecError("cannot resolve deployment revision")
    return result.stdout.strip()


def _git_status(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        raise SupplementSpecError("cannot inspect deployment checkout")
    return result.stdout


def _verify_deployment(
    manifest_binding: FileBinding,
    deployment: DeploymentBinding,
    *,
    error_type: type[SupplementError],
) -> Mapping[str, Any]:
    if (
        manifest_binding.path.is_symlink()
        or not manifest_binding.path.is_file()
        or file_sha256(manifest_binding.path) != manifest_binding.sha256
    ):
        raise error_type("deployment manifest is missing or changed")
    try:
        manifest = verify_deployment_manifest(manifest_binding.path)
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
        artifact = files.get(f"module:{module_name}")
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


def _absolute_manifest_path(raw: Any, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SupplementSpecError(f"{role} path is missing")
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve()) != raw:
        raise SupplementSpecError(f"{role} path is not absolute and canonical")
    return path


def _bound_transitive_file(
    raw_path: Any,
    raw_hash: Any,
    role: str,
    frozen: list[FileBinding],
) -> FileBinding:
    path = _absolute_manifest_path(raw_path, role)
    digest = _require_sha256(raw_hash, f"{role} hash")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise SupplementSpecError(f"{role} is missing or changed")
    binding = FileBinding(path, digest)
    frozen.append(binding)
    return binding


def _verify_primary_prefilter(
    binding: FileBinding,
    *,
    models: Mapping[str, FileBinding],
    analysis_config: FileBinding,
) -> tuple[PrimaryPrefilter, tuple[FileBinding, ...]]:
    frozen: list[FileBinding] = [binding]
    try:
        manifest = _load_json_object(binding.path, "primary prefilter manifest")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("contract") != PREFILTER_CONTRACT
            or manifest.get("advisory_only") is not True
            or manifest.get("requires_full_machine_consensus") is not True
        ):
            raise SupplementSpecError("primary prefilter contract is unsupported")
        label = manifest.get("label")
        if label not in PRIMARY_LABELS:
            raise SupplementSpecError(
                f"primary prefilter label must be one of {list(PRIMARY_LABELS)}"
            )
        validate_prefilter_artifact(
            binding.path,
            expected_label=label,
            expected_model_hashes={
                model: models[model].sha256 for model in ("original", "champion")
            },
        )

        selected_value = manifest.get("selected")
        if not isinstance(selected_value, Mapping):
            raise SupplementSpecError("primary prefilter selected binding is missing")
        selected = _bound_transitive_file(
            selected_value.get("path"),
            selected_value.get("sha256"),
            "primary prefilter selected source",
            frozen,
        )
        positions = _normalized_positions(selected.path)
        if selected.path.read_bytes() != _canonical_jsonl(positions):
            raise SupplementSpecError(
                "primary prefilter selected source is not canonical normalized JSONL"
            )
        semantic_ids = tuple(row["semanticSha256"] for row in positions)
        orbits = tuple(symmetry_orbit_sha256(row) for row in positions)
        if (
            selected_value.get("row_count") != len(positions)
            or selected_value.get("symmetry_orbit_count") != len(set(orbits))
            or len(orbits) != len(set(orbits))
        ):
            raise SupplementSpecError("primary prefilter selected inventory changed")

        normalized = manifest.get("normalized")
        if not isinstance(normalized, Mapping):
            raise SupplementSpecError("primary prefilter normalized binding is missing")
        normalized_binding = _bound_transitive_file(
            normalized.get("path"),
            normalized.get("sha256"),
            "primary prefilter normalized source",
            frozen,
        )
        normalized_rows = _normalized_positions(normalized_binding.path)
        normalized_ids = [row["semanticSha256"] for row in normalized_rows]
        if (
            normalized.get("row_count") != len(normalized_rows)
            or normalized.get("semantic_ids_sha256") != canonical_sha256(normalized_ids)
            or not set(semantic_ids).issubset(normalized_ids)
        ):
            raise SupplementSpecError("primary prefilter normalized inventory changed")

        expected_models = {
            model: models[model].sha256 for model in ("original", "champion")
        }
        if manifest.get("model_hashes") != expected_models:
            raise SupplementSpecError(
                "primary prefilter uses different original/champion models"
            )
        analyses = manifest.get("analyses")
        if not isinstance(analyses, Mapping) or set(analyses) != set(PREFILTER_ROLES):
            raise SupplementSpecError(
                "primary prefilter analysis inventory is incomplete"
            )
        for role in PREFILTER_ROLES:
            identity = analyses[role]
            if not isinstance(identity, Mapping):
                raise SupplementSpecError(
                    f"primary prefilter analysis {role} is malformed"
                )
            model = role.split("/", 1)[0]
            output = _bound_transitive_file(
                identity.get("path"),
                identity.get("sha256"),
                f"primary {role} analysis",
                frozen,
            )
            execution = _bound_transitive_file(
                identity.get("manifest_path"),
                identity.get("manifest_sha256"),
                f"primary {role} execution manifest",
                frozen,
            )
            query = _bound_transitive_file(
                identity.get("query_path"),
                identity.get("query_sha256"),
                f"primary {role} query",
                frozen,
            )
            if (
                identity.get("model_sha256") != models[model].sha256
                or identity.get("config_sha256") != analysis_config.sha256
            ):
                raise SupplementSpecError(f"primary {role} execution identity changed")
            execution_value = _load_json_object(
                execution.path, f"primary {role} execution manifest"
            )
            if (
                execution_value.get("contract") != ANALYSIS_RUN_CONTRACT
                or execution_value.get("output_path") != str(output.path)
                or execution_value.get("output_sha256") != output.sha256
                or execution_value.get("query_path") != str(query.path)
                or execution_value.get("query_sha256") != query.sha256
                or execution_value.get("model_sha256") != models[model].sha256
                or execution_value.get("config_sha256") != analysis_config.sha256
            ):
                raise SupplementSpecError(
                    f"primary {role} execution provenance changed"
                )
        return (
            PrimaryPrefilter(
                manifest=binding,
                label=label,
                selected=selected,
                row_count=len(positions),
                semantic_ids=semantic_ids,
                symmetry_orbits=orbits,
            ),
            tuple(frozen),
        )
    except SupplementSpecError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SupplementSpecError(
            f"primary prefilter {binding.path} is invalid: {exc}"
        ) from exc


def _load_legacy_read_only_spec(
    requested: Path, raw: Mapping[str, Any]
) -> SupplementSpec:
    """Load v1 provenance fixtures without permitting legacy execution.

    The downstream pipeline accepted these summaries before the production
    command contract was tightened.  They remain verifiable as historical
    ancestry, but :class:`CurationSupplement` refuses to execute them.
    """

    _require_exact_keys(raw, _LEGACY_READ_ONLY_SPEC_KEYS, "supplement specification")
    payload = dict(raw)
    identity = payload.pop("spec_sha256", None)
    if (
        raw.get("schema_version") != 1
        or raw.get("contract") != SPEC_CONTRACT
        or _require_sha256(identity, "supplement specification identity")
        != canonical_sha256(payload)
    ):
        raise SupplementSpecError("legacy supplement specification is invalid")
    run_root = _required_directory(raw["run_root"], "run root")
    training = _required_directory(raw["training_input_root"], "training input root")
    work_root = _future_directory(raw["work_root"], "supplement work root")
    katago = _file_binding(raw["katago"], "KataGo binary")
    analysis_config = _file_binding(raw["analysis_config"], "analysis config")
    selfplay_config = _file_binding(raw["selfplay_config"], "self-play config")
    policy = _file_binding(raw["policy"], "promotion policy")
    models_value = _require_exact_keys(
        raw["models"], {"original", "champion"}, "model bindings"
    )
    models = {
        role: _file_binding(models_value[role], f"{role} model")
        for role in ("original", "champion")
    }
    policy_value = load_policy(policy.path)
    minima = policy_pool_minima(policy_value)
    topology_value = _require_exact_keys(
        raw["topology"],
        {"shards_per_role", "gpus", "per_gpu_parallelism"},
        "supplement topology",
    )
    raw_gpus = topology_value["gpus"]
    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise SupplementSpecError("legacy topology has no GPUs")
    topology = Topology(
        int(topology_value["shards_per_role"]),
        tuple(raw_gpus),
        tuple(raw_gpus),
        int(topology_value["per_gpu_parallelism"]),
    )
    primary_bindings = tuple(
        _file_binding(value, f"primary prefilter manifest {index}")
        for index, value in enumerate(raw["primary_prefilter_manifests"])
    )
    primary: list[PrimaryPrefilter] = []
    frozen: list[FileBinding] = [
        katago,
        analysis_config,
        selfplay_config,
        policy,
        models["original"],
        models["champion"],
    ]
    for binding in primary_bindings:
        source, source_frozen = _verify_primary_prefilter(
            binding, models=models, analysis_config=analysis_config
        )
        primary.append(source)
        frozen.extend(source_frozen)
    target_value = _require_exact_keys(
        raw["target_counts"], set(LEAD_LABELS), "target Lead counts"
    )
    targets = {label: int(target_value[label]) for label in LEAD_LABELS}
    placeholder_directory = DirectoryBinding(
        models["original"].path.parent,
        canonical_sha256(
            [
                {
                    "path": models["original"].path.name,
                    "size": models["original"].path.stat().st_size,
                    "sha256": models["original"].sha256,
                }
            ]
        ),
    )
    revision = "0" * 40
    return SupplementSpec(
        path=requested.resolve(),
        file_sha256=file_sha256(requested),
        identity=identity,
        raw=raw,
        deployment=DeploymentBinding(
            run_root, revision, hashlib.sha256(revision.encode()).hexdigest()
        ),
        deployment_manifest=selfplay_config,
        run_root=run_root,
        training_input_root=training,
        work_root=work_root,
        katago=katago,
        analysis_config=analysis_config,
        selfplay_config=selfplay_config,
        selfplay_models_directory=placeholder_directory,
        selfplay_override_args=(),
        policy=policy,
        models=models,
        game_count=int(raw["game_count"]),
        topology=topology,
        policy_minima={label: int(minima[label]) for label in LEAD_LABELS},
        consensus_reserve_fraction=0.0,
        target_counts=targets,
        primary_prefilter_inventory=(
            primary_bindings[0] if primary_bindings else selfplay_config
        ),
        primary_prefilters=tuple(primary),
        round=1,
        prior_round_summaries=(),
        downstream_accepted_counts=None,
        frozen_files=tuple(
            {(binding.path, binding.sha256): binding for binding in frozen}.values()
        ),
        legacy_read_only=True,
    )


def load_supplement_spec(
    path: Path,
    *,
    revision_reader: Callable[[Path], str] = _git_revision,
    repository_status_reader: Callable[[Path], str] = _git_status,
) -> SupplementSpec:
    """Load and fully validate the strict canonical v1 specification."""

    requested = Path(path)
    try:
        raw = _load_json_object(requested, "curation supplement specification")
    except ValueError as exc:
        raise SupplementSpecError(str(exc)) from exc
    if set(raw) == _LEGACY_READ_ONLY_SPEC_KEYS:
        return _load_legacy_read_only_spec(requested, raw)
    _require_exact_keys(raw, _SPEC_KEYS, "supplement specification")
    if raw.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise SupplementSpecError("supplement schema_version must be 1")
    if raw.get("contract") != SPEC_CONTRACT:
        raise SupplementSpecError("supplement specification contract is unsupported")
    payload = dict(raw)
    identity = payload.pop("spec_sha256", None)
    if _require_sha256(
        identity, "supplement specification identity"
    ) != canonical_sha256(payload):
        raise SupplementSpecError("supplement specification self-hash is invalid")

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
        raise SupplementSpecError(
            "deployment source_revision must be a lowercase Git hash"
        )
    revision_hash = _require_sha256(
        deployment_value["source_sha256"], "deployment source hash"
    )
    if revision_hash != hashlib.sha256(revision.encode("utf-8")).hexdigest():
        raise SupplementSpecError("deployment source hash does not bind revision")
    if revision_reader(repository) != revision:
        raise SupplementSpecError("deployment repository revision changed")
    if repository_status_reader(repository):
        raise SupplementSpecError("deployment repository has uncommitted changes")
    deployment = DeploymentBinding(repository, revision, revision_hash)
    deployment_manifest = _file_binding(
        raw["deployment_manifest"], "deployment manifest"
    )
    _verify_deployment(deployment_manifest, deployment, error_type=SupplementSpecError)

    run_root = _required_directory(raw["run_root"], "run root")
    training_input_root = _required_directory(
        raw["training_input_root"], "training input root"
    )
    work_root = _future_directory(raw["work_root"], "supplement work root")
    if not _is_within(work_root, run_root) or work_root == run_root:
        raise SupplementSpecError("work root must be strictly beneath run root")
    if _paths_overlap(work_root, training_input_root):
        raise SupplementSpecError(
            "supplement work root must be quarantined outside training input root"
        )

    katago = _file_binding(raw["katago"], "KataGo binary")
    analysis_config = _file_binding(raw["analysis_config"], "analysis config")
    selfplay_config = _file_binding(raw["selfplay_config"], "self-play config")
    selfplay_models_directory = _directory_binding(
        raw["selfplay_models_directory"], "self-play models directory"
    )
    selfplay_override_args = _validate_override_args(raw["selfplay_override_args"])
    policy = _file_binding(raw["policy"], "promotion policy")
    try:
        validate_deterministic_analysis_config(analysis_config.path)
        policy_value = load_policy(policy.path)
        if not isinstance(policy_value, Mapping):
            raise ValueError("policy root is not an object")
    except (KeyError, TypeError, ValueError) as exc:
        raise SupplementSpecError(f"frozen policy/config is invalid: {exc}") from exc

    model_values = _require_exact_keys(
        raw["models"], {"original", "champion"}, "model bindings"
    )
    models = {
        role: _file_binding(model_values[role], f"{role} model")
        for role in ("original", "champion")
    }
    if (
        models["original"].path == models["champion"].path
        or models["original"].sha256 == models["champion"].sha256
    ):
        raise SupplementSpecError("original and champion models must be distinct")
    models_inventory = _directory_inventory(selfplay_models_directory.path)
    expected_model_path = selfplay_models_directory.path / "model.bin.gz"
    if (
        len(models_inventory) != 1
        or models_inventory[0]["path"] != "model.bin.gz"
        or models_inventory[0]["sha256"] != models["original"].sha256
        or models["original"].path != expected_model_path
    ):
        raise SupplementSpecError(
            "self-play models directory must contain only the selected "
            "original model as model.bin.gz"
        )

    game_count = raw["game_count"]
    if type(game_count) is not int or not 1 <= game_count <= 1_000_000_000:
        raise SupplementSpecError("game_count must be an integer between 1 and 1e9")

    topology_value = _require_exact_keys(
        raw["topology"],
        {"shards_per_role", "gpus", "selfplay_gpus", "per_gpu_parallelism"},
        "supplement topology",
    )
    shards = topology_value["shards_per_role"]
    parallelism = topology_value["per_gpu_parallelism"]
    if type(shards) is not int or not 1 <= shards <= 64:
        raise SupplementSpecError("shards_per_role must be between 1 and 64")
    if type(parallelism) is not int or not 1 <= parallelism <= 64:
        raise SupplementSpecError("per_gpu_parallelism must be between 1 and 64")
    parsed_gpus: dict[str, tuple[str, ...]] = {}
    for field in ("gpus", "selfplay_gpus"):
        raw_gpus = topology_value[field]
        if not isinstance(raw_gpus, list) or not raw_gpus:
            raise SupplementSpecError(f"topology {field} must be a nonempty array")
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
                raise SupplementSpecError(
                    "GPU identifiers must be unique nonempty tokens"
                )
            gpus.append(gpu)
        parsed_gpus[field] = tuple(gpus)
    topology = Topology(
        shards, parsed_gpus["gpus"], parsed_gpus["selfplay_gpus"], parallelism
    )
    _validate_selfplay_gpu_mapping(selfplay_config, topology.selfplay_gpus)

    reserve_fraction = raw["consensus_reserve_fraction"]
    if (
        isinstance(reserve_fraction, bool)
        or not isinstance(reserve_fraction, (int, float))
        or not math.isfinite(float(reserve_fraction))
        or not 0 < float(reserve_fraction) <= 10
    ):
        raise SupplementSpecError(
            "consensus_reserve_fraction must be finite and in (0, 10]"
        )
    reserve_fraction = float(reserve_fraction)
    try:
        minima_value = policy_pool_minima(policy_value)
        policy_minima = {label: int(minima_value[label]) for label in LEAD_LABELS}
    except (KeyError, TypeError, ValueError) as exc:
        raise SupplementSpecError(
            "promotion policy does not define Lead consensus minima"
        ) from exc

    target_values = _require_exact_keys(
        raw["target_counts"], set(LEAD_LABELS), "target Lead counts"
    )
    target_counts: dict[str, int] = {}
    for label in LEAD_LABELS:
        value = target_values[label]
        if type(value) is not int or value < 1:
            raise SupplementSpecError(f"target count for {label} must be positive")
        reserved_minimum = policy_minima[label] + math.ceil(
            policy_minima[label] * reserve_fraction
        )
        if value < reserved_minimum:
            raise SupplementSpecError(
                f"target count for {label} must be at least {reserved_minimum} "
                "under the configured consensus reserve"
            )
        target_counts[label] = value

    primary_values = raw["primary_prefilter_manifests"]
    if not isinstance(primary_values, list):
        raise SupplementSpecError("primary_prefilter_manifests must be an array")
    primary_bindings = [
        _file_binding(value, f"primary prefilter manifest {index}")
        for index, value in enumerate(primary_values)
    ]
    if [str(item.path) for item in primary_bindings] != sorted(
        str(item.path) for item in primary_bindings
    ):
        raise SupplementSpecError(
            "primary_prefilter_manifests must be sorted by canonical path"
        )
    if len({item.path for item in primary_bindings}) != len(primary_bindings):
        raise SupplementSpecError("primary prefilter manifest is listed more than once")

    primary_inventory = _file_binding(
        raw["primary_prefilter_inventory"], "primary prefilter inventory"
    )
    try:
        inventory = _load_json_object(
            primary_inventory.path, "primary prefilter inventory"
        )
        inventory_payload = dict(inventory)
        inventory_identity = inventory_payload.pop("inventory_sha256", None)
        expected_entries = [
            {"path": str(binding.path), "sha256": binding.sha256}
            for binding in primary_bindings
        ]
        if (
            set(inventory)
            != {
                "schema_version",
                "contract",
                "manifests",
                "inventory_sha256",
            }
            or inventory.get("schema_version") != 1
            or inventory.get("contract") != PRIMARY_INVENTORY_CONTRACT
            or inventory_identity != canonical_sha256(inventory_payload)
            or inventory.get("manifests") != expected_entries
        ):
            raise SupplementSpecError(
                "primary prefilter inventory is incomplete or changed"
            )
    except SupplementSpecError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise SupplementSpecError(
            f"primary prefilter inventory is invalid: {exc}"
        ) from exc

    round_number = raw["round"]
    if type(round_number) is not int or round_number < 1:
        raise SupplementSpecError("round must be a positive integer")
    prior_values = raw["prior_round_summaries"]
    if not isinstance(prior_values, list):
        raise SupplementSpecError("prior_round_summaries must be an array")
    prior_summaries = tuple(
        _file_binding(value, f"prior round summary {index}")
        for index, value in enumerate(prior_values)
    )
    if [str(item.path) for item in prior_summaries] != sorted(
        str(item.path) for item in prior_summaries
    ) or len({item.path for item in prior_summaries}) != len(prior_summaries):
        raise SupplementSpecError(
            "prior_round_summaries must be unique and path-sorted"
        )
    downstream_value = raw["downstream_accepted_counts"]
    downstream_counts: Mapping[str, int] | None
    if downstream_value is None:
        downstream_counts = None
    else:
        accepted = _require_exact_keys(
            downstream_value, set(LEAD_LABELS), "downstream accepted counts"
        )
        if any(
            type(accepted[label]) is not int or accepted[label] < 0
            for label in LEAD_LABELS
        ):
            raise SupplementSpecError(
                "downstream accepted counts must be nonnegative integers"
            )
        downstream_counts = {label: accepted[label] for label in LEAD_LABELS}
    if round_number == 1 and (prior_summaries or downstream_counts is not None):
        raise SupplementSpecError(
            "round 1 may not bind prior summaries or downstream counts"
        )
    if round_number > 1 and (
        len(prior_summaries) != round_number - 1 or downstream_counts is None
    ):
        raise SupplementSpecError(
            "later rounds require every prior summary and downstream counts"
        )

    primary: list[PrimaryPrefilter] = []
    frozen: list[FileBinding] = [
        deployment_manifest,
        katago,
        analysis_config,
        selfplay_config,
        policy,
        models["original"],
        models["champion"],
        primary_inventory,
        *prior_summaries,
    ]
    for binding in primary_bindings:
        source, source_frozen = _verify_primary_prefilter(
            binding, models=models, analysis_config=analysis_config
        )
        primary.append(source)
        frozen.extend(source_frozen)

    seen_semantic: dict[str, Path] = {}
    seen_orbits: dict[str, Path] = {}
    for source in primary:
        for semantic_id, orbit in zip(
            source.semantic_ids, source.symmetry_orbits, strict=True
        ):
            if semantic_id in seen_semantic:
                raise SupplementSpecError(
                    "primary prefilter sources contain duplicate semantic positions"
                )
            if orbit in seen_orbits:
                raise SupplementSpecError(
                    "primary prefilter sources contain duplicate symmetry orbits"
                )
            seen_semantic[semantic_id] = source.manifest.path
            seen_orbits[orbit] = source.manifest.path

    for protected in {requested.resolve(), *(binding.path for binding in frozen)}:
        if _is_within(protected, work_root):
            raise SupplementSpecError(
                "frozen inputs may not be stored beneath supplement work root"
            )

    unique_frozen = {(binding.path, binding.sha256): binding for binding in frozen}
    return SupplementSpec(
        path=requested.resolve(),
        file_sha256=file_sha256(requested),
        identity=identity,
        raw=raw,
        deployment=deployment,
        deployment_manifest=deployment_manifest,
        run_root=run_root,
        training_input_root=training_input_root,
        work_root=work_root,
        katago=katago,
        analysis_config=analysis_config,
        selfplay_config=selfplay_config,
        selfplay_models_directory=selfplay_models_directory,
        selfplay_override_args=selfplay_override_args,
        policy=policy,
        models=models,
        game_count=game_count,
        topology=topology,
        policy_minima=policy_minima,
        consensus_reserve_fraction=reserve_fraction,
        target_counts=target_counts,
        primary_prefilter_inventory=primary_inventory,
        primary_prefilters=tuple(primary),
        round=round_number,
        prior_round_summaries=prior_summaries,
        downstream_accepted_counts=downstream_counts,
        frozen_files=tuple(
            unique_frozen[key]
            for key in sorted(unique_frozen, key=lambda item: str(item[0]))
        ),
    )


load_spec = load_supplement_spec


def supplement_layout(spec: SupplementSpec) -> SupplementLayout:
    work = spec.work_root
    selfplay = work / "selfplay-corpus"
    selfplay_attempt = work / "selfplay-attempt"
    query_directory = work / "prefilter-query-bundle"
    return SupplementLayout(
        selfplay_attempt_root=selfplay_attempt,
        selfplay_attempt_output=selfplay_attempt / "working",
        selfplay_attempt_journal=selfplay_attempt / "attempt.json",
        selfplay_orphans=selfplay_attempt / "orphaned",
        selfplay_directory=selfplay,
        selfplay_receipt=selfplay / "receipt.json",
        harvest_plan=work / "harvest-plan.json",
        harvest_directory=work / "harvested",
        normalized=work / "normalized.jsonl",
        normalized_manifest=work / "normalized.manifest.json",
        rejected_duplicates=work / "normalized.rejected-duplicates.jsonl",
        rejected_duplicates_manifest=work
        / "normalized.rejected-duplicates.manifest.json",
        query_directory=query_directory,
        query_manifest=query_directory / "manifest.json",
        query_shards=work / "query-shards",
        analysis_shards=work / "analysis-shards",
        analyses=work / "analyses",
        selected=work / "selected",
        summary=work / "summary.json",
        gpu_ownership_archive=work / "gpu-ownership-released.json",
        stage_attempts=work / "stage-attempts",
        status=work / "status.json",
        lock=work / ".supplement.lock",
    )


def primary_counts(spec: SupplementSpec) -> Mapping[str, int]:
    return {
        label: sum(
            source.row_count
            for source in spec.primary_prefilters
            if source.label == label
        )
        for label in LEAD_LABELS
    }


def source_deficits(
    counts: Mapping[str, int], targets: Mapping[str, int]
) -> Mapping[str, int]:
    return {
        label: int(targets[label]) - int(counts.get(label, 0))
        for label in LEAD_LABELS
        if int(counts.get(label, 0)) < int(targets[label])
    }


def deficit_basis_counts(spec: SupplementSpec) -> Mapping[str, int]:
    return (
        primary_counts(spec)
        if spec.downstream_accepted_counts is None
        else dict(spec.downstream_accepted_counts)
    )


def generation_limits(
    spec: SupplementSpec, deficits: Mapping[str, int]
) -> Mapping[str, int]:
    """Reserve for downstream attrition instead of generating exact deficits."""

    return {
        label: spec.target_counts[label] for label in LEAD_LABELS if label in deficits
    }


def render_selfplay_argv(
    spec: SupplementSpec, *, output_directory: Path | None = None
) -> tuple[str, ...]:
    """Render the exact reviewed finite KataGo self-play command."""

    output = (
        supplement_layout(spec).selfplay_attempt_output
        if output_directory is None
        else Path(output_directory)
    )
    argv = [
        str(spec.katago.path),
        "selfplay",
        "-max-games-total",
        str(spec.game_count),
        "-output-dir",
        str(output),
        "-models-dir",
        str(spec.selfplay_models_directory.path),
        "-config",
        str(spec.selfplay_config.path),
    ]
    for flag, value in spec.selfplay_override_args:
        argv.extend((flag, value))
    return tuple(argv)


build_selfplay_argv = render_selfplay_argv


def plan_selfplay_command(spec: SupplementSpec) -> Mapping[str, Any]:
    return {
        "argv": list(render_selfplay_argv(spec)),
        "shell": False,
        "game_count": spec.game_count,
        "output_directory": str(supplement_layout(spec).selfplay_attempt_output),
        "models_directory": str(spec.selfplay_models_directory.path),
        "selfplay_gpus": list(spec.topology.selfplay_gpus),
    }


def plan_analysis_jobs(spec: SupplementSpec) -> tuple[AnalysisJob, ...]:
    """Return deterministic role/shard/GPU assignments."""

    layout = supplement_layout(spec)
    jobs: list[AnalysisJob] = []
    index = 0
    for role in PREFILTER_ROLES:
        model, mode = role.split("/", 1)
        for shard_index in range(spec.topology.shards_per_role):
            jobs.append(
                AnalysisJob(
                    role=role,
                    model=model,
                    mode=mode,
                    shard_index=shard_index,
                    gpu=spec.topology.gpus[index % len(spec.topology.gpus)],
                    query_path=layout.query_shards
                    / mode
                    / f"shard-{shard_index:03d}.jsonl",
                    output_path=layout.analysis_shards
                    / model
                    / mode
                    / f"shard-{shard_index:03d}.jsonl",
                )
            )
            index += 1
    return tuple(jobs)


def plan_analysis_commands(spec: SupplementSpec) -> tuple[Mapping[str, Any], ...]:
    """Render every deterministic analysis process assignment."""

    return tuple(
        {
            "role": job.role,
            "shard_index": job.shard_index,
            "gpu": job.gpu,
            "argv": [
                str(spec.katago.path),
                "analysis",
                "-config",
                str(spec.analysis_config.path),
                "-model",
                str(spec.models[job.model].path),
            ],
            "query_path": str(job.query_path),
            "output_path": str(job.output_path),
            "environment": {"CUDA_VISIBLE_DEVICES": job.gpu},
        }
        for job in plan_analysis_jobs(spec)
    )


def _proc_argv_sha256(argv: Sequence[str]) -> str:
    command = b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    return hashlib.sha256(command).hexdigest()


def _owned_command_hashes(spec: SupplementSpec) -> tuple[str, ...]:
    commands = [render_selfplay_argv(spec)]
    for model in ("original", "champion"):
        commands.append(
            (
                str(spec.katago.path),
                "analysis",
                "-config",
                str(spec.analysis_config.path),
                "-model",
                str(spec.models[model].path),
            )
        )
    return tuple(sorted({_proc_argv_sha256(argv) for argv in commands}))


def _owned_gpu_ids(spec: SupplementSpec) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(spec.topology.selfplay_gpus) | set(spec.topology.gpus),
            key=lambda value: (
                not value.isdigit(),
                int(value) if value.isdigit() else value,
            ),
        )
    )


def _ownership_topology(spec: SupplementSpec) -> Mapping[str, Any]:
    return {
        "gpus": list(_owned_gpu_ids(spec)),
        "selfplay_gpus": list(spec.topology.selfplay_gpus),
        "analysis_gpus": list(spec.topology.gpus),
        "per_gpu_parallelism": spec.topology.per_gpu_parallelism,
        "shards_per_role": spec.topology.shards_per_role,
        "round": spec.round,
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(path: Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise SupplementContradiction("immutable output parent is unsafe")
    if _lexists(target):
        if target.is_symlink() or not target.is_file() or target.read_bytes() != data:
            raise SupplementContradiction(
                f"existing immutable output conflicts: {target}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.rename(os.fspath(temporary), os.fspath(target))
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _publish_immutable(path, (canonical_json(value) + "\n").encode("utf-8"))


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise SupplementContradiction("status parent is not a regular directory")
    if _lexists(target) and (target.is_symlink() or not target.is_file()):
        raise SupplementContradiction("status path is not a regular file")
    data = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(os.fspath(temporary), os.fspath(target))
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_work_root(spec: SupplementSpec, *, create: bool) -> None:
    if not _lexists(spec.work_root):
        if not create:
            return
        relative = spec.work_root.relative_to(spec.run_root)
        current = spec.run_root
        for part in relative.parts:
            current = current / part
            if _lexists(current):
                if current.is_symlink() or not current.is_dir():
                    raise SupplementContradiction(
                        f"unsafe work-root path component: {current}"
                    )
            else:
                current.mkdir()
        return
    if (
        spec.work_root.is_symlink()
        or not spec.work_root.is_dir()
        or str(spec.work_root.resolve()) != str(spec.work_root)
    ):
        raise SupplementContradiction("supplement work root is unsafe")


def _validate_sgf_game(game: str, source: str) -> None:
    try:
        properties = _sgf_properties(game, source)
    except PromotionAuditorError as exc:
        raise SupplementContradiction(f"malformed self-play SGF: {exc}") from exc
    if (
        properties.get("GM") != ("1",)
        or properties.get("FF") != ("4",)
        or len(properties.get("SZ", ())) != 1
        or len(properties.get("RE", ())) != 1
        or _TERMINAL_RESULT_RE.fullmatch(properties["RE"][0]) is None
    ):
        raise SupplementContradiction(
            f"{source}: SGF lacks required Go/FF[4]/size/terminal-result properties"
        )


def _inventory_selfplay(root: Path) -> tuple[list[dict[str, Any]], int]:
    if root.is_symlink() or not root.is_dir():
        raise SupplementContradiction("self-play output is not a regular directory")
    inventory: list[dict[str, Any]] = []
    games = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SupplementContradiction(
                f"self-play output contains a symlink: {path}"
            )
        if not path.is_file() or path == root / "receipt.json":
            continue
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        game_rows = 0
        if suffix == ".sgfs":
            try:
                data = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise SupplementContradiction(
                    f"self-play SGFS is not UTF-8: {path}"
                ) from exc
            if not data.endswith("\n"):
                raise SupplementContradiction(
                    f"self-play SGFS is not newline-terminated: {path}"
                )
            lines = data.splitlines()
            if not lines or any(not line for line in lines):
                raise SupplementContradiction(f"self-play SGFS is malformed: {path}")
            for line_number, game in enumerate(lines, start=1):
                _validate_sgf_game(game, f"{path}:{line_number}")
            game_rows = len(lines)
        elif suffix == ".sgf":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise SupplementContradiction(
                    f"self-play SGF is not UTF-8: {path}"
                ) from exc
            _validate_sgf_game(text, str(path))
            game_rows = 1
        elif path.name.endswith((".tmp", ".partial", ".incomplete")):
            raise SupplementContradiction(
                f"partial self-play artifact is forbidden: {path}"
            )
        games += game_rows
        inventory.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "game_count": game_rows,
            }
        )
    if not inventory or games <= 0:
        raise SupplementContradiction("self-play produced no SGF/SGFS games")
    return inventory, games


def _selfplay_receipt_value(
    spec: SupplementSpec,
    inventory: Sequence[Mapping[str, Any]],
    *,
    process_identity: Mapping[str, Any],
    attempt_generation: int,
) -> dict[str, Any]:
    layout = supplement_layout(spec)
    value: dict[str, Any] = {
        "schema_version": 1,
        "contract": SELFPLAY_RECEIPT_CONTRACT,
        "spec": {
            "path": str(spec.path),
            "sha256": spec.file_sha256,
            "identity": spec.identity,
        },
        "argv": list(render_selfplay_argv(spec)),
        "shell": False,
        "returncode": 0,
        "process_identity": dict(process_identity),
        "attempt_generation": attempt_generation,
        "katago_sha256": spec.katago.sha256,
        "model_sha256": spec.models["original"].sha256,
        "selfplay_config_sha256": spec.selfplay_config.sha256,
        "models_directory": {
            "path": str(spec.selfplay_models_directory.path),
            "sha256": spec.selfplay_models_directory.sha256,
        },
        "selfplay_gpus": list(spec.topology.selfplay_gpus),
        "game_count": spec.game_count,
        "output_directory": str(layout.selfplay_directory),
        "outputs": list(inventory),
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _harvest_source_directories(
    layout: SupplementLayout,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    inventory, _ = _inventory_selfplay(layout.selfplay_directory)
    has_sgfs = any(item["path"].lower().endswith(".sgfs") for item in inventory)
    has_sgf = any(item["path"].lower().endswith(".sgf") for item in inventory)
    return (
        (layout.selfplay_directory,) if has_sgfs else (),
        (layout.selfplay_directory,) if has_sgf else (),
    )


def _expected_harvest_plan(spec: SupplementSpec) -> Mapping[str, Any]:
    layout = supplement_layout(spec)
    sgfs_dirs, sgf_dirs = _harvest_source_directories(layout)
    argv = build_harvest_argv(
        katago=spec.katago.path,
        sgfs_dirs=sgfs_dirs,
        sgf_dirs=sgf_dirs,
        training_input_roots=[spec.training_input_root],
        output_dir=layout.harvest_directory,
        threads=1,
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "contract": HARVEST_PLAN_CONTRACT,
        "katago_path": str(spec.katago.path),
        "katago_sha256": spec.katago.sha256,
        "argv": list(argv),
        "inputs": _source_inventory(sgfs_dirs=sgfs_dirs, sgf_dirs=sgf_dirs),
        "training_input_roots": [str(spec.training_input_root)],
        "output_dir": str(layout.harvest_directory),
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _harvest_position_sources(layout: SupplementLayout) -> tuple[Path, ...]:
    receipt = _load_json_object(
        layout.harvest_directory / "receipt.json", "harvest receipt"
    )
    paths = tuple(
        sorted(
            (
                layout.harvest_directory / item["path"]
                for item in receipt.get("outputs", [])
                if isinstance(item, Mapping)
                and isinstance(item.get("path"), str)
                and item["path"].endswith(".startposes.txt")
            ),
            key=str,
        )
    )
    if not paths:
        raise SupplementContradiction("harvest receipt has no PositionSample output")
    return paths


def _filter_normalized_positions(
    spec: SupplementSpec, sources: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    rows, base = normalize_sources(sources)
    primary_semantic = {
        semantic_id: source.manifest
        for source in spec.primary_prefilters
        for semantic_id in source.semantic_ids
    }
    primary_orbits = {
        orbit: source.manifest
        for source in spec.primary_prefilters
        for orbit in source.symmetry_orbits
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_orbits: dict[str, str] = {}
    for row in rows:
        semantic = row["semanticSha256"]
        orbit = symmetry_orbit_sha256(row)
        reason = None
        duplicate_binding = None
        if semantic in primary_semantic:
            reason = "primary_semantic_duplicate"
            duplicate_binding = primary_semantic[semantic]
        elif orbit in primary_orbits:
            reason = "primary_symmetry_orbit_duplicate"
            duplicate_binding = primary_orbits[orbit]
        elif orbit in accepted_orbits:
            reason = "supplement_symmetry_orbit_duplicate"
        if reason is None:
            accepted.append(row)
            accepted_orbits[orbit] = semantic
            continue
        rejected.append(
            {
                "semanticSha256": semantic,
                "symmetryOrbitSha256": orbit,
                "reason": reason,
                "curationSource": row["curationSource"],
                "duplicateOfSemanticSha256": accepted_orbits.get(orbit),
                "primaryManifest": (
                    None
                    if duplicate_binding is None
                    else {
                        "path": str(duplicate_binding.path),
                        "sha256": duplicate_binding.sha256,
                    }
                ),
            }
        )
    return accepted, rejected, base


def _assert_expected_files(root: Path, allowed: Iterable[Path], role: str) -> None:
    if not _lexists(root):
        return
    if root.is_symlink() or not root.is_dir():
        raise SupplementContradiction(f"{role} root is unsafe")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SupplementContradiction(f"{role} contains a symlink: {path}")
        if path.is_file():
            actual.add(path.resolve())
    allowed_set = {Path(path).resolve() for path in allowed}
    extra = actual.difference(allowed_set)
    if extra:
        raise SupplementContradiction(
            f"{role} contains unexpected artifacts: {sorted(map(str, extra))}"
        )


def _analysis_output_path(layout: SupplementLayout, role: str) -> Path:
    return layout.analyses.joinpath(*role.split("/")).with_suffix(".jsonl")


def _selected_paths(layout: SupplementLayout, label: str) -> tuple[Path, Path]:
    return (
        layout.selected / f"{label}.jsonl",
        layout.selected / f"{label}.manifest.json",
    )


def _artifact_binding(path: Path) -> Mapping[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _summary_value(
    spec: SupplementSpec,
    *,
    counts: Mapping[str, int],
    deficits: Mapping[str, int],
    supplemental: Mapping[str, int],
    state: str,
) -> dict[str, Any]:
    layout = supplement_layout(spec)
    basis_counts = (
        counts
        if spec.downstream_accepted_counts is None
        else spec.downstream_accepted_counts
    )
    final_counts = {
        label: int(basis_counts[label]) + int(supplemental.get(label, 0))
        for label in LEAD_LABELS
    }
    selected: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    limits = generation_limits(spec, deficits)
    if deficits:
        for label in LEAD_LABELS:
            if label not in deficits:
                continue
            output, manifest = _selected_paths(layout, label)
            selected[label] = {
                "limit": limits[label],
                "row_count": supplemental[label],
                "output": _artifact_binding(output),
                "manifest": _artifact_binding(manifest),
            }
        for role in PREFILTER_ROLES:
            output = _analysis_output_path(layout, role)
            analyses[role] = {
                "output": _artifact_binding(output),
                "manifest": _artifact_binding(Path(str(output) + ".manifest.json")),
                "shards": [
                    {
                        "output": _artifact_binding(job.output_path),
                        "manifest": _artifact_binding(job.manifest_path),
                        "attempt": _artifact_binding(
                            layout.stage_attempts
                            / (
                                f"analysis-{job.model}-{job.mode}-"
                                f"{job.shard_index:03d}.json"
                            )
                        ),
                    }
                    for job in plan_analysis_jobs(spec)
                    if job.role == role
                ],
            }
    value: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "contract": SUMMARY_CONTRACT,
        "spec": {
            "path": str(spec.path),
            "sha256": spec.file_sha256,
            "identity": spec.identity,
        },
        "state": state,
        "primary_counts": dict(counts),
        "target_counts": dict(spec.target_counts),
        "generation_limits": dict(limits),
        "supplemental_counts": {
            label: int(supplemental.get(label, 0)) for label in LEAD_LABELS
        },
        "final_counts": final_counts,
        "primary_prefilter_manifests": [
            {
                "label": source.label,
                "row_count": source.row_count,
                "path": str(source.manifest.path),
                "sha256": source.manifest.sha256,
            }
            for source in spec.primary_prefilters
        ],
        "selfplay": (
            None
            if not deficits
            else {
                "game_count": spec.game_count,
                "receipt": _artifact_binding(layout.selfplay_receipt),
                "attempt": _artifact_binding(layout.selfplay_attempt_journal),
                "gpu_ownership": _artifact_binding(layout.gpu_ownership_archive),
                "deployment_manifest": _artifact_binding(spec.deployment_manifest.path),
                "round": spec.round,
                "prior_round_summaries": [
                    _artifact_binding(binding.path)
                    for binding in spec.prior_round_summaries
                ],
                "downstream_accepted_counts": (
                    None
                    if spec.downstream_accepted_counts is None
                    else dict(spec.downstream_accepted_counts)
                ),
            }
        ),
        "harvest": (
            None
            if not deficits
            else {
                "plan": _artifact_binding(layout.harvest_plan),
                "receipt": _artifact_binding(layout.harvest_directory / "receipt.json"),
                "attempt": _artifact_binding(layout.stage_attempts / "harvest.json"),
            }
        ),
        "normalized": (
            None
            if not deficits
            else {
                "positions": _artifact_binding(layout.normalized),
                "manifest": _artifact_binding(layout.normalized_manifest),
                "rejected_duplicates": _artifact_binding(layout.rejected_duplicates),
                "rejected_duplicates_manifest": _artifact_binding(
                    layout.rejected_duplicates_manifest
                ),
            }
        ),
        "query_bundle": (
            None
            if not deficits
            else {
                "manifest": _artifact_binding(layout.query_manifest),
                "shards": {
                    mode: _artifact_binding(
                        layout.query_shards / mode / "manifest.json"
                    )
                    for mode in ("standard-2000", "powered-2000")
                },
            }
        ),
        "analyses": analyses,
        "selected": selected,
    }
    value["summary_sha256"] = canonical_sha256(value)
    return value


def plan_next_stage(snapshot: SupplementSnapshot) -> str:
    if snapshot.summary_complete:
        return "complete"
    if not snapshot.deficits:
        return "publish_summary"
    if not snapshot.selfplay_complete:
        return "run_selfplay"
    if not snapshot.harvest_plan_complete:
        return "create_harvest_plan"
    if not snapshot.harvest_complete:
        return "execute_harvest"
    if not snapshot.normalized_complete:
        return "normalize"
    if not snapshot.queries_complete:
        return "generate_queries"
    if snapshot.split_roles_complete < 2:
        return "split_queries"
    if snapshot.analysis_shards_complete < snapshot.analysis_shards_total:
        return "run_analyses"
    if snapshot.merged_roles_complete < len(PREFILTER_ROLES):
        return "merge_analyses"
    if any(
        not snapshot.prefilters_complete.get(label, False)
        for label in snapshot.deficits
    ):
        return "run_prefilters"
    return "publish_summary"


class CurationSupplement:
    """Single-writer, artifact-derived supplemental Lead coordinator."""

    def __init__(
        self,
        spec: SupplementSpec | Path,
        *,
        runners: SupplementRunners = DEFAULT_RUNNERS,
        process_runner: Callable[..., Any] | None = None,
        process_launcher: Callable[..., Any] | None = None,
        revision_reader: Callable[[Path], str] = _git_revision,
        repository_status_reader: Callable[[Path], str] = _git_status,
        ownership_probes: Any = None,
    ) -> None:
        self.spec = (
            spec
            if isinstance(spec, SupplementSpec)
            else load_supplement_spec(
                Path(spec),
                revision_reader=revision_reader,
                repository_status_reader=repository_status_reader,
            )
        )
        if self.spec.legacy_read_only:
            raise SupplementSpecError(
                "legacy template-based supplement specs are provenance-only and "
                "cannot execute"
            )
        replacements = {}
        if process_runner is not None:
            replacements["process"] = process_runner
        if process_launcher is not None:
            replacements["launcher"] = process_launcher
        self.runners = replace(runners, **replacements) if replacements else runners
        self.revision_reader = revision_reader
        self.repository_status_reader = repository_status_reader
        from risk_score.curation_pipeline import (
            DEFAULT_OWNERSHIP_PROBES,
            GpuOwnershipManager,
        )

        self.ownership_probes = (
            DEFAULT_OWNERSHIP_PROBES if ownership_probes is None else ownership_probes
        )
        for name in (
            "process",
            "launcher",
            "harvest_plan",
            "harvest",
            "queries",
            "split",
            "analysis",
            "merge",
            "prefilter",
        ):
            if not callable(getattr(self.runners, name)):
                raise ValueError(f"supplement runner {name!r} must be callable")
        self.layout = supplement_layout(self.spec)
        self._frozen_input_validation_token: (
            tuple[tuple[str, int, int, int, int, int, int], ...] | None
        ) = None
        self.gpu_ownership = GpuOwnershipManager(
            claim_path=self.spec.run_root
            / "evaluation"
            / ".curation-gpu-ownership.json",
            spec_path=self.spec.path,
            spec_sha256=self.spec.file_sha256,
            spec_identity=self.spec.identity,
            configured_gpu_ids=_owned_gpu_ids(self.spec),
            topology_binding=_ownership_topology(self.spec),
            expected_command_sha256s=_owned_command_hashes(self.spec),
            run_root=self.spec.run_root,
            probes=self.ownership_probes,
        )

    def _frozen_input_token(
        self,
    ) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
        model_directory = self.spec.selfplay_models_directory.path
        if model_directory.is_symlink() or not model_directory.is_dir():
            raise SupplementContradiction("self-play models directory changed")
        paths = {self.spec.path, *(binding.path for binding in self.spec.frozen_files)}
        try:
            deployment = _load_json_object(
                self.spec.deployment_manifest.path,
                "deployment manifest",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SupplementContradiction(
                "deployment manifest changed or became invalid"
            ) from exc
        deployment_files = deployment.get("files")
        if not isinstance(deployment_files, Mapping):
            raise SupplementContradiction(
                "deployment manifest file inventory is malformed"
            )
        for role, record in deployment_files.items():
            if (
                not isinstance(role, str)
                or not isinstance(record, Mapping)
                or not isinstance(record.get("path"), str)
            ):
                raise SupplementContradiction(
                    "deployment manifest file inventory is malformed"
                )
            paths.add(_canonical_path(record["path"], f"deployment file {role}"))
        paths.update(model_directory.iterdir())
        observed_paths = [model_directory, *sorted(paths, key=str)]
        token = []
        for path in observed_paths:
            try:
                observed = path.lstat()
            except OSError as exc:
                raise SupplementContradiction(
                    f"frozen input changed: {path}"
                ) from exc
            is_directory = path == model_directory
            if stat.S_ISLNK(observed.st_mode) or (
                not stat.S_ISDIR(observed.st_mode)
                if is_directory
                else not stat.S_ISREG(observed.st_mode)
            ):
                raise SupplementContradiction(f"frozen input changed: {path}")
            token.append(
                (
                    str(path),
                    observed.st_mode,
                    observed.st_dev,
                    observed.st_ino,
                    observed.st_size,
                    observed.st_mtime_ns,
                    observed.st_ctime_ns,
                )
            )
        return tuple(token)

    def _assert_frozen_inputs(self) -> None:
        for root, role in (
            (self.spec.run_root, "run root"),
            (self.spec.training_input_root, "training input root"),
        ):
            if (
                root.is_symlink()
                or not root.is_dir()
                or str(root.resolve()) != str(root)
            ):
                raise SupplementContradiction(f"{role} changed or became unsafe")
        if _paths_overlap(self.spec.work_root, self.spec.training_input_root):
            raise SupplementContradiction(
                "supplement work root now overlaps training input root"
            )
        _safe_work_root(self.spec, create=False)
        if self.revision_reader(self.spec.deployment.repository_path) != (
            self.spec.deployment.source_revision
        ):
            raise SupplementContradiction("deployment repository revision changed")
        if self.repository_status_reader(self.spec.deployment.repository_path):
            raise SupplementContradiction("deployment repository became dirty")
        before = self._frozen_input_token()
        if self._frozen_input_validation_token == before:
            return
        if file_sha256(self.spec.path) != self.spec.file_sha256:
            raise SupplementContradiction(
                "supplement specification changed during execution"
            )
        _verify_deployment(
            self.spec.deployment_manifest,
            self.spec.deployment,
            error_type=SupplementContradiction,
        )
        if (
            canonical_sha256(
                list(_directory_inventory(self.spec.selfplay_models_directory.path))
            )
            != self.spec.selfplay_models_directory.sha256
        ):
            raise SupplementContradiction("self-play models directory changed")
        for binding in self.spec.frozen_files:
            if (
                binding.path.is_symlink()
                or not binding.path.is_file()
                or file_sha256(binding.path) != binding.sha256
            ):
                raise SupplementContradiction(f"frozen input changed: {binding.path}")
        for source in self.spec.primary_prefilters:
            try:
                validate_prefilter_artifact(
                    source.manifest.path,
                    expected_label=source.label,
                    expected_model_hashes={
                        model: self.spec.models[model].sha256
                        for model in ("original", "champion")
                    },
                )
            except (OSError, TypeError, ValueError) as exc:
                raise SupplementContradiction(
                    f"primary prefilter recomputation failed: {source.manifest.path}"
                ) from exc
        after = self._frozen_input_token()
        if after != before:
            raise SupplementContradiction("frozen inputs changed during validation")
        self._frozen_input_validation_token = after

    def _validate_existing_status(self) -> None:
        if not _lexists(self.layout.status):
            return
        try:
            status = _load_json_object(self.layout.status, "supplement status")
        except ValueError as exc:
            raise SupplementContradiction(
                f"supplement status is invalid: {exc}"
            ) from exc
        payload = dict(status)
        identity = payload.pop("status_sha256", None)
        binding = status.get("spec")
        expected_keys = {
            "schema_version",
            "contract",
            "spec",
            "mode",
            "state",
            "next_stage",
            "work_root",
            "primary_counts",
            "target_counts",
            "deficits",
            "supplemental_counts",
            "selfplay_command",
            "topology",
            "progress",
            "artifacts",
            "error",
            "status_sha256",
        }
        expected_topology = {
            "shards_per_role": self.spec.topology.shards_per_role,
            "gpus": list(self.spec.topology.gpus),
            "selfplay_gpus": list(self.spec.topology.selfplay_gpus),
            "per_gpu_parallelism": self.spec.topology.per_gpu_parallelism,
        }
        if (
            set(status) != expected_keys
            or status.get("schema_version") != STATUS_SCHEMA_VERSION
            or status.get("contract") != STATUS_CONTRACT
            or identity != canonical_sha256(payload)
            or not isinstance(binding, Mapping)
            or binding.get("path") != str(self.spec.path)
            or binding.get("sha256") != self.spec.file_sha256
            or binding.get("identity") != self.spec.identity
            or status.get("work_root") != str(self.spec.work_root)
            or status.get("primary_counts") != dict(primary_counts(self.spec))
            or status.get("target_counts") != dict(self.spec.target_counts)
            or status.get("deficits")
            != dict(
                source_deficits(
                    deficit_basis_counts(self.spec), self.spec.target_counts
                )
            )
            or status.get("selfplay_command") != plan_selfplay_command(self.spec)
            or status.get("topology") != expected_topology
        ):
            raise SupplementContradiction(
                "existing status contradicts the frozen supplement specification"
            )

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self._assert_frozen_inputs()
        _safe_work_root(self.spec, create=True)
        if self.layout.lock.is_symlink():
            raise SupplementContradiction("supplement lock may not be a symlink")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(self.layout.lock), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SupplementContradiction("supplement lock is not a regular file")
            if fcntl is None:
                raise SupplementError("supplement coordination requires Unix locking")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SupplementBusy(
                    f"another supplement coordinator holds {self.layout.lock}"
                ) from exc
            yield
        finally:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _assert_absent(self, paths: Iterable[Path], reason: str) -> None:
        present = [str(path) for path in paths if _lexists(path)]
        if present:
            raise SupplementContradiction(f"{reason}: {present}")

    def _current_owner(self) -> ProcessIdentity:
        owner = self.ownership_probes.current_process()
        if (
            not isinstance(owner, ProcessIdentity)
            or not owner.is_verifiable
            or owner.boot_id is None
            or owner.process_group_id is None
        ):
            raise SupplementContradiction(
                "attempt owner lacks boot/process-group identity"
            )
        return owner

    def _owner_is_alive(self, expected: ProcessIdentity) -> bool:
        try:
            os.kill(expected.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise SupplementContradiction(
                "cannot prove whether an attempt owner is alive"
            ) from exc
        actual = self.ownership_probes.process_identity(expected.pid)
        return actual.same_process_as(expected)

    def _stage_attempt_path(self, key: str) -> Path:
        return self.layout.stage_attempts / f"{key}.json"

    def _stage_attempt_value(
        self,
        *,
        key: str,
        state: str,
        generation: int,
        owner: ProcessIdentity,
    ) -> Mapping[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "contract": STAGE_ATTEMPT_CONTRACT,
            "spec_identity": self.spec.identity,
            "key": key,
            "state": state,
            "generation": generation,
            "owner": owner.to_dict(),
        }
        value["attempt_sha256"] = canonical_sha256(value)
        return value

    def _load_stage_attempt(self, key: str) -> Mapping[str, Any] | None:
        path = self._stage_attempt_path(key)
        if not _lexists(path):
            return None
        try:
            value = _load_json_object(path, f"{key} attempt")
            payload = dict(value)
            supplied = payload.pop("attempt_sha256", None)
            owner = ProcessIdentity.from_dict(value["owner"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupplementContradiction(f"{key} attempt journal is invalid") from exc
        if (
            value.get("contract") != STAGE_ATTEMPT_CONTRACT
            or value.get("spec_identity") != self.spec.identity
            or value.get("key") != key
            or value.get("state") not in {"running", "failed", "complete", "recovered"}
            or type(value.get("generation")) is not int
            or value["generation"] < 1
            or supplied != canonical_sha256(payload)
            or not owner.is_verifiable
        ):
            raise SupplementContradiction(f"{key} attempt journal changed")
        return value

    def _reconcile_stage_temporary(
        self, *, key: str, temporary_paths: Sequence[Path]
    ) -> int:
        attempt = self._load_stage_attempt(key)
        present = [path for path in temporary_paths if _lexists(path)]
        generation = 0 if attempt is None else int(attempt["generation"])
        if not present:
            return generation
        if attempt is None:
            raise SupplementContradiction(f"{key} has unowned temporary artifacts")
        owner = ProcessIdentity.from_dict(attempt["owner"])
        if self._owner_is_alive(owner):
            raise SupplementBusy(f"{key} attempt owner is still alive")
        for path in present:
            if path.is_symlink():
                raise SupplementContradiction(f"{key} temporary path is a symlink")
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
            else:
                raise SupplementContradiction(f"{key} temporary artifact is nonregular")
        _atomic_replace_json(
            self._stage_attempt_path(key),
            self._stage_attempt_value(
                key=key,
                state="recovered",
                generation=generation,
                owner=owner,
            ),
        )
        return generation

    def _begin_stage_attempt(
        self, *, key: str, temporary_paths: Sequence[Path]
    ) -> tuple[int, ProcessIdentity]:
        generation = self._reconcile_stage_temporary(
            key=key, temporary_paths=temporary_paths
        )
        owner = self._current_owner()
        generation += 1
        _atomic_replace_json(
            self._stage_attempt_path(key),
            self._stage_attempt_value(
                key=key,
                state="running",
                generation=generation,
                owner=owner,
            ),
        )
        return generation, owner

    def _finish_stage_attempt(
        self,
        *,
        key: str,
        generation: int,
        owner: ProcessIdentity,
        state: str,
    ) -> None:
        _atomic_replace_json(
            self._stage_attempt_path(key),
            self._stage_attempt_value(
                key=key,
                state=state,
                generation=generation,
                owner=owner,
            ),
        )

    @property
    def _gpu_claim_path(self) -> Path:
        return self.spec.run_root / "evaluation" / ".curation-gpu-ownership.json"

    def _archive_released_claim(self, *, require_own: bool) -> None:
        claim_path = self._gpu_claim_path
        if not _lexists(claim_path):
            if require_own and not _lexists(self.layout.gpu_ownership_archive):
                raise SupplementContradiction(
                    "released GPU ownership evidence is missing"
                )
            return
        try:
            claim = _load_json_object(claim_path, "global GPU ownership claim")
        except ValueError as exc:
            raise SupplementContradiction(
                "global GPU ownership claim is invalid"
            ) from exc
        claim_payload = dict(claim)
        claim_identity = claim_payload.pop("claim_sha256", None)
        if claim.get(
            "contract"
        ) != GPU_OWNERSHIP_CONTRACT or claim_identity != canonical_sha256(
            claim_payload
        ):
            raise SupplementContradiction(
                "global GPU ownership claim self-hash is invalid"
            )
        if claim.get("state") != "released":
            return
        own = (
            isinstance(claim.get("spec"), Mapping)
            and claim["spec"].get("identity") == self.spec.identity
        )
        if require_own and not own:
            raise SupplementContradiction(
                "released GPU ownership belongs to another specification"
            )
        target = (
            self.layout.gpu_ownership_archive
            if own
            else self.spec.work_root
            / "prior-gpu-ownership"
            / f"{claim.get('claim_sha256', 'unknown')}.json"
        )
        _publish_immutable(target, claim_path.read_bytes())
        claim_path.unlink()
        _fsync_directory(claim_path.parent)

    def _gpu_released(self) -> bool:
        if not _lexists(self.layout.gpu_ownership_archive):
            return False
        try:
            claim = _load_json_object(
                self.layout.gpu_ownership_archive, "released GPU ownership"
            )
        except ValueError as exc:
            raise SupplementContradiction("released GPU ownership is invalid") from exc
        payload = dict(claim)
        supplied = payload.pop("claim_sha256", None)
        return (
            supplied == canonical_sha256(payload)
            and claim.get("contract") == GPU_OWNERSHIP_CONTRACT
            and claim.get("state") == "released"
            and isinstance(claim.get("spec"), Mapping)
            and claim["spec"].get("identity") == self.spec.identity
            and claim.get("topology") == _ownership_topology(self.spec)
        )

    def _next_stage(self, snapshot: SupplementSnapshot) -> str:
        action = plan_next_stage(snapshot)
        if (
            snapshot.deficits
            and snapshot.analysis_shards_complete == snapshot.analysis_shards_total
            and not self._gpu_released()
        ):
            return "release_gpu_ownership"
        return action

    @contextlib.contextmanager
    def _owned_gpu_stage(self, *, release_after: bool) -> Iterator[None]:
        with self.gpu_ownership.global_lock():
            self._archive_released_claim(require_own=False)
            self.gpu_ownership.acquire(poll_interval=0.1)
            completed = False
            try:
                yield
                completed = True
            finally:
                if release_after and completed:
                    self.gpu_ownership.release()
                    self._archive_released_claim(require_own=True)

    def _release_gpu_ownership(self) -> None:
        with self.gpu_ownership.global_lock():
            if not _lexists(self._gpu_claim_path):
                raise SupplementContradiction(
                    "GPU ownership claim disappeared before release"
                )
            self.gpu_ownership.acquire(poll_interval=0.1)
            self.gpu_ownership.release()
            self._archive_released_claim(require_own=True)

    def _validate_selfplay_receipt(
        self,
        root: Path,
        inventory: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        receipt_path = root / "receipt.json"
        try:
            receipt = _load_json_object(receipt_path, "self-play exit receipt")
            payload = dict(receipt)
            supplied = payload.pop("receipt_sha256", None)
            identity = ProcessIdentity.from_dict(receipt["process_identity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupplementContradiction("self-play exit receipt is invalid") from exc
        expected_keys = {
            "schema_version",
            "contract",
            "spec",
            "argv",
            "shell",
            "returncode",
            "process_identity",
            "attempt_generation",
            "katago_sha256",
            "model_sha256",
            "selfplay_config_sha256",
            "models_directory",
            "selfplay_gpus",
            "game_count",
            "output_directory",
            "outputs",
            "receipt_sha256",
        }
        if (
            set(receipt) != expected_keys
            or receipt.get("schema_version") != 1
            or receipt.get("contract") != SELFPLAY_RECEIPT_CONTRACT
            or supplied != canonical_sha256(payload)
            or receipt.get("spec")
            != {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            }
            or receipt.get("argv") != list(render_selfplay_argv(self.spec))
            or receipt.get("shell") is not False
            or receipt.get("returncode") != 0
            or type(receipt.get("attempt_generation")) is not int
            or receipt["attempt_generation"] < 1
            or identity.boot_id is None
            or identity.process_group_id is None
            or identity.command_sha256
            != _proc_argv_sha256(render_selfplay_argv(self.spec))
            or receipt.get("katago_sha256") != self.spec.katago.sha256
            or receipt.get("model_sha256") != self.spec.models["original"].sha256
            or receipt.get("selfplay_config_sha256") != self.spec.selfplay_config.sha256
            or receipt.get("models_directory")
            != {
                "path": str(self.spec.selfplay_models_directory.path),
                "sha256": self.spec.selfplay_models_directory.sha256,
            }
            or receipt.get("selfplay_gpus") != list(self.spec.topology.selfplay_gpus)
            or receipt.get("game_count") != self.spec.game_count
            or receipt.get("output_directory") != str(self.layout.selfplay_directory)
            or receipt.get("outputs") != list(inventory)
        ):
            raise SupplementContradiction(
                "self-play exit receipt contradicts command or outputs"
            )
        return receipt

    def _inspect_selfplay(self) -> bool:
        if not _lexists(self.layout.selfplay_directory):
            return False
        inventory, games = _inventory_selfplay(self.layout.selfplay_directory)
        if games != self.spec.game_count:
            raise SupplementContradiction(
                f"self-play contains {games} games, expected {self.spec.game_count}"
            )
        if not _lexists(self.layout.selfplay_receipt):
            return False
        self._validate_selfplay_receipt(self.layout.selfplay_directory, inventory)
        return True

    def _inspect_harvest_plan(self) -> bool:
        if not _lexists(self.layout.harvest_plan):
            return False
        try:
            actual = _validate_harvest_plan(self.layout.harvest_plan)
            expected = _expected_harvest_plan(self.spec)
        except (OSError, TypeError, ValueError) as exc:
            raise SupplementContradiction(f"harvest plan is invalid: {exc}") from exc
        if actual != expected:
            raise SupplementContradiction(
                "harvest plan contradicts the frozen self-play corpus"
            )
        return True

    def _inspect_harvest(self) -> bool:
        if not _lexists(self.layout.harvest_directory):
            return False
        receipt_path = self.layout.harvest_directory / "receipt.json"
        if not _lexists(receipt_path):
            raise SupplementContradiction("harvest directory has no receipt")
        try:
            receipt = _load_json_object(receipt_path, "harvest receipt")
            payload = dict(receipt)
            identity = payload.pop("manifest_sha256", None)
            if (
                receipt.get("schema_version") != 1
                or receipt.get("contract") != HARVEST_RECEIPT_CONTRACT
                or identity != canonical_sha256(payload)
                or receipt.get("plan_path") != str(self.layout.harvest_plan)
                or receipt.get("plan_sha256") != file_sha256(self.layout.harvest_plan)
                or receipt.get("output_dir") != str(self.layout.harvest_directory)
                or receipt.get("outputs")
                != _harvest_output_inventory(self.layout.harvest_directory)
            ):
                raise SupplementContradiction(
                    "harvest receipt contradicts its plan or outputs"
                )
            _harvest_position_sources(self.layout)
        except SupplementContradiction:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise SupplementContradiction(f"harvest output is invalid: {exc}") from exc
        return True

    def _inspect_normalized(self) -> bool:
        output_exists = _lexists(self.layout.normalized)
        manifest_exists = _lexists(self.layout.normalized_manifest)
        if not output_exists and not manifest_exists:
            return False
        if manifest_exists and not output_exists:
            raise SupplementContradiction(
                "normalized manifest exists without its output"
            )
        try:
            sources = _harvest_position_sources(self.layout)
            rows, rejected, base = _filter_normalized_positions(self.spec, sources)
            expected_data = _canonical_jsonl(rows)
            rejected_data = _canonical_jsonl(rejected)
            if self.layout.normalized.read_bytes() != expected_data:
                raise SupplementContradiction(
                    "normalized positions contradict harvested inputs"
                )
            if not manifest_exists:
                return False
            if (
                not _lexists(self.layout.rejected_duplicates)
                or not _lexists(self.layout.rejected_duplicates_manifest)
                or self.layout.rejected_duplicates.read_bytes() != rejected_data
            ):
                raise SupplementContradiction(
                    "normalized duplicate rejection provenance changed"
                )
            rejected_manifest: dict[str, Any] = {
                "schema_version": 1,
                "contract": REJECTED_DUPLICATES_CONTRACT,
                "primary_inventory": _artifact_binding(
                    self.spec.primary_prefilter_inventory.path
                ),
                "output": _artifact_binding(self.layout.rejected_duplicates),
                "row_count": len(rejected),
                "semantic_ids_sha256": canonical_sha256(
                    [row["semanticSha256"] for row in rejected]
                ),
            }
            rejected_manifest["manifest_sha256"] = canonical_sha256(rejected_manifest)
            if (
                _load_json_object(
                    self.layout.rejected_duplicates_manifest,
                    "normalized duplicate rejection manifest",
                )
                != rejected_manifest
            ):
                raise SupplementContradiction(
                    "normalized duplicate rejection manifest changed"
                )
            expected: dict[str, Any] = {
                **base,
                "row_count": len(rows),
                "semantic_hashes_sha256": canonical_sha256(
                    [row["semanticSha256"] for row in rows]
                ),
                "unfiltered_row_count": len(rows) + len(rejected),
                "duplicate_rejections": _artifact_binding(
                    self.layout.rejected_duplicates_manifest
                ),
                "output_path": str(self.layout.normalized),
                "output_sha256": hashlib.sha256(expected_data).hexdigest(),
            }
            expected_payload = dict(expected)
            expected_payload.pop("manifest_sha256", None)
            expected["manifest_sha256"] = canonical_sha256(expected_payload)
            actual = _load_json_object(
                self.layout.normalized_manifest, "normalized manifest"
            )
            if actual != expected or actual.get("contract") != CURATION_CONTRACT:
                raise SupplementContradiction(
                    "normalized positions contradict harvested inputs"
                )
        except SupplementContradiction:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise SupplementContradiction(
                f"normalized artifact is invalid: {exc}"
            ) from exc
        return True

    def _inspect_queries(self) -> bool:
        if not _lexists(self.layout.query_directory):
            return False
        if (
            self.layout.query_directory.is_symlink()
            or not self.layout.query_directory.is_dir()
        ):
            raise SupplementContradiction("query bundle root is unsafe")
        try:
            manifest = _load_json_object(
                self.layout.query_manifest, "prefilter query manifest"
            )
            positions = _normalized_positions(self.layout.normalized)
            ids = [row["semanticSha256"] for row in positions]
            query_entries: dict[str, Any] = {}
            expected_files = {self.layout.query_manifest.resolve()}
            for mode, powered in (("standard-2000", False), ("powered-2000", True)):
                query_path = self.layout.query_directory / "queries" / f"{mode}.jsonl"
                expected_files.add(query_path.resolve())
                expected_data = _canonical_jsonl(
                    build_analysis_query(
                        position,
                        query_id=position["semanticSha256"],
                        max_visits=2000,
                        powered=powered,
                    )
                    for position in positions
                )
                if query_path.is_symlink() or not query_path.is_file():
                    raise SupplementContradiction(f"missing prefilter query {mode}")
                if query_path.read_bytes() != expected_data:
                    raise SupplementContradiction(f"prefilter query {mode} changed")
                query_entries[mode] = {
                    "path": f"queries/{mode}.jsonl",
                    "sha256": file_sha256(query_path),
                    "row_count": len(positions),
                    "ids_sha256": canonical_sha256(ids),
                    "powered": powered,
                    "visits": 2000,
                }
            expected: dict[str, Any] = {
                "schema_version": 1,
                "contract": PREFILTER_QUERY_BUNDLE_CONTRACT,
                "source_path": str(self.layout.normalized),
                "source_sha256": file_sha256(self.layout.normalized),
                "position_count": len(positions),
                "semantic_ids_sha256": canonical_sha256(ids),
                "katago_path": str(self.spec.katago.path),
                "katago_sha256": self.spec.katago.sha256,
                "config_path": str(self.spec.analysis_config.path),
                "config_sha256": self.spec.analysis_config.sha256,
                "model_path": str(self.spec.models["champion"].path),
                "model_sha256": self.spec.models["champion"].sha256,
                "policy_path": str(self.spec.policy.path),
                "policy_sha256": self.spec.policy.sha256,
                "policy_hash": canonical_sha256(load_policy(self.spec.policy.path)),
                "queries": query_entries,
            }
            expected["manifest_sha256"] = canonical_sha256(expected)
            if manifest != expected:
                raise SupplementContradiction(
                    "prefilter query bundle uses different frozen inputs"
                )
            _assert_expected_files(
                self.layout.query_directory, expected_files, "prefilter query bundle"
            )
        except SupplementContradiction:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise SupplementContradiction(f"query bundle is invalid: {exc}") from exc
        return True

    def _inspect_splits(self) -> int:
        complete = 0
        for mode in ("standard-2000", "powered-2000"):
            root = self.layout.query_shards / mode
            if not _lexists(root):
                continue
            if (
                root.is_symlink()
                or not root.is_dir()
                or str(root.resolve()) != str(root)
            ):
                raise SupplementContradiction(f"{mode} query shard root is unsafe")
            query = self.layout.query_directory / "queries" / f"{mode}.jsonl"
            try:
                manifest, shard_specs = _load_query_shard_manifest(
                    root / "manifest.json", query
                )
            except (OSError, TypeError, ValueError) as exc:
                raise SupplementContradiction(
                    f"{mode} query shards are invalid: {exc}"
                ) from exc
            if (
                manifest.get("contract") != QUERY_SHARDS_CONTRACT
                or manifest.get("shard_count") != self.spec.topology.shards_per_role
            ):
                raise SupplementContradiction(
                    f"{mode} query shards contradict configured topology"
                )
            _assert_expected_files(
                root,
                [
                    root / "manifest.json",
                    *(root / spec["path"] for spec in shard_specs.values()),
                ],
                f"{mode} query shards",
            )
            complete += 1
        if _lexists(self.layout.query_shards):
            if (
                self.layout.query_shards.is_symlink()
                or not self.layout.query_shards.is_dir()
            ):
                raise SupplementContradiction("query shard root is unsafe")
            expected_roots = {
                (self.layout.query_shards / mode).resolve()
                for mode in ("standard-2000", "powered-2000")
            }
            for child in self.layout.query_shards.iterdir():
                if (
                    child.is_symlink()
                    or not child.is_dir()
                    or child.resolve() not in expected_roots
                ):
                    raise SupplementContradiction(
                        f"unexpected query shard artifact: {child}"
                    )
        return complete

    def _inspect_analysis_job(self, job: AnalysisJob) -> bool:
        output_exists = _lexists(job.output_path)
        manifest_exists = _lexists(job.manifest_path)
        if not output_exists and not manifest_exists:
            return False
        if manifest_exists and not output_exists:
            raise SupplementContradiction(
                "analysis manifest exists without output for "
                f"{job.role}/{job.shard_index}"
            )
        try:
            query_rows = _load_canonical_jsonl(
                job.query_path, f"{job.role} query shard"
            )
            expected_ids = sorted(row.get("id") for row in query_rows)
            output_rows = _load_canonical_jsonl(
                job.output_path, f"{job.role} analysis shard"
            )
            output_ids = [row.get("id") for row in output_rows]
            if (
                any(not isinstance(value, str) or not value for value in output_ids)
                or any("error" in row for row in output_rows)
                or output_ids != expected_ids
            ):
                raise SupplementContradiction(
                    f"analysis output IDs changed for {job.role}/{job.shard_index}"
                )
            if not manifest_exists:
                return False
            manifest = _load_json_object(
                job.manifest_path, f"{job.role} analysis shard manifest"
            )
            payload = dict(manifest)
            identity = payload.pop("manifest_sha256", None)
            expected_argv = [
                str(self.spec.katago.path),
                "analysis",
                "-config",
                str(self.spec.analysis_config.path),
                "-model",
                str(self.spec.models[job.model].path),
            ]
            expected_keys = {
                "schema_version",
                "contract",
                "argv",
                "katago_sha256",
                "config_sha256",
                "model_sha256",
                "cuda_visible_devices",
                "query_path",
                "query_sha256",
                "output_path",
                "output_sha256",
                "row_count",
                "manifest_sha256",
            }
            if (
                set(manifest) != expected_keys
                or manifest.get("schema_version") != 1
                or manifest.get("contract") != ANALYSIS_RUN_CONTRACT
                or identity != canonical_sha256(payload)
                or manifest.get("argv") != expected_argv
                or manifest.get("katago_sha256") != self.spec.katago.sha256
                or manifest.get("config_sha256") != self.spec.analysis_config.sha256
                or manifest.get("model_sha256") != self.spec.models[job.model].sha256
                or manifest.get("cuda_visible_devices") != job.gpu
                or manifest.get("query_path") != str(job.query_path)
                or manifest.get("query_sha256") != file_sha256(job.query_path)
                or manifest.get("output_path") != str(job.output_path)
                or manifest.get("output_sha256") != file_sha256(job.output_path)
                or manifest.get("row_count") != len(output_rows)
            ):
                raise SupplementContradiction(
                    f"analysis provenance changed for {job.role}/{job.shard_index}"
                )
        except SupplementContradiction:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise SupplementContradiction(
                f"analysis artifact is invalid for {job.role}/{job.shard_index}: {exc}"
            ) from exc
        return True

    def _inspect_analyses(self) -> int:
        jobs = plan_analysis_jobs(self.spec)
        allowed = [
            path for job in jobs for path in (job.output_path, job.manifest_path)
        ]
        for job in jobs:
            temporary = tuple(
                job.output_path.parent.glob(f".{job.output_path.name}.analysis-*")
            )
            if temporary:
                key = f"analysis-{job.model}-{job.mode}-{job.shard_index:03d}"
                if self._load_stage_attempt(key) is None:
                    raise SupplementContradiction(
                        f"{key} has unowned temporary artifacts"
                    )
                allowed.extend(temporary)
        _assert_expected_files(self.layout.analysis_shards, allowed, "analysis shard")
        return sum(self._inspect_analysis_job(job) for job in jobs)

    def _inspect_merged(self) -> int:
        allowed: list[Path] = []
        complete = 0
        for role in PREFILTER_ROLES:
            output = _analysis_output_path(self.layout, role)
            manifest = Path(str(output) + ".manifest.json")
            allowed.extend((output, manifest))
            output_exists = _lexists(output)
            manifest_exists = _lexists(manifest)
            if not output_exists and not manifest_exists:
                continue
            if manifest_exists and not output_exists:
                raise SupplementContradiction(
                    f"merged manifest exists without output for {role}"
                )
            mode = role.split("/", 1)[1]
            jobs = [job for job in plan_analysis_jobs(self.spec) if job.role == role]
            if not manifest_exists:
                try:
                    rows = _load_canonical_jsonl(output, f"{role} merged analysis")
                    query_rows = _load_canonical_jsonl(
                        self.layout.query_directory / "queries" / f"{mode}.jsonl",
                        f"{role} full query",
                    )
                except (OSError, TypeError, ValueError) as exc:
                    raise SupplementContradiction(
                        f"recoverable merged output is invalid for {role}: {exc}"
                    ) from exc
                if [row.get("id") for row in rows] != sorted(
                    row.get("id") for row in query_rows
                ):
                    raise SupplementContradiction(
                        f"recoverable merged output IDs changed for {role}"
                    )
                continue
            try:
                receipt = merge_analysis(
                    query_path=self.layout.query_directory
                    / "queries"
                    / f"{mode}.jsonl",
                    split_manifest_path=self.layout.query_shards
                    / mode
                    / "manifest.json",
                    shard_outputs=[job.output_path for job in jobs],
                    output=output,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise SupplementContradiction(
                    f"merged analysis is invalid for {role}: {exc}"
                ) from exc
            if receipt.get("output_path") != str(output) or receipt.get(
                "output_sha256"
            ) != file_sha256(output):
                raise SupplementContradiction(
                    f"merged analysis binding changed for {role}"
                )
            complete += 1
        _assert_expected_files(self.layout.analyses, allowed, "merged analysis")
        return complete

    def _inspect_prefilters(
        self, deficits: Mapping[str, int]
    ) -> tuple[Mapping[str, bool], Mapping[str, int]]:
        allowed: list[Path] = []
        completion = {label: label not in deficits for label in LEAD_LABELS}
        counts = {label: 0 for label in LEAD_LABELS}
        limits = generation_limits(self.spec, deficits)
        for label in LEAD_LABELS:
            output, manifest_path = _selected_paths(self.layout, label)
            if label not in deficits:
                if _lexists(output) or _lexists(manifest_path):
                    raise SupplementContradiction(
                        f"prefilter artifacts exist for non-deficit label {label}"
                    )
                continue
            allowed.extend((output, manifest_path))
            output_exists = _lexists(output)
            manifest_exists = _lexists(manifest_path)
            if not output_exists and not manifest_exists:
                continue
            if manifest_exists and not output_exists:
                raise SupplementContradiction(
                    f"prefilter manifest exists without output for {label}"
                )
            if not manifest_exists:
                try:
                    rows = _load_canonical_jsonl(
                        output, f"{label} recoverable prefilter", allow_empty=True
                    )
                except (OSError, TypeError, ValueError) as exc:
                    raise SupplementContradiction(
                        f"recoverable {label} prefilter output is invalid: {exc}"
                    ) from exc
                if len(rows) > limits[label]:
                    raise SupplementContradiction(
                        f"recoverable {label} prefilter exceeds its limit"
                    )
                continue
            try:
                actual = validate_prefilter_artifact(
                    manifest_path,
                    expected_label=label,
                    expected_model_hashes={
                        model: self.spec.models[model].sha256
                        for model in ("original", "champion")
                    },
                )
                rows = _load_canonical_jsonl(
                    output, f"{label} prefilter output", allow_empty=True
                )
            except (OSError, TypeError, ValueError) as exc:
                raise SupplementContradiction(
                    f"{label} prefilter artifact is invalid: {exc}"
                ) from exc
            if (
                actual.get("label") != label
                or actual.get("limit") != limits[label]
                or actual.get("selected", {}).get("row_count") != len(rows)
                or len(rows) > limits[label]
            ):
                raise SupplementContradiction(
                    f"{label} prefilter limit or inventory changed"
                )
            completion[label] = True
            counts[label] = len(rows)
        _assert_expected_files(self.layout.selected, allowed, "selected prefilter")
        return completion, counts

    def _inspect_summary(
        self,
        counts: Mapping[str, int],
        deficits: Mapping[str, int],
        supplemental: Mapping[str, int],
    ) -> tuple[bool, str | None]:
        if not _lexists(self.layout.summary):
            return False, None
        basis = (
            counts
            if self.spec.downstream_accepted_counts is None
            else self.spec.downstream_accepted_counts
        )
        final_counts = {
            label: basis[label] + supplemental.get(label, 0) for label in LEAD_LABELS
        }
        if not deficits:
            state = "noop"
        elif all(
            final_counts[label] >= self.spec.target_counts[label]
            for label in LEAD_LABELS
        ):
            state = "complete"
        else:
            state = "insufficient_candidates"
        expected = _summary_value(
            self.spec,
            counts=counts,
            deficits=deficits,
            supplemental=supplemental,
            state=state,
        )
        try:
            actual = _load_json_object(self.layout.summary, "supplement summary")
        except ValueError as exc:
            raise SupplementContradiction(
                f"supplement summary is invalid: {exc}"
            ) from exc
        if actual != expected:
            raise SupplementContradiction(
                "supplement summary contradicts immutable artifacts"
            )
        return True, state

    def _snapshot(self) -> SupplementSnapshot:
        self._assert_frozen_inputs()
        counts = primary_counts(self.spec)
        deficits = source_deficits(
            deficit_basis_counts(self.spec), self.spec.target_counts
        )
        total_jobs = len(plan_analysis_jobs(self.spec))
        empty_completion = {label: label not in deficits for label in LEAD_LABELS}
        empty_counts = {label: 0 for label in LEAD_LABELS}

        operational = (
            self.layout.selfplay_directory,
            self.layout.harvest_plan,
            self.layout.harvest_directory,
            self.layout.normalized,
            self.layout.normalized_manifest,
            self.layout.query_directory,
            self.layout.query_shards,
            self.layout.analysis_shards,
            self.layout.analyses,
            self.layout.selected,
        )
        if not deficits:
            self._assert_absent(
                operational
                + (
                    self.layout.selfplay_attempt_root,
                    self.layout.rejected_duplicates,
                    self.layout.rejected_duplicates_manifest,
                    self.layout.gpu_ownership_archive,
                    self.layout.stage_attempts,
                ),
                "supplement artifacts exist although primary reserves are sufficient",
            )
            summary_complete, summary_state = self._inspect_summary(
                counts, deficits, empty_counts
            )
            return SupplementSnapshot(
                counts,
                deficits,
                False,
                False,
                False,
                False,
                False,
                0,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                summary_complete,
                summary_state,
            )

        selfplay = self._inspect_selfplay()
        if not selfplay:
            self._assert_absent(
                operational[1:] + (self.layout.summary,),
                "downstream artifact exists before complete self-play",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                False,
                False,
                False,
                False,
                False,
                0,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        harvest_plan = self._inspect_harvest_plan()
        if not harvest_plan:
            self._assert_absent(
                operational[2:] + (self.layout.summary,),
                "downstream artifact exists before harvest plan",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                False,
                False,
                False,
                False,
                0,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        harvest = self._inspect_harvest()
        if not harvest:
            self._assert_absent(
                operational[3:] + (self.layout.summary,),
                "downstream artifact exists before harvest completion",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                False,
                False,
                False,
                0,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        normalized = self._inspect_normalized()
        if not normalized:
            self._assert_absent(
                operational[5:] + (self.layout.summary,),
                "downstream artifact exists before normalization",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                True,
                False,
                False,
                0,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        queries = self._inspect_queries()
        if not queries:
            self._assert_absent(
                operational[6:] + (self.layout.summary,),
                "downstream artifact exists before query generation",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                True,
                True,
                False,
                0,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        splits = self._inspect_splits()
        if splits < 2:
            self._assert_absent(
                operational[7:] + (self.layout.summary,),
                "downstream artifact exists before query sharding",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                True,
                True,
                True,
                splits,
                0,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        analyses = self._inspect_analyses()
        if analyses < total_jobs:
            self._assert_absent(
                operational[8:] + (self.layout.summary,),
                "downstream artifact exists before all analysis shards",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                True,
                True,
                True,
                splits,
                analyses,
                total_jobs,
                0,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        merged = self._inspect_merged()
        if merged < len(PREFILTER_ROLES):
            self._assert_absent(
                (self.layout.selected, self.layout.summary),
                "prefilter artifact exists before merged analyses",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                True,
                True,
                True,
                splits,
                analyses,
                total_jobs,
                merged,
                empty_completion,
                empty_counts,
                False,
                None,
            )

        prefilters, supplemental = self._inspect_prefilters(deficits)
        if any(not prefilters[label] for label in deficits):
            self._assert_absent(
                (self.layout.summary,),
                "summary exists before all required prefilters",
            )
            return SupplementSnapshot(
                counts,
                deficits,
                True,
                True,
                True,
                True,
                True,
                splits,
                analyses,
                total_jobs,
                merged,
                prefilters,
                supplemental,
                False,
                None,
            )

        summary_complete, summary_state = self._inspect_summary(
            counts, deficits, supplemental
        )
        self._assert_frozen_inputs()
        return SupplementSnapshot(
            counts,
            deficits,
            True,
            True,
            True,
            True,
            True,
            splits,
            analyses,
            total_jobs,
            merged,
            prefilters,
            supplemental,
            summary_complete,
            summary_state,
        )

    def _run_selfplay(self) -> None:
        if _lexists(self.layout.selfplay_directory):
            if not self._inspect_selfplay():
                raise SupplementContradiction(
                    "published self-play corpus lacks a valid exit receipt"
                )
            return
        self.layout.selfplay_attempt_root.mkdir(parents=True, exist_ok=True)
        generation = 0
        journal: Mapping[str, Any] | None = None
        if _lexists(self.layout.selfplay_attempt_journal):
            journal = _load_json_object(
                self.layout.selfplay_attempt_journal, "self-play attempt journal"
            )
            payload = dict(journal)
            supplied = payload.pop("attempt_sha256", None)
            try:
                prior_identity = ProcessIdentity.from_dict(journal["process_identity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SupplementContradiction(
                    "self-play attempt identity is invalid"
                ) from exc
            if (
                journal.get("contract") != SELFPLAY_ATTEMPT_CONTRACT
                or journal.get("spec_identity") != self.spec.identity
                or journal.get("argv") != list(render_selfplay_argv(self.spec))
                or journal.get("output_directory")
                != str(self.layout.selfplay_attempt_output)
                or journal.get("state")
                not in {
                    "launching",
                    "running",
                    "failed",
                    "receipt_published",
                    "published",
                }
                or type(journal.get("generation")) is not int
                or journal["generation"] < 1
                or supplied != canonical_sha256(payload)
            ):
                raise SupplementContradiction("self-play attempt journal changed")
            generation = journal["generation"]
            if journal["state"] in {"launching", "running"} and self._owner_is_alive(
                prior_identity
            ):
                raise SupplementBusy("finite self-play attempt remains alive")
            if journal["state"] == "receipt_published" and _lexists(
                self.layout.selfplay_attempt_output
            ):
                inventory, games = _inventory_selfplay(
                    self.layout.selfplay_attempt_output
                )
                if games != self.spec.game_count:
                    raise SupplementContradiction(
                        "receipt-published self-play attempt is incomplete"
                    )
                self._validate_selfplay_receipt(
                    self.layout.selfplay_attempt_output, inventory
                )
                os.replace(
                    self.layout.selfplay_attempt_output,
                    self.layout.selfplay_directory,
                )
                _fsync_directory(self.layout.selfplay_directory.parent)
                return
            if _lexists(self.layout.selfplay_attempt_output):
                if (
                    self.layout.selfplay_attempt_output.is_symlink()
                    or not self.layout.selfplay_attempt_output.is_dir()
                ):
                    raise SupplementContradiction("self-play attempt output is unsafe")
                self.layout.selfplay_orphans.mkdir(parents=True, exist_ok=True)
                orphan = self.layout.selfplay_orphans / f"generation-{generation:06d}"
                if _lexists(orphan):
                    raise SupplementContradiction(
                        "self-play orphan generation already exists"
                    )
                os.replace(self.layout.selfplay_attempt_output, orphan)
                _fsync_directory(orphan.parent)

        generation += 1
        argv = render_selfplay_argv(self.spec)

        def journal_value(
            state: str,
            process_identity: ProcessIdentity,
            returncode: int | None,
        ) -> Mapping[str, Any]:
            value: dict[str, Any] = {
                "schema_version": 1,
                "contract": SELFPLAY_ATTEMPT_CONTRACT,
                "spec_identity": self.spec.identity,
                "generation": generation,
                "state": state,
                "argv": list(argv),
                "output_directory": str(self.layout.selfplay_attempt_output),
                "process_identity": process_identity.to_dict(),
                "returncode": returncode,
            }
            value["attempt_sha256"] = canonical_sha256(value)
            return value

        launching_owner = self._current_owner()
        _atomic_replace_json(
            self.layout.selfplay_attempt_journal,
            journal_value("launching", launching_owner, None),
        )
        process = self.runners.launcher(
            argv,
            shell=False,
            start_new_session=False,
        )
        pid = getattr(process, "pid", None)
        if type(pid) is not int or pid <= 0:
            raise SupplementError("self-play launcher did not expose a valid PID")
        identity = self.ownership_probes.process_identity(pid)
        if (
            not isinstance(identity, ProcessIdentity)
            or identity.boot_id is None
            or identity.process_group_id is None
            or identity.command_sha256 != _proc_argv_sha256(argv)
        ):
            raise SupplementContradiction(
                "self-play child identity does not bind the reviewed command"
            )

        _atomic_replace_json(
            self.layout.selfplay_attempt_journal,
            journal_value("running", identity, None),
        )
        returncode = process.wait()
        if returncode != 0:
            _atomic_replace_json(
                self.layout.selfplay_attempt_journal,
                journal_value("failed", identity, returncode),
            )
            raise SupplementError(f"finite self-play failed with status {returncode!r}")
        inventory, games = _inventory_selfplay(self.layout.selfplay_attempt_output)
        if games != self.spec.game_count:
            _atomic_replace_json(
                self.layout.selfplay_attempt_journal,
                journal_value("failed", identity, returncode),
            )
            raise SupplementError(
                f"finite self-play produced {games} games, "
                f"expected {self.spec.game_count}"
            )
        self._assert_frozen_inputs()
        _publish_immutable_json(
            self.layout.selfplay_attempt_output / "receipt.json",
            _selfplay_receipt_value(
                self.spec,
                inventory,
                process_identity=identity.to_dict(),
                attempt_generation=generation,
            ),
        )
        _atomic_replace_json(
            self.layout.selfplay_attempt_journal,
            journal_value("receipt_published", identity, 0),
        )
        os.replace(
            self.layout.selfplay_attempt_output,
            self.layout.selfplay_directory,
        )
        _fsync_directory(self.layout.selfplay_directory.parent)
        _atomic_replace_json(
            self.layout.selfplay_attempt_journal,
            journal_value("published", identity, 0),
        )

    def _create_harvest_plan(self) -> None:
        sgfs_dirs, sgf_dirs = _harvest_source_directories(self.layout)
        self.runners.harvest_plan(
            katago=self.spec.katago.path,
            sgfs_dirs=sgfs_dirs,
            sgf_dirs=sgf_dirs,
            training_input_roots=[self.spec.training_input_root],
            output_dir=self.layout.harvest_directory,
            manifest_path=self.layout.harvest_plan,
            threads=1,
        )

    def _execute_harvest(self) -> None:
        key = "harvest"
        temporary = tuple(
            self.layout.harvest_directory.parent.glob(
                f".{self.layout.harvest_directory.name}.harvest-*"
            )
        )
        generation, owner = self._begin_stage_attempt(
            key=key, temporary_paths=temporary
        )
        try:
            self.runners.harvest(
                self.layout.harvest_plan, subprocess_runner=self.runners.process
            )
        except BaseException:
            self._finish_stage_attempt(
                key=key, generation=generation, owner=owner, state="failed"
            )
            raise
        self._finish_stage_attempt(
            key=key, generation=generation, owner=owner, state="complete"
        )

    def _normalize(self) -> None:
        sources = _harvest_position_sources(self.layout)
        accepted, rejected, base = _filter_normalized_positions(self.spec, sources)
        accepted_data = _canonical_jsonl(accepted)
        rejected_data = _canonical_jsonl(rejected)
        _publish_immutable(self.layout.normalized, accepted_data)
        _publish_immutable(self.layout.rejected_duplicates, rejected_data)
        rejected_manifest: dict[str, Any] = {
            "schema_version": 1,
            "contract": REJECTED_DUPLICATES_CONTRACT,
            "primary_inventory": _artifact_binding(
                self.spec.primary_prefilter_inventory.path
            ),
            "output": _artifact_binding(self.layout.rejected_duplicates),
            "row_count": len(rejected),
            "semantic_ids_sha256": canonical_sha256(
                [row["semanticSha256"] for row in rejected]
            ),
        }
        rejected_manifest["manifest_sha256"] = canonical_sha256(rejected_manifest)
        _publish_immutable_json(
            self.layout.rejected_duplicates_manifest, rejected_manifest
        )
        manifest = {
            **base,
            "row_count": len(accepted),
            "semantic_hashes_sha256": canonical_sha256(
                [row["semanticSha256"] for row in accepted]
            ),
            "unfiltered_row_count": len(accepted) + len(rejected),
            "duplicate_rejections": _artifact_binding(
                self.layout.rejected_duplicates_manifest
            ),
            "output_path": str(self.layout.normalized),
            "output_sha256": hashlib.sha256(accepted_data).hexdigest(),
        }
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _publish_immutable_json(self.layout.normalized_manifest, manifest)

    def _generate_queries(self) -> None:
        self.runners.queries(
            normalized_path=self.layout.normalized,
            output_dir=self.layout.query_directory,
            katago_path=self.spec.katago.path,
            config_path=self.spec.analysis_config.path,
            model_path=self.spec.models["champion"].path,
            policy_path=self.spec.policy.path,
        )

    def _split_queries(self) -> None:
        for mode in ("standard-2000", "powered-2000"):
            output = self.layout.query_shards / mode
            if _lexists(output):
                continue
            self.runners.split(
                self.layout.query_directory / "queries" / f"{mode}.jsonl",
                output,
                shard_count=self.spec.topology.shards_per_role,
            )

    def _execute_analysis_job(self, job: AnalysisJob) -> Mapping[str, Any]:
        if self._inspect_analysis_job(job):
            return {"reused": True}
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = job.gpu
        key = f"analysis-{job.model}-{job.mode}-{job.shard_index:03d}"
        temporary = tuple(
            job.output_path.parent.glob(f".{job.output_path.name}.analysis-*")
        )
        generation, owner = self._begin_stage_attempt(
            key=key, temporary_paths=temporary
        )
        try:
            result = self.runners.analysis(
                katago=self.spec.katago.path,
                config=self.spec.analysis_config.path,
                model=self.spec.models[job.model].path,
                queries=job.query_path,
                output=job.output_path,
                env=environment,
                subprocess_runner=self.runners.process,
            )
        except BaseException:
            self._finish_stage_attempt(
                key=key, generation=generation, owner=owner, state="failed"
            )
            raise
        self._finish_stage_attempt(
            key=key, generation=generation, owner=owner, state="complete"
        )
        if not self._inspect_analysis_job(job):
            raise SupplementError(
                f"analysis runner did not publish {job.role}/{job.shard_index}"
            )
        return result

    def _run_analyses(self) -> None:
        pending = [
            job
            for job in plan_analysis_jobs(self.spec)
            if not self._inspect_analysis_job(job)
        ]
        executors = {
            gpu: concurrent.futures.ThreadPoolExecutor(
                max_workers=self.spec.topology.per_gpu_parallelism,
                thread_name_prefix=f"supplement-gpu-{gpu}",
            )
            for gpu in self.spec.topology.gpus
        }
        futures: dict[concurrent.futures.Future[Mapping[str, Any]], AnalysisJob] = {}
        failures: list[tuple[AnalysisJob, Exception]] = []
        try:
            for job in pending:
                futures[executors[job.gpu].submit(self._execute_analysis_job, job)] = (
                    job
                )
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    failures.append((job, exc))
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True)
        if failures:
            failures.sort(key=lambda item: item[0].key)
            details = "; ".join(
                f"{job.role}/{job.shard_index}: {type(exc).__name__}: {exc}"
                for job, exc in failures
            )
            raise SupplementError(f"one or more analysis shards failed: {details}")

    def _merge_analyses(self) -> None:
        for role in PREFILTER_ROLES:
            output = _analysis_output_path(self.layout, role)
            if _lexists(output) and _lexists(Path(str(output) + ".manifest.json")):
                continue
            mode = role.split("/", 1)[1]
            jobs = [job for job in plan_analysis_jobs(self.spec) if job.role == role]
            self.runners.merge(
                query_path=self.layout.query_directory / "queries" / f"{mode}.jsonl",
                split_manifest_path=self.layout.query_shards / mode / "manifest.json",
                shard_outputs=[job.output_path for job in jobs],
                output=output,
            )

    def _run_prefilters(self, deficits: Mapping[str, int]) -> None:
        analyses = {
            role: _analysis_output_path(self.layout, role) for role in PREFILTER_ROLES
        }
        limits = generation_limits(self.spec, deficits)
        for label in LEAD_LABELS:
            if label not in deficits:
                continue
            output, manifest = _selected_paths(self.layout, label)
            if _lexists(output) and _lexists(manifest):
                continue
            self.runners.prefilter(
                normalized_path=self.layout.normalized,
                analysis_paths=analyses,
                label=label,
                output_path=output,
                manifest_path=manifest,
                maximum_score_spread=PREFILTER_MAXIMUM_SCORE_SPREAD,
                threshold_buffer=PREFILTER_THRESHOLD_BUFFER,
                limit=limits[label],
                allow_empty=True,
            )

    def _publish_summary(self, snapshot: SupplementSnapshot) -> None:
        basis = (
            snapshot.primary_counts
            if self.spec.downstream_accepted_counts is None
            else self.spec.downstream_accepted_counts
        )
        final_counts = {
            label: basis[label] + snapshot.supplemental_counts.get(label, 0)
            for label in LEAD_LABELS
        }
        if not snapshot.deficits:
            state = "noop"
        elif all(
            final_counts[label] >= self.spec.target_counts[label]
            for label in LEAD_LABELS
        ):
            state = "complete"
        else:
            state = "insufficient_candidates"
        value = _summary_value(
            self.spec,
            counts=snapshot.primary_counts,
            deficits=snapshot.deficits,
            supplemental=snapshot.supplemental_counts,
            state=state,
        )
        _publish_immutable_json(self.layout.summary, value)

    def _dispatch(self, action: str, snapshot: SupplementSnapshot) -> None:
        if action == "run_selfplay":
            with self._owned_gpu_stage(release_after=False):
                self._run_selfplay()
            return
        if action == "create_harvest_plan":
            self._create_harvest_plan()
            return
        if action == "execute_harvest":
            self._execute_harvest()
            return
        if action == "normalize":
            self._normalize()
            return
        if action == "generate_queries":
            self._generate_queries()
            return
        if action == "split_queries":
            self._split_queries()
            return
        if action == "run_analyses":
            with self._owned_gpu_stage(release_after=False):
                self._run_analyses()
            return
        if action == "release_gpu_ownership":
            self._release_gpu_ownership()
            return
        if action == "merge_analyses":
            self._merge_analyses()
            return
        if action == "run_prefilters":
            self._run_prefilters(snapshot.deficits)
            return
        if action == "publish_summary":
            self._publish_summary(snapshot)
            return
        raise SupplementError(f"stage {action!r} is not executable")

    def _build_status(
        self,
        snapshot: SupplementSnapshot,
        *,
        mode: str,
        state: str | None = None,
        error: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        action = self._next_stage(snapshot)
        if snapshot.summary_complete:
            resolved_state = snapshot.summary_state or "complete"
        else:
            resolved_state = state or action
        status: dict[str, Any] = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "contract": STATUS_CONTRACT,
            "spec": {
                "path": str(self.spec.path),
                "sha256": self.spec.file_sha256,
                "identity": self.spec.identity,
            },
            "mode": mode,
            "state": resolved_state,
            "next_stage": None if action == "complete" else action,
            "work_root": str(self.spec.work_root),
            "primary_counts": dict(snapshot.primary_counts),
            "target_counts": dict(self.spec.target_counts),
            "deficits": dict(snapshot.deficits),
            "supplemental_counts": dict(snapshot.supplemental_counts),
            "selfplay_command": plan_selfplay_command(self.spec),
            "topology": {
                "shards_per_role": self.spec.topology.shards_per_role,
                "gpus": list(self.spec.topology.gpus),
                "selfplay_gpus": list(self.spec.topology.selfplay_gpus),
                "per_gpu_parallelism": self.spec.topology.per_gpu_parallelism,
            },
            "progress": {
                "selfplay": snapshot.selfplay_complete,
                "harvest_plan": snapshot.harvest_plan_complete,
                "harvest": snapshot.harvest_complete,
                "normalized": snapshot.normalized_complete,
                "queries": snapshot.queries_complete,
                "split_roles_complete": snapshot.split_roles_complete,
                "split_roles_total": 2,
                "analysis_shards_complete": snapshot.analysis_shards_complete,
                "analysis_shards_total": snapshot.analysis_shards_total,
                "merged_roles_complete": snapshot.merged_roles_complete,
                "merged_roles_total": len(PREFILTER_ROLES),
                "prefilters": dict(snapshot.prefilters_complete),
                "summary": snapshot.summary_complete,
            },
            "artifacts": {
                "selfplay_receipt": str(self.layout.selfplay_receipt),
                "harvest_plan": str(self.layout.harvest_plan),
                "harvest_receipt": str(self.layout.harvest_directory / "receipt.json"),
                "normalized": str(self.layout.normalized),
                "query_manifest": str(self.layout.query_manifest),
                "summary": str(self.layout.summary),
            },
            "error": None if error is None else dict(error),
        }
        status["status_sha256"] = canonical_sha256(status)
        return status

    def _persist(
        self,
        snapshot: SupplementSnapshot,
        *,
        mode: str,
        state: str | None = None,
        error: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        status = self._build_status(snapshot, mode=mode, state=state, error=error)
        _atomic_replace_json(self.layout.status, status)
        return status

    def status(self) -> Mapping[str, Any]:
        """Inspect without creating the work root, lock, or status."""

        self._validate_existing_status()
        return self._build_status(self._snapshot(), mode="status")

    def _run_locked(self, *, mode: str) -> Mapping[str, Any]:
        self._validate_existing_status()
        while True:
            snapshot = self._snapshot()
            action = self._next_stage(snapshot)
            if action == "complete":
                return self._persist(snapshot, mode=mode)
            self._persist(snapshot, mode=mode, state=f"running_{action}")
            try:
                self._dispatch(action, snapshot)
                next_snapshot = self._snapshot()
            except BaseException as exc:
                failed_snapshot = snapshot
                with contextlib.suppress(Exception):
                    failed_snapshot = self._snapshot()
                with contextlib.suppress(Exception):
                    self._persist(
                        failed_snapshot,
                        mode=mode,
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
            next_action = self._next_stage(next_snapshot)
            if next_action == action:
                error = SupplementError(
                    f"stage runner returned without publishing {action}"
                )
                self._persist(
                    next_snapshot,
                    mode=mode,
                    state="failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
                raise error
            status = self._persist(next_snapshot, mode=mode)
            if mode == "once":
                return status

    def once(self) -> Mapping[str, Any]:
        """Advance exactly one restartable stage."""

        with self._exclusive_lock():
            return self._run_locked(mode="once")

    def watch(self, *, poll_interval: float = 30.0) -> Mapping[str, Any]:
        """Run all finite stages to a terminal summary."""

        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(float(poll_interval))
            or poll_interval < 0
        ):
            raise ValueError("poll_interval must be finite and nonnegative")
        # Every child command is finite and synchronous.  The argument is kept
        # for a uniform status/once/watch CLI and future external-stage polling.
        with self._exclusive_lock():
            return self._run_locked(mode="watch")


SupplementCoordinator = CurationSupplement
LeadSourceCoordinator = CurationSupplement
CurationSupplementCoordinator = CurationSupplement


def coordinate(
    *,
    mode: str,
    spec_path: Path,
    runners: SupplementRunners = DEFAULT_RUNNERS,
    poll_interval: float = 30.0,
    process_runner: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    coordinator = CurationSupplement(
        spec_path, runners=runners, process_runner=process_runner
    )
    if mode == "status":
        return coordinator.status()
    if mode == "once":
        return coordinator.once()
    if mode == "watch":
        return coordinator.watch(poll_interval=poll_interval)
    raise ValueError(f"unsupported supplement mode: {mode}")


run_supplement = coordinate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "once", "watch"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runners: SupplementRunners = DEFAULT_RUNNERS,
    process_runner: Callable[..., Any] | None = None,
) -> int:
    args = parse_args(argv)
    try:
        status = coordinate(
            mode=args.mode,
            spec_path=args.spec,
            runners=runners,
            poll_interval=args.poll_interval,
            process_runner=process_runner,
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
    return 0 if status["state"] in {"complete", "noop"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
