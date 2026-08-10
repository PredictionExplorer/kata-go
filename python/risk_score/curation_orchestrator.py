#!/usr/bin/env python3
"""Restartable GPU orchestration for machine-consensus position analysis.

The query bundle, query shards, analysis outputs, and their manifests are
immutable.  Only ``status.json`` is replaced in place, using canonical JSON and
an fsynced atomic rename.  On every pass, artifact state is reconstructed from
the files rather than trusted from the previous status snapshot.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production curation runs on Unix.
    fcntl = None  # type: ignore[assignment]

from risk_score.curate_position_bank import (
    ANALYSIS_RUN_CONTRACT,
    CONSENSUS_QUERY_BUNDLE_CONTRACT,
    _validate_consensus_query_bundle,
    merge_analysis,
    run_analysis,
    split_queries,
)
from risk_score.position_samples import canonical_json, canonical_sha256, file_sha256

STATUS_CONTRACT = "risk-score-machine-consensus-curation-status-v1"
STATUS_SCHEMA_VERSION = 1
EXPECTED_ROLE_COUNT = 8


class CurationContradiction(ValueError):
    """An existing artifact contradicts the frozen orchestration plan."""


@dataclass(frozen=True)
class ShardJob:
    role: str
    model: str
    model_path: Path
    model_sha256: str
    index: int
    query_path: Path
    query_sha256: str
    role_query_sha256: str
    query_ids: Tuple[str, ...]
    row_count: int
    split_manifest_path: Path
    output_path: Path

    @property
    def key(self) -> Tuple[str, int]:
        return (self.role, self.index)

    @property
    def manifest_path(self) -> Path:
        return Path(str(self.output_path) + ".manifest.json")


@dataclass(frozen=True)
class RolePlan:
    role: str
    model: str
    model_path: Path
    model_sha256: str
    query_path: Path
    query_sha256: str
    split_manifest_path: Path
    split_manifest_sha256: str
    split_manifest_identity: str
    jobs: Tuple[ShardJob, ...]
    merged_output_path: Path


@dataclass(frozen=True)
class PreparedPlan:
    context: Mapping[str, Any]
    manifest_path: Path
    manifest_file_sha256: str
    manifest_identity: str
    work_dir: Path
    roles: Mapping[str, RolePlan]

    @property
    def jobs(self) -> Tuple[ShardJob, ...]:
        return tuple(
            job for role in sorted(self.roles) for job in self.roles[role].jobs
        )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _load_canonical_json(path: Path, role: str) -> Dict[str, Any]:
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
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise ValueError(f"{role} must be canonical newline-terminated JSON")
    return value


def _load_canonical_jsonl(path: Path, role: str) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    try:
        data = source.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot load {role} {source}: {exc}") from exc
    rows: List[Dict[str, Any]] = []
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
    if not rows or data != expected:
        raise ValueError(f"{role} must be nonempty canonical JSONL")
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise CurationContradiction(
            f"status parent must be a regular non-symlink directory: {target.parent}"
        )
    if _lexists(target) and (target.is_symlink() or not target.is_file()):
        raise CurationContradiction(
            f"status path must be a regular non-symlink file: {target}"
        )
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _ensure_directory_beneath(path: Path, root: Path, role: str) -> None:
    root = root.resolve()
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise CurationContradiction(f"{role} escapes the work directory") from exc
    current = root
    for part in parts:
        current = current / part
        if _lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise CurationContradiction(
                    f"{role} contains an unsafe path component: {current}"
                )
            continue
        current.mkdir()


def _normalize_gpus(values: Sequence[Any]) -> Tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    gpus: List[str] = []
    for raw_value in values:
        for item in str(raw_value).split(","):
            gpu = item.strip()
            if not gpu:
                raise ValueError("GPU identifiers must be nonempty")
            if any(character.isspace() for character in gpu) or "," in gpu:
                raise ValueError(f"invalid GPU identifier: {gpu!r}")
            if gpu in gpus:
                raise ValueError(f"GPU identifier was supplied more than once: {gpu}")
            gpus.append(gpu)
    if not gpus:
        raise ValueError("at least one explicit GPU identifier is required")
    return tuple(gpus)


def _round_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


class CurationOrchestrator:
    """Reconcile and execute the eight roles in a consensus query bundle."""

    def __init__(
        self,
        *,
        query_manifest_path: Path,
        work_dir: Path,
        shard_count: int,
        gpus: Sequence[Any],
        per_gpu_parallelism: int = 1,
        analysis_runner: Callable[..., Mapping[str, Any]] = run_analysis,
        subprocess_runner: Callable[..., Any] = subprocess.run,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        if type(shard_count) is not int or not 1 <= shard_count <= 64:
            raise ValueError("shards per role must be between 1 and 64")
        if type(per_gpu_parallelism) is not int or not 1 <= per_gpu_parallelism <= 64:
            raise ValueError("per-GPU parallelism must be between 1 and 64")
        if not callable(analysis_runner):
            raise ValueError("analysis runner must be callable")
        if not callable(subprocess_runner):
            raise ValueError("subprocess runner must be callable")
        self.query_manifest_path = Path(query_manifest_path)
        self.requested_work_dir = Path(work_dir)
        self.shard_count = shard_count
        self.gpus = _normalize_gpus(gpus)
        self.per_gpu_parallelism = per_gpu_parallelism
        self.analysis_runner = analysis_runner
        self.subprocess_runner = subprocess_runner
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.progress_callback = progress_callback
        self._started_at: Optional[float] = None
        self._history: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _load_context(self) -> Tuple[Path, Mapping[str, Any]]:
        requested_manifest = self.query_manifest_path.expanduser()
        if requested_manifest.is_symlink():
            raise ValueError("consensus query bundle manifest must not be a symlink")
        manifest_path = requested_manifest.resolve()
        raw_manifest = _load_canonical_json(
            manifest_path, "consensus query bundle manifest"
        )
        if raw_manifest.get("contract") != CONSENSUS_QUERY_BUNDLE_CONTRACT:
            raise ValueError(
                "query manifest is not a "
                "risk-score-position-analysis-query-bundle-v2 bundle"
            )
        normalized_value = raw_manifest.get("normalized_path")
        if not isinstance(normalized_value, str) or not normalized_value:
            raise ValueError("consensus query manifest lacks normalized_path")
        normalized_path = Path(normalized_value)
        if (
            not normalized_path.is_absolute()
            or str(normalized_path.resolve()) != normalized_value
        ):
            raise ValueError("consensus normalized_path is not canonical and absolute")
        context = _validate_consensus_query_bundle(normalized_path, manifest_path)
        if len(context["manifest"]["queries"]) != EXPECTED_ROLE_COUNT:
            raise ValueError("consensus query bundle must contain exactly eight roles")
        return manifest_path, context

    def _ensure_work_dir(self, manifest_path: Path) -> Path:
        requested = self.requested_work_dir.expanduser()
        if not requested.is_absolute():
            requested = Path.cwd() / requested
        if _lexists(requested) and (requested.is_symlink() or not requested.is_dir()):
            raise ValueError("work directory must be a regular non-symlink directory")
        work_dir = requested.resolve()
        bundle_root = manifest_path.parent.resolve()
        if _is_within(work_dir, bundle_root) or _is_within(bundle_root, work_dir):
            raise ValueError(
                "work directory may not overlap the immutable consensus query bundle"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        if work_dir.is_symlink() or not work_dir.is_dir():
            raise ValueError("work directory must be a regular non-symlink directory")
        return work_dir

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        manifest_path, _ = self._load_context()
        work_dir = self._ensure_work_dir(manifest_path)
        lock_path = work_dir / ".curation.lock"
        if lock_path.is_symlink():
            raise CurationContradiction("curation lock may not be a symlink")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(os.fspath(lock_path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise CurationContradiction("curation lock is not a regular file")
            if fcntl is None:
                raise RuntimeError("curation orchestration requires Unix file locking")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"another curation orchestrator holds {lock_path}"
                ) from exc
            yield
        finally:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _prepare_locked(self) -> PreparedPlan:
        manifest_path, context = self._load_context()
        work_dir = self._ensure_work_dir(manifest_path)
        query_manifest = context["manifest"]
        roles: Dict[str, RolePlan] = {}
        for role in sorted(query_manifest["queries"]):
            role_parts = tuple(role.split("/"))
            if len(role_parts) != 2 or any(
                not part or part in {".", ".."} for part in role_parts
            ):
                raise ValueError(f"unsafe consensus query role: {role!r}")
            artifact = query_manifest["queries"][role]
            model = artifact["model"]
            role_root = work_dir.joinpath("roles", *role_parts)
            split_dir = role_root / "query-shards"
            analysis_dir = role_root / "analysis-shards"
            _ensure_directory_beneath(role_root, work_dir, f"{role} work directory")
            if _lexists(split_dir) and (
                split_dir.is_symlink() or not split_dir.is_dir()
            ):
                raise CurationContradiction(
                    f"query shard directory is unsafe for role {role}"
                )
            if _lexists(analysis_dir) and (
                analysis_dir.is_symlink() or not analysis_dir.is_dir()
            ):
                raise CurationContradiction(
                    f"analysis shard directory is unsafe for role {role}"
                )
            query_path = Path(context["query_paths"][role]).resolve()
            split_manifest = split_queries(
                query_path, split_dir, shard_count=self.shard_count
            )
            split_manifest_path = (split_dir / "manifest.json").resolve()
            if (
                split_manifest.get("source_path") != str(query_path)
                or split_manifest.get("source_sha256") != artifact["sha256"]
                or split_manifest.get("shard_count") != self.shard_count
                or split_manifest.get("manifest_sha256")
                != canonical_sha256(
                    {
                        key: value
                        for key, value in split_manifest.items()
                        if key != "manifest_sha256"
                    }
                )
            ):
                raise CurationContradiction(
                    f"query shard manifest contradicts role {role}"
                )
            _ensure_directory_beneath(
                analysis_dir, work_dir, f"{role} analysis directory"
            )
            jobs: List[ShardJob] = []
            for shard in split_manifest["shards"]:
                shard_path = (split_dir / shard["path"]).resolve()
                rows = _load_canonical_jsonl(
                    shard_path, f"{role} query shard {shard['index']}"
                )
                ids = tuple(row.get("id") for row in rows)
                if (
                    any(
                        not isinstance(record_id, str) or not record_id
                        for record_id in ids
                    )
                    or len(ids) != len(set(ids))
                    or shard.get("row_count") != len(rows)
                    or shard.get("ids_sha256") != canonical_sha256(list(ids))
                    or shard.get("sha256") != file_sha256(shard_path)
                ):
                    raise CurationContradiction(
                        f"query shard {role}/{shard['index']} changed"
                    )
                output_path = (
                    analysis_dir / f"shard-{shard['index']:03d}.jsonl"
                ).resolve()
                jobs.append(
                    ShardJob(
                        role=role,
                        model=model,
                        model_path=Path(context["model_paths"][model]).resolve(),
                        model_sha256=query_manifest["models"][model]["sha256"],
                        index=shard["index"],
                        query_path=shard_path,
                        query_sha256=shard["sha256"],
                        role_query_sha256=artifact["sha256"],
                        query_ids=ids,
                        row_count=shard["row_count"],
                        split_manifest_path=split_manifest_path,
                        output_path=output_path,
                    )
                )
            roles[role] = RolePlan(
                role=role,
                model=model,
                model_path=Path(context["model_paths"][model]).resolve(),
                model_sha256=query_manifest["models"][model]["sha256"],
                query_path=query_path,
                query_sha256=artifact["sha256"],
                split_manifest_path=split_manifest_path,
                split_manifest_sha256=file_sha256(split_manifest_path),
                split_manifest_identity=split_manifest["manifest_sha256"],
                jobs=tuple(sorted(jobs, key=lambda job: job.index)),
                merged_output_path=(role_root / "analysis.jsonl").resolve(),
            )
        if len(roles) != EXPECTED_ROLE_COUNT:
            raise ValueError("consensus query inventory must contain eight roles")
        prepared = PreparedPlan(
            context=context,
            manifest_path=manifest_path,
            manifest_file_sha256=file_sha256(manifest_path),
            manifest_identity=context["manifest_identity"],
            work_dir=work_dir,
            roles=roles,
        )
        self._restore_status_history(prepared)
        return prepared

    def _status_path(self, prepared: PreparedPlan) -> Path:
        return prepared.work_dir / "status.json"

    def _restore_status_history(self, prepared: PreparedPlan) -> None:
        status_path = self._status_path(prepared)
        if not _lexists(status_path):
            if self._started_at is None:
                started_at = float(self.clock())
                if not math.isfinite(started_at):
                    raise RuntimeError("wall clock returned a non-finite value")
                self._started_at = started_at
            return
        try:
            status = _load_canonical_json(status_path, "curation status")
        except ValueError as exc:
            raise CurationContradiction(f"curation status is invalid: {exc}") from exc
        payload = dict(status)
        status_hash = payload.pop("status_sha256", None)
        bundle = status.get("query_bundle")
        if (
            status.get("schema_version") != STATUS_SCHEMA_VERSION
            or status.get("contract") != STATUS_CONTRACT
            or status_hash != canonical_sha256(payload)
            or not isinstance(bundle, Mapping)
            or bundle.get("path") != str(prepared.manifest_path)
            or bundle.get("sha256") != prepared.manifest_file_sha256
            or bundle.get("identity") != prepared.manifest_identity
            or status.get("work_directory") != str(prepared.work_dir)
            or status.get("shards_per_role") != self.shard_count
        ):
            raise CurationContradiction(
                "existing curation status contradicts the current plan"
            )
        started_at = status.get("started_at_unix")
        if (
            isinstance(started_at, bool)
            or not isinstance(started_at, (int, float))
            or not math.isfinite(float(started_at))
        ):
            raise CurationContradiction("existing curation status has invalid timing")
        self._started_at = float(started_at)
        raw_roles = status.get("roles")
        if not isinstance(raw_roles, Mapping):
            return
        known_jobs = {job.key: job for job in prepared.jobs}
        for role, raw_role in raw_roles.items():
            if not isinstance(raw_role, Mapping):
                continue
            raw_shards = raw_role.get("shards")
            if not isinstance(raw_shards, list):
                continue
            for raw_shard in raw_shards:
                if not isinstance(raw_shard, Mapping):
                    continue
                index = raw_shard.get("index")
                key = (role, index)
                job = known_jobs.get(key)
                if (
                    job is None
                    or raw_shard.get("query_sha256") != job.query_sha256
                    or raw_shard.get("output_path") != str(job.output_path)
                ):
                    continue
                history: Dict[str, Any] = {}
                duration = raw_shard.get("duration_seconds")
                if (
                    isinstance(duration, (int, float))
                    and not isinstance(duration, bool)
                    and math.isfinite(float(duration))
                    and duration >= 0
                ):
                    history["duration_seconds"] = _round_seconds(float(duration))
                gpu = raw_shard.get("gpu")
                if isinstance(gpu, str) and gpu:
                    history["gpu"] = gpu
                if history:
                    self._history[key] = history

    def _assert_frozen_inputs(self, prepared: PreparedPlan) -> None:
        for source, expected_hash, role in prepared.context["frozen_files"]:
            path = Path(source)
            if (
                path.is_symlink()
                or not path.is_file()
                or file_sha256(path) != expected_hash
            ):
                raise CurationContradiction(
                    f"{role} changed during curation orchestration"
                )
        for role_plan in prepared.roles.values():
            if (
                role_plan.split_manifest_path.is_symlink()
                or not role_plan.split_manifest_path.is_file()
                or file_sha256(role_plan.split_manifest_path)
                != role_plan.split_manifest_sha256
            ):
                raise CurationContradiction(
                    f"query shard manifest changed for role {role_plan.role}"
                )
            for job in role_plan.jobs:
                if (
                    job.query_path.is_symlink()
                    or not job.query_path.is_file()
                    or file_sha256(job.query_path) != job.query_sha256
                ):
                    raise CurationContradiction(
                        f"query shard changed for {job.role}/{job.index}"
                    )

    def _validate_output_rows(self, job: ShardJob) -> Mapping[str, Any]:
        rows = _load_canonical_jsonl(
            job.output_path, f"{job.role} analysis shard {job.index}"
        )
        ids: List[str] = []
        seen = set()
        for row in rows:
            record_id = row.get("id")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in seen
                or "error" in row
            ):
                raise ValueError("analysis responses have invalid or duplicate IDs")
            ids.append(record_id)
            seen.add(record_id)
        if (
            len(rows) != job.row_count
            or set(ids) != set(job.query_ids)
            or ids != sorted(job.query_ids)
        ):
            raise ValueError("analysis response IDs do not match the query shard")
        return {
            "row_count": len(rows),
            "output_sha256": file_sha256(job.output_path),
        }

    def _validate_complete_job(self, job: ShardJob) -> Mapping[str, Any]:
        output = self._validate_output_rows(job)
        manifest = _load_canonical_json(
            job.manifest_path,
            f"{job.role} shard {job.index} run-analysis manifest",
        )
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
        payload = dict(manifest)
        manifest_identity = payload.pop("manifest_sha256", None)
        query_manifest = self._current_manifest_for_job(job)
        expected_argv = [
            query_manifest["katago_path"],
            "analysis",
            "-config",
            query_manifest["analysis_config_path"],
            "-model",
            str(job.model_path),
        ]
        gpu = manifest.get("cuda_visible_devices")
        if (
            set(manifest) != expected_keys
            or manifest.get("schema_version") != 1
            or manifest.get("contract") != ANALYSIS_RUN_CONTRACT
            or manifest_identity != canonical_sha256(payload)
            or manifest.get("argv") != expected_argv
            or manifest.get("katago_sha256") != query_manifest["katago_sha256"]
            or manifest.get("config_sha256") != query_manifest["analysis_config_sha256"]
            or manifest.get("model_sha256") != job.model_sha256
            or not isinstance(gpu, str)
            or not gpu
            or "," in gpu
            or any(character.isspace() for character in gpu)
            or manifest.get("query_path") != str(job.query_path)
            or manifest.get("query_sha256") != job.query_sha256
            or manifest.get("output_path") != str(job.output_path)
            or manifest.get("output_sha256") != output["output_sha256"]
            or manifest.get("row_count") != output["row_count"]
        ):
            raise ValueError("run-analysis manifest is not bound to this shard")
        return {
            **output,
            "gpu": gpu,
            "manifest_identity": manifest_identity,
            "manifest_sha256": file_sha256(job.manifest_path),
        }

    def _current_manifest_for_job(self, job: ShardJob) -> Mapping[str, Any]:
        # Every job was built from the currently validated bundle.  Reloading it
        # here also catches replacement between planning and reconciliation.
        manifest = _load_canonical_json(
            self.query_manifest_path.expanduser().resolve(),
            "consensus query bundle manifest",
        )
        artifact = manifest.get("queries", {}).get(job.role)
        model = manifest.get("models", {}).get(job.model)
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(model, Mapping)
            or artifact.get("sha256") != job.role_query_sha256
            or artifact.get("model") != job.model
            or artifact.get("model_sha256") != job.model_sha256
            or model.get("path") != str(job.model_path)
            or model.get("sha256") != job.model_sha256
        ):
            raise CurationContradiction(
                f"query manifest role binding changed for {job.role}"
            )
        return manifest

    def _inspect_job(self, job: ShardJob) -> Mapping[str, Any]:
        output_exists = _lexists(job.output_path)
        manifest_exists = _lexists(job.manifest_path)
        if not output_exists and not manifest_exists:
            return {"state": "pending"}
        if manifest_exists and not output_exists:
            raise CurationContradiction(
                f"{job.role} shard {job.index} has a manifest without its output"
            )
        if job.output_path.is_symlink() or not job.output_path.is_file():
            raise CurationContradiction(
                f"{job.role} shard {job.index} output is not a regular file"
            )
        if not manifest_exists:
            try:
                output = self._validate_output_rows(job)
            except ValueError as exc:
                return {
                    "state": "invalid",
                    "recoverable_output": False,
                    "reason": str(exc),
                }
            return {
                "state": "invalid",
                "recoverable_output": True,
                **output,
            }
        try:
            complete = self._validate_complete_job(job)
        except (OSError, ValueError) as exc:
            if isinstance(exc, CurationContradiction):
                raise
            raise CurationContradiction(
                f"{job.role} shard {job.index} artifact contradiction: {exc}"
            ) from exc
        return {"state": "complete", **complete}

    def _quarantine_invalid_output(self, prepared: PreparedPlan, job: ShardJob) -> None:
        if not _lexists(job.output_path):
            return
        if job.output_path.is_symlink() or not job.output_path.is_file():
            raise CurationContradiction(
                f"cannot quarantine unsafe output for {job.role}/{job.index}"
            )
        digest = file_sha256(job.output_path)
        destination_dir = prepared.work_dir.joinpath("orphaned", *job.role.split("/"))
        _ensure_directory_beneath(
            destination_dir,
            prepared.work_dir,
            f"{job.role} orphan quarantine",
        )
        destination = destination_dir / f"{job.output_path.name}.{digest}.orphan"
        if _lexists(destination):
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != job.output_path.read_bytes()
            ):
                raise CurationContradiction(
                    f"orphan quarantine conflicts for {job.role}/{job.index}"
                )
            job.output_path.unlink()
            _fsync_directory(job.output_path.parent)
            return
        os.rename(os.fspath(job.output_path), os.fspath(destination))
        _fsync_directory(job.output_path.parent)
        _fsync_directory(destination_dir)

    def _execute_job(
        self, prepared: PreparedPlan, job: ShardJob, gpu: str
    ) -> Mapping[str, Any]:
        inspection = self._inspect_job(job)
        if inspection["state"] == "complete":
            return inspection
        if inspection["state"] == "invalid" and not inspection.get(
            "recoverable_output"
        ):
            self._quarantine_invalid_output(prepared, job)
        started = self.monotonic()
        if not math.isfinite(float(started)):
            raise RuntimeError("monotonic clock returned a non-finite value")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        self.analysis_runner(
            katago=Path(prepared.context["manifest"]["katago_path"]),
            config=Path(prepared.context["manifest"]["analysis_config_path"]),
            model=job.model_path,
            queries=job.query_path,
            output=job.output_path,
            env=environment,
            subprocess_runner=self.subprocess_runner,
        )
        finished = self.monotonic()
        if not math.isfinite(float(finished)):
            raise RuntimeError("monotonic clock returned a non-finite value")
        duration = _round_seconds(finished - started)
        complete = self._inspect_job(job)
        if complete["state"] != "complete":
            raise RuntimeError(
                f"analysis runner did not publish a complete shard for "
                f"{job.role}/{job.index}"
            )
        if complete.get("gpu") != gpu:
            raise RuntimeError(
                f"analysis runner recorded GPU {complete.get('gpu')!r}; "
                f"expected {gpu!r}"
            )
        return {**complete, "duration_seconds": duration}

    def _validate_merged_role(
        self, prepared: PreparedPlan, role_plan: RolePlan
    ) -> Mapping[str, Any]:
        output_path = role_plan.merged_output_path
        manifest_path = Path(str(output_path) + ".manifest.json")
        if (
            output_path.is_symlink()
            or not output_path.is_file()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise ValueError(f"merged analysis is incomplete for role {role_plan.role}")
        rows = _load_canonical_jsonl(output_path, f"{role_plan.role} merged analysis")
        ids = [row.get("id") for row in rows]
        expected_ids = set().union(*(set(job.query_ids) for job in role_plan.jobs))
        if (
            any(not isinstance(record_id, str) or not record_id for record_id in ids)
            or len(ids) != len(set(ids))
            or set(ids) != expected_ids
        ):
            raise ValueError(f"merged analysis IDs changed for role {role_plan.role}")
        manifest = _load_canonical_json(
            manifest_path, f"{role_plan.role} merged analysis manifest"
        )
        payload = dict(manifest)
        identity = payload.pop("manifest_sha256", None)
        expected_shards = []
        for job in role_plan.jobs:
            complete = self._validate_complete_job(job)
            expected_shards.append(
                {
                    "index": job.index,
                    "path": str(job.manifest_path),
                    "sha256": complete["manifest_sha256"],
                    "query_path": str(job.query_path),
                    "query_sha256": job.query_sha256,
                    "output_sha256": complete["output_sha256"],
                    "cuda_visible_devices": complete["gpu"],
                }
            )
        query_manifest = prepared.context["manifest"]
        if (
            manifest.get("schema_version") != 1
            or manifest.get("contract") != ANALYSIS_RUN_CONTRACT
            or identity != canonical_sha256(payload)
            or manifest.get("argv") != ["merge-analysis"]
            or manifest.get("katago_sha256") != query_manifest["katago_sha256"]
            or manifest.get("config_sha256") != query_manifest["analysis_config_sha256"]
            or manifest.get("model_sha256") != role_plan.model_sha256
            or manifest.get("query_path") != str(role_plan.query_path)
            or manifest.get("query_sha256") != role_plan.query_sha256
            or manifest.get("split_manifest_path") != str(role_plan.split_manifest_path)
            or manifest.get("split_manifest_sha256") != role_plan.split_manifest_sha256
            or manifest.get("split_manifest_identity")
            != role_plan.split_manifest_identity
            or manifest.get("output_path") != str(output_path)
            or manifest.get("output_sha256") != file_sha256(output_path)
            or manifest.get("row_count") != len(rows)
            or manifest.get("shards") != expected_shards
            or manifest.get("cuda_visible_devices")
            != sorted({str(item["cuda_visible_devices"]) for item in expected_shards})
        ):
            raise ValueError(
                f"merged analysis manifest changed for role {role_plan.role}"
            )
        return {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            "row_count": len(rows),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_identity": identity,
        }

    def _merge_role(
        self, prepared: PreparedPlan, role_plan: RolePlan
    ) -> Mapping[str, Any]:
        try:
            merge_analysis(
                query_path=role_plan.query_path,
                split_manifest_path=role_plan.split_manifest_path,
                shard_outputs=[job.output_path for job in role_plan.jobs],
                output=role_plan.merged_output_path,
            )
            return self._validate_merged_role(prepared, role_plan)
        except (OSError, ValueError) as exc:
            raise CurationContradiction(
                f"cannot reconcile merged analysis for {role_plan.role}: {exc}"
            ) from exc

    def _progress(
        self,
        prepared: PreparedPlan,
        inspections: Mapping[Tuple[str, int], Mapping[str, Any]],
        *,
        now: float,
        running: Iterable[Tuple[str, int]],
        failures: Mapping[Tuple[str, int], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        running_set = set(running)
        completed_jobs = [
            job for job in prepared.jobs if inspections[job.key]["state"] == "complete"
        ]
        completed_rows = sum(job.row_count for job in completed_jobs)
        total_rows = sum(job.row_count for job in prepared.jobs)
        total_shards = len(prepared.jobs)
        started_at = now if self._started_at is None else self._started_at
        elapsed = _round_seconds(now - float(started_at))
        rows_per_second: Optional[float] = None
        eta_seconds: Optional[float] = None
        if completed_rows > 0 and elapsed > 0:
            raw_rows_per_second = completed_rows / elapsed
            rows_per_second = round(raw_rows_per_second, 6)
            eta_seconds = round(
                max(0, total_rows - completed_rows) / raw_rows_per_second, 6
            )
        return {
            "completed_shards": len(completed_jobs),
            "total_shards": total_shards,
            "running_shards": len(running_set),
            "failed_shards": len(failures),
            "remaining_shards": total_shards - len(completed_jobs),
            "completed_rows": completed_rows,
            "total_rows": total_rows,
            "completion_fraction": round(len(completed_jobs) / total_shards, 6),
            "elapsed_seconds": elapsed,
            "rows_per_second": rows_per_second,
            "eta_seconds": eta_seconds,
        }

    def _build_status(
        self,
        prepared: PreparedPlan,
        *,
        mode: str,
        state: str,
        running: Iterable[Tuple[str, int]] = (),
        failures: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None,
        assignments: Optional[Mapping[Tuple[str, int], str]] = None,
        merged: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        failures = dict(failures or {})
        assignments = dict(assignments or {})
        merged = dict(merged or {})
        running_set = set(running)
        inspections = {job.key: self._inspect_job(job) for job in prepared.jobs}
        role_status: Dict[str, Any] = {}
        for role, role_plan in sorted(prepared.roles.items()):
            shard_status = []
            for job in role_plan.jobs:
                inspection = inspections[job.key]
                shard_state = inspection["state"]
                if job.key in running_set:
                    shard_state = "running"
                elif job.key in failures and shard_state != "complete":
                    shard_state = "failed"
                history = self._history.get(job.key, {})
                gpu = inspection.get(
                    "gpu", assignments.get(job.key, history.get("gpu"))
                )
                item: Dict[str, Any] = {
                    "index": job.index,
                    "state": shard_state,
                    "query_path": str(job.query_path),
                    "query_sha256": job.query_sha256,
                    "row_count": job.row_count,
                    "output_path": str(job.output_path),
                    "manifest_path": str(job.manifest_path),
                    "gpu": gpu,
                }
                if "output_sha256" in inspection:
                    item["output_sha256"] = inspection["output_sha256"]
                if "manifest_sha256" in inspection:
                    item["manifest_sha256"] = inspection["manifest_sha256"]
                    item["manifest_identity"] = inspection["manifest_identity"]
                if "duration_seconds" in history:
                    item["duration_seconds"] = history["duration_seconds"]
                if job.key in failures and shard_state != "complete":
                    item["failure"] = failures[job.key]
                if inspection.get("reason"):
                    item["invalid_reason"] = inspection["reason"]
                shard_status.append(item)
            role_status[role] = {
                "model": role_plan.model,
                "model_path": str(role_plan.model_path),
                "model_sha256": role_plan.model_sha256,
                "query_path": str(role_plan.query_path),
                "query_sha256": role_plan.query_sha256,
                "split_manifest_path": str(role_plan.split_manifest_path),
                "split_manifest_sha256": role_plan.split_manifest_sha256,
                "split_manifest_identity": role_plan.split_manifest_identity,
                "merged_output_path": str(role_plan.merged_output_path),
                "merged": merged.get(role),
                "shards": shard_status,
            }
        now = float(self.clock())
        if not math.isfinite(now):
            raise RuntimeError("wall clock returned a non-finite value")
        status: Dict[str, Any] = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "contract": STATUS_CONTRACT,
            "mode": mode,
            "state": state,
            "query_bundle": {
                "contract": CONSENSUS_QUERY_BUNDLE_CONTRACT,
                "path": str(prepared.manifest_path),
                "sha256": prepared.manifest_file_sha256,
                "identity": prepared.manifest_identity,
            },
            "work_directory": str(prepared.work_dir),
            "shards_per_role": self.shard_count,
            "scheduler": {
                "gpus": list(self.gpus),
                "per_gpu_parallelism": self.per_gpu_parallelism,
                "maximum_parallel_processes": len(self.gpus) * self.per_gpu_parallelism,
            },
            "started_at_unix": float(
                now if self._started_at is None else self._started_at
            ),
            "updated_at_unix": now,
            "progress": self._progress(
                prepared,
                inspections,
                now=now,
                running=running_set,
                failures=failures,
            ),
            "ready_for_labeling": state == "complete",
            "analysis_outputs": (
                {role: metadata["path"] for role, metadata in sorted(merged.items())}
                if state == "complete"
                else {}
            ),
            "roles": role_status,
        }
        status["status_sha256"] = canonical_sha256(status)
        return status

    def _persist_status(
        self, prepared: PreparedPlan, status: Mapping[str, Any], *, event: str
    ) -> None:
        _atomic_replace_json(self._status_path(prepared), status)
        if self.progress_callback is not None:
            progress = status["progress"]
            self.progress_callback(
                {
                    "event": event,
                    "state": status["state"],
                    "completed_shards": progress["completed_shards"],
                    "total_shards": progress["total_shards"],
                    "completed_rows": progress["completed_rows"],
                    "total_rows": progress["total_rows"],
                    "running_shards": progress["running_shards"],
                    "failed_shards": progress["failed_shards"],
                    "rows_per_second": progress["rows_per_second"],
                    "eta_seconds": progress["eta_seconds"],
                }
            )

    def plan(self) -> Mapping[str, Any]:
        """Validate and materialize shard plans without launching analysis."""

        with self._exclusive_lock():
            prepared = self._prepare_locked()
            self._assert_frozen_inputs(prepared)
            status = self._build_status(prepared, mode="plan", state="planned")
            self._persist_status(prepared, status, event="planned")
            return status

    def _run_once_locked(self, *, mode: str) -> Mapping[str, Any]:
        prepared = self._prepare_locked()
        self._assert_frozen_inputs(prepared)
        initial = {job.key: self._inspect_job(job) for job in prepared.jobs}
        jobs = [job for job in prepared.jobs if initial[job.key]["state"] != "complete"]
        pending = deque(jobs)
        assignments: Dict[Tuple[str, int], str] = {}
        running = set()
        failures: Dict[Tuple[str, int], Mapping[str, Any]] = {}

        # Reserve only genuinely available worker slots.  All other work stays
        # unassigned in the global queue until a slot finishes its current job.
        initial_claims: List[Tuple[ShardJob, str]] = []
        for _ in range(self.per_gpu_parallelism):
            for gpu in self.gpus:
                if not pending:
                    break
                job = pending.popleft()
                assignments[job.key] = gpu
                running.add(job.key)
                initial_claims.append((job, gpu))
            if not pending:
                break

        status = self._build_status(
            prepared,
            mode=mode,
            state="running",
            running=running,
            assignments=assignments,
        )
        self._persist_status(prepared, status, event="started")

        executors = {
            gpu: concurrent.futures.ThreadPoolExecutor(
                max_workers=self.per_gpu_parallelism,
                thread_name_prefix=f"curation-gpu-{gpu}",
            )
            for gpu in self.gpus
        }
        futures: Dict[
            concurrent.futures.Future[Mapping[str, Any]], Tuple[ShardJob, str]
        ] = {}
        interrupted = False
        try:
            for job, gpu in initial_claims:
                future = executors[gpu].submit(self._execute_job, prepared, job, gpu)
                futures[future] = (job, gpu)

            job_order = {job.key: index for index, job in enumerate(jobs)}
            while futures:
                done, _ = concurrent.futures.wait(
                    tuple(futures),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in sorted(
                    done,
                    key=lambda item: job_order[futures[item][0].key],
                ):
                    job, gpu = futures.pop(future)
                    try:
                        result = future.result()
                        self._history[job.key] = {
                            "gpu": gpu,
                            "duration_seconds": result.get("duration_seconds", 0.0),
                        }
                    except Exception as exc:  # Individual failures do not hide siblings.
                        failures[job.key] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    running.discard(job.key)

                    # The slot that just became available claims the next global
                    # shard, so faster GPUs naturally consume more of the queue.
                    if pending:
                        next_job = pending.popleft()
                        next_future = executors[gpu].submit(
                            self._execute_job, prepared, next_job, gpu
                        )
                        futures[next_future] = (next_job, gpu)
                        assignments[next_job.key] = gpu
                        running.add(next_job.key)

                    status = self._build_status(
                        prepared,
                        mode=mode,
                        state="running",
                        running=running,
                        failures=failures,
                        assignments=assignments,
                    )
                    self._persist_status(
                        prepared, status, event="shard-finished"
                    )
        except BaseException:
            interrupted = True
            for future in futures:
                future.cancel()
            running = {
                job.key for future, (job, _) in futures.items() if not future.done()
            }
            status = self._build_status(
                prepared,
                mode=mode,
                state="interrupted",
                running=running,
                failures=failures,
                assignments=assignments,
            )
            self._persist_status(prepared, status, event="interrupted")
            raise
        finally:
            for executor in executors.values():
                executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

        self._assert_frozen_inputs(prepared)
        final_inspections = {job.key: self._inspect_job(job) for job in prepared.jobs}
        failures = {
            key: failure
            for key, failure in failures.items()
            if final_inspections[key]["state"] != "complete"
        }
        incomplete = [
            job
            for job in prepared.jobs
            if final_inspections[job.key]["state"] != "complete"
        ]
        if incomplete:
            status = self._build_status(
                prepared,
                mode=mode,
                state="failed" if failures else "incomplete",
                failures=failures,
                assignments=assignments,
            )
            self._persist_status(prepared, status, event=status["state"])
            return status

        merged = {
            role: self._merge_role(prepared, role_plan)
            for role, role_plan in sorted(prepared.roles.items())
        }
        self._assert_frozen_inputs(prepared)
        status = self._build_status(
            prepared,
            mode=mode,
            state="complete",
            assignments=assignments,
            merged=merged,
        )
        self._persist_status(prepared, status, event="complete")
        return status

    def once(self) -> Mapping[str, Any]:
        """Run one reconciliation/scheduling pass."""

        with self._exclusive_lock():
            return self._run_once_locked(mode="once")

    def _watch(
        self,
        *,
        poll_interval: float = 30.0,
        max_passes: Optional[int] = None,
        mode: str,
    ) -> Mapping[str, Any]:
        """Resume until complete, retrying non-contradictory failed shards."""

        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(float(poll_interval))
            or poll_interval < 0
        ):
            raise ValueError("watch poll interval must be finite and nonnegative")
        if max_passes is not None and (type(max_passes) is not int or max_passes < 1):
            raise ValueError("watch max passes must be a positive integer")
        with self._exclusive_lock():
            passes = 0
            while True:
                passes += 1
                status = self._run_once_locked(mode=mode)
                if status["state"] == "complete":
                    return status
                if max_passes is not None and passes >= max_passes:
                    return status
                if self.progress_callback is not None:
                    self.progress_callback(
                        {
                            "event": "waiting",
                            "state": status["state"],
                            "pass": passes,
                            "poll_interval_seconds": float(poll_interval),
                            "eta_seconds": status["progress"]["eta_seconds"],
                        }
                    )
                self.sleeper(float(poll_interval))

    def watch(
        self, *, poll_interval: float = 30.0, max_passes: Optional[int] = None
    ) -> Mapping[str, Any]:
        return self._watch(
            poll_interval=poll_interval, max_passes=max_passes, mode="watch"
        )

    def resume(
        self, *, poll_interval: float = 30.0, max_passes: Optional[int] = None
    ) -> Mapping[str, Any]:
        return self._watch(
            poll_interval=poll_interval, max_passes=max_passes, mode="resume"
        )


def orchestrate(
    *,
    mode: str,
    query_manifest_path: Path,
    work_dir: Path,
    shard_count: int,
    gpus: Sequence[Any],
    per_gpu_parallelism: int = 1,
    poll_interval: float = 30.0,
    max_passes: Optional[int] = None,
    analysis_runner: Callable[..., Mapping[str, Any]] = run_analysis,
    subprocess_runner: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Mapping[str, Any]:
    """Convenience API used by the CLI and automation callers."""

    orchestrator = CurationOrchestrator(
        query_manifest_path=query_manifest_path,
        work_dir=work_dir,
        shard_count=shard_count,
        gpus=gpus,
        per_gpu_parallelism=per_gpu_parallelism,
        analysis_runner=analysis_runner,
        subprocess_runner=subprocess_runner,
        clock=clock,
        monotonic=monotonic,
        sleeper=sleeper,
        progress_callback=progress_callback,
    )
    if mode == "plan":
        return orchestrator.plan()
    if mode == "once":
        return orchestrator.once()
    if mode == "watch":
        return orchestrator.watch(poll_interval=poll_interval, max_passes=max_passes)
    if mode == "resume":
        return orchestrator.resume(poll_interval=poll_interval, max_passes=max_passes)
    raise ValueError(f"unsupported curation mode: {mode}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restart and schedule an immutable machine-consensus analysis bundle."
        )
    )
    parser.add_argument("mode", choices=("plan", "once", "watch", "resume"))
    parser.add_argument(
        "--query-manifest",
        "--manifest",
        required=True,
        type=Path,
        dest="query_manifest",
    )
    parser.add_argument(
        "--work-dir", "--output-dir", required=True, type=Path, dest="work_dir"
    )
    parser.add_argument(
        "--shards-per-role", "--shards", type=int, default=1, dest="shard_count"
    )
    gpu_group = parser.add_mutually_exclusive_group(required=True)
    gpu_group.add_argument(
        "--gpus",
        nargs="+",
        help="Explicit GPU identifiers (space- or comma-separated).",
    )
    gpu_group.add_argument(
        "--gpu",
        action="append",
        help="Explicit GPU identifier; repeat for multiple GPUs.",
    )
    parser.add_argument("--per-gpu-parallelism", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--max-passes", type=int)
    return parser.parse_args(argv)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    analysis_runner: Callable[..., Mapping[str, Any]] = run_analysis,
    subprocess_runner: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    args = parse_args(argv)

    def emit(event: Mapping[str, Any]) -> None:
        print(canonical_json(event), file=sys.stderr, flush=True)

    try:
        status = orchestrate(
            mode=args.mode,
            query_manifest_path=args.query_manifest,
            work_dir=args.work_dir,
            shard_count=args.shard_count,
            gpus=(args.gpus if args.gpus is not None else args.gpu),
            per_gpu_parallelism=args.per_gpu_parallelism,
            poll_interval=args.poll_interval,
            max_passes=args.max_passes,
            analysis_runner=analysis_runner,
            subprocess_runner=subprocess_runner,
            clock=clock,
            monotonic=monotonic,
            sleeper=sleeper,
            progress_callback=emit,
        )
    except KeyboardInterrupt:
        print(
            canonical_json(
                {"error": {"type": "KeyboardInterrupt", "message": "interrupted"}}
            ),
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            canonical_json(
                {"error": {"type": type(exc).__name__, "message": str(exc)}}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(status))
    if status["state"] in {"planned", "complete"}:
        return 0
    if status["state"] == "interrupted":
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
