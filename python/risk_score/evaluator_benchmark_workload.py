#!/usr/bin/env python3
"""Run the immutable production workload for evaluator-topology benchmarks.

The generic topology benchmark deliberately knows nothing about KataGo.  This
adapter verifies a second, self-hashed workload specification, proves that the
requested physical GPU is idle, runs one KataGo analysis process per shard, and
publishes one topology-independent canonical analysis artifact.

Intermediate queries, raw responses, manifests, and stderr logs are retained
when a run fails.  They are removed only after every child and merged output
has been validated, because the generic runner requires the successful output
tree to contain exactly the artifacts declared by the receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
WORKLOAD_SPEC_CONTRACT = (
    "risk-score-evaluator-topology-benchmark-workload-spec-v1"
)
BENCHMARK_RECEIPT_CONTRACT = (
    "risk-score-evaluator-topology-benchmark-receipt-v1"
)
CHILD_OUTPUT_MANIFEST_CONTRACT = (
    "risk-score-evaluator-topology-benchmark-child-output-v1"
)

PROCESS_COUNTS = (4, 8, 16)
MAX_SPEC_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_QUERY_BYTES = 256 * 1024 * 1024
MAX_QUERY_ROWS = 100_000
MAX_QUERY_LINE_BYTES = 4 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 4 * 1024 * 1024 * 1024
MAX_MODEL_BYTES = 64 * 1024 * 1024 * 1024
MAX_VISITS = 100_000_000
MAX_TIMEOUT_SECONDS = 24 * 60 * 60
MAX_PATH_BYTES = 4096
MAX_GPU_PROBE_BYTES = 1024 * 1024
GPU_PROBE_TIMEOUT_SECONDS = 10.0
MAX_CHILD_STDERR_BYTES = 16 * 1024 * 1024
MAX_CHILD_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_TOTAL_CHILD_OUTPUT_BYTES = 512 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_MIG_UUID_RE = re.compile(r"^MIG-[A-Za-z0-9_./-]+$")
_NVIDIA_L_GPU_RE = re.compile(
    r"^GPU ([0-9]+): .+ \(UUID: (GPU-[A-Za-z0-9-]+)\)$"
)
_NVIDIA_L_MIG_RE = re.compile(
    r"^\s+MIG .+ Device ([0-9]+): "
    r"\(UUID: (MIG-[A-Za-z0-9_./-]+)\)$"
)
_QUERY_KEYS = {
    "id",
    "moves",
    "initialStones",
    "initialPlayer",
    "rules",
    "komi",
    "boardXSize",
    "boardYSize",
    "includePolicy",
    "maxVisits",
    "overrideSettings",
}
_QUERY_OVERRIDE_KEYS = {
    "useScoreMaximizingUtility",
    "scorePower",
    "scoreScale",
    "winWeight",
    "rootNoiseEnabled",
    "rootNumSymmetriesToSample",
}
_SPEC_KEYS = {
    "schema_version",
    "contract",
    "katago",
    "analysis_config",
    "model",
    "queries",
    "nvidia_smi",
    "gpu",
    "timeout_seconds",
    "spec_sha256",
}
_BINDING_KEYS = {"path", "sha256"}
_QUERY_BINDING_KEYS = {"path", "sha256", "row_count", "max_visits"}
_GPU_KEYS = {"index", "uuid"}
_RECEIPT_KEYS = {
    "schema_version",
    "contract",
    "process_count",
    "completed_work_count",
    "elapsed_seconds",
    "output_manifest",
    "output_manifest_sha256",
    "receipt_sha256",
}
_ARTIFACT_KEYS = {"path", "sha256", "size_bytes", "row_count"}
_CHILD_MANIFEST_KEYS = {
    "schema_version",
    "contract",
    "shard_index",
    "returncode",
    "argv",
    "gpu",
    "query",
    "output",
    "manifest_sha256",
}
_CHILD_GPU_KEYS = {"index", "uuid", "cuda_visible_devices"}
_CHILD_FILE_KEYS = {
    "path",
    "sha256",
    "size_bytes",
    "row_count",
    "ids_sha256",
}
_ATTEMPT_NAME = ".evaluator-benchmark-workload-attempt"
_ARTIFACT_RELATIVE = PurePosixPath("artifacts/analysis.jsonl")


class EvaluatorBenchmarkWorkloadError(RuntimeError):
    """The workload specification, execution, or publication is unsafe."""


class WorkloadSpecError(EvaluatorBenchmarkWorkloadError, ValueError):
    """The immutable workload specification is malformed or stale."""


class WorkloadExecutionError(EvaluatorBenchmarkWorkloadError):
    """A GPU probe, KataGo child, or output validation failed."""


class WorkloadConflictError(EvaluatorBenchmarkWorkloadError):
    """Existing filesystem state contradicts this immutable run."""


def canonical_json(value: Any) -> str:
    """Return compact, sorted, finite JSON used by all workload contracts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorBenchmarkWorkloadError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvaluatorBenchmarkWorkloadError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _ensure_finite_json(value: Any, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} contains a non-finite number"
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise EvaluatorBenchmarkWorkloadError(
                    f"{role} contains a non-string object key"
                )
            _ensure_finite_json(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{role}[{index}]")


def _json_value(data: bytes, role: str) -> Any:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EvaluatorBenchmarkWorkloadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} is invalid JSON: {exc}"
        ) from exc
    _ensure_finite_json(value, role)
    return value


def _exact_keys(
    value: Any, expected: set[str], role: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluatorBenchmarkWorkloadError(f"{role} must be an object")
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} keys differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be a lowercase 64-character SHA-256"
        )
    return value


def _positive_int(value: Any, role: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be an integer between 1 and {maximum}"
        )
    return value


def _positive_number(value: Any, role: str, *, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= maximum
    ):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be a positive finite number no greater than {maximum:g}"
        )
    return float(value)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} has a symlinked path component: {current}"
            )
        if current.parent == current:
            return
        current = current.parent


def _canonical_absolute(raw: Any, role: str) -> Path:
    if not isinstance(raw, (str, os.PathLike)):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be an absolute path"
        )
    text = os.fspath(raw)
    if (
        not isinstance(text, str)
        or not text
        or "\x00" in text
        or "\n" in text
        or "\r" in text
        or len(os.fsencode(text)) > MAX_PATH_BYTES
    ):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be a safe nonempty path"
        )
    path = Path(text)
    normalized = Path(os.path.abspath(text))
    if not path.is_absolute() or path != normalized:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be an absolute lexically normalized path"
        )
    _reject_symlink_ancestors(path, role)
    if path.resolve(strict=False) != path:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must contain no symlink components"
        )
    return path


def _spec_absolute(raw: Any, role: str) -> Path:
    try:
        return _canonical_absolute(raw, role)
    except EvaluatorBenchmarkWorkloadError as exc:
        if isinstance(exc, WorkloadSpecError):
            raise
        raise WorkloadSpecError(str(exc)) from exc


def _strictly_within(path: Path, root: Path) -> bool:
    if path == root:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _metadata_identity(metadata: os.stat_result) -> Tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _read_stable_regular_file(
    path: Path,
    role: str,
    *,
    maximum_bytes: int,
    require_single_link: bool = False,
) -> bytes:
    source = Path(path)
    _reject_symlink_ancestors(source, role)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(source), flags)
    except OSError as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be an existing regular non-symlink file: {exc}"
        ) from exc
    chunks = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} must be a regular file"
            )
        if require_single_link and before.st_nlink != 1:
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} must have exactly one hard link"
            )
        if before.st_size <= 0:
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} must be nonempty"
            )
        if before.st_size > maximum_bytes:
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} exceeds the {maximum_bytes}-byte limit"
            )
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        if remaining:
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} was truncated while read"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = source.lstat()
    except OSError as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} disappeared while it was read"
        ) from exc
    if (
        _metadata_identity(before) != _metadata_identity(after)
        or _metadata_identity(after) != _metadata_identity(path_after)
    ):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} changed while it was read"
        )
    _reject_symlink_ancestors(source, role)
    return b"".join(chunks)


def _stable_file_sha256(
    path: Path,
    role: str,
    *,
    maximum_bytes: int,
    executable: bool = False,
) -> str:
    source = Path(path)
    _reject_symlink_ancestors(source, role)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(source), flags)
    except OSError as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be an existing regular non-symlink file: {exc}"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} must be a regular file"
            )
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} size is outside the supported bound"
            )
        if executable and before.st_mode & 0o111 == 0:
            raise EvaluatorBenchmarkWorkloadError(
                f"{role} must be executable"
            )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = source.lstat()
    except OSError as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} disappeared while it was hashed"
        ) from exc
    if (
        _metadata_identity(before) != _metadata_identity(after)
        or _metadata_identity(after) != _metadata_identity(path_after)
    ):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} changed while it was hashed"
        )
    _reject_symlink_ancestors(source, role)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return _stable_file_sha256(
        Path(path),
        f"file {path}",
        maximum_bytes=MAX_MODEL_BYTES,
    )


def _load_canonical_object(
    path: Path,
    role: str,
    *,
    maximum_bytes: int = MAX_SPEC_BYTES,
    require_single_link: bool = True,
) -> Dict[str, Any]:
    data = _read_stable_regular_file(
        path,
        role,
        maximum_bytes=maximum_bytes,
        require_single_link=require_single_link,
    )
    value = _json_value(data, role)
    if not isinstance(value, dict):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} root must be an object"
        )
    try:
        expected = (canonical_json(value) + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} cannot be represented canonically"
        ) from exc
    if data != expected:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str
    maximum_bytes: int
    executable: bool = False

    def spec_value(self) -> Mapping[str, str]:
        return {"path": os.fspath(self.path), "sha256": self.sha256}

    def verify(self, role: str) -> None:
        observed = _stable_file_sha256(
            self.path,
            role,
            maximum_bytes=self.maximum_bytes,
            executable=self.executable,
        )
        if observed != self.sha256:
            raise WorkloadSpecError(
                f"hash-bound input changed: {self.path}; "
                f"expected {self.sha256}, observed {observed}"
            )


@dataclass(frozen=True)
class WorkloadSpec:
    path: Path
    file_sha256: str
    identity: str
    raw: Mapping[str, Any]
    katago: FileBinding
    analysis_config: FileBinding
    model: FileBinding
    queries: FileBinding
    nvidia_smi: FileBinding
    gpu_index: int
    expected_gpu_uuid: str
    row_count: int
    max_visits: int
    timeout_seconds: float

    def assert_frozen(self) -> None:
        observed = _stable_file_sha256(
            self.path,
            "workload specification",
            maximum_bytes=MAX_SPEC_BYTES,
        )
        if observed != self.file_sha256:
            raise WorkloadSpecError(
                "workload specification changed during execution"
            )
        self.katago.verify("KataGo binary")
        self.analysis_config.verify("analysis config")
        self.model.verify("analysis model")
        self.queries.verify("analysis queries")
        self.nvidia_smi.verify("nvidia-smi binary")


def _binding(
    value: Any,
    role: str,
    *,
    maximum_bytes: int,
    executable: bool = False,
) -> FileBinding:
    raw = _exact_keys(value, _BINDING_KEYS, role)
    path = _canonical_absolute(raw["path"], f"{role} path")
    digest = _require_sha256(raw["sha256"], f"{role} sha256")
    binding = FileBinding(path, digest, maximum_bytes, executable)
    binding.verify(role)
    return binding


def _validate_deterministic_config(path: Path) -> None:
    data = _read_stable_regular_file(
        path,
        "analysis config",
        maximum_bytes=MAX_CONFIG_BYTES,
    )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkloadSpecError("analysis config must be UTF-8") from exc
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in values:
            raise WorkloadSpecError(
                f"analysis config has a duplicate/empty key on line {line_number}"
            )
        values[key] = value.lower()
    required = {
        "forDeterministicTesting": "true",
        "numAnalysisThreads": "1",
        "numSearchThreadsPerAnalysisThread": "1",
        "nnRandomize": "false",
        "rootNoiseEnabled": "false",
        "rootNumSymmetriesToSample": "1",
        "useUncertainty": "false",
        "cpuctUtilityStdevScale": "0",
        "reportAnalysisWinratesAs": "sidetomove",
    }
    conflicts = {
        key: {"expected": expected, "actual": values.get(key)}
        for key, expected in required.items()
        if values.get(key) != expected
    }
    if conflicts:
        raise WorkloadSpecError(
            "analysis config is not deterministic and perspective-fixed: "
            f"{conflicts}"
        )


def _load_query_rows(
    path: Path,
    *,
    row_count: int,
    max_visits: int,
) -> Tuple[Mapping[str, Any], ...]:
    data = _read_stable_regular_file(
        path,
        "analysis queries",
        maximum_bytes=MAX_QUERY_BYTES,
    )
    if not data.endswith(b"\n"):
        raise WorkloadSpecError(
            "analysis queries must be newline-terminated JSONL"
        )
    raw_lines = data[:-1].split(b"\n")
    if (
        not raw_lines
        or any(not line for line in raw_lines)
        or len(raw_lines) > MAX_QUERY_ROWS
    ):
        raise WorkloadSpecError(
            "analysis queries have an invalid or excessive row count"
        )
    rows = []
    seen_ids = set()
    for index, raw_line in enumerate(raw_lines):
        if len(raw_line) > MAX_QUERY_LINE_BYTES:
            raise WorkloadSpecError(
                f"analysis query row {index} exceeds the line-size limit"
            )
        try:
            row = _json_value(raw_line, f"analysis query row {index}")
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadSpecError(str(exc)) from exc
        if not isinstance(row, Mapping):
            raise WorkloadSpecError(
                f"analysis query row {index} must be an object"
            )
        try:
            _exact_keys(row, _QUERY_KEYS, f"analysis query row {index}")
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadSpecError(str(exc)) from exc
        query_id = row.get("id")
        visits = row.get("maxVisits")
        if (
            not isinstance(query_id, str)
            or not query_id
            or "\x00" in query_id
            or query_id in seen_ids
        ):
            raise WorkloadSpecError(
                "analysis query IDs must be unique nonempty strings"
            )
        if type(visits) is not int or not 1 <= visits <= max_visits:
            raise WorkloadSpecError(
                f"analysis query {query_id!r} maxVisits is outside "
                f"the bound 1..{max_visits}"
            )
        for field in ("boardXSize", "boardYSize"):
            if type(row[field]) is not int or not 1 <= row[field] <= 25:
                raise WorkloadSpecError(
                    f"analysis query {query_id!r} {field} is invalid"
                )
        if (
            row["initialPlayer"] not in {"B", "W"}
            or not isinstance(row["rules"], str)
            or not row["rules"]
            or isinstance(row["komi"], bool)
            or not isinstance(row["komi"], (int, float))
            or not math.isfinite(float(row["komi"]))
            or row["includePolicy"] is not True
        ):
            raise WorkloadSpecError(
                f"analysis query {query_id!r} is not one normal position request"
            )
        for field in ("moves", "initialStones"):
            records = row[field]
            if not isinstance(records, list) or len(records) > 10_000:
                raise WorkloadSpecError(
                    f"analysis query {query_id!r} {field} is invalid"
                )
            for record in records:
                if (
                    not isinstance(record, list)
                    or len(record) != 2
                    or record[0] not in {"B", "W"}
                    or not isinstance(record[1], str)
                    or not record[1]
                    or len(record[1]) > 32
                    or any(character in record[1] for character in "\x00\r\n")
                ):
                    raise WorkloadSpecError(
                        f"analysis query {query_id!r} {field} entry is invalid"
                    )
        try:
            overrides = _exact_keys(
                row["overrideSettings"],
                _QUERY_OVERRIDE_KEYS,
                f"analysis query {query_id!r} overrides",
            )
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadSpecError(str(exc)) from exc
        if (
            type(overrides["useScoreMaximizingUtility"]) is not bool
            or overrides["rootNoiseEnabled"] is not False
            or overrides["rootNumSymmetriesToSample"] != 1
            or isinstance(
                overrides["rootNumSymmetriesToSample"], bool
            )
        ):
            raise WorkloadSpecError(
                f"analysis query {query_id!r} overrides are nondeterministic"
            )
        for field in ("scorePower", "scoreScale", "winWeight"):
            value = overrides[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise WorkloadSpecError(
                    f"analysis query {query_id!r} override {field} is invalid"
                )
        seen_ids.add(query_id)
        rows.append(dict(row))
    if len(rows) < row_count:
        raise WorkloadSpecError(
            "analysis query file has fewer rows than the frozen prefix"
        )
    return tuple(rows[:row_count])


def _load_workload_spec(
    path: Path,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> WorkloadSpec:
    try:
        source = _canonical_absolute(path, "workload specification")
        raw = _load_canonical_object(source, "workload specification")
        _exact_keys(raw, _SPEC_KEYS, "workload specification")
    except EvaluatorBenchmarkWorkloadError as exc:
        raise WorkloadSpecError(str(exc)) from exc
    if (
        raw["schema_version"] != SCHEMA_VERSION
        or isinstance(raw["schema_version"], bool)
        or raw["contract"] != WORKLOAD_SPEC_CONTRACT
    ):
        raise WorkloadSpecError(
            "workload specification contract is unsupported"
        )
    body = dict(raw)
    identity = _require_sha256(
        body.pop("spec_sha256", None), "workload specification identity"
    )
    if identity != canonical_sha256(body):
        raise WorkloadSpecError(
            "workload specification self-hash is invalid"
        )
    spec_data = _read_stable_regular_file(
        source,
        "workload specification",
        maximum_bytes=MAX_SPEC_BYTES,
        require_single_link=True,
    )
    file_identity = hashlib.sha256(spec_data).hexdigest()
    if expected_spec_sha256 is not None:
        expected = _require_sha256(
            expected_spec_sha256,
            "expected workload specification hash",
        )
        if expected not in {file_identity, identity}:
            raise WorkloadSpecError(
                "workload specification does not match expected SHA-256"
            )

    try:
        katago = _binding(
            raw["katago"],
            "KataGo binary",
            maximum_bytes=MAX_EXECUTABLE_BYTES,
            executable=True,
        )
        config = _binding(
            raw["analysis_config"],
            "analysis config",
            maximum_bytes=MAX_CONFIG_BYTES,
        )
        model = _binding(
            raw["model"],
            "analysis model",
            maximum_bytes=MAX_MODEL_BYTES,
        )
        nvidia_smi = _binding(
            raw["nvidia_smi"],
            "nvidia-smi binary",
            maximum_bytes=MAX_EXECUTABLE_BYTES,
            executable=True,
        )
        query_raw = _exact_keys(
            raw["queries"], _QUERY_BINDING_KEYS, "analysis queries"
        )
        queries = FileBinding(
            _canonical_absolute(
                query_raw["path"], "analysis queries path"
            ),
            _require_sha256(
                query_raw["sha256"], "analysis queries sha256"
            ),
            MAX_QUERY_BYTES,
        )
        queries.verify("analysis queries")
        row_count = _positive_int(
            query_raw["row_count"],
            "analysis query row_count",
            maximum=MAX_QUERY_ROWS,
        )
        if row_count < max(PROCESS_COUNTS):
            raise WorkloadSpecError(
                "analysis query row_count must cover the largest topology"
            )
        max_visits = _positive_int(
            query_raw["max_visits"],
            "analysis query max_visits",
            maximum=MAX_VISITS,
        )
        gpu = _exact_keys(raw["gpu"], _GPU_KEYS, "workload GPU")
        gpu_index = gpu["index"]
        if (
            type(gpu_index) is not int
            or not 0 <= gpu_index <= 1024
        ):
            raise WorkloadSpecError(
                "workload GPU index must be an integer between 0 and 1024"
            )
        expected_gpu_uuid = gpu["uuid"]
        if (
            not isinstance(expected_gpu_uuid, str)
            or _GPU_UUID_RE.fullmatch(expected_gpu_uuid) is None
        ):
            raise WorkloadSpecError("workload GPU UUID is malformed")
        timeout_seconds = _positive_number(
            raw["timeout_seconds"],
            "workload timeout_seconds",
            maximum=MAX_TIMEOUT_SECONDS,
        )
    except EvaluatorBenchmarkWorkloadError as exc:
        if isinstance(exc, WorkloadSpecError):
            raise
        raise WorkloadSpecError(str(exc)) from exc

    paths = (
        katago.path,
        config.path,
        model.path,
        queries.path,
        nvidia_smi.path,
        source,
    )
    if len(set(paths)) != len(paths):
        raise WorkloadSpecError(
            "workload specification and bound input paths must be distinct"
        )
    _validate_deterministic_config(config.path)
    _load_query_rows(
        queries.path,
        row_count=row_count,
        max_visits=max_visits,
    )
    return WorkloadSpec(
        path=source,
        file_sha256=file_identity,
        identity=identity,
        raw=raw,
        katago=katago,
        analysis_config=config,
        model=model,
        queries=queries,
        nvidia_smi=nvidia_smi,
        gpu_index=gpu_index,
        expected_gpu_uuid=expected_gpu_uuid,
        row_count=row_count,
        max_visits=max_visits,
        timeout_seconds=timeout_seconds,
    )


def load_workload_spec(
    path: Path,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> WorkloadSpec:
    """Load and verify a canonical workload spec and every bound input."""

    return _load_workload_spec(
        Path(path), expected_spec_sha256=expected_spec_sha256
    )


def _directory_identity(path: Path, role: str) -> Tuple[int, int, int]:
    _reject_symlink_ancestors(path, role)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} is missing: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EvaluatorBenchmarkWorkloadError(
            f"{role} must be a non-symlink directory"
        )
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(os.fspath(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory_chain(root: Path, target: Path) -> None:
    if target != root and not _strictly_within(target, root):
        raise WorkloadConflictError(
            f"output directory escapes work_root: {target}"
        )
    _directory_identity(root, "work_root")
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _directory_identity(current, f"output directory {current}")


def _recover_owned_publication_link(path: Path) -> None:
    if not _lexists(path):
        return
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise WorkloadConflictError(
            f"published output is unsafe: {path}"
        )
    if metadata.st_nlink == 1:
        return
    if metadata.st_nlink != 2:
        raise WorkloadConflictError(
            f"published output has unexpected hard links: {path}"
        )
    candidates = []
    for candidate in path.parent.glob(f".{path.name}.*.tmp"):
        candidate_metadata = candidate.lstat()
        if (
            stat.S_ISREG(candidate_metadata.st_mode)
            and not stat.S_ISLNK(candidate_metadata.st_mode)
            and candidate_metadata.st_dev == metadata.st_dev
            and candidate_metadata.st_ino == metadata.st_ino
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise WorkloadConflictError(
            f"cannot recover owned publication hard link: {path}"
        )
    candidates[0].unlink()
    _fsync_directory(path.parent)
    if path.lstat().st_nlink != 1:
        raise WorkloadConflictError(
            f"publication hard-link recovery did not converge: {path}"
        )


def _publish_immutable_bytes(
    path: Path,
    data: bytes,
    *,
    root: Optional[Path] = None,
    maximum_bytes: int,
) -> None:
    target = Path(path)
    if not data or len(data) > maximum_bytes:
        raise WorkloadConflictError(
            f"immutable output size is invalid: {target}"
        )
    if root is not None:
        if not _strictly_within(target, root):
            raise WorkloadConflictError(
                f"immutable output escapes work_root: {target}"
            )
        _ensure_directory_chain(root, target.parent)
    else:
        _reject_symlink_ancestors(target, f"immutable output {target}")
        _directory_identity(target.parent, "immutable output parent")
    if _lexists(target):
        _recover_owned_publication_link(target)
        existing = _read_stable_regular_file(
            target,
            f"existing immutable output {target}",
            maximum_bytes=maximum_bytes,
            require_single_link=True,
        )
        if existing != data:
            raise WorkloadConflictError(
                f"existing immutable output conflicts: {target}"
            )
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=os.fspath(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(
                os.fspath(temporary),
                os.fspath(target),
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = _read_stable_regular_file(
                target,
                f"existing immutable output {target}",
                maximum_bytes=maximum_bytes,
                require_single_link=True,
            )
            if existing != data:
                raise WorkloadConflictError(
                    f"existing immutable output conflicts: {target}"
                )
        temporary.unlink()
        _fsync_directory(target.parent)
        metadata = target.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise WorkloadConflictError(
                f"published immutable output is unsafe: {target}"
            )
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        with contextlib.suppress(OSError):
            _fsync_directory(target.parent)


def publish_workload_spec(
    path: Path,
    *,
    katago: Path,
    analysis_config: Path,
    model: Path,
    queries: Path,
    gpu_index: int,
    expected_gpu_uuid: str,
    max_visits: int,
    timeout_seconds: float,
    row_count: Optional[int] = None,
    query_count: Optional[int] = None,
    nvidia_smi: Optional[Path] = None,
) -> WorkloadSpec:
    """Publish a canonical immutable workload spec for a provisioner.

    ``query_count`` is accepted as an API alias for ``row_count``.  The stored
    contract always calls the deterministic prefix bound ``row_count``.
    """

    if row_count is None:
        row_count = query_count
    elif query_count is not None and query_count != row_count:
        raise WorkloadSpecError(
            "row_count and query_count aliases disagree"
        )
    if row_count is None:
        raise WorkloadSpecError("row_count is required")
    if nvidia_smi is None:
        discovered = shutil.which("nvidia-smi")
        if discovered is None:
            raise WorkloadSpecError(
                "nvidia-smi was not found; pass its frozen absolute path"
            )
        nvidia_smi = Path(discovered).resolve(strict=True)

    destination = _spec_absolute(path, "workload specification output")
    katago_path = _spec_absolute(katago, "KataGo binary")
    config_path = _spec_absolute(
        analysis_config, "analysis config"
    )
    model_path = _spec_absolute(model, "analysis model")
    query_path = _spec_absolute(queries, "analysis queries")
    nvidia_path = _spec_absolute(nvidia_smi, "nvidia-smi binary")
    bindings = {
        "katago": FileBinding(
            katago_path,
            _stable_file_sha256(
                katago_path,
                "KataGo binary",
                maximum_bytes=MAX_EXECUTABLE_BYTES,
                executable=True,
            ),
            MAX_EXECUTABLE_BYTES,
            True,
        ),
        "analysis_config": FileBinding(
            config_path,
            _stable_file_sha256(
                config_path,
                "analysis config",
                maximum_bytes=MAX_CONFIG_BYTES,
            ),
            MAX_CONFIG_BYTES,
        ),
        "model": FileBinding(
            model_path,
            _stable_file_sha256(
                model_path,
                "analysis model",
                maximum_bytes=MAX_MODEL_BYTES,
            ),
            MAX_MODEL_BYTES,
        ),
        "queries": FileBinding(
            query_path,
            _stable_file_sha256(
                query_path,
                "analysis queries",
                maximum_bytes=MAX_QUERY_BYTES,
            ),
            MAX_QUERY_BYTES,
        ),
        "nvidia_smi": FileBinding(
            nvidia_path,
            _stable_file_sha256(
                nvidia_path,
                "nvidia-smi binary",
                maximum_bytes=MAX_EXECUTABLE_BYTES,
                executable=True,
            ),
            MAX_EXECUTABLE_BYTES,
            True,
        ),
    }
    row_count = _positive_int(
        row_count, "analysis query row_count", maximum=MAX_QUERY_ROWS
    )
    if row_count < max(PROCESS_COUNTS):
        raise WorkloadSpecError(
            "analysis query row_count must cover the largest topology"
        )
    max_visits = _positive_int(
        max_visits, "analysis query max_visits", maximum=MAX_VISITS
    )
    timeout_seconds = _positive_number(
        timeout_seconds,
        "workload timeout_seconds",
        maximum=MAX_TIMEOUT_SECONDS,
    )
    if (
        type(gpu_index) is not int
        or not 0 <= gpu_index <= 1024
    ):
        raise WorkloadSpecError(
            "workload GPU index must be between 0 and 1024"
        )
    if (
        not isinstance(expected_gpu_uuid, str)
        or _GPU_UUID_RE.fullmatch(expected_gpu_uuid) is None
    ):
        raise WorkloadSpecError("workload GPU UUID is malformed")
    if len({binding.path for binding in bindings.values()}) != len(bindings):
        raise WorkloadSpecError("workload bound input paths must be distinct")
    if destination in {binding.path for binding in bindings.values()}:
        raise WorkloadSpecError(
            "workload specification output collides with a bound input"
        )
    _validate_deterministic_config(config_path)
    _load_query_rows(
        query_path,
        row_count=row_count,
        max_visits=max_visits,
    )

    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": WORKLOAD_SPEC_CONTRACT,
        "katago": bindings["katago"].spec_value(),
        "analysis_config": bindings["analysis_config"].spec_value(),
        "model": bindings["model"].spec_value(),
        "queries": {
            **bindings["queries"].spec_value(),
            "row_count": row_count,
            "max_visits": max_visits,
        },
        "nvidia_smi": bindings["nvidia_smi"].spec_value(),
        "gpu": {"index": gpu_index, "uuid": expected_gpu_uuid},
        "timeout_seconds": timeout_seconds,
    }
    value["spec_sha256"] = canonical_sha256(value)
    data = (canonical_json(value) + "\n").encode("utf-8")
    _publish_immutable_bytes(
        destination,
        data,
        maximum_bytes=MAX_SPEC_BYTES,
    )
    return load_workload_spec(
        destination,
        expected_spec_sha256=hashlib.sha256(data).hexdigest(),
    )


def build_benchmark_argv_template(
    spec: WorkloadSpec | Path,
    *,
    python_executable: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
) -> Tuple[str, ...]:
    """Build the immutable argv template consumed by the generic runner."""

    loaded = (
        spec
        if isinstance(spec, WorkloadSpec)
        else load_workload_spec(Path(spec))
    )
    python_path = _canonical_absolute(
        Path(sys.executable).resolve()
        if python_executable is None
        else python_executable,
        "Python executable",
    )
    adapter = _canonical_absolute(
        Path(__file__).resolve() if adapter_path is None else adapter_path,
        "workload adapter",
    )
    _stable_file_sha256(
        python_path,
        "Python executable",
        maximum_bytes=MAX_EXECUTABLE_BYTES,
        executable=True,
    )
    _stable_file_sha256(
        adapter,
        "workload adapter",
        maximum_bytes=MAX_CONFIG_BYTES,
    )
    return (
        os.fspath(python_path),
        os.fspath(adapter),
        "--spec",
        os.fspath(loaded.path),
        "--expected-spec-sha256",
        loaded.file_sha256,
        "--process-count",
        "{process_count}",
        "--output",
        "{output}",
        "--work-root",
        "{work_root}",
    )


def benchmark_input_bindings(
    spec: WorkloadSpec | Path,
    *,
    python_executable: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
) -> Tuple[Mapping[str, str], ...]:
    """Return sorted bindings suitable for the generic runner's ``inputs``."""

    loaded = (
        spec
        if isinstance(spec, WorkloadSpec)
        else load_workload_spec(Path(spec))
    )
    python_path = _canonical_absolute(
        Path(sys.executable).resolve()
        if python_executable is None
        else python_executable,
        "Python executable",
    )
    adapter = _canonical_absolute(
        Path(__file__).resolve() if adapter_path is None else adapter_path,
        "workload adapter",
    )
    paths = {
        python_path,
        adapter,
        loaded.path,
        loaded.katago.path,
        loaded.analysis_config.path,
        loaded.model.path,
        loaded.queries.path,
        loaded.nvidia_smi.path,
    }
    bindings = [
        {"path": os.fspath(path), "sha256": file_sha256(path)}
        for path in paths
    ]
    return tuple(sorted(bindings, key=lambda item: item["path"]))


def _publisher_bound_path(
    value: Any,
    role: str,
    *,
    maximum_bytes: int,
) -> Path:
    if isinstance(value, Mapping):
        binding = _exact_keys(value, _BINDING_KEYS, role)
        path = _spec_absolute(binding["path"], f"{role} path")
        expected = _require_sha256(binding["sha256"], f"{role} sha256")
        observed = _stable_file_sha256(
            path, role, maximum_bytes=maximum_bytes
        )
        if observed != expected:
            raise WorkloadSpecError(f"{role} hash binding changed")
        return path
    return _spec_absolute(value, role)


def publish_topology_specs(
    *,
    workload_spec_path: Path,
    benchmark_spec_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    python_executable: Optional[Path] = None,
    katago_binary: Optional[Path] = None,
    models: Optional[Sequence[Any]] = None,
    model_probe_config: Optional[Path] = None,
    process_counts: Sequence[int] = PROCESS_COUNTS,
    work_root: Optional[Path] = None,
    evidence_output: Optional[Path] = None,
    timeout_seconds: Optional[float] = None,
    **options: Any,
) -> Mapping[str, Any]:
    """Publish both specs expected by ``autonomy_provisioner``.

    The provisioner supplies deployment-wide coordinates directly and carries
    workload-specific coordinates in ``topology_benchmark.publisher_options``.
    Required workload options are the fixed query JSONL, row/visit bounds, GPU
    index and UUID, and (when more than one model is supplied) an explicit
    model path or SHA-256 selector.
    """

    def take(*names: str, default: Any = None) -> Any:
        found = [name for name in names if name in options]
        if len(found) > 1:
            values = [options.pop(name) for name in found]
            if any(value != values[0] for value in values[1:]):
                raise WorkloadSpecError(
                    f"publisher aliases disagree: {found}"
                )
            return values[0]
        if found:
            return options.pop(found[0])
        return default

    benchmark_destination_value = (
        benchmark_spec_path
        if benchmark_spec_path is not None
        else output_path
    )
    if benchmark_destination_value is None:
        raise WorkloadSpecError("benchmark_spec_path is required")
    if (
        benchmark_spec_path is not None
        and output_path is not None
        and Path(benchmark_spec_path) != Path(output_path)
    ):
        raise WorkloadSpecError(
            "benchmark_spec_path and output_path disagree"
        )
    if tuple(process_counts) != PROCESS_COUNTS:
        raise WorkloadSpecError(
            "topology publisher process_counts must be exactly 4, 8, 16"
        )
    if work_root is None or evidence_output is None or timeout_seconds is None:
        raise WorkloadSpecError(
            "work_root, evidence_output, and timeout_seconds are required"
        )

    query_value = take(
        "queries",
        "query_jsonl",
        "fixed_query_jsonl",
        "query_path",
    )
    if query_value is None:
        raise WorkloadSpecError(
            "topology publisher requires a fixed query JSONL"
        )
    config_value = take(
        "analysis_config",
        "config",
        default=model_probe_config,
    )
    if config_value is None:
        raise WorkloadSpecError(
            "topology publisher requires an analysis config"
        )
    katago_value = take("katago", default=katago_binary)
    if katago_value is None:
        raise WorkloadSpecError(
            "topology publisher requires the KataGo binary"
        )
    nvidia_value = take(
        "nvidia_smi",
        "nvidia_smi_path",
        default=None,
    )
    gpu_index = take("gpu_index", default=None)
    expected_gpu_uuid = take(
        "expected_gpu_uuid", "gpu_uuid", default=None
    )
    gpu_value = take("gpu", default=None)
    if gpu_value is not None:
        gpu = _exact_keys(gpu_value, _GPU_KEYS, "topology publisher GPU")
        if gpu_index is not None and gpu_index != gpu["index"]:
            raise WorkloadSpecError(
                "gpu.index contradicts the explicit gpu_index"
            )
        if (
            expected_gpu_uuid is not None
            and expected_gpu_uuid != gpu["uuid"]
        ):
            raise WorkloadSpecError(
                "gpu.uuid contradicts the explicit expected_gpu_uuid"
            )
        gpu_index = gpu["index"]
        expected_gpu_uuid = gpu["uuid"]
    row_count = take("row_count", "query_count", default=None)
    max_visits = take(
        "max_visits", "maximum_visits", default=None
    )
    adapter_path = take("adapter_path", default=None)
    workload_timeout = take(
        "workload_timeout_seconds",
        "child_timeout_seconds",
        default=None,
    )
    model_value = take(
        "model",
        "analysis_model",
        "reference_model",
        default=None,
    )
    model_sha256 = take(
        "model_sha256",
        "analysis_model_sha256",
        default=None,
    )
    model_index = take("model_index", default=None)

    # These are useful provisioner coordinates but are not mutable inputs to
    # this publisher.  Accepting them explicitly prevents accidental treatment
    # as workload options while retaining a strict unknown-option check.
    for ignored in ("repository", "run_root", "suite_manifest"):
        options.pop(ignored, None)
    if options:
        raise WorkloadSpecError(
            "unsupported topology publisher options: "
            + ", ".join(sorted(options))
        )

    candidate_models = tuple(models or ())
    if model_value is None:
        if model_sha256 is not None:
            expected_model_hash = _require_sha256(
                model_sha256, "topology model selector"
            )
            matches = [
                candidate
                for candidate in candidate_models
                if isinstance(candidate, Mapping)
                and candidate.get("sha256") == expected_model_hash
            ]
            if len(matches) != 1:
                raise WorkloadSpecError(
                    "topology model SHA-256 does not select exactly one model"
                )
            model_value = matches[0]
        elif model_index is not None:
            if (
                type(model_index) is not int
                or not 0 <= model_index < len(candidate_models)
            ):
                raise WorkloadSpecError(
                    "topology model_index is outside the model inventory"
                )
            model_value = candidate_models[model_index]
        elif len(candidate_models) == 1:
            model_value = candidate_models[0]
        else:
            raise WorkloadSpecError(
                "topology publisher must explicitly select one analysis model"
            )
    model_path = _publisher_bound_path(
        model_value,
        "topology analysis model",
        maximum_bytes=MAX_MODEL_BYTES,
    )
    if candidate_models:
        inventory_paths = {
            _publisher_bound_path(
                candidate,
                f"topology model inventory item {index}",
                maximum_bytes=MAX_MODEL_BYTES,
            )
            for index, candidate in enumerate(candidate_models)
        }
        if model_path not in inventory_paths:
            raise WorkloadSpecError(
                "selected topology model is absent from the frozen inventory"
            )

    outer_timeout = _positive_number(
        timeout_seconds,
        "topology benchmark timeout_seconds",
        maximum=MAX_TIMEOUT_SECONDS,
    )
    if workload_timeout is None:
        workload_timeout = outer_timeout * 0.9
    workload_timeout = _positive_number(
        workload_timeout,
        "topology workload timeout_seconds",
        maximum=MAX_TIMEOUT_SECONDS,
    )
    if workload_timeout >= outer_timeout:
        raise WorkloadSpecError(
            "workload timeout must be less than the generic command timeout"
        )
    if row_count is None or max_visits is None:
        raise WorkloadSpecError(
            "topology publisher requires row_count and max_visits"
        )
    if gpu_index is None or expected_gpu_uuid is None:
        raise WorkloadSpecError(
            "topology publisher requires gpu_index and expected_gpu_uuid"
        )

    workload_destination = _spec_absolute(
        workload_spec_path, "workload specification output"
    )
    benchmark_destination = _spec_absolute(
        benchmark_destination_value,
        "topology benchmark specification output",
    )
    if workload_destination == benchmark_destination:
        raise WorkloadSpecError(
            "workload and generic benchmark specs must be distinct"
        )
    outer_root = _spec_absolute(work_root, "topology benchmark work_root")
    evidence_path = _spec_absolute(
        evidence_output, "topology benchmark evidence_output"
    )
    if not _strictly_within(evidence_path, outer_root):
        raise WorkloadSpecError(
            "topology evidence_output must be beneath work_root"
        )
    if any(
        path == outer_root or _strictly_within(path, outer_root)
        for path in (workload_destination, benchmark_destination)
    ):
        raise WorkloadSpecError(
            "topology specifications may not be stored beneath work_root"
        )

    workload = publish_workload_spec(
        workload_destination,
        katago=_publisher_bound_path(
            katago_value,
            "topology KataGo binary",
            maximum_bytes=MAX_EXECUTABLE_BYTES,
        ),
        analysis_config=_publisher_bound_path(
            config_value,
            "topology analysis config",
            maximum_bytes=MAX_CONFIG_BYTES,
        ),
        model=model_path,
        queries=_publisher_bound_path(
            query_value,
            "topology fixed queries",
            maximum_bytes=MAX_QUERY_BYTES,
        ),
        nvidia_smi=(
            None
            if nvidia_value is None
            else _publisher_bound_path(
                nvidia_value,
                "topology nvidia-smi binary",
                maximum_bytes=MAX_EXECUTABLE_BYTES,
            )
        ),
        gpu_index=gpu_index,
        expected_gpu_uuid=expected_gpu_uuid,
        row_count=row_count,
        max_visits=max_visits,
        timeout_seconds=workload_timeout,
    )
    argv = build_benchmark_argv_template(
        workload,
        python_executable=python_executable,
        adapter_path=adapter_path,
    )
    inputs = benchmark_input_bindings(
        workload,
        python_executable=python_executable,
        adapter_path=adapter_path,
    )
    benchmark: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": "risk-score-evaluator-topology-benchmark-spec-v1",
        "benchmark_argv_template": list(argv),
        "inputs": list(inputs),
        "timeout_seconds": outer_timeout,
        "work_root": os.fspath(outer_root),
        "evidence_output": os.fspath(evidence_path),
    }
    benchmark["spec_sha256"] = canonical_sha256(benchmark)
    benchmark_data = (canonical_json(benchmark) + "\n").encode("utf-8")
    _publish_immutable_bytes(
        benchmark_destination,
        benchmark_data,
        maximum_bytes=MAX_SPEC_BYTES,
    )
    return {
        "workload_spec": {
            "path": os.fspath(workload.path),
            "sha256": workload.file_sha256,
            "spec_sha256": workload.identity,
        },
        "benchmark_spec": {
            "path": os.fspath(benchmark_destination),
            "sha256": hashlib.sha256(benchmark_data).hexdigest(),
            "spec_sha256": benchmark["spec_sha256"],
        },
        "benchmark_argv_template": list(argv),
    }


def _canonical_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    chunks = []
    total = 0
    try:
        for row in rows:
            chunk = (canonical_json(row) + "\n").encode("utf-8")
            total += len(chunk)
            if total > MAX_TOTAL_CHILD_OUTPUT_BYTES:
                raise WorkloadExecutionError(
                    "canonical merged analysis output exceeds its size limit"
                )
            chunks.append(chunk)
    except (TypeError, ValueError, RecursionError) as exc:
        raise WorkloadExecutionError(
            "analysis output cannot be represented canonically"
        ) from exc
    if not chunks:
        raise WorkloadExecutionError("canonical analysis output is empty")
    return b"".join(chunks)


def _write_new_file(path: Path, data: bytes) -> None:
    if not data:
        raise WorkloadExecutionError(f"refusing to write an empty file: {path}")
    try:
        with path.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise WorkloadConflictError(
            f"forensic attempt path already exists: {path}"
        ) from exc


def _run_probe_command(
    binding: FileBinding,
    arguments: Sequence[str],
    *,
    role: str,
    timeout_seconds: float = GPU_PROBE_TIMEOUT_SECONDS,
) -> str:
    argv = (os.fspath(binding.path), *arguments)
    if (
        len(argv) > 16
        or any(
            not isinstance(part, str)
            or not part
            or "\x00" in part
            or len(os.fsencode(part)) > MAX_PATH_BYTES
            for part in argv
        )
    ):
        raise WorkloadExecutionError(f"{role} argv is unsafe")
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            close_fds=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkloadExecutionError(f"{role} timed out") from exc
    except OSError as exc:
        raise WorkloadExecutionError(
            f"{role} could not be started: {exc}"
        ) from exc
    if (
        len(result.stdout) > MAX_GPU_PROBE_BYTES
        or len(result.stderr) > MAX_GPU_PROBE_BYTES
    ):
        raise WorkloadExecutionError(f"{role} output is excessive")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise WorkloadExecutionError(
            f"{role} failed with status {result.returncode}{suffix}"
        )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkloadExecutionError(
            f"{role} output is not UTF-8"
        ) from exc


def _relative_path(path: Path, root: Path, role: str) -> str:
    if not _strictly_within(path, root):
        raise WorkloadExecutionError(f"{role} escapes work_root")
    relative = path.relative_to(root)
    text = relative.as_posix()
    if (
        not text
        or text == "."
        or "\\" in text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise WorkloadExecutionError(f"{role} is not a safe relative path")
    return text


def _inventory_tree(root: Path) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    directories = []
    files = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        _directory_identity(current, f"workload directory {current}")
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
                metadata.st_mode
            ):
                raise WorkloadExecutionError(
                    f"workload tree contains an unsafe directory: {child}"
                )
            directories.append(child.relative_to(root).as_posix())
        for name in file_names:
            child = current / name
            metadata = child.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise WorkloadExecutionError(
                    f"workload tree contains an unsafe file: {child}"
                )
            files.append(child.relative_to(root).as_posix())
    return tuple(sorted(directories)), tuple(sorted(files))


@dataclass(frozen=True)
class _Child:
    index: int
    process: subprocess.Popen[Any]
    query_path: Path
    response_path: Path
    stderr_path: Path
    query_rows: Tuple[Mapping[str, Any], ...]


class EvaluatorBenchmarkWorkload:
    """Fail-closed executor for one 4/8/16-process benchmark invocation."""

    def __init__(
        self,
        spec: WorkloadSpec | Path,
        *,
        process_count: int,
        output: Path,
        work_root: Path,
        expected_spec_sha256: Optional[str] = None,
    ) -> None:
        if isinstance(spec, WorkloadSpec):
            if expected_spec_sha256 is not None:
                expected = _require_sha256(
                    expected_spec_sha256,
                    "expected workload specification hash",
                )
                if expected not in {spec.file_sha256, spec.identity}:
                    raise WorkloadSpecError(
                        "loaded workload specification does not match "
                        "expected SHA-256"
                    )
            self.spec = spec
        else:
            self.spec = load_workload_spec(
                Path(spec),
                expected_spec_sha256=expected_spec_sha256,
            )
        if type(process_count) is not int or process_count not in PROCESS_COUNTS:
            raise WorkloadSpecError(
                "process_count must be one of 4, 8, or 16"
            )
        self.process_count = process_count
        try:
            self.work_root = _canonical_absolute(work_root, "work_root")
            self.output = _canonical_absolute(output, "receipt output")
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadSpecError(str(exc)) from exc
        if self.work_root == Path(self.work_root.anchor):
            raise WorkloadSpecError("work_root may not be a filesystem root")
        try:
            self._root_identity = _directory_identity(
                self.work_root, "work_root"
            )
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadSpecError(str(exc)) from exc
        if not _strictly_within(self.output, self.work_root):
            raise WorkloadSpecError(
                "receipt output must be strictly beneath work_root"
            )
        self.artifact = self.work_root.joinpath(*_ARTIFACT_RELATIVE.parts)
        attempt_path = self.work_root / _ATTEMPT_NAME
        if (
            self.output == self.artifact
            or _strictly_within(self.output, self.artifact)
            or _strictly_within(self.artifact, self.output)
            or self.output == attempt_path
            or _strictly_within(self.output, attempt_path)
        ):
            raise WorkloadSpecError(
                "receipt output collides with a reserved workload path"
            )
        frozen_paths = (
            self.spec.path,
            self.spec.katago.path,
            self.spec.analysis_config.path,
            self.spec.model.path,
            self.spec.queries.path,
            self.spec.nvidia_smi.path,
        )
        if any(
            path == self.work_root
            or _strictly_within(path, self.work_root)
            for path in frozen_paths
        ):
            raise WorkloadSpecError(
                "work_root may not contain the spec or a bound input"
            )

    def _assert_root(self) -> None:
        if (
            _directory_identity(self.work_root, "work_root")
            != self._root_identity
        ):
            raise WorkloadExecutionError(
                "workload replaced its work_root"
            )

    def _verify_gpu(
        self,
        allowed_pids: Optional[set[int]] = None,
        *,
        require_all: bool = False,
        deadline: Optional[float] = None,
    ) -> set[int]:
        def probe_timeout() -> float:
            if deadline is None:
                return GPU_PROBE_TIMEOUT_SECONDS
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkloadExecutionError(
                    "GPU attribution timeout deadline expired"
                )
            return min(GPU_PROBE_TIMEOUT_SECONDS, remaining)

        inventory = _run_probe_command(
            self.spec.nvidia_smi,
            (
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ),
            role="CUDA GPU inventory",
            timeout_seconds=probe_timeout(),
        )
        by_index: Dict[int, str] = {}
        seen_uuids = set()
        for raw_line in inventory.splitlines():
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",", 1)]
            try:
                index = int(fields[0]) if len(fields) == 2 else -1
            except ValueError:
                index = -1
            uuid = fields[1] if len(fields) == 2 else ""
            if _MIG_UUID_RE.fullmatch(uuid) is not None:
                if index == self.spec.gpu_index:
                    raise WorkloadExecutionError(
                        "MIG inventory ambiguously overlaps the target GPU index"
                    )
                continue
            if (
                index < 0
                or index in by_index
                or _GPU_UUID_RE.fullmatch(uuid) is None
                or uuid in seen_uuids
            ):
                raise WorkloadExecutionError(
                    "CUDA GPU inventory is malformed"
                )
            by_index[index] = uuid
            seen_uuids.add(uuid)
        if not by_index:
            raise WorkloadExecutionError("CUDA GPU inventory is empty")
        observed = by_index.get(self.spec.gpu_index)
        if observed != self.spec.expected_gpu_uuid:
            raise WorkloadExecutionError(
                "requested CUDA GPU UUID differs from the frozen UUID"
            )

        topology = _run_probe_command(
            self.spec.nvidia_smi,
            ("-L",),
            role="CUDA MIG topology inventory",
            timeout_seconds=probe_timeout(),
        )
        listed_by_index: Dict[int, str] = {}
        mig_parents: Dict[str, str] = {}
        mig_devices = set()
        current_parent: Optional[str] = None
        for raw_line in topology.splitlines():
            if not raw_line.strip():
                continue
            gpu_match = _NVIDIA_L_GPU_RE.fullmatch(raw_line)
            if gpu_match is not None:
                gpu_index = int(gpu_match.group(1))
                gpu_uuid = gpu_match.group(2)
                if (
                    gpu_index in listed_by_index
                    or gpu_uuid in listed_by_index.values()
                ):
                    raise WorkloadExecutionError(
                        "CUDA MIG topology has duplicate physical GPUs"
                    )
                listed_by_index[gpu_index] = gpu_uuid
                current_parent = gpu_uuid
                continue
            mig_match = _NVIDIA_L_MIG_RE.fullmatch(raw_line)
            if mig_match is None or current_parent is None:
                raise WorkloadExecutionError(
                    "CUDA MIG topology inventory is malformed"
                )
            device_index = int(mig_match.group(1))
            mig_uuid = mig_match.group(2)
            device_identity = (current_parent, device_index)
            if (
                mig_uuid in mig_parents
                or device_identity in mig_devices
            ):
                raise WorkloadExecutionError(
                    "CUDA MIG topology maps one MIG UUID more than once "
                    "or repeats a device index"
                )
            mig_parents[mig_uuid] = current_parent
            mig_devices.add(device_identity)
        if listed_by_index != by_index:
            raise WorkloadExecutionError(
                "CUDA MIG topology contradicts the physical GPU inventory"
            )

        processes = _run_probe_command(
            self.spec.nvidia_smi,
            (
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ),
            role="CUDA process inventory",
            timeout_seconds=probe_timeout(),
        )
        permitted = set() if allowed_pids is None else set(allowed_pids)
        observed_target_pids = set()
        for raw_line in processes.splitlines():
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",", 2)]
            try:
                pid = int(fields[1]) if len(fields) == 3 else -1
            except ValueError:
                pid = -1
            if (
                len(fields) != 3
                or (
                    _GPU_UUID_RE.fullmatch(fields[0]) is None
                    and _MIG_UUID_RE.fullmatch(fields[0]) is None
                )
                or pid <= 0
                or not fields[2]
            ):
                raise WorkloadExecutionError(
                    "CUDA process inventory is malformed"
                )
            if _MIG_UUID_RE.fullmatch(fields[0]) is not None:
                if pid in permitted:
                    raise WorkloadExecutionError(
                        "an evaluator child was attributed to an unexpected MIG GPU"
                    )
                mapped_parent = mig_parents.get(fields[0])
                embedded_matches = [
                    gpu_uuid
                    for gpu_uuid in seen_uuids
                    if fields[0].startswith(f"MIG-{gpu_uuid}/")
                ]
                if len(embedded_matches) > 1:
                    raise WorkloadExecutionError(
                        "MIG process row cannot be attributed unambiguously "
                        "to one physical GPU"
                    )
                embedded_parent = (
                    embedded_matches[0] if embedded_matches else None
                )
                if (
                    mapped_parent is not None
                    and embedded_parent is not None
                    and mapped_parent != embedded_parent
                ):
                    raise WorkloadExecutionError(
                        "MIG topology mapping contradicts the process UUID"
                    )
                parent = (
                    mapped_parent
                    if mapped_parent is not None
                    else embedded_parent
                )
                if parent is None:
                    raise WorkloadExecutionError(
                        "MIG process row has no unambiguous physical mapping"
                    )
                if parent == observed:
                    raise WorkloadExecutionError(
                        "target CUDA GPU has an unexpected MIG compute process"
                    )
                continue
            if pid in permitted and fields[0] != observed:
                raise WorkloadExecutionError(
                    "an evaluator child appeared on another CUDA GPU"
                )
            if fields[0] == observed and pid not in permitted:
                raise WorkloadExecutionError(
                    "target CUDA GPU has a foreign compute process"
                )
            if fields[0] == observed:
                observed_target_pids.add(pid)
        if require_all and observed_target_pids != permitted:
            missing = sorted(permitted.difference(observed_target_pids))
            raise WorkloadExecutionError(
                "not every live evaluator child was attributed to the "
                f"expected GPU; missing={missing}"
            )
        return observed_target_pids

    def _load_replay(
        self, query_rows: Tuple[Mapping[str, Any], ...]
    ) -> Optional[Mapping[str, Any]]:
        if not _lexists(self.output):
            return None
        try:
            _recover_owned_publication_link(self.output)
            _recover_owned_publication_link(self.artifact)
            receipt = _load_canonical_object(
                self.output,
                "benchmark workload receipt",
                maximum_bytes=MAX_SPEC_BYTES,
            )
            _exact_keys(
                receipt, _RECEIPT_KEYS, "benchmark workload receipt"
            )
            body = dict(receipt)
            supplied = _require_sha256(
                body.pop("receipt_sha256", None),
                "benchmark workload receipt identity",
            )
            if supplied != canonical_sha256(body):
                raise WorkloadConflictError(
                    "benchmark workload receipt self-hash is invalid"
                )
            if (
                receipt["schema_version"] != SCHEMA_VERSION
                or isinstance(receipt["schema_version"], bool)
                or receipt["contract"] != BENCHMARK_RECEIPT_CONTRACT
                or receipt["process_count"] != self.process_count
                or isinstance(receipt["process_count"], bool)
                or receipt["completed_work_count"] != self.spec.row_count
                or isinstance(receipt["completed_work_count"], bool)
            ):
                raise WorkloadConflictError(
                    "benchmark workload receipt invocation is invalid"
                )
            elapsed = receipt["elapsed_seconds"]
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or not 0 < float(elapsed) <= self.spec.timeout_seconds
            ):
                raise WorkloadConflictError(
                    "benchmark workload receipt elapsed time is invalid"
                )
            manifest = receipt["output_manifest"]
            if (
                not isinstance(manifest, list)
                or len(manifest) != 1
                or receipt["output_manifest_sha256"]
                != canonical_sha256(manifest)
            ):
                raise WorkloadConflictError(
                    "benchmark workload output manifest is invalid"
                )
            artifact = _exact_keys(
                manifest[0], _ARTIFACT_KEYS, "benchmark output artifact"
            )
            if (
                artifact["path"] != _ARTIFACT_RELATIVE.as_posix()
                or artifact["row_count"] != self.spec.row_count
                or isinstance(artifact["row_count"], bool)
                or type(artifact["size_bytes"]) is not int
                or artifact["size_bytes"] <= 0
            ):
                raise WorkloadConflictError(
                    "benchmark output artifact declaration is invalid"
                )
            expected_hash = _require_sha256(
                artifact["sha256"], "benchmark output artifact sha256"
            )
            artifact_data = _read_stable_regular_file(
                self.artifact,
                "benchmark output artifact",
                maximum_bytes=MAX_TOTAL_CHILD_OUTPUT_BYTES,
                require_single_link=True,
            )
            if (
                len(artifact_data) != artifact["size_bytes"]
                or hashlib.sha256(artifact_data).hexdigest() != expected_hash
            ):
                raise WorkloadConflictError(
                    "benchmark output artifact bytes are invalid"
                )
            responses = self._responses_by_id(
                self.artifact,
                query_rows,
                role="benchmark output artifact",
                maximum_bytes=MAX_TOTAL_CHILD_OUTPUT_BYTES,
            )
            canonical = _canonical_jsonl(
                responses[str(row["id"])] for row in query_rows
            )
            if canonical != artifact_data:
                raise WorkloadConflictError(
                    "benchmark artifact is not in original query order"
                )
            _, files = _inventory_tree(self.work_root)
            expected_files = tuple(
                sorted(
                    (
                        _ARTIFACT_RELATIVE.as_posix(),
                        self.output.relative_to(self.work_root).as_posix(),
                    )
                )
            )
            if files != expected_files:
                raise WorkloadConflictError(
                    "completed workload tree has undeclared files"
                )
            return receipt
        except WorkloadConflictError:
            raise
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadConflictError(str(exc)) from exc

    def _require_fresh_root(self) -> None:
        try:
            entry = next(self.work_root.iterdir(), None)
        except OSError as exc:
            raise WorkloadConflictError(
                f"cannot inspect work_root: {exc}"
            ) from exc
        if entry is not None:
            raise WorkloadConflictError(
                "work_root contains an incomplete or conflicting prior attempt"
            )

    def _write_shards(
        self,
        attempt: Path,
        query_rows: Tuple[Mapping[str, Any], ...],
    ) -> Tuple[Tuple[Path, Tuple[Mapping[str, Any], ...]], ...]:
        query_dir = attempt / "queries"
        response_dir = attempt / "responses"
        stderr_dir = attempt / "stderr"
        manifest_dir = attempt / "manifests"
        for directory in (
            query_dir,
            response_dir,
            stderr_dir,
            manifest_dir,
        ):
            directory.mkdir(mode=0o700)
            _directory_identity(directory, f"attempt directory {directory}")
        shards = []
        for index in range(self.process_count):
            rows = tuple(query_rows[index :: self.process_count])
            if not rows:
                raise WorkloadExecutionError(
                    "deterministic query sharding produced an empty shard"
                )
            path = query_dir / f"shard-{index:03d}.jsonl"
            _write_new_file(path, _canonical_jsonl(rows))
            shards.append((path, rows))
        return tuple(shards)

    @staticmethod
    def _kill_process(process: subprocess.Popen[Any]) -> None:
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.kill()

    @staticmethod
    def _process_group_pids() -> set[int]:
        if os.name != "posix":  # pragma: no cover - Unix production.
            return {os.getpid()}
        try:
            result = subprocess.run(
                (
                    "/bin/ps",
                    "-axo",
                    "pid=,ppid=,pgid=,comm=",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=False,
                close_fds=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorkloadExecutionError(
                f"could not inventory the adapter process group: {exc}"
            ) from exc
        if (
            result.returncode != 0
            or len(result.stdout) > MAX_GPU_PROBE_BYTES
        ):
            raise WorkloadExecutionError(
                "could not inventory the adapter process group"
            )
        process_group = os.getpgrp()
        pids = set()
        for raw_line in result.stdout.decode(
            "utf-8", errors="replace"
        ).splitlines():
            fields = raw_line.split(None, 3)
            if len(fields) != 4:
                raise WorkloadExecutionError(
                    "process-group inventory is malformed"
                )
            try:
                pid, parent_pid, pgid = map(int, fields[:3])
            except ValueError as exc:
                raise WorkloadExecutionError(
                    "process-group inventory is malformed"
                ) from exc
            if (
                parent_pid == os.getpid()
                and Path(fields[3]).name == "ps"
            ):
                continue
            if pgid == process_group:
                pids.add(pid)
        if os.getpid() not in pids:
            raise WorkloadExecutionError(
                "process-group inventory omitted the adapter"
            )
        return pids

    def _unexpected_group_pids(self, baseline: set[int]) -> set[int]:
        return self._process_group_pids().difference(baseline).difference(
            {os.getpid()}
        )

    def _kill_children(
        self, children: Sequence[_Child], baseline: set[int]
    ) -> None:
        for child in children:
            self._kill_process(child.process)
        # Descendants inherit the adapter's outer process group.  Snapshotting
        # the pre-launch members lets internal failure clean only processes
        # created by this workload without detaching children from the generic
        # runner's timeout-kill group.
        for _attempt in range(50):
            unexpected = self._unexpected_group_pids(baseline)
            if not unexpected:
                break
            for pid in unexpected:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
            time.sleep(0.02)
        for child in children:
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.process.wait(timeout=5)
        unexpected = self._unexpected_group_pids(baseline)
        if unexpected:
            raise WorkloadExecutionError(
                "could not drain workload descendants from the adapter "
                f"process group: {sorted(unexpected)}"
            )

    @staticmethod
    def _stderr_tail(path: Path, limit: int = 8192) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - limit))
                return stream.read().decode(
                    "utf-8", errors="replace"
                ).strip()
        except OSError:
            return ""

    def _monitor_sizes(self, children: Sequence[_Child]) -> None:
        total = 0
        for child in children:
            for path, per_file_limit, role in (
                (
                    child.response_path,
                    MAX_CHILD_OUTPUT_BYTES,
                    "child analysis output",
                ),
                (
                    child.stderr_path,
                    MAX_CHILD_STDERR_BYTES,
                    "child stderr",
                ),
            ):
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    raise WorkloadExecutionError(
                        f"{role} disappeared during execution"
                    ) from exc
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > per_file_limit
                ):
                    raise WorkloadExecutionError(
                        f"{role} metadata or size is unsafe"
                    )
                total += metadata.st_size
        if total > MAX_TOTAL_CHILD_OUTPUT_BYTES:
            raise WorkloadExecutionError(
                "aggregate child output exceeds its size limit"
            )

    def _run_children(
        self,
        attempt: Path,
        shards: Sequence[
            Tuple[Path, Tuple[Mapping[str, Any], ...]]
        ],
        *,
        deadline: float,
    ) -> Tuple[Tuple[_Child, ...], Tuple[str, ...]]:
        argv = (
            os.fspath(self.spec.katago.path),
            "analysis",
            "-config",
            os.fspath(self.spec.analysis_config.path),
            "-model",
            os.fspath(self.spec.model.path),
        )
        if len(argv) > 16 or any(
            not part
            or "\x00" in part
            or len(os.fsencode(part)) > MAX_PATH_BYTES
            for part in argv
        ):
            raise WorkloadSpecError("KataGo analysis argv is unsafe")
        environment = dict(os.environ)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(self.spec.gpu_index)
        response_dir = attempt / "responses"
        stderr_dir = attempt / "stderr"
        children = []
        baseline_group_pids = self._process_group_pids()
        children_started = time.monotonic()
        try:
            for index, (query_path, query_rows) in enumerate(shards):
                response_path = (
                    response_dir / f"shard-{index:03d}.raw.jsonl"
                )
                stderr_path = stderr_dir / f"shard-{index:03d}.stderr"
                with query_path.open("rb") as source, response_path.open(
                    "xb"
                ) as destination, stderr_path.open("xb") as errors:
                    try:
                        process = subprocess.Popen(
                            argv,
                            cwd=os.fspath(attempt),
                            stdin=source,
                            stdout=destination,
                            stderr=errors,
                            env=environment,
                            close_fds=True,
                            shell=False,
                        )
                    except OSError as exc:
                        raise WorkloadExecutionError(
                            f"could not start analysis shard {index}: {exc}"
                        ) from exc
                children.append(
                    _Child(
                        index=index,
                        process=process,
                        query_path=query_path,
                        response_path=response_path,
                        stderr_path=stderr_path,
                        query_rows=query_rows,
                    )
                )
        except BaseException:
            self._kill_children(children, baseline_group_pids)
            raise
        if len(children) != self.process_count:
            self._kill_children(children, baseline_group_pids)
            raise WorkloadExecutionError(
                "did not launch exactly process_count analysis children"
            )

        attribution_grace = min(
            1.0,
            max(0.05, max(0.0, deadline - children_started) * 0.25),
        )
        attribution_deadline = min(
            deadline, children_started + attribution_grace
        )
        next_gpu_probe = 0.0
        attributed_pids: set[int] = set()
        try:
            while True:
                self._monitor_sizes(children)
                failed = [
                    child
                    for child in children
                    if child.process.poll() not in (None, 0)
                ]
                if failed:
                    child = failed[0]
                    detail = self._stderr_tail(child.stderr_path)
                    suffix = f": {detail}" if detail else ""
                    raise WorkloadExecutionError(
                        f"analysis shard {child.index} failed with status "
                        f"{child.process.returncode}{suffix}"
                    )
                now = time.monotonic()
                alive_pids = {
                    child.process.pid
                    for child in children
                    if child.process.poll() is None
                }
                if alive_pids and now >= next_gpu_probe:
                    observed = self._verify_gpu(
                        alive_pids,
                        require_all=now >= attribution_deadline,
                        deadline=deadline,
                    )
                    attributed_pids.update(observed)
                    next_gpu_probe = time.monotonic() + 0.05
                if all(
                    child.process.poll() == 0 for child in children
                ):
                    missing = {
                        child.process.pid for child in children
                    }.difference(attributed_pids)
                    if missing:
                        raise WorkloadExecutionError(
                            "analysis children completed without positive GPU "
                            f"attribution: {sorted(missing)}"
                        )
                    unexpected = self._unexpected_group_pids(
                        baseline_group_pids
                    )
                    if unexpected:
                        for pid in unexpected:
                            with contextlib.suppress(
                                ProcessLookupError, PermissionError
                            ):
                                os.kill(pid, signal.SIGKILL)
                        raise WorkloadExecutionError(
                            "analysis children exited while descendants "
                            f"remained: {sorted(unexpected)}"
                        )
                    return tuple(children), argv
                if now >= deadline:
                    raise WorkloadExecutionError(
                        "analysis children reached their inner timeout"
                    )
                time.sleep(min(0.02, max(0.001, deadline - now)))
        except BaseException:
            self._kill_children(children, baseline_group_pids)
            raise

    def _responses_by_id(
        self,
        path: Path,
        expected_rows: Sequence[Mapping[str, Any]],
        *,
        role: str,
        maximum_bytes: int = MAX_CHILD_OUTPUT_BYTES,
    ) -> Dict[str, Mapping[str, Any]]:
        data = _read_stable_regular_file(
            path,
            role,
            maximum_bytes=maximum_bytes,
            require_single_link=True,
        )
        if not data.endswith(b"\n"):
            raise WorkloadExecutionError(
                f"{role} must be newline-terminated JSONL"
            )
        raw_lines = data[:-1].split(b"\n")
        if not raw_lines or any(not line for line in raw_lines):
            raise WorkloadExecutionError(
                f"{role} contains empty response rows"
            )
        expected = {
            str(row["id"]): (len(row["moves"]), int(row["maxVisits"]))
            for row in expected_rows
        }
        if len(expected) != len(expected_rows):
            raise WorkloadExecutionError(
                f"{role} expected query IDs are not unique"
            )
        by_id: Dict[str, Mapping[str, Any]] = {}
        for index, raw_line in enumerate(raw_lines):
            if len(raw_line) > MAX_QUERY_LINE_BYTES * 8:
                raise WorkloadExecutionError(
                    f"{role} row {index} exceeds its size limit"
                )
            try:
                row = _json_value(raw_line, f"{role} row {index}")
            except EvaluatorBenchmarkWorkloadError as exc:
                raise WorkloadExecutionError(str(exc)) from exc
            if not isinstance(row, Mapping):
                raise WorkloadExecutionError(
                    f"{role} row {index} must be an object"
                )
            record_id = row.get("id")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in by_id
                or "error" in row
            ):
                raise WorkloadExecutionError(
                    f"{role} has a missing, duplicate, or failed response ID"
                )
            expected_coordinates = expected.get(record_id)
            root_info = row.get("rootInfo")
            move_infos = row.get("moveInfos")
            turn_number = row.get("turnNumber")
            if (
                expected_coordinates is None
                or not isinstance(root_info, Mapping)
                or not isinstance(move_infos, list)
                or not move_infos
                or any(not isinstance(move, Mapping) for move in move_infos)
                or type(turn_number) is not int
                or turn_number != expected_coordinates[0]
                or row.get("isDuringSearch") is not False
                or type(root_info.get("visits")) is not int
                or root_info["visits"] < expected_coordinates[1]
            ):
                raise WorkloadExecutionError(
                    f"{role} response {record_id!r} is not one final "
                    "normal analysis result"
                )
            by_id[record_id] = dict(row)
        if set(by_id) != set(expected) or len(by_id) != len(expected):
            raise WorkloadExecutionError(
                f"{role} IDs do not match their query shard"
            )
        return by_id

    def _child_manifest(
        self,
        attempt: Path,
        child: _Child,
        argv: Sequence[str],
        responses: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        query_data = _read_stable_regular_file(
            child.query_path,
            f"analysis shard {child.index} queries",
            maximum_bytes=MAX_QUERY_BYTES,
            require_single_link=True,
        )
        response_data = _read_stable_regular_file(
            child.response_path,
            f"analysis shard {child.index} output",
            maximum_bytes=MAX_CHILD_OUTPUT_BYTES,
            require_single_link=True,
        )
        query_ids = tuple(str(row["id"]) for row in child.query_rows)
        value: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": CHILD_OUTPUT_MANIFEST_CONTRACT,
            "shard_index": child.index,
            "returncode": child.process.returncode,
            "argv": list(argv),
            "gpu": {
                "index": self.spec.gpu_index,
                "uuid": self.spec.expected_gpu_uuid,
                "cuda_visible_devices": str(self.spec.gpu_index),
            },
            "query": {
                "path": _relative_path(
                    child.query_path, attempt, "child query path"
                ),
                "sha256": hashlib.sha256(query_data).hexdigest(),
                "size_bytes": len(query_data),
                "row_count": len(query_ids),
                "ids_sha256": canonical_sha256(list(query_ids)),
            },
            "output": {
                "path": _relative_path(
                    child.response_path, attempt, "child output path"
                ),
                "sha256": hashlib.sha256(response_data).hexdigest(),
                "size_bytes": len(response_data),
                "row_count": len(responses),
                "ids_sha256": canonical_sha256(sorted(responses)),
            },
        }
        value["manifest_sha256"] = canonical_sha256(value)
        return value

    def _validate_child_manifest(
        self,
        path: Path,
        expected: Mapping[str, Any],
    ) -> None:
        try:
            observed = _load_canonical_object(
                path,
                "child output manifest",
                maximum_bytes=MAX_SPEC_BYTES,
            )
            _exact_keys(
                observed,
                _CHILD_MANIFEST_KEYS,
                "child output manifest",
            )
            _exact_keys(
                observed["gpu"],
                _CHILD_GPU_KEYS,
                "child output manifest GPU",
            )
            _exact_keys(
                observed["query"],
                _CHILD_FILE_KEYS,
                "child output manifest query",
            )
            _exact_keys(
                observed["output"],
                _CHILD_FILE_KEYS,
                "child output manifest output",
            )
            body = dict(observed)
            supplied = _require_sha256(
                body.pop("manifest_sha256", None),
                "child output manifest identity",
            )
            if (
                supplied != canonical_sha256(body)
                or observed != expected
                or observed["returncode"] != 0
            ):
                raise WorkloadExecutionError(
                    "child output manifest contradicts its execution"
                )
        except WorkloadExecutionError:
            raise
        except EvaluatorBenchmarkWorkloadError as exc:
            raise WorkloadExecutionError(str(exc)) from exc

    def _collect_responses(
        self,
        attempt: Path,
        children: Sequence[_Child],
        argv: Sequence[str],
    ) -> Dict[str, Mapping[str, Any]]:
        merged: Dict[str, Mapping[str, Any]] = {}
        manifest_dir = attempt / "manifests"
        for child in children:
            if child.process.returncode != 0:
                raise WorkloadExecutionError(
                    f"analysis shard {child.index} did not complete"
                )
            responses = self._responses_by_id(
                child.response_path,
                child.query_rows,
                role=f"analysis shard {child.index} output",
            )
            if set(merged).intersection(responses):
                raise WorkloadExecutionError(
                    "analysis shard outputs contain duplicate IDs"
                )
            manifest = self._child_manifest(
                attempt, child, argv, responses
            )
            manifest_path = (
                manifest_dir / f"shard-{child.index:03d}.json"
            )
            _write_new_file(
                manifest_path,
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
            self._validate_child_manifest(manifest_path, manifest)
            merged.update(responses)
        return merged

    def _remove_successful_attempt(
        self, attempt: Path, identity: Tuple[int, int, int]
    ) -> None:
        if (
            attempt.parent != self.work_root
            or attempt.name != _ATTEMPT_NAME
            or _directory_identity(attempt, "workload attempt") != identity
        ):
            raise WorkloadConflictError(
                "refusing to clean an unowned workload attempt"
            )
        _inventory_tree(attempt)
        try:
            shutil.rmtree(attempt)
            _fsync_directory(self.work_root)
        except OSError as exc:
            raise WorkloadConflictError(
                f"could not remove successful workload attempt: {exc}"
            ) from exc

    def run(
        self, *, workload_started: Optional[float] = None
    ) -> Mapping[str, Any]:
        started = (
            time.monotonic()
            if workload_started is None
            else workload_started
        )
        overall_deadline = started + self.spec.timeout_seconds
        cleanup_margin = min(
            5.0,
            max(0.02, self.spec.timeout_seconds * 0.1),
            self.spec.timeout_seconds * 0.25,
        )
        child_deadline = overall_deadline - cleanup_margin
        self.spec.assert_frozen()
        query_rows = _load_query_rows(
            self.spec.queries.path,
            row_count=self.spec.row_count,
            max_visits=self.spec.max_visits,
        )
        replay = self._load_replay(query_rows)
        if replay is not None:
            self.spec.assert_frozen()
            self._assert_root()
            return replay
        self._require_fresh_root()
        self._verify_gpu(deadline=child_deadline)
        self.spec.assert_frozen()
        self._assert_root()
        if time.monotonic() >= child_deadline:
            raise WorkloadExecutionError(
                "workload preflight exhausted the child timeout budget"
            )

        attempt = self.work_root / _ATTEMPT_NAME
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise WorkloadConflictError(
                "an incomplete workload attempt already exists"
            ) from exc
        attempt_identity = _directory_identity(
            attempt, "workload attempt"
        )
        shards = self._write_shards(attempt, query_rows)
        children, argv = self._run_children(
            attempt, shards, deadline=child_deadline
        )
        self.spec.assert_frozen()
        self._assert_root()
        self._verify_gpu(deadline=overall_deadline)
        merged = self._collect_responses(
            attempt, children, argv
        )
        expected_ids = tuple(str(row["id"]) for row in query_rows)
        if set(merged) != set(expected_ids):
            raise WorkloadExecutionError(
                "analysis shards do not cover the frozen query prefix"
            )
        artifact_data = _canonical_jsonl(
            merged[record_id] for record_id in expected_ids
        )
        _publish_immutable_bytes(
            self.artifact,
            artifact_data,
            root=self.work_root,
            maximum_bytes=MAX_TOTAL_CHILD_OUTPUT_BYTES,
        )
        artifact_hash = hashlib.sha256(artifact_data).hexdigest()
        persisted = _read_stable_regular_file(
            self.artifact,
            "published analysis artifact",
            maximum_bytes=MAX_TOTAL_CHILD_OUTPUT_BYTES,
            require_single_link=True,
        )
        if persisted != artifact_data:
            raise WorkloadExecutionError(
                "published analysis artifact bytes changed"
            )
        self._remove_successful_attempt(attempt, attempt_identity)
        self._assert_root()
        self.spec.assert_frozen()
        _, files = _inventory_tree(self.work_root)
        if files != (_ARTIFACT_RELATIVE.as_posix(),):
            raise WorkloadExecutionError(
                "successful workload tree contains undeclared files"
            )
        elapsed = time.monotonic() - started
        if elapsed > self.spec.timeout_seconds:
            raise WorkloadExecutionError(
                "validated workload exceeded its overall timeout"
            )

        manifest = [
            {
                "path": _ARTIFACT_RELATIVE.as_posix(),
                "sha256": artifact_hash,
                "size_bytes": len(artifact_data),
                "row_count": self.spec.row_count,
            }
        ]
        receipt: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "contract": BENCHMARK_RECEIPT_CONTRACT,
            "process_count": self.process_count,
            "completed_work_count": self.spec.row_count,
            "elapsed_seconds": elapsed,
            "output_manifest": manifest,
            "output_manifest_sha256": canonical_sha256(manifest),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipt_data = (canonical_json(receipt) + "\n").encode("utf-8")
        _publish_immutable_bytes(
            self.output,
            receipt_data,
            root=self.work_root,
            maximum_bytes=MAX_SPEC_BYTES,
        )
        published_receipt = _read_stable_regular_file(
            self.output,
            "published benchmark receipt",
            maximum_bytes=MAX_SPEC_BYTES,
            require_single_link=True,
        )
        if published_receipt != receipt_data:
            raise WorkloadConflictError(
                "published receipt did not replay byte-for-byte"
            )
        _, final_files = _inventory_tree(self.work_root)
        expected_files = tuple(
            sorted(
                (
                    _ARTIFACT_RELATIVE.as_posix(),
                    self.output.relative_to(self.work_root).as_posix(),
                )
            )
        )
        if final_files != expected_files:
            raise WorkloadConflictError(
                "published workload tree contains undeclared files"
            )
        self._assert_root()
        return receipt


def run_workload(
    spec: WorkloadSpec | Path,
    *,
    process_count: int,
    output: Path,
    work_root: Path,
    expected_spec_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    started = time.monotonic()
    workload = EvaluatorBenchmarkWorkload(
        spec,
        process_count=process_count,
        output=output,
        work_root=work_root,
        expected_spec_sha256=expected_spec_sha256,
    )
    return workload.run(workload_started=started)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--expected-spec-sha256",
        "--expected-spec-hash",
        dest="expected_spec_sha256",
        required=True,
    )
    parser.add_argument(
        "--process-count",
        required=True,
        type=int,
        choices=PROCESS_COUNTS,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        receipt = run_workload(
            args.spec,
            process_count=args.process_count,
            output=args.output,
            work_root=args.work_root,
            expected_spec_sha256=args.expected_spec_sha256,
        )
    except (EvaluatorBenchmarkWorkloadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    print(canonical_json(receipt))
    return 0


load_spec = load_workload_spec
publish_spec = publish_workload_spec
publish_benchmark_specs = publish_topology_specs
publish_specs = publish_topology_specs
run = run_workload
benchmark_argv_template = build_benchmark_argv_template


__all__ = [
    "BENCHMARK_RECEIPT_CONTRACT",
    "CHILD_OUTPUT_MANIFEST_CONTRACT",
    "EvaluatorBenchmarkWorkload",
    "EvaluatorBenchmarkWorkloadError",
    "FileBinding",
    "PROCESS_COUNTS",
    "SCHEMA_VERSION",
    "WORKLOAD_SPEC_CONTRACT",
    "WorkloadConflictError",
    "WorkloadExecutionError",
    "WorkloadSpec",
    "WorkloadSpecError",
    "benchmark_argv_template",
    "benchmark_input_bindings",
    "build_benchmark_argv_template",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
    "load_spec",
    "load_workload_spec",
    "main",
    "parse_args",
    "publish_benchmark_specs",
    "publish_spec",
    "publish_specs",
    "publish_topology_specs",
    "publish_workload_spec",
    "run",
    "run_workload",
]


if __name__ == "__main__":
    raise SystemExit(main())
