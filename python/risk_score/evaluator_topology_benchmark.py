#!/usr/bin/env python3
"""Run the fixed evaluator-topology benchmark and publish bootstrap evidence.

The benchmark command is an immutable, hash-bound argv template.  It receives
one fresh work directory per repetition and must write a canonical receipt to
``{output}``.  The receipt carries a relative artifact manifest; this runner
re-hashes and inventories those artifacts before accepting any timing data.
Only after all six isolated runs agree on their deterministic output does the
runner publish the generic evidence consumed by :mod:`autonomy_bootstrap`.
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
import string
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are Unix.
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
SPEC_CONTRACT = "risk-score-evaluator-topology-benchmark-spec-v1"
BENCHMARK_RECEIPT_CONTRACT = "risk-score-evaluator-topology-benchmark-receipt-v1"
COMPLETION_RECEIPT_CONTRACT = (
    "risk-score-evaluator-topology-benchmark-completion-v1"
)
GATE_EVIDENCE_CONTRACT = "risk-score-autonomy-gate-evidence-v1"
GATE_ID = "evaluator-topology-benchmark"

PROCESS_COUNTS = (4, 8, 16)
TOPOLOGY_CHOICES = PROCESS_COUNTS
REPETITION_COUNT = 2
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_ARTIFACTS = 100_000
MAX_TOTAL_OUTPUT_BYTES = 4 * 1024 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 24 * 60 * 60

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TEMPLATE_FIELDS = frozenset({"process_count", "output", "work_root"})
_SPEC_KEYS = {
    "schema_version",
    "contract",
    "benchmark_argv_template",
    "inputs",
    "timeout_seconds",
    "work_root",
    "evidence_output",
    "spec_sha256",
}
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
_COMPLETION_KEYS = {
    "schema_version",
    "contract",
    "spec",
    "benchmark_argv_template_sha256",
    "inputs",
    "repetitions",
    "evidence_output",
    "evidence",
    "completion_sha256",
}
_REPETITION_KEYS = {
    "process_count",
    "repetition",
    "completed_work_count",
    "elapsed_seconds",
    "throughput_per_second",
    "output_manifest_sha256",
    "receipt_sha256",
}
_ATTEMPT_PREFIX = ".evaluator-topology-run-"
_LOCK_NAME = ".evaluator-topology-benchmark.lock"
_COMPLETION_NAME = ".evaluator-topology-benchmark-completion.json"
_COMMAND_RECEIPT_NAME = "benchmark-receipt.json"


class EvaluatorTopologyBenchmarkError(RuntimeError):
    """Benchmark input, execution, or evidence is unsafe or contradictory."""


class BenchmarkSpecError(EvaluatorTopologyBenchmarkError, ValueError):
    """The immutable benchmark specification is malformed or stale."""


class BenchmarkExecutionError(EvaluatorTopologyBenchmarkError):
    """A benchmark command failed or returned invalid evidence."""


class BenchmarkConflictError(EvaluatorTopologyBenchmarkError):
    """Existing state contradicts the requested immutable benchmark."""


class BenchmarkBusy(EvaluatorTopologyBenchmarkError):
    """Another benchmark runner owns the work-root lock."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def spec_value(self) -> Mapping[str, str]:
        return {"path": os.fspath(self.path), "sha256": self.sha256}

    def verify(self) -> None:
        observed = _stable_file_sha256(self.path, f"bound input {self.path}")
        if observed != self.sha256:
            raise BenchmarkSpecError(
                f"hash-bound input changed: {self.path}; "
                f"expected {self.sha256}, observed {observed}"
            )


@dataclass(frozen=True)
class BenchmarkSpec:
    path: Path
    file_sha256: str
    identity: str
    raw: Mapping[str, Any]
    benchmark_argv_template: Tuple[str, ...]
    inputs: Tuple[FileBinding, ...]
    timeout_seconds: float
    work_root: Path
    evidence_output: Path

    @property
    def completion_path(self) -> Path:
        return self.work_root / _COMPLETION_NAME

    @property
    def lock_path(self) -> Path:
        return self.work_root / _LOCK_NAME


@dataclass(frozen=True)
class BenchmarkObservation:
    process_count: int
    repetition: int
    completed_work_count: int
    elapsed_seconds: float
    output_manifest_sha256: str
    receipt_sha256: str

    @property
    def throughput_per_second(self) -> float:
        return self.completed_work_count / self.elapsed_seconds

    def completion_value(self) -> Mapping[str, Any]:
        return {
            "process_count": self.process_count,
            "repetition": self.repetition,
            "completed_work_count": self.completed_work_count,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_per_second": self.throughput_per_second,
            "output_manifest_sha256": self.output_manifest_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    wall_elapsed_seconds: float
    stderr_tail: str


def canonical_json(value: Any) -> str:
    """Return the repository's canonical compact JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return _stable_file_sha256(Path(path), f"file {path}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluatorTopologyBenchmarkError(f"{role} must be an object")
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} keys differ from contract; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be a lowercase 64-character SHA-256"
        )
    return value


def _positive_int(value: Any, role: str) -> int:
    if type(value) is not int or not 1 <= value <= 10**18:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be a positive integer no greater than 1e18"
        )
    return value


def _positive_number(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be a positive finite number"
        )
    return float(value)


def _ensure_finite_json(value: Any, role: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluatorTopologyBenchmarkError(f"{role} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise EvaluatorTopologyBenchmarkError(
                    f"{role} contains a non-string object key"
                )
            _ensure_finite_json(child, f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{role}[{index}]")


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorTopologyBenchmarkError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvaluatorTopologyBenchmarkError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_symlink_ancestors(path: Path, role: str) -> None:
    current = Path(path)
    while True:
        if current.is_symlink():
            raise EvaluatorTopologyBenchmarkError(
                f"{role} has a symlinked path component: {current}"
            )
        if current.parent == current:
            return
        current = current.parent


def _canonical_absolute(raw: Any, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be a nonempty absolute path"
        )
    path = Path(raw)
    normalized = Path(os.path.abspath(os.fspath(path)))
    if not path.is_absolute() or path != normalized:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be an absolute lexically normalized path"
        )
    _reject_symlink_ancestors(path, role)
    if path.resolve(strict=False) != path:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must contain no symlink components"
        )
    return path


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
    maximum_bytes: Optional[int] = None,
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
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be an existing regular non-symlink file: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluatorTopologyBenchmarkError(f"{role} must be a regular file")
        if require_single_link and before.st_nlink != 1:
            raise EvaluatorTopologyBenchmarkError(
                f"{role} must not be hard-linked outside its work directory"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise EvaluatorTopologyBenchmarkError(
                f"{role} exceeds the {maximum_bytes}-byte limit"
            )
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        if remaining:
            raise EvaluatorTopologyBenchmarkError(f"{role} was truncated while read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = source.lstat()
    except OSError as exc:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} disappeared while it was read"
        ) from exc
    if (
        _metadata_identity(before) != _metadata_identity(after)
        or _metadata_identity(after) != _metadata_identity(path_after)
    ):
        raise EvaluatorTopologyBenchmarkError(f"{role} changed while it was read")
    _reject_symlink_ancestors(source, role)
    return b"".join(chunks)


def _stable_file_sha256(
    path: Path, role: str, *, require_single_link: bool = False
) -> str:
    source = Path(path)
    _reject_symlink_ancestors(source, role)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(source), flags)
    except OSError as exc:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be an existing regular non-symlink file: {exc}"
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluatorTopologyBenchmarkError(f"{role} must be a regular file")
        if require_single_link and before.st_nlink != 1:
            raise EvaluatorTopologyBenchmarkError(
                f"{role} must not be hard-linked outside its work directory"
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
        raise EvaluatorTopologyBenchmarkError(
            f"{role} disappeared while it was hashed"
        ) from exc
    if (
        _metadata_identity(before) != _metadata_identity(after)
        or _metadata_identity(after) != _metadata_identity(path_after)
    ):
        raise EvaluatorTopologyBenchmarkError(f"{role} changed while it was hashed")
    _reject_symlink_ancestors(source, role)
    return digest.hexdigest()


def _stable_record_count(path: Path, role: str) -> int:
    source = Path(path)
    _reject_symlink_ancestors(source, role)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(source), flags)
    except OSError as exc:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be an existing regular non-symlink file: {exc}"
        ) from exc
    count = 0
    previous_was_newline = True
    saw_data = False
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvaluatorTopologyBenchmarkError(f"{role} must be a regular file")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            saw_data = True
            for byte in block:
                if byte == 10:
                    if previous_was_newline:
                        raise EvaluatorTopologyBenchmarkError(
                            f"{role} contains an empty output record"
                        )
                    count += 1
                    previous_was_newline = True
                else:
                    previous_was_newline = False
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = source.lstat()
    if (
        _metadata_identity(before) != _metadata_identity(after)
        or _metadata_identity(after) != _metadata_identity(path_after)
    ):
        raise EvaluatorTopologyBenchmarkError(f"{role} changed while records were counted")
    if not saw_data or not previous_was_newline or count <= 0:
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must contain nonempty newline-terminated output records"
        )
    return count


def _load_canonical_object(path: Path, role: str) -> Dict[str, Any]:
    data = _read_stable_regular_file(
        Path(path),
        role,
        maximum_bytes=MAX_JSON_BYTES,
        require_single_link=True,
    )
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EvaluatorTopologyBenchmarkError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluatorTopologyBenchmarkError(f"{role} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluatorTopologyBenchmarkError(f"{role} root must be an object")
    _ensure_finite_json(value, role)
    if data != (canonical_json(value) + "\n").encode("utf-8"):
        raise EvaluatorTopologyBenchmarkError(
            f"{role} must be canonical newline-terminated JSON"
        )
    return value


def _file_binding(value: Any, role: str) -> FileBinding:
    binding = _exact_keys(value, {"path", "sha256"}, role)
    path = _canonical_absolute(binding["path"], f"{role} path")
    digest = _require_sha256(binding["sha256"], f"{role} sha256")
    result = FileBinding(path=path, sha256=digest)
    result.verify()
    return result


def _template_field_counts(argv: Sequence[str]) -> Counter[str]:
    formatter = string.Formatter()
    counts: Counter[str] = Counter()
    for index, part in enumerate(argv):
        try:
            parsed = tuple(formatter.parse(part))
        except ValueError as exc:
            raise BenchmarkSpecError(
                f"benchmark argv token {index} has malformed braces: {exc}"
            ) from exc
        for _literal, field, format_spec, conversion in parsed:
            if field is None:
                continue
            if (
                field not in _TEMPLATE_FIELDS
                or format_spec
                or conversion is not None
            ):
                raise BenchmarkSpecError(
                    f"benchmark argv token {index} has unsupported placeholder "
                    f"{field!r}"
                )
            counts[field] += 1
    return counts


def _validate_argv_template(
    value: Any, inputs: Sequence[FileBinding]
) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(part, str)
            or not part
            or "\x00" in part
            or "\n" in part
            or "\r" in part
            for part in value
        )
    ):
        raise BenchmarkSpecError(
            "benchmark_argv_template must be a nonempty argv string array"
        )
    counts = _template_field_counts(value)
    if counts != Counter({field: 1 for field in _TEMPLATE_FIELDS}):
        raise BenchmarkSpecError(
            "benchmark argv must contain each of {process_count}, {output}, and "
            "{work_root} exactly once"
        )
    if any(field in value[0] for field in ("{process_count}", "{output}", "{work_root}")):
        raise BenchmarkSpecError("benchmark executable may not contain placeholders")

    bound_paths = {binding.path for binding in inputs}
    executable = _canonical_absolute(value[0], "benchmark executable")
    if (
        executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise BenchmarkSpecError(
            "benchmark executable must be an executable regular non-symlink file"
        )
    if executable not in bound_paths:
        raise BenchmarkSpecError("benchmark executable must be a hash-bound input")

    for index, part in enumerate(value):
        if "{" in part or "}" in part or not Path(part).is_absolute():
            continue
        fixed_path = _canonical_absolute(part, f"benchmark argv token {index}")
        if (
            not fixed_path.is_file()
            or fixed_path.is_symlink()
            or fixed_path not in bound_paths
        ):
            raise BenchmarkSpecError(
                f"fixed absolute argv path must be a hash-bound regular file: "
                f"{fixed_path}"
            )
    return tuple(value)


def _load_benchmark_spec(
    path: Path, *, expected_spec_sha256: Optional[str] = None
) -> BenchmarkSpec:
    """Load and fully verify one canonical immutable benchmark specification.

    ``expected_spec_sha256`` binds the complete canonical spec file, including
    its internal ``spec_sha256`` identity.
    """

    try:
        source = _canonical_absolute(
            os.fspath(Path(path)), "benchmark specification"
        )
        raw = _load_canonical_object(source, "benchmark specification")
        _exact_keys(raw, _SPEC_KEYS, "benchmark specification")
    except EvaluatorTopologyBenchmarkError as exc:
        raise BenchmarkSpecError(str(exc)) from exc
    if raw.get("schema_version") != SCHEMA_VERSION or isinstance(
        raw.get("schema_version"), bool
    ):
        raise BenchmarkSpecError("benchmark specification schema_version must be 1")
    if raw.get("contract") != SPEC_CONTRACT:
        raise BenchmarkSpecError("benchmark specification contract is unsupported")
    body = dict(raw)
    identity = _require_sha256(
        body.pop("spec_sha256", None), "benchmark specification identity"
    )
    if identity != canonical_sha256(body):
        raise BenchmarkSpecError("benchmark specification self-hash is invalid")
    source_data = _read_stable_regular_file(
        source,
        "benchmark specification",
        maximum_bytes=MAX_JSON_BYTES,
        require_single_link=True,
    )
    file_identity = hashlib.sha256(source_data).hexdigest()
    if expected_spec_sha256 is not None:
        expected = _require_sha256(
            expected_spec_sha256, "expected benchmark specification hash"
        )
        if expected != file_identity:
            raise BenchmarkSpecError(
                "benchmark specification file does not match expected SHA-256"
            )

    raw_inputs = raw["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise BenchmarkSpecError("benchmark inputs must be a nonempty array")
    try:
        inputs = tuple(
            _file_binding(item, f"benchmark input {index}")
            for index, item in enumerate(raw_inputs)
        )
    except EvaluatorTopologyBenchmarkError as exc:
        if isinstance(exc, BenchmarkSpecError):
            raise
        raise BenchmarkSpecError(str(exc)) from exc
    input_paths = [os.fspath(binding.path) for binding in inputs]
    if input_paths != sorted(input_paths) or len(set(input_paths)) != len(input_paths):
        raise BenchmarkSpecError(
            "benchmark inputs must be unique and sorted by canonical path"
        )

    argv_template = _validate_argv_template(raw["benchmark_argv_template"], inputs)
    timeout_seconds = _positive_number(
        raw["timeout_seconds"], "benchmark timeout_seconds"
    )
    if timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise BenchmarkSpecError(
            f"benchmark timeout_seconds may not exceed {MAX_TIMEOUT_SECONDS}"
        )

    work_root = _canonical_absolute(raw["work_root"], "benchmark work_root")
    if work_root == Path(work_root.anchor):
        raise BenchmarkSpecError("benchmark work_root may not be a filesystem root")
    if _lexists(work_root) and (
        work_root.is_symlink() or not work_root.is_dir()
    ):
        raise BenchmarkSpecError(
            "benchmark work_root must be a non-symlink directory when present"
        )
    if not work_root.exists() and (
        work_root.parent.is_symlink() or not work_root.parent.is_dir()
    ):
        raise BenchmarkSpecError(
            "benchmark work_root parent must be an existing non-symlink directory"
        )

    evidence_output = _canonical_absolute(
        raw["evidence_output"], "benchmark evidence_output"
    )
    if not _strictly_within(evidence_output, work_root):
        raise BenchmarkSpecError(
            "benchmark evidence_output must be strictly beneath work_root"
        )
    if _lexists(evidence_output) and (
        evidence_output.is_symlink() or not evidence_output.is_file()
    ):
        raise BenchmarkSpecError(
            "benchmark evidence_output must be a regular file when present"
        )
    if _strictly_within(source, work_root):
        raise BenchmarkSpecError(
            "benchmark specification may not be stored beneath work_root"
        )
    for binding in inputs:
        if _strictly_within(binding.path, work_root) or binding.path == work_root:
            raise BenchmarkSpecError(
                f"hash-bound input may not be stored beneath work_root: {binding.path}"
            )
    reserved = {
        work_root / _LOCK_NAME,
        work_root / _COMPLETION_NAME,
    }
    if evidence_output in reserved:
        raise BenchmarkSpecError("benchmark evidence_output uses a reserved path")

    return BenchmarkSpec(
        path=source,
        file_sha256=file_identity,
        identity=identity,
        raw=raw,
        benchmark_argv_template=argv_template,
        inputs=inputs,
        timeout_seconds=timeout_seconds,
        work_root=work_root,
        evidence_output=evidence_output,
    )


def load_benchmark_spec(
    path: Path, *, expected_spec_sha256: Optional[str] = None
) -> BenchmarkSpec:
    try:
        return _load_benchmark_spec(
            path, expected_spec_sha256=expected_spec_sha256
        )
    except BenchmarkSpecError:
        raise
    except EvaluatorTopologyBenchmarkError as exc:
        raise BenchmarkSpecError(str(exc)) from exc


load_spec = load_benchmark_spec


def _directory_identity(path: Path, role: str) -> Tuple[int, int, int]:
    _reject_symlink_ancestors(path, role)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BenchmarkExecutionError(f"{role} is missing: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BenchmarkExecutionError(f"{role} must be a non-symlink directory")
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode)


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
        raise BenchmarkConflictError(f"output parent escapes work_root: {target}")
    _directory_identity(root, "benchmark work_root")
    current = root
    relative = target.relative_to(root)
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _directory_identity(current, f"output directory {current}")


def _publish_immutable(path: Path, value: Mapping[str, Any], *, root: Path) -> None:
    target = Path(path)
    if not _strictly_within(target, root):
        raise BenchmarkConflictError(f"immutable output escapes work_root: {target}")
    _ensure_directory_chain(root, target.parent)
    data = (canonical_json(dict(value)) + "\n").encode("utf-8")
    if _lexists(target):
        _recover_owned_publication_link(target)
        existing = _read_stable_regular_file(
            target,
            f"existing immutable output {target}",
            maximum_bytes=MAX_JSON_BYTES,
            require_single_link=True,
        )
        if existing != data:
            raise BenchmarkConflictError(
                f"existing immutable output conflicts: {target}"
            )
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=os.fspath(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(os.fspath(temporary), os.fspath(target), follow_symlinks=False)
        except FileExistsError:
            existing = _read_stable_regular_file(
                target,
                f"existing immutable output {target}",
                maximum_bytes=MAX_JSON_BYTES,
                require_single_link=True,
            )
            if existing != data:
                raise BenchmarkConflictError(
                    f"existing immutable output conflicts: {target}"
                )
        _fsync_directory(target.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        with contextlib.suppress(OSError):
            _fsync_directory(target.parent)


def _recover_owned_publication_link(path: Path) -> None:
    target = Path(path)
    if not _lexists(target):
        return
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BenchmarkConflictError(f"published output is unsafe: {target}")
    if metadata.st_nlink == 1:
        return
    if metadata.st_nlink != 2:
        raise BenchmarkConflictError(
            f"published output has unexpected hard links: {target}"
        )
    candidates = []
    for candidate in target.parent.glob(f".{target.name}.*.tmp"):
        candidate_metadata = candidate.lstat()
        if (
            stat.S_ISREG(candidate_metadata.st_mode)
            and not stat.S_ISLNK(candidate_metadata.st_mode)
            and candidate_metadata.st_dev == metadata.st_dev
            and candidate_metadata.st_ino == metadata.st_ino
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise BenchmarkConflictError(
            f"cannot recover owned publication hard link: {target}"
        )
    candidates[0].unlink()
    _fsync_directory(target.parent)
    final = target.lstat()
    if final.st_nlink != 1 or final.st_ino != metadata.st_ino:
        raise BenchmarkConflictError(
            f"published output hard-link recovery did not converge: {target}"
        )


def _stderr_tail(stream: Any, limit: int = 8192) -> str:
    try:
        stream.flush()
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return ""


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:  # pragma: no cover - production targets are Unix.
        with contextlib.suppress(OSError):
            process.kill()


def _invoke_benchmark(
    argv: Sequence[str], *, cwd: Path, timeout_seconds: float
) -> _ProcessResult:
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+b") as stdout, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=os.fspath(cwd),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise BenchmarkExecutionError(
                f"could not start benchmark command: {exc}"
            ) from exc
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(process)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            detail = _stderr_tail(stderr)
            suffix = f": {detail}" if detail else ""
            raise BenchmarkExecutionError(
                f"benchmark command timed out after {timeout_seconds:g} seconds"
                f"{suffix}"
            ) from exc
        finally:
            # A successful parent is not allowed to leave workers running.
            _kill_process_group(process)
        elapsed = time.monotonic() - started
        return _ProcessResult(
            returncode=int(process.returncode),
            wall_elapsed_seconds=elapsed,
            stderr_tail=_stderr_tail(stderr),
        )


def _relative_artifact_path(value: Any, role: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        raise BenchmarkExecutionError(
            f"{role} must be a nonempty normalized POSIX relative path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BenchmarkExecutionError(
            f"{role} must be a normalized relative path without '..'"
        )
    return path


def _inventory_output_files(root: Path, receipt_path: Path) -> Tuple[str, ...]:
    receipt_relative = receipt_path.relative_to(root).as_posix()
    observed = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        _directory_identity(current, f"benchmark output directory {current}")
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise BenchmarkExecutionError(
                    f"benchmark output tree contains unsafe directory: {child}"
                )
        for name in file_names:
            child = current / name
            relative = child.relative_to(root).as_posix()
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise BenchmarkExecutionError(
                    f"benchmark output tree contains non-regular artifact: {child}"
                )
            if metadata.st_nlink != 1:
                raise BenchmarkExecutionError(
                    f"benchmark artifact is hard-linked: {child}"
                )
            if relative != receipt_relative:
                observed.append(relative)
    return tuple(sorted(observed))


def _validate_command_receipt(
    path: Path,
    *,
    run_root: Path,
    process_count: int,
    repetition: int,
    timeout_seconds: float,
    wall_elapsed_seconds: float,
) -> BenchmarkObservation:
    try:
        receipt = _load_canonical_object(path, "benchmark command receipt")
        _exact_keys(receipt, _RECEIPT_KEYS, "benchmark command receipt")
    except EvaluatorTopologyBenchmarkError as exc:
        if isinstance(exc, BenchmarkExecutionError):
            raise
        raise BenchmarkExecutionError(str(exc)) from exc
    body = dict(receipt)
    supplied_receipt_hash = _require_sha256(
        body.pop("receipt_sha256", None), "benchmark command receipt identity"
    )
    if supplied_receipt_hash != canonical_sha256(body):
        raise BenchmarkExecutionError("benchmark command receipt self-hash is invalid")
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or isinstance(receipt["schema_version"], bool)
        or receipt["contract"] != BENCHMARK_RECEIPT_CONTRACT
    ):
        raise BenchmarkExecutionError(
            "benchmark command receipt contract is unsupported"
        )
    if receipt["process_count"] != process_count or isinstance(
        receipt["process_count"], bool
    ):
        raise BenchmarkExecutionError(
            "benchmark command receipt process_count does not match its invocation"
        )
    completed = _positive_int(
        receipt["completed_work_count"], "benchmark completed_work_count"
    )
    elapsed = _positive_number(
        receipt["elapsed_seconds"], "benchmark elapsed_seconds"
    )
    if elapsed > timeout_seconds:
        raise BenchmarkExecutionError(
            "benchmark receipt elapsed_seconds exceeds the command timeout"
        )
    # Process startup and wait-loop granularity dominate tiny test/smoke runs.
    # Production benchmarks run for minutes, where the 5% bound dominates.
    elapsed_tolerance = max(0.500, wall_elapsed_seconds * 0.05)
    if abs(elapsed - wall_elapsed_seconds) > elapsed_tolerance:
        raise BenchmarkExecutionError(
            "benchmark receipt elapsed_seconds differs from the independently "
            "measured command lifetime"
        )

    raw_manifest = receipt["output_manifest"]
    if not isinstance(raw_manifest, list) or not raw_manifest:
        raise BenchmarkExecutionError(
            "benchmark output_manifest must be a nonempty array"
        )
    if len(raw_manifest) > MAX_OUTPUT_ARTIFACTS:
        raise BenchmarkExecutionError("benchmark output_manifest is too large")
    manifest_hash = _require_sha256(
        receipt["output_manifest_sha256"],
        "benchmark output_manifest_sha256",
    )
    if manifest_hash != canonical_sha256(raw_manifest):
        raise BenchmarkExecutionError("benchmark output manifest hash is invalid")

    declared_paths = []
    total_size = 0
    total_rows = 0
    for index, raw_artifact in enumerate(raw_manifest):
        artifact = _exact_keys(
            raw_artifact, _ARTIFACT_KEYS, f"benchmark output artifact {index}"
        )
        relative = _relative_artifact_path(
            artifact["path"], f"benchmark output artifact {index} path"
        )
        relative_text = relative.as_posix()
        if relative_text == path.relative_to(run_root).as_posix():
            raise BenchmarkExecutionError(
                "benchmark receipt may not list itself as an output artifact"
            )
        digest = _require_sha256(
            artifact["sha256"], f"benchmark output artifact {index} sha256"
        )
        size = artifact["size_bytes"]
        if type(size) is not int or size < 0:
            raise BenchmarkExecutionError(
                f"benchmark output artifact {index} size_bytes must be nonnegative"
            )
        total_size += size
        if total_size > MAX_TOTAL_OUTPUT_BYTES:
            raise BenchmarkExecutionError(
                "benchmark output artifacts exceed the total size limit"
            )
        artifact_path = run_root.joinpath(*relative.parts)
        if not _strictly_within(artifact_path, run_root):
            raise BenchmarkExecutionError("benchmark artifact escapes its run root")
        try:
            metadata = artifact_path.lstat()
        except OSError as exc:
            raise BenchmarkExecutionError(
                f"benchmark output artifact is missing: {relative_text}"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != size
        ):
            raise BenchmarkExecutionError(
                f"benchmark output artifact metadata is invalid: {relative_text}"
            )
        observed_hash = _stable_file_sha256(
            artifact_path,
            f"benchmark output artifact {relative_text}",
            require_single_link=True,
        )
        if observed_hash != digest:
            raise BenchmarkExecutionError(
                f"benchmark output artifact hash is invalid: {relative_text}"
            )
        row_count = _positive_int(
            artifact["row_count"],
            f"benchmark output artifact {index} row_count",
        )
        observed_rows = _stable_record_count(
            artifact_path,
            f"benchmark output artifact {relative_text}",
        )
        if observed_rows != row_count:
            raise BenchmarkExecutionError(
                f"benchmark output artifact row count is invalid: {relative_text}"
            )
        total_rows += observed_rows
        declared_paths.append(relative_text)

    if declared_paths != sorted(set(declared_paths)):
        raise BenchmarkExecutionError(
            "benchmark output manifest paths must be sorted and unique"
        )
    observed_paths = _inventory_output_files(run_root, path)
    if tuple(declared_paths) != observed_paths:
        raise BenchmarkExecutionError(
            "benchmark output manifest does not exactly inventory the output tree"
        )
    if completed != total_rows:
        raise BenchmarkExecutionError(
            "benchmark completed_work_count does not match validated output rows"
        )
    return BenchmarkObservation(
        process_count=process_count,
        repetition=repetition,
        completed_work_count=completed,
        elapsed_seconds=elapsed,
        output_manifest_sha256=manifest_hash,
        receipt_sha256=supplied_receipt_hash,
    )


def _select_topology(benchmarks: Sequence[Mapping[str, Any]]) -> int:
    # Importing the authoritative selector avoids a second tie-breaking contract.
    from risk_score.autonomy_bootstrap import select_evaluator_topology

    try:
        return select_evaluator_topology(benchmarks)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise BenchmarkExecutionError(
            f"autonomy bootstrap rejected topology benchmark evidence: {exc}"
        ) from exc


def _derive_checks(
    observations: Sequence[BenchmarkObservation],
) -> Mapping[str, Any]:
    expected_order = [
        (count, repetition)
        for count in PROCESS_COUNTS
        for repetition in range(1, REPETITION_COUNT + 1)
    ]
    observed_order = [
        (observation.process_count, observation.repetition)
        for observation in observations
    ]
    if observed_order != expected_order:
        raise BenchmarkExecutionError(
            "benchmark observations do not cover two ordered repetitions of 4/8/16"
        )

    benchmarks = []
    all_hashes = set()
    for count in PROCESS_COUNTS:
        repeated = [
            observation
            for observation in observations
            if observation.process_count == count
        ]
        first, second = repeated
        if first.output_manifest_sha256 != second.output_manifest_sha256:
            raise BenchmarkExecutionError(
                f"benchmark output is not deterministic for process_count={count}"
            )
        total_work = sum(item.completed_work_count for item in repeated)
        total_elapsed = sum(item.elapsed_seconds for item in repeated)
        throughput = total_work / total_elapsed
        if not math.isfinite(throughput) or throughput <= 0:
            raise BenchmarkExecutionError(
                f"derived throughput is invalid for process_count={count}"
            )
        all_hashes.add(first.output_manifest_sha256)
        benchmarks.append(
            {
                "process_count": count,
                "throughput_per_second": throughput,
                "output_sha256": first.output_manifest_sha256,
                "repeat_output_sha256": second.output_manifest_sha256,
            }
        )
    if len(all_hashes) != 1:
        raise BenchmarkExecutionError(
            "deterministic benchmark output changed across process counts"
        )
    selected = _select_topology(benchmarks)
    return {
        "benchmarks": benchmarks,
        "selected_process_count": selected,
    }


def _generic_evidence(checks: Mapping[str, Any]) -> Mapping[str, Any]:
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": GATE_EVIDENCE_CONTRACT,
        "gate_id": GATE_ID,
        "decision": "PASS",
        "checks": dict(checks),
    }
    _ensure_finite_json(value, "generic gate evidence")
    value["evidence_sha256"] = canonical_sha256(value)
    return value


def _validate_generic_evidence(value: Any) -> Mapping[str, Any]:
    evidence = _exact_keys(
        value,
        {
            "schema_version",
            "contract",
            "gate_id",
            "decision",
            "checks",
            "evidence_sha256",
        },
        "generic gate evidence",
    )
    body = dict(evidence)
    supplied = _require_sha256(
        body.pop("evidence_sha256", None), "generic gate evidence identity"
    )
    if (
        evidence["schema_version"] != SCHEMA_VERSION
        or isinstance(evidence["schema_version"], bool)
        or evidence["contract"] != GATE_EVIDENCE_CONTRACT
        or evidence["gate_id"] != GATE_ID
        or evidence["decision"] != "PASS"
        or supplied != canonical_sha256(body)
    ):
        raise BenchmarkConflictError("generic gate evidence identity is invalid")
    checks = _exact_keys(
        evidence["checks"],
        {"benchmarks", "selected_process_count"},
        "generic gate evidence checks",
    )
    benchmarks = checks["benchmarks"]
    if not isinstance(benchmarks, list):
        raise BenchmarkConflictError("generic topology benchmarks must be an array")
    selected = _select_topology(benchmarks)
    if checks["selected_process_count"] != selected:
        raise BenchmarkConflictError(
            "generic gate evidence topology selection is invalid"
        )
    return evidence


def _completion_value(
    spec: BenchmarkSpec,
    observations: Sequence[BenchmarkObservation],
    evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    value: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": COMPLETION_RECEIPT_CONTRACT,
        "spec": {
            "path": os.fspath(spec.path),
            "file_sha256": spec.file_sha256,
            "spec_sha256": spec.identity,
        },
        "benchmark_argv_template_sha256": canonical_sha256(
            list(spec.benchmark_argv_template)
        ),
        "inputs": [binding.spec_value() for binding in spec.inputs],
        "repetitions": [
            observation.completion_value() for observation in observations
        ],
        "evidence_output": os.fspath(spec.evidence_output),
        "evidence": dict(evidence),
    }
    value["completion_sha256"] = canonical_sha256(value)
    return value


def _observation_from_completion(value: Any, index: int) -> BenchmarkObservation:
    item = _exact_keys(value, _REPETITION_KEYS, f"completion repetition {index}")
    process_count = item["process_count"]
    repetition = item["repetition"]
    if type(process_count) is not int or process_count not in PROCESS_COUNTS:
        raise BenchmarkConflictError(
            f"completion repetition {index} process_count is invalid"
        )
    if type(repetition) is not int or not 1 <= repetition <= REPETITION_COUNT:
        raise BenchmarkConflictError(
            f"completion repetition {index} number is invalid"
        )
    completed = _positive_int(
        item["completed_work_count"],
        f"completion repetition {index} completed_work_count",
    )
    elapsed = _positive_number(
        item["elapsed_seconds"], f"completion repetition {index} elapsed_seconds"
    )
    throughput = _positive_number(
        item["throughput_per_second"],
        f"completion repetition {index} throughput_per_second",
    )
    if throughput != completed / elapsed:
        raise BenchmarkConflictError(
            f"completion repetition {index} throughput was not independently derived"
        )
    return BenchmarkObservation(
        process_count=process_count,
        repetition=repetition,
        completed_work_count=completed,
        elapsed_seconds=elapsed,
        output_manifest_sha256=_require_sha256(
            item["output_manifest_sha256"],
            f"completion repetition {index} output hash",
        ),
        receipt_sha256=_require_sha256(
            item["receipt_sha256"], f"completion repetition {index} receipt hash"
        ),
    )


class EvaluatorTopologyBenchmark:
    """Fail-closed coordinator for the six benchmark command invocations."""

    def __init__(self, spec: BenchmarkSpec | Path) -> None:
        self.spec = (
            spec if isinstance(spec, BenchmarkSpec) else load_benchmark_spec(Path(spec))
        )

    def run(self) -> Mapping[str, Any]:
        self._ensure_work_root()
        with self._exclusive_lock():
            self._assert_frozen()
            replay = self._load_replay()
            if replay is not None:
                return replay
            self._reject_stale_attempts()

            observations = []
            for process_count in PROCESS_COUNTS:
                for repetition in range(1, REPETITION_COUNT + 1):
                    observations.append(
                        self._run_one(process_count, repetition=repetition)
                    )
            checks = _derive_checks(observations)
            evidence = _generic_evidence(checks)
            _validate_generic_evidence(evidence)
            completion = _completion_value(self.spec, observations, evidence)
            self._assert_frozen()

            # Completion is published first so a crash before generic evidence can
            # replay the already-validated result without re-running benchmarks.
            _publish_immutable(
                self.spec.completion_path, completion, root=self.spec.work_root
            )
            _publish_immutable(
                self.spec.evidence_output, evidence, root=self.spec.work_root
            )
            self._assert_frozen()
            return evidence

    def _ensure_work_root(self) -> None:
        root = self.spec.work_root
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        try:
            _directory_identity(root, "benchmark work_root")
        except EvaluatorTopologyBenchmarkError as exc:
            raise BenchmarkSpecError(str(exc)) from exc

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if fcntl is None:  # pragma: no cover - production targets are Unix.
            raise BenchmarkBusy("fcntl is required for exclusive benchmark locking")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(os.fspath(self.spec.lock_path), flags, 0o600)
        except OSError as exc:
            raise BenchmarkConflictError(f"cannot open benchmark lock: {exc}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BenchmarkConflictError(
                    "benchmark lock must be a non-hard-linked regular file"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BenchmarkBusy(
                    "another evaluator-topology benchmark is running"
                ) from exc
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _assert_frozen(self) -> None:
        observed_spec_hash = _stable_file_sha256(
            self.spec.path, "benchmark specification"
        )
        if observed_spec_hash != self.spec.file_sha256:
            raise BenchmarkSpecError("benchmark specification changed during execution")
        for binding in self.spec.inputs:
            binding.verify()

    def _load_replay(self) -> Optional[Mapping[str, Any]]:
        _recover_owned_publication_link(self.spec.completion_path)
        _recover_owned_publication_link(self.spec.evidence_output)
        completion_exists = _lexists(self.spec.completion_path)
        evidence_exists = _lexists(self.spec.evidence_output)
        if not completion_exists:
            if evidence_exists:
                raise BenchmarkConflictError(
                    "generic evidence exists without its benchmark completion receipt"
                )
            return None

        try:
            completion = _load_canonical_object(
                self.spec.completion_path, "benchmark completion receipt"
            )
            _exact_keys(
                completion, _COMPLETION_KEYS, "benchmark completion receipt"
            )
        except EvaluatorTopologyBenchmarkError as exc:
            if isinstance(exc, BenchmarkConflictError):
                raise
            raise BenchmarkConflictError(str(exc)) from exc
        body = dict(completion)
        supplied = _require_sha256(
            body.pop("completion_sha256", None), "benchmark completion identity"
        )
        if (
            completion["schema_version"] != SCHEMA_VERSION
            or isinstance(completion["schema_version"], bool)
            or completion["contract"] != COMPLETION_RECEIPT_CONTRACT
            or supplied != canonical_sha256(body)
        ):
            raise BenchmarkConflictError("benchmark completion receipt is invalid")
        expected_spec = {
            "path": os.fspath(self.spec.path),
            "file_sha256": self.spec.file_sha256,
            "spec_sha256": self.spec.identity,
        }
        if (
            completion["spec"] != expected_spec
            or completion["benchmark_argv_template_sha256"]
            != canonical_sha256(list(self.spec.benchmark_argv_template))
            or completion["inputs"]
            != [binding.spec_value() for binding in self.spec.inputs]
            or completion["evidence_output"] != os.fspath(self.spec.evidence_output)
        ):
            raise BenchmarkConflictError(
                "benchmark completion does not bind the current immutable spec"
            )
        raw_repetitions = completion["repetitions"]
        if not isinstance(raw_repetitions, list):
            raise BenchmarkConflictError("completion repetitions must be an array")
        observations = [
            _observation_from_completion(item, index)
            for index, item in enumerate(raw_repetitions)
        ]
        expected_evidence = _generic_evidence(_derive_checks(observations))
        embedded_evidence = _validate_generic_evidence(completion["evidence"])
        if embedded_evidence != expected_evidence:
            raise BenchmarkConflictError(
                "benchmark completion evidence contradicts its repetitions"
            )
        _publish_immutable(
            self.spec.evidence_output,
            embedded_evidence,
            root=self.spec.work_root,
        )
        return embedded_evidence

    def _reject_stale_attempts(self) -> None:
        stale = sorted(
            entry.name
            for entry in self.spec.work_root.iterdir()
            if entry.name.startswith(_ATTEMPT_PREFIX)
        )
        if stale:
            raise BenchmarkConflictError(
                "incomplete benchmark attempt directories require explicit cleanup: "
                + ", ".join(stale)
            )

    @contextlib.contextmanager
    def _attempt_directory(self) -> Iterator[Path]:
        name = tempfile.mkdtemp(prefix=_ATTEMPT_PREFIX, dir=self.spec.work_root)
        root = Path(name)
        _directory_identity(root, "isolated benchmark run root")
        try:
            yield root
        finally:
            self._remove_attempt(root)

    def _remove_attempt(self, root: Path) -> None:
        if root.parent != self.spec.work_root or not root.name.startswith(
            _ATTEMPT_PREFIX
        ):
            raise BenchmarkConflictError("refusing to clean an unowned benchmark path")
        if not _lexists(root):
            return
        metadata = root.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            root.unlink()
            raise BenchmarkConflictError(
                "benchmark command replaced its isolated root with a symlink"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            root.unlink()
            raise BenchmarkConflictError(
                "benchmark command replaced its isolated root with a non-directory"
            )
        try:
            shutil.rmtree(root)
            _fsync_directory(self.spec.work_root)
        except OSError as exc:
            raise BenchmarkConflictError(
                f"could not clean isolated benchmark run root: {exc}"
            ) from exc

    def _expanded_argv(
        self, process_count: int, *, output: Path, work_root: Path
    ) -> Tuple[str, ...]:
        replacements = {
            "process_count": str(process_count),
            "output": os.fspath(output),
            "work_root": os.fspath(work_root),
        }
        try:
            expanded = tuple(
                part.format_map(replacements)
                for part in self.spec.benchmark_argv_template
            )
        except (KeyError, ValueError) as exc:  # Defensive after loader validation.
            raise BenchmarkSpecError(f"cannot expand benchmark argv: {exc}") from exc
        if any(not part or "\x00" in part for part in expanded):
            raise BenchmarkSpecError("expanded benchmark argv is malformed")
        return expanded

    def _run_one(
        self, process_count: int, *, repetition: int
    ) -> BenchmarkObservation:
        with self._attempt_directory() as run_root:
            root_identity = _directory_identity(
                run_root, "isolated benchmark run root"
            )
            output = run_root / _COMMAND_RECEIPT_NAME
            argv = self._expanded_argv(
                process_count, output=output, work_root=run_root
            )
            invocation_error: Optional[BenchmarkExecutionError] = None
            result: Optional[_ProcessResult] = None
            try:
                result = _invoke_benchmark(
                    argv,
                    cwd=run_root,
                    timeout_seconds=self.spec.timeout_seconds,
                )
            except BenchmarkExecutionError as exc:
                invocation_error = exc
            self._assert_frozen()
            if invocation_error is not None:
                raise invocation_error
            assert result is not None
            if result.returncode != 0:
                detail = f": {result.stderr_tail}" if result.stderr_tail else ""
                raise BenchmarkExecutionError(
                    f"benchmark command failed with exit code {result.returncode}"
                    f"{detail}"
                )
            if (
                _directory_identity(run_root, "isolated benchmark run root")
                != root_identity
            ):
                raise BenchmarkExecutionError(
                    "benchmark command replaced its isolated work directory"
                )
            observation = _validate_command_receipt(
                output,
                run_root=run_root,
                process_count=process_count,
                repetition=repetition,
                timeout_seconds=self.spec.timeout_seconds,
                wall_elapsed_seconds=result.wall_elapsed_seconds,
            )
            self._assert_frozen()
            return observation


def run_benchmark_gate(
    spec: BenchmarkSpec | Path,
    *,
    expected_spec_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    if isinstance(spec, BenchmarkSpec):
        if expected_spec_sha256 is not None:
            expected = _require_sha256(
                expected_spec_sha256, "expected benchmark specification hash"
            )
            if spec.file_sha256 != expected:
                raise BenchmarkSpecError(
                    "loaded benchmark specification does not match expected SHA-256"
                )
        loaded = spec
    else:
        loaded = load_benchmark_spec(
            Path(spec), expected_spec_sha256=expected_spec_sha256
        )
    return EvaluatorTopologyBenchmark(loaded).run()


run_benchmarks = run_benchmark_gate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        evidence = run_benchmark_gate(
            args.spec,
            expected_spec_sha256=args.expected_spec_sha256,
        )
    except (EvaluatorTopologyBenchmarkError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    print(canonical_json(evidence))
    return 0


__all__ = [
    "BENCHMARK_RECEIPT_CONTRACT",
    "BenchmarkBusy",
    "BenchmarkConflictError",
    "BenchmarkExecutionError",
    "BenchmarkObservation",
    "BenchmarkSpec",
    "BenchmarkSpecError",
    "COMPLETION_RECEIPT_CONTRACT",
    "EvaluatorTopologyBenchmark",
    "EvaluatorTopologyBenchmarkError",
    "GATE_EVIDENCE_CONTRACT",
    "GATE_ID",
    "PROCESS_COUNTS",
    "REPETITION_COUNT",
    "SCHEMA_VERSION",
    "SPEC_CONTRACT",
    "TOPOLOGY_CHOICES",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
    "load_benchmark_spec",
    "load_spec",
    "main",
    "parse_args",
    "run_benchmark_gate",
    "run_benchmarks",
]


if __name__ == "__main__":
    raise SystemExit(main())
